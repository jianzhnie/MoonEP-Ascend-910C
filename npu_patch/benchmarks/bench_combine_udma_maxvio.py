"""MoonEP-Ascend combine-UDMA benchmark with a MaxVio sweep.

This is the combine counterpart of ``bench_dispatch_udma_maxvio.py``.  Every
MaxVio case first runs the real planning kernel; the timed loop reuses the plan
and contains only ``launch_combine``.  The symmetric dispatch-output buffers are
initialized in place, so no extra ``[S*K, H]`` input allocation is kept alive.

PlanningOp produces routing and combine execution metadata before timing.
Combine never builds execution metadata. Preparation time is reported separately;
moving it out is not an end-to-end speedup by itself.
Remote payload accounting covers canonical hidden and per-route weight gathers;
intra-server relay is reported separately. --check validates nonzero data before
timing. Only end-to-end latency and payload bandwidth are measured; the combine
kernel does not collect per-core profiling counters or timeline events.

Examples:

    # One node, 8 ranks
    torchrun --nproc-per-node=8 benchmarks/bench_combine_udma_maxvio.py

    # Two nodes, 8 ranks per node (run on both nodes with NODE_RANK=0/1)
    MASTER_ADDR=192.168.1.10 NODES=2 NODE_RANK=0 \
        bash benchmarks/run_bench.sh bench_combine_udma_maxvio.py

    # Cyclic plan used by the multi-node correctness tests
    torchrun --nproc-per-node=8 benchmarks/bench_combine_udma_maxvio.py \
        --plan-mode cyclic --no-weights
"""

import argparse
import csv
import json
import math
import statistics
import os

import torch
import torch_npu  # noqa: F401
import torch.distributed as dist

from _bench_common import generate_topk_routing, setup
from bench_dispatch_maxvio import TcpStoreStats, setup_no_dist

from npu_patch import Buffer, ShmemRuntime
from npu_patch.combine import COMBINE_PUT_CORES, combine_put_roles
from npu_patch.buffer_c import reset_symmetric_descriptors
from npu_patch._device import local_device_index
from npu_patch.shmem_runtime import default_shmem_ip_port
from npu_patch.launchers import (
    launch_combine,
    launch_planning,
)
from npu_patch.planning import allocate_planning_outputs, prepare_combine_plan


# Planning diagnostics are read only after preparation, outside timed combine.
# NPU Event intervals include synchronization/submission gaps; fused counters
# report device phase cycles. Do not mix the two measurement methods.
STAGES = ('base', 'combine_added', 'launch_gap', 'total')
CORE_STAGES = ('entry', 'publish_routes', 'scan_work', 'scan_tail_wait',
               'prefix', 'compact_or_fill', 'final_table', 'publish_plan')


def decode_planning_stages(values):
    """Reserved header: per-stage min/sum/max/slowest-core, device 1GHz cycles."""
    if len(values) != 36:
        raise ValueError('invalid planning detail header length')
    if int(values[0]) == 0:  # Tiny shapes cannot fit per-core diagnostic scratch.
        return None
    magic, cores, stages, path = map(int, values[:4])
    if magic != 0x504c414e53544731 or not 1 <= cores <= 64 or stages != 8 or path not in (0, 1, 2, 3, 4):
        raise ValueError('invalid planning detail header')
    result = dict(cores=cores, path=('two_pass', 'source_staged', 'token_tiled', 'token_tiled_simt', 'token_tiled_simt_fused')[path], stages={})
    for i, stage in enumerate(CORE_STAGES):
        low, total, high, owner = map(int, values[4+i*4:8+i*4])
        if low < 0 or not low*cores <= total <= high*cores or not 0 <= owner < cores:
            raise ValueError('invalid planning per-core stage: ' + stage)
        result['stages'][stage] = dict(min_us=low/1000., mean_us=total/cores/1000.,
                                       max_us=high/1000., max_core=owner)
    return result


def decode_fused_planning_timing(values, rank):
    magic, cores, base, added, total, mode = map(int, values)
    if (magic != 0x504c414e54494d45 or not 1 <= cores <= 64 or mode != 1
            or min(base, added) < 0 or base + added != total):
        raise ValueError('invalid fused planning timing header')
    return dict(rank=int(rank), base=base/1000.0, combine_added=added/1000.0,
                launch_gap=0.0, total=total/1000.0,
                timing_method='device_core0_phase_cycles_1ghz_single_preparation')


def read_planning_timing(ctx, plan, execution):
    """Call only after planning, never from a timed combine iteration."""
    execution.prepare_end.synchronize()
    if getattr(execution, 'fused', False):
        import torch
        values = execution.storage[64:416].cpu().view(torch.int64).tolist()
        row = decode_fused_planning_timing(values[:6], ctx['rank'])
        row['core_detail'] = decode_planning_stages(values[8:44])
        return row
    us = lambda a, b: float(a.elapsed_time(b)) * 1000.0
    row = dict(rank=int(ctx['rank']), base=None, launch_gap=None, total=None,
               combine_added=us(execution.prepare_begin, execution.prepare_end))
    events = ctx.get('_planning_timing_events')
    if (events is not None and events['routing_id'] == id(plan) and
            events['generation'] == execution.generation == ctx.get('_combine_plan_generation', 0)):
        row.update(base=us(events['begin'], events['end']),
                   launch_gap=us(events['end'], execution.prepare_begin),
                   total=us(events['begin'], execution.prepare_end))
    # Synthetic routes have no base planner sample; do not report a fake zero
    # or use the preceding case's event pair as this case's total.
    return row


def summarize_planning_timing(rows):
    rows = sorted(rows, key=lambda row: row['rank'])
    if not rows or [x['rank'] for x in rows] != list(range(len(rows))):
        raise ValueError('planning timing requires exactly one sample per rank')
    methods = {x.get('timing_method', 'npu_event_stream_interval_single_preparation') for x in rows}
    if len(methods) != 1:
        raise ValueError('cannot mix fused device-cycle and event timing samples')
    summary = {'_method': methods.pop()}
    for stage in STAGES:
        values = [x[stage] for x in rows]
        if all(x is None for x in values):
            summary[stage] = None
            continue
        if any(x is None or not math.isfinite(x) or x < 0 for x in values):
            raise ValueError('invalid or incomplete planning timing: ' + stage)
        slow = max(range(len(rows)), key=lambda i: values[i])
        summary[stage] = dict(rank0_us=values[0], p50_us=statistics.median(values),
                              max_us=values[slow], max_rank=rows[slow]['rank'],
                              per_rank_us=values)
    details = [row.get('core_detail') for row in rows]
    if any(detail is not None for detail in details):
        if any(detail is None for detail in details):
            raise ValueError('incomplete planning per-core details')
        if len({(d['cores'], d['path']) for d in details}) != 1:
            raise ValueError('mixed planning paths/core counts')
        summary['_core_detail'] = dict(cores=details[0]['cores'], path=details[0]['path'], stages={})
        for stage in CORE_STAGES:
            data = [d['stages'][stage] for d in details]
            slow = max(range(len(data)), key=lambda i: data[i]['max_us'])
            summary['_core_detail']['stages'][stage] = dict(
                rank0=data[0], p50_core_max_us=statistics.median(d['max_us'] for d in data),
                max_us=data[slow]['max_us'], max_rank=rows[slow]['rank'], max_core=data[slow]['max_core'])
    return summary


def planning_timing_csv(summary):
    result = {'planning_timing_method': summary.get('_method', 'npu_event_stream_interval_single_preparation')}
    for stage in STAGES:
        data = summary[stage]
        for metric in ('rank0_us', 'p50_us', 'max_us', 'max_rank'):
            result['planning_' + stage + '_' + metric] = data[metric] if data else ''
        result['planning_' + stage + '_per_rank_us'] = (
            ';'.join(f'{x:.3f}' for x in data['per_rank_us']) if data else '')
    base, extra, total = summary['base'], summary['combine_added'], summary['total']
    result['planning_combine_added_percent_rank0'] = (
        100.0 * extra['rank0_us'] / total['rank0_us']
        if base and extra and total and total['rank0_us'] > 0 else '')
    detail = summary.get('_core_detail')
    if detail:
        result['planning_builder_path'] = detail['path']
        result['planning_builder_cores'] = detail['cores']
        for stage, data in detail['stages'].items():
            prefix = 'planning_detail_' + stage + '_'
            for key in ('p50_core_max_us', 'max_us', 'max_rank', 'max_core'):
                result[prefix + key] = data[key]
            for key, value in data['rank0'].items():
                result[prefix + 'rank0_' + key] = value
    return result


def format_planning_timing(summary):
    lines = ['[planning timing] us; single preparation; p50 across ranks, not iterations; '
             'NPU stream intervals include kernel waits/submission gaps; excluded from combine timing']
    if summary.get('_method', '').startswith('device_core0_'):
        lines[0] = ('[planning timing] us; fused single kernel; core0 phase cycles at 1GHz; '
                    'p50 across ranks, not iterations; includes phase waits, excludes host launch gaps; '
                    'excluded from combine timing')
    for stage in STAGES:
        data = summary[stage]
        lines.append(f'[planning timing] {stage}: ' + (
            f"rank0={data['rank0_us']:.3f} p50={data['p50_us']:.3f} "
            f"max={data['max_us']:.3f} (rank {data['max_rank']})"
            if data else 'N/A (base planner not measured; e.g. synthetic routes)'))
    share = planning_timing_csv(summary)['planning_combine_added_percent_rank0']
    if share != '':
        lines.append(f'[planning timing] combine_added / total (rank0) = {share:.2f}%')
    detail = summary.get('_core_detail')
    if detail:
        lines.append(f"[planning detail] path={detail['path']} cores={detail['cores']}; us; "
                     'scan_work excludes local scan join; other intervals may include waits; '
                     'stage maxima are not additive; diagnostic reduction is included in combine_added')
        for stage, data in detail['stages'].items():
            d = data['rank0']
            lines.append(f"[planning detail] {stage}: rank0 core min/mean/max="
                         f"{d['min_us']:.3f}/{d['mean_us']:.3f}/{d['max_us']:.3f} "
                         f"(core {d['max_core']}); p50_core_max={data['p50_core_max_us']:.3f}; "
                         f"max={data['max_us']:.3f} (rank {data['max_rank']} core {data['max_core']})")
    return '\n'.join(lines)


DEFAULT_ACTIVE_QP_COUNT = 4
def created_qp_count(active_qp_count: int) -> int:
    """Map the hot-QP count to dispatch's physical QP layout."""
    return active_qp_count if active_qp_count < 4 else 6


def gbps(nbytes: int, us: float) -> float:
    return nbytes / (us * 1e-6) / 1e9 if us > 0 else 0.0


def print_box_table(headers, rows, alignments, min_widths=None):
    """Print a compact ASCII table with borders around every cell and row."""
    header_lines = [str(header).split("\n") for header in headers]
    if min_widths is None:
        min_widths = (0,) * len(headers)
    widths = [
        max(
            [min_widths[index]]
            + [len(line) for line in header_lines[index]]
            + [len(str(row[index])) for row in rows]
        )
        for index in range(len(headers))
    ]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def format_row(values, row_alignments):
        cells = []
        for value, width, alignment in zip(values, widths, row_alignments):
            value = str(value)
            if alignment == "left":
                value = value.ljust(width)
            elif alignment == "center":
                value = value.center(width)
            else:
                value = value.rjust(width)
            cells.append(f" {value} ")
        return "|" + "|".join(cells) + "|"

    print(border)
    header_height = max(len(lines) for lines in header_lines)
    for line_index in range(header_height):
        values = [
            lines[line_index] if line_index < len(lines) else ""
            for lines in header_lines
        ]
        print(format_row(values, ("center",) * len(headers)))
    print(border)
    for row in rows:
        print(format_row(row, alignments))
        print(border)


def print_summary_table(rows):
    """Print the operator-level summary using one consistent max-rank latency."""
    first = rows[0]
    print()
    print("MoonEP Combine UDMA Summary")
    print(
        f"EP={first['ep']} S={first['S']} E={first['E']} H={first['H']} "
        f"K={first['K']} QP={first['qp']} Weights="
        f"{'on' if first['with_weights'] else 'off'}"
    )
    print()
    table_rows = []
    for row in rows:
        dedup = row["remote_dedup_ratio"]
        table_rows.append(
            (
                f"{row['peak_avg_load']:.3f}x",
                f"{row['combine_max_us']:.1f}",
                f"{row['logical_hidden_gbps']:.1f}",
                f"{row['remote_hidden_gbps_per_rank']:.1f}",
                f"{row['remote_weight_gbps_per_rank']:.3f}",
                f"{row['remote_total_gbps_per_rank']:.1f}",
                f"{dedup:.3f}" if dedup != "" else "n/a",
            )
        )
    print_box_table(
        (
            "Peak/Avg\n",
            "Latency\n(us)",
            "Logical BW\n(GB/s/rank)",
            "Remote Hidden\n(GB/s/rank)",
            "Remote Weight\n(GB/s/rank)",
            "Remote Total\n(GB/s/rank)",
            "Dedup\n",
        ),
        table_rows,
        ("right",) * 7,
        (8, 8, 11, 12, 13, 12, 5),
    )
    print()
    print("Peak/Avg = max expert load / average expert load.")
    print("Latency = slowest-rank latency.")
    print("Logical BW/rank = S*K*H*2 bytes / Latency.")
    print(
        "Remote Hidden BW/rank = "
        "all-rank remote hidden payload / EP / Latency."
    )
    print(
        "Remote Weight BW/rank = "
        "all-rank remote weight payload / EP / Latency."
    )
    print("Remote Total BW/rank = Remote Hidden BW/rank + Remote Weight BW/rank.")
    print("Dedup = unique remote hidden rows / remote route count.")
    print()


def tcp_all_gather(stats, value):
    """CPU-only all-rank view for check routes and small planning/traffic summaries."""
    key = f"combine-all/{stats.step}"
    stats.step += 1
    stats.store.set(f"{key}/{stats.rank}", json.dumps(value))
    values = [json.loads(stats.store.get(f"{key}/{peer}"))
              for peer in range(stats.world_size)]
    # No reader may lose its data during cleanup. Use dispatch's release/ack protocol.
    stats.barrier()
    if stats.rank == 0:
        for peer in range(stats.world_size):
            stats.store.delete_key(f"{key}/{peer}")
    return values


def time_op(fn, group, rank, warmup=20, iters=50, verbose=True, stats=None):
    """Return per-rank, mean, and slowest-rank NPU-event latency."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    if stats is None:
        dist.barrier(group=group)
    else:
        stats.barrier()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    local_us = start.elapsed_time(end) / iters * 1e3

    if stats is not None:
        # Same CPU TCPStore collector as dispatch; all collection is outside NPU events.
        return stats.collect(local_us, 0, 0, 0)

    dev = torch.device(f"npu:{torch.npu.current_device()}")
    world_size = dist.get_world_size(group=group)
    local = torch.tensor([local_us], dtype=torch.float32, device=dev)
    gathered = [
        torch.empty(1, dtype=torch.float32, device=dev)
        for _ in range(world_size)
    ]
    dist.all_gather(gathered, local, group=group)
    times = [value.item() for value in gathered]
    mean_us = sum(times) / len(times)
    max_us = max(times)
    if verbose and rank == 0:
        print(
            f"  per-rank latency: {[f'{value:.1f}' for value in times]} "
            f"max={max_us:.1f} mean={mean_us:.1f} us",
            flush=True,
        )
    dist.barrier(group=group)
    return {"per_rank_us": times, "mean_us": mean_us, "max_us": max_us}


def _maxvio_for_sigma(sigma, S, K, E, R, dev):
    total = torch.zeros(E, dtype=torch.int64, device=dev)
    for rank in range(R):
        _, tokens_per_expert = generate_topk_routing(
            S, K, E, R, sigma, dev, 1234, rank=rank
        )
        total += tokens_per_expert.to(torch.int64)
    return total.max().item() / (total.sum().item() / E) - 1.0


def solve_sigma_for_maxvio(target, S, K, E, R, dev):
    """Find a deterministic routing sigma whose realized MaxVio is closest."""
    lo, hi = 1e-5, 10.0
    mv_lo = _maxvio_for_sigma(lo, S, K, E, R, dev)
    if mv_lo >= target:
        return lo, mv_lo
    best_sigma, best_mv = lo, mv_lo
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        mv = _maxvio_for_sigma(mid, S, K, E, R, dev)
        if abs(mv - target) < abs(best_mv - target):
            best_sigma, best_mv = mid, mv
        if abs(mv - target) <= max(0.02 * target, 0.005):
            return mid, mv
        if mv < target:
            lo = mid
        else:
            hi = mid
    return best_sigma, best_mv


# CPU routing oracle and nonzero correctness checks used by --check.
def server_cyclic_plan(tokens, topk, ranks, rank):
    """Return server-dedup ABI lists, including full-address duplicate members."""
    if ranks % 8:
        raise ValueError("server cyclic plan requires complete 8-rank servers")
    n = tokens * topk
    dst = [0] * n
    groups, members = [], []
    prefixes = []
    for source in range(ranks):
        prefixes.append(len(groups))
        for token in range(tokens):
            domains = {}
            for j in range(topk):
                route = token * topk + j
                peer = (source - (token + j)) % ranks
                address = peer * n + route
                domains.setdefault(peer // 8, []).append((route, address))
                if source == rank:
                    dst[route] = address
            for entries in domains.values():
                if len(entries) < 2:
                    continue
                route, address = entries[0]
                if source == rank:
                    dst[route] = (address | 0x80000000) - (1 << 32)
                    for secondary_route, secondary_address in entries[1:]:
                        dst[secondary_route] = -secondary_address - 1
                if address // n == rank:
                    groups.append([address % n, len(members),
                                   (route << 5) | (len(entries) - 1)])
                    members.extend(a for _, a in entries[1:])
    counts = [len(groups), len(members)] + prefixes
    assert len(groups) <= n and len(members) <= 8 * n
    groups.extend([[-1, -1, -1] for _ in range(n - len(groups))])
    members.extend([-1] * (8 * n - len(members)))
    return dst, groups, members, counts


def decode_route(encoded):
    if encoded & 0xc0000000 == 0x80000000:
        return encoded & 0x3fffffff
    return encoded if encoded >= 0 else -encoded - 1




def check_nonzero(runner, group, epochs=3, stats=None):
    """Check real plan routing against a source-side oracle (not kernel buffers).

    Values depend on original source/route/epoch/dimension, so missed routes,
    stale flags, wrong destinations and missing weights are observable. Inputs
    are binary fractions and sums fit BF16 exactly for K<=16, allowing exact
    equality independent of the reduction order. All epochs use the production combine kernel.
    """
    import torch
    import torch.distributed as dist
    rank, ranks, n, h = runner.rank, runner.R, runner.S * runner.K, runner.H
    local = runner.plan.dst.to(torch.int64)
    if stats is None:
        gathered = [torch.empty_like(local) for _ in range(ranks)]
        dist.all_gather(gathered, local, group=group)
        routes = [item.cpu().tolist() for item in gathered]
    else:
        routes = tcp_all_gather(stats, local.cpu().tolist())
    # Prepared plans remain fixed across epochs; only hidden/weights change.
    # Validate both their lifecycle guard and the actual returned payloads.
    if getattr(runner, 'execution_plan', None) is not None:
        runner.execution_plan.validate(runner.ctx, runner.plan)
    owner_source = torch.full((n,), -1, dtype=torch.int64)
    owner_route = torch.full((n,), -1, dtype=torch.int64)
    for source, encoded_routes in enumerate(routes):
        for route, encoded in enumerate(encoded_routes):
            owner, loff = divmod(decode_route(encoded), n)
            if owner == rank:
                if owner_source[loff] != -1:
                    raise AssertionError(f"plan aliases rank={rank} loff={loff}")
                owner_source[loff], owner_route[loff] = source, route
    dim = (torch.arange(h) % 3).float() / 16
    valid = owner_source >= 0
    device = runner.output_sh.device
    for epoch in range(epochs):
        values = ((owner_source % 5 + owner_route % 7 + epoch % 3).float()[:, None]
                  + (torch.arange(h) % 3).float()[None, :]) / 16
        values[~valid] = 0
        runner.ctx["hidden_buf_local"].copy_(values.to(device, dtype=torch.bfloat16))
        if runner.with_weights:
            weights = (owner_source * n + owner_route).float() / 16 + epoch / 4
            weights[~valid] = 0
            runner.ctx["weights_buf_local"].view(torch.float32).copy_(weights.to(device))
        torch.npu.synchronize()
        if stats is None:
            dist.barrier(group=group)
        else:
            stats.barrier()
        if rank == 0:
            print(f"[PUT check] begin epoch={epoch}", flush=True)
        runner.combine()
        torch.npu.synchronize()
        source_values = ((rank % 5 + torch.arange(n) % 7 + epoch % 3).float()[:, None]
                         / 16 + dim[None, :])
        expected = source_values.reshape(runner.S, runner.K, h).sum(1).to(torch.bfloat16)
        torch.testing.assert_close(runner.output_sh.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(runner.ctx["hidden_buf_local"].cpu(),
                                   values.to(torch.bfloat16), rtol=0, atol=0)
        if runner.with_weights:
            expected_w = ((rank * n + torch.arange(n)).float() / 16 + epoch / 4)
            torch.testing.assert_close(runner.output_sk.cpu(), expected_w.reshape(runner.S, runner.K),
                                       rtol=0, atol=0)
        if stats is None:
            dist.barrier(group=group)
        else:
            stats.barrier()
    if rank == 0:
        print(f"[PUT check] PASS ranks={ranks} S={runner.S} K={runner.K} H={h} "
              f"epochs={epochs} weights={runner.with_weights} hidden+input-preservation", flush=True)
    runner.activate_symmetric_views()


def make_cyclic_plan_tensors(S, K, R, rank, dev, server=False):
    """Build the cyclic ordered plan used by the UDMA correctness tests."""
    if server:
        return tuple(torch.tensor(value, dtype=torch.int32, device=dev)
                     for value in server_cyclic_plan(S, K, R, rank))
    slots = S * K
    src_ranks = torch.arange(R).reshape(-1, 1, 1)
    token_ids = torch.arange(S).reshape(1, -1, 1)
    topk_ids = torch.arange(K).reshape(1, 1, -1)
    slot_idx = token_ids * K + topk_ids
    target_ranks = (
        src_ranks - ((token_ids + topk_ids) % R) + R
    ) % R
    destination_by_route = (
        target_ranks * slots + slot_idx
    ).reshape(-1).to(torch.int32).contiguous()

    source_start = rank * slots
    local_raw_dst = destination_by_route[source_start:source_start + slots]
    local_dest_ranks = torch.div(
        local_raw_dst, slots, rounding_mode="floor"
    ).reshape(S, K)
    canonical = torch.ones((S, K), dtype=torch.bool)
    for topk_id in range(1, K):
        canonical[:, topk_id] = torch.all(
            local_dest_ranks[:, :topk_id]
            != local_dest_ranks[:, topk_id].unsqueeze(1),
            dim=1,
        )
    dst_local = torch.where(
        canonical.reshape(-1), local_raw_dst, -local_raw_dst - 1
    ).contiguous()

    destination_start = rank * slots
    destination_end = destination_start + slots
    local_destination_mask = (
        (destination_by_route >= destination_start)
        & (destination_by_route < destination_end)
    )
    source_route_indices = torch.nonzero(
        local_destination_mask, as_tuple=False
    ).reshape(-1)
    if int(source_route_indices.numel()) != slots:
        raise AssertionError("cyclic plan does not cover every local output slot")
    local_loffs = (
        destination_by_route[source_route_indices] - destination_start
    ).to(torch.int64)
    received_source_routes = torch.empty(slots, dtype=torch.int32)
    received_source_routes[local_loffs] = source_route_indices.to(torch.int32)

    sorted_source_routes, sorted_loffs = torch.sort(received_source_routes)
    global_tokens = torch.div(
        sorted_source_routes, K, rounding_mode="floor"
    )
    _, token_counts = torch.unique_consecutive(
        global_tokens, return_counts=True
    )
    token_starts = torch.cumsum(token_counts, dim=0) - token_counts
    duplicate_groups = token_counts > 1
    group_counts = token_counts[duplicate_groups]
    group_starts = token_starts[duplicate_groups]
    duplicate_counts = group_counts - 1
    duplicate_starts = torch.cumsum(duplicate_counts, dim=0) - duplicate_counts

    dup_groups_local = torch.full((slots, 3), -1, dtype=torch.int32)
    group_count = int(group_counts.numel())
    if group_count > 0:
        dup_groups_local[:group_count, 0] = sorted_loffs[group_starts].to(
            torch.int32
        )
        dup_groups_local[:group_count, 1] = duplicate_starts.to(torch.int32)
        dup_groups_local[:group_count, 2] = duplicate_counts.to(torch.int32)

    duplicate_members = torch.zeros(slots, dtype=torch.bool)
    if slots > 1:
        duplicate_members[1:] = global_tokens[1:] == global_tokens[:-1]
    duplicate_loffs = sorted_loffs[duplicate_members].to(torch.int32)
    duplicate_count = int(duplicate_loffs.numel())
    dup_loffs_local = torch.full((slots,), -1, dtype=torch.int32)
    dup_loffs_local[:duplicate_count] = duplicate_loffs
    dup_counts_local = torch.tensor(
        [group_count, duplicate_count], dtype=torch.int32
    )

    return (
        dst_local.to(dev),
        dup_groups_local.to(dev),
        dup_loffs_local.to(dev),
        dup_counts_local.to(dev),
    )


class CombineUdmaRunner:
    """Own the Buffer, plan, symmetric inputs, and fixed combine outputs."""

    def __init__(self, group, R, S, K, E, H, num_sms, local_rank, with_weights,
                 rank=None, stats=None, plan_source='kernel'):
        self.group = group
        # 'kernel': device planner (Ascend950/SIMT only); 'golden': host plan via
        # test_npu/planing_reference -- required on A2/A3. (The URMA engine
        # itself stays A5-only: UBC_CTP/URMA channels do not exist on A3.)
        self.plan_source = plan_source
        self.rank = rank if rank is not None else dist.get_rank(group=group)
        self.stats = stats
        self.R = R
        self.S = S
        self.K = K
        self.H = H
        self.with_weights = with_weights
        self.execution_plan = None
        self.execution_plan_prepare_us = 0.0
        self.buffer = Buffer(
            S,
            H,
            K,
            E,
            R,
            num_sms=num_sms,
            group=group,
            explicitly_destroy=True,
            use_udma=False,
            combine_use_udma=True,
            rank=rank,
        )
        self.ctx = self.buffer._require_ctx()
        # Instrument only this benchmark; normal PlanningOp callers need not
        # pay for extra events. No device synchronize is inserted between stages.
        self.ctx['_planning_timing_enabled'] = True
        self.plan, self.cu_seqlens = allocate_planning_outputs(self.ctx, dedup_mode='server')
        dev = torch.device(f"npu:{local_rank}")
        self.output_sh = torch.empty(
            (S, H), dtype=torch.bfloat16, device=dev
        )
        self.output_sk = None
        if with_weights:
            self.output_sk = torch.empty(
                (S, K), dtype=torch.float32, device=dev
            )

    def activate_symmetric_views(self):
        self.buffer._ensure_shmem_views()
        # This benchmark uses zero hidden for timing; the separate PUT
        # regression checks nonzero hidden and repeated-call correctness.
        self.ctx["hidden_buf_local"].zero_()
        if self.with_weights:
            self.ctx["weights_buf_local"].view(torch.float32).fill_(
                self.rank + 0.5
            )
        self.ctx["combine_sync_buffer"].zero_()
        torch.npu.synchronize()
        ShmemRuntime.barrier(group=self.group)

    def prepare(self, topk, tokens_per_expert):
        """Run base routing and the automatic combine execution-plan substage."""
        topk_flat = topk.reshape(-1).contiguous()
        if self.plan_source == 'golden':
            from test_npu.planing_reference import reference_plan_and_cu_seqlens
            self.plan, self.cu_seqlens = reference_plan_and_cu_seqlens(
                self.ctx, topk_flat, tokens_per_expert)
        else:
            launch_planning(
                self.ctx,
                topk_flat,
                tokens_per_expert,
                self.cu_seqlens,
                self.plan,
            )
        torch.npu.synchronize()
        ShmemRuntime.barrier(group=self.group)

    def prepare_execution_plan(self):
        # Real routing already produced the plan in PlanningOp. Only synthetic
        # cyclic routes need the planning substage explicitly after manual writes.
        self.execution_plan = self.plan.combine_execution_plan
        if self.execution_plan is None:
            self.execution_plan = prepare_combine_plan(self.ctx, self.plan)
        local_timing = read_planning_timing(self.ctx, self.plan, self.execution_plan)
        self.execution_plan_prepare_us = local_timing['combine_added']
        if self.stats is None:
            rank_timings = [None] * self.R
            dist.all_gather_object(rank_timings, local_timing, group=self.group)
        else:
            rank_timings = tcp_all_gather(self.stats, local_timing)
        self.planning_timing_summary = summarize_planning_timing(rank_timings)
        if self.rank == 0:
            print(f"[combine prepare] explicit plan, rank0={self.execution_plan_prepare_us:.3f} us "
                  f"storage={self.execution_plan.storage.numel()} bytes; excluded from combine timing",
                  flush=True)
            print(format_planning_timing(self.planning_timing_summary), flush=True)

    def combine(self):
        launch_combine(
            self.ctx,
            self.output_sh,
            self.plan,
            self.output_sk,
        )

    def local_remote_traffic(self):
        """Count canonical return payload and intra-server relay payload."""
        NvS = int(self.ctx["NvS"])
        encoded = self.plan.dst.to(torch.int64)
        forwarded = (encoded & 0xc0000000) == 0x80000000
        canonical = (encoded >= 0) | forwarded
        raw_dst = torch.where(forwarded, encoded & 0x3fffffff,
                              torch.where(encoded >= 0, encoded, -encoded - 1))
        peer_rank = torch.div(raw_dst, NvS, rounding_mode="floor")
        server_mode = self.plan.dup_counts.numel() == self.R + 2
        if server_mode:
            # Phase A returns every same-server original row (no partial sum).
            canonical |= torch.div(peer_rank, 8, rounding_mode="floor") == self.rank // 8
        is_remote = peer_rank != self.rank
        remote_hidden_rows = (canonical & is_remote).sum(dtype=torch.int64)
        remote_weight_values = is_remote.sum(dtype=torch.int64)
        if not self.with_weights:
            remote_weight_values.zero_()
        server_intra_hidden_rows = 0
        if server_mode:
            dup_count = int(self.plan.dup_counts[1].item())
            dup_count = min(dup_count, self.plan.dup_loffs.numel())
            if dup_count > 0:
                source_is_remote = torch.ones(dup_count, dtype=torch.bool, device=encoded.device)
                prefixes = self.plan.dup_counts.cpu().tolist()
                # Groups/dups are source-major in the public server plan. Local
                # source groups bypass relay/reduce entirely in the aligned path.
                for source in range(self.rank // 8 * 8, min(self.R, self.rank // 8 * 8 + 8)):
                    gb = prefixes[2 + source]
                    ge = prefixes[3 + source] if source + 1 < self.R else prefixes[0]
                    if ge > gb:
                        first = int(self.plan.dup_groups[gb, 1].item())
                        last = int(self.plan.dup_groups[ge - 1, 1].item())
                        last += int(self.plan.dup_groups[ge - 1, 2].item()) & 31
                        source_is_remote[first:last] = False
                source_rank = torch.div(
                    self.plan.dup_loffs[:dup_count].to(torch.int64),
                    NvS,
                    rounding_mode="floor",
                )
                same_server = torch.div(
                    source_rank, 8, rounding_mode="floor"
                ) == self.rank // 8
                server_intra_hidden_rows = int(
                    (
                        (source_rank >= 0)
                        & (source_rank < self.R)
                        & (source_rank != self.rank)
                        & same_server
                        & source_is_remote
                    ).sum().item()
                )
        return (
            int(remote_hidden_rows.item()),
            int(remote_weight_values.item()),
            server_intra_hidden_rows,
        )

    def destroy(self):
        self.buffer.destroy()


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [f"{row['maxvio']:.2g}" for row in rows]
    xpos = np.arange(len(rows))
    means = [row["combine_mean_us"] for row in rows]
    maxima = [row["combine_max_us"] for row in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.36
    ax.bar(xpos - width / 2, means, width, label="rank mean")
    ax.bar(xpos + width / 2, maxima, width, label="slowest rank")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_xlabel("realized MaxVio")
    ax.set_ylabel("combine latency (us)")
    ax.set_title(
        f"MoonEP-Ascend combine-UDMA, EP={rows[0]['ep']}, "
        f"S={rows[0]['S']}, K={rows[0]['K']}, H={rows[0]['H']}, qp=2"
    )
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=('urma',), default='urma',
                        help='PUT combine benchmark requires Ascend950 UDMA/URMA')
    parser.add_argument(
        "--out",
        default="bench_combine_udma_maxvio_results.csv",
        help="CSV output path on rank 0 (default: %(default)s)",
    )
    parser.add_argument(
        "--plot", default=None, help="Optional latency figure path on rank 0"
    )
    parser.add_argument(
        "--maxvios",
        default="0.2,1,10,20",
        help="Comma-separated target MaxVio values (default: %(default)s)",
    )
    parser.add_argument(
        "--plan-mode",
        choices=("maxvio", "cyclic"),
        default="maxvio",
        help="sigma-based MaxVio sweep or cyclic correctness-test plan",
    )
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--plan-source", choices=("kernel", "golden"),
                        default="kernel",
                        help="kernel: device planner (Ascend950/SIMT only); "
                             "golden: host plan via test_npu/planing_reference "
                             "(required on A2/A3). The URMA engine itself stays "
                             "A5-only.")
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-sms", type=int, default=32)
    parser.add_argument(
        "--active-qp-count",
        type=int,
        default=DEFAULT_ACTIVE_QP_COUNT,
        help="Hot UDMA QP count (default: %(default)s)",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--check", action="store_true",
                        help="Check nonzero hidden, all weights, 3 epochs before timing")
    parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Benchmark hidden-only combine",
    )
    parser.add_argument(
        "--ip-port",
        default=None,
        help=(
            "SHMEM DEFAULT-bootstrap endpoint. Multi-node runs derive it from "
            "MASTER_ADDR and MASTER_PORT (+1) when omitted"
        ),
    )
    parser.add_argument('--no-dist', action='store_true',
                        help='Use torchrun only as a launcher; SHMEM bootstrap and CPU TCPStore statistics')
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.iters <= 0:
        parser.error("--iters must be > 0")
    if not 1 <= args.active_qp_count <= 4:
        parser.error(
            "--active-qp-count must be in [1, 4], "
            f"got {args.active_qp_count}"
        )

    if args.no_dist:
        rank, world_size, local_rank = setup_no_dist()
        group = None
    else:
        rank, world_size, local_rank = setup()
        group = dist.group.WORLD
    local_rank = local_device_index(local_rank)
    from npu_patch.api import _is_ascend950
    if not _is_ascend950(local_rank, torch.npu.get_device_properties(local_rank)):
        if rank == 0:
            print("SKIP: combine UDMA benchmark requires Ascend950; 910C uses MTE", flush=True)
        if dist.is_initialized():
            dist.destroy_process_group()
        return
    stats = None
    R = world_size
    S, K, E, H = args.tokens, args.topk, args.experts, args.hidden
    dev = f"npu:{local_rank}"
    assert R >= 2, f"combine benchmark requires at least 2 ranks, got {R}"
    assert E % R == 0, f"E ({E}) must be divisible by world size ({R})"
    assert K <= E, f"K ({K}) must be <= E ({E})"
    targets = [float(value) for value in args.maxvios.split(",")]
    if args.plan_mode == "cyclic":
        targets = [0.0]

    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    multi_node = world_size > local_world_size
    runner = CombineUdmaRunner(
        group,
        R,
        S,
        K,
        E,
        H,
        args.num_sms,
        local_rank,
        with_weights=not args.no_weights,
        rank=rank if args.no_dist else None,
        plan_source=args.plan_source,
    )
    qp_count = created_qp_count(args.active_qp_count)
    if not ShmemRuntime.is_initialized():
        if args.no_dist:
            ip_port = args.ip_port or default_shmem_ip_port(
                os.environ.get('MASTER_ADDR', '127.0.0.1'),
                int(os.environ.get('MASTER_PORT', '29500')))
            ShmemRuntime.init_with_ip_port(
                ip_port=ip_port, rank=rank, world_size=R,
                enable_udma=True, qp_count=qp_count)
        elif args.ip_port is not None or multi_node:
            ShmemRuntime.init_with_ip_port(
                group=group,
                ip_port=args.ip_port,
                enable_udma=True,
                qp_count=qp_count,
            )
        else:
            ShmemRuntime.init_with_buffer(
                group=group,
                enable_udma=True,
                qp_count=qp_count,
            )
    runner.activate_symmetric_views()

    if args.no_dist:
        stats = TcpStoreStats(rank, R)
        runner.stats = stats

    if rank == 0:
        print(
            f"moonep-udma combine bench: ep={R} S={S} E={E} H={H} K={K} "
            f"qp={args.active_qp_count} (created={qp_count}) "
            f"with_weights={not args.no_weights} stats={'TCPStore' if args.no_dist else 'HCCL'}",
            flush=True,
        )

    rows = []
    try:
        for target in targets:
            if args.plan_mode == "cyclic":
                if rank == 0:
                    print("--- plan-mode cyclic ---", flush=True)
                sigma = 0.0
                maxvio = 0.0
                note = ""
                plan_tensors = make_cyclic_plan_tensors(
                    S, K, R, rank, dev, server=runner.plan.dup_counts.numel() == R + 2)
                runner.plan.dst.copy_(plan_tensors[0])
                runner.plan.dup_groups.copy_(plan_tensors[1])
                runner.plan.dup_loffs.copy_(plan_tensors[2])
                runner.plan.dup_counts.copy_(plan_tensors[3])
                object.__setattr__(runner.plan, 'combine_execution_plan', None)
                runner.ctx.pop('_planning_timing_events', None)
                torch.npu.synchronize()
            else:
                solved = [None, None]
                if args.no_dist or rank == 0:
                    solved[:] = solve_sigma_for_maxvio(
                        target, S, K, E, R, dev
                    )
                if not args.no_dist:
                    dist.broadcast_object_list(solved, src=0, group=group)
                sigma, realized = solved
                topk, tokens_per_expert = generate_topk_routing(
                    S, K, E, R, sigma, dev, 1234, rank=rank
                )
                if args.no_dist:
                    # Match dispatch: each rank solves the same deterministic routing distribution.
                    maxvio = realized
                else:
                    global_tpe = tokens_per_expert.to(torch.int64)
                    dist.all_reduce(global_tpe, group=group)
                    maxvio = (
                        global_tpe.max().item()
                        / (global_tpe.sum().item() / E)
                        - 1.0
                    )
                note = (
                    ""
                    if abs(maxvio - target) <= max(0.05 * target, 0.01)
                    else f" (target {target} unreachable, floored)"
                )
                runner.prepare(topk, tokens_per_expert)

            runner.prepare_execution_plan()
            # Report actual groups; the PUT reducer partitions source batches,
            # not equal ranges of dup_groups as the old GET prologue did.
            phase1_groups = int(runner.plan.dup_counts[0].item())
            phase1_dups = int(runner.plan.dup_counts[1].item())
            role_layout = combine_put_roles(R)
            role_summary = "/".join(str(role_layout.count(role)) for role in range(4))
            print(
                f"[rank {rank}] PUT plan: groups={phase1_groups} "
                f"dups={phase1_dups} roles={role_summary}(relay/reduce/sender/final-only) "
                f"source_lanes=16 aiv={COMBINE_PUT_CORES} batch_rows=64",
                flush=True,
            )

            (
                local_hidden_rows,
                local_weight_values,
                local_server_intra_hidden_rows,
            ) = runner.local_remote_traffic()
            counts = torch.tensor(
                [
                    local_hidden_rows,
                    local_weight_values,
                    local_server_intra_hidden_rows,
                ],
                dtype=torch.int64,
                device=dev,
            )
            if stats is None:
                gathered_counts = [torch.empty_like(counts) for _ in range(R)]
                dist.all_gather(gathered_counts, counts, group=group)
                counts_by_rank = torch.stack(gathered_counts).cpu()
            else:
                counts_by_rank = torch.tensor(
                    tcp_all_gather(stats, counts.cpu().tolist()), dtype=torch.int64)
            counts = counts_by_rank[:, :2].sum(dim=0)
            remote_hidden_rows, remote_weight_values = [
                int(value) for value in counts
            ]
            if args.check:
                check_nonzero(runner, group, stats=stats)
            timing = time_op(
                runner.combine,
                group,
                rank,
                warmup=args.warmup,
                iters=args.iters,
                verbose=False,
                stats=stats,
            )
            remote_hidden_bytes = remote_hidden_rows * H * 2
            remote_weight_bytes = remote_weight_values * 4
            server_intra_hidden_rows = int(counts_by_rank[:, 2].sum().item())
            server_intra_hidden_bytes = server_intra_hidden_rows * H * 2
            remote_bytes = remote_hidden_bytes + remote_weight_bytes
            logical_hidden_bytes = S * K * H * 2
            mean_us = timing["mean_us"]
            max_us = timing["max_us"]
            if not args.no_weights and remote_weight_values > 0:
                remote_dedup_ratio = remote_hidden_rows / remote_weight_values
                dedup_saved_bytes = (
                    remote_weight_values - remote_hidden_rows
                ) * H * 2
            else:
                remote_dedup_ratio = ""
                dedup_saved_bytes = ""

            case_row = {
                "lib": "moonep-udma-combine",
                "combine_plan_mode": "planning",
                "combine_plan_prepare_rank0_us": runner.execution_plan_prepare_us,
                **planning_timing_csv(runner.planning_timing_summary),
                "server_dedup": runner.plan.dup_counts.numel() == R + 2,
                "ep": R,
                "S": S,
                "E": E,
                "H": H,
                "K": K,
                "qp": args.active_qp_count,
                "with_weights": not args.no_weights,
                "sigma": sigma,
                "target_maxvio": target,
                "maxvio": maxvio,
                "peak_avg_load": 1.0 + maxvio,
                "combine_mean_us": mean_us,
                "combine_max_us": max_us,
                "logical_hidden_gbps": gbps(logical_hidden_bytes, max_us),
                "aggregate_remote_payload_gbps": gbps(remote_bytes, max_us),
                "remote_hidden_gbps_per_rank": (
                    gbps(remote_hidden_bytes, max_us) / R
                ),
                "remote_weight_gbps_per_rank": (
                    gbps(remote_weight_bytes, max_us) / R
                ),
                "remote_total_gbps_per_rank": gbps(remote_bytes, max_us) / R,
                "aggregate_remote_data_mb": remote_bytes / 1e6,
                "remote_hidden_mb": remote_hidden_bytes / 1e6,
                "remote_weight_mb": remote_weight_bytes / 1e6,
                "remote_dedup_ratio": (
                    round(remote_dedup_ratio, 4)
                    if remote_dedup_ratio != ""
                    else ""
                ),
                "dedup_saved_mb": (
                    dedup_saved_bytes / 1e6
                    if dedup_saved_bytes != ""
                    else ""
                ),
                "remote_hidden_rows": remote_hidden_rows,
                "remote_weight_values": remote_weight_values,
                "server_intra_hidden_rows": server_intra_hidden_rows,
                "server_intra_plan_hidden_bytes": server_intra_hidden_bytes,
                "per_rank_us": ";".join(
                    f"{value:.3f}" for value in timing["per_rank_us"]
                ),
            }
            rows.append(case_row)
            # Persist completed measurements before SHMEM resource teardown.
            if rank == 0 and args.out:
                write_csv(args.out, rows)
                print(f"[result] checkpoint CSV={args.out}", flush=True)
            if rank == 0:
                dedup_str = (
                    f"{remote_dedup_ratio:.3f}"
                    if remote_dedup_ratio != ""
                    else "n/a"
                )
                print(
                    f"completed Peak/Avg={1.0 + maxvio:.3f}x{note} "
                    f"Latency={max_us:.1f} us "
                    f"Logical_BW/rank={gbps(logical_hidden_bytes, max_us):.1f} GB/s "
                    f"Remote_Hidden_BW/rank="
                    f"{gbps(remote_hidden_bytes, max_us) / R:.1f} GB/s "
                    f"Remote_Weight_BW/rank="
                    f"{gbps(remote_weight_bytes, max_us) / R:.3f} GB/s "
                    f"Remote_Total_BW/rank="
                    f"{gbps(remote_bytes, max_us) / R:.1f} GB/s "
                    f"Dedup={dedup_str}",
                    flush=True,
                )
    finally:
        if ShmemRuntime.is_initialized():
            ShmemRuntime.finalize()
        runner.destroy()
        reset_symmetric_descriptors()

    if rank == 0 and rows:
        print_summary_table(rows)
    if rank == 0 and args.out:
        write_csv(args.out, rows)
        print(f"wrote {args.out}")
    if rank == 0 and args.plot:
        plot_rows(rows, args.plot)
        print(f"wrote {args.plot}")
    if not args.no_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
