"""MoonEP-Ascend dispatch-only benchmark with a MaxVio sweep.

This is the dispatch counterpart of ``bench_grad_reduce_maxvio.py``.  Every
case first runs the real planning kernel to materialize ``MoonEPCommPlan``;
the timed loop then reuses that plan and contains only the fused dispatch plus
local duplicate-expansion kernel.

The report contains two bandwidth views:

* ``logical_hidden_gbps`` uses ``S*K*H*2 / mean-rank latency``.
* ``aggregate_remote_payload_gbps`` decodes ``plan.dst`` and counts bytes that
  all EP ranks actually send to another rank.  Canonical hidden rows and fp32
  route weights are counted; local writes and local duplicate expansion are
  excluded.  This is aggregate EP traffic divided by slowest-rank latency.

EP group = the whole torchrun world.  Examples:

    # One node, 8 ranks (UID bootstrap)
    torchrun --nproc-per-node=8 npu_patch/benchmarks/bench_dispatch_maxvio.py

    # One node, 8 ranks, torchrun launcher without a process group
    bash npu_patch/benchmarks/run_bench.sh bench_dispatch_maxvio.py "--no-dist"

    # Two nodes, 8 ranks per node (run on each node with NODE_RANK=0/1)
    MASTER_ADDR=192.168.1.10 NODES=2 NODE_RANK=0 \
        bash npu_patch/benchmarks/run_bench.sh bench_dispatch_maxvio.py

    # Explicit SHMEM endpoint
    torchrun --nnodes=2 --nproc-per-node=8 --node-rank=0 \
        --master-addr=192.168.1.10 --master-port=29500 \
        npu_patch/benchmarks/bench_dispatch_maxvio.py \
        --ip-port tcp://192.168.1.10:29501
"""

import argparse
import csv
import json
import os
from datetime import timedelta

import torch
import torch_npu
import torch.distributed as dist

from _bench_common import (
    add_engine_args,
    apply_engine,
    generate_topk_routing,
    setup,
)

from npu_patch import Buffer, ShmemRuntime
from npu_patch._device import local_device_index
from npu_patch.buffer_c import reset_symmetric_descriptors
from npu_patch.launchers import launch_dispatch, launch_dispatch_profile, launch_planning
from npu_patch.planning import allocate_planning_outputs
from npu_patch.shmem_runtime import default_shmem_ip_port


def gbps(nbytes: int, us: float) -> float:
    return nbytes / (us * 1e-6) / 1e9 if us > 0 else 0.0


def print_box_table(headers, rows, alignments, min_widths):
    """Print a two-line header and a border after every row."""
    header_lines = [header.split("\n") for header in headers]
    widths = [max([min_widths[i]] + [len(line) for line in header_lines[i]] +
                  [len(str(row[i])) for row in rows]) for i in range(len(headers))]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def format_row(values, align):
        cells = []
        for value, width, side in zip(values, widths, align):
            cell = str(value)
            if side == "left":
                cell = cell.ljust(width)
            elif side == "center":
                cell = cell.center(width)
            else:
                cell = cell.rjust(width)
            cells.append(f" {cell} ")
        return "|" + "|".join(cells) + "|"

    print(border)
    for line in range(max(len(lines) for lines in header_lines)):
        values = tuple(lines[line] if line < len(lines) else "" for lines in header_lines)
        print(format_row(values, ("center",) * len(headers)))
    print(border)
    for row in rows:
        print(format_row(row, alignments))
        print(border)


def summarize_profile(profile, cycles_per_us, duplicate_rows_by_rank, hidden_size):
    """Take each rank max AIV time and total bytes, then average across ranks."""
    matrix = profile.tolist()
    stages = (("token", 1, 4), ("weight", 6, 7), ("duplicate", 8, None), ("operator", 0, 4))
    result = {name: {"time": [], "bw": []} for name, _, _ in stages}
    invalid_rows = 0
    # Zero out -1 fields that were never written back; mark Duplicate bandwidth NaN on incomplete ranks.
    for rank, rank_rows in enumerate(matrix):
        rank_invalid = any(value == -1 for row in rank_rows for value in row)
        invalid_rows += sum(any(value == -1 for value in row) for row in rank_rows)
        valid_rows = [[0 if value == -1 else value for value in row] for row in rank_rows]
        # Every bandwidth uses this rank stage max time as the denominator; Duplicate additionally checks complete
        # write-back.
        for name, cycle_field, byte_field in stages:
            stage_us = max(row[cycle_field] for row in valid_rows) / cycles_per_us
            if byte_field is None:
                byte_count = duplicate_rows_by_rank[rank] * hidden_size * 2
            else:
                byte_count = sum(row[byte_field] for row in valid_rows)
            if name == "duplicate" and (rank_invalid or (stage_us == 0 and byte_count != 0)):
                bandwidth = float("nan")
            else:
                bandwidth = byte_count / stage_us / 1000 if stage_us > 0 else 0.0
            result[name]["time"].append(stage_us)
            result[name]["bw"].append(bandwidth)
    for name, _, _ in stages:
        result[name] = (sum(result[name]["time"]) / len(matrix), sum(result[name]["bw"]) / len(matrix))
    result["invalid_rows"] = invalid_rows
    return result


def print_summary_tables(rows, profiles, dedup_mode):
    """Print the algorithm bandwidth table and the four-stage Kernel Timing table."""
    first = rows[0]
    print("MoonEP Dispatch UDMA Summary")
    print(f"EP={first['ep']} S={first['S']} E={first['E']} H={first['H']} K={first['K']} "
          f"QP=2 Weights={'on' if first['with_weights'] else 'off'} plan_mode=maxvio dedup_mode={dedup_mode}")
    print()
    summary_rows = []
    for row in rows:
        dedup = row["remote_dedup_ratio"]
        summary_rows.append((
            f"{1 + row['maxvio']:.3f}x", f"{row['dispatch_max_us']:.1f}",
            f"{row['logical_hidden_gbps']:.1f}", f"{row['remote_token_gbps_per_rank']:.1f}",
            f"{row['remote_weight_gbps_per_rank']:.3f}", f"{row['remote_total_gbps_per_rank']:.1f}",
            f"{dedup:.3f}" if dedup is not None else "n/a",
        ))
    print_box_table(
        ("Peak/Avg\n", "Latency\n(us)", "Logical BW\n(GB/s/rank)", "Remote Token\n(GB/s/rank)",
         "Remote Weight\n(GB/s/rank)", "Remote Total\n(GB/s/rank)", "Dedup\n"),
        summary_rows, ("right",) * 7, (8, 8, 11, 12, 13, 12, 5),
    )
    print("Latency / Remote BW use max_us; Logical BW uses mean_us.")
    print("\nKernel Timing Summary\n")
    stage_rows = []
    for case, (row, profile) in enumerate(zip(rows, profiles)):
        for index, (name, key) in enumerate((
            ("Token Scatter", "token"), ("Weight Scatter", "weight"),
            ("Duplicate", "duplicate"), ("Operator", "operator"),
        )):
            stage_us, stage_bw = profile[key]
            stage_rows.append((f"{1 + row['maxvio']:.3f}x" if index == 0 else "", name,
                               f"{stage_us:.1f}", f"{stage_bw:.3f}"))
        if case + 1 < len(rows):
            stage_rows.append(("", "", "", ""))
    print_box_table(
        ("Peak/Avg", "Stage", "Avg Rank Time\n(us)", "Avg Rank BW\n(GB/s)"),
        stage_rows, ("right", "left", "right", "right"), (13, 19, 23, 23),
    )
    invalid_rows = sum(profile["invalid_rows"] for profile in profiles)
    if invalid_rows:
        print(f"WARNING: {invalid_rows} invalid profile row(s)")


def time_op(fn, group, rank, warmup=20, iters=50, stats=None,
            remote_counts=(0, 0, 0)):
    """Return per-rank, mean, and slowest-rank eager NPU-event latency."""
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
        return stats.collect(local_us, *remote_counts)

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

    if rank == 0:
        print(
            f"  [time_op] per-rank: {[f'{value:.1f}' for value in times]} "
            f"max={max_us:.1f} mean={mean_us:.1f} us",
            flush=True,
        )

    dist.barrier(group=group)
    return {"per_rank_us": times, "mean_us": mean_us, "max_us": max_us}


def _maxvio_for_sigma(sigma, S, K, E, R, dev):
    total = torch.zeros(E, dtype=torch.int64, device=dev)
    for rank in range(R):
        _, tpe = generate_topk_routing(
            S, K, E, R, sigma, dev, 1234, rank=rank
        )
        total += tpe.to(torch.int64)
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


class DispatchRunner:
    """Own the Buffer, plan, and fixed dispatch inputs for one benchmark."""

    def __init__(self, group, R, S, K, E, H, num_sms, local_rank, with_weights,
                 rank=None, use_udma=False, dedup_mode="rank",
                 plan_source="kernel"):
        self.group = group
        # 'kernel': device planner (Ascend950/SIMT only); 'golden': host plan via
        # test_npu/planing_reference -- the only way to run on A2/A3.
        self.plan_source = plan_source
        self.rank = rank if rank is not None else dist.get_rank(group=group)
        self.R = R
        self.S = S
        self.K = K
        self.H = H
        self.with_weights = with_weights
        self.use_udma = use_udma
        self.buffer = Buffer(
            S,
            H,
            K,
            E,
            R,
            num_sms=num_sms,
            group=group,
            use_udma=use_udma,
            explicitly_destroy=True,
            rank=rank,
        )
        self.ctx = self.buffer._require_ctx()
        self.plan, self.cu_seqlens = allocate_planning_outputs(self.ctx, dedup_mode=dedup_mode)

        dev = f"npu:{local_rank}"
        generator = torch.Generator(device=dev).manual_seed(7777 + self.rank)
        self.hidden = torch.randn(
            S, H, dtype=torch.bfloat16, device=dev, generator=generator
        )
        self.weights = None
        if with_weights:
            self.weights = torch.rand(
                S, K, dtype=torch.float32, device=dev, generator=generator
            )
        self.profile_cycles = None

    def activate_symmetric_views(self):
        self.buffer._ensure_shmem_views()
        torch.npu.synchronize()
        ShmemRuntime.barrier(group=self.group)

    def prepare(self, topk, tokens_per_expert):
        """Run real planning once; no planning work is in ``dispatch``."""
        topk_flat = topk.reshape(-1).contiguous()
        if self.plan_source == "golden":
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

    def dispatch(self):
        launch_dispatch(
            self.ctx,
            self.hidden,
            self.weights,
            self.plan,
            build_dedup_map=False,
        )

    def dispatch_profile(self):
        """Run the profiling kernel alone; the AIV matrix keeps the result of the last call."""
        if self.profile_cycles is None:
            self.profile_cycles = torch.full((32, 9), -1, dtype=torch.int64, device=self.hidden.device)
        self.profile_cycles.fill_(-1)
        launch_dispatch_profile(
            self.ctx, self.hidden, self.weights, self.plan, self.profile_cycles,
            build_dedup_map=False,
        )

    def local_remote_traffic(self):
        """Decode this rank's plan.dst and count its inter-rank payload."""
        NvS = int(self.ctx["NvS"])
        encoded = self.plan.dst.to(torch.int64)
        canonical = encoded >= 0
        raw_dst = torch.where(canonical, encoded, -encoded - 1)
        dest_rank = torch.div(raw_dst, NvS, rounding_mode="floor")
        is_remote = dest_rank != self.rank

        remote_hidden_rows = (canonical & is_remote).sum(dtype=torch.int64)
        remote_weight_values = is_remote.sum(dtype=torch.int64)
        if not self.with_weights:
            remote_weight_values.zero_()
        return int(remote_hidden_rows.item()), int(remote_weight_values.item())

    def local_duplicate_rows(self):
        """Read the number of secondary output rows Planning produced for this primary rank."""
        return int(self.plan.dup_counts[1].item())

    def destroy(self):
        self.buffer.destroy()


class TcpStoreStats:
    """Aggregate per-rank timings, traffic, and profiling matrices over a CPU TCPStore."""

    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.step = 0
        # torchrun occupies MASTER_PORT and SHMEM defaults to +1; statistics use +2 on its own.
        address = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = int(os.environ.get("MASTER_PORT", "29500")) + 2
        self.store = dist.TCPStore(address, port, world_size, rank == 0, timeout=timedelta(minutes=5))

    def gather(self, value):
        """Each rank writes its data; rank 0 reads everything and then signals all ranks to continue."""
        key = f"stats/{self.step}"
        self.step += 1
        self.store.set(f"{key}/rank/{self.rank}", json.dumps(value))
        values = None
        # Let only rank 0 read the full data, so ranks do not each fetch the profiling matrices again.
        if self.rank == 0:
            values = [json.loads(self.store.get(f"{key}/rank/{peer}")) for peer in range(self.world_size)]
            self.store.set(f"{key}/release", "1")
        self.store.get(f"{key}/release")
        self.store.set(f"{key}/read/{self.rank}", "1")
        # rank 0 waits until every rank receives the release, so the last gather round completes before exit.
        if self.rank == 0:
            self.store.wait([f"{key}/read/{peer}" for peer in range(self.world_size)])
        return values

    def barrier(self):
        self.gather(None)

    def collect(self, local_us, remote_hidden_rows, remote_weight_values, duplicate_rows):
        gathered = self.gather([local_us, remote_hidden_rows, remote_weight_values, duplicate_rows])

        if self.rank == 0:
            times = [row[0] for row in gathered]
            hidden_rows = sum(row[1] for row in gathered)
            weight_values = sum(row[2] for row in gathered)
            duplicate_rows_by_rank = [row[3] for row in gathered]
        else:
            times = [local_us]
            hidden_rows = remote_hidden_rows
            weight_values = remote_weight_values
            duplicate_rows_by_rank = [duplicate_rows]

        if self.rank == 0:
            print(
                f"  [time_op] per-rank: {[f'{value:.1f}' for value in times]} "
                f"max={max(times):.1f} mean={sum(times) / len(times):.1f} us",
                flush=True,
            )

        return {
            "per_rank_us": times,
            "mean_us": sum(times) / len(times),
            "max_us": max(times),
            "remote_hidden_rows": hidden_rows,
            "remote_weight_values": weight_values,
            "duplicate_rows_by_rank": duplicate_rows_by_rank,
        }

    def collect_profile(self, local_matrix):
        """After the profiling kernel finishes, gather each rank 32x9 raw matrix on the CPU."""
        local = local_matrix.cpu()
        gathered = self.gather(local.tolist())
        return torch.tensor(gathered, dtype=local.dtype) if self.rank == 0 else None


def setup_no_dist():
    """Read torchrun rank metadata without creating a process group."""
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            f"--no-dist requires torchrun environment variables: {missing}"
        )
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_device_index(local_rank))
    return rank, world_size, local_rank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_args(parser)
    parser.add_argument(
        "--plan-source",
        choices=("kernel", "golden"),
        default="kernel",
        help="kernel: device planner (Ascend950/SIMT only); golden: build the "
             "plan on the host with test_npu/planing_reference (required on "
             "A2/A3, where the SIMT planner is not built).",
    )
    parser.add_argument(
        "--out",
        default="bench_dispatch_maxvio_results.csv",
        help="CSV output path on rank 0 (default: %(default)s)",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Optional latency figure path on rank 0",
    )
    parser.add_argument(
        "--maxvios",
        default="0.2,1,10,20",
        help="Comma-separated target MaxVio values (default: %(default)s)",
    )
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-sms", type=int, default=32)
    parser.add_argument("--dedup-mode", choices=("rank", "server"), default="rank")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--kernel-timing", action="store_true", help="collect AIV stage cycles per MaxVio case")
    parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Benchmark hidden-only dispatch instead of forward dispatch",
    )
    parser.add_argument(
        "--ip-port",
        default=None,
        help=(
            "SHMEM DEFAULT-bootstrap endpoint. Multi-node runs derive it from "
            "MASTER_ADDR and MASTER_PORT (+1) when omitted"
        ),
    )
    parser.add_argument(
        "--no-dist",
        action="store_true",
        help="Use torchrun only as a process launcher; SHMEM bootstrap and CPU TCPStore statistics",
    )
    args = parser.parse_args()

    if args.no_dist:
        rank, world_size, local_rank = setup_no_dist()
        group = None
    else:
        rank, world_size, local_rank = setup()
        group = dist.group.WORLD
    local_rank = local_device_index(local_rank)
    use_udma = apply_engine(args.engine)
    if args.kernel_timing and not use_udma:
        raise RuntimeError("--kernel-timing requires the URMA dispatch engine")
    if args.dedup_mode == "server" and not use_udma:
        raise RuntimeError("server dedup requires the URMA dispatch engine")
    R = world_size
    S, K, E, H = args.tokens, args.topk, args.experts, args.hidden
    dev = f"npu:{local_rank}"
    assert R >= 2, f"dispatch benchmark requires at least 2 ranks, got {R}"
    assert E % R == 0, f"E ({E}) must be divisible by world size ({R})"
    assert K <= E, f"K ({K}) must be <= E ({E})"
    targets = [float(value) for value in args.maxvios.split(",")]

    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    multi_node = world_size > local_world_size

    # Build the Buffer first so its VMM descriptors are registered before SHMEM initializes.
    stats = None
    runner = DispatchRunner(
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
        use_udma=use_udma,
        dedup_mode=args.dedup_mode,
        plan_source=args.plan_source,
    )
    if not ShmemRuntime.is_initialized():
        if args.no_dist:
            ip_port = args.ip_port or default_shmem_ip_port(
                os.environ.get("MASTER_ADDR", "127.0.0.1"),
                int(os.environ.get("MASTER_PORT", "29500")),
            )
            ShmemRuntime.init_with_ip_port(
                ip_port=ip_port,
                rank=rank,
                world_size=R,
                enable_udma=use_udma,
            )
        elif args.ip_port is not None or multi_node:
            ShmemRuntime.init_with_ip_port(group=group, ip_port=args.ip_port,
                                           enable_udma=use_udma)
        else:
            ShmemRuntime.init_with_buffer(group=group, enable_udma=use_udma)
    runner.activate_symmetric_views()
    if args.no_dist:
        stats = TcpStoreStats(rank, R)

    rows = []
    profiles = []
    try:
        for target in targets:
            if args.no_dist:
                sigma, realized = solve_sigma_for_maxvio(
                    target, S, K, E, R, dev
                )
            elif rank == 0:
                sigma, realized = solve_sigma_for_maxvio(
                    target, S, K, E, R, dev
                )
            else:
                sigma, realized = None, None
            if not args.no_dist:
                solved = [sigma, realized]
                dist.broadcast_object_list(solved, src=0, group=group)
                sigma, realized = solved

            topk, tpe = generate_topk_routing(
                S, K, E, R, sigma, dev, 1234, rank=rank
            )
            if args.no_dist:
                maxvio = realized
            else:
                global_tpe = tpe.to(torch.int64)
                dist.all_reduce(global_tpe, group=group)
                maxvio = (
                    global_tpe.max().item()
                    / (global_tpe.sum().item() / E)
                    - 1.0
                )
            if rank == 0:
                note = (
                    ""
                    if abs(maxvio - target) <= max(0.05 * target, 0.01)
                    else f" (target {target} unreachable, floored)"
                )
                print(
                    f"--- target MaxVio={target}: sigma={sigma:.5f}, "
                    f"realized MaxVio={maxvio:.3f}{note}",
                    flush=True,
                )

            runner.prepare(topk, tpe)
            local_hidden_rows, local_weight_values = runner.local_remote_traffic()
            local_duplicate_rows = runner.local_duplicate_rows()
            if not args.no_dist:
                counts = torch.tensor(
                    [local_hidden_rows, local_weight_values],
                    dtype=torch.int64,
                    device=dev,
                )
                dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)
                remote_hidden_rows, remote_weight_values = [
                    int(value) for value in counts.cpu()
                ]
                local_duplicate = torch.tensor([local_duplicate_rows], dtype=torch.int32, device=dev)
                gathered_duplicates = [torch.empty_like(local_duplicate) for _ in range(R)]
                dist.all_gather(gathered_duplicates, local_duplicate, group=group)
                duplicate_rows_by_rank = [int(value.item()) for value in gathered_duplicates]
            timing = time_op(
                runner.dispatch,
                group,
                rank,
                warmup=args.warmup,
                iters=args.iters,
                stats=stats,
                remote_counts=(local_hidden_rows, local_weight_values, local_duplicate_rows),
            )
            if args.no_dist:
                remote_hidden_rows = timing["remote_hidden_rows"]
                remote_weight_values = timing["remote_weight_values"]
                duplicate_rows_by_rank = timing["duplicate_rows_by_rank"]
            remote_bytes = (
                remote_hidden_rows * H * 2 + remote_weight_values * 4
            )

            logical_hidden_bytes = S * K * H * 2
            mean_us = timing["mean_us"]
            max_us = timing["max_us"]
            profile_summary = None
            if args.kernel_timing:
                # The normal Dispatch determines the algorithm bandwidth; the profiling Dispatch only provides module
                # timings.
                time_op(runner.dispatch_profile, group, rank, warmup=args.warmup,
                        iters=args.iters, stats=stats)
                if stats is not None:
                    profile = stats.collect_profile(runner.profile_cycles)
                else:
                    gathered = [torch.empty_like(runner.profile_cycles) for _ in range(R)]
                    dist.all_gather(gathered, runner.profile_cycles, group=group)
                    profile = torch.stack(gathered).cpu() if rank == 0 else None
                if rank == 0:
                    device_name = torch.npu.get_device_name(torch.npu.current_device())
                    cycle_rate = 1000.0 if "950" in device_name else 50.0
                    profile_summary = summarize_profile(profile, cycle_rate, duplicate_rows_by_rank, H)
                    profiles.append(profile_summary)
            rows.append(
                {
                    "lib": "moonep",
                    "ep": R,
                    "S": S,
                    "E": E,
                    "H": H,
                    "K": K,
                    "with_weights": not args.no_weights,
                    "sigma": sigma,
                    "target_maxvio": target,
                    "maxvio": maxvio,
                    "dispatch_mean_us": mean_us,
                    "dispatch_max_us": max_us,
                    "logical_hidden_gbps": gbps(logical_hidden_bytes, mean_us),
                    "aggregate_remote_payload_gbps": gbps(remote_bytes, max_us),
                    "remote_token_gbps_per_rank": gbps(remote_hidden_rows * H * 2, max_us) / R,
                    "remote_weight_gbps_per_rank": gbps(remote_weight_values * 4, max_us) / R,
                    "remote_total_gbps_per_rank": gbps(remote_bytes, max_us) / R,
                    "remote_dedup_ratio": (remote_hidden_rows / remote_weight_values
                                           if remote_weight_values else None),
                    "aggregate_remote_data_mb": remote_bytes / 1e6,
                    "remote_hidden_rows": remote_hidden_rows,
                    "remote_weight_values": remote_weight_values,
                    "per_rank_us": ";".join(
                        f"{value:.3f}" for value in timing["per_rank_us"]
                    ),
                }
            )
            if rank == 0:
                print(
                    f"{'moonep':>8} ep={R} S={S} E={E} H={H} K={K} "
                    f"sigma={sigma:.5f}: maxvio={maxvio:6.2f} | "
                    f"dispatch mean={mean_us:7.1f} max={max_us:7.1f} us | "
                    f"logical hidden={gbps(logical_hidden_bytes, mean_us):5.1f} "
                    f"GB/s | aggregate remote={gbps(remote_bytes, max_us):5.1f} "
                    f"GB/s ({remote_bytes / 1e6:.1f} MB)",
                    flush=True,
                )
    finally:
        if ShmemRuntime.is_initialized():
            ShmemRuntime.finalize()
        runner.destroy()
        reset_symmetric_descriptors()

    if rank == 0 and args.out:
        with open(args.out, "w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.out}")
    if rank == 0 and args.plot:
        plot_rows(rows, args.plot)
        print(f"wrote {args.plot}")
    if rank == 0 and args.kernel_timing:
        print("BENCH_TABLE_BEGIN", flush=True)
        print_summary_tables(rows, profiles, args.dedup_mode)
        print("BENCH_TABLE_END", flush=True)

    if not args.no_dist:
        dist.destroy_process_group()


def plot_rows(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [f"{row['maxvio']:.2g}" for row in rows]
    xpos = np.arange(len(rows))
    means = [row["dispatch_mean_us"] for row in rows]
    maxima = [row["dispatch_max_us"] for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.36
    ax.bar(xpos - width / 2, means, width, label="rank mean")
    ax.bar(xpos + width / 2, maxima, width, label="slowest rank")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_xlabel("realized MaxVio")
    ax.set_ylabel("dispatch latency (us)")
    ax.set_title(
        f"MoonEP-Ascend dispatch, EP={rows[0]['ep']}, "
        f"S={rows[0]['S']}, K={rows[0]['K']}, H={rows[0]['H']}"
    )
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
