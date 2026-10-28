"""Minimal end-to-end MoonEP Buffer example for A2/A3 (no device planner).

Run inside the ``hyper-parallel-env`` container from the repository root:

    export ASCEND_MOONEP_HEAP=shmem HCCL_NPU_SOCKET_PORT_RANGE=61000-61031
    torchrun --nproc_per_node=2 test_npu/mte/example_a3_buffer_usage.py \
        --S 64 --K 4 --E 8 --H 128

On A5 (Ascend950) the plan comes from the device planner
(``launch_planning``); on A2/A3 that planner is SIMT-only and not built, so the
plan is produced by the host reference (``test_npu/planing_reference.py``) and
handed to dispatch/combine. Everything else (Buffer, shmem heap, kernels) is the
same code path the A5 flow uses.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  (registers the npu device)

from npu_patch import Buffer, ShmemRuntime
from npu_patch._device import local_device_index
from test_npu.generate_topk_routing import generate_topk_routing
from test_npu.planing_reference import reference_plan_and_cu_seqlens


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--S", type=int, default=64, help="tokens per rank")
    p.add_argument("--K", type=int, default=4, help="experts per token (topk)")
    p.add_argument("--E", type=int, default=8, help="total experts")
    p.add_argument("--H", type=int, default=128, help="hidden size")
    p.add_argument("--bias-ratio", type=float, default=1.0,
                   help="lognormal routing sigma (0 = balanced)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--num-sms", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    dist.init_process_group("hccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    group = dist.group.WORLD
    device_id = local_device_index(rank)
    torch.npu.set_device(device_id)
    dev = f"npu:{device_id}"

    R, S, K, E, H = world_size, args.S, args.K, args.E, args.H
    assert E % R == 0, f"E ({E}) must be divisible by world_size ({R})"
    B = E // R  # per-rank expert capacity (moonep composite layout requires B == epn)

    # 1) Buffer owns the symmetric chunks; heap mode (ASCEND_MOONEP_HEAP=shmem)
    #    takes them from shmem's ordinary heap. Construct it first, then let
    #    ShmemRuntime initialize (a no-op in heap mode, required in VMM mode).
    buf = Buffer(S, H, K, E, R, B=B, num_sms=args.num_sms, use_udma=False,
                 group=group, explicitly_destroy=True)
    if not ShmemRuntime.is_initialized():
        ShmemRuntime.init_with_buffer(group=group, enable_udma=False)
    ctx = buf._require_ctx()

    try:
        # 2) Plan: A3 has no device planner, so build it with the host reference.
        #    Regenerate it whenever the routing changes.
        topk, tpe = generate_topk_routing(S, K, E, R, args.bias_ratio, dev,
                                          args.seed, rank=rank, all_ranks=True)
        plan, _cu_seqlens = reference_plan_and_cu_seqlens(ctx, topk, tpe)

        # 3) dispatch: hidden [S, H] bf16 + route weights [S, K] fp32 -> expert-major
        #    rows on this rank. The expert GEMMs would run here on another stream.
        hidden = torch.randn(S, H, dtype=torch.bfloat16, device=dev)
        weights = torch.rand(S, K, dtype=torch.float32, device=dev)
        hidden_nvsh, weights_nvs, _cu, plan = buf.dispatch(
            hidden, weights, plan=plan, zero_copy=True)

        # 4) combine: gather the expert outputs back to token-major [S, H].
        #    c_f (hidden only) returns no weights; c_b also gathers the route
        #    weights, so it takes the dispatched [NvS] weights back as input.
        out, no_weights, _event = buf.combine(plan=plan, hidden_nvsh=hidden_nvsh)
        out_weighted, gathered_weights, _event = buf.combine(
            plan=plan, hidden_nvsh=hidden_nvsh, route_weights_nvs=weights_nvs)
        torch.npu.synchronize()

        ok = (tuple(out.shape) == (S, H)
              and no_weights is None
              and tuple(out_weighted.shape) == (S, H)
              and gathered_weights is not None
              and bool(torch.isfinite(out).all().item())
              and tuple(gathered_weights.shape) == (S, K))
        print(f"[rank {rank}] dispatch hidden {tuple(hidden_nvsh.shape)} / "
              f"weights {tuple(weights_nvs.shape)}; combine c_f {tuple(out.shape)} / "
              f"c_b {tuple(out_weighted.shape)} + gathered "
              f"{tuple(gathered_weights.shape)}; finite={ok}", flush=True)
        if not ok:
            raise SystemExit(f"[rank {rank}] unexpected output shapes/values")

    finally:
        torch.npu.synchronize()
        ShmemRuntime.finalize()
        buf.destroy()
        dist.destroy_process_group()
    if rank == 0:
        print("EXAMPLE PASS", flush=True)


if __name__ == "__main__":
    main()
