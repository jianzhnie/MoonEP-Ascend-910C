"""Single-process host-planner latency, including input/output transfers.

This models R logical ranks and is not an R-rank collective measurement.
"""

import argparse
import csv
import time

import torch
import torch_npu  # noqa: F401

from npu_patch.planning_host import build_plan_host


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'npu'), default='cpu')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--out')
    args = parser.parse_args()
    if args.warmup < 0 or args.iters <= 0:
        parser.error('warmup must be nonnegative and iters positive')
    if args.device == 'npu':
        torch.npu.set_device(0)
    rows = []
    for tokens in (64, 1024, 4096, 8192):
        world, experts, topk = 8, 128, 8
        generator = torch.Generator().manual_seed(1234)
        routes = torch.randint(experts, (world, tokens * topk), generator=generator,
                               dtype=torch.int32)
        counts = torch.stack([torch.bincount(r.long(), minlength=experts)
                              for r in routes]).int().to(args.device)
        route = routes[0].to(args.device)

        def run():
            plan = build_plan_host(counts, counts[0], route, 0,
                                   experts // world, tokens * topk)
            if args.device == 'npu':
                torch.npu.synchronize()
            return plan

        for _ in range(args.warmup):
            run()
        start = time.perf_counter()
        for _ in range(args.iters):
            plan = run()
        elapsed_us = (time.perf_counter() - start) * 1e6 / args.iters
        assert int(plan['cu_seqlens'][-1]) == tokens * topk
        row = dict(device=args.device, R=world, E=experts, S=tokens, K=topk,
                   warmup=args.warmup, iters=args.iters, e2e_us=elapsed_us)
        rows.append(row)
        print(row, flush=True)
    if args.out:
        with open(args.out, 'w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
