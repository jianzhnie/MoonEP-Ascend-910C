"""Unfused BF16 MoonEP + grouped GEMM + SwiGLU + shared expert forward.

Static routing and resident expert replicas: planning/weight distribution excluded.
Run with run_bench.sh; see docs/moonep_megamoe_benchmark.md.
"""
import argparse
import csv
import math
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu

from _bench_common import setup, check_all_ranks, time_kernel
from npu_patch import Buffer, ShmemRuntime
from npu_patch.launchers import launch_dispatch, launch_combine, launch_inter_rank_sync
from test_npu.planing_reference import reference_plan_and_cu_seqlens


def weights(expert, h, intermediate, dev):
    # Identical global expert ids have identical weights on every rank.
    g = torch.Generator(device=dev).manual_seed(81000 + expert)
    w1 = torch.randn((h, 2 * intermediate), device=dev, dtype=torch.bfloat16, generator=g) / math.sqrt(h)
    w2 = torch.randn((intermediate, h), device=dev, dtype=torch.bfloat16, generator=g) / math.sqrt(intermediate)
    return w1, w2


def expert_forward(x, w1, w2):
    gate, up = (x @ w1).chunk(2, dim=-1)
    return (F.silu(gate) * up) @ w2


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('tokens', 'hidden', 'intermediate', 'experts'):
        p.add_argument('--' + name, type=int, required=True)
    p.add_argument('--topk', type=int, default=6)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--peak-tflops', type=float, default=376.0,
                   help='Assumed BF16 dense peak per logical device; utilization is conditional on this value.')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    rank, world, _ = setup()
    S,H,I,E,K = a.tokens,a.hidden,a.intermediate,a.experts,a.topk
    assert E % world == 0 and 0 < K <= E and min(S,H,I,a.iters,a.peak_tflops)>0
    dev = f'npu:{torch.npu.current_device()}'
    group = dist.group.WORLD
    torch.manual_seed(1234 + rank)
    generator = torch.Generator(device=dev).manual_seed(23000 + rank)
    scores = torch.randn((S,E),device=dev,generator=generator)
    values, routes = torch.topk(scores,K,dim=-1,sorted=False)
    routes=routes.int()
    rw=values.softmax(-1)
    tpe=torch.bincount(routes.flatten().long(),minlength=E).int()
    x=torch.randn((S,H),device=dev,dtype=torch.bfloat16)
    buf=Buffer(S,H,K,E,world,B=E//world,num_sms=32,group=group,use_udma=False,explicitly_destroy=True)
    ctx=buf._require_ctx()
    ShmemRuntime.init_with_buffer(group=group,enable_udma=False)
    buf._ensure_shmem_views()
    try:
        plan,cu=reference_plan_and_cu_seqlens(ctx,routes,tpe)
        ends=cu.cpu().tolist()
        copies=plan.experts_to_copy[rank].cpu().tolist()
        ids=[]; boundaries=[];start=0
        for g,end in enumerate(ends):
            if end>start:
                ids.append(g if g<E else copies[g-E]);boundaries.append(end)
            start=end
        assert ends[-1]==S*K
        print(f'rank={rank} active_experts={len(ids)} rows={ends[-1]}',flush=True)
        pairs=[weights(e,H,I,dev) for e in ids]
        w1=torch.stack([w[0] for w in pairs]);w2=torch.stack([w[1] for w in pairs])
        del pairs
        group_list=torch.tensor(boundaries,device=dev,dtype=torch.int64)
        shared1,shared2=weights(E,H,I,dev)
        out=torch.empty_like(x)
        result=None

        def forward():
            nonlocal result
            launch_dispatch(ctx,x,rw,plan,build_dedup_map=False)
            first=torch_npu.npu_grouped_matmul([ctx['hidden_buf_local']],[w1],group_list=group_list,split_item=3,group_type=0)[0]
            gate,up=first.chunk(2,dim=-1)
            act=F.silu(gate)*up
            y=torch_npu.npu_grouped_matmul([act],[w2],group_list=group_list,split_item=3,group_type=0)[0]
            y.mul_(ctx['weights_buf_local'].view(torch.float32).reshape(-1,1))
            ctx['hidden_buf_local'].copy_(y)
            launch_inter_rank_sync(ctx)
            launch_combine(ctx,out,plan)
            result=out+expert_forward(x,shared1,shared2)

        forward()
        torch.npu.synchronize()
        # Independent token-level oracle bypasses dispatch, group layout and combine.
        def check():
            reference=torch.zeros((1,H),device=dev,dtype=torch.float32)
            for e,rweight in zip(routes[0].cpu().tolist(),rw[0].cpu().tolist()):
                p1,p2=weights(e,H,I,dev)
                term=expert_forward(x[:1],p1,p2)
                reference+=(term.float()*rweight).bfloat16().float()
                del p1,p2,term
            reference=reference.bfloat16()+expert_forward(x[:1],shared1,shared2)
            torch.testing.assert_close(result[:1].float(),reference.float(),rtol=0.05,atol=0.05)
            assert bool(torch.isfinite(result).all())
        check_all_ranks(check,group)
        if rank==0: print('FULL FORWARD token oracle + finite PASS',flush=True)
        timing=time_kernel(forward,group,rank,warmup=a.warmup,iters=a.iters,agg='max',cudagraph=False)
        # Mean routed work per rank is S*K, plus one local shared expert.
        flops=6*H*I*S*(K+1)
        tf=flops/timing['us']/1e6
        row=dict(ep=world,experts=E,hidden=H,intermediate=I,tokens=S,topk=K,shared_experts=1,
                 forward_us_max_rank=timing['us'],gemm_flops_per_rank=flops,tflops_per_rank=tf,
                 assumed_peak_tflops_per_rank=a.peak_tflops,utilization_pct_assumed_peak=100*tf/a.peak_tflops,
                 warmup=a.warmup,iters=a.iters,validation='token_oracle_and_all_finite_PASS')
        if rank==0:
            dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('w') as f:
                writer=csv.DictWriter(f,fieldnames=row.keys());writer.writeheader();writer.writerow(row)
            print(row,flush=True)
    finally:
        torch.npu.synchronize()
        ShmemRuntime.finalize()
        buf.destroy()
        dist.destroy_process_group()


if __name__=='__main__':
    main()
