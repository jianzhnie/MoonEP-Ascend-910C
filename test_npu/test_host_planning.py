"""CPU contract checks for the compiled host planner; no NPU allocations."""

import pytest
import torch

from npu_patch import _C


@pytest.mark.parametrize("world,epn", [(2, 1), (4, 2), (8, 4)])
def test_host_plan_conserves_rows(world, epn):
    generator = torch.Generator().manual_seed(1234)
    cap = 64
    experts = world * epn
    for _ in range(20):
        popularity = torch.rand(experts, generator=generator).pow(8)
        routes = torch.multinomial(popularity, world * cap, replacement=True,
                                   generator=generator).reshape(world, cap).int()
        counts = torch.stack([torch.bincount(r.long(), minlength=experts)
                              for r in routes]).int()
        destinations = []
        for rank in range(world):
            plan = _C.plan_host(counts, counts[rank].contiguous(), routes[rank],
                                rank, epn, cap)
            dst = plan["dst"].long()
            assert bool(((dst >= 0) & (dst < world * cap)).all())
            assert int(plan["cu_seqlens"][-1]) == cap
            assert torch.equal(plan["alloc"].sum(1), torch.full((world,), cap))
            assert torch.equal(plan["alloc"].sum(0), counts.sum(0))
            destinations.append(dst)
        assert torch.equal(torch.cat(destinations).sort().values,
                           torch.arange(world * cap))


def test_host_plan_rejects_stale_histogram():
    counts = torch.tensor([[1, 1], [1, 1]], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="histogram"):
        _C.plan_host(counts, counts[0], torch.tensor([0, 0], dtype=torch.int32), 0, 1, 2)


def test_host_plan_rejects_mismatched_local_counts():
    counts = torch.tensor([[1, 1], [1, 1]], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="counts_all"):
        _C.plan_host(counts, torch.tensor([2, 0], dtype=torch.int32),
                     torch.tensor([0, 0], dtype=torch.int32), 0, 1, 2)


def test_host_plan_rejects_negative_counts():
    counts = torch.tensor([[1, 1], [-1, 3]], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="nonnegative"):
        _C.plan_host(counts, counts[0], torch.tensor([0, 1], dtype=torch.int32), 0, 1, 2)
