"""Run with torchrun --standalone --nproc_per_node=4 -m pytest -q this_file."""

import os

import pytest
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from npu_patch import ShmemRuntime, buffer_c


def test_nonzero_root_ep_group():
    if int(os.environ.get('WORLD_SIZE', '1')) < 4:
        pytest.skip('requires four torchrun ranks')
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl')
    group = dist.new_group([2, 3])
    try:
        if dist.get_rank() in (2, 3):
            buffer_c.ensure_shmem_heap(4 * 1024 * 1024, group=group)
            ShmemRuntime.init_with_buffer(group=group, enable_udma=False)
            tensor, _, _ = buffer_c.create_vmm_physical([1024], torch.int32,
                                                        torch.npu.current_device())
            tensor.fill_(dist.get_rank())
            torch.npu.synchronize()
            assert buffer_c._HEAP_STATE['scope'][2:] == (dist.get_rank(group), 2)
            with pytest.raises(RuntimeError, match='scope'):
                ShmemRuntime.init_with_buffer(group=None, enable_udma=False)
            ShmemRuntime.finalize()
            tensor._vmm_allocation.destroy()
        dist.barrier()
    finally:
        dist.destroy_process_group()
