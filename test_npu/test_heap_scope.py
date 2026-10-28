"""Heap bootstrap guards, exercised without allocating device memory."""

from unittest.mock import patch

import pytest

from npu_patch import buffer_c


def test_allocation_requires_explicit_bootstrap():
    with patch.dict(buffer_c._HEAP_STATE, inited=False), \
         patch.object(buffer_c, 'acl_shmem_heap_alloc') as allocate:
        with pytest.raises(RuntimeError, match="ensure_shmem_heap"):
            buffer_c._heap_alloc(1024)
        allocate.assert_not_called()


def test_heap_rejects_scope_change():
    scope = (None, 0, 0, 1)
    with patch.dict(buffer_c._HEAP_STATE, inited=True, scope=scope), \
         patch('torch.distributed.is_initialized', return_value=False):
        buffer_c.ensure_shmem_heap(0, device_id=0, rank=0, world_size=1)
        for device, rank, world in [(1, 0, 1), (0, 1, 2), (0, 0, 2)]:
            with pytest.raises(RuntimeError, match="scope"):
                buffer_c.ensure_shmem_heap(0, device_id=device, rank=rank, world_size=world)
