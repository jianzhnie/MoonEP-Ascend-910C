"""CPU regression for the all2all combine contract (mocked transport)."""

import sys
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'npu_patch' / 'benchmarks'))
from bench_vs_all2all import All2AllRunner  # noqa: E402


def test_combine_backward_gathers_weights_without_weighting_hidden():
    runner = All2AllRunner.__new__(All2AllRunner)
    runner.S, runner.K, runner.H = 1, 2, 1
    runner.group = None
    runner.order = torch.tensor([1, 0])
    runner.inv_perm = torch.tensor([0, 1])
    runner.x_sorted = torch.tensor([[2.0], [1.0]], dtype=torch.bfloat16)
    runner.w_sorted = torch.tensor([[0.75], [0.25]])
    runner.y_back = torch.empty_like(runner.x_sorted)
    runner.w_back = torch.empty_like(runner.w_sorted)
    runner.out_sh = torch.empty((1, 1), dtype=torch.bfloat16)
    runner.send_counts_list = runner.recv_counts_list = [2]

    def transport(output, source, **kwargs):
        output.copy_(source)

    with patch('torch.distributed.all_to_all_single', side_effect=transport):
        runner.combine_bwd()
    assert runner.out_sh.item() == 3.0
    torch.testing.assert_close(runner.w_back.flatten(), torch.tensor([0.25, 0.75]))


def test_mutating_kernel_restores_input_outside_timed_events():
    from contextlib import ExitStack
    from _bench_common import time_kernel

    calls = []

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            calls.append('event')

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 1.0

    def gather(outputs, source, **kwargs):
        outputs[0].copy_(source)

    tensor = torch.tensor
    with ExitStack() as stack:
        stack.enter_context(patch('torch.npu.Event', Event))
        stack.enter_context(patch('torch.npu.synchronize'))
        stack.enter_context(patch('torch.npu.current_device', return_value=0))
        stack.enter_context(patch('torch.distributed.barrier'))
        stack.enter_context(patch('torch.distributed.get_world_size', return_value=1))
        stack.enter_context(patch('torch.distributed.all_gather', side_effect=gather))
        stack.enter_context(patch('torch.tensor', side_effect=lambda data, **kw:
                                  tensor(data, dtype=kw.get('dtype'))))
        result = time_kernel(lambda: calls.append('kernel'), None, 0,
                             warmup=1, iters=2, cudagraph=True,
                             percentiles=True,
                             prepare=lambda: calls.append('prepare'))
    assert calls == ['prepare', 'kernel'] + ['prepare', 'event', 'kernel', 'event'] * 2
    assert result == {'status': 'ok', 'us': 1000.0, 'p50': 1000.0, 'p90': 1000.0}
