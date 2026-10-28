"""A2/A3 host planner fast path.

The Ascend 950 planner is a SIMT device kernel and is not built for 910C.
``_C.plan_host`` implements the same plan on the host in C++, avoiding the
per-token Python loops of ``test_npu.planing_reference``.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["build_plan_host", "available"]


def _extension() -> Any:
    """Return the compiled ``_C`` extension (lazy import)."""
    from . import _C  # pylint: disable=import-outside-toplevel
    return _C


def available() -> bool:
    """Whether the compiled host planner is available."""
    try:
        return hasattr(_extension(), "plan_host")
    except ImportError:
        return False


def build_plan_host(
    counts_all: torch.Tensor,
    local_counts: torch.Tensor,
    route_ids: torch.Tensor,
    rank: int,
    epn: int,
    nvs: int,
    route_ids_cpu: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Build a MoonEP plan on the host and return tensors on the input device.

    Args:
        counts_all: ``[R, E]`` routed rows per rank and expert.
        local_counts: This rank's ``[E]`` row counts.
        route_ids: ``[N]`` global expert id per routed row.
        rank: This rank's EP index.
        epn: Local experts per rank.
        nvs: Per-rank receive capacity (``S * K``).
        route_ids_cpu: Optional pre-copied CPU int32 ``route_ids`` (e.g. from a
            caller-side fingerprint), skipping one D2H copy.

    Returns:
        The same fields as ``build_moonep_plan`` in HyperParallel.
    """
    device = counts_all.device

    def _cpu_i32(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().to(device="cpu", dtype=torch.int32).contiguous()

    if route_ids_cpu is None:
        route_ids_cpu = _cpu_i32(route_ids)
    raw = _extension().plan_host(
        _cpu_i32(counts_all),
        _cpu_i32(local_counts),
        route_ids_cpu,
        int(rank),
        int(epn),
        int(nvs),
    )
    result: dict[str, Any] = {}
    for name in (
        "dst",
        "cu_seqlens",
        "experts_to_copy",
        "remote_stats",
        "zero_fill_ranges",
        "alloc",
        "needed_remote",
    ):
        result[name] = raw[name].to(device)
    result["local_span"] = tuple(int(v) for v in raw["local_span"])
    return result
