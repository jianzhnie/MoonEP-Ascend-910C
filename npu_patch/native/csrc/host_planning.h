#pragma once

#include <torch/extension.h>

#include <cstdint>

struct HostPlanResult {
    torch::Tensor dst;
    torch::Tensor cu_seqlens;
    torch::Tensor experts_to_copy;
    torch::Tensor remote_stats;
    torch::Tensor zero_fill_ranges;
    torch::Tensor alloc;
    torch::Tensor needed_remote;
    int64_t span_begin;
    int64_t span_end;
};

// Host-side MoonEP plan for the A2/A3 (non-SIMT) path.
//
// Inputs must be CPU, contiguous int32 tensors.  The contract matches
// hyper_parallel/distributed/expert_parallel/moonep_plan.py:
//   counts_all: [R, E], local_counts: [E], route_ids: [N]
// and returns the same fields as build_moonep_plan (alloc is [R, E]).
HostPlanResult BuildHostPlan(
    const torch::Tensor& counts_all,
    const torch::Tensor& local_counts,
    const torch::Tensor& route_ids,
    int64_t rank,
    int64_t epn,
    int64_t nvs);
