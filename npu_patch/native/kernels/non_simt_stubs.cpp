// Copyright 2026 Huawei Technologies Co., Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
// Non-SIMT (A2/A3) build: stand-ins for the Ascend950-only kernel entry points.
//
// Two groups of kernels are SIMT-only and therefore cannot be compiled for
// dav-c220 (A2/A3):
//   * the planner (`planning.cpp`: threadIdx / Reg::UpdateMask / asc_atomic_add
//     / simt_api/*), and
//   * the UDMA/URMA variants (`*_udma.cpp`: simt_api headers).
// Everything else in the library (dispatch / combine / prefetch / grad_reduce in
// their MTE form) is plain AscendC and does build for A2/A3.
//
// The host launchers and `npu_patch._C` reference these entry points, so the
// library must still define them; each one fails loudly at call time instead of
// resolving to garbage. On A2/A3 the plan has to come from the host-side planner
// (`test_npu/planing_reference.py`, tests: `--mode golden`) and the MTE kernels
// are selected automatically (`_is_ascend950()` gates UDMA).
#include <cstdio>
#include <stdexcept>

#include "ascend_moonep/kernels/launch.h"

namespace {

[[noreturn]] void unavailable(const char* name, const char* hint) {
    std::fprintf(stderr,
                 "[moonep] %s is not built for this SoC: it is Ascend950-only "
                 "(SIMT). %s\n",
                 name, hint);
    throw std::runtime_error(std::string(name) +
                             " is not available on this SoC (Ascend950/SIMT "
                             "only); " + hint);
}

}  // namespace

extern "C" {

ASCEND_MOONEP_EXPORT void planning_kernel_do(
    void* stream, uint8_t* topk_idx, uint8_t* symmetric_buffer, uint8_t* dst,
    uint8_t* cu_seqlens, uint8_t* experts_to_copy, uint8_t* dup_groups,
    uint8_t* dup_loffs, uint8_t* dup_counts, uint8_t* tpe_all,
    uint8_t* tpe_prefix, uint8_t* balance, uint8_t* z_tensor,
    uint8_t* alloc_tensor, uint8_t* alloc_cumsum, uint8_t* expert_offsets,
    uint8_t* core_expert_counts, uint8_t* dup_workspace,
    uint8_t* dup_key_workspace, uint8_t* tiling_host, uint64_t ffts_addr) {
    (void)stream; (void)topk_idx; (void)symmetric_buffer; (void)dst;
    (void)cu_seqlens; (void)experts_to_copy; (void)dup_groups; (void)dup_loffs;
    (void)dup_counts; (void)tpe_all; (void)tpe_prefix; (void)balance;
    (void)z_tensor; (void)alloc_tensor; (void)alloc_cumsum;
    (void)expert_offsets; (void)core_expert_counts; (void)dup_workspace;
    (void)dup_key_workspace; (void)tiling_host; (void)ffts_addr;
    unavailable("planning_kernel_do",
                "supply a host-computed plan (tests: --mode golden)");
}

ASCEND_MOONEP_EXPORT void moonep_dispatch_udma_kernel_do(
    void* stream, uint8_t* hidden_sh, uint8_t* route_weights, uint8_t* dst,
    uint8_t* dup_groups, uint8_t* dup_loffs, uint8_t* dup_counts,
    uint8_t* hidden_buf, uint8_t* weights_buf, uint8_t* sync_buffer,
    uint8_t* tiling_host, uint64_t magic) {
    (void)stream; (void)hidden_sh; (void)route_weights; (void)dst;
    (void)dup_groups; (void)dup_loffs; (void)dup_counts; (void)hidden_buf;
    (void)weights_buf; (void)sync_buffer; (void)tiling_host; (void)magic;
    unavailable("moonep_dispatch_udma_kernel_do",
                "use the MTE engine (ASCEND_MOONEP_USE_UDMA=0)");
}

ASCEND_MOONEP_EXPORT void moonep_dispatch_udma_profile_kernel_do(
    void* stream, uint8_t* hidden_sh, uint8_t* route_weights, uint8_t* dst,
    uint8_t* dup_groups, uint8_t* dup_loffs, uint8_t* dup_counts,
    uint8_t* hidden_buf, uint8_t* weights_buf, uint8_t* sync_buffer,
    uint8_t* profile_cycles, uint8_t* tiling_host, uint64_t magic) {
    (void)stream; (void)hidden_sh; (void)route_weights; (void)dst;
    (void)dup_groups; (void)dup_loffs; (void)dup_counts; (void)hidden_buf;
    (void)weights_buf; (void)sync_buffer; (void)profile_cycles;
    (void)tiling_host; (void)magic;
    unavailable("moonep_dispatch_udma_profile_kernel_do",
                "use the MTE engine (ASCEND_MOONEP_USE_UDMA=0)");
}

ASCEND_MOONEP_EXPORT void moonep_udma_combine_plan_kernel_do(
    void* stream, uint8_t* hidden, uint8_t* weights, uint8_t* dst,
    uint8_t* groups, uint8_t* dups, uint8_t* counts, uint8_t* out,
    uint8_t* outw, uint8_t* workspace, uint8_t* plan, uint8_t* tiling,
    uint64_t magic) {
    (void)stream; (void)hidden; (void)weights; (void)dst; (void)groups;
    (void)dups; (void)counts; (void)out; (void)outw; (void)workspace;
    (void)plan; (void)tiling; (void)magic;
    unavailable("moonep_udma_combine_plan_kernel_do",
                "use the MTE engine (ASCEND_MOONEP_USE_UDMA=0)");
}

ASCEND_MOONEP_EXPORT void prefetch_weight_kernel_udma_do(
    void* stream, uint8_t* remote_expert, uint8_t* prefetch_buffers,
    uint8_t* experts_to_copy, int64_t num_experts, int64_t hidden,
    int64_t hidden_prime, int64_t num_slots, int64_t block_dim) {
    (void)stream; (void)remote_expert; (void)prefetch_buffers;
    (void)experts_to_copy; (void)num_experts; (void)hidden; (void)hidden_prime;
    (void)num_slots; (void)block_dim;
    unavailable("prefetch_weight_kernel_udma_do",
                "use the MTE engine (ASCEND_MOONEP_USE_UDMA=0)");
}

ASCEND_MOONEP_EXPORT int moonep_shmem_grad_reduce_udma(
    void* stream, void* expert_grads, void* reduce_buffers, void* workspace,
    const int32_t* experts_to_copy, int32_t epn, int32_t hidden,
    int32_t projection, int32_t ranks, int32_t slots, int32_t rank,
    int32_t blocks) {
    (void)stream; (void)expert_grads; (void)reduce_buffers; (void)workspace;
    (void)experts_to_copy; (void)epn; (void)hidden; (void)projection;
    (void)ranks; (void)slots; (void)rank; (void)blocks;
    unavailable("moonep_shmem_grad_reduce_udma",
                "use the MTE engine (ASCEND_MOONEP_USE_UDMA=0)");
}

}  // extern "C"
