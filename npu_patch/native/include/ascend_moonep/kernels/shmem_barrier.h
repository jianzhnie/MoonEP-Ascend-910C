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
// Cross-rank barrier used by the MoonEP device kernels.
//
// The kernels synchronize right after issuing cross-rank RMA; the barrier also
// guarantees completion/visibility of those previously issued remote updates.
// Ascend950 keeps shmem's vector-core barrier (`aclshmemx_barrier_all_vec()`);
// on A2/A3 `aclshmem_barrier_all()` is used together with the SyncAll ->
// soft-core-sync switch below (see the comment there for why it is required).
#pragma once

#if !defined(CATLASS_ARCH) || CATLASS_ARCH != 3510
// A2/A3 (dav-c220): shmem picks AscendC::SyncAll<>() for its intra-device core
// sync whenever __CCE_AICORE_ENABLE_MIX__ is defined (it is, for the
// [[bisheng::core_ratio(0, 1)]] entries). SyncAll is a cross-core
// "FFTS"/sync-base instruction, and this build is configured with
// --cce-auto-set-ffts=false and never calls util_set_ffts_config(), so the
// instruction faults on AICore (507015 / Illegal instruction). Undefining the
// macro for the shmem headers selects aclshmemi_sync_core_soft(), the portable
// GM-flag based core sync, which needs no sync base address.
//
// The #ifdef lives inside the shmem headers, so it is evaluated while they are
// preprocessed: undef before the include, restore immediately after. Guarded
// so translation units where the macro was never defined are left untouched.
#ifdef __CCE_AICORE_ENABLE_MIX__
#undef __CCE_AICORE_ENABLE_MIX__
#define MOONEP_RESTORE_AICORE_ENABLE_MIX
#endif
#endif

#include "shmem.h"

#ifdef MOONEP_RESTORE_AICORE_ENABLE_MIX
#define __CCE_AICORE_ENABLE_MIX__
#undef MOONEP_RESTORE_AICORE_ENABLE_MIX
#endif

#if defined(MOONEP_NO_BARRIER)
// Diagnostic build: drop the cross-rank barrier entirely (used to check whether
// a fault/hang is caused by the barrier itself rather than by the kernel body).
#define MOONEP_SHMEM_BARRIER_ALL() ((void)0)
#elif defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
#define MOONEP_SHMEM_BARRIER_ALL() aclshmemx_barrier_all_vec()
#else
#define MOONEP_SHMEM_BARRIER_ALL() aclshmem_barrier_all()
#endif
