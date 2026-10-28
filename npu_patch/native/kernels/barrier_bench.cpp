// Must come before any other shmem include: shmem_barrier.h adapts shmem's
// intra-device core sync for A2/A3 (see its header comment).
#include "ascend_moonep/kernels/shmem_barrier.h"

#include "ascend_moonep/kernels/launch.h"

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
device_barrier_bench_kernel(int32_t iterations) {
    for (int32_t i = 0; i < iterations; ++i) {
        MOONEP_SHMEM_BARRIER_ALL();
    }
}

extern "C" ASCEND_MOONEP_EXPORT void device_barrier_bench_kernel_do(
    void* stream,
    int32_t iterations,
    int32_t block_dim) {
    device_barrier_bench_kernel<<<block_dim, nullptr, stream>>>(iterations);
}
