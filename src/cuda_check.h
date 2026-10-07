// Internal helper that turns CUDA runtime errors into gravity::CudaError.
// Usable from both .cpp and .cu files.
#pragma once

#include <cuda_runtime.h>

#include <string>

#include "gravity/gravity.h"

namespace gravity::detail {

inline void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) return;
    // Reset the runtime's "last error" so a later cudaGetLastError() check
    // (e.g. after a kernel launch) does not report this stale failure.
    (void)cudaGetLastError();
    throw CudaError(static_cast<int>(err),
                    std::string(file) + ":" + std::to_string(line) + ": " + expr +
                        " failed: " + cudaGetErrorName(err) + " (" +
                        cudaGetErrorString(err) + ")");
}

}  // namespace gravity::detail

#define GRAVITY_CUDA_CHECK(expr) \
    ::gravity::detail::cuda_check((expr), #expr, __FILE__, __LINE__)
