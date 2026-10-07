// SAXPY (y = a*x + y): an example kernel plus its host-side wrappers.
#include <algorithm>
#include <cstddef>

#include "cuda_check.h"
#include "device_buffer.h"
#include "gravity/gravity.h"

namespace gravity {
namespace {

constexpr int kBlockSize = 256;
constexpr std::size_t kMaxBlocks = 65535;

// Grid-stride loop: correct for any n, whatever grid size is launched.
__global__ void saxpy_kernel(float a, const float* __restrict__ x,
                             float* __restrict__ y, std::size_t n) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

}  // namespace

void saxpy_device(float a, const float* d_x, float* d_y, std::size_t n) {
    if (n == 0) return;
    const auto blocks = static_cast<unsigned>(
        std::min<std::size_t>((n + kBlockSize - 1) / kBlockSize, kMaxBlocks));
    saxpy_kernel<<<blocks, kBlockSize>>>(a, d_x, d_y, n);
    GRAVITY_CUDA_CHECK(cudaGetLastError());
    GRAVITY_CUDA_CHECK(cudaDeviceSynchronize());
}

void saxpy(float a, const float* x, float* y, std::size_t n) {
    if (n == 0) return;
    detail::DeviceBuffer<float> d_x(n);
    detail::DeviceBuffer<float> d_y(n);
    d_x.copy_from_host(x);
    d_y.copy_from_host(y);
    saxpy_device(a, d_x.get(), d_y.get(), n);
    d_y.copy_to_host(y);
}

}  // namespace gravity
