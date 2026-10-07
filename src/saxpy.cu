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

__global__ void calculate_gravity_kernel_1D(const float* __restrict__ masses,
                                         const float* __restrict__ positions,
                                         float* __restrict__ accelerations,
                                         std::size_t n) {

    const std::size_t n_threads = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    const std::size_t offset = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (std::size_t i = offset; i < n; i += n_threads) {
        const float m_i = masses[i];
        const float x_i = positions[3 * i];
        const float y_i = positions[3 * i + 1];
        const float z_i = positions[3 * i + 2];
        float a_x = 0.0f;
        float a_y = 0.0f;
        float a_z = 0.0f;
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            const float m_j = masses[j];
            const float x_j = positions[3 * j];
            const float y_j = positions[3 * j + 1];
            const float z_j = positions[3 * j + 2];
            const float dx = x_j - x_i;
            const float dy = y_j - y_i;
            const float dz = z_j - z_i;
            const float dist_sqr = dx * dx + dy * dy + dz * dz + 1e-10f;
            const float inv_dist = rsqrtf(dist_sqr);
            const float inv_dist3 = inv_dist * inv_dist * inv_dist;
            const float f = m_j * inv_dist3;
            a_x += f * dx;
            a_y += f * dy;
            a_z += f * dz;
        }
        accelerations[3 * i] = a_x;
        accelerations[3 * i + 1] = a_y;
        accelerations[3 * i + 2] = a_z;
    }
}

__global__ void calculate_gravity_kernel_2D(const float* __restrict__ masses,
                                         const float* __restrict__ positions,
                                         float* __restrict__ accelerations,
                                         std::size_t n) {

    // The output accelerations will be a 2D array corresponding to the 2D grid of threads.
    // So size of the output array will be 3 * n * n

    // Implementation for a 2D grid version of the gravity calculation would go here.
    const std::size_t n_threads_x = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    const std::size_t offset_x = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const std::size_t n_threads_y = static_cast<std::size_t>(blockDim.y) * gridDim.y;
    const std::size_t offset_y = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;

    for (std::size_t i = offset_x; i < n; i += n_threads_x) {
        for (std::size_t j = offset_y; j < n; j += n_threads_y) {
            // Implementation for the 2D grid version would go here.
            const float m_i = masses[i];
            const float x_i = positions[3 * i];
            const float y_i = positions[3 * i + 1];
            const float z_i = positions[3 * i + 2];

            const float m_j = masses[j];
            const float x_j = positions[3 * j];
            const float y_j = positions[3 * j + 1];
            const float z_j = positions[3 * j + 2];

            const float dx = x_j - x_i;
            const float dy = y_j - y_i;
            const float dz = z_j - z_i;
            const float dist_sqr = dx * dx + dy * dy + dz * dz + 1e-10f;
            const float inv_dist = rsqrtf(dist_sqr);
            const float inv_dist3 = inv_dist * inv_dist * inv_dist;
            const float f = m_j * inv_dist3;
            
            const float a_x = f * dx;
            const float a_y = f * dy;
            const float a_z = f * dz;

            const std::size_t acc_idx = i * n + j;

            accelerations[3 * acc_idx] = a_x;
            accelerations[3 * acc_idx + 1] = a_y;
            accelerations[3 * acc_idx + 2] = a_z;
        }
    }
}

__global__ void accelerations_2D_to_1D(const float* __restrict__ accelerations_2D,
                                        float* __restrict__ accelerations_1D,
                                        std::size_t n) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = idx; i < n; i += stride) {
        float a_x = 0.0f;
        float a_y = 0.0f;
        float a_z = 0.0f;
        for (std::size_t j = 0; j < n; ++j) {
            const std::size_t acc_2D_idx = i * n + j;
            const std::size_t acc_1D_idx = i;
            a_x += accelerations_2D[3 * acc_2D_idx];
            a_y += accelerations_2D[3 * acc_2D_idx + 1];
            a_z += accelerations_2D[3 * acc_2D_idx + 2];
        }
        accelerations_1D[3 * i] = a_x;
        accelerations_1D[3 * i + 1] = a_y;
        accelerations_1D[3 * i + 2] = a_z;
        }
    }
}

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


void calculate_gravity_device(const float* d_masses, const float* d_positions, float* d_accelerations, std::size_t n) {
    if (n == 0) return;
    calculate_gravity_kernel_1D<<<std::min<std::size_t>((n + kBlockSize - 1) / kBlockSize, kMaxBlocks), kBlockSize>>>(
        d_masses, d_positions, d_accelerations, n);
    GRAVITY_CUDA_CHECK(cudaGetLastError());
    GRAVITY_CUDA_CHECK(cudaDeviceSynchronize());
}

void calculate_gravity(const float* masses, const float* positions, float* accelerations, std::size_t n) {
    if (n == 0) return;
    detail::DeviceBuffer<float> d_masses(n);
    detail::DeviceBuffer<float> d_positions(3 * n);
    detail::DeviceBuffer<float> d_accelerations(3 * n);
    d_masses.copy_from_host(masses);
    d_positions.copy_from_host(positions);
    // Assuming a kernel calculate_gravity_kernel_1D is defined elsewhere
    calculate_gravity_kernel_1D<<<std::min<std::size_t>((n + kBlockSize - 1) / kBlockSize, kMaxBlocks), kBlockSize>>>(
        d_masses.get(), d_positions.get(), d_accelerations.get(), n);
    GRAVITY_CUDA_CHECK(cudaGetLastError());
    GRAVITY_CUDA_CHECK(cudaDeviceSynchronize());
    d_accelerations.copy_to_host(accelerations);
}

}  // namespace gravity
