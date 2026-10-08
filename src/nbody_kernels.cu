// N-body kernels (gravitational accelerations, velocity and position updates)
// and their host-side wrappers.
#include <algorithm>
#include <cstddef>

#include "cuda_check.h"
#include "device_buffer.h"
#include "gravity/gravity.h"

namespace gravity
{
namespace
{

constexpr int kBlockSize = 256;
constexpr std::size_t kMaxBlocks = 65535;

__global__ void scale_1D(float a,
                      float* __restrict__ y,
                      std::size_t n)
{
    const std::size_t offset = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = offset; i < n; i += stride)
    {
        y[i] = a * y[i];
    }
}


__global__ void calculate_gravity_kernel_1D(const float* __restrict__ masses,
                                            const float* __restrict__ positions,
                                            float* __restrict__ accelerations,
                                            const float G,
                                            std::size_t n)
{
    const std::size_t n_threads = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    const std::size_t offset = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (std::size_t i = offset; i < n; i += n_threads)
    {
        const float x_i = positions[3 * i];
        const float y_i = positions[3 * i + 1];
        const float z_i = positions[3 * i + 2];
        float a_x = 0.0f;
        float a_y = 0.0f;
        float a_z = 0.0f;
        for (std::size_t j = 0; j < n; ++j)
        {
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
        accelerations[3 * i] = G * a_x ;
        accelerations[3 * i + 1] = G * a_y;
        accelerations[3 * i + 2] = G * a_z;
    }
}

__global__ void calculate_gravity_kernel_2D(const float* __restrict__ masses,
                                            const float* __restrict__ positions,
                                            float* __restrict__ accelerations,
                                            const float G,
                                            std::size_t n)
{
    // The output accelerations will be a 2D array corresponding to the 2D grid of threads.
    // So size of the output array will be 3 * n * n

    // Implementation for a 2D grid version of the gravity calculation would go here.
    const std::size_t n_threads_x = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    const std::size_t offset_x = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const std::size_t n_threads_y = static_cast<std::size_t>(blockDim.y) * gridDim.y;
    const std::size_t offset_y = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;

    for (std::size_t i = offset_x; i < n; i += n_threads_x)
    {
        for (std::size_t j = offset_y; j < n; j += n_threads_y)
        {
            // Implementation for the 2D grid version would go here.
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

            accelerations[3 * acc_idx] = G * a_x;
            accelerations[3 * acc_idx + 1] = G * a_y;
            accelerations[3 * acc_idx + 2] = G * a_z;
        }
    }
}

__global__ void accelerations_2D_to_1D(const float* __restrict__ accelerations_2D,
                                       float* __restrict__ accelerations_1D,
                                       std::size_t n)
{
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = idx; i < n; i += stride)
    {
        float a_x = 0.0f;
        float a_y = 0.0f;
        float a_z = 0.0f;
        for (std::size_t j = 0; j < n; ++j)
        {
            const std::size_t acc_2D_idx = i * n + j;
            a_x += accelerations_2D[3 * acc_2D_idx];
            a_y += accelerations_2D[3 * acc_2D_idx + 1];
            a_z += accelerations_2D[3 * acc_2D_idx + 2];
        }
        accelerations_1D[3 * i] = a_x;
        accelerations_1D[3 * i + 1] = a_y;
        accelerations_1D[3 * i + 2] = a_z;
    }
}


__global__ void calculate_velocity_kernel_1D(const float* __restrict__ velocities,
                                             const float* __restrict__ accelerations,
                                             float* __restrict__ new_velocities,
                                             const float T,
                                             std::size_t n)
{
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = idx; i < n; i += stride)
    {
        new_velocities[3 * i] = velocities[3 * i] + T * accelerations[3 * i];
        new_velocities[3 * i + 1] = velocities[3 * i + 1] + T * accelerations[3 * i + 1];
        new_velocities[3 * i + 2] = velocities[3 * i + 2] + T * accelerations[3 * i + 2];
    }
}


__global__ void calculate_velocity_and_position_kernel_1D(const float* __restrict__ positions,
                                                          const float* __restrict__ velocities,
                                                          const float* __restrict__ accelerations,
                                                          float* __restrict__ new_positions,
                                                          float* __restrict__ new_velocities,
                                                          const float T,
                                                          std::size_t n)
{
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = idx; i < n; i += stride)
    {
        new_positions[3 * i] = positions[3 * i] + T * velocities[3 * i] + 0.5f * T * T * accelerations[3 * i];
        new_positions[3 * i + 1] = positions[3 * i + 1] + T * velocities[3 * i + 1] + 0.5f * T * T * accelerations[3 * i + 1];
        new_positions[3 * i + 2] = positions[3 * i + 2] + T * velocities[3 * i + 2] + 0.5f * T * T * accelerations[3 * i + 2];

        new_velocities[3 * i] = velocities[3 * i] + T * accelerations[3 * i];
        new_velocities[3 * i + 1] = velocities[3 * i + 1] + T * accelerations[3 * i + 1];
        new_velocities[3 * i + 2] = velocities[3 * i + 2] + T * accelerations[3 * i + 2];
    }
}

}  // namespace gravity

void calculate_gravity_device(const float* d_masses,
                              const float* d_positions,
                              float* d_accelerations,
                              const float G,
                              std::size_t n)
{
    if (n == 0) return;
    calculate_gravity_kernel_1D<<<
        std::min<std::size_t>((n + kBlockSize - 1) / kBlockSize, kMaxBlocks), kBlockSize>>>(
        d_masses, d_positions, d_accelerations, G, n);
    GRAVITY_CUDA_CHECK(cudaGetLastError());
    GRAVITY_CUDA_CHECK(cudaDeviceSynchronize());
}

void calculate_gravity(const float* masses,
                       const float* positions,
                       float* accelerations,
                       const float G,
                       std::size_t n)
{
    if (n == 0) return;
    detail::DeviceBuffer<float> d_masses(n);
    detail::DeviceBuffer<float> d_positions(3 * n);
    detail::DeviceBuffer<float> d_accelerations(3 * n);
    d_masses.copy_from_host(masses);
    d_positions.copy_from_host(positions);
    // Assuming a kernel calculate_gravity_kernel_1D is defined elsewhere
    calculate_gravity_kernel_1D<<<
        std::min<std::size_t>((n + kBlockSize - 1) / kBlockSize, kMaxBlocks), kBlockSize>>>(
        d_masses.get(), d_positions.get(), d_accelerations.get(), G, n);
    GRAVITY_CUDA_CHECK(cudaGetLastError());
    GRAVITY_CUDA_CHECK(cudaDeviceSynchronize());
    d_accelerations.copy_to_host(accelerations);
}

void calculate_velocity_device(const float* d_velocities,
                               const float* d_accelerations,
                               float* d_new_velocities,
                               std::size_t n,
                               const float T)
{
    if (n == 0) return;
    calculate_velocity_kernel_1D<<<
        std::min<std::size_t>((n + kBlockSize - 1) / kBlockSize, kMaxBlocks), kBlockSize>>>(
        d_velocities, d_accelerations, d_new_velocities, T, n);
    GRAVITY_CUDA_CHECK(cudaGetLastError());
    GRAVITY_CUDA_CHECK(cudaDeviceSynchronize());
}

void calculate_velocity(const float* velocities,
                        const float* accelerations,
                        float* new_velocities,
                        const float T,
                        std::size_t n)
{
    if (n == 0) return;
    detail::DeviceBuffer<float> d_velocities(3 * n);
    detail::DeviceBuffer<float> d_accelerations(3 * n);
    detail::DeviceBuffer<float> d_new_velocities(3 * n);
    d_velocities.copy_from_host(velocities);
    d_accelerations.copy_from_host(accelerations);
    calculate_velocity_device(d_velocities.get(), d_accelerations.get(), d_new_velocities.get(), n, T);
    d_new_velocities.copy_to_host(new_velocities);
}

void calculate_velocity_and_position_device(const float* d_positions,
                                            const float* d_velocities,
                                            const float* d_accelerations,
                                            float* d_new_positions,
                                            float* d_new_velocities,
                                            const float T,
                                            std::size_t n)
{
    if (n == 0) return;
    calculate_velocity_and_position_kernel_1D<<<
        std::min<std::size_t>((n + kBlockSize - 1) / kBlockSize, kMaxBlocks), kBlockSize>>>(
        d_positions, d_velocities, d_accelerations, d_new_positions, d_new_velocities, T, n);
    GRAVITY_CUDA_CHECK(cudaGetLastError());
    GRAVITY_CUDA_CHECK(cudaDeviceSynchronize());
}

void calculate_velocity_and_position(const float* positions,
                                     const float* velocities,
                                     const float* accelerations,
                                     float* new_positions,
                                     float* new_velocities,
                                     const float T,
                                     std::size_t n)
{
    if (n == 0) return;
    detail::DeviceBuffer<float> d_positions(3 * n);
    detail::DeviceBuffer<float> d_velocities(3 * n);
    detail::DeviceBuffer<float> d_accelerations(3 * n);
    detail::DeviceBuffer<float> d_new_positions(3 * n);
    detail::DeviceBuffer<float> d_new_velocities(3 * n);
    d_positions.copy_from_host(positions);
    d_velocities.copy_from_host(velocities);
    d_accelerations.copy_from_host(accelerations);
    calculate_velocity_and_position_device(d_positions.get(), d_velocities.get(), d_accelerations.get(),
                                           d_new_positions.get(), d_new_velocities.get(), T, n);
    d_new_positions.copy_to_host(new_positions);
    d_new_velocities.copy_to_host(new_velocities);
}

}  // namespace gravity
