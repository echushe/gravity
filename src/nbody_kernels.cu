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

__global__ void scale_1D(double a,
                      double* __restrict__ y,
                      std::size_t n)
{
    const std::size_t offset = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = offset; i < n; i += stride)
    {
        y[i] = a * y[i];
    }
}


__global__ void calculate_gravity_kernel_1D(const double* __restrict__ masses,
                                            const double* __restrict__ positions,
                                            double* __restrict__ accelerations,
                                            const double G,
                                            std::size_t n)
{
    const std::size_t n_threads = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    const std::size_t offset = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (std::size_t i = offset; i < n; i += n_threads)
    {
        const double x_i = positions[3 * i];
        const double y_i = positions[3 * i + 1];
        const double z_i = positions[3 * i + 2];
        double a_x = 0.0;
        double a_y = 0.0;
        double a_z = 0.0;
        for (std::size_t j = 0; j < n; ++j)
        {
            if (i == j) continue;
            const double m_j = masses[j];
            const double x_j = positions[3 * j];
            const double y_j = positions[3 * j + 1];
            const double z_j = positions[3 * j + 2];
            const double dx = x_j - x_i;
            const double dy = y_j - y_i;
            const double dz = z_j - z_i;
            const double dist_sqr = dx * dx + dy * dy + dz * dz + 1e-10;
            const double inv_dist = rsqrt(dist_sqr);
            const double inv_dist3 = inv_dist * inv_dist * inv_dist;
            const double f = m_j * inv_dist3;
            a_x += f * dx;
            a_y += f * dy;
            a_z += f * dz;
        }
        accelerations[3 * i] = G * a_x ;
        accelerations[3 * i + 1] = G * a_y;
        accelerations[3 * i + 2] = G * a_z;
    }
}

__global__ void calculate_gravity_kernel_2D(const double* __restrict__ masses,
                                            const double* __restrict__ positions,
                                            double* __restrict__ accelerations,
                                            const double G,
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
            if (i == j)
            {
                accelerations[3 * (i * n + j)] = 0.0;
                accelerations[3 * (i * n + j) + 1] = 0.0;
                accelerations[3 * (i * n + j) + 2] = 0.0;
                continue;
            }

            const double x_i = positions[3 * i];
            const double y_i = positions[3 * i + 1];
            const double z_i = positions[3 * i + 2];

            const double m_j = masses[j];
            const double x_j = positions[3 * j];
            const double y_j = positions[3 * j + 1];
            const double z_j = positions[3 * j + 2];

            const double dx = x_j - x_i;
            const double dy = y_j - y_i;
            const double dz = z_j - z_i;
            const double dist_sqr = dx * dx + dy * dy + dz * dz + 1e-10;
            const double inv_dist = rsqrt(dist_sqr);
            const double inv_dist3 = inv_dist * inv_dist * inv_dist;
            const double f = m_j * inv_dist3;

            const double a_x = f * dx;
            const double a_y = f * dy;
            const double a_z = f * dz;

            const std::size_t acc_idx = i * n + j;

            accelerations[3 * acc_idx] = G * a_x;
            accelerations[3 * acc_idx + 1] = G * a_y;
            accelerations[3 * acc_idx + 2] = G * a_z;
        }
    }
}

__global__ void accelerations_2D_to_1D(const double* __restrict__ accelerations_2D,
                                       double* __restrict__ accelerations_1D,
                                       std::size_t n)
{
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = idx; i < n; i += stride)
    {
        double a_x = 0.0;
        double a_y = 0.0;
        double a_z = 0.0;
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


__global__ void calculate_velocity_kernel_1D(const double* __restrict__ velocities,
                                             const double* __restrict__ accelerations,
                                             double* __restrict__ new_velocities,
                                             const double T,
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


__global__ void calculate_velocity_and_position_kernel_1D(const double* __restrict__ positions,
                                                          const double* __restrict__ velocities,
                                                          const double* __restrict__ accelerations,
                                                          double* __restrict__ new_positions,
                                                          double* __restrict__ new_velocities,
                                                          const double T,
                                                          std::size_t n)
{
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = idx; i < n; i += stride)
    {
        new_positions[3 * i] = positions[3 * i] + T * velocities[3 * i] + 0.5 * T * T * accelerations[3 * i];
        new_positions[3 * i + 1] = positions[3 * i + 1] + T * velocities[3 * i + 1] + 0.5 * T * T * accelerations[3 * i + 1];
        new_positions[3 * i + 2] = positions[3 * i + 2] + T * velocities[3 * i + 2] + 0.5 * T * T * accelerations[3 * i + 2];

        new_velocities[3 * i] = velocities[3 * i] + T * accelerations[3 * i];
        new_velocities[3 * i + 1] = velocities[3 * i + 1] + T * accelerations[3 * i + 1];
        new_velocities[3 * i + 2] = velocities[3 * i + 2] + T * accelerations[3 * i + 2];
    }
}

}  // namespace gravity

void calculate_gravity_device(const double* d_masses,
                              const double* d_positions,
                              double* d_accelerations,
                              const double G,
                              std::size_t n)
{
    if (n == 0) return;
    calculate_gravity_kernel_1D<<<
        std::min<std::size_t>((n + kBlockSize - 1) / kBlockSize, kMaxBlocks), kBlockSize>>>(
        d_masses, d_positions, d_accelerations, G, n);
    GRAVITY_CUDA_CHECK(cudaGetLastError());
    GRAVITY_CUDA_CHECK(cudaDeviceSynchronize());
}

void calculate_gravity(const double* masses,
                       const double* positions,
                       double* accelerations,
                       const double G,
                       std::size_t n)
{
    if (n == 0) return;
    detail::DeviceBuffer<double> d_masses(n);
    detail::DeviceBuffer<double> d_positions(3 * n);
    detail::DeviceBuffer<double> d_accelerations(3 * n);
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

void calculate_velocity_device(const double* d_velocities,
                               const double* d_accelerations,
                               double* d_new_velocities,
                               const double T,
                               std::size_t n)
{
    if (n == 0) return;
    calculate_velocity_kernel_1D<<<
        std::min<std::size_t>((n + kBlockSize - 1) / kBlockSize, kMaxBlocks), kBlockSize>>>(
        d_velocities, d_accelerations, d_new_velocities, T, n);
    GRAVITY_CUDA_CHECK(cudaGetLastError());
    GRAVITY_CUDA_CHECK(cudaDeviceSynchronize());
}

void calculate_velocity(const double* velocities,
                        const double* accelerations,
                        double* new_velocities,
                        const double T,
                        std::size_t n)
{
    if (n == 0) return;
    detail::DeviceBuffer<double> d_velocities(3 * n);
    detail::DeviceBuffer<double> d_accelerations(3 * n);
    detail::DeviceBuffer<double> d_new_velocities(3 * n);
    d_velocities.copy_from_host(velocities);
    d_accelerations.copy_from_host(accelerations);
    calculate_velocity_device(d_velocities.get(), d_accelerations.get(), d_new_velocities.get(), T, n);
    d_new_velocities.copy_to_host(new_velocities);
}

void calculate_velocity_and_position_device(const double* d_positions,
                                            const double* d_velocities,
                                            const double* d_accelerations,
                                            double* d_new_positions,
                                            double* d_new_velocities,
                                            const double T,
                                            std::size_t n)
{
    if (n == 0) return;
    calculate_velocity_and_position_kernel_1D<<<
        std::min<std::size_t>((n + kBlockSize - 1) / kBlockSize, kMaxBlocks), kBlockSize>>>(
        d_positions, d_velocities, d_accelerations, d_new_positions, d_new_velocities, T, n);
    GRAVITY_CUDA_CHECK(cudaGetLastError());
    GRAVITY_CUDA_CHECK(cudaDeviceSynchronize());
}

void calculate_velocity_and_position(const double* positions,
                                     const double* velocities,
                                     const double* accelerations,
                                     double* new_positions,
                                     double* new_velocities,
                                     const double T,
                                     std::size_t n)
{
    if (n == 0) return;
    detail::DeviceBuffer<double> d_positions(3 * n);
    detail::DeviceBuffer<double> d_velocities(3 * n);
    detail::DeviceBuffer<double> d_accelerations(3 * n);
    detail::DeviceBuffer<double> d_new_positions(3 * n);
    detail::DeviceBuffer<double> d_new_velocities(3 * n);
    d_positions.copy_from_host(positions);
    d_velocities.copy_from_host(velocities);
    d_accelerations.copy_from_host(accelerations);
    calculate_velocity_and_position_device(d_positions.get(), d_velocities.get(), d_accelerations.get(),
                                           d_new_positions.get(), d_new_velocities.get(), T, n);
    d_new_positions.copy_to_host(new_positions);
    d_new_velocities.copy_to_host(new_velocities);
}

void calculate_gravity_velocity_and_position(const double* masses,
                                             const double* positions,
                                             const double* velocities,
                                             double* new_positions,
                                             double* new_velocities,
                                             const double G,
                                             const double T,
                                             std::size_t n)
{
    if (n == 0) return;
    // values in
    detail::DeviceBuffer<double> d_masses(n);
    detail::DeviceBuffer<double> d_positions(3 * n);
    detail::DeviceBuffer<double> d_velocities(3 * n);
    // values tmp
    detail::DeviceBuffer<double> d_accelerations(3 * n);
    // values out
    detail::DeviceBuffer<double> d_new_positions(3 * n);
    detail::DeviceBuffer<double> d_new_velocities(3 * n);

    d_masses.copy_from_host(masses);
    d_positions.copy_from_host(positions);
    d_velocities.copy_from_host(velocities);

    calculate_gravity_device(d_masses.get(), d_positions.get(), d_accelerations.get(), G, n);
    calculate_velocity_and_position_device(d_positions.get(), d_velocities.get(), d_accelerations.get(),
                                           d_new_positions.get(), d_new_velocities.get(), T, n);

    d_new_positions.copy_to_host(new_positions);
    d_new_velocities.copy_to_host(new_velocities);
}

}  // namespace gravity
