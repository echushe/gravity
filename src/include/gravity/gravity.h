// Public API of the gravity CUDA library.
//
// This header is plain C++17: it does not include any CUDA headers, so code
// that only calls the library can be compiled with a regular host compiler.
#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace gravity
{

// Thrown when a CUDA runtime call made by the library fails.
class CudaError : public std::runtime_error
{
public:
    CudaError(int code, const std::string& what) : std::runtime_error(what), code_(code) {}

    // The underlying cudaError_t value.
    int code() const noexcept { return code_; }

private:
    int code_;
};

// Properties of a CUDA device.
struct DeviceInfo
{
    int id = 0;
    std::string name;
    int compute_major = 0;
    int compute_minor = 0;
    std::size_t total_global_mem = 0;  // bytes
    int multiprocessor_count = 0;
};

// Number of CUDA devices visible to this process. Returns 0 when there is no
// device or no usable driver.
int device_count();

// Properties of the given device. Throws CudaError for an invalid id.
DeviceInfo device_info(int device = 0);

// y[i] = a * x[i] + y[i] for i in [0, n).
// x and y are host pointers: the data is copied to the current device,
// computed there, and the result is copied back into y.
void saxpy(float a, const float* x, float* y, std::size_t n);

// Same as saxpy(), but d_x and d_y are device pointers already resident on
// the current device. Blocks until the computation has finished.
void saxpy_device(float a, const float* d_x, float* d_y, std::size_t n);

// Calculates gravitational accelerations for a set of bodies.
// masses: array of body masses of length n.
// positions: array of body positions of length 3*n (x, y, z for each body).
// accelerations: output array of length 3*n.
// n: number of bodies.
void calculate_gravity(const float* masses,
                       const float* positions,
                       const float G,
                       float* accelerations,
                       std::size_t n);

void calculate_gravity_device(const float* d_masses,
                              const float* d_positions,
                              float* d_accelerations,
                              const float G,
                              std::size_t n);

void calculate_velocity(const float* velocities,
                        const float* accelerations,
                        float* new_velocities,
                        const float T,
                        std::size_t n);

void calculate_velocity_device(const float* d_velocities,
                               const float* d_accelerations,
                               float* d_new_velocities,
                               const float T,
                               std::size_t n);

void calculate_velocity_and_position(const float* positions,
                                     const float* velocities,
                                     const float* accelerations,
                                     float* new_positions,
                                     float* new_velocities,
                                     const float T,
                                     std::size_t n);

void calculate_velocity_and_position_device(const float* d_positions,
                                            const float* d_velocities,
                                            const float* d_accelerations,
                                            float* d_new_positions,
                                            float* d_new_velocities,
                                            const float T,
                                            std::size_t n);

}  // namespace gravity
