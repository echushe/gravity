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

// Calculates gravitational accelerations for a set of bodies.
// masses: array of body masses of length n.
// positions: array of body positions of length 3*n (x, y, z for each body).
// accelerations: output array of length 3*n.
// G: gravitational constant.
// epsilon: softening length (Plummer softening): every squared distance r^2
//   becomes r^2 + epsilon^2, which caps the force of close encounters. See
//   Simulation::default_softening() for a value that suits the setup.
// n: number of bodies.
// All arrays are host memory. Everything is computed in double precision.
void calculate_gravity(const double* masses,
                       const double* positions,
                       double* accelerations,
                       const double G,
                       const double epsilon,
                       std::size_t n);

void calculate_gravity_device(const double* d_masses,
                              const double* d_positions,
                              double* d_accelerations,
                              const double G,
                              const double epsilon,
                              std::size_t n);

void calculate_velocity(const double* velocities,
                        const double* accelerations,
                        double* new_velocities,
                        const double T,
                        std::size_t n);

void calculate_velocity_device(const double* d_velocities,
                               const double* d_accelerations,
                               double* d_new_velocities,
                               const double T,
                               std::size_t n);

void calculate_velocity_and_position(const double* positions,
                                     const double* velocities,
                                     const double* accelerations,
                                     double* new_positions,
                                     double* new_velocities,
                                     const double T,
                                     std::size_t n);

void calculate_velocity_and_position_device(const double* d_positions,
                                            const double* d_velocities,
                                            const double* d_accelerations,
                                            double* d_new_positions,
                                            double* d_new_velocities,
                                            const double T,
                                            std::size_t n);

void calculate_gravity_velocity_and_position(const double* masses,
                                             const double* positions,
                                             const double* velocities,
                                             double* new_positions,
                                             double* new_velocities,
                                             const double G,
                                             const double epsilon,
                                             const double T,
                                             std::size_t n);

// One leapfrog (kick-drift-kick, also called velocity Verlet) step of T
// seconds, in place:
//   v += a*T/2;  x += v*T;  a = gravity at the new x;  v += a*T/2.
// accelerations must hold the accelerations at the current positions (from
// calculate_gravity with the same G and epsilon) and are replaced by those at
// the new positions, ready for the next step. One gravity calculation per
// step. Unlike calculate_gravity_velocity_and_position, which updates
// velocities with the old acceleration only, this is second order and
// symplectic: the energy error stays bounded instead of growing every step.
// All arrays are host memory; masses has length n, the others 3*n.
void leapfrog_step(const double* masses,
                   double* positions,
                   double* velocities,
                   double* accelerations,
                   const double G,
                   const double epsilon,
                   const double T,
                   std::size_t n);

// The same step on arrays that are already in device memory.
void leapfrog_step_device(const double* d_masses,
                          double* d_positions,
                          double* d_velocities,
                          double* d_accelerations,
                          const double G,
                          const double epsilon,
                          const double T,
                          std::size_t n);

}  // namespace gravity
