// Simulation setup. Plain C++: no CUDA in this file.
#include "gravity/simulation.h"

#include "gravity/gravity.h"

#include <random>
#include <stdexcept>

namespace gravity
{

Simulation::Simulation(float cube_size, std::size_t num_objects, float mass_mean, float mass_stddev)
    : Simulation(cube_size, num_objects, mass_mean, mass_stddev, std::random_device{}())
{
}

Simulation::Simulation(float cube_size,
                       std::size_t num_objects,
                       float mass_mean,
                       float mass_stddev,
                       std::uint64_t seed)
    : cube_size_(cube_size),
      masses_(num_objects),
      positions_(num_objects),
      velocities_(num_objects),
      accelerations_(num_objects)
{
    // Written as !(x > 0) so that NaN is rejected as well.
    if (!(cube_size > 0.0f)) throw std::invalid_argument("Simulation: cube_size must be > 0");
    if (!(mass_mean > 0.0f)) throw std::invalid_argument("Simulation: mass_mean must be > 0");
    if (!(mass_stddev >= 0.0f))
    {
        throw std::invalid_argument("Simulation: mass_stddev must be >= 0");
    }

    std::mt19937_64 rng(seed);

    if (mass_stddev > 0.0f)
    {
        std::normal_distribution<float> mass_dist(mass_mean, mass_stddev);
        for (std::size_t i = 0; i < num_objects; ++i)
        {
            float m;
            do
            {
                m = mass_dist(rng);
            } while (!(m > 0.0f));
            masses_[i] = m;
        }
    }
    else
    {
        // normal_distribution requires stddev > 0; with stddev 0 every mass is the mean.
        for (std::size_t i = 0; i < num_objects; ++i) masses_[i] = mass_mean;
    }

    // uniform_real_distribution<float> can return its upper bound through
    // rounding, so redraw that case to keep positions in [0, cube_size).
    std::uniform_real_distribution<float> position_dist(0.0f, cube_size);
    auto draw_position = [&] {
        float p;
        do
        {
            p = position_dist(rng);
        } while (p >= cube_size);
        return p;
    };
    for (std::size_t i = 0; i < num_objects; ++i)
    {
        positions_.x(i) = draw_position();
        positions_.y(i) = draw_position();
        positions_.z(i) = draw_position();
    }

    // velocities_ and accelerations_ are already zero-filled by their constructors.
}

void Simulation::step(float time_step)
{
    if (!(time_step > 0.0f)) throw std::invalid_argument("Simulation::step: time_step must be > 0");

    calculate_gravity_velocity_and_position(masses_.data(),
                                            positions_.data(),
                                            velocities_.data(),
                                            positions_.data(),
                                            velocities_.data(),
                                            6.6743e-11,
                                            time_step,
                                            masses_.size());
    
    // Placeholder: currently does nothing.
}

}  // namespace gravity
