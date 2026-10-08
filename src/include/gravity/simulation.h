// Simulation of objects with mass moving inside a cube.
//
// This class does not use CUDA directly; GPU work goes through the functions
// declared in gravity.h.
#pragma once

#include <cstddef>
#include <cstdint>

#include "gravity/object_data.h"

namespace gravity
{

class Simulation
{
public:
    // Creates num_objects objects inside a cube of side cube_size:
    //   - masses are drawn from a normal distribution with mean mass_mean and
    //     standard deviation mass_stddev; non-positive draws are redrawn, so
    //     every mass is > 0,
    //   - positions are drawn uniformly from [0, cube_size) on each axis,
    //   - velocities and accelerations start at zero.
    // Throws std::invalid_argument unless cube_size > 0, mass_mean > 0 and
    // mass_stddev >= 0.
    //
    // This overload uses a random seed, so every run is different.
    Simulation(double cube_size, std::size_t num_objects, double mass_mean, double mass_stddev);

    // Same, but with a fixed seed: equal seeds give identical initial states.
    Simulation(double cube_size,
               std::size_t num_objects,
               double mass_mean,
               double mass_stddev,
               std::uint64_t seed);

public:
    // Advances the simulation by the given time step.
    void step(double time_step);
    PositionData positions() const { return this->positions_; }
    VelocityData velocities() const { return this->velocities_; }
    MassData masses() const { return this->masses_; }

private:
    double cube_size_;
    MassData masses_;
    PositionData positions_;
    VelocityData velocities_;
};

}  // namespace gravity
