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
    // Gravitational constant in m^3 kg^-1 s^-2: the simulation uses SI units.
    static constexpr double kGravitationalConstant = 6.6743e-11;

    // The softening length is this fraction of the mean distance between
    // objects, cube_size / cbrt(num_objects).
    static constexpr double kSofteningFraction = 0.05;

    // The time step is this fraction of sqrt(epsilon^3 / (G * m)), roughly the
    // time a close pass at distance epsilon takes between objects of mass m.
    static constexpr double kTimeStepFraction = 0.1;

    // kSofteningFraction * cube_size / cbrt(num_objects), in metres. A
    // num_objects of 0 counts as 1.
    static double default_softening(double cube_size, std::size_t num_objects);

    // kTimeStepFraction * sqrt(softening^3 / (G * mean_mass)), in seconds.
    // Throws std::invalid_argument unless softening > 0 and mean_mass > 0.
    static double default_time_step(double softening, double mean_mass);

    // Creates num_objects objects inside a cube of side cube_size:
    //   - masses are drawn from a normal distribution with mean mass_mean and
    //     standard deviation mass_stddev; non-positive draws are redrawn, so
    //     every mass is > 0,
    //   - positions are drawn uniformly from [0, cube_size) on each axis,
    //   - velocities start at zero,
    //   - softening() is default_softening(cube_size, num_objects), and
    //     time_step() is default_time_step(softening(), mean of the drawn
    //     masses), or of mass_mean when there are no objects.
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
    // Advances the simulation by time_step() seconds.
    void step();
    // Advances the simulation by the given time step instead, with the same
    // softening. Throws std::invalid_argument unless time_step > 0.
    void step(double time_step);

    // Softening length epsilon used for the gravity calculation, in metres.
    double softening() const { return this->softening_; }
    // Time step used by step(), in seconds.
    double time_step() const { return this->time_step_; }

    PositionData positions() const { return this->positions_; }
    VelocityData velocities() const { return this->velocities_; }
    MassData masses() const { return this->masses_; }

private:
    double cube_size_;
    double softening_ = 0.0;
    double time_step_ = 0.0;
    MassData masses_;
    PositionData positions_;
    VelocityData velocities_;
};

}  // namespace gravity
