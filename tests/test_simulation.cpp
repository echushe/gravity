// Tests the per-object data classes and Simulation construction.
// CPU only: runs even without a GPU.
#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "gravity/simulation.h"
#include "test_common.h"

namespace
{

// The 3D classes are distinct types, so they cannot be mixed up.
static_assert(!std::is_convertible_v<gravity::VelocityData, gravity::PositionData>);
static_assert(!std::is_convertible_v<gravity::PositionData, gravity::AccelerationData>);

// All per-object state is stored in double precision.
static_assert(std::is_same_v<decltype(gravity::MassData().data()), double*>);
static_assert(std::is_same_v<decltype(gravity::PositionData().data()), double*>);
static_assert(std::is_same_v<decltype(gravity::VelocityData().data()), double*>);
static_assert(std::is_same_v<decltype(gravity::AccelerationData().data()), double*>);

void test_default_constructed_is_empty()
{
    CHECK(gravity::MassData().size() == 0);
    CHECK(gravity::PositionData().size() == 0);
    CHECK(gravity::VelocityData().size() == 0);
    CHECK(gravity::AccelerationData().size() == 0);
}

void test_mass_data()
{
    gravity::MassData masses(5);
    CHECK(masses.size() == 5);
    for (std::size_t i = 0; i < masses.size(); ++i) CHECK(masses[i] == 0.0);

    masses[3] = 2.5;
    CHECK(masses.data()[3] == 2.5);
}

void test_vector3_data()
{
    gravity::PositionData positions(4);
    CHECK(positions.size() == 4);  // objects, not doubles
    for (std::size_t i = 0; i < 3 * positions.size(); ++i) CHECK(positions.data()[i] == 0.0);

    // x, y, z of object i are interleaved at data()[3*i + 0..2].
    positions.x(2) = 1.0;
    positions.y(2) = 2.0;
    positions.z(2) = 3.0;
    CHECK(positions.data()[6] == 1.0);
    CHECK(positions.data()[7] == 2.0);
    CHECK(positions.data()[8] == 3.0);

    const gravity::PositionData& view = positions;
    CHECK(view.x(2) == 1.0 && view.y(2) == 2.0 && view.z(2) == 3.0);
}

void test_simulation_construction()
{
    gravity::Simulation random_seed(100.0, 1000, 5.0, 1.0);
    gravity::Simulation fixed_seed(100.0, 1000, 5.0, 1.0, 42);
    gravity::Simulation empty(100.0, 0, 5.0, 1.0);
    gravity::Simulation equal_masses(1.0, 10, 5.0, 0.0);
    // A large stddev makes many draws non-positive; they must be redrawn, not hang.
    gravity::Simulation wide_masses(1.0, 1000, 1.0, 100.0, 7);
}

void test_simulation_rejects_bad_arguments()
{
    CHECK_THROWS(gravity::Simulation(0.0, 10, 5.0, 1.0), std::invalid_argument);
    CHECK_THROWS(gravity::Simulation(-1.0, 10, 5.0, 1.0), std::invalid_argument);
    CHECK_THROWS(gravity::Simulation(1.0, 10, 0.0, 1.0), std::invalid_argument);
    CHECK_THROWS(gravity::Simulation(1.0, 10, -5.0, 1.0), std::invalid_argument);
    CHECK_THROWS(gravity::Simulation(1.0, 10, 5.0, -1.0), std::invalid_argument);
    CHECK_THROWS(gravity::Simulation(NAN, 10, 5.0, 1.0), std::invalid_argument);

    // step() checks the time step before doing any GPU work.
    gravity::Simulation sim(1.0, 10, 5.0, 1.0, 3);
    CHECK_THROWS(sim.step(0.0), std::invalid_argument);
    CHECK_THROWS(sim.step(-0.1), std::invalid_argument);
    CHECK_THROWS(sim.step(NAN), std::invalid_argument);
}

void test_softening_and_time_step()
{
    using gravity::Simulation;
    const double G = Simulation::kGravitationalConstant;

    // 1000 objects in a 100 m cube are 10 m apart on average.
    CHECK_NEAR(Simulation::default_softening(100.0, 1000), Simulation::kSofteningFraction * 10.0, 1e-12);
    // No objects count as one: the mean spacing is the cube size.
    CHECK_NEAR(Simulation::default_softening(100.0, 0), Simulation::kSofteningFraction * 100.0, 1e-12);

    const double eps = 0.5;
    const double m = 2.0e10;
    CHECK_NEAR(Simulation::default_time_step(eps, m),
               Simulation::kTimeStepFraction * std::sqrt(eps * eps * eps / (G * m)),
               1e-15);
    CHECK_THROWS(Simulation::default_time_step(0.0, m), std::invalid_argument);
    CHECK_THROWS(Simulation::default_time_step(eps, 0.0), std::invalid_argument);
    CHECK_THROWS(Simulation::default_time_step(NAN, m), std::invalid_argument);

    // Equal masses: the time step uses mass_mean itself.
    const Simulation equal_masses(100.0, 1000, m, 0.0, 1);
    CHECK_NEAR(equal_masses.softening(), Simulation::default_softening(100.0, 1000), 1e-12);
    CHECK_NEAR(equal_masses.time_step(), Simulation::default_time_step(equal_masses.softening(), m), 1e-15);

    // Varying masses: the time step uses the mean of the masses actually drawn.
    const Simulation varied_masses(100.0, 1000, m, 0.5 * m, 2);
    const gravity::MassData masses = varied_masses.masses();
    double total = 0.0;
    for (std::size_t i = 0; i < masses.size(); ++i) total += masses[i];
    CHECK_NEAR(varied_masses.time_step(),
               Simulation::default_time_step(varied_masses.softening(), total / masses.size()),
               1e-15);

    // No objects: the time step falls back to mass_mean.
    const Simulation empty(100.0, 0, m, 0.0);
    CHECK_NEAR(empty.time_step(), Simulation::default_time_step(empty.softening(), m), 1e-15);
}

}  // namespace

int main()
{
    return test::run(
        "test_simulation",
        [] {
            test_default_constructed_is_empty();
            test_mass_data();
            test_vector3_data();
            test_simulation_construction();
            test_simulation_rejects_bad_arguments();
            test_softening_and_time_step();
        },
        /*needs_gpu=*/false);
}
