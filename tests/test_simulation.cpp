// Tests the per-object data classes and Simulation construction.
// CPU only: runs even without a GPU.
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
        },
        /*needs_gpu=*/false);
}
