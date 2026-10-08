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
    for (std::size_t i = 0; i < masses.size(); ++i) CHECK(masses[i] == 0.0f);

    masses[3] = 2.5f;
    CHECK(masses.data()[3] == 2.5f);
}

void test_vector3_data()
{
    gravity::PositionData positions(4);
    CHECK(positions.size() == 4);  // objects, not floats
    for (std::size_t i = 0; i < 3 * positions.size(); ++i) CHECK(positions.data()[i] == 0.0f);

    // x, y, z of object i are interleaved at data()[3*i + 0..2].
    positions.x(2) = 1.0f;
    positions.y(2) = 2.0f;
    positions.z(2) = 3.0f;
    CHECK(positions.data()[6] == 1.0f);
    CHECK(positions.data()[7] == 2.0f);
    CHECK(positions.data()[8] == 3.0f);

    const gravity::PositionData& view = positions;
    CHECK(view.x(2) == 1.0f && view.y(2) == 2.0f && view.z(2) == 3.0f);
}

void test_simulation_construction()
{
    gravity::Simulation random_seed(100.0f, 1000, 5.0f, 1.0f);
    gravity::Simulation fixed_seed(100.0f, 1000, 5.0f, 1.0f, 42);
    gravity::Simulation empty(100.0f, 0, 5.0f, 1.0f);
    gravity::Simulation equal_masses(1.0f, 10, 5.0f, 0.0f);
    // A large stddev makes many draws non-positive; they must be redrawn, not hang.
    gravity::Simulation wide_masses(1.0f, 1000, 1.0f, 100.0f, 7);
}

void test_simulation_rejects_bad_arguments()
{
    CHECK_THROWS(gravity::Simulation(0.0f, 10, 5.0f, 1.0f), std::invalid_argument);
    CHECK_THROWS(gravity::Simulation(-1.0f, 10, 5.0f, 1.0f), std::invalid_argument);
    CHECK_THROWS(gravity::Simulation(1.0f, 10, 0.0f, 1.0f), std::invalid_argument);
    CHECK_THROWS(gravity::Simulation(1.0f, 10, -5.0f, 1.0f), std::invalid_argument);
    CHECK_THROWS(gravity::Simulation(1.0f, 10, 5.0f, -1.0f), std::invalid_argument);
    CHECK_THROWS(gravity::Simulation(NAN, 10, 5.0f, 1.0f), std::invalid_argument);
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
