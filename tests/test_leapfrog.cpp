// Tests the leapfrog integrator: leapfrog_step() against a CPU reference,
// energy conservation, and Simulation::step(). Needs a GPU.
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "gravity/simulation.h"
#include "test_common.h"

namespace
{

constexpr double G = gravity::Simulation::kGravitationalConstant;

// Softened gravitational accelerations, computed on the CPU in the same way
// as the kernel.
std::vector<double> cpu_gravity(const std::vector<double>& m, const std::vector<double>& x, double eps)
{
    const std::size_t n = m.size();
    std::vector<double> a(3 * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            if (i == j) continue;
            double d[3];
            double r2 = eps * eps;
            for (int k = 0; k < 3; ++k)
            {
                d[k] = x[3 * j + k] - x[3 * i + k];
                r2 += d[k] * d[k];
            }
            const double f = G * m[j] / (r2 * std::sqrt(r2));
            for (int k = 0; k < 3; ++k) a[3 * i + k] += f * d[k];
        }
    }
    return a;
}

// Kinetic plus softened potential energy.
double energy(const std::vector<double>& m, const std::vector<double>& x, const std::vector<double>& v, double eps)
{
    const std::size_t n = m.size();
    double e = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        e += 0.5 * m[i] * (v[3 * i] * v[3 * i] + v[3 * i + 1] * v[3 * i + 1] + v[3 * i + 2] * v[3 * i + 2]);
        for (std::size_t j = i + 1; j < n; ++j)
        {
            const double dx = x[3 * j] - x[3 * i];
            const double dy = x[3 * j + 1] - x[3 * i + 1];
            const double dz = x[3 * j + 2] - x[3 * i + 2];
            e -= G * m[i] * m[j] / std::sqrt(dx * dx + dy * dy + dz * dz + eps * eps);
        }
    }
    return e;
}

std::vector<double> to_vector(const double* data, std::size_t count)
{
    return std::vector<double>(data, data + count);
}

void test_one_step_matches_cpu()
{
    const std::size_t n = 50;
    const double eps = 0.3;
    const double T = 0.05;
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> u(0.0, 10.0);
    std::vector<double> m(n), x(3 * n), v(3 * n);
    for (auto& e : m) e = 1e9 * (1.0 + u(rng));
    for (auto& e : x) e = u(rng);
    for (auto& e : v) e = 0.01 * (u(rng) - 5.0);

    // Reference: kick, drift, new accelerations, kick.
    const std::vector<double> a0 = cpu_gravity(m, x, eps);
    std::vector<double> x_ref = x;
    std::vector<double> v_ref = v;
    for (std::size_t i = 0; i < 3 * n; ++i)
    {
        v_ref[i] += 0.5 * T * a0[i];
        x_ref[i] += T * v_ref[i];
    }
    const std::vector<double> a_ref = cpu_gravity(m, x_ref, eps);
    for (std::size_t i = 0; i < 3 * n; ++i) v_ref[i] += 0.5 * T * a_ref[i];

    std::vector<double> a(3 * n);
    gravity::calculate_gravity(m.data(), x.data(), a.data(), G, eps, n);
    gravity::leapfrog_step(m.data(), x.data(), v.data(), a.data(), G, eps, T, n);

    for (std::size_t i = 0; i < 3 * n; ++i)
    {
        CHECK_NEAR(x[i], x_ref[i], 1e-12 * std::fabs(x_ref[i]) + 1e-15);
        CHECK_NEAR(v[i], v_ref[i], 1e-12 * std::fabs(v_ref[i]) + 1e-15);
        CHECK_NEAR(a[i], a_ref[i], 1e-12 * std::fabs(a_ref[i]) + 1e-18);
    }

    // n == 0 is a no-op.
    gravity::leapfrog_step(nullptr, nullptr, nullptr, nullptr, G, eps, T, 0);
}

void test_circular_orbit_conserves_energy()
{
    // Two equal masses d apart, each circling their centre of mass at speed
    // sqrt(G*M / (2*d)). No softening, 1000 steps per orbit, two orbits.
    const double M = 1e10;
    const double d = 10.0;
    const double speed = std::sqrt(G * M / (2.0 * d));
    const double period = 2.0 * M_PI * (0.5 * d) / speed;
    const double T = period / 1000.0;

    std::vector<double> m = {M, M};
    std::vector<double> x = {-0.5 * d, 0.0, 0.0, 0.5 * d, 0.0, 0.0};
    std::vector<double> v = {0.0, -speed, 0.0, 0.0, speed, 0.0};
    std::vector<double> a(6);
    gravity::calculate_gravity(m.data(), x.data(), a.data(), G, 0.0, 2);

    const double e0 = energy(m, x, v, 0.0);
    double worst_energy = 0.0;
    double worst_distance = 0.0;
    for (int s = 0; s < 2000; ++s)
    {
        gravity::leapfrog_step(m.data(), x.data(), v.data(), a.data(), G, 0.0, T, 2);
        worst_energy = std::fmax(worst_energy, std::fabs((energy(m, x, v, 0.0) - e0) / e0));
        const double dist = std::hypot(x[3] - x[0], x[4] - x[1], x[5] - x[2]);
        worst_distance = std::fmax(worst_distance, std::fabs(dist - d) / d);
    }
    std::printf("  circular orbit, 2 orbits: max |dE/E| %.2e, max |dr/r| %.2e\n", worst_energy, worst_distance);
    CHECK(worst_energy < 1e-5);
    CHECK(worst_distance < 1e-4);
}

void test_simulation_step_is_leapfrog()
{
    gravity::Simulation sim(100.0, 200, 2.0e10, 4.0e9, 9);
    const double eps = sim.softening();
    const double T = sim.time_step();
    const std::size_t n = 200;

    // The same three steps by hand, starting from the same state.
    std::vector<double> m = to_vector(sim.masses().data(), n);
    std::vector<double> x = to_vector(sim.positions().data(), 3 * n);
    std::vector<double> v = to_vector(sim.velocities().data(), 3 * n);
    std::vector<double> a(3 * n);
    gravity::calculate_gravity(m.data(), x.data(), a.data(), G, eps, n);
    for (int s = 0; s < 3; ++s)
    {
        gravity::leapfrog_step(m.data(), x.data(), v.data(), a.data(), G, eps, T, n);
        sim.step();
    }

    const gravity::PositionData positions = sim.positions();
    const gravity::VelocityData velocities = sim.velocities();
    for (std::size_t i = 0; i < 3 * n; ++i)
    {
        CHECK(positions.data()[i] == x[i]);
        CHECK(velocities.data()[i] == v[i]);
    }
}

void test_simulation_cold_collapse_conserves_energy()
{
    // 500 objects at rest collapse under their own gravity; run for 1.5
    // free-fall times, through the densest point, with the default softening
    // and time step.
    const std::size_t n = 500;
    const double L = 100.0;
    gravity::Simulation sim(L, n, 2.0e10, 4.0e9, 4);
    const double eps = sim.softening();
    const std::vector<double> m = to_vector(sim.masses().data(), n);
    double total_mass = 0.0;
    for (double mi : m) total_mass += mi;
    const double free_fall = std::sqrt(3.0 * M_PI / (32.0 * G * total_mass / (L * L * L)));
    const int steps = static_cast<int>(std::lround(1.5 * free_fall / sim.time_step()));

    const double e0 = energy(m, to_vector(sim.positions().data(), 3 * n), to_vector(sim.velocities().data(), 3 * n), eps);
    double worst = 0.0;
    for (int s = 1; s <= steps; ++s)
    {
        sim.step();
        if (s % 50 == 0 || s == steps)
        {
            const double e = energy(m, to_vector(sim.positions().data(), 3 * n), to_vector(sim.velocities().data(), 3 * n), eps);
            worst = std::fmax(worst, std::fabs((e - e0) / e0));
        }
    }
    std::printf("  cold collapse, %zu objects, %d steps: max |dE/E| %.2e\n", n, steps, worst);
    CHECK(worst < 1e-2);
}

}  // namespace

int main()
{
    return test::run("test_leapfrog", [] {
        test_one_step_matches_cpu();
        test_circular_orbit_conserves_energy();
        test_simulation_step_is_leapfrog();
        test_simulation_cold_collapse_conserves_energy();
    });
}
