// Per-object state: masses, positions, velocities and accelerations.
//
// Each class owns one contiguous float array, in the layout that the
// functions in gravity.h expect, so data() can be passed to them directly:
//   MassData:                                m0, m1, m2, ...              (size() floats)
//   PositionData, VelocityData,
//   AccelerationData:                        x0, y0, z0, x1, y1, z1, ...  (3 * size() floats)
#pragma once

#include <cstddef>
#include <vector>

namespace gravity
{

class MassData
{
public:
    // n objects, all with mass 0.
    explicit MassData(std::size_t n = 0) : values_(n, 0.0f) {}

    // Number of objects.
    std::size_t size() const noexcept { return values_.size(); }

    float& operator[](std::size_t i) { return values_[i]; }
    float operator[](std::size_t i) const { return values_[i]; }

    float* data() noexcept { return values_.data(); }
    const float* data() const noexcept { return values_.data(); }

private:
    std::vector<float> values_;
};

namespace detail
{

// Storage shared by the per-object 3D vector quantities.
class Vector3Data
{
public:
    // n objects, all set to (0, 0, 0).
    explicit Vector3Data(std::size_t n = 0) : values_(3 * n, 0.0f) {}

    // Number of objects; data() holds 3 * size() floats.
    std::size_t size() const noexcept { return values_.size() / 3; }

    float& x(std::size_t i) { return values_[3 * i]; }
    float& y(std::size_t i) { return values_[3 * i + 1]; }
    float& z(std::size_t i) { return values_[3 * i + 2]; }
    float x(std::size_t i) const { return values_[3 * i]; }
    float y(std::size_t i) const { return values_[3 * i + 1]; }
    float z(std::size_t i) const { return values_[3 * i + 2]; }

    float* data() noexcept { return values_.data(); }
    const float* data() const noexcept { return values_.data(); }

private:
    std::vector<float> values_;
};

}  // namespace detail

// Distinct types, so that e.g. a VelocityData cannot be passed where a
// PositionData is expected.
class PositionData : public detail::Vector3Data
{
public:
    using Vector3Data::Vector3Data;
};

class VelocityData : public detail::Vector3Data
{
public:
    using Vector3Data::Vector3Data;
};

class AccelerationData : public detail::Vector3Data
{
public:
    using Vector3Data::Vector3Data;
};

}  // namespace gravity
