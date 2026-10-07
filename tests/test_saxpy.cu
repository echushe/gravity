// Tests saxpy() (host pointers) and saxpy_device() (device pointers) against
// a CPU reference. A .cu file, so it can call the CUDA runtime directly.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "test_common.h"

namespace {

constexpr float kA = 2.5f;

std::vector<float> make_input(std::size_t n, float scale) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = scale * static_cast<float>(i % 1000) - 3.0f;
    return v;
}

std::vector<float> saxpy_reference(float a, const std::vector<float>& x, std::vector<float> y) {
    for (std::size_t i = 0; i < y.size(); ++i) y[i] = a * x[i] + y[i];
    return y;
}

// Compares element-wise; reports the first mismatch only, to avoid flooding.
void check_all_near(const std::vector<float>& actual, const std::vector<float>& expected) {
    CHECK(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float tol = 1e-5f * std::max(1.0f, std::fabs(expected[i]));
        if (std::fabs(actual[i] - expected[i]) > tol) {
            CHECK_NEAR(actual[i], expected[i], tol);
            std::printf("  (first mismatch at index %zu of %zu)\n", i, actual.size());
            return;
        }
    }
}

void test_host_api(std::size_t n) {
    const std::vector<float> x = make_input(n, 0.5f);
    std::vector<float> y = make_input(n, -1.25f);
    const std::vector<float> expected = saxpy_reference(kA, x, y);

    gravity::saxpy(kA, x.data(), y.data(), n);
    check_all_near(y, expected);
}

void test_device_api(std::size_t n) {
    const std::vector<float> x = make_input(n, 0.75f);
    std::vector<float> y = make_input(n, 2.0f);
    const std::vector<float> expected = saxpy_reference(kA, x, y);

    const std::size_t bytes = n * sizeof(float);
    float* d_x = nullptr;
    float* d_y = nullptr;
    CHECK(cudaMalloc(&d_x, bytes) == cudaSuccess);
    CHECK(cudaMalloc(&d_y, bytes) == cudaSuccess);
    CHECK(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(d_y, y.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess);

    gravity::saxpy_device(kA, d_x, d_y, n);

    CHECK(cudaMemcpy(y.data(), d_y, bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
    cudaFree(d_x);
    cudaFree(d_y);
    check_all_near(y, expected);
}

}  // namespace

int main() {
    return test::run("test_saxpy", [] {
        // Edge sizes around the block size (256), plus one larger than
        // 65535 blocks * 256 threads to exercise the grid-stride loop.
        const std::size_t sizes[] = {0, 1, 255, 256, 257, 1 << 20, 65535ull * 256 + 7};
        for (std::size_t n : sizes) test_host_api(n);

        test_device_api(1 << 20);
    });
}
