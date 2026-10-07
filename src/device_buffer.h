// Internal RAII owner of a device allocation, so device memory is released
// even when a CUDA call throws.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "cuda_check.h"

namespace gravity::detail {

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_ > 0) GRAVITY_CUDA_CHECK(cudaMalloc(&ptr_, bytes()));
    }
    ~DeviceBuffer() { cudaFree(ptr_); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return count_; }

    void copy_from_host(const T* src) {
        GRAVITY_CUDA_CHECK(cudaMemcpy(ptr_, src, bytes(), cudaMemcpyHostToDevice));
    }
    void copy_to_host(T* dst) const {
        GRAVITY_CUDA_CHECK(cudaMemcpy(dst, ptr_, bytes(), cudaMemcpyDeviceToHost));
    }

private:
    std::size_t bytes() const noexcept { return count_ * sizeof(T); }

    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

}  // namespace gravity::detail
