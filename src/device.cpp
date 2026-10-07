// Device enumeration. Uses only the CUDA runtime API (no kernels), so it is
// compiled as plain C++ by the host compiler.
#include <cuda_runtime.h>

#include "cuda_check.h"
#include "gravity/gravity.h"

namespace gravity {

int device_count() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        (void)cudaGetLastError();  // no device / no driver: clear and report 0
        return 0;
    }
    return count;
}

DeviceInfo device_info(int device) {
    cudaDeviceProp prop{};
    GRAVITY_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    DeviceInfo info;
    info.id = device;
    info.name = prop.name;
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;
    info.total_global_mem = prop.totalGlobalMem;
    info.multiprocessor_count = prop.multiProcessorCount;
    return info;
}

}  // namespace gravity
