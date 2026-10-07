// Tests device enumeration and CudaError reporting.
// Plain C++: uses only the public gravity API, compiled with the host compiler.
#include "test_common.h"

int main() {
    return test::run("test_device", [] {
        const int count = gravity::device_count();
        CHECK(count > 0);

        for (int i = 0; i < count; ++i) {
            const gravity::DeviceInfo info = gravity::device_info(i);
            std::printf("  device %d: %s, sm_%d%d, %d SMs, %.1f GiB\n", info.id,
                        info.name.c_str(), info.compute_major, info.compute_minor,
                        info.multiprocessor_count,
                        info.total_global_mem / (1024.0 * 1024.0 * 1024.0));
            CHECK(info.id == i);
            CHECK(!info.name.empty());
            CHECK(info.compute_major > 0);
            CHECK(info.multiprocessor_count > 0);
            CHECK(info.total_global_mem > 0);
        }

        // Invalid ids must raise CudaError carrying the CUDA error code.
        CHECK_THROWS(gravity::device_info(count), gravity::CudaError);
        try {
            gravity::device_info(-1);
            CHECK(false && "device_info(-1) did not throw");
        } catch (const gravity::CudaError& e) {
            CHECK(e.code() != 0);
        }
    });
}
