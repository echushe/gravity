# libgravity

A C++17 library with CUDA kernels. It builds into a static library
(`libgravity.a`) and a shared library (`libgravity.so`).

The public header is plain C++ and includes no CUDA headers. Code that only
calls the library can therefore be compiled with `g++` or `clang++`; only the
library itself needs `nvcc`.

## Layout

```
src/
├── include/gravity/gravity.h   public API (the only header consumers include)
├── cuda_check.h                internal: GRAVITY_CUDA_CHECK -> gravity::CudaError
├── device_buffer.h             internal: RAII wrapper around cudaMalloc/cudaFree
├── device.cpp                  device queries (CUDA runtime API only, built with g++)
├── saxpy.cu                    example kernel + host wrappers (built with nvcc)
└── Makefile
```

Build output goes to `../build/` (outside the source tree):

```
build/obj/            object files and dependency (.d) files
build/lib/libgravity.a
build/lib/libgravity.so
```

## Requirements

- CUDA Toolkit (tested with 13.2), with `nvcc` at `/usr/local/cuda/bin/nvcc` by default
- A C++17 host compiler (tested with g++ 11.4)
- GNU make
- An NVIDIA GPU and driver to run anything. Building does not need a GPU, but
  you then have to set `GPU_ARCH` explicitly, because `native` needs a GPU.

## Building

From this directory:

```sh
make                 # static + shared, release build
make static          # only libgravity.a
make shared          # only libgravity.so
make BUILD=debug     # -g -G -O0 (device-side debugging with cuda-gdb)
make clean
```

From the project root, `make` builds the library and the tests, and `make test`
also runs the tests.

| Variable    | Default           | Meaning |
|-------------|-------------------|---------|
| `GPU_ARCH`  | `native`          | Passed as `nvcc -arch=...`. `native` targets the GPU in this machine. Use e.g. `sm_86` to build for a specific architecture, or to build on a machine with no GPU. |
| `BUILD`     | `release`         | `release` (`-O3 -lineinfo`) or `debug` (`-g -G -O0`). Run `make clean` when switching, because both use the same object directory. |
| `CUDA_PATH` | `/usr/local/cuda` | CUDA Toolkit root. |
| `BUILD_DIR` | `../build`        | Where objects and libraries are written. |

Header dependencies are tracked automatically, so editing a header rebuilds
the objects that include it.

## Using the library

```cpp
#include <gravity/gravity.h>

int main() {
    float x[] = {1, 2, 3};
    float y[] = {10, 20, 30};
    gravity::saxpy(2.0f, x, y, 3);   // y is now {12, 24, 36}
}
```

Link against the **static** library. It needs the CUDA runtime on the link
line:

```sh
g++ -std=c++17 app.cpp -I<project>/src/include <project>/build/lib/libgravity.a \
    -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lcudart
```

Link against the **shared** library. The CUDA runtime is already built into
`libgravity.so`, so `-lcudart` is not needed:

```sh
g++ -std=c++17 app.cpp -I<project>/src/include -L<project>/build/lib \
    -Wl,-rpath,<project>/build/lib -lgravity
```

## API summary

See [include/gravity/gravity.h](include/gravity/gravity.h) for the full
comments.

| Function / type | Description |
|-----------------|-------------|
| `int device_count()` | Number of visible CUDA devices. Returns 0 if there is no device or no usable driver; it never throws. |
| `DeviceInfo device_info(int device = 0)` | Name, compute capability, SM count and memory size of a device. |
| `void saxpy(a, x, y, n)` | `y = a*x + y` on **host** arrays. The library handles the copies to and from the GPU. |
| `void saxpy_device(a, d_x, d_y, n)` | Same operation on arrays already in **device** memory. Blocks until finished. |
| `class CudaError` | Thrown when any CUDA call fails. `what()` gives the file, line, failed call and CUDA error name; `code()` gives the `cudaError_t` value. |

## Adding code

- **New source file:** put a `.cu` file (kernels, built with `nvcc`) or a
  `.cpp` file (host-only code, built with `g++`) in this directory. The
  Makefile picks it up automatically.
- **Public declarations** go in `include/gravity/gravity.h`, or in a new
  header under `include/gravity/`. Keep public headers free of CUDA types such
  as `cudaStream_t` or `float3`, so that consumers can keep using a plain C++
  compiler.
- **Internal headers** (`*.h` directly in `src/`) can include
  `<cuda_runtime.h>` freely.
- **Error checking:** wrap every CUDA runtime call in `GRAVITY_CUDA_CHECK(...)`.
  After a kernel launch, check both `cudaGetLastError()` (catches launch
  errors) and, when you synchronize, `cudaDeviceSynchronize()` (catches errors
  during execution). See `saxpy.cu`.
- **Device memory** in host wrappers: use `detail::DeviceBuffer<T>` so that
  memory is freed when an exception is thrown.
- Kernels are compiled without relocatable device code (`-rdc`), so a
  `__device__` function must be defined in the same `.cu` file that calls it,
  or in a header that file includes. If you need device code shared across
  files, add `-rdc=true` and a device-link step to the Makefile.
