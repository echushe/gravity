# libgravity

A C++17 library with CUDA kernels, built with CMake. It builds as a static
library (`libgravity.a`) by default, or as a shared library (`libgravity.so`)
if you ask for one.

The public header is plain C++ and includes no CUDA headers. Code that only
calls the library can therefore be compiled with `g++` or `clang++`; only the
library itself needs `nvcc`.

## Layout

```
gravity/
├── CMakeLists.txt              top-level project: compiler settings, GPU target, options
└── src/
    ├── CMakeLists.txt          the `gravity` library target
    ├── include/gravity/gravity.h   public API (the only header consumers include)
    ├── cuda_check.h            internal: GRAVITY_CUDA_CHECK -> gravity::CudaError
    ├── device_buffer.h         internal: RAII wrapper around cudaMalloc/cudaFree
    ├── device.cpp              device queries (CUDA runtime API only, built with g++)
    └── saxpy.cu                example kernel + host wrappers (built with nvcc)
```

Build output goes into the build directory you choose (`build/` below). The
library ends up in `build/lib/`.

## Requirements

- CMake 3.22 or newer
- CUDA Toolkit (tested with 13.2)
- A C++17 host compiler (tested with g++ 11.4)
- An NVIDIA GPU and driver to run anything. Building does not need a GPU, but
  you then have to set `CMAKE_CUDA_ARCHITECTURES` explicitly (see below).

## Building

Run these from the project root:

```sh
cmake -B build                          # configure (Release by default)
cmake --build build -j                  # build the library and the tests
ctest --test-dir build --output-on-failure    # run the tests
```

Other configurations each go in their own build directory:

```sh
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug     # -g -G -O0, for cuda-gdb
cmake -B build-shared -DBUILD_SHARED_LIBS=ON      # libgravity.so instead of .a
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=86      # build for sm_86 specifically
cmake --build build --target gravity              # build only the library
```

| Option | Default | Meaning |
|--------|---------|---------|
| `CMAKE_BUILD_TYPE` | `Release` | `Release` (`-O3`, plus `-lineinfo` for profilers) or `Debug` (`-g -G`). |
| `CMAKE_CUDA_ARCHITECTURES` | the GPU in this machine | Which GPU architectures to compile for, e.g. `86` or `"75;86;89"`. Set it when building for other machines or on a machine with no GPU. |
| `BUILD_SHARED_LIBS` | `OFF` | `ON` builds `libgravity.so` instead of `libgravity.a`. |
| `GRAVITY_BUILD_TESTS` | `ON` when this is the top-level project | Build `tests/`. It is off when another project includes gravity with `add_subdirectory`. |

CMake also writes `build/compile_commands.json`. clangd and the VS Code C/C++
extension can use it for code completion and navigation.

## Using the library

```cpp
#include <gravity/gravity.h>

int main() {
    float x[] = {1, 2, 3};
    float y[] = {10, 20, 30};
    gravity::saxpy(2.0f, x, y, 3);   // y is now {12, 24, 36}
}
```

**From another CMake project.** Add gravity as a subdirectory and link the
`gravity::gravity` target. This sets the include path and the CUDA runtime
for you, and the consuming project does not need to enable CUDA itself:

```cmake
add_subdirectory(path/to/gravity gravity)
target_link_libraries(my_app PRIVATE gravity::gravity)
```

**By hand, with the static library.** The CUDA runtime must be on the link
line:

```sh
g++ -std=c++17 app.cpp -I<project>/src/include <project>/build/lib/libgravity.a \
    -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lcudart
```

**By hand, with the shared library** (built with `-DBUILD_SHARED_LIBS=ON`).
`libgravity.so` loads the CUDA runtime itself, so `-lcudart` is not needed:

```sh
g++ -std=c++17 app.cpp -I<project>/src/include -L<project>/build-shared/lib \
    -Wl,-rpath,<project>/build-shared/lib -lgravity
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

- **New source file:** add a `.cu` file (kernels, built with `nvcc`) or a
  `.cpp` file (host-only code, built with `g++`) to the `add_library(gravity
  ...)` list in [CMakeLists.txt](CMakeLists.txt).
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
- **Device code across files:** kernels are compiled without relocatable
  device code, so a `__device__` function must be defined in the same `.cu`
  file that calls it, or in a header that file includes. If you need device
  code shared across files, add
  `set_target_properties(gravity PROPERTIES CUDA_SEPARABLE_COMPILATION ON)`
  to [CMakeLists.txt](CMakeLists.txt).
- **Compiler flags** shared by the library and the tests (warnings, `-G`,
  `-lineinfo`) are set in the top-level `CMakeLists.txt`.
