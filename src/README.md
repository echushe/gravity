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
├── demo/                       OpenGL viewer for the simulation (see ../demo/README.md)
├── tests/                      test programs (see ../tests/README.md)
├── WORKLOG.md                  record of the work done on the project, and open follow-ups
└── src/
    ├── CMakeLists.txt          the `gravity` library target
    ├── include/gravity/        public headers (the only ones consumers include)
    │   ├── gravity.h           device queries, CudaError, GPU compute functions
    │   ├── object_data.h       MassData, PositionData, VelocityData, AccelerationData
    │   └── simulation.h        Simulation: objects with mass moving in a cube
    ├── cuda_check.h            internal: GRAVITY_CUDA_CHECK -> gravity::CudaError
    ├── device_buffer.h         internal: RAII wrapper around cudaMalloc/cudaFree
    ├── device.cpp              device queries (CUDA runtime API only, built with g++)
    ├── nbody_kernels.cu        N-body kernels + host wrappers (built with nvcc)
    └── simulation.cpp          Simulation setup and step() (plain C++, built with g++;
                                its GPU work goes through the functions in gravity.h)
```

Build output goes into the build directory you choose (`build/` below). The
library ends up in `build/lib/`.

## Requirements

- CMake 3.22 or newer
- CUDA Toolkit (tested with 13.2)
- A C++17 host compiler (tested with g++ 11.4)
- An NVIDIA GPU and driver to run anything. Building does not need a GPU, but
  you then have to set `CMAKE_CUDA_ARCHITECTURES` explicitly (see below).
- For the demo only: GLFW, GLEW, GLM and the OpenGL development files
  (`sudo apt install libglfw3-dev libglew-dev libglm-dev` on Ubuntu), and
  internet access the first time you configure, because CMake downloads
  `stb_image_write.h`. Without these, configure with
  `-DGRAVITY_BUILD_DEMO=OFF`.

## Building

Run these from the project root:

```sh
cmake -B build                          # configure (Release by default)
cmake --build build -j                  # build the library, the tests and the demo
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
| `GRAVITY_BUILD_DEMO` | `ON` when this is the top-level project | Build the OpenGL viewer in `demo/`. Configuring fails with install instructions if GLFW, GLEW or GLM is missing; set it to `OFF` to skip the demo. |

CMake also writes `build/compile_commands.json`. clangd and the VS Code C/C++
extension can use it for code completion and navigation.

## Using the library

```cpp
#include <cstdio>
#include <gravity/simulation.h>

int main() {
    // 1000 objects in a 100 m cube, masses drawn from N(2e10, 4e9) kg,
    // velocity components uniform in [-2, 2] m/s.
    gravity::Simulation sim(100.0, 1000, 2.0e10, 4.0e9, 2.0);
    for (int i = 0; i < 100; ++i) sim.step();  // 100 steps of sim.time_step() seconds, on the GPU
    const gravity::PositionData p = sim.positions();
    std::printf("object 0 is at (%g, %g, %g) m\n", p.x(0), p.y(0), p.z(0));
}
```

Units are SI: `step()` uses the real gravitational constant, so masses must be
large (around 1e10 kg in a 100 m cube) for any visible motion.

All floating-point values in the library are `double`, on the CPU and in the
CUDA kernels. Consumer and laptop NVIDIA GPUs run FP64 far slower than FP32
(1/64 of the rate on the RTX A2000 this was developed on): one `step()` takes
about 9 ms for 5,000 objects and 125 ms for 20,000, against 1.5 ms and 7 ms
when the library used `float`.

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
| `class CudaError` | Thrown when any CUDA call fails. `what()` gives the file, line, failed call and CUDA error name; `code()` gives the `cudaError_t` value. |
| `MassData`, `PositionData`, `VelocityData`, `AccelerationData` | Per-object state ([object_data.h](include/gravity/object_data.h)). `size()` is the number of objects. `data()` is a flat `double` array in the layout the GPU functions take: `m0, m1, ...` for masses and `x0, y0, z0, x1, ...` for the 3D quantities. |
| `Simulation(L, n, mass_mean, mass_stddev, max_velocity[, seed])` | `n` objects in a cube of side `L` ([simulation.h](include/gravity/simulation.h)): masses from a normal distribution (redrawn until positive), positions uniform in `[0, L)`, each velocity component uniform in `[-max_velocity, max_velocity]` (all zero for 0). Also sets the softening length and the time step (next rows). Pass `seed` for a reproducible setup; velocities are drawn last, so `max_velocity` does not change the masses or positions for a given seed. Throws `std::invalid_argument` unless `L > 0`, `mass_mean > 0`, `mass_stddev >= 0` and `max_velocity` is finite and `>= 0`. Uses no CUDA directly. |
| `softening()`, `default_softening(L, n)` | Softening length ε = `kSofteningFraction` × L / ∛n, a fraction (0.05) of the mean distance between objects. Gravity uses r² + ε² in place of r², which caps the force of close encounters. |
| `time_step()`, `default_time_step(ε, m̄)` | Time step T = `kTimeStepFraction` × √(ε³ / (G·m̄)), a fraction (0.1) of the time a close pass at distance ε takes. m̄ is the mean of the drawn masses. |
| `Simulation::step()`, `step(T)` | Advances the simulation by `time_step()`, or by `T` seconds, with one leapfrog step on the GPU (`leapfrog_step`, below), with G = `kGravitationalConstant` = 6.6743e-11 (SI units). The accelerations are kept between steps; the first call computes them, so constructing a `Simulation` needs no GPU. Each call copies the state to the GPU and back. `step(T)` throws `std::invalid_argument` unless `T > 0`. |
| `leapfrog_step`, `leapfrog_step_device` | One kick-drift-kick (velocity Verlet) step, in place: `v += ½·a·T`, `x += v·T`, `a` = gravity at the new `x`, `v += ½·a·T`. `accelerations` goes in holding the accelerations at the current positions (from `calculate_gravity`) and comes out holding those at the new positions. Second order and symplectic, so the energy error stays bounded: about 0.5% through the demo's whole collapse. One gravity calculation per step. |
| `Simulation::positions()`, `velocities()`, `masses()` | Copies of the current state, in the `data()` layout above. |
| `calculate_gravity`, `calculate_velocity`, `calculate_velocity_and_position`, `calculate_gravity_velocity_and_position` | GPU compute functions that take host arrays. Each call allocates device memory, copies the inputs over, runs the kernels and copies the results back. The gravity functions take a softening length `epsilon` after `G`. `calculate_velocity_and_position` and `calculate_gravity_velocity_and_position` do a first-order step (`x += v·T + ½·a·T²`, `v += a·T`) that steadily gains energy; `Simulation` uses `leapfrog_step` instead. |
| `calculate_gravity_device`, `calculate_velocity_device`, `calculate_velocity_and_position_device` | The same computations on arrays that are already in device memory: no allocation, no copies. Each takes its arguments in the same order as its host-array counterpart. |

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
  during execution). See `nbody_kernels.cu`.
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
