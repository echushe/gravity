# libgravity tests

Each `test_*.cpp` or `test_*.cu` file in this directory builds into its own
standalone executable, and is registered as a CTest test. You can run the
tests through `ctest` or run any executable directly. There is no test
framework to install: everything needed is in `test_common.h`.

## Layout

```
tests/
├── CMakeLists.txt     one executable + one CTest test per test_* file
├── test_common.h      CHECK / CHECK_NEAR / CHECK_THROWS macros and test::run()
├── test_device.cpp    device enumeration, DeviceInfo fields, CudaError on bad ids
├── test_leapfrog.cpp  leapfrog_step vs a CPU reference, energy conservation
│                      (circular orbit, cold collapse), Simulation::step()
└── test_simulation.cpp  data classes, Simulation construction, initial
                       velocities, softening and time step (CPU only)
```

Executables are written to `build/tests/`.

Not covered yet: the other GPU compute functions in `gravity.h`
(`calculate_velocity`, `calculate_velocity_and_position` and their `_device`
versions, `calculate_gravity_velocity_and_position`) have no tests.
`calculate_gravity` is tested indirectly through `test_leapfrog`.

## Building and running

Run these from the project root:

```sh
cmake -B build                                  # configure (once)
cmake --build build -j                          # build the library and all tests
ctest --test-dir build --output-on-failure      # run all tests
ctest --test-dir build -R simulation -V         # run tests matching a pattern, with full output
cmake --build build --target test_simulation    # build one test
./build/tests/test_simulation                   # run one test directly
```

`cmake -B build` also configures the OpenGL demo, which needs GLFW, GLEW and
GLM. On a machine without them (a CI runner, say), add
`-DGRAVITY_BUILD_DEMO=OFF` to build only the library and the tests.

`ctest` exits non-zero if any test fails, so you can use it in CI.
`--output-on-failure` prints a failing test's output; `-V` prints the output
of every test.

To run the tests against a debug or shared build, configure a separate build
directory (see [../src/README.md](../src/README.md)):

```sh
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build-debug -j
ctest --test-dir build-debug --output-on-failure
```

Running a test directly prints output like this:

```
$ ./build/tests/test_device
  device 0: NVIDIA RTX A2000 8GB Laptop GPU, sm_86, 20 SMs, 7.7 GiB
[PASS] test_device
```

## Test conventions

- **Exit status:** 0 means pass or skip; 1 means at least one check failed.
- **No GPU:** if no CUDA device is visible, a test prints `[SKIP]` and exits 0.
  Tests that don't need a GPU pass `false` as the third argument of
  `test::run()`, so they still run (see `test_simulation.cpp`).
- **Checks don't abort.** `CHECK(cond)`, `CHECK_NEAR(actual, expected, tol)`
  and `CHECK_THROWS(expr, Type)` print the file, line and expression of a
  failure and keep going, so one run reports every failing check.
- **Unexpected exceptions** (for example a `gravity::CudaError` thrown from
  inside a test) are caught by `test::run()`, printed, and counted as a
  failure.

## Adding a test

1. Create `test_<name>.cpp` if the test uses only the public `gravity` API.
   Create `test_<name>.cu` if it defines kernels; it is then compiled with
   `nvcc`. Either kind can call the CUDA runtime (`cudaMalloc` and so on),
   because every test links `CUDA::cudart`.
2. Write it like this:

   ```cpp
   #include "test_common.h"

   int main() {
       return test::run("test_<name>", [] {
           CHECK(gravity::device_count() > 0);
           // ...
       });
   }
   ```

3. Run `cmake --build build` and then `ctest --test-dir build`. New `test_*`
   files are picked up automatically, without re-running `cmake -B build`.

`.cpp` tests are compiled with `g++`. Building them that way also checks that
the public header still compiles without CUDA headers.
