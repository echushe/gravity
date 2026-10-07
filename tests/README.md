# libgravity tests

Each `test_*.cpp` or `test_*.cu` file in this directory builds into its own
standalone executable, which you can run directly. There is no test framework
to install: everything needed is in `test_common.h`.

## Layout

```
tests/
├── test_common.h      CHECK / CHECK_NEAR / CHECK_THROWS macros and test::run()
├── test_device.cpp    device enumeration, DeviceInfo fields, CudaError on bad ids
├── test_saxpy.cu      saxpy() and saxpy_device() against a CPU reference
└── Makefile
```

Executables are written to `../build/tests/`.

## Building and running

From this directory:

```sh
make                    # build every test (builds ../src's libgravity.a first if needed)
make run                # build and run every test, then print a summary
make run-test_saxpy     # build and run one test
make test_saxpy         # build one test without running it
make clean              # remove the test binaries (the library is left alone)
```

You can also run a built test directly:

```sh
../build/tests/test_saxpy
```

From the project root, `make test` does the same as `make run`.

`make run` output looks like this:

```
== test_device
  device 0: NVIDIA RTX A2000 8GB Laptop GPU, sm_86, 20 SMs, 7.7 GiB
[PASS] test_device
== test_saxpy
[PASS] test_saxpy
----
2 passed, 0 failed
```

The variables `GPU_ARCH`, `BUILD` and `CUDA_PATH` work the same way as for
the library (see [../src/README.md](../src/README.md)), and the Makefile
passes them on to the library build. For example, `make run BUILD=debug`
builds the library and the tests with debug info. Run `make clean` in both
directories when you switch.

## Test conventions

- **Exit status:** 0 means pass or skip; 1 means at least one check failed.
  `make run` exits non-zero if any test failed, so it can be used in CI.
- **No GPU:** if no CUDA device is visible, a test prints `[SKIP]` and exits 0.
- **Checks don't abort.** `CHECK(cond)`, `CHECK_NEAR(actual, expected, tol)`
  and `CHECK_THROWS(expr, Type)` print the file, line and expression of a
  failure and keep going, so one run reports every failing check.
- **Unexpected exceptions** (for example a `gravity::CudaError` thrown from
  inside a test) are caught by `test::run()`, printed, and counted as a
  failure.

## Adding a test

1. Create `test_<name>.cpp` if the test uses only the public `gravity` API.
   Create `test_<name>.cu` if it also needs to call CUDA directly or define
   kernels; it is then built and linked with `nvcc`.
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

3. Run `make run`. The Makefile finds new `test_*` files automatically.

`.cpp` tests are compiled with `g++` and linked against `libgravity.a` and
`-lcudart`. Building them that way also checks that the public header still
compiles without CUDA headers.
