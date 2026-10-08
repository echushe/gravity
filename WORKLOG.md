# Work log

What was done on the project, in order, with the reasons behind the less
obvious changes. Open follow-ups are collected at the end.

## 2026-10-08 (with Claude Code)

### 1. Fixed `Simulation::step()` not compiling

- `src/simulation.cpp` called `calculate_gravity_velocity_and_position`
  without including the header that declares it. Added
  `#include "gravity/gravity.h"`.
- In `src/nbody_kernels.cu`, `calculate_gravity_velocity_and_position` called
  `calculate_gravity`, the host-array version, with device pointers: it would
  have copied "host" arrays that were really device memory. Changed it to
  `calculate_gravity_device`. (At the time this was described as a wrong
  argument order that would not compile. That was wrong: the call matched the
  definition in the `.cu` file. It was the header's declaration that had a
  different order, as found in item 10.)
- Verified: the whole project builds and both tests pass.
- Commits: `2deb9d5`, `03dd7e2`.

### 2. Chose a display engine for the demo

- Compared raylib (built-in 3D camera, least code), SDL (2D renderer,
  projection by hand) and GLFW + OpenGL (most code, but allows CUDA–OpenGL
  interop later). GLFW + OpenGL was chosen, accepting a CPU → GPU copy of the
  positions every frame for now.

### 3. Added the OpenGL demo (`demo/`)

- `demo/main.cpp`: GLFW window, OpenGL 3.3 core profile, GLEW, GLM. Orbit
  camera (drag to rotate, scroll to zoom), Space to pause, Esc to quit. Draws
  the cube outline and one point per object; the vertex shader does the
  perspective projection.
- `demo/CMakeLists.txt` and the `GRAVITY_BUILD_DEMO` option in the top-level
  `CMakeLists.txt`. Configuring fails with install instructions if a
  dependency is missing.
- Dependencies: `sudo apt install libglfw3-dev libglew-dev libglm-dev`.
- Masses and the time step were chosen so that the cloud visibly collapses
  with the real G (2e10 kg per object in a 100 m cube).
- Commit: `81027d6`. The public `positions()`, `velocities()` and `masses()`
  accessors the demo needs came in `3a8a344`.

### 4. Explained how the main loop is paced

- No code change. The loop runs until the window's "should close" flag is set
  (Esc or the close button); `glfwSwapBuffers` waits for vsync, and
  `glfwPollEvents` handles input without waiting. Now described in
  `demo/README.md`.

### 5. Logged positions and added stdout logging

- The first 10 positions are printed at start-up and every half second while
  running.
- `log_message()` prints timestamped lines for start-up steps, the GPU used
  for drawing, the simulation settings, pause/resume, Esc, and a summary at
  exit.
- Commit: `a0d1c5e` (together with item 6).

### 6. Fixed the demo showing only the cube

- Symptom: the positions in the log were correct, but no objects were drawn.
- Diagnosis: a debug build read the framebuffer back. All objects were inside
  the view, the vertex buffer held the right data and there were no OpenGL
  errors, yet there were 0 point-coloured pixels. Turning off the
  round-point `discard` brought back about 14,000. Rendering `gl_PointCoord`
  as a colour showed that Mesa's Intel driver (23.2.1) leaves it at (0, 0),
  which the disc test always discards. On the NVIDIA GPU it was correct.
- Fix: draw plain square points (no visible difference at 3 px) and leave a
  comment explaining why.
- Verified: about 14,000 point pixels on the Intel GPU.
- Commit: `a0d1c5e`.

### 7. Added PNG snapshots

- Every `kSnapshotInterval` iterations of the main loop, the frame is read
  with `glReadPixels` (before the buffer swap) and written to
  `snapshots/frame_<iteration>.png`. The folder is created at start-up and
  added to `.gitignore`.
- PNG writing uses `stb_image_write.h` v1.16 (public domain / MIT), which
  CMake downloads with `FetchContent`, pinned to commit `2c980bb` and checked
  against its SHA-256, so no extra `sudo` install is needed.
- Verified with a snapshot every 5 iterations: 42 correct 1280×800 images in
  4 s, at about 55 fps instead of 60.
- Commit: `257b7f1`.

### 8. Made a video from the snapshots

- Combined 590 snapshots (frames 5 to 2950) into `snapshots/gravity_demo.mp4`
  with ffmpeg: H.264, 24 fps (twice real speed), 24.6 s, 19.7 MB. The command
  is in `demo/README.md`.

### 9. Updated the documentation

- `src/README.md`: project layout (demo, tests, this log), demo requirements,
  the `GRAVITY_BUILD_DEMO` option, a usage example with `step()` and
  `positions()`, and API entries for `step()`, the accessors and the GPU
  compute functions.
- `tests/README.md`: how to build without the demo's dependencies, and what is
  not tested yet.
- `demo/README.md` (new): building, running, controls, settings, output
  (log, snapshots, video), how the main loop works, known issues.
- `WORKLOG.md` (this file).

### 10. Converted all floating-point maths to double (CPU and GPU)

- Library: `MassData` and the 3D data classes store `double`; `Simulation`
  takes and draws `double`s (`normal_distribution<double>`,
  `uniform_real_distribution<double>`); `step(double)`.
- CUDA: every kernel and host wrapper uses `double`, `DeviceBuffer<double>`,
  `rsqrt` instead of `rsqrtf`, and literals without the `f` suffix.
- Found and fixed two public declarations in `gravity.h` that did not match
  their definitions, so calling them as declared would have failed to link:
  `calculate_gravity` (header had `G` before the output array) and
  `calculate_velocity_device` (definition had `n` before `T`). Both now use
  the same order as their host/device counterpart.
- Tests: literals changed to `double`, and `static_assert`s check that every
  data class stores `double`.
- Demo: settings, simulated time, camera maths (`glm::dvec3`/`dmat4`) and
  vertex buffers (`GL_DOUBLE`) are `double`. Only the drawing stays `float`
  (GLSL 3.30 has no doubles; see "Precision" in `demo/README.md`).
- Verified: everything builds without new warnings and both tests pass. A
  separate program called all seven compute functions (all now link) and
  matched a CPU double reference to 1e-14 or better. The demo still draws
  every point (framebuffer check) and runs at 60 fps with 5,000 objects.
- Cost: FP64 runs at 1/64 of the FP32 rate on the RTX A2000. `step()` went
  from 1.5 to 9.2 ms for 5,000 objects and from 7.3 to 124 ms for 20,000.

### 11. Derived the softening length and the time step from the setup

- Before: the kernels added a fixed 1e-10 m² to r² (ε = 10 µm) and the demo
  used a fixed T. A close pass at ε lasts about √(ε³/(G·m)) ≈ 3e-8 s, so T was
  ~400,000 times too long and close pairs were flung out of the cube.
- Now `Simulation` sets:
  - ε = `kSofteningFraction` × L / ∛N (0.05 of the mean spacing; ∛N with N
    at least 1),
  - T = `kTimeStepFraction` × √(ε³ / (G·m̄)) (0.1), with m̄ the mean of the
    drawn masses (`mass_mean` when there are no objects).
  `default_softening()` and `default_time_step()` are public, `softening()`
  and `time_step()` return the values in use, `step()` advances by
  `time_step()`, and `step(T)` still takes an explicit step. G is
  `Simulation::kGravitationalConstant`.
- `calculate_gravity`, `calculate_gravity_device`,
  `calculate_gravity_velocity_and_position` and both gravity kernels take
  `epsilon` (a length) after `G` and use r² + ε².
- The demo no longer has `kTimeStep`; it calls `step()` and logs ε and T
  (0.292 m and 0.0137 s for its settings), and still runs at 60 fps.
- Tests: the two formulas, the mean-mass rule (equal, varied and no masses),
  and `step(T)` rejecting T ≤ 0 or NaN.
- Choosing the fractions: a cold collapse (N = 2,000, uniform cube at rest,
  run for 1.5 free-fall times) was simulated for k_ε ∈ {0.02, 0.05, 0.1} and
  k_T ∈ {0.03, 0.1, 0.3}, measuring the relative energy error and the bodies
  beyond 3 L:
  - old fixed values: energy error 814× the initial energy, 95 bodies out;
  - new rule with the current integrator: 0 bodies out in every case, but
    energy errors of 8–190%, roughly proportional to T and almost
    independent of ε. The cause is the integrator: `v += a·T` with only the
    old acceleration is first order and adds energy;
  - new rule with a leapfrog (kick-drift-kick) integrator: 1e-5 to 7e-2;
    1e-3 for k_ε = 0.05, k_T = 0.1. At N = 5,000: 0.4% (leapfrog) against 98%
    (current), 0 bodies out with either.
  k_ε = 0.05 and k_T = 0.1 were kept: good accuracy with a proper integrator,
  and the same simulation speed on screen as before.

## Open follow-ups

- **Integrator.** Switch `step()` to leapfrog / velocity Verlet: half kick
  `v += ½·a·T`, drift `x += v·T`, new accelerations, half kick. It needs the
  accelerations kept between steps (one force calculation per step, as now)
  and cuts the energy error from ~100% to ~0.4% in the demo's collapse.
- **Tests.** `Simulation::step()` and the GPU compute functions in
  `gravity.h` have no tests.
- **FP64 speed.** If large object counts matter more than the last digits,
  consider a compile-time choice of precision (a `using real = double;` alias
  in one header), since FP64 is 1/64 rate on this GPU.
- **Copies.** `positions()`, `velocities()` and `masses()` return copies, and
  `step()` copies the whole state to the GPU and back on every call. Returning
  `const&`, and later keeping the state on the GPU with CUDA–OpenGL interop,
  would remove them if they become a bottleneck.
- **Unused kernels.** nvcc warns that `scale_1D`,
  `calculate_gravity_kernel_2D` and `accelerations_2D_to_1D` are never used.
- **Doc comments.** Most functions in `gravity.h` have no comments.
- **Editor errors.** VS Code's code analysis reported "no member
  `positions`" in `demo/main.cpp` while the real build compiled fine; it was
  working from an out-of-date copy of the header.
