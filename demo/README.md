# gravity_demo

An OpenGL viewer for `gravity::Simulation`. It opens a window, advances the
simulation by one step per frame, and draws every object as a point inside the
outline of the simulation cube. A perspective camera orbits the cube's centre,
and the vertex shader does the 3D → 2D projection. While it runs, the demo
logs to stdout and can save frames as PNG files.

It uses GLFW (window and input), GLEW (OpenGL loader), GLM (matrices) and
OpenGL 3.3 core profile. Everything is in [main.cpp](main.cpp).

## Building

Install the dependencies once (Ubuntu):

```sh
sudo apt install libglfw3-dev libglew-dev libglm-dev
```

Then, from the project root:

```sh
cmake -B build                                 # the demo is on by default
cmake --build build --target gravity_demo
```

The first configure downloads `stb_image_write.h` (v1.16, public domain / MIT),
which writes the PNG files. The download is pinned to a commit and checked
against its SHA-256 (see [CMakeLists.txt](CMakeLists.txt)), and is reused
afterwards. Configure with `-DGRAVITY_BUILD_DEMO=OFF` to skip the demo.

## Running

```sh
./build/demo/gravity_demo
```

Run it from the directory where you want the `snapshots/` folder; normally
the project root.

On a laptop with two GPUs, OpenGL runs on the integrated GPU by default (the
log shows which one). The CUDA simulation always runs on the NVIDIA GPU. To
draw on the NVIDIA GPU as well:

```sh
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./build/demo/gravity_demo
```

| Control | Action |
|---------|--------|
| Left mouse drag | Rotate the camera |
| Scroll wheel | Zoom |
| Space | Pause / resume the simulation (the view can still be rotated) |
| Esc, or the window's close button | Quit |

The window title shows the number of objects, the simulated time and the
frame rate.

## Settings

All settings are constants at the top of [main.cpp](main.cpp); change them and
rebuild.

| Constant | Meaning |
|----------|---------|
| `kCubeSize` | Side of the simulation cube, in metres. |
| `kNumObjects` | Number of objects. |
| `kMassMean`, `kMassStddev` | Normal distribution of the masses, in kg. |
| `kTimeStep` | Simulated seconds per frame. |
| `kWindowWidth`, `kWindowHeight` | Initial window size, in pixels. |
| `kPointSize` | Size of each drawn object, in pixels. |
| `kNumLoggedPositions` | How many positions are printed to stdout. |
| `kSnapshotInterval` | A PNG is saved every this many loop iterations. |
| `kSnapshotDir` | Folder for the PNG files, relative to the working directory. |

The simulation uses SI units and the real gravitational constant, so the
masses have to be large to see anything move: with 2e10 kg per object in a
100 m cube, the cloud collapses in about ten simulated seconds.

## Output

**Log.** Every line on stdout starts with the seconds since start-up: start-up
steps, the GPU used for drawing, the simulation settings, pause/resume and
Esc, and a summary at exit. Every half second (while not paused) the
positions of the first `kNumLoggedPositions` objects are printed:

```
[   0.186] OpenGL 4.6 (Core Profile) Mesa 23.2.1-1ubuntu3.1~22.04.4 on Mesa Intel(R) Graphics (ADL GT2)
[   0.196] simulation created: 5000 objects in a 100 m cube, mass 2e+10 +/- 0 kg, time step 0.01 s
[   0.206] entering the main loop (Space: pause / resume, Esc or close button: quit)
[   0.694] t = 0.34 s, positions of the first 10 objects:
   0: (   99.5982,    64.1734,    79.7446)
   ...
```

**Snapshots.** Every `kSnapshotInterval` iterations of the main loop, the
frame is saved as `snapshots/frame_<iteration>.png` (iteration numbers are
zero-padded to six digits). The folder is created on start-up and is ignored
by git. A 1280×800 frame is about 100 KB. A new run overwrites files with the
same names but leaves higher-numbered frames from a longer earlier run, so
clear the folder before making a video.

**Video.** To combine the snapshots into an MP4 (the frame numbers are not
consecutive, so match them with a glob):

```sh
ffmpeg -framerate 24 -pattern_type glob -i 'snapshots/frame_*.png' \
       -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart snapshots/gravity_demo.mp4
```

At 60 fps with a snapshot every 5 iterations, the demo saves 12 frames per
second, so `-framerate 12` plays back at real speed and 24 at twice that.

## How the main loop works

Each iteration of the `while` loop in `run()`:

1. calls `simulation.step(kTimeStep)` (skipped while paused),
2. copies `simulation.positions()` into the vertex buffer with
   `glBufferSubData`; `PositionData` is already in the vertex layout
   (`x0, y0, z0, x1, ...`),
3. draws the cube outline (`GL_LINES`) and the objects (`GL_POINTS`),
4. on every `kSnapshotInterval`-th iteration, reads the frame back with
   `glReadPixels` and writes the PNG. This has to happen before the swap,
   which leaves the back buffer undefined,
5. calls `glfwSwapBuffers`, which waits for the monitor's next refresh
   (vsync), so the loop runs at the refresh rate (about 60 times a second),
6. calls `glfwPollEvents`, which runs the key and mouse callbacks for any
   input since the last iteration. It does not wait for input.

The loop ends when the window's "should close" flag is set, by Esc or the
close button.

## Known issues

- **Square points.** Points are drawn as plain squares. Rounding them with
  `gl_PointCoord` does not work on Mesa's Intel driver (23.2.1), which leaves
  `gl_PointCoord` at (0, 0), so every fragment was discarded and nothing but
  the cube was visible. At 3 px a square looks the same as a disc. For larger
  round points, compare `gl_FragCoord` with the point's centre instead.
- **Objects thrown far away.** The kernel adds only 1e-10 m² to each squared
  distance, so two objects that pass very close get enormous accelerations
  and fly off. A larger softening (around 1 m² for the default settings) or a
  smaller time step reduces this.
- **Copies every frame.** `step()` copies the whole state to the GPU and back,
  `positions()` returns a copy, and the demo uploads the positions to OpenGL
  again. This is fine for thousands of objects. If it becomes a bottleneck,
  keep the state on the GPU and share the vertex buffer with CUDA
  (CUDA–OpenGL interop).
- **Snapshots cost time.** `glReadPixels` stalls the GPU and PNG encoding
  runs on the CPU. With a snapshot every 5 iterations, the frame rate drops
  from about 60 to about 55 fps.
