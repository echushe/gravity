// OpenGL viewer for gravity::Simulation.
//
// Each frame advances the simulation by one step, copies the positions into
// an OpenGL vertex buffer and draws every object as a point, together with
// the outline of the simulation cube. The 3D -> 2D projection happens in the
// vertex shader, with a perspective camera that orbits the cube's centre.
//
// Controls:
//   left mouse drag   rotate the camera
//   scroll wheel      zoom
//   space             pause / resume
//   escape            quit
//
// Every kSnapshotInterval-th frame is also saved as a PNG file in the
// directory kSnapshotDir, relative to the current working directory.
#include <GL/glew.h>  // before GLFW, so that GLFW does not include its own gl.h
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#define STB_IMAGE_WRITE_IMPLEMENTATION  // compile the stb implementation into this file
#include <stb_image_write.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "gravity/simulation.h"

namespace
{

// Simulation set-up, in SI units: Simulation::step() uses the real G. The
// masses are chosen so that the cloud collapses within a few simulated
// seconds. The softening and the time step (one step per frame) are not set
// here: Simulation derives them from these values.
constexpr double kCubeSize = 100.0;  // m
constexpr std::size_t kNumObjects = 5000;
constexpr double kMassMean = 2.0e10;    // kg
constexpr double kMassStddev = 0.0; //0.4e10;  // kg
// Each velocity component is drawn uniformly from [-kMaxVelocity, kMaxVelocity].
// For the settings above: 0 gives a cold collapse, about 8 m/s keeps the cloud
// in virial equilibrium (2K = |U|), and above about 11.2 m/s it flies apart.
constexpr double kMaxVelocity = 0.0;  // m/s

constexpr int kWindowWidth = 1280;
constexpr int kWindowHeight = 800;
constexpr float kPointSize = 3.0f;  // pixels

// A PNG snapshot is saved every kSnapshotInterval iterations of the main loop,
// as kSnapshotDir/frame_<iteration>.png.
constexpr std::size_t kSnapshotInterval = 5;
const char* const kSnapshotDir = "snapshots";

// Prints one line to stdout, prefixed with the seconds since the first call,
// e.g. "[   1.234] window created". Takes printf-style arguments.
void log_message(const char* format, ...)
{
    static const auto start = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    std::printf("[%8.3f] ", elapsed.count());
    va_list args;
    va_start(args, format);
    std::vprintf(format, args);
    va_end(args);
    std::printf("\n");
    std::fflush(stdout);  // show it straight away even when stdout is piped
}

const char* const kVertexShader = R"(
#version 330 core
layout(location = 0) in vec3 position;
uniform mat4 view_projection;
uniform float point_size;
void main()
{
    gl_Position = view_projection * vec4(position, 1.0);
    gl_PointSize = point_size;
}
)";

const char* const kFragmentShader = R"(
#version 330 core
uniform vec4 color;
out vec4 frag_color;
void main()
{
    frag_color = color;
}
)";

struct Camera
{
    double yaw = 0.6;    // radians, around the y axis
    double pitch = 0.4;  // radians, above the x-z plane
    double distance = 2.5 * kCubeSize;
    bool dragging = false;
    double last_x = 0.0;
    double last_y = 0.0;
};

struct AppState
{
    Camera camera;
    bool paused = false;
};

AppState& app_state(GLFWwindow* window)
{
    return *static_cast<AppState*>(glfwGetWindowUserPointer(window));
}

void on_glfw_error(int code, const char* description)
{
    std::fprintf(stderr, "GLFW error %d: %s\n", code, description);
}

void on_key(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/)
{
    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_ESCAPE)
    {
        log_message("Esc pressed: closing the window");
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    if (key == GLFW_KEY_SPACE)
    {
        bool& paused = app_state(window).paused;
        paused = !paused;
        log_message("simulation %s", paused ? "paused" : "resumed");
    }
}

void on_mouse_button(GLFWwindow* window, int button, int action, int /*mods*/)
{
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;
    Camera& camera = app_state(window).camera;
    camera.dragging = (action == GLFW_PRESS);
    glfwGetCursorPos(window, &camera.last_x, &camera.last_y);
}

void on_cursor_pos(GLFWwindow* window, double x, double y)
{
    Camera& camera = app_state(window).camera;
    if (!camera.dragging) return;
    constexpr double kRadiansPerPixel = 0.005;
    camera.yaw -= kRadiansPerPixel * (x - camera.last_x);
    camera.pitch += kRadiansPerPixel * (y - camera.last_y);
    // Stay short of straight up / down, where lookAt's up vector degenerates.
    camera.pitch = std::clamp(camera.pitch, -1.5, 1.5);
    camera.last_x = x;
    camera.last_y = y;
}

void on_scroll(GLFWwindow* window, double /*dx*/, double dy)
{
    Camera& camera = app_state(window).camera;
    camera.distance *= std::pow(0.9, dy);
    camera.distance = std::clamp(camera.distance, 0.1 * kCubeSize, 20.0 * kCubeSize);
}

// Computed in double; converted to float only for the shader, because GLSL
// 3.30 has no double uniforms and OpenGL rasterises in float anyway.
glm::mat4 view_projection(const Camera& camera, double aspect)
{
    const glm::dvec3 centre(0.5 * kCubeSize);
    const glm::dvec3 direction(std::cos(camera.pitch) * std::sin(camera.yaw),
                               std::sin(camera.pitch),
                               std::cos(camera.pitch) * std::cos(camera.yaw));
    const glm::dmat4 view = glm::lookAt(centre + camera.distance * direction, centre, glm::dvec3(0.0, 1.0, 0.0));
    const glm::dmat4 projection =
        glm::perspective(glm::radians(45.0), aspect, 0.01 * kCubeSize, 100.0 * kCubeSize);
    return glm::mat4(projection * view);
}

GLuint compile_shader(GLenum type, const char* source)
{
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[1024];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        throw std::runtime_error(std::string("shader compilation failed:\n") + log);
    }
    return shader;
}

GLuint link_program(const char* vertex_source, const char* fragment_source)
{
    const GLuint vertex_shader = compile_shader(GL_VERTEX_SHADER, vertex_source);
    const GLuint fragment_shader = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
    const GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);
    // The shaders are only flagged for deletion here; they live as long as the program.
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char log[1024];
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        throw std::runtime_error(std::string("shader linking failed:\n") + log);
    }
    return program;
}

// A vertex array with a single buffer of vec3 positions at attribute 0.
struct Mesh
{
    GLuint vao = 0;
    GLuint vbo = 0;
    GLsizei num_vertices = 0;
};

// The buffer holds doubles; OpenGL converts each vertex to the shader's float
// vec3 when it reads it.
Mesh make_mesh(const double* xyz, std::size_t num_vertices, GLenum usage)
{
    Mesh mesh;
    mesh.num_vertices = static_cast<GLsizei>(num_vertices);
    glGenVertexArrays(1, &mesh.vao);
    glBindVertexArray(mesh.vao);
    glGenBuffers(1, &mesh.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
    glBufferData(GL_ARRAY_BUFFER, 3 * num_vertices * sizeof(double), xyz, usage);
    glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, 3 * sizeof(double), nullptr);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
    return mesh;
}

// The 12 edges of the simulation cube, as pairs of vertices for GL_LINES.
std::vector<double> cube_edges()
{
    const double s = kCubeSize;
    const double corners[8][3] = {
        {0, 0, 0}, {s, 0, 0}, {s, s, 0}, {0, s, 0}, {0, 0, s}, {s, 0, s}, {s, s, s}, {0, s, s}};
    const int edges[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
    std::vector<double> xyz;
    for (const auto& edge : edges)
    {
        for (int corner : edge) xyz.insert(xyz.end(), corners[corner], corners[corner] + 3);
    }
    return xyz;
}

// Prints a summary of the simulation state to stdout: the softening length
// epsilon and time step T, the object farthest from the mean position, the
// mean position, the fastest object and the mean velocity. The means are
// plain averages over all objects, not weighted by mass.
void print_status(const gravity::Simulation& simulation, double simulated_time)
{
    log_message("t = %.2f s, epsilon = %.4g m, T = %.4g s",
                simulated_time,
                simulation.softening(),
                simulation.time_step());

    const gravity::PositionData positions = simulation.positions();
    const gravity::VelocityData velocities = simulation.velocities();
    const std::size_t n = positions.size();
    if (n == 0) return;
    auto position = [&](std::size_t i) { return glm::dvec3(positions.x(i), positions.y(i), positions.z(i)); };
    auto velocity = [&](std::size_t i) { return glm::dvec3(velocities.x(i), velocities.y(i), velocities.z(i)); };

    glm::dvec3 mean_position(0.0);
    glm::dvec3 mean_velocity(0.0);
    for (std::size_t i = 0; i < n; ++i)
    {
        mean_position += position(i);
        mean_velocity += velocity(i);
    }
    mean_position /= static_cast<double>(n);
    mean_velocity /= static_cast<double>(n);

    std::size_t farthest = 0;
    std::size_t fastest = 0;
    double max_distance = -1.0;
    double max_speed = -1.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        const double distance = glm::length(position(i) - mean_position);
        const double speed = glm::length(velocity(i));
        if (distance > max_distance)
        {
            max_distance = distance;
            farthest = i;
        }
        if (speed > max_speed)
        {
            max_speed = speed;
            fastest = i;
        }
    }

    const glm::dvec3 p = position(farthest);
    const glm::dvec3 v = velocity(fastest);
    char label[48];
    std::snprintf(label, sizeof(label), "farthest (#%zu):", farthest);
    std::printf("  %-18s position (%10.4f, %10.4f, %10.4f) m, %.4f m from the mean position\n",
                label, p.x, p.y, p.z, max_distance);
    std::printf("  %-18s position (%10.4f, %10.4f, %10.4f) m\n",
                "mean:", mean_position.x, mean_position.y, mean_position.z);
    std::snprintf(label, sizeof(label), "fastest (#%zu):", fastest);
    std::printf("  %-18s velocity (%10.4f, %10.4f, %10.4f) m/s, speed %.4f m/s\n",
                label, v.x, v.y, v.z, max_speed);
    // Scientific notation: with equal masses the mean velocity is the
    // centre-of-mass velocity, which stays at rounding level (~1e-16).
    std::printf("  %-18s velocity (%10.3e, %10.3e, %10.3e) m/s, speed %.3e m/s\n",
                "mean:", mean_velocity.x, mean_velocity.y, mean_velocity.z, glm::length(mean_velocity));
    std::fflush(stdout);  // show it straight away even when stdout is piped
}

// Saves the back buffer, i.e. the frame that the next glfwSwapBuffers() will
// show, as a PNG file. Returns false if the file could not be written.
bool save_snapshot(const char* path, int width, int height)
{
    std::vector<unsigned char> rgb(static_cast<std::size_t>(width) * height * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);  // rows of RGB bytes, not padded to 4 bytes
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, rgb.data());
    stbi_flip_vertically_on_write(1);  // OpenGL's first row is the bottom one, PNG's the top one
    return stbi_write_png(path, width, height, 3, rgb.data(), width * 3) != 0;
}

// Calls glfwTerminate() on every way out of run(). That also destroys the
// window and its OpenGL context, which frees every GL object created in it.
struct GlfwSession
{
    GlfwSession()
    {
        if (!glfwInit()) throw std::runtime_error("glfwInit failed");
        log_message("GLFW %s initialised", glfwGetVersionString());
    }
    ~GlfwSession()
    {
        glfwTerminate();
        log_message("GLFW terminated");
    }
    GlfwSession(const GlfwSession&) = delete;
    GlfwSession& operator=(const GlfwSession&) = delete;
};

void run()
{
    log_message("starting gravity_demo");
    glfwSetErrorCallback(on_glfw_error);
    GlfwSession glfw;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    GLFWwindow* window = glfwCreateWindow(kWindowWidth, kWindowHeight, "gravity demo", nullptr, nullptr);
    if (!window) throw std::runtime_error("could not create an OpenGL 3.3 window");
    log_message("created %dx%d window", kWindowWidth, kWindowHeight);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);  // vsync: one simulation step per displayed frame

    glewExperimental = GL_TRUE;  // load every entry point, which a core profile needs
    if (const GLenum error = glewInit(); error != GLEW_OK)
    {
        throw std::runtime_error(std::string("glewInit failed: ") +
                                 reinterpret_cast<const char*>(glewGetErrorString(error)));
    }
    log_message("GLEW %s initialised", reinterpret_cast<const char*>(glewGetString(GLEW_VERSION)));
    log_message("OpenGL %s on %s",
                reinterpret_cast<const char*>(glGetString(GL_VERSION)),
                reinterpret_cast<const char*>(glGetString(GL_RENDERER)));

    AppState state;
    glfwSetWindowUserPointer(window, &state);
    glfwSetKeyCallback(window, on_key);
    glfwSetMouseButtonCallback(window, on_mouse_button);
    glfwSetCursorPosCallback(window, on_cursor_pos);
    glfwSetScrollCallback(window, on_scroll);

    const GLuint program = link_program(kVertexShader, kFragmentShader);
    const GLint view_projection_location = glGetUniformLocation(program, "view_projection");
    const GLint point_size_location = glGetUniformLocation(program, "point_size");
    const GLint color_location = glGetUniformLocation(program, "color");
    log_message("shader program compiled and linked");

    gravity::Simulation simulation(kCubeSize, kNumObjects, kMassMean, kMassStddev, kMaxVelocity);
    log_message("simulation created: %zu objects in a %.0f m cube, mass %.3g +/- %.3g kg, "
                "velocity components up to %.3g m/s, softening %.3g m, time step %.3g s",
                kNumObjects,
                kCubeSize,
                kMassMean,
                kMassStddev,
                kMaxVelocity,
                simulation.softening(),
                simulation.time_step());

    const std::vector<double> edges = cube_edges();
    const Mesh cube = make_mesh(edges.data(), edges.size() / 3, GL_STATIC_DRAW);
    // PositionData is laid out as x0, y0, z0, x1, ..., which is exactly the
    // vertex layout, so it is copied into the buffer as is.
    const Mesh objects = make_mesh(simulation.positions().data(), kNumObjects, GL_DYNAMIC_DRAW);
    log_message("vertex buffers created: %d cube vertices, %d object vertices",
                cube.num_vertices,
                objects.num_vertices);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_MULTISAMPLE);
    glClearColor(0.02f, 0.02f, 0.05f, 1.0f);

    double simulated_time = 0.0;
    std::size_t num_frames = 0;
    std::size_t num_steps = 0;
    std::size_t num_snapshots = 0;
    double title_time = glfwGetTime();
    int title_frames = 0;
    print_status(simulation, simulated_time);

    std::filesystem::create_directories(kSnapshotDir);
    log_message("saving a snapshot every %zu frames to %s",
                kSnapshotInterval,
                std::filesystem::absolute(kSnapshotDir).c_str());

    log_message("entering the main loop (Space: pause / resume, Esc or close button: quit)");
    while (!glfwWindowShouldClose(window))
    {
        ++num_frames;
        if (!state.paused)
        {
            simulation.step();
            simulated_time += simulation.time_step();
            ++num_steps;
            const gravity::PositionData positions = simulation.positions();
            glBindBuffer(GL_ARRAY_BUFFER, objects.vbo);
            glBufferSubData(GL_ARRAY_BUFFER, 0, 3 * positions.size() * sizeof(double), positions.data());
        }

        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // A minimised window has a zero-sized framebuffer: nothing to draw.
        if (width > 0 && height > 0)
        {
            const glm::mat4 vp = view_projection(state.camera, static_cast<double>(width) / height);
            glUseProgram(program);
            glUniformMatrix4fv(view_projection_location, 1, GL_FALSE, glm::value_ptr(vp));
            glUniform1f(point_size_location, kPointSize);

            glUniform4f(color_location, 0.35f, 0.35f, 0.45f, 1.0f);
            glBindVertexArray(cube.vao);
            glDrawArrays(GL_LINES, 0, cube.num_vertices);

            // Plain square points. Do not round them with gl_PointCoord: Mesa's
            // Intel driver leaves it at (0, 0), so every fragment would be cut.
            glUniform4f(color_location, 1.0f, 0.85f, 0.55f, 1.0f);
            glBindVertexArray(objects.vao);
            glDrawArrays(GL_POINTS, 0, objects.num_vertices);

            // Read the frame back before glfwSwapBuffers(), after which the
            // back buffer's contents are undefined.
            if (num_frames % kSnapshotInterval == 0)
            {
                char path[256];
                std::snprintf(path, sizeof(path), "%s/frame_%06zu.png", kSnapshotDir, num_frames);
                if (save_snapshot(path, width, height))
                {
                    ++num_snapshots;
                }
                else
                {
                    log_message("could not write %s", path);
                }
            }
        }

        glfwSwapBuffers(window);
        glfwPollEvents();

        ++title_frames;
        const double now = glfwGetTime();
        if (now - title_time >= 0.5)
        {
            char title[128];
            std::snprintf(title,
                          sizeof(title),
                          "gravity demo - %zu objects - t = %.1f s - %.0f fps%s",
                          kNumObjects,
                          simulated_time,
                          title_frames / (now - title_time),
                          state.paused ? " - paused" : "");
            glfwSetWindowTitle(window, title);
            title_time = now;
            title_frames = 0;
            if (!state.paused) print_status(simulation, simulated_time);
        }
    }
    log_message("main loop finished: %zu frames drawn, %zu simulation steps, t = %.2f s, %zu snapshots saved",
                num_frames,
                num_steps,
                simulated_time,
                num_snapshots);
}

}  // namespace

int main()
{
    try
    {
        run();
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "gravity_demo: %s\n", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
