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
#include <GL/glew.h>  // before GLFW, so that GLFW does not include its own gl.h
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "gravity/simulation.h"

namespace
{

// Simulation set-up, in SI units: Simulation::step() uses the real G. The
// masses are chosen so that the cloud collapses in about ten simulated
// seconds, i.e. a few seconds on screen at 60 frames per second.
constexpr float kCubeSize = 100.0f;  // m
constexpr std::size_t kNumObjects = 2000;
constexpr float kMassMean = 2.0e10f;    // kg
constexpr float kMassStddev = 0.4e10f;  // kg
constexpr float kTimeStep = 0.02f;      // simulated seconds per frame

constexpr int kWindowWidth = 1280;
constexpr int kWindowHeight = 800;
constexpr float kPointSize = 3.0f;  // pixels

// How many positions are printed to stdout, every half second.
constexpr std::size_t kNumLoggedPositions = 10;

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
    float yaw = 0.6f;    // radians, around the y axis
    float pitch = 0.4f;  // radians, above the x-z plane
    float distance = 2.5f * kCubeSize;
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
    constexpr float kRadiansPerPixel = 0.005f;
    camera.yaw -= kRadiansPerPixel * static_cast<float>(x - camera.last_x);
    camera.pitch += kRadiansPerPixel * static_cast<float>(y - camera.last_y);
    // Stay short of straight up / down, where lookAt's up vector degenerates.
    camera.pitch = std::clamp(camera.pitch, -1.5f, 1.5f);
    camera.last_x = x;
    camera.last_y = y;
}

void on_scroll(GLFWwindow* window, double /*dx*/, double dy)
{
    Camera& camera = app_state(window).camera;
    camera.distance *= std::pow(0.9f, static_cast<float>(dy));
    camera.distance = std::clamp(camera.distance, 0.1f * kCubeSize, 20.0f * kCubeSize);
}

glm::mat4 view_projection(const Camera& camera, float aspect)
{
    const glm::vec3 centre(0.5f * kCubeSize);
    const glm::vec3 direction(std::cos(camera.pitch) * std::sin(camera.yaw),
                              std::sin(camera.pitch),
                              std::cos(camera.pitch) * std::cos(camera.yaw));
    const glm::mat4 view = glm::lookAt(centre + camera.distance * direction, centre, glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::mat4 projection =
        glm::perspective(glm::radians(45.0f), aspect, 0.01f * kCubeSize, 100.0f * kCubeSize);
    return projection * view;
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

Mesh make_mesh(const float* xyz, std::size_t num_vertices, GLenum usage)
{
    Mesh mesh;
    mesh.num_vertices = static_cast<GLsizei>(num_vertices);
    glGenVertexArrays(1, &mesh.vao);
    glBindVertexArray(mesh.vao);
    glGenBuffers(1, &mesh.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
    glBufferData(GL_ARRAY_BUFFER, 3 * num_vertices * sizeof(float), xyz, usage);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
    return mesh;
}

// The 12 edges of the simulation cube, as pairs of vertices for GL_LINES.
std::vector<float> cube_edges()
{
    const float s = kCubeSize;
    const float corners[8][3] = {
        {0, 0, 0}, {s, 0, 0}, {s, s, 0}, {0, s, 0}, {0, 0, s}, {s, 0, s}, {s, s, s}, {0, s, s}};
    const int edges[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
    std::vector<float> xyz;
    for (const auto& edge : edges)
    {
        for (int corner : edge) xyz.insert(xyz.end(), corners[corner], corners[corner] + 3);
    }
    return xyz;
}

// Prints the positions of the first kNumLoggedPositions objects to stdout.
void print_positions(const gravity::PositionData& positions, float simulated_time)
{
    const std::size_t n = std::min(kNumLoggedPositions, positions.size());
    log_message("t = %.2f s, positions of the first %zu objects:", simulated_time, n);
    for (std::size_t i = 0; i < n; ++i)
    {
        std::printf("  %2zu: (%10.4f, %10.4f, %10.4f)\n", i, positions.x(i), positions.y(i), positions.z(i));
    }
    std::fflush(stdout);  // show it straight away even when stdout is piped
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

    gravity::Simulation simulation(kCubeSize, kNumObjects, kMassMean, kMassStddev);
    log_message("simulation created: %zu objects in a %.0f m cube, mass %.3g +/- %.3g kg, time step %.3g s",
                kNumObjects,
                kCubeSize,
                kMassMean,
                kMassStddev,
                kTimeStep);

    const std::vector<float> edges = cube_edges();
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

    float simulated_time = 0.0f;
    std::size_t num_frames = 0;
    std::size_t num_steps = 0;
    double title_time = glfwGetTime();
    int title_frames = 0;
    print_positions(simulation.positions(), simulated_time);

    log_message("entering the main loop (Space: pause / resume, Esc or close button: quit)");
    while (!glfwWindowShouldClose(window))
    {
        ++num_frames;
        if (!state.paused)
        {
            simulation.step(kTimeStep);
            simulated_time += kTimeStep;
            ++num_steps;
            const gravity::PositionData positions = simulation.positions();
            glBindBuffer(GL_ARRAY_BUFFER, objects.vbo);
            glBufferSubData(GL_ARRAY_BUFFER, 0, 3 * positions.size() * sizeof(float), positions.data());
        }

        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // A minimised window has a zero-sized framebuffer: nothing to draw.
        if (width > 0 && height > 0)
        {
            const glm::mat4 vp = view_projection(state.camera, static_cast<float>(width) / height);
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
            if (!state.paused) print_positions(simulation.positions(), simulated_time);
        }
    }
    log_message("main loop finished: %zu frames drawn, %zu simulation steps, t = %.2f s",
                num_frames,
                num_steps,
                simulated_time);
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
