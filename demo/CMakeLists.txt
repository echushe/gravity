# gravity_demo: OpenGL viewer for the simulation. Needs GLFW, GLEW, GLM and
# the OpenGL development files; on Ubuntu:
#   sudo apt install libglfw3-dev libglew-dev libglm-dev
find_package(OpenGL)
find_package(GLEW)
find_package(glfw3 3.3 CONFIG)
find_package(glm CONFIG)

if(NOT (OpenGL_FOUND AND GLEW_FOUND AND glfw3_FOUND AND glm_FOUND))
    message(FATAL_ERROR
        "The demo needs OpenGL, GLEW, GLFW and GLM. Install them with\n"
        "  sudo apt install libglfw3-dev libglew-dev libglm-dev\n"
        "or configure with -DGRAVITY_BUILD_DEMO=OFF to skip the demo.")
endif()

# stb_image_write.h (v1.16, public domain / MIT) writes the PNG snapshots. It
# is downloaded once per build directory, pinned to a commit and checked
# against its hash.
include(FetchContent)
FetchContent_Declare(stb
    URL https://github.com/nothings/stb/archive/2c980bb59875b0d32144a71867fbdebb2f77cd20.tar.gz
    URL_HASH SHA256=9a955b1b49a4410088a2e0ee2a9c057c3c907d0c1d75454144cb980aca0ba515
)
FetchContent_MakeAvailable(stb)
add_library(stb_image_write INTERFACE)
# SYSTEM: no -Wall / -Wextra warnings from the stb implementation.
target_include_directories(stb_image_write SYSTEM INTERFACE ${stb_SOURCE_DIR})

add_executable(gravity_demo main.cpp)
target_link_libraries(gravity_demo PRIVATE gravity::gravity glfw GLEW::GLEW OpenGL::GL glm::glm stb_image_write)
