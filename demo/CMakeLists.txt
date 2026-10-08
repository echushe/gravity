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

add_executable(gravity_demo main.cpp)
target_link_libraries(gravity_demo PRIVATE gravity::gravity glfw GLEW::GLEW OpenGL::GL glm::glm)
