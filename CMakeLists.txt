cmake_minimum_required(VERSION 3.22)

# Default to the GPU in this machine unless -DCMAKE_CUDA_ARCHITECTURES=... is
# given. CMake >= 3.24 understands "native" itself; older versions get nvcc's
# own -arch=native instead (added to the compile options below).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES OFF)
        set(gravity_nvcc_native_arch ON)
    endif()
endif()

project(gravity VERSION 0.1.0 LANGUAGES CXX CUDA)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
    message(FATAL_ERROR "In-source builds are not supported. Use: cmake -B build")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Release or Debug)" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Link the shared CUDA runtime everywhere, so the library and any CUDA code in
# the same process use a single runtime instance.
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

find_package(CUDAToolkit REQUIRED)

add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>"
    # Device-side debug info in Debug; source line info for profilers otherwise.
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>"
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-lineinfo>"
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${gravity_nvcc_native_arch}>>:-arch=native>"
)

add_subdirectory(src)

option(GRAVITY_BUILD_TESTS "Build the test programs" ${PROJECT_IS_TOP_LEVEL})
if(GRAVITY_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
