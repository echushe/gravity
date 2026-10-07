# libgravity: static by default, shared with -DBUILD_SHARED_LIBS=ON.
add_library(gravity
    device.cpp
    saxpy.cu
)
add_library(gravity::gravity ALIAS gravity)

target_include_directories(gravity PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(gravity PUBLIC cxx_std_17)

# The public header needs no CUDA headers, so the CUDA runtime is a private
# dependency. CMake still passes it on to consumers' link lines when the
# library is static.
target_link_libraries(gravity PRIVATE CUDA::cudart)
