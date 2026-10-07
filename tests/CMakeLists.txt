# One standalone executable, registered as one CTest test, per test_*.cpp /
# test_*.cu file in this directory. New files are picked up on the next build.
file(GLOB test_sources CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cu
)

foreach(source IN LISTS test_sources)
    get_filename_component(name ${source} NAME_WE)
    add_executable(${name} ${source})
    # CUDA::cudart lets tests call the CUDA runtime directly, even from .cpp files.
    target_link_libraries(${name} PRIVATE gravity::gravity CUDA::cudart)
    add_test(NAME ${name} COMMAND ${name})
endforeach()
