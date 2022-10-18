# Enumerate source files
file(GLOB_RECURSE SRC_FILES *.cpp *.c *.cu)
file(GLOB_RECURSE INC_FILES *.h *.hpp *.cuh)

# Add executable and library
add_library(Utility SHARED)
add_library(Prim3D::Utility ALIAS Utility)
set_target_properties(Utility PROPERTIES OUTPUT_NAME "Prim3D-Utility")

target_include_directories(
    Utility PRIVATE
    ..
    ${OptiX_INCLUDE}
    ${PYTHON_INCLUDE_DIRS})

target_link_libraries(
    Utility PUBLIC
    Prim3D::Core
    ${optix_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY})

target_sources(Utility PUBLIC ${SRC_FILES} ${INC_FILES})
