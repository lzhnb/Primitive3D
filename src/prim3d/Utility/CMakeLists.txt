# Add executable and library
add_library(Utility SHARED)
add_library(Prim3D::Utility ALIAS Utility)
set_target_properties(Utility PROPERTIES OUTPUT_NAME "Prim3D-Utility")

target_include_directories(
    Utility PRIVATE
    ..
    ${PYTHON_INCLUDE_DIRS})

target_link_libraries(
    Utility PUBLIC
    Prim3D::Core
    ${optix_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY})

if(OptiX_FOUND OR OptiX_INCLUDE)
    add_subdirectory(optix_ext)
    # target_link_libraries(
    #     Utility PUBLIC
    #     Prim3D::Utility::OptixEXT)
    message("CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}/optix_ext")
    target_include_directories(Utility PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/optix_ext)
    # target_sources(Utility PUBLIC ${OPTIX_PTX_HEADER})
endif()

# Enumerate source files
file(GLOB SRC_FILES *.cpp *.c *.cu)
file(GLOB INC_FILES *.h *.hpp *.cuh)
target_sources(Utility PUBLIC ${SRC_FILES} ${INC_FILES})
