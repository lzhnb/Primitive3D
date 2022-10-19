# include_directories(${OptiX_INCLUDE})
add_library(optix_program OBJECT
    device_programs.cu
)
target_include_directories(optix_program PRIVATE ${Optix_INCLUDE})

set_target_properties(optix_program PROPERTIES CUDA_PTX_COMPILATION ON)
set_target_properties(optix_program PROPERTIES CUDA_ARCHITECTURES OFF)
target_compile_options(optix_program PUBLIC "--expt-relaxed-constexpr")

# OptiX programs will be compiled as PTX and packaged
# as headers to be included from the binary dir.
include_directories(${CMAKE_CURRENT_BINARY_DIR})
set(OPTIX_PTX_HEADER ${CMAKE_CURRENT_BINARY_DIR}/optix_ptx.h)

find_program(bin_to_c NAMES bin2c PATHS ${CUDA_COMPILER_BIN})
if (NOT bin_to_c)
    message(FATAL_ERROR
        "bin2c not found:\n"
        "  CMAKE_CUDA_COMPILER='${CMAKE_CUDA_COMPILER}'\n"
        "  CUDA_COMPILER_BIN='${CUDA_COMPILER_BIN}'\n"
    )
endif()

add_custom_command(
    OUTPUT "${OPTIX_PTX_HEADER}"
    COMMAND ${CMAKE_COMMAND}
    "-DBIN_TO_C_COMMAND=${bin_to_c}"
    "-DOBJECTS=$<TARGET_OBJECTS:optix_program>"
    "-DOUTPUT=${OPTIX_PTX_HEADER}"
    -P ${PROJECT_SOURCE_DIR}/cmake/bin2c_wrapper.cmake
    VERBATIM
    DEPENDS optix_program
    COMMENT "Converting PTX files to a C header"
)

# message("optix_ptx.h path: ${OPTIX_PTX_HEADER}")
# add_library(
#     OptixEXT SHARED
#     ray_casting.cu
#     ${OPTIX_PTX_HEADER})
# add_library(Prim3D::Utility::OptixEXT ALIAS OptixEXT)
# set_target_properties(OptixEXT PROPERTIES OUTPUT_NAME "Prim3D-OptixEXT")
# target_include_directories(
#     OptixEXT PRIVATE
#     ${CMAKE_CURRENT_BINARY_DIR}
#     ${OptiX_INCLUDE})
# target_link_libraries(
#     OptixEXT PUBLIC
#     ${optix_LIBRARY}
#     ${CUDA_LIBRARIES}
#     ${CUDA_CUDA_LIBRARY})
