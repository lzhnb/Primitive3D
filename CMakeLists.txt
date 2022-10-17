cmake_minimum_required(VERSION 3.4)

# minimum version for modern cmake
project(
    Prim3D
    DESCRIPTION "Primitive3D: Self-Pratice Library for 3D Data Processing." # Set PROJECT_DESCRIPTION
    LANGUAGES CXX CUDA
)

# Primitive3D build options
option(CUDA_ENABLED "Whether to enable CUDA, if available" ON)
option(BUILD_PYTHON_MODULE "Build the python module" ON)

add_definitions("-DCUDA_ENABLED")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# torch extension
find_package(Torch REQUIRED)
find_package(PythonLibs REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")


# Add library
add_subdirectory(src)
