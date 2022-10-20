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

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Optix
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message("CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
find_package(Optix)

# set OptiX_INSTALL_DIR via your environment if it's not found automatically
if(OptiX_FOUND OR OptiX_INCLUDE)
	include_directories("${OptiX_INCLUDE}")
	message("successfully include OptiX")
	# Find CUDA
	find_package(CUDA REQUIRED)
	# compile nvidia support libraries
	add_definitions(-DENABLE_OPTIX)
else()
	message(WARNING
		"OptiX was not found."
	)
endif()

# torch extension
find_package(Torch REQUIRED)
find_package(PythonLibs REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

# Add library
add_subdirectory(src)
