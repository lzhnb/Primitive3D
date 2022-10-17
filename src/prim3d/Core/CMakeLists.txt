# Enumerate source files
file(GLOB_RECURSE SRC_FILES *.cpp *.c *.cu)
file(GLOB_RECURSE INC_FILES *.h *.hpp *.cuh)

# Add executable and library
add_library(Core SHARED)
add_library(Prim3D::Core ALIAS Core)
set_target_properties(Core PROPERTIES OUTPUT_NAME "Prim3D-Core")

target_link_libraries(
    Core PUBLIC
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY})

target_include_directories(Core PRIVATE ${PYTHON_INCLUDE_DIRS})

target_sources(Core PUBLIC ${SRC_FILES} ${INC_FILES})
