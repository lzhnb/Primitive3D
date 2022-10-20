# Add executable and library
add_library(Geometry SHARED)
add_library(Prim3D::Geometry ALIAS Geometry)
set_target_properties(Geometry PROPERTIES OUTPUT_NAME "Prim3D-Geometry")

target_include_directories(
    Geometry PRIVATE
    ..
    ${PYTHON_INCLUDE_DIRS})

target_link_libraries(
    Geometry PUBLIC
    Prim3D::Core
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY})

# Enumerate source files
file(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.c ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
file(GLOB INC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.h ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh)
target_sources(Geometry PUBLIC ${SRC_FILES} ${INC_FILES})
