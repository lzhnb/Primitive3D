# Add executable and library
add_library(Core SHARED)
add_library(Prim3D::Core ALIAS Core)
set_target_properties(Core PROPERTIES OUTPUT_NAME "Prim3D-Core")

target_include_directories(Core PRIVATE ${PYTHON_INCLUDE_DIRS})

target_link_libraries(
    Core PUBLIC
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY})

# Enumerate source files
file(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.c ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
file(GLOB INC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.h ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh)
target_sources(Core PUBLIC ${SRC_FILES} ${INC_FILES})
