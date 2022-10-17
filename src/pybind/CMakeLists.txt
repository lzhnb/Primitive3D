add_library(libPrim3D SHARED)

target_sources(libPrim3D PRIVATE bindings.cpp)

target_link_libraries(
    libPrim3D PRIVATE
    Prim3D::Core
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY})

set_target_properties(libPrim3D PROPERTIES POSITION_INDEPENDENT_CODE ON) # -fPIC
set_target_properties(libPrim3D PROPERTIES PREFIX "") # not "lib" prefix
target_compile_definitions(libPrim3D PUBLIC TORCH_EXTENSION_NAME=libPrim3D)
target_compile_definitions(libPrim3D PUBLIC TORCH_API_INCLUDE_EXTENSION_H)
target_compile_definitions(libPrim3D PUBLIC ${TORCH_CXX_FLAGS})

target_include_directories(libPrim3D SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR}/src/prim3d
    ${PYTHON_INCLUDE_DIRS})