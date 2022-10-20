// Copyright 2022 Zhihao Liang
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <Core/utils.h>
#include <Utility/ray_cast_optix.h>

namespace py = pybind11;

// TODO: decouple the pybind11 module
namespace prim3d {
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test", &test);

#ifdef ENABLE_OPTIX
    py::class_<RayCaster>(m, "RayCaster").def("cast", &RayCaster::cast);
    m.def("create_raycaster", &create_raycaster);
#endif
}
}  // namespace prim3d
