// Copyright 2022 Zhihao Liang
#include <Core/utils.h>
#include <Utility/ray_cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

// TODO: decouple the pybind11 module
namespace prim3d {
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Primitive3D: Self-Pratice Library for pytorch-based 3D Data Processing.";

#ifdef ENABLE_OPTIX
    m.attr("enable_optix") = true;
#else
    m.attr("enable_optix") = false;
#endif

    m.def("test", &test);

    py::class_<RayCaster>(m, "RayCaster").def("invoke", &RayCaster::invoke);
    m.def("create_raycaster", &create_raycaster);
}
}  // namespace prim3d
