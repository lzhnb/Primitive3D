// Copyright 2022 Zhihao Liang
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <Core/utils.h>

namespace py = pybind11;

// TODO: decouple the pybind11 module
namespace prim3d {
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("test", &test); }
}  // namespace prim3d
