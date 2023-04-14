// Copyright 2023 Zhihao Liang
#pragma once
#include <Core/common.h>
#include <Core/utils.h>
#include <Core/vec_math.h>
#include <torch/script.h>

#include <string>
#include <vector>

using std::vector;
using torch::Tensor;

namespace prim3d {
vector<Tensor> marching_tetrahedras(const Tensor &points, Tensor &tetrahedras, const Tensor &sdfs,
                                    const float thresh);
}

// clang-format off
// Triangle table for marching tetrahedras
static PRIM_DEVICE int8_t tetrahedra_triangle_table[16][6] = {
    {-1, -1, -1, -1, -1, -1},
    {1, 0, 2, -1, -1, -1},
    {4, 0, 3, -1, -1, -1},
    {1, 4, 2, 1, 3, 4},
    {3, 1, 5, -1, -1, -1},
    {2, 3, 0, 2, 5, 3},
    {1, 4, 0, 1, 5, 4},
    {4, 2, 5, -1, -1, -1},
    {4, 5, 2, -1, -1, -1},
    {4, 1, 0, 4, 5, 1},
    {3, 2, 0, 3, 5, 2},
    {1, 3, 5, -1, -1, -1},
    {4, 1, 2, 4, 3, 1},
    {3, 0, 4, -1, -1, -1},
    {2, 0, 1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1}
};

static PRIM_DEVICE int8_t tetrahedra_num_triangles_table[16] = {
    0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0
};

static PRIM_DEVICE int8_t tetrahedra_edge_to_vertices_table[6][2] = {
    {0, 1},
    {0, 2},
    {0, 3},
    {1, 2},
    {1, 3},
    {2, 3}
};

// clang-format on
