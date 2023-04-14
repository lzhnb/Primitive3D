// Copyright 2023 Zhihao Liang
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "marching_tetrahedras.h"

__global__ void correct_tetrahedras(const float *__restrict__ points_ptr,
                                    int32_t *__restrict__ tetrahedras_ptr, // need to be corrected
                                    const int32_t num_tetrahedras) {
    /* confirm the structure of tetrahedras */
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tetrahedras)
        return;
    int32_t *tetra_ptr = tetrahedras_ptr + idx * 4;
    const int32_t idx0 = tetra_ptr[0], idx1 = tetra_ptr[1], idx2 = tetra_ptr[2],
                  idx3 = tetra_ptr[3];

    const float3 p0 = make_float3(points_ptr[idx0 * 3 + 0], points_ptr[idx0 * 3 + 1],
                                  points_ptr[idx0 * 3 + 2]),
                 p1 = make_float3(points_ptr[idx1 * 3 + 0], points_ptr[idx1 * 3 + 1],
                                  points_ptr[idx1 * 3 + 2]),
                 p2 = make_float3(points_ptr[idx2 * 3 + 0], points_ptr[idx2 * 3 + 1],
                                  points_ptr[idx2 * 3 + 2]),
                 p3 = make_float3(points_ptr[idx3 * 3 + 0], points_ptr[idx3 * 3 + 1],
                                  points_ptr[idx3 * 3 + 2]);

    const float3 v01 = p1 - p0, v12 = p2 - p1, v03 = p3 - p0;
    const float3 cross012 = cross(v01, v12);
    const float inner_product = dot(cross012, v03);

    if (inner_product >= 0) { // NOTE: ignore the 0 case
        return;
    } else {
        const int32_t tmp = tetra_ptr[0];
        tetra_ptr[0] = tetra_ptr[1];
        tetra_ptr[1] = tmp;
    }
}

__global__ void tetrahedra_count_vertices_faces_kernel(const int32_t *__restrict__ tetrahedras_ptr,
                                                       const float *__restrict__ sdfs_ptr,
                                                       const float thresh,
                                                       const int32_t num_tetrahedras,
                                                       // output
                                                       int32_t *__restrict__ face_counters_ptr) {
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tetrahedras)
        return;

    // query the num_triangle_table
    const int32_t *tetra_ptr = tetrahedras_ptr + idx * 4;
    int32_t num_idx = 0;
    int32_t scale = 1;
#pragma unroll 4
    for (int i = 0; i < 4; ++i) {
        const int32_t tet = tetra_ptr[i];
        const float sdf = sdfs_ptr[tet];
        num_idx += sdf > 0 ? scale : 0;
        scale *= 2;
    }
    face_counters_ptr[idx] = tetrahedra_num_triangles_table[num_idx];
}

__global__ void tetrahedra_gen_vertices_faces_kernel(
    const float *__restrict__ points_ptr, const int32_t *__restrict__ tetrahedras_ptr,
    const float *__restrict__ sdfs_ptr, const int32_t *__restrict__ bias_ptr,
    const int32_t *__restrict__ face_counters_ptr, const float thresh,
    const int32_t num_tetrahedras,
    // output
    float *__restrict__ vertices_ptr, int32_t *__restrict__ faces_ptr) {
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tetrahedras)
        return;

    // query the num_triangle_table
    const int32_t *tetra_ptr = tetrahedras_ptr + idx * 4;
    const int32_t bias = bias_ptr[idx];
    const int32_t num_faces = face_counters_ptr[idx];
    if (num_faces == 0)
        return;

    // interpolate the vertices
    int32_t num_idx = 0;
    int32_t scale = 1;
#pragma unroll 4
    for (int i = 0; i < 4; ++i) {
        const int32_t tet = tetra_ptr[i];
        const float sdf = sdfs_ptr[tet];
        num_idx += sdf > 0 ? scale : 0;
        scale *= 2;
    }
    // get the triangle and get vertices
    for (int face_id = 0; face_id < num_faces; ++face_id) {
        float vertices[4][3];
        float sdfs[4];

        // get the vertces and sdf values for the tetrahedra
#pragma unroll 4
        for (int i = 0; i < 4; ++i) {
            const int32_t tet = tetra_ptr[i];
            vertices[i][0] = points_ptr[tet * 3 + 0];
            vertices[i][1] = points_ptr[tet * 3 + 1];
            vertices[i][2] = points_ptr[tet * 3 + 2];
            sdfs[i] = sdfs_ptr[tet];
        }

        const int8_t edges[3] = {tetrahedra_triangle_table[num_idx][face_id * 3 + 0],
                                 tetrahedra_triangle_table[num_idx][face_id * 3 + 1],
                                 tetrahedra_triangle_table[num_idx][face_id * 3 + 2]};

        // process each vertices in this triangle
#pragma unroll 3
        for (int v_idx = 0; v_idx < 3; ++v_idx) {
            const int8_t edge = edges[v_idx];
            const int8_t *point_ids = tetrahedra_edge_to_vertices_table[edge];
            const float *vert1 = vertices[point_ids[0]];
            const float *vert2 = vertices[point_ids[1]];
            const float sdf1 = abs(sdfs[point_ids[0]]);
            const float sdf2 = abs(sdfs[point_ids[1]]);
            // interpolate
            const float denominator = sdf1 + sdf2;
            vertices_ptr[((bias + face_id) * 3 + v_idx) * 3 + 0] =
                sdf1 / denominator * vert1[0] + sdf2 / denominator * vert2[0];
            vertices_ptr[((bias + face_id) * 3 + v_idx) * 3 + 1] =
                sdf1 / denominator * vert1[1] + sdf2 / denominator * vert2[1];
            vertices_ptr[((bias + face_id) * 3 + v_idx) * 3 + 2] =
                sdf1 / denominator * vert1[2] + sdf2 / denominator * vert2[2];
        }
        // generate faces
        faces_ptr[(bias + face_id) * 3 + 0] = (bias + face_id) * 3 + 0;
        faces_ptr[(bias + face_id) * 3 + 1] = (bias + face_id) * 3 + 1;
        faces_ptr[(bias + face_id) * 3 + 2] = (bias + face_id) * 3 + 2;
    }
}

namespace prim3d {
vector<Tensor> marching_tetrahedras(const Tensor &points, Tensor &tetrahedras, const Tensor &sdfs,
                                    const float thresh) {
    // check
    CHECK_INPUT(points);
    CHECK_INPUT(tetrahedras);
    CHECK_INPUT(sdfs);
    TORCH_CHECK(points.ndimension() == 2)
    TORCH_CHECK(tetrahedras.ndimension() == 2)
    TORCH_CHECK(sdfs.ndimension() == 1)

    const int32_t num_tetrahedras = tetrahedras.size(0);
    const int32_t threads = 256;
    const int32_t blocks = (num_tetrahedras - 1) / threads + 1;
    correct_tetrahedras<<<blocks, threads>>>(points.data_ptr<float>(),
                                             tetrahedras.data_ptr<int32_t>(), num_tetrahedras);

    const torch::Device curr_device = points.device();
    Tensor face_counters = torch::zeros(
        {num_tetrahedras}, torch::TensorOptions().dtype(torch::kInt).device(curr_device));

    // count only
    tetrahedra_count_vertices_faces_kernel<<<blocks, threads>>>(
        tetrahedras.data_ptr<int32_t>(), sdfs.data_ptr<float>(), thresh, num_tetrahedras,
        // output
        face_counters.data_ptr<int32_t>());

    // init the essential tensor(memory space)
    const int32_t num_faces = face_counters.sum().item<int32_t>();
    const int32_t num_vertices = num_faces * 3;
    Tensor vertices = torch::zeros({num_vertices, 3},
                                   torch::TensorOptions().dtype(torch::kFloat).device(curr_device));
    Tensor faces =
        torch::zeros({num_faces, 3}, torch::TensorOptions().dtype(torch::kInt).device(curr_device));

    Tensor bias = face_counters.clone();
    thrust::exclusive_scan(thrust::device, bias.data_ptr<int32_t>(),
                           bias.data_ptr<int32_t>() + num_tetrahedras, bias.data_ptr<int32_t>(), 0);
    tetrahedra_gen_vertices_faces_kernel<<<blocks, threads>>>(
        points.data_ptr<float>(), tetrahedras.data_ptr<int32_t>(), sdfs.data_ptr<float>(),
        bias.data_ptr<int32_t>(), face_counters.data_ptr<int32_t>(), thresh, num_tetrahedras,
        // output
        vertices.data_ptr<float>(), faces.data_ptr<int32_t>());

    vector<Tensor> results(2);
    results[0] = vertices;
    results[1] = faces;

    return results;
}

} // namespace prim3d
