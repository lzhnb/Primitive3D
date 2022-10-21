#include <Core/utils.h>
#include <assert.h>

#include <algorithm>
#include <stack>

#include "bounding_box.h"
#include "bvh.h"
#include "triangle.h"

namespace prim3d {
// kernels
constexpr float MAX_DIST    = 10.0f;
constexpr float MAX_DIST_SQ = MAX_DIST * MAX_DIST;

// declaration
__global__ void raytrace_kernel(
    uint32_t n_elements,
    const float* __restrict__ rays_o,
    const float* __restrict__ rays_d,
    float* __restrict__ depth,
    float* __restrict__ normals,
    int32_t* __restrict__ face_id,
    const TriangleBvhNode* __restrict__ nodes,
    const Triangle* __restrict__ triangles);

struct DistAndIdx {
    float dist;
    uint32_t idx;

    // Sort in descending order!
    PRIM_HOST_DEVICE bool operator<(const DistAndIdx& other) { return dist < other.dist; }
};

template <typename T>
PRIM_HOST_DEVICE void inline compare_and_swap(T& t1, T& t2) {
    if (t1 < t2) {
        T tmp{t1};
        t1 = t2;
        t2 = tmp;
    }
}

// Sorting networks from
// http://users.telenet.be/bertdobbelaere/SorterHunter/sorting_networks.html#N4L5D3
template <uint32_t N, typename T>
PRIM_HOST_DEVICE void sorting_network(T values[N]) {
    static_assert(N <= 8, "Sorting networks are only implemented up to N==8");
    if (N <= 1) {
        return;
    } else if (N == 2) {
        compare_and_swap(values[0], values[1]);
    } else if (N == 3) {
        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[1], values[2]);
    } else if (N == 4) {
        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[1], values[3]);
        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[1], values[2]);
    } else if (N == 5) {
        compare_and_swap(values[0], values[3]);
        compare_and_swap(values[1], values[4]);

        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[1], values[3]);

        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[4]);

        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);

        compare_and_swap(values[2], values[3]);
    } else if (N == 6) {
        compare_and_swap(values[0], values[5]);
        compare_and_swap(values[1], values[3]);
        compare_and_swap(values[2], values[4]);

        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);

        compare_and_swap(values[0], values[3]);
        compare_and_swap(values[2], values[5]);

        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[4], values[5]);

        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);
    } else if (N == 7) {
        compare_and_swap(values[0], values[6]);
        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[4], values[5]);

        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[1], values[4]);
        compare_and_swap(values[3], values[6]);

        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[5]);
        compare_and_swap(values[3], values[4]);

        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[4], values[6]);

        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[4], values[5]);

        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);
        compare_and_swap(values[5], values[6]);
    } else if (N == 8) {
        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[1], values[3]);
        compare_and_swap(values[4], values[6]);
        compare_and_swap(values[5], values[7]);

        compare_and_swap(values[0], values[4]);
        compare_and_swap(values[1], values[5]);
        compare_and_swap(values[2], values[6]);
        compare_and_swap(values[3], values[7]);

        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[4], values[5]);
        compare_and_swap(values[6], values[7]);

        compare_and_swap(values[2], values[4]);
        compare_and_swap(values[3], values[5]);

        compare_and_swap(values[1], values[4]);
        compare_and_swap(values[3], values[6]);

        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);
        compare_and_swap(values[5], values[6]);
    }
}

template <uint32_t BRANCHING_FACTOR>
class TriangleBvhWithBranchingFactor : public TriangleBvh {
public:
    PRIM_HOST_DEVICE static std::pair<int, float> ray_intersect(
        float3 ro,
        float3 rd,
        const TriangleBvhNode* __restrict__ bvhnodes,
        const Triangle* __restrict__ triangles) {
        FixedIntStack query_stack;
        query_stack.push(0);

        float mint       = MAX_DIST;
        int shortest_idx = -1;

        while (!query_stack.empty()) {
            int idx = query_stack.pop();

            const TriangleBvhNode& node = bvhnodes[idx];

            if (node.left_idx < 0) {
                int end = -node.right_idx - 1;
                for (int i = -node.left_idx - 1; i < end; ++i) {
                    float t = triangles[i].ray_intersect(ro, rd);
                    if (t < mint) {
                        mint         = t;
                        shortest_idx = i;
                    }
                }
            } else {
                DistAndIdx children[BRANCHING_FACTOR];

                uint32_t first_child = node.left_idx;

#pragma unroll
                for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
                    children[i] = {
                        bvhnodes[i + first_child].bb.ray_intersect(ro, rd).x, i + first_child};
                }

                sorting_network<BRANCHING_FACTOR>(children);

#pragma unroll
                for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
                    if (children[i].dist < mint) { query_stack.push(children[i].idx); }
                }
            }
        }

        return {shortest_idx, mint};
    }

    void ray_trace_gpu(
        uint32_t n_elements,
        const float* rays_o,
        const float* rays_d,
        float* depth,
        float* normals,
        int32_t* face_id,
        const Triangle* gpu_triangles,
        cudaStream_t stream) override {
        const int32_t blocks = n_blocks_linear(n_elements);
        raytrace_kernel<<<blocks, n_threads_linear>>>(
            n_elements, rays_o, rays_d, depth, normals, face_id, m_nodes_gpu, gpu_triangles);
    }

    void build(std::vector<Triangle>& triangles, uint32_t n_primitives_per_leaf) override {
        m_nodes.clear();

        // Root
        m_nodes.emplace_back();
        m_nodes.front().bb = BoundingBox(std::begin(triangles), std::end(triangles));

        struct BuildNode {
            int node_idx;
            std::vector<Triangle>::iterator begin;
            std::vector<Triangle>::iterator end;
        };

        std::stack<BuildNode> build_stack;
        build_stack.push({0, std::begin(triangles), std::end(triangles)});

        // cpu bvh construction
        while (!build_stack.empty()) {
            const BuildNode& curr = build_stack.top();
            size_t node_idx       = curr.node_idx;

            std::array<BuildNode, BRANCHING_FACTOR> children;
            children[0].begin = curr.begin;
            children[0].end   = curr.end;

            build_stack.pop();

            // Partition the triangles into the children
            int n_children = 1;
            while (n_children < BRANCHING_FACTOR) {
                for (int i = n_children - 1; i >= 0; --i) {
                    const BuildNode& child = children[i];

                    // Choose axis with maximum standard deviation
                    float3 mean = make_float3(0.0f);
                    for (auto it = child.begin; it != child.end; ++it) {
                        mean = mean + it->centroid();
                    }
                    mean /= (float)std::distance(child.begin, child.end);

                    float3 var = make_float3(0.0f);
                    for (auto it = child.begin; it != child.end; ++it) {
                        float3 diff = it->centroid() - mean;
                        var         = var + square_norm(diff);
                    }
                    var /= (float)std::distance(child.begin, child.end);

                    const int32_t axis = max_axis(var);

                    std::vector<Triangle>::iterator m = child.begin + std::distance(child.begin, child.end) / 2;
                    std::nth_element(
                        child.begin, m, child.end, [&](const Triangle& tri1, const Triangle& tri2) {
                            return tri1.centroid(axis) < tri2.centroid(axis);
                        });

                    children[i * 2].begin   = children[i].begin;
                    children[i * 2 + 1].end = children[i].end;
                    children[i * 2].end = children[i * 2 + 1].begin = m;
                }

                n_children *= 2;
            }

            // Create next build nodes
            m_nodes[node_idx].left_idx = (int)m_nodes.size();
            for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
                BuildNode& child = children[i];
                assert(child.begin != child.end);
                child.node_idx = (int)m_nodes.size();

                m_nodes.emplace_back();
                m_nodes.back().bb = BoundingBox(child.begin, child.end);

                if (std::distance(child.begin, child.end) <= n_primitives_per_leaf) {
                    m_nodes.back().left_idx =
                        -(int)std::distance(std::begin(triangles), child.begin) - 1;
                    m_nodes.back().right_idx =
                        -(int)std::distance(std::begin(triangles), child.end) - 1;
                } else {
                    build_stack.push(child);
                }
            }
            m_nodes[node_idx].right_idx = (int)m_nodes.size();
        }

        // Put the bvh on gpu
        CUDA_CHECK(cudaMalloc((void**)&m_nodes_gpu, sizeof(TriangleBvhNode) * m_nodes.size()));
        CUDA_CHECK(cudaMemcpy(
            m_nodes_gpu,
            &m_nodes[0],
            sizeof(TriangleBvhNode) * m_nodes.size(),
            cudaMemcpyHostToDevice));
    }

    TriangleBvhWithBranchingFactor() {}
};

using TriangleBvh4 = TriangleBvhWithBranchingFactor<4>;

std::unique_ptr<TriangleBvh> TriangleBvh::make() {
    return std::unique_ptr<TriangleBvh>(new TriangleBvh4());
}

__global__ void raytrace_kernel(
    uint32_t n_elements,
    const float* __restrict__ rays_o,
    const float* __restrict__ rays_d,
    float* __restrict__ depth,
    float* __restrict__ normals,
    int32_t* __restrict__ face_id,
    const TriangleBvhNode* __restrict__ nodes,
    const Triangle* __restrict__ triangles) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    float3 ro = make_float3(rays_o[i * 3 + 0], rays_o[i * 3 + 1], rays_o[i * 3 + 2]);
    float3 rd = make_float3(rays_d[i * 3 + 0], rays_d[i * 3 + 1], rays_d[i * 3 + 2]);

    auto p = TriangleBvh4::ray_intersect(ro, rd, nodes, triangles);

    // write depth
    depth[i] = p.second;

    // face normal is written to directions.
    if (p.first >= 0) {
        const float3 normal = triangles[p.first].normal();
        normals[i * 3 + 0]  = normal.x;
        normals[i * 3 + 1]  = normal.y;
        normals[i * 3 + 2]  = normal.z;
        face_id[i]          = triangles[p.first].idx;
    } else {
        normals[i * 3 + 0] = 0.0f;
        normals[i * 3 + 1] = 0.0f;
        normals[i * 3 + 2] = 0.0f;
        face_id[i]         = -1;
    }
}

}  // namespace prim3d
