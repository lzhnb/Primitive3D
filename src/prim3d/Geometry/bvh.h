// Copyright 2022 Zhihao Liang
#pragma once
#include <Core/common.h>
#include <Core/vec_math.h>

#include <memory>
#include <vector>

#include "bounding_box.h"

namespace prim3d {
struct TriangleBvhNode {
    BoundingBox bb;
    int left_idx; // negative values indicate leaves
    int right_idx;
};

template <typename T, int MAX_SIZE = 32>
class FixedStack {
  public:
    PRIM_HOST_DEVICE void push(T val) {
        if (m_count >= MAX_SIZE - 1) {
            printf("WARNING TOO BIG\n");
        }
        m_elems[m_count++] = val;
    }

    PRIM_HOST_DEVICE T pop() {
        return m_elems[--m_count];
    }

    PRIM_HOST_DEVICE bool empty() const {
        return m_count <= 0;
    }

  private:
    T m_elems[MAX_SIZE];
    int m_count = 0;
};

using FixedIntStack = FixedStack<int>;

class TriangleBvh {
  protected:
    std::vector<TriangleBvhNode> m_nodes;
    TriangleBvhNode *m_nodes_gpu;
    TriangleBvh(){};

  public:
    virtual void build(std::vector<Triangle> &triangles, uint32_t n_primitives_per_leaf) = 0;
    virtual void ray_trace_gpu(uint32_t n_elements, const float *rays_o, const float *rays_d,
                               float *depth, float *normals, int32_t *face_id,
                               const Triangle *gpu_triangles, cudaStream_t stream) = 0;

    static std::unique_ptr<TriangleBvh> make();
};
} // namespace prim3d
