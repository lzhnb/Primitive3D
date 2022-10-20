// Copyright 2022 Zhihao Liang
#pragma once
#include <Core/common.h>
#include <Core/vec_math.h>

namespace prim3d {
struct Triangle {
    PRIM_HOST_DEVICE float3 normal() const { return normalize(cross(b - a, c - a)); }

    // based on https://www.iquilezles.org/www/articles/intersectors/intersectors.htm
    PRIM_HOST_DEVICE float ray_intersect(const float3 &ro, const float3 &rd, float3 &n) const {
        float3 v1v0 = b - a;
        float3 v2v0 = c - a;
        float3 rov0 = ro - a;
        n           = cross(v1v0, v2v0);
        float3 q    = cross(rov0, rd);
        float d     = 1.0f / dot(rd, n);
        float u     = d * -dot(q, v2v0);
        float v     = d * dot(q, v1v0);
        float t     = d * -dot(n, rov0);
        if (u < 0.0f || u > 1.0f || v < 0.0f || (u + v) > 1.0f || t < 0.0f) {
            t = 1e38f;  // No intersection
        }
        return t;
    }

    PRIM_HOST_DEVICE float ray_intersect(const float3 &ro, const float3 &rd) const {
        float3 n;
        return ray_intersect(ro, rd, n);
    }

    PRIM_HOST_DEVICE float3 centroid() const { return (a + b + c) / 3.0f; }

    PRIM_HOST_DEVICE float centroid(int axis) const {
        if (axis == 0) {
            return (a.x + b.x + c.x) / 3;
        } else if (axis == 1) {
            return (a.y + b.y + c.y) / 3;
        } else if (axis == 2) {
            return (a.z + b.z + c.z) / 3;
        } else {
            return (a.x + b.x + c.x) / 3;
        }
    }

    PRIM_HOST_DEVICE void get_vertices(float3 v[3]) const {
        v[0] = a;
        v[1] = b;
        v[2] = c;
    }

    float3 a, b, c;
    int32_t idx;
};

/* TODO: complete the Mesh structure*/
struct TriangleMesh {
    Triangle *triangles;
    int32_t num_triangles;
};
}  // namespace prim3d
