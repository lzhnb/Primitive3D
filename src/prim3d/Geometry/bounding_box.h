// Copyright 2022 Zhihao Liang
#pragma once
#include <Core/common.h>
#include <Core/vec_math.h>

#include <limits>
#include <vector>

#include "triangle.h"

namespace prim3d {
template <int N_POINTS>
PRIM_HOST_DEVICE inline void project(
    float3 points[N_POINTS], const float3& axis, float& min, float& max) {
    min = std::numeric_limits<float>::infinity();
    max = -std::numeric_limits<float>::infinity();

#pragma unroll
    for (uint32_t i = 0; i < N_POINTS; ++i) {
        float val = dot(axis, points[i]);

        if (val < min) { min = val; }

        if (val > max) { max = val; }
    }
}

struct BoundingBox {
    float3 min = make_float3(std::numeric_limits<float>::infinity());
    float3 max = make_float3(-std::numeric_limits<float>::infinity());

    PRIM_HOST_DEVICE BoundingBox() {}

    PRIM_HOST_DEVICE BoundingBox(const float3& a, const float3& b) : min{a}, max{b} {}

    PRIM_HOST_DEVICE explicit BoundingBox(const Triangle& tri) {
        min = max = tri.a;
        enlarge(tri.b);
        enlarge(tri.c);
    }

    BoundingBox(std::vector<Triangle>::iterator begin, std::vector<Triangle>::iterator end) {
        min = max = begin->a;
        for (auto it = begin; it != end; ++it) { enlarge(*it); }
    }

    PRIM_HOST_DEVICE void enlarge(const BoundingBox& other) {
        min = fminf(min, other.min);
        max = fmaxf(max, other.max);
    }

    PRIM_HOST_DEVICE void enlarge(const Triangle& tri) {
        enlarge(tri.a);
        enlarge(tri.b);
        enlarge(tri.c);
    }

    PRIM_HOST_DEVICE void enlarge(const float3& point) {
        min = fminf(min, point);
        max = fmaxf(max, point);
    }

    PRIM_HOST_DEVICE void inflate(float amount) {
        min = min - amount;
        max = max + amount;
    }

    PRIM_HOST_DEVICE float3 diag() const { return max - min; }

    PRIM_HOST_DEVICE float3 relative_pos(const float3& pos) const { return (pos - min) / diag(); }

    PRIM_HOST_DEVICE float3 center() const { return (max + min) * 0.5f; }

    PRIM_HOST_DEVICE BoundingBox intersection(const BoundingBox& other) const {
        BoundingBox result = *this;
        result.min         = fmaxf(result.min, other.min);
        result.max         = fminf(result.max, other.max);
        return result;
    }

    PRIM_HOST_DEVICE bool intersects(const BoundingBox& other) const {
        return !intersection(other).is_empty();
    }

    // Based on the separating axis theorem
    // (https://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox_tam.pdf)
    // Code adapted from a C# implementation at stack overflow
    // https://stackoverflow.com/a/17503268
    PRIM_HOST_DEVICE bool intersects(const Triangle& triangle) const {
        float triangle_min, triangle_max;
        float box_min, box_max;

        // Test the box normals (x-, y- and z-axes)
        float3 box_normals[3] = {
            make_float3(1.0f, 0.0f, 0.0f),
            make_float3(0.0f, 1.0f, 0.0f),
            make_float3(0.0f, 0.0f, 1.0f),
        };

        float3 triangle_normal = triangle.normal();
        float3 triangle_verts[3];
        triangle.get_vertices(triangle_verts);

        for (int i = 0; i < 3; i++) {
            project<3>(triangle_verts, box_normals[i], triangle_min, triangle_max);
            switch (i) {
                case 0:
                    if (triangle_max < min.x || triangle_min > max.x) {
                        return false;  // No intersection possible.
                    }
                    break;
                case 1:
                    if (triangle_max < min.y || triangle_min > max.y) {
                        return false;  // No intersection possible.
                    }
                    break;
                case 2:
                    if (triangle_max < min.z || triangle_min > max.z) {
                        return false;  // No intersection possible.
                    }
                    break;

                default:
                    break;
            }
            // if (triangle_max < min[i] || triangle_min > max[i]) {
            //     return false;  // No intersection possible.
            // }
        }

        float3 verts[8];
        get_vertices(verts);

        // Test the triangle normal
        float triangle_offset = dot(triangle_normal, triangle.a);
        project<8>(verts, triangle_normal, box_min, box_max);
        if (box_max < triangle_offset || box_min > triangle_offset) {
            return false;  // No intersection possible.
        }

        // Test the nine edge cross-products
        float3 edges[3] = {
            triangle.a - triangle.b,
            triangle.a - triangle.c,
            triangle.b - triangle.c,
        };

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                // The box normals are the same as it's edge tangents
                float3 axis = cross(edges[i], box_normals[j]);
                project<8>(verts, axis, box_min, box_max);
                project<3>(triangle_verts, axis, triangle_min, triangle_max);
                if (box_max < triangle_min || box_min > triangle_max)
                    return false;  // No intersection possible
            }
        }

        // No separating axis found.
        return true;
    }

    PRIM_HOST_DEVICE float2 ray_intersect(float3 pos, float3 dir) const {
        float tmin = (min.x - pos.x) / dir.x;
        float tmax = (max.x - pos.x) / dir.x;

        if (tmin > tmax) { host_device_swap(tmin, tmax); }

        float tymin = (min.y - pos.y) / dir.y;
        float tymax = (max.y - pos.y) / dir.y;

        if (tymin > tymax) { host_device_swap(tymin, tymax); }

        if (tmin > tymax || tymin > tmax) {
            return {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
        }

        if (tymin > tmin) { tmin = tymin; }

        if (tymax < tmax) { tmax = tymax; }

        float tzmin = (min.z - pos.z) / dir.z;
        float tzmax = (max.z - pos.z) / dir.z;

        if (tzmin > tzmax) { host_device_swap(tzmin, tzmax); }

        if (tmin > tzmax || tzmin > tmax) {
            return {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
        }

        if (tzmin > tmin) { tmin = tzmin; }

        if (tzmax < tmax) { tmax = tzmax; }

        return make_float2(tmin, tmax);
    }

    PRIM_HOST_DEVICE bool is_empty() const {
        return (max.x < min.x || max.y < min.y || max.z < min.z);
    }

    PRIM_HOST_DEVICE bool contains(const float3& p) const {
        return p.x >= min.x && p.x <= max.x && p.y >= min.y && p.y <= max.y && p.z >= min.z &&
               p.z <= max.z;
    }

    /// Calculate the squared point-AABB distance
    PRIM_HOST_DEVICE float distance(const float3& p) const { return sqrt(distance_sq(p)); }

    PRIM_HOST_DEVICE float distance_sq(const float3& p) const {
        const float3 zeros = make_float3(0.0f);
        const float3 diag  = fmaxf(fmaxf(min - p, p - max), zeros);
        return square_norm(diag);
        // return (min - p).cwiseMax(p - max).cwiseMax(0.0f).squaredNorm();
    }

    PRIM_HOST_DEVICE float signed_distance(const float3& p) const {
        const float3 q         = abs(p - min) - diag();
        const float3 zeros     = make_float3(0.0f);
        const float3 q_no_zero = fmaxf(q, zeros);

        return square_norm(q_no_zero) + std::min(std::max(q.x, std::max(q.y, q.z)), 0.0f);
    }

    PRIM_HOST_DEVICE void get_vertices(float3 v[8]) const {
        v[0] = {min.x, min.y, min.z};
        v[1] = {min.x, min.y, max.z};
        v[2] = {min.x, max.y, min.z};
        v[3] = {min.x, max.y, max.z};
        v[4] = {max.x, min.y, min.z};
        v[5] = {max.x, min.y, max.z};
        v[6] = {max.x, max.y, min.z};
        v[7] = {max.x, max.y, max.z};
    }
};
}  // namespace prim3d
