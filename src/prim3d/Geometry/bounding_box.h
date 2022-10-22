// Copyright 2022 Zhihao Liang
#pragma once
#include <Core/common.h>
#include <Core/vec_math.h>

#include <Eigen/Dense>
#include <limits>
#include <vector>

#include "triangle.h"

using Eigen::Vector3f;

namespace prim3d {
template <int N_POINTS>
PRIM_HOST_DEVICE inline void project(
    Vector3f points[N_POINTS], const Vector3f& axis, float& min, float& max) {
    min = std::numeric_limits<float>::infinity();
    max = -std::numeric_limits<float>::infinity();

#pragma unroll
    for (uint32_t i = 0; i < N_POINTS; ++i) {
        float val = axis.dot(points[i]);

        if (val < min) { min = val; }

        if (val > max) { max = val; }
    }
}

struct BoundingBox {
    PRIM_HOST_DEVICE BoundingBox() {}

    PRIM_HOST_DEVICE BoundingBox(const Vector3f& a, const Vector3f& b) : min{a}, max{b} {}

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
        min = min.cwiseMin(other.min);
        max = max.cwiseMax(other.max);
    }

    PRIM_HOST_DEVICE void enlarge(const Triangle& tri) {
        enlarge(tri.a);
        enlarge(tri.b);
        enlarge(tri.c);
    }

    PRIM_HOST_DEVICE void enlarge(const Vector3f& point) {
        min = min.cwiseMin(point);
        max = max.cwiseMax(point);
    }

    PRIM_HOST_DEVICE void inflate(float amount) {
        min -= Vector3f::Constant(amount);
        max += Vector3f::Constant(amount);
    }

    PRIM_HOST_DEVICE Vector3f diag() const { return max - min; }

    PRIM_HOST_DEVICE Vector3f relative_pos(const Vector3f& pos) const {
        return (pos - min).cwiseQuotient(diag());
    }

    PRIM_HOST_DEVICE Vector3f center() const { return (max + min) * 0.5f; }

    PRIM_HOST_DEVICE BoundingBox intersection(const BoundingBox& other) const {
        BoundingBox result = *this;
        result.min         = result.min.cwiseMax(other.min);
        result.max         = result.max.cwiseMin(other.max);
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
        Vector3f box_normals[3] = {
            Vector3f{1.0f, 0.0f, 0.0f},
            Vector3f{0.0f, 1.0f, 0.0f},
            Vector3f{0.0f, 0.0f, 1.0f},
        };

        Vector3f triangle_normal = triangle.normal();
        Vector3f triangle_verts[3];
        triangle.get_vertices(triangle_verts);

        for (int i = 0; i < 3; i++) {
            project<3>(triangle_verts, box_normals[i], triangle_min, triangle_max);
            if (triangle_max < min[i] || triangle_min > max[i]) {
                return false;  // No intersection possible.
            }
        }

        Vector3f verts[8];
        get_vertices(verts);

        // Test the triangle normal
        float triangle_offset = triangle_normal.dot(triangle.a);
        project<8>(verts, triangle_normal, box_min, box_max);
        if (box_max < triangle_offset || box_min > triangle_offset) {
            return false;  // No intersection possible.
        }

        // Test the nine edge cross-products
        Vector3f edges[3] = {
            triangle.a - triangle.b,
            triangle.a - triangle.c,
            triangle.b - triangle.c,
        };

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                // The box normals are the same as it's edge tangents
                Vector3f axis = edges[i].cross(box_normals[j]);
                project<8>(verts, axis, box_min, box_max);
                project<3>(triangle_verts, axis, triangle_min, triangle_max);
                if (box_max < triangle_min || box_min > triangle_max)
                    return false;  // No intersection possible
            }
        }

        // No separating axis found.
        return true;
    }

    PRIM_HOST_DEVICE Eigen::Vector2f ray_intersect(const Vector3f& pos, const Vector3f& dir) const {
        float tmin = (min.x() - pos.x()) / dir.x();
        float tmax = (max.x() - pos.x()) / dir.x();

        if (tmin > tmax) { host_device_swap(tmin, tmax); }

        float tymin = (min.y() - pos.y()) / dir.y();
        float tymax = (max.y() - pos.y()) / dir.y();

        if (tymin > tymax) { host_device_swap(tymin, tymax); }

        if (tmin > tymax || tymin > tmax) {
            return {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
        }

        if (tymin > tmin) { tmin = tymin; }

        if (tymax < tmax) { tmax = tymax; }

        float tzmin = (min.z() - pos.z()) / dir.z();
        float tzmax = (max.z() - pos.z()) / dir.z();

        if (tzmin > tzmax) { host_device_swap(tzmin, tzmax); }

        if (tmin > tzmax || tzmin > tmax) {
            return {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
        }

        if (tzmin > tmin) { tmin = tzmin; }

        if (tzmax < tmax) { tmax = tzmax; }

        return {tmin, tmax};
    }

    PRIM_HOST_DEVICE bool is_empty() const {
        return (max.x() < min.x() || max.y() < min.y() || max.z() < min.z());
    }

    PRIM_HOST_DEVICE bool contains(const Vector3f& p) const {
        return p.x() >= min.x() && p.x() <= max.x() &&  //
               p.y() >= min.y() && p.y() <= max.y() &&  //
               p.z() >= min.z() && p.z() <= max.z();
    }

    /// Calculate the squared point-AABB distance
    PRIM_HOST_DEVICE float distance(const Vector3f& p) const { return sqrt(distance_sq(p)); }

    PRIM_HOST_DEVICE float distance_sq(const Vector3f& p) const {
        return (min - p).cwiseMax(p - max).cwiseMax(0.0f).squaredNorm();
    }

    PRIM_HOST_DEVICE float signed_distance(const Vector3f& p) const {
        const Vector3f q = (p - min).cwiseAbs() - diag();
		return q.cwiseMax(0.0f).norm() + std::min(std::max(q.x(), std::max(q.y(), q.z())), 0.0f);
    }

    PRIM_HOST_DEVICE void get_vertices(Vector3f v[8]) const {
		v[0] = {min.x(), min.y(), min.z()};
		v[1] = {min.x(), min.y(), max.z()};
		v[2] = {min.x(), max.y(), min.z()};
		v[3] = {min.x(), max.y(), max.z()};
		v[4] = {max.x(), min.y(), min.z()};
		v[5] = {max.x(), min.y(), max.z()};
		v[6] = {max.x(), max.y(), min.z()};
		v[7] = {max.x(), max.y(), max.z()};
	}

    Vector3f min = Vector3f::Constant(std::numeric_limits<float>::infinity());
    Vector3f max = Vector3f::Constant(-std::numeric_limits<float>::infinity());
};
}  // namespace prim3d
