// Copyright 2022 Zhihao Liang
#pragma once
#include <Core/common.h>
#include <Core/vec_math.h>

#include <Eigen/Dense>
#include <limits>

using Eigen::Vector3f;

namespace prim3d {
struct Triangle {
    PRIM_HOST_DEVICE Vector3f normal() const { return (b - a).cross(c - a).normalized(); }

    // based on https://www.iquilezles.org/www/articles/intersectors/intersectors.htm
    PRIM_HOST_DEVICE float ray_intersect(
        const Vector3f &ro, const Vector3f &rd, Vector3f &n) const {
        Eigen::Vector3f v1v0 = b - a;
        Eigen::Vector3f v2v0 = c - a;
        Eigen::Vector3f rov0 = ro - a;
        n                    = v1v0.cross(v2v0);
        Eigen::Vector3f q    = rov0.cross(rd);
        float d              = 1.0f / rd.dot(n);
        float u              = d * -q.dot(v2v0);
        float v              = d * q.dot(v1v0);
        float t              = d * -n.dot(rov0);
        if (u < 0.0f || u > 1.0f || v < 0.0f || (u + v) > 1.0f || t < 0.0f) {
            t = std::numeric_limits<float>::max();  // No intersection
        }
        return t;
    }

    PRIM_HOST_DEVICE float ray_intersect(const Vector3f &ro, const Vector3f &rd) const {
        Vector3f n;
        return ray_intersect(ro, rd, n);
    }

    PRIM_HOST_DEVICE Vector3f centroid() const { return (a + b + c) / 3.0f; }

    PRIM_HOST_DEVICE float centroid(int axis) const {
		return (a[axis] + b[axis] + c[axis]) / 3;
	}

    PRIM_HOST_DEVICE void get_vertices(Vector3f v[3]) const {
        v[0] = a;
        v[1] = b;
        v[2] = c;
    }

    Vector3f a, b, c;
    int32_t idx;
};

/* TODO: complete the Mesh structure*/
struct TriangleMesh {
    Triangle *triangles;
    int32_t num_triangles;
};
}  // namespace prim3d
