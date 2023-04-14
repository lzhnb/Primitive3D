// Copyright 2022 Zhihao Liang
#include <Core/vec_math.h>
#include <optix.h>
#include <optix_stubs.h>
// NOTE: optix_stubs must be before!!!
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>

#include <Eigen/Dense>

using Eigen::Vector3f;

namespace prim3d {

// NOTE: OptixTriangle just contain a,b,c (for the trianglearray)
struct OptixTriangle {
    SUTIL_HOSTDEVICE Vector3f normal() const {
        return (b - a).cross(c - a).normalized();
    }

    Vector3f a, b, c;
};

struct RayCast {
    struct Params {
        float *ray_origins;
        float *ray_directions;
        float *output_depths;
        float *output_normals;
        int32_t *output_primitive_ids;
        OptixTraversableHandle handle;
    };
    struct RayGenData {};
    struct MissData {};
    struct HitGroupData {
        OptixTriangle *data;
    };
};
} // namespace prim3d
