// Copyright 2022 Zhihao Liang
#include <Core/vec_math.h>
#include <optix.h>
#include <optix_stubs.h>
// NOTE: optix_stubs must be before!!!
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>

namespace prim3d {

// NOTE: OptixTriangle just contain a,b,c (for the trianglearray)
struct OptixTriangle {
    SUTIL_HOSTDEVICE float3 normal() const { return normalize(cross(b - a, c - a)); }

    float3 a, b, c;
};

struct RayCast {
    struct Params {
        float* ray_origins;
        float* ray_directions;
        float* output_depths;
        float* output_normals;
        int32_t* output_primitive_ids;
        OptixTraversableHandle handle;
    };
    struct RayGenData {};
    struct MissData {};
    struct HitGroupData {
        OptixTriangle *data;
    };
};
}  // namespace prim3d
