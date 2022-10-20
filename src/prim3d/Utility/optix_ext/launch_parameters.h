// Copyright 2022 Zhihao Liang
#include <Geometry/triangle.h>
#include <optix.h>
#include <optix_stubs.h>
// NOTE: optix_stubs must be before!!!
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>

namespace prim3d {

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
        TriangleMesh data;
    };
};
}  // namespace prim3d
