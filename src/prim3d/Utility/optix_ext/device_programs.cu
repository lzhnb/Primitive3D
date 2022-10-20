#include <optix.h>
#include "launch_parameters.h"

namespace prim3d {

extern "C" {
__constant__ RayCast::Params params;
}

// ray generation program
extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    float3 ray_origin    = params.ray_origins[idx.x];
    float3 ray_direction = params.ray_directions[idx.x];

    unsigned int prim_idx, t, nx, ny, nz;  // holder for the payload
    optixTrace(
        params.handle,
        ray_origin,
        ray_direction,
        0.0f,                      // Min intersection distance
        1e16f,                     // Max intersection distance
        0.0f,                      // rayTime -- used for motion blur
        OptixVisibilityMask(255),  // Specify always visible
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0,  // SBT offset
        1,  // SBT stride
        0,  // missSBTIndex
        // payload
        prim_idx,
        t,
        nx,
        ny,
        nz);

    // Hit position
    const int32_t ray_id  = idx.x;
    params.hits[ray_id].w = int_as_float(t);

    // If a triangle was hit, prim_idx is its index, otherwise prim_idx is -1.
    // Write out the triangle's normal if it (abuse the direction buffer).
    if ((int32_t)prim_idx == -1) {
        return;
    }
    const float3 n        = params.triangles[prim_idx].normal();
    params.hits[ray_id].x = n.x;
    params.hits[ray_id].y = n.y;
    params.hits[ray_id].z = n.z;
}

// miss program
extern "C" __global__ void __miss__ms() {
    optixSetPayload_0((uint32_t)-1);
    optixSetPayload_1(float_as_int(optixGetRayTmax()));
    optixSetPayload_2(float_as_int(1.0f));
    optixSetPayload_3(float_as_int(0.0f));
    optixSetPayload_4(float_as_int(0.0f));
}

// closest-hit program
extern "C" __global__ void __closesthit__ch() {
    const int32_t prim_idx                = (int32_t)optixGetPrimitiveIndex();
    // const RayCast::HitGroupData* sbt_data = (RayCast::HitGroupData*)optixGetSbtDataPointer();
    // const Triangle* triangles             = (Triangle*)(sbt_data->data);
    // const Triangle hit_triangle           = triangles[prim_idx];
    // const float3 n                        = hit_triangle.normal();

    optixSetPayload_0(prim_idx);
    optixSetPayload_1(float_as_int(optixGetRayTmax()));
    optixSetPayload_2(float_as_int(1.0f));
    optixSetPayload_3(float_as_int(0.0f));
    optixSetPayload_4(float_as_int(0.0f));
}

}  // namespace prim3d
