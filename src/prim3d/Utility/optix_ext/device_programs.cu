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

    float3 ray_origin = params.ray_origins[idx.x];
    float3 ray_direction = params.ray_directions[idx.x];

    unsigned int p0, p1; // holder for the payload
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
        p0,
        p1);

    // Hit position
    const int32_t ray_id = idx.x;
    params.hits[ray_id].w = int_as_float(p1);

    // If a triangle was hit, p0 is its index, otherwise p0 is -1.
    // Write out the triangle's normal if it (abuse the direction buffer).
    if ((int)p0 == -1) {
        return;
    }
    const float3 n = params.triangles[p0].normal();
    params.hits[ray_id].x = n.x;
    params.hits[ray_id].y = n.y;
    params.hits[ray_id].z = n.z;
}

// miss program
extern "C" __global__ void __miss__ms() {
    optixSetPayload_0((uint32_t)-1);
    optixSetPayload_1(__float_as_int(optixGetRayTmax()));
}

// closest-hit program
extern "C" __global__ void __closesthit__ch() {
    optixSetPayload_0(optixGetPrimitiveIndex());
    optixSetPayload_1(__float_as_int(optixGetRayTmax()));
}

}  // namespace prim3d
