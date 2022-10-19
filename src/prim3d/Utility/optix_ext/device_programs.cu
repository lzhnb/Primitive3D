#include <optix.h>
#include "launch_parameters.h"

namespace prim3d {

extern "C" {
__constant__ Params params;
}

// ray generation program
extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    float3 ray_origin = params.ray_origins[idx.x];
    float3 ray_direction = params.ray_directions[idx.x];

    unsigned int t, nx, ny, nz;  // holder for the payload
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
        t,
        nx,
        ny,
        nz);

    // Hit position
    params.ray_origins[idx.x].x = int_as_float(t);

    params.ray_directions[idx.x] =
        make_float3(int_as_float(nx), int_as_float(ny), int_as_float(nz));
}

// miss program
extern "C" __global__ void __miss__ms() {
    optixSetPayload_0(float_as_int(-1.0f));
    optixSetPayload_1(float_as_int(1.0f));
    optixSetPayload_2(float_as_int(0.0f));
    optixSetPayload_3(float_as_int(0.0f));
}

// closest-hit program
extern "C" __global__ void __closesthit__ch() {
    const unsigned int t = optixGetRayTmax();

    const Triangle* hit_triangle = (Triangle*)optixGetSbtDataPointer();
    const float3 normal =
        normalize(cross(hit_triangle->b - hit_triangle->a, hit_triangle->c - hit_triangle->a));

    optixSetPayload_0(float_as_int(t));
    optixSetPayload_1(float_as_int(normal.x));
    optixSetPayload_2(float_as_int(normal.y));
    optixSetPayload_3(float_as_int(normal.z));
}

}  // namespace prim3d
