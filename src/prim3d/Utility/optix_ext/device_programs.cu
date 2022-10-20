// Copyright 2022 Zhihao Liang
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

    const float3 ray_origin = make_float3(
        params.ray_origins[idx.x * 3 + 0],
        params.ray_origins[idx.x * 3 + 1],
        params.ray_origins[idx.x * 3 + 2]);
    const float3 ray_direction = make_float3(
        params.ray_directions[idx.x * 3 + 0],
        params.ray_directions[idx.x * 3 + 1],
        params.ray_directions[idx.x * 3 + 2]);

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
    const int32_t ray_id         = idx.x;
    params.output_depths[ray_id] = int_as_float(t);

    // If a triangle was hit, prim_idx is its index, otherwise prim_idx is -1.
    // Write out the triangle's normal if it (abuse the direction buffer).
    if ((int32_t)prim_idx == -1) { return; }
    params.output_primitive_ids[ray_id]   = (int32_t)prim_idx;
    params.output_normals[ray_id * 3 + 0] = int_as_float(nx);
    params.output_normals[ray_id * 3 + 1] = int_as_float(ny);
    params.output_normals[ray_id * 3 + 2] = int_as_float(nz);
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
    const RayCast::HitGroupData& sbt_data = *(RayCast::HitGroupData*)optixGetSbtDataPointer();
    const int32_t prim_idx                = (int32_t)optixGetPrimitiveIndex();
    const TriangleMesh mesh               = (TriangleMesh)(sbt_data.data);
    const Triangle hit_triangle           = mesh.triangles[prim_idx];
    const float3 n                        = hit_triangle.normal();

    optixSetPayload_0(prim_idx);
    optixSetPayload_1(float_as_int(optixGetRayTmax()));
    optixSetPayload_2(float_as_int(n.x));
    optixSetPayload_3(float_as_int(n.y));
    optixSetPayload_4(float_as_int(n.z));
}

}  // namespace prim3d
