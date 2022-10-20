#pragma once
#include <torch/script.h>
#include <iostream>
#include <memory>

#include <Core/torch_utils.h>
#include "optix_ext/launch_parameters.h"
#include "optix_ext/utils.h"

using torch::Tensor;

namespace prim3d {
template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct RayCastingState {
    OptixDeviceContext context = 0;
    OptixTraversableHandle gas_handle = 0;  // Traversable handle for triangle AS
    CUdeviceptr gas_output_buffer = 0;    // Triangle AS memory

    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixModule ptx_module = 0;
    OptixPipeline pipeline = 0;

    // ProgramGroups
    OptixProgramGroup raygen_prog_group = 0;
    OptixProgramGroup miss_prog_group = 0;
    OptixProgramGroup hit_prog_group = 0;

    RayCast::Params params = {};
    OptixShaderBindingTable sbt = {};
};

// abstract class of RayCaster
class RayCaster {
public:
    RayCaster() {}
    virtual ~RayCaster() {}

    virtual void build_gas(Triangle* triangles, const int32_t num_triangles) = 0;
    virtual void build_pipeline() = 0;
    virtual void invoke(const RayCast::Params& params, const int32_t num_rays) = 0;
    virtual void cast(
        const Tensor& origins, const Tensor& directions, Tensor& depths, Tensor& normals) = 0;
};

// function to create an implementation of RayCaster
RayCaster* create_raycaster(const Tensor& vertices, const Tensor& triangles);
}  // namespace prim3d
