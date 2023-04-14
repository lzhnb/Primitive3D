// Copyright 2022 Zhihao Liang
#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <Core/common.h>
#include <Core/utils.h>
#include <Geometry/triangle.h>
#include <torch/script.h>

#include <Eigen/Dense>
#include <iostream>
#include <memory>

#ifdef ENABLE_OPTIX
#include "optix_ext/launch_parameters.h"
#else
#include <Geometry/bvh.h>
#endif

using Eigen::Vector3f;
using torch::Tensor;

namespace prim3d {

#ifdef ENABLE_OPTIX
template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayCast::RayGenData> RayGenSbtRecord;
typedef SbtRecord<RayCast::MissData> MissSbtRecord;
typedef SbtRecord<RayCast::HitGroupData> HitGroupSbtRecord;

struct RayCastingState {
    OptixDeviceContext context = 0;
    OptixTraversableHandle gas_handle = 0; // Traversable handle for triangle AS
    CUdeviceptr gas_output_buffer = 0;     // Triangle AS memory

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
#endif

// abstract class of RayCaster
class RayCaster {
  public:
    RayCaster() {
    }
    virtual ~RayCaster() {
    }

#ifdef ENABLE_OPTIX
    virtual void build_gas(const Tensor &vertices, const Tensor &faces) = 0;
    virtual void build_pipeline() = 0;
    virtual void launch_optix(const RayCast::Params &params, const int32_t num_rays) = 0;
#else
    virtual void build_bvh(const Tensor &vertices, const Tensor &faces) = 0;
#endif
    virtual void invoke(const Tensor &origins, const Tensor &directions, Tensor &depths,
                        Tensor &normals, Tensor &primitives_ids) = 0;
};

// function to create an implementation of RayCaster
RayCaster *create_raycaster(const Tensor &vertices, const Tensor &triangles);
} // namespace prim3d
