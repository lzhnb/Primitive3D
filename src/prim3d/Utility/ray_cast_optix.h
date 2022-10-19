#pragma once
#include <iostream>
#include <memory>

#include "optix_ext/utils.h"
#include "optix_ext/launch_parameters.h"

namespace prim3d {
template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

class RayCaster {
public:
    RayCaster() {
        /* init optix*/
        // Initialize CUDA with a no-op call to the the CUDA runtime API
        cudaFree(nullptr);

        // Initialize the OptiX API, loading all API entry points
        OPTIX_CHECK_THROW(optixInit());

        // Specify options for this context. We will use the default options.
        m_options = {};

        // Associate a CUDA context (and therefore a specific GPU) with this
        // device context
        m_cuCtx = 0;  // NULL means take the current active context

        m_context = nullptr;
        OPTIX_CHECK_THROW(optixDeviceContextCreate(m_cuCtx, &m_options, &m_context));
    };
    ~RayCaster(){};
    void build_gas();
    void build_pipeline();

    void invoke(const RayCast::Params& params, const int32_t height, const int32_t width);

private:
    OptixDeviceContextOptions m_options;
    OptixDeviceContext m_context;
    CUcontext m_cuCtx;
    OptixTraversableHandle m_gas_handle;
    CUdeviceptr m_gas_output_buffer;
    OptixModule m_module;  // The output module
    OptixPipeline m_pipeline;
    OptixShaderBindingTable m_sbt;
};
}  // namespace prim3d
