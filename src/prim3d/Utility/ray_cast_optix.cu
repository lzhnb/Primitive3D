// #include <optix/ray_casting.h>

#include <array>
#include "ray_cast_optix.h"
namespace optix_ptx {
	#include <optix_ptx.h>
}

namespace prim3d {
typedef SbtRecord<RayCast::RayGenData> RayGenSbtRecord;
typedef SbtRecord<RayCast::MissData> MissSbtRecord;
typedef SbtRecord<RayCast::HitGroupData> HitGroupSbtRecord;

void RayCaster::build_gas() {
    // Specify options for the build. We use default options for simplicity.
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Triangle build input: simple list of three vertices
    const std::array<float3, 3> vertices = {
        {{-0.5f, -0.5f, 0.0f}, {0.5f, -0.5f, 0.0f}, {0.0f, 0.5f, 0.0f}}};

    // Allocate and copy device memory for our input triangle vertices
    const size_t vertices_size = sizeof(float3) * vertices.size();
    CUdeviceptr d_vertices = 0;
    cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertices_size);
    cudaMemcpy(
        reinterpret_cast<void*>(d_vertices),
        vertices.data(),
        vertices_size,
        cudaMemcpyHostToDevice);

    // Populate the build input struct with our triangle data as well as
    // information about the sizes and types of our data
    const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.numVertices = vertices.size();
    triangle_input.triangleArray.vertexBuffers = &d_vertices;
    triangle_input.triangleArray.flags = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;

    // Query OptiX for the memory requirements for our GAS
    OptixAccelBufferSizes gas_buffer_sizes;
    optixAccelComputeMemoryUsage(
        m_context,  // The device context we are using
        &accel_options,
        &triangle_input,  // Describes our geometry
        1,                // Number of build inputs, could have multiple
        &gas_buffer_sizes);

    // Allocate device memory for the scratch space buffer as well
    // as the GAS itself
    CUdeviceptr d_temp_buffer_gas;
    cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes);
    cudaMalloc(reinterpret_cast<void**>(&m_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes);

    optixAccelBuild(
        m_context,
        0,  // CUDA stream
        &accel_options,
        &triangle_input,
        1,  // num build inputs
        d_temp_buffer_gas,
        gas_buffer_sizes.tempSizeInBytes,
        m_gas_output_buffer,
        gas_buffer_sizes.outputSizeInBytes,
        &m_gas_handle,  // Output handle to the struct
        nullptr,        // emitted property list
        0);             // num emitted properties

    // We can now free scratch space used during the build
    cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas));
}

void RayCaster::build_pipeline() {
    char log[2048];  // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof(log);

    /* Convert CUDA code into PTX, NVIDIA's intermediate code
       and convert the PTX into optixModules
     */
    // Pipeline options must be consistent for all modules used in a single pipeline
    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur = false;
    {
        // Default options for our module.
        OptixModuleCompileOptions module_compile_options = {};

        // This option is important to ensure we compile code which is optimal
        // for our scene hierarchy. We use a single GAS – no instancing or
        // multi-level hierarchies
        pipeline_compile_options.traversableGraphFlags =
            OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;

        // Our device code uses 3 payload registers (r,g,b output value)
        pipeline_compile_options.numPayloadValues = 3;

        // This is the name of the param struct variable in our device code
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

        // read the ptx code during the compling
        const char* ptx_code = (char*) optix_ptx::device_programs_ptx;
        const size_t ptx_size = sizeof(optix_ptx::device_programs_ptx);
        // const std::string ptx = "ptx.cu";
        size_t sizeof_log = sizeof(log);

        m_module = nullptr;  // The output module
        optixModuleCreateFromPTX(
            m_context,
            &module_compile_options,
            &pipeline_compile_options,
            // ptx.c_str(),
            // ptx.size(),
            ptx_code,
            ptx_size,
            log,
            &sizeof_log,
            &m_module);
    }

    /* convert the optixModules into the optixProgramGroup */
    OptixProgramGroup raygen_prog_group = nullptr;
    OptixProgramGroup miss_prog_group = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;
    {
        OptixProgramGroupOptions program_group_options = {};  // Initialize to zeros

        OptixProgramGroupDesc raygen_prog_group_desc = {};  //
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = m_module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
        OPTIX_CHECK_THROW_LOG(optixProgramGroupCreate(
            m_context,
            &raygen_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &raygen_prog_group));

        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = m_module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
        OPTIX_CHECK_THROW_LOG(optixProgramGroupCreate(
            m_context,
            &miss_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &miss_prog_group));

        OptixProgramGroupDesc hitgroup_prog_group_desc = {};
        hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prog_group_desc.hitgroup.moduleCH = m_module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        OPTIX_CHECK_THROW_LOG(optixProgramGroupCreate(
            m_context,
            &hitgroup_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &hitgroup_prog_group));
    }

    /* link */
    {
        const uint32_t max_trace_depth = 1;
        OptixProgramGroup program_groups[] = {
            raygen_prog_group, miss_prog_group, hitgroup_prog_group};

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth = max_trace_depth;
        pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;

        m_pipeline = nullptr;
        optixPipelineCreate(
            m_context,
            &pipeline_compile_options,
            &pipeline_link_options,
            program_groups,
            sizeof(program_groups) / sizeof(program_groups[0]),
            log,
            &sizeof_log,
            &m_pipeline);

        // TODO: complete here
    }

    /* The SBT shader binding table */
    {
        CUdeviceptr raygen_record;
        const size_t raygen_record_size = sizeof(RayGenSbtRecord);
        cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size);
        RayGenSbtRecord rg_sbt;
        OPTIX_CHECK_THROW(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
        cudaMemcpy(
            reinterpret_cast<void*>(raygen_record),
            &rg_sbt,
            raygen_record_size,
            cudaMemcpyHostToDevice);

        CUdeviceptr miss_record;
        size_t miss_record_size = sizeof(MissSbtRecord);
        cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size);
        MissSbtRecord ms_sbt;
        OPTIX_CHECK_THROW(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
        cudaMemcpy(
            reinterpret_cast<void*>(miss_record),
            &ms_sbt,
            miss_record_size,
            cudaMemcpyHostToDevice);

        CUdeviceptr hitgroup_record;
        size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);

        cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size);
        HitGroupSbtRecord hg_sbt;
        OPTIX_CHECK_THROW(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
        cudaMemcpy(
            reinterpret_cast<void*>(hitgroup_record),
            &hg_sbt,
            hitgroup_record_size,
            cudaMemcpyHostToDevice);

        m_sbt = {};
        m_sbt.raygenRecord = raygen_record;
        m_sbt.missRecordBase = miss_record;
        m_sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
        m_sbt.missRecordCount = 1;
        m_sbt.hitgroupRecordBase = hitgroup_record;
        m_sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        m_sbt.hitgroupRecordCount = 1;
    }
}

void RayCaster::invoke(const RayCast::Params& params, const int32_t height, const int32_t width) {
    // Transfer params to the device
    CUdeviceptr d_param;
    cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(RayCast::Params));
    cudaMemcpy(reinterpret_cast<void*>(d_param), &params, sizeof(params), cudaMemcpyHostToDevice);
    OPTIX_CHECK_THROW(
        optixLaunch(m_pipeline, 0, d_param, sizeof(RayCast::Params), &m_sbt, width, height, 1));
}
}  // namespace prim3d
