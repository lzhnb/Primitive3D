#include <array>

#include <Core/utils.h>

#include "ray_cast_optix.h"

namespace optix_ptx {
#include <optix_ptx.h>
}

// Kernels

__global__ void vertices_faces_to_triangles(
    const int32_t num_triangles,
    const float* __restrict__ vertices_ptr,
    const int32_t* __restrict__ faces_ptr,
    // output
    prim3d::Triangle* __restrict__ triangles_ptr) {
    const int32_t triangle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangle_id >= num_triangles) return;

    // found points' ids
    const int32_t p1 = faces_ptr[triangle_id * 3 + 0];
    const int32_t p2 = faces_ptr[triangle_id * 3 + 1];
    const int32_t p3 = faces_ptr[triangle_id * 3 + 2];

    // fullfill the points
    triangles_ptr[triangle_id].a.x = vertices_ptr[p1 * 3 + 0];
    triangles_ptr[triangle_id].a.y = vertices_ptr[p1 * 3 + 1];
    triangles_ptr[triangle_id].a.z = vertices_ptr[p1 * 3 + 2];
    triangles_ptr[triangle_id].b.x = vertices_ptr[p2 * 3 + 0];
    triangles_ptr[triangle_id].b.y = vertices_ptr[p2 * 3 + 1];
    triangles_ptr[triangle_id].b.z = vertices_ptr[p2 * 3 + 2];
    triangles_ptr[triangle_id].c.x = vertices_ptr[p3 * 3 + 0];
    triangles_ptr[triangle_id].c.y = vertices_ptr[p3 * 3 + 1];
    triangles_ptr[triangle_id].c.z = vertices_ptr[p3 * 3 + 2];
}

__global__ void origins_directions_to_float3s(
    const int32_t num_rays,
    const float* __restrict__ origins_ptr,
    const float* __restrict__ directions_ptr,
    // output
    float3* __restrict__ ray_origins_ptr,
    float3* __restrict__ ray_directions_ptr
) {
    const int32_t ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_id >= num_rays) return;

    ray_origins_ptr[ray_id].x = origins_ptr[ray_id * 3 + 0];
    ray_origins_ptr[ray_id].y = origins_ptr[ray_id * 3 + 1];
    ray_origins_ptr[ray_id].z = origins_ptr[ray_id * 3 + 2];
    ray_directions_ptr[ray_id].x = directions_ptr[ray_id * 3 + 0];
    ray_directions_ptr[ray_id].y = directions_ptr[ray_id * 3 + 1];
    ray_directions_ptr[ray_id].z = directions_ptr[ray_id * 3 + 2];
}

__global__ void export_depth_normals(
    const int32_t num_rays,
    const float3* __restrict__ depths_ptr,
    const float3* __restrict__ normals_ptr,
    // output
    float* __restrict__ output_depths_ptr,
    float* __restrict__ output_normals_ptr
) {
    const int32_t ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_id >= num_rays) return;

    output_depths_ptr[ray_id] = depths_ptr[ray_id].x;
    output_normals_ptr[ray_id * 3 + 0] = normals_ptr[ray_id].x;
    output_normals_ptr[ray_id * 3 + 1] = normals_ptr[ray_id].y;
    output_normals_ptr[ray_id * 3 + 2] = normals_ptr[ray_id].z;
}

namespace prim3d {
typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

class RayCasterImpl : public RayCaster {
public:
    // accept numpy array (cpu) to init
    RayCasterImpl() : RayCaster() {
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
    }

    void build_gas(Triangle* triangles, const int32_t num_triangles) {
        // set the gpu_triangles
        m_gpu_triangles = triangles;

        // Specify options for the build. We use default options for simplicity.
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        // Populate the build input struct with our triangle data as well as
        // information about the sizes and types of our data
        const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
        OptixBuildInput triangle_input = {};

        CUdeviceptr d_triangles = (CUdeviceptr)(uintptr_t)triangles;

        triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.numVertices = num_triangles * 3;
        triangle_input.triangleArray.vertexBuffers = &d_triangles;
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
        cudaMalloc(
            reinterpret_cast<void**>(&m_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes);

        OPTIX_CHECK_THROW(optixAccelBuild(
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
            0));            // num emitted properties

        // We can now free scratch space used during the build
        cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas));
    }

    void build_pipeline() {
        char log[2048];  // For error reporting from OptiX creation functions
        size_t sizeof_log = sizeof(log);

        /* convert the PTX into optixModules */
        // Pipeline options must be consistent for all modules used in a single pipeline
        OptixPipelineCompileOptions pipeline_compile_options = {};
        pipeline_compile_options.usesMotionBlur = false;
        m_module = nullptr;  // The output module
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
            const char* ptx_code = (char*)optix_ptx::device_programs_ptx;
            const size_t ptx_size = sizeof(optix_ptx::device_programs_ptx);
            size_t sizeof_log = sizeof(log);

            optixModuleCreateFromPTX(
                m_context,
                &module_compile_options,
                &pipeline_compile_options,
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

            OptixProgramGroupDesc raygen_prog_group_desc = {};
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

        /* linking */
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

            OptixStackSizes stack_sizes = {};
            for (auto& prog_group : program_groups) {
                OPTIX_CHECK_THROW(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
            }

            uint32_t direct_callable_stack_size_from_traversal;
            uint32_t direct_callable_stack_size_from_state;
            uint32_t continuation_stack_size;
            OPTIX_CHECK_THROW(optixUtilComputeStackSizes(
                &stack_sizes,
                max_trace_depth,
                0,  // maxCCDepth
                0,  // maxDCDEpth
                &direct_callable_stack_size_from_traversal,
                &direct_callable_stack_size_from_state,
                &continuation_stack_size));
            OPTIX_CHECK_THROW(optixPipelineSetStackSize(
                m_pipeline,
                direct_callable_stack_size_from_traversal,
                direct_callable_stack_size_from_state,
                continuation_stack_size,
                1  // maxTraversableDepth
                ));
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

    void invoke(const Params& params, const int32_t num_rays) {
        // Transfer params to the device
        CUdeviceptr d_param;
        cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params));
        cudaMemcpy(
            reinterpret_cast<void*>(d_param), &params, sizeof(params), cudaMemcpyHostToDevice);
        OPTIX_CHECK_THROW(
            optixLaunch(m_pipeline, 0, d_param, sizeof(Params), &m_sbt, num_rays, 1, 1));
    }

    void cast(const Tensor& origins, const Tensor& directions, Tensor& depths, Tensor& normals) {
        CHECK_INPUT(origins);
        CHECK_INPUT(directions);

        const int32_t num_rays = origins.size(0);
        float3* ray_origins = NULL;
        cudaMalloc((void**)&ray_origins, sizeof(float3) * num_rays);
        float3* ray_directions = NULL;
        cudaMalloc((void**)&ray_directions, sizeof(float3) * num_rays);
        const int32_t blocks = n_blocks_linear(num_rays);

        /* convert origins and directions to float3s */
        origins_directions_to_float3s<<<blocks, n_threads_linear>>>(
            num_rays,
            origins.data_ptr<float>(),
            directions.data_ptr<float>(),
            // output
            ray_origins,
            ray_directions
        );

        // invode optix ray casting
        Params params = {ray_origins, ray_directions, m_gpu_triangles, m_gas_handle};
        invoke(params, num_rays);

        // get the output
        export_depth_normals<<<blocks, n_threads_linear>>>(
            num_rays,
            params.ray_origins,
            params.ray_directions,
            // output
            depths.data_ptr<float>(),
            normals.data_ptr<float>()
        );
    }

private:
    OptixDeviceContextOptions m_options;
    OptixDeviceContext m_context;
    CUcontext m_cuCtx;
    OptixTraversableHandle m_gas_handle;
    CUdeviceptr m_gas_output_buffer;
    OptixModule m_module;  // The output module
    OptixPipeline m_pipeline;
    OptixShaderBindingTable m_sbt;
    Triangle* m_gpu_triangles;
};

RayCaster* create_raycaster(const Tensor& vertices, const Tensor& faces) {
    CHECK_INPUT(vertices);
    CHECK_INPUT(faces);

    // conver the vertices and faces into triangles
    const int32_t num_triangles = faces.size(0);
    Triangle* triangles = NULL;
    cudaMalloc((void**)&triangles, sizeof(Triangle) * num_triangles);
    const int32_t blocks = n_blocks_linear(num_triangles);

    vertices_faces_to_triangles<<<blocks, n_threads_linear>>>(
        num_triangles, vertices.data_ptr<float>(), faces.data_ptr<int32_t>(), triangles);

    RayCaster* ray_caster = new RayCasterImpl{};
    ray_caster->build_gas(triangles, num_triangles);
    ray_caster->build_pipeline();
    return ray_caster;
}
}  // namespace prim3d
