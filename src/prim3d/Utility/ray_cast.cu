// Copyright 2022 Zhihao Liang
#include <array>
#include <vector>

#include "ray_cast.h"

#ifdef ENABLE_OPTIX
namespace optix_ptx {
#include <optix_ptx.h>
}

// Kernels

__global__ void vertices_faces_to_triangles(
    const int32_t num_triangles,
    const float* __restrict__ vertices_ptr,
    const int32_t* __restrict__ faces_ptr,
    // output
    prim3d::OptixTriangle* __restrict__ triangles_ptr) {
    const int32_t triangle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangle_id >= num_triangles) return;

    // found points' ids
    const int32_t p1 = faces_ptr[triangle_id * 3 + 0];
    const int32_t p2 = faces_ptr[triangle_id * 3 + 1];
    const int32_t p3 = faces_ptr[triangle_id * 3 + 2];

    // fullfill the points
    triangles_ptr[triangle_id].a =
        Vector3f{vertices_ptr[p1 * 3 + 0], vertices_ptr[p1 * 3 + 1], vertices_ptr[p1 * 3 + 2]};
    triangles_ptr[triangle_id].b =
        Vector3f{vertices_ptr[p2 * 3 + 0], vertices_ptr[p2 * 3 + 1], vertices_ptr[p2 * 3 + 2]};
    triangles_ptr[triangle_id].c =
        Vector3f{vertices_ptr[p3 * 3 + 0], vertices_ptr[p3 * 3 + 1], vertices_ptr[p3 * 3 + 2]};
}

#endif
namespace prim3d {

class RayCasterImpl : public RayCaster {
#ifdef ENABLE_OPTIX
public:
    RayCasterImpl() : RayCaster() {
        /* init optix and create context*/
        // Initialize CUDA with a no-op call to the the CUDA runtime API
        cudaFree(nullptr);

        // Initialize the OptiX API, loading all API entry points
        OPTIX_CHECK(optixInit());

        // Specify options for this context. We will use the default options.
        OptixDeviceContextOptions options = {};

        // Associate a CUDA context (and therefore a specific GPU) with this
        // device context
        CUcontext cuCtx = 0;  // NULL means take the current active context

        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &m_state.context));
    }

    void build_gas(const Tensor& vertices, const Tensor& faces) {
        CHECK_INPUT(vertices);
        CHECK_INPUT(faces);
        // conver the vertices and faces into triangles
        const int32_t num_triangles = faces.size(0);
        m_triangles                 = NULL;
        cudaMalloc((void**)&m_triangles, sizeof(OptixTriangle) * num_triangles);
        const int32_t blocks = n_blocks_linear(num_triangles);

        vertices_faces_to_triangles<<<blocks, n_threads_linear>>>(
            num_triangles, vertices.data_ptr<float>(), faces.data_ptr<int32_t>(), m_triangles);

        // Specify options for the build. We use default options for simplicity.
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags             = OPTIX_BUILD_FLAG_NONE;
        accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

        // Populate the build input struct with our triangle data as well as
        // information about the sizes and types of our data
        const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
        OptixBuildInput triangle_input         = {};

        CUdeviceptr d_triangles = (CUdeviceptr)(uintptr_t)m_triangles;

        triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.numVertices   = num_triangles * 3;
        triangle_input.triangleArray.vertexBuffers = &d_triangles;
        triangle_input.triangleArray.flags         = triangle_input_flags;
        triangle_input.triangleArray.numSbtRecords = 1;

        // Query OptiX for the memory requirements for our GAS
        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            m_state.context,  // The device context we are using
            &accel_options,
            &triangle_input,  // Describes our geometry
            1,                // Number of build inputs, could have multiple
            &gas_buffer_sizes));

        // Allocate device memory for the scratch space buffer as well
        // as the GAS itself
        CUdeviceptr d_temp_buffer_gas;
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&m_state.gas_output_buffer),
            gas_buffer_sizes.outputSizeInBytes));

        OPTIX_CHECK(optixAccelBuild(
            m_state.context,
            0,  // CUDA stream
            &accel_options,
            &triangle_input,
            1,  // num build inputs
            d_temp_buffer_gas,
            gas_buffer_sizes.tempSizeInBytes,
            m_state.gas_output_buffer,
            gas_buffer_sizes.outputSizeInBytes,
            &m_state.gas_handle,  // Output handle to the struct
            nullptr,              // emitted property list
            0));                  // num emitted properties

        // We can now free scratch space used during the build
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
    }

    void build_pipeline() {
        char log[2048];  // For error reporting from OptiX creation functions
        size_t sizeof_log = sizeof(log);

        /* convert the PTX into optixModules */
        {
            // Default options for our module.
            OptixModuleCompileOptions module_compile_options = {};

            // Pipeline options must be consistent for all modules used in a single pipeline
            m_state.pipeline_compile_options.usesMotionBlur = false;

            // This option is important to ensure we compile code which is optimal
            // for our scene hierarchy. We use a single GAS – no instancing or
            // multi-level hierarchies
            m_state.pipeline_compile_options.traversableGraphFlags =
                OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;

            // NOTE: the number of payload registers
            m_state.pipeline_compile_options.numPayloadValues = 5;

            // This is the name of the param struct variable in our device code
            m_state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

            // read the ptx code during the compling
            const char* ptx_code  = (char*)optix_ptx::device_programs_ptx;
            const size_t ptx_size = sizeof(optix_ptx::device_programs_ptx);
            size_t sizeof_log     = sizeof(log);

            optixModuleCreateFromPTX(
                m_state.context,
                &module_compile_options,
                &m_state.pipeline_compile_options,
                ptx_code,
                ptx_size,
                log,
                &sizeof_log,
                &m_state.ptx_module);
        }

        /* convert the optixModules into the optixProgramGroup */
        {
            OptixProgramGroupOptions program_group_options = {};  // Initialize to zeros

            OptixProgramGroupDesc raygen_prog_group_desc    = {};
            raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module            = m_state.ptx_module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                m_state.context,
                &raygen_prog_group_desc,
                1,  // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &m_state.raygen_prog_group));

            OptixProgramGroupDesc miss_prog_group_desc  = {};
            miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
            miss_prog_group_desc.miss.module            = m_state.ptx_module;
            miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                m_state.context,
                &miss_prog_group_desc,
                1,  // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &m_state.miss_prog_group));

            OptixProgramGroupDesc hit_prog_group_desc        = {};
            hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hit_prog_group_desc.hitgroup.moduleCH            = m_state.ptx_module;
            hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                m_state.context,
                &hit_prog_group_desc,
                1,  // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &m_state.hit_prog_group));
        }

        /* linking */
        {
            const uint32_t max_trace_depth      = 1;
            OptixProgramGroup program_groups[3] = {
                m_state.raygen_prog_group, m_state.miss_prog_group, m_state.hit_prog_group};

            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth            = max_trace_depth;
            pipeline_link_options.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;

            // m_pipeline = nullptr;
            OPTIX_CHECK_LOG(optixPipelineCreate(
                m_state.context,
                &m_state.pipeline_compile_options,
                &pipeline_link_options,
                program_groups,
                sizeof(program_groups) / sizeof(program_groups[0]),
                log,
                &sizeof_log,
                &m_state.pipeline));

            OptixStackSizes stack_sizes = {};
            for (auto& prog_group : program_groups) {
                OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
            }

            uint32_t direct_callable_stack_size_from_traversal;
            uint32_t direct_callable_stack_size_from_state;
            uint32_t continuation_stack_size;
            OPTIX_CHECK(optixUtilComputeStackSizes(
                &stack_sizes,
                max_trace_depth,
                0,  // maxCCDepth
                0,  // maxDCDEpth
                &direct_callable_stack_size_from_traversal,
                &direct_callable_stack_size_from_state,
                &continuation_stack_size));
            OPTIX_CHECK(optixPipelineSetStackSize(
                m_state.pipeline,
                direct_callable_stack_size_from_traversal,
                direct_callable_stack_size_from_state,
                continuation_stack_size,
                1  // maxTraversableDepth
                ));
        }

        /* create SBT shader binding table */
        {
            CUdeviceptr d_raygen_record     = 0;
            const size_t raygen_record_size = sizeof(RayGenSbtRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), raygen_record_size));
            RayGenSbtRecord rg_sbt_record;
            OPTIX_CHECK(optixSbtRecordPackHeader(m_state.raygen_prog_group, &rg_sbt_record));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(d_raygen_record),
                &rg_sbt_record,
                raygen_record_size,
                cudaMemcpyHostToDevice));

            CUdeviceptr d_miss_record = 0;
            size_t miss_record_size   = sizeof(MissSbtRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_record), miss_record_size));
            MissSbtRecord ms_sbt_record;
            OPTIX_CHECK(optixSbtRecordPackHeader(m_state.miss_prog_group, &ms_sbt_record));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(d_miss_record),
                &ms_sbt_record,
                miss_record_size,
                cudaMemcpyHostToDevice));

            int numObjects = 1;
            // we only have a single object type so far
            HitGroupSbtRecord hg_sbt_record = {};
            OPTIX_CHECK(optixSbtRecordPackHeader(m_state.hit_prog_group, &hg_sbt_record));
            hg_sbt_record.data.data = m_triangles;

            // HitGroupSbtRecord hg_sbt_record = {};
            // OPTIX_CHECK(optixSbtRecordPackHeader(m_state.hit_prog_group, &hg_sbt_record));
            CUdeviceptr d_hitgroup_record = 0;
            size_t mesh_record_size       = sizeof(HitGroupSbtRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_record), mesh_record_size));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(d_hitgroup_record),
                &hg_sbt_record,
                mesh_record_size,
                cudaMemcpyHostToDevice));

            m_state.sbt.raygenRecord                = d_raygen_record;
            m_state.sbt.missRecordBase              = d_miss_record;
            m_state.sbt.missRecordStrideInBytes     = static_cast<uint32_t>(miss_record_size);
            m_state.sbt.missRecordCount             = 1;
            m_state.sbt.hitgroupRecordBase          = d_hitgroup_record;
            m_state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(mesh_record_size);
            m_state.sbt.hitgroupRecordCount         = 1;  // the number of objects in hitgroup
        }
    }

    void launch_optix(const RayCast::Params& params, const int32_t num_rays) {
        /* Transfer params to the device */
        RayCast::Params* d_params = NULL;

        // cuda parameters
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(RayCast::Params)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_params),
            &params,
            sizeof(RayCast::Params),
            cudaMemcpyHostToDevice));

        // launch optix
        OPTIX_CHECK(optixLaunch(
            m_state.pipeline,
            0,
            reinterpret_cast<CUdeviceptr>(d_params),
            sizeof(RayCast::Params),
            &m_state.sbt,
            num_rays,
            1,
            1));

        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));
    }

#else

public:
    RayCasterImpl() : RayCaster() {}

    void build_bvh(const Tensor& vertices, const Tensor& faces) {
        // confirm the cpu tensor
        CHECK_CPU_INPUT(vertices);
        CHECK_CPU_INPUT(faces);

        // conver the vertices and faces into triangles
        const int32_t num_triangles = faces.size(0);
        std::vector<Triangle> triangles_cpu;
        triangles_cpu.resize(num_triangles);

        const int32_t* faces_ptr  = faces.data_ptr<int32_t>();
        const float* vertices_ptr = vertices.data_ptr<float>();
        for (size_t tri_id = 0; tri_id < num_triangles; tri_id++) {
            const int32_t p1_id   = faces_ptr[tri_id * 3 + 0];
            const int32_t p2_id   = faces_ptr[tri_id * 3 + 1];
            const int32_t p3_id   = faces_ptr[tri_id * 3 + 2];
            triangles_cpu[tri_id] = {
                Vector3f{
                    vertices_ptr[p1_id * 3 + 0],
                    vertices_ptr[p1_id * 3 + 1],
                    vertices_ptr[p1_id * 3 + 2]},
                Vector3f{
                    vertices_ptr[p2_id * 3 + 0],
                    vertices_ptr[p2_id * 3 + 1],
                    vertices_ptr[p2_id * 3 + 2]},
                Vector3f{
                    vertices_ptr[p3_id * 3 + 0],
                    vertices_ptr[p3_id * 3 + 1],
                    vertices_ptr[p3_id * 3 + 2]},
                (int32_t)tri_id};
        }

        if (!triangle_bvh) { triangle_bvh = TriangleBvh::make(); }

        triangle_bvh->build(triangles_cpu, 8);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMalloc((void**)&triangles_gpu, sizeof(Triangle) * num_triangles));
        CUDA_CHECK(cudaMemcpy(
            triangles_gpu,
            &triangles_cpu[0],
            sizeof(Triangle) * num_triangles,
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
#endif

    void invoke(
        const Tensor& origins,
        const Tensor& directions,
        Tensor& depths,
        Tensor& normals,
        Tensor& primitives_ids) {
        CHECK_INPUT(origins);
        CHECK_INPUT(directions);
        CHECK_INPUT(depths);
        CHECK_INPUT(normals);
        CHECK_INPUT(primitives_ids);

        const int32_t num_rays = origins.size(0);

#ifdef ENABLE_OPTIX
        // invode optix ray casting
        launch_optix(
            {origins.data_ptr<float>(),
             directions.data_ptr<float>(),
             depths.data_ptr<float>(),
             normals.data_ptr<float>(),
             primitives_ids.data_ptr<int32_t>(),
             m_state.gas_handle},
            num_rays);
#else
        // TODO
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        triangle_bvh->ray_trace_gpu(
            num_rays,
            origins.data_ptr<float>(),
            directions.data_ptr<float>(),
            depths.data_ptr<float>(),
            normals.data_ptr<float>(),
            primitives_ids.data_ptr<int32_t>(),
            triangles_gpu,
            stream);
#endif
    }

private:
#ifdef ENABLE_OPTIX
    RayCastingState m_state    = {};
    OptixTriangle* m_triangles = {};
#else
private:
    std::shared_ptr<TriangleBvh> triangle_bvh;
    Triangle* triangles_gpu = NULL;
#endif
};

RayCaster* create_raycaster(const Tensor& vertices, const Tensor& faces) {
    RayCaster* ray_caster = new RayCasterImpl{};
#ifdef ENABLE_OPTIX
    // build geometry acceleration structure
    ray_caster->build_gas(vertices, faces);
    // build the shading pipeline
    ray_caster->build_pipeline();
#else
    // build the bounding volume hierarchy
    ray_caster->build_bvh(vertices, faces);
#endif

    return ray_caster;
}

}  // namespace prim3d
