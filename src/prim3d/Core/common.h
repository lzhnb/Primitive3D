#pragma once

#ifdef ENABLE_OPTIX
#include <optix.h>
#endif
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef __NVCC__
#define PRIM_HOST_DEVICE __host__ __device__
#else
#define PRIM_HOST_DEVICE
#endif

//////////////////////////////////////
// CUDA ERROR HANDLING (EXCEPTIONS) //
//////////////////////////////////////

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)

/// Checks the result of a cudaXXXXXX call and throws an error on failure
#define CUDA_CHECK(x)                                                                              \
    do {                                                                                           \
        cudaError_t result = x;                                                                    \
        if (result != cudaSuccess)                                                                 \
            throw std::runtime_error(                                                              \
                std::string(FILE_LINE " " #x " failed with error ") + cudaGetErrorString(result)); \
    } while (0)

#ifdef ENABLE_OPTIX
#define OPTIX_CHECK(x)                                                            \
    do {                                                                          \
        OptixResult res = x;                                                      \
        if (res != OPTIX_SUCCESS) {                                               \
            throw std::runtime_error(std::string("Optix call '" #x "' failed.")); \
        }                                                                         \
    } while (0)

#define OPTIX_CHECK_LOG(x)                                                                      \
    do {                                                                                        \
        OptixResult res                  = x;                                                   \
        const size_t sizeof_log_returned = sizeof_log;                                          \
        sizeof_log                       = sizeof(log); /* reset sizeof_log for future calls */ \
        if (res != OPTIX_SUCCESS) {                                                             \
            throw std::runtime_error(                                                           \
                std::string("Optix call '" #x "' failed. Log:\n") + log +                       \
                (sizeof_log_returned == sizeof_log ? "" : "<truncated>"));                      \
        }                                                                                       \
    } while (0)
#endif

/* torch tensor check */
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)
#define CHECK_CPU_INPUT(x) \
    CHECK_CPU(x);          \
    CHECK_CONTIGUOUS(x)
