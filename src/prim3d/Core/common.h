#pragma once

#include <array>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

//////////////////////////////////////
// CUDA ERROR HANDLING (EXCEPTIONS) //
//////////////////////////////////////

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)

/// Checks the result of a cudaXXXXXX call and throws an error on failure
#define CUDA_CHECK_THROW(x)                                                                        \
    do {                                                                                           \
        cudaError_t result = x;                                                                    \
        if (result != cudaSuccess)                                                                 \
            throw std::runtime_error(                                                              \
                std::string(FILE_LINE " " #x " failed with error ") + cudaGetErrorString(result)); \
    } while (0)
