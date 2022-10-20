/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   raytrace.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Minimal optix program.
 */

#pragma once

#include <optix.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// #include "vec_math.h"
// #include "triangle.cuh"

#define OPTIX_CHECK(x)                                                            \
    do {                                                                          \
        OptixResult res = x;                                                      \
        if (res != OPTIX_SUCCESS) {                                               \
            throw std::runtime_error(std::string("Optix call '" #x "' failed.")); \
        }                                                                         \
    } while (0)

#define OPTIX_CHECK_LOG(x)                                                 \
    do {                                                                   \
        OptixResult res = x;                                               \
        const size_t sizeof_log_returned = sizeof_log;                     \
        sizeof_log = sizeof(log); /* reset sizeof_log for future calls */  \
        if (res != OPTIX_SUCCESS) {                                        \
            throw std::runtime_error(                                      \
                std::string("Optix call '" #x "' failed. Log:\n") + log +  \
                (sizeof_log_returned == sizeof_log ? "" : "<truncated>")); \
        }                                                                  \
    } while (0)

static std::vector<char> read_data(std::string const& filename) {
    std::ifstream inputData(filename, std::ios::binary);

    if (inputData.fail()) {
        std::cerr << "ERROR: read_data() Failed to open file " << filename << '\n';
        return std::vector<char>();
    }

    // Copy the input buffer to a char vector.
    std::vector<char> data(std::istreambuf_iterator<char>(inputData), {});

    if (inputData.fail()) {
        std::cerr << "ERROR: read_data() Failed to read file " << filename << '\n';
        return std::vector<char>();
    }

    return data;
}
