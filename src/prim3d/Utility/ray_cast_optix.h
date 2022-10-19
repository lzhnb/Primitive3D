#pragma once
#include <iostream>
#include <memory>

#include "optix_ext/launch_parameters.h"
#include "optix_ext/utils.h"

namespace prim3d {
template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

// abstract class of RayCaster
class RayCaster {
public:
    RayCaster() {}
    virtual ~RayCaster() {}

    virtual void build_gas() = 0;
    virtual void build_pipeline() = 0;
    virtual void invoke(
        const RayCast::Params& params, const int32_t height, const int32_t width) = 0;
};

// function to create an implementation of RayCaster
RayCaster* create_raycaster();
}  // namespace prim3d
