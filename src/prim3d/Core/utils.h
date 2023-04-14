// Copyright 2022 Zhihao Liang
#pragma once

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "common.h"

namespace prim3d {

void test();

static constexpr float PI = 3.14159265358979323846f;
static constexpr float SQRT2 = 1.41421356237309504880f;

constexpr uint32_t n_threads_linear = 1024;

template <typename T>
PRIM_HOST_DEVICE T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}

template <typename T>
constexpr uint32_t n_blocks_linear(T n_elements) {
    return (uint32_t)div_round_up(n_elements, (T)n_threads_linear);
}

inline PRIM_HOST_DEVICE float sign(float x) {
    return copysignf(1.0, x);
}

template <typename T>
PRIM_HOST_DEVICE T clamp(T val, T lower, T upper) {
    return val < lower ? lower : (upper < val ? upper : val);
}

template <typename T>
PRIM_HOST_DEVICE void host_device_swap(T &a, T &b) {
    T c(a);
    a = b;
    b = c;
}

} // namespace prim3d