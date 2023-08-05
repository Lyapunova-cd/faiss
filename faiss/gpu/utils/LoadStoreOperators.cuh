/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/utils/Float16.cuh>

#ifndef __HALF2_TO_UI
// cuda_fp16.hpp doesn't export this
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int*>(&(var)))
#endif

//
// Templated wrappers to express load/store for different scalar and vector
// types, so kernels can have the same written form but can operate
// over half and float, and on vector types transparently
//

namespace faiss {
namespace gpu {

template <typename T>
struct LoadStore {
    static inline __device__ T load(void* p) {
        return *((T*)p);
    }

    static inline __device__ void store(void* p, const T& v) {
        *((T*)p) = v;
    }
};

template <>
struct LoadStore<Half4> {
    static inline __device__ Half4 load(void* p) {
        Half4 out;
        uint32_t* ptr = static_cast<uint32_t*>(p);
        out.a.x = __uint2half_rn(ptr[0] & 0xFFFF);
        out.a.y = __uint2half_rn(ptr[0] >> 16);
        out.b.x = __uint2half_rn(ptr[1] & 0xFFFF);
        out.b.y = __uint2half_rn(ptr[1] >> 16);
        return out;
    }

    static inline __device__ void store(void* p, Half4& v) {
        uint32_t* ptr = static_cast<uint32_t*>(p);
        ptr[0] = static_cast<uint32_t>(__half_as_short(v.a.x)) |
                (static_cast<uint32_t>(__half_as_short(v.a.y)) << 16);
        ptr[1] = static_cast<uint32_t>(__half_as_short(v.b.x)) |
                (static_cast<uint32_t>(__half_as_short(v.b.y)) << 16);
    }
};

template <>
struct LoadStore<Half8> {
    static inline __device__ Half8 load(void* p) {
        Half8 out;
        uint32_t* ptr = static_cast<uint32_t*>(p);
        out.a.a.x = __uint2half_rn(ptr[0] & 0xFFFF);
        out.a.a.y = __uint2half_rn(ptr[0] >> 16);
        out.a.b.x = __uint2half_rn(ptr[1] & 0xFFFF);
        out.a.b.y = __uint2half_rn(ptr[1] >> 16);
        out.b.a.x = __uint2half_rn(ptr[2] & 0xFFFF);
        out.b.a.y = __uint2half_rn(ptr[2] >> 16);
        out.b.b.x = __uint2half_rn(ptr[3] & 0xFFFF);
        out.b.b.y = __uint2half_rn(ptr[3] >> 16);
        return out;
    }

    static inline __device__ void store(void* p, Half8& v) {
        uint32_t* ptr = static_cast<uint32_t*>(p);
        ptr[0] = static_cast<uint32_t>(__half_as_short(v.a.a.x)) |
                (static_cast<uint32_t>(__half_as_short(v.a.a.y)) << 16);
        ptr[1] = static_cast<uint32_t>(__half_as_short(v.a.b.x)) |
                (static_cast<uint32_t>(__half_as_short(v.a.b.y)) << 16);
        ptr[2] = static_cast<uint32_t>(__half_as_short(v.b.a.x)) |
                (static_cast<uint32_t>(__half_as_short(v.b.a.y)) << 16);
        ptr[3] = static_cast<uint32_t>(__half_as_short(v.b.b.x)) |
                (static_cast<uint32_t>(__half_as_short(v.b.b.y)) << 16);
    }
};

} // namespace gpu
} // namespace faiss
