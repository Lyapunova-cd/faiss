/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/utils/PtxUtils.cuh>

namespace faiss {
namespace gpu {

#if __CUDA_ARCH__ >= 350
// Use the CC 3.5+ read-only texture cache (nc)
#define LD_NC_V1 "ld.global.cs.nc.u32"
#define LD_NC_V2 "ld.global.cs.nc.v2.u32"
#define LD_NC_V4 "ld.global.cs.nc.v4.u32"
#else
// Read normally
#define LD_NC_V1 "ld.global.cs.u32"
#define LD_NC_V2 "ld.global.cs.v2.u32"
#define LD_NC_V4 "ld.global.cs.v4.u32"
#endif // __CUDA_ARCH__

///
/// This file contains loader functions for PQ codes of various byte
/// length.
///

// Type-specific wrappers around the PTX bfe.* instruction, for
// quantization code extraction
inline __device__ unsigned int getByte(unsigned char v, int pos, int width) {
    return v;
}

inline __device__ unsigned int getByte(unsigned short v, int pos, int width) {
    return getBitfield((unsigned int)v, pos, width);
}

inline __device__ unsigned int getByte(unsigned int v, int pos, int width) {
    return getBitfield(v, pos, width);
}

inline __device__ unsigned int getByte(uint64_t v, int pos, int width) {
    return getBitfield(v, pos, width);
}

template <int NumSubQuantizers>
struct LoadCode32 {};

template <>
struct LoadCode32<1> {
    static inline __device__ void load(
            unsigned int code32[1],
            uint8_t* p,
            int offset) {
        p += offset * 1;
        code32[0] = *reinterpret_cast<unsigned int*>(p);
    }
};

template <>
struct LoadCode32<2> {
    static inline __device__ void load(
            unsigned int code32[1],
            uint8_t* p,
            int offset) {
        p += offset * 2;
        code32[0] = *reinterpret_cast<unsigned int*>(p);
    }
};

template <>
struct LoadCode32<3> {
    static inline __device__ void load(
            unsigned int code32[1],
            uint8_t* p,
            int offset) {
        p += offset * 3;
        unsigned int a = p[0];
        unsigned int b = p[1];
        unsigned int c = p[2];

        // FIXME: this is a non-coalesced, unaligned, non-vectorized load
        // unfortunately need to reorganize memory layout by warp

        // FIXME: this is also slow, since we have to recover the
        // individual bytes loaded
        code32[0] = (c << 16) | (b << 8) | a;
    }
};

template <>
struct LoadCode32<4> {
    static inline __device__ void load(
            unsigned int code32[1],
            uint8_t* p,
            int offset) {
        p += offset * 4;
        code32[0] = *reinterpret_cast<unsigned int*>(p);
    }
};

template <>
struct LoadCode32<8> {
    static inline __device__ void load(
            unsigned int code32[2],
            uint8_t* p,
            int offset) {
        p += offset * 8;
        unsigned int* p_cast = reinterpret_cast<unsigned int*>(p);
        code32[0] = p_cast[0];
        code32[1] = p_cast[1];
    }
};

template <>
struct LoadCode32<12> {
    static inline __device__ void load(
            unsigned int code32[3],
            uint8_t* p,
            int offset) {
        p += offset * 12;
        // FIXME: this is a non-coalesced, unaligned, non-vectorized load
        // unfortunately need to reorganize memory layout by warp
        unsigned int* p_cast = reinterpret_cast<unsigned int*>(p);
        code32[0] = p_cast[0];
        code32[1] = p_cast[1];
        code32[2] = p_cast[2];
    }
};

template <>
struct LoadCode32<16> {
    static inline __device__ void load(
            unsigned int code32[4],
            uint8_t* p,
            int offset) {
        p += offset * 16;
        unsigned int* p_cast = reinterpret_cast<unsigned int*>(p);
        code32[0] = p_cast[0];
        code32[1] = p_cast[1];
        code32[2] = p_cast[2];
        code32[3] = p_cast[3];
    }
};

template <>
struct LoadCode32<20> {
    static inline __device__ void load(
            unsigned int code32[5],
            uint8_t* p,
            int offset) {
        p += offset * 20;
        // FIXME: this is a non-coalesced, unaligned, non-vectorized load
        // unfortunately need to reorganize memory layout by warp
        unsigned int* p_cast = reinterpret_cast<unsigned int*>(p);
        code32[0] = p_cast[0];
        code32[1] = p_cast[1];
        code32[2] = p_cast[2];
        code32[3] = p_cast[3];
        code32[4] = p_cast[4];
    }
};

template <>
struct LoadCode32<24> {
    static inline __device__ void load(
            unsigned int code32[6],
            uint8_t* p,
            int offset) {
        p += offset * 24;
        // FIXME: this is a non-coalesced, unaligned, 2-vectorized load
        // unfortunately need to reorganize memory layout by warp
        unsigned int* p_cast = reinterpret_cast<unsigned int*>(p);
        code32[0] = p_cast[0];
        code32[1] = p_cast[1];
        code32[2] = p_cast[2];
        code32[3] = p_cast[3];
        code32[4] = p_cast[4];
        code32[5] = p_cast[5];
    }
};

template <>
struct LoadCode32<28> {
    static inline __device__ void load(
            unsigned int code32[7],
            uint8_t* p,
            int offset) {
        p += offset * 28;
        unsigned int* p_cast = reinterpret_cast<unsigned int*>(p);
        for (int i = 0; i < 7; i++) {
            code32[i] = p_cast[i];
        }
    }
};

template <>
struct LoadCode32<32> {
    static inline __device__ void load(
            unsigned int code32[8],
            uint8_t* p,
            int offset) {
        p += offset * 32;
        unsigned int* p_cast = reinterpret_cast<unsigned int*>(p);
        for (int i = 0; i < 8; i++) {
            code32[i] = p_cast[i];
        }
    }
};

template <>
struct LoadCode32<40> {
    static inline __device__ void load(
            unsigned int code32[10],
            uint8_t* p,
            int offset) {
        p += offset * 40;
        unsigned int* p_cast = reinterpret_cast<unsigned int*>(p);
        for (int i = 0; i < 10; i++) {
            code32[i] = p_cast[i];
        }
    }
};

template <>
struct LoadCode32<48> {
    static inline __device__ void load(
            unsigned int code32[12],
            uint8_t* p,
            int offset) {
        p += offset * 48;
        // FIXME: this is a non-coalesced load
        // unfortunately need to reorganize memory layout by warp
        unsigned int* p_cast = reinterpret_cast<unsigned int*>(p);
        for (int i = 0; i < 12; i++) {
            code32[i] = p_cast[i];
        }
    }
};

template <>
struct LoadCode32<56> {
    static inline __device__ void load(
            unsigned int code32[14],
            uint8_t* p,
            int offset) {
        p += offset * 56;
        // FIXME: this is a non-coalesced, unaligned, 2-vectorized load
        // unfortunately need to reorganize memory layout by warp
        unsigned int* p_cast = reinterpret_cast<unsigned int*>(p);
        for (int i = 0; i < 14; i++) {
            code32[i] = p_cast[i];
        }
    }
};

template <>
struct LoadCode32<64> {
    static inline __device__ void load(
            unsigned int code32[16],
            uint8_t* p,
            int offset) {
        p += offset * 64;
        // FIXME: this is a non-coalesced load
        // unfortunately need to reorganize memory layout by warp
        unsigned int* p_cast = reinterpret_cast<unsigned int*>(p);
        for (int i = 0; i < 16; i++) {
            code32[i] = p_cast[i];
        }
    }
};

template <>
struct LoadCode32<96> {
    static inline __device__ void load(
            unsigned int code32[24],
            uint8_t* p,
            int offset) {
        p += offset * 96;
        // FIXME: this is a non-coalesced load
        // unfortunately need to reorganize memory layout by warp
        unsigned int* p_cast = reinterpret_cast<unsigned int*>(p);
        for (int i = 0; i < 24; i++) {
            code32[i] = p_cast[i];
        }
    }
};

} // namespace gpu
} // namespace faiss
