/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda.h>

namespace faiss {
namespace gpu {

template <typename T>
__device__ __forceinline__ T getBitfield(T val, int pos, int len) {
    T mask = (static_cast<T>(1) << len) - 1;
    return (val >> pos) & mask;
}

} // namespace gpu
} // namespace faiss
