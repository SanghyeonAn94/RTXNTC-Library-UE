/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#if defined(__CUDACC__)

#include "tin_common.h"
#include <mma.h>

namespace tin {

    // Using inline PTX as using tiled group to get lane ID fails in Optix
    __forceinline__ __device__ unsigned _lane_id() {
        unsigned ret;
        asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
        return ret;
    }

    // Using inline PTX as using tiled group to get lane ID fails in Optix
    __forceinline__ __device__ unsigned _warp_id() {
        uint32_t thread_rank = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
        return thread_rank / warpSize;
    }

    // Using inline PTX as movm is not exposed in CUDA
    __forceinline__ __device__ uint32_t _transpose(uint32_t in) {
        uint32_t ret;
        asm ("movmatrix.sync.trans.aligned.m8n8.b16 %0, %1;" : "=r"(ret) : "r"(in));
        return ret;
    }

    // Using inline PTX to avoid branches introduced with CUDA atomics for detecting shared memory atomics
    __forceinline__ __device__ void _atomic_addh2(half2* addr, half2 in) {
        int in_int = *((int*)&in);
        asm (
            "red.relaxed.gpu.global.add.noftz.f16x2 [%0], %1;" :: "l"(addr), "r"(in_int));
    }

    __forceinline__ __device__ void _atomic_addf(float* addr, float in) {
        int in_int = *((int*)&in);
        asm (
            "red.relaxed.gpu.global.add.f32 [%0], %1;" :: "l"(addr), "r"(in_int));
    }

}

#endif
