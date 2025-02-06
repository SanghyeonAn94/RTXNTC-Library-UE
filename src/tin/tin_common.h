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
#define TIN_DEVICE      __device__
#define TIN_DEVICE_HOST __device__ __host__
#define TIN_HOST        __host__

    #if defined(__CUDACC_RTC__)
    #  define TIN_UNROLL      _Pragma("unroll")
    #else
    #  define TIN_UNROLL      #pragma unroll
    #endif
#else
#define TIN_DEVICE
#define TIN_DEVICE_HOST
#define TIN_HOST
#define TIN_UNROLL
#endif

#include <cuda_fp16.h>
#include <cuda.h>

namespace tin {

    template<typename T>
    using PackedType = typename
        std::conditional<
            std::is_same<T, half>::value, half2,
            std::conditional<
                std::is_same<T, char>::value, char4,
                std::conditional<
                    std::is_same<T, unsigned char>::value, uchar4, T
                >
            >
        >::type;

    template<typename T>
    TIN_DEVICE inline PackedType<T> unpack(uint32_t x) {
        return *((PackedType<T> *)(&x));
    }

    template<typename T>
    TIN_DEVICE inline uint32_t pack(PackedType<T> x) {
        return *((uint32_t*)(&x));
    }

    template <class T>
    constexpr TIN_DEVICE_HOST int num_packed() {

        if (std::is_same<T, signed char>::value |
            std::is_same<T, unsigned char>::value) {
            return 4;
        }
        else if (std::is_same<T, half>::value) {
            return 2;
        }
        else {
            return 1;
        }
    };

    static const int WarpSize = 32;

}
