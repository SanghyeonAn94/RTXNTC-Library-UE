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
#include "tin_common.h"
#include <cuda_fp8.h>

namespace tin {

    enum class Quantization
    {
        None,
        Int8,
        FP8
    };

    struct ActNone {
        TIN_DEVICE ActNone(Quantization quant = Quantization::None) {};
        TIN_DEVICE half forward(half x) { return x; };
        TIN_DEVICE half backward(half x_d, half x) { return x_d; };
    };

    struct ActReLU {
        TIN_DEVICE ActReLU(Quantization quant = Quantization::None) {};
        TIN_DEVICE half forward(half x) { return std::max(x, half(0)); };
        TIN_DEVICE half backward(half x_d, half x) { return x > half(0) ? x_d : half(0); };
    };

    struct ActHGELU {
        TIN_DEVICE ActHGELU(Quantization quant = Quantization::None) {};
        TIN_DEVICE half forward(half x) { return x * __hmin(__hmax(x + half(1.5f), 0.f), 3.f) * half(1 / 3.f); };
        TIN_DEVICE half backward(half x_d, half x) { return __hgt(x, -1.5f) ? __hlt(x, 1.5f) ? half(0.5f) * x_d + half(2.f / 3.f) * x * x_d : x_d : half(0.f); };
    };

    inline __device__ half RoundHalfToFloatE4M3(half x)
    {
    #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
        // Use hardware conversion on SM8.9 or above
        return half(__nv_fp8_e4m3(x));
    #else
        // The CUDA software implementation of half<->fp8 conversions goes through double and is very slow.
        // Use a custom one based on the bit representations of these floats, and optimize it by not actually encoding
        // FP8 numbers because we only need to round to that precision.

        // Get the binary representation of the half
        uint16_t bits = __half_raw(x).x;

        // Sign bit
        uint16_t sign = bits & 0x8000;
        
        // Mantissa bits that don't fit into fp8
        uint16_t remainder = bits & 0x7f;
        
        // Remove the sign and the remainder
        bits &= 0x7f80;

        // Round to nearest, ties to even:
        // If the remainder is more than 0.5, round up.
        // If the remainder is exactly 0.5, and the bit above it is 1, round up.
        if (remainder > 0x40 || remainder == 0x40 && (bits & 0x80) != 0)
            bits += 0x80;

        // Clamp to 448.0 which is the max representable value in FP8-E4M3
        bits = std::min(bits, uint16_t(0x5f00));
        
        // Restore the sign bit
        bits |= sign;

        // Convert the bits back into the half type
        return half(__half_raw{bits});
    #endif
    }

    // Clamped HGELU for int8 quantization
    struct ActHGELUClamp {

        TIN_DEVICE ActHGELUClamp(Quantization quant = Quantization::None) : quant(quant) {};

        TIN_DEVICE half forward(half x) { 

            half y = __hmin(x, pos_lim) * __hfma_sat(half(1 / 3.f), x, half(0.5f));

            if (quant == Quantization::Int8)
            {
                y = y * half(invStep);
                y = hrint(y);
                y = y * half(step);
            }
            else if (quant == Quantization::FP8)
            {
                y = RoundHalfToFloatE4M3(y);
            }

            return y;
        };

        TIN_DEVICE half backward(half x_d, half x) { 
            half dy = __hgt(x, -1.5f) ? __hlt(x, 1.5f) ? __hfma(half(2.f / 3.f), x, half(0.5f)) * x_d : x_d : half(0.f);
            dy = __hgt(x, pos_lim) ? half(0.f) : dy;
            return dy;
        };

        static constexpr float minval = -3.f / 16;
        static constexpr float maxval = 3.f;

        static constexpr int bins = 256;
        static constexpr float step = (maxval - minval) / float(bins - 1);
        static constexpr float invStep = 1.f / step;
        static constexpr int qmax = int(maxval / step);
        static constexpr int qmin = qmax - bins + 1;
        static constexpr float pos_lim = qmax * step;
        static constexpr int bias = -(bins / 2) - qmin;

        Quantization quant;
    };
}
