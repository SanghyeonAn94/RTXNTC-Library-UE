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
#include <algorithm>
#include <math.h>

struct QuantizationParameters
{
    float scale;
    float step;
    float bias;
    float qmin;
    float qmax;
};

#ifdef _NV_COMPILER_NVCC
__host__ __device__
#endif
inline QuantizationParameters GetLatentQuantization(int bits)
{
    QuantizationParameters params{};
    // Note: the 127.5 limit is there to make 8-bit latent quantization compatible with input quantization,
    // which also uses 127.5 scale. See FillLatentEncodingConstants(...) for more info.
    params.scale = std::min(float(1 << (bits - 1)) + 0.5f, 127.5f);
    params.step = 1.f / params.scale;
    params.bias = (1.5f - params.scale) * params.step;
    params.qmin = nextafterf(-params.scale + 1.f, 0.f);
    params.qmax = nextafterf(params.scale, 0.f);
    return params;
}
