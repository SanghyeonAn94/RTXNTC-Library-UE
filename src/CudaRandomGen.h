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
#include <cuda_fp16.h>
#include <cstdint>

namespace ntc
{

class CudaRandomGen
{
public:
    CudaRandomGen() = default;

    uint32_t GetSeed() const { return m_seed; }

    void SetSeed(uint32_t seed) { m_seed = seed; }

    void RandomizeSeed();
    
    // Fills the buffer with uniform random numbers in the range [0, UINT_MAX].
    void FillRandomUint(
        uint32_t* buffer,
        uint32_t length);

    // Fills the buffers with random numbers drawn from a normal (Gaussian) distribution
    // centered at 0 with sigma = 1.0, scaled by 'scale' and offset by 'bias', truncated to [minValue, maxValue].
    void FillRandomNormalHalf(
        half* buffer,
        uint32_t length,
        float scale,
        float bias,
        float minValue,
        float maxValue);
    
private:
    
    uint32_t m_seed = 0;
};

}