/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstddef>

namespace ntc
{

class FeatureGridMath
{
public:
    enum class Grid
    {
        HighRes,
        LowRes
    };

    static constexpr int NeuralMipRatio = 2;

    static size_t CalculateQuantizedLatentsSize(int imageWidth, int imageHeight, int imageMips, int highResGridScale,
        int highResFeatures, int lowResFeatures, int highResQuantBits, int lowResQuantBits);

    static int GetQuantizedLatentSizeUints(int num_weights, int quant_bits);

    static int LodToNeuralLod(int lod, int highResGridScale, int neuralLods);

    static int NeuralLodToColorLod(int neuralLod, int highResGridScale);

    static int GetGridDimension(Grid grid, int imageDimension, int neuralLod, int highResScale);

    static void GetPositionLodAndScale(int neuralLod, int mipLevel, float& outPositionLod, float& outPositionScale);

protected:
    static int CalculateNumNeuralMipLevels(int imageWidth, int imageHeight, int highResGridScale);

    static int CalculateNumLatentsInNeuralMip(Grid grid, int imageWidth, int imageHeight, int highResGridScale, int mip);
};

}