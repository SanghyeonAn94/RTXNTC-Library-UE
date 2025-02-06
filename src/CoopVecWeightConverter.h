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

#include <cstdint>
#include <cstdlib>
#include <libntc/shaders/InferenceConstants.h>

namespace ntc
{

class GraphicsResources;

class CoopVecWeightConverter
{
public:
    CoopVecWeightConverter(GraphicsResources const* resources, bool useFP8, int inputChannels, int hiddenChannels,
        int outputChannels, int numHiddenLayers);

    bool IsConversionSupported() const { return m_isSupported; }
    size_t GetConvertedWeightSize() const { return m_dstTotalSize; }
    size_t GetSourceWeightSize() const { return m_srcTotalSize; }

    void ConvertWeights(const uint8_t* src, uint8_t* dst);
    void GetConvertedWeightOffsets(int weightOffsets[NTC_MLP_LAYERS]);

    // These are mostly just NVAPI function wrappers that let us avoid including nvapi.h
    // into other source files (makes Intellisense slow)
#if NTC_WITH_DX12
    static bool InitializeNVAPI();
    static void IsDX12CoopVecSupported(GraphicsResources const* resources, bool& outInt8Supported, bool& outFP8Supported);
    static void UnloadNVAPI();
#endif
#if NTC_WITH_VULKAN
    static void IsVkCoopVecSupported(GraphicsResources const* resources, bool& outInt8Supported, bool& outFP8Supported);
#endif

private:
    GraphicsResources const* m_resources;
    bool m_useFP8;
    bool m_isSupported;
    int m_inputChannels = 0;
    int m_hiddenChannels = 0;
    int m_outputChannels = 0;
    int m_numHiddenLayers = 0;
    size_t m_srcTotalSize = 0;
    size_t m_dstTotalSize = 0;
    size_t m_dstWeightSizeInput = 0;
    size_t m_dstWeightSizeHidden = 0;
    size_t m_dstWeightSizeOutput = 0;

    void CalculateOutputSizes();
};

}