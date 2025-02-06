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

namespace ntc
{

// The MlpDesc structure describes the geometry of the MLP used to decode neural textures.
// There are only a few versions of the MLP defined in the library, described by the
// NTC_NETWORK_(SMALL..XLARGE) constants, and there is a corresponding table in MlpDesc.cpp
// listing the parameters of each network size.
struct MlpDesc
{
    int networkVersion;
    int highResFeatures;
    int lowResFeatures;
    int inputChannels;
    
    int GetInputChannels() const { return inputChannels; }
    int GetHiddenChannels() const;
    int GetOutputChannels() const;
    int GetHiddenLayers() const;

    // Returns the total number of weights in all layers.
    int GetWeightCount() const;

    // Returns the total number of outputs from all layers.
    int GetLayerOutputCount() const;

    // Returns the number of inputs for a specific layer by index.
    int GetLayerInputChannels(int layer) const;

    // Returns the number of outputs for a specific layer by index.
    int GetLayerOutputChannels(int layer) const;

    // Finds the MLP version corresponding to a NTC_NETWORK_... constant.
    // If networkVersion has an unsupported value, returns nullptr.
    static MlpDesc const* FromNetworkVersion(int networkVersion);

    // Finds the MLP version with exactly matching feature counts.
    // If no such version found, returns nullptr.
    static MlpDesc const* FromFeatureCounts(int highResFeatures, int lowResFeatures);

    // Finds the MLP version matching input channel count.
    // If no such version found, returns nullptr.
    static MlpDesc const* FromInputChannels(int inputChannels);

    // Finds the smallest MLP version that can fit the specified feature counts.
    // If no such version found, returns nullptr.
    static MlpDesc const* PickOptimalConfig(int highResFeatures, int lowResFeatures);
};

}