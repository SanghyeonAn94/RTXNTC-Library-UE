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

#ifndef NTC_INFERENCE_CONSTANTS_H
#define NTC_INFERENCE_CONSTANTS_H

#ifdef __cplusplus
#define NTC_UINT unsigned int
#else
#define NTC_UINT uint
#endif

#define NTC_MAX_MIPS        16
#define NTC_MAX_CHANNELS    16
#define NTC_MAX_NEURAL_MIPS 8

#define NTC_NETWORK_UNKNOWN 0
#define NTC_NETWORK_SMALL   1
#define NTC_NETWORK_MEDIUM  2
#define NTC_NETWORK_LARGE   3
#define NTC_NETWORK_XLARGE  4
#define NTC_NETWORK_COUNT   (NTC_NETWORK_XLARGE)

#define NTC_MLP_LAYERS                4
#define NTC_MLP_HR_FEATURES_SMALL     4
#define NTC_MLP_HR_FEATURES_MEDIUM    8
#define NTC_MLP_HR_FEATURES_LARGE     12
#define NTC_MLP_HR_FEATURES_XLARGE    16
#define NTC_MLP_LR_FEATURES           16
#define NTC_MLP_POS_ENC_SCALE         8
#define NTC_MLP_INPUT_CHANNELS_SMALL  48
#define NTC_MLP_INPUT_CHANNELS_MEDIUM 64
#define NTC_MLP_INPUT_CHANNELS_LARGE  80
#define NTC_MLP_INPUT_CHANNELS_XLARGE 96
#define NTC_MLP_HIDDEN_CHANNELS       64
#define NTC_MLP_OUTPUT_CHANNELS       16

struct NtcLatentEncodingConstants
{
    int numFeatures;
    int quantBits;
    int logElementsPerUint; // == log2(32 / quantBits)
    int pad;

    NTC_UINT addressMask; // == (1u << logElementsPerUint) - 1u
    NTC_UINT dataMask;    // == (1u << quantBits) - 1u
    int quantizedScale;   // scale and bias parameters to translate quantized latent to quantized input
    int quantizedBias;
};

struct NtcNeuralMipConstants
{
    NTC_UINT dataOffset;
#ifdef __cplusplus
    unsigned short imageWidth;
    unsigned short imageHeight;
    unsigned short sliceLeft;
    unsigned short sliceTop;
    unsigned short sliceWidth;
    unsigned short sliceHeight;
#else
    // Use 32-bit uints in HLSL to avoid compatibility issues.
    // The struct's binary representation doesn't matter on HLSL side.
    NTC_UINT sliceLeft;
    NTC_UINT sliceTop;
    NTC_UINT sliceWidth;
    NTC_UINT sliceHeight;
    NTC_UINT imageWidth;
    NTC_UINT imageHeight;
#endif
};

struct NtcColorMipConstants
{
    int neuralMip;
    float positionLod;
    float positionScale;
    int pad;
};

struct NtcTextureSetConstants
{
#ifdef __cplusplus
    NtcLatentEncodingConstants highResEncoding;
    NtcLatentEncodingConstants lowResEncoding;
    NtcNeuralMipConstants highResNeuralMips[NTC_MAX_NEURAL_MIPS];
    NtcNeuralMipConstants lowResNeuralMips[NTC_MAX_NEURAL_MIPS];
    NtcColorMipConstants colorMips[NTC_MAX_MIPS];
    int networkWeightOffsets[NTC_MLP_LAYERS];
#else
    // These structures are packed as int4 for compatibility with engines
    // that don't support structs in constant buffers.
    int4 highResEncoding[2];
    int4 lowResEncoding[2];
    int4 highResNeuralMips[NTC_MAX_NEURAL_MIPS];
    int4 lowResNeuralMips[NTC_MAX_NEURAL_MIPS];
    int4 colorMips[NTC_MAX_MIPS];
    
    // This maps to the int[] array on the host side
    // but doesn't require 16-byte alignment for each element.
    int4 networkWeightOffsets;
#endif

    int imageWidth;
    int imageHeight;
    int imageMips;    
    int networkScaleBiasOffset;

    NTC_UINT validChannelMask;
    NTC_UINT channelColorSpaces; // Packed with 2 bits for each channel
    int pad1;
    int pad2;
};

#endif
