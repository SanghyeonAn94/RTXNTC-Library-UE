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

#define USE_COOPVEC

#include "DecompressCommon.hlsli"
#include "libntc/shaders/InferenceCoopVec.hlsli"

groupshared uint32_t s_Latents[LATENTS_COUNT / 4][HR_LATENTS_HEIGHT][HR_LATENTS_WIDTH];

#define TOTAL_BIAS_CHANNELS (Params::HIDDEN_LAYER_CHANNELS * 3 + Params::OUTPUT_CHANNELS)

groupshared float s_Scale[TOTAL_BIAS_CHANNELS];
groupshared float s_Outputs[DECOMPRESS_CS_BLOCK_HEIGHT * DECOMPRESS_CS_BLOCK_WIDTH * (Params::OUTPUT_CHANNELS + 4)];

void PreloadLatents<let NUM_FEATURES: int>(
    NtcLatentEncodingConstants encoding,
    NtcNeuralMipConstants neuralMip,
    float2 colorToNeuralScale,
    int2 baseLatentPos,
    int latentOffset,
    int2 threadIndex)
{
    // Rename the threads into a 2D group of a different size, iterate over partitions of that group
    // if the original group size is smaller.
    const int groupWidth = int(ceil(float(DECOMPRESS_CS_BLOCK_WIDTH) * colorToNeuralScale.x)) + PRELOAD_MARGIN;
    const int groupHeight = int(ceil(float(DECOMPRESS_CS_BLOCK_HEIGHT) * colorToNeuralScale.y)) + PRELOAD_MARGIN;
    int linearThreadIndex = threadIndex.x + threadIndex.y * DECOMPRESS_CS_BLOCK_WIDTH;
    while (linearThreadIndex < groupWidth * groupHeight)
    {
        const int2 renamedThreadIdx = int2(linearThreadIndex % groupWidth, linearThreadIndex / groupWidth);
        const int2 sliceOrigin = int2(neuralMip.sliceLeft, neuralMip.sliceTop);
        const int2 sliceSize = int2(neuralMip.sliceWidth, neuralMip.sliceHeight);
        const int2 latentPos = clamp(baseLatentPos + renamedThreadIdx - sliceOrigin, 0, sliceSize - 1);
        int addr = (latentPos.y * neuralMip.sliceWidth + latentPos.x) * encoding.numFeatures;
    
        [unroll]
        for (int i = 0; i < NUM_FEATURES / 4; i++)
        {
            int4 inp = NtcLoadFourInputQuantizedLatents(t_InputFile, 0, encoding, neuralMip, addr);
            addr += 4;
        
            s_Latents[latentOffset + i][renamedThreadIdx.y][renamedThreadIdx.x] = NtcPackInt8x4(inp);
        }
        linearThreadIndex += DECOMPRESS_CS_BLOCK_WIDTH * DECOMPRESS_CS_BLOCK_HEIGHT;
    }
}

void SampleLatentGridShared<let NUM_FEATURES: int, let ALL_CORNERS: bool>(
    NtcLatentEncodingConstants encoding,
    NtcNeuralMipConstants neuralMip,
    float2 uv,
    int2 baseLatentPos,
    int latentOffset,
    int outputOffset,
    inout uint32_t[Params::INPUT_CHANNELS / 4] outputArray)
{
    int2 topLeftPos;
    float4 weights;
    NtcSetupLatentBilinearFilter(neuralMip, uv, topLeftPos, weights);
    int4 iweights = int4(weights * 256.f);
    
    // Shift right the interpolated weights by 8 to undo the 256 factor above
    const int normalizationShift = 8;

    const int2 sharedPos = topLeftPos - baseLatentPos;
    
    [unroll]
    for (int i = 0; i < NUM_FEATURES / 4; i++)
    {
        if (i >= encoding.numFeatures / 4)
            break;

        const uint32_t u00 = s_Latents[latentOffset + i][sharedPos.y    ][sharedPos.x    ];
        const uint32_t u01 = s_Latents[latentOffset + i][sharedPos.y    ][sharedPos.x + 1];
        const uint32_t u10 = s_Latents[latentOffset + i][sharedPos.y + 1][sharedPos.x    ];
        const uint32_t u11 = s_Latents[latentOffset + i][sharedPos.y + 1][sharedPos.x + 1];

        // Unpack the latents into int4's for blending and multiply by weights.
        const int4 x00 = NtcUnpackInt8x4(u00) * iweights.x;
        const int4 x01 = NtcUnpackInt8x4(u01) * iweights.y;
        const int4 x10 = NtcUnpackInt8x4(u10) * iweights.z;
        const int4 x11 = NtcUnpackInt8x4(u11) * iweights.w;

        if (ALL_CORNERS)
        {
            // Copy the latents for the 4 pixels into the network inputs.
            outputArray[outputOffset + (NUM_FEATURES / 4) * 0 + i] = NtcPackInt8x4(x00 >> normalizationShift);
            outputArray[outputOffset + (NUM_FEATURES / 4) * 1 + i] = NtcPackInt8x4(x01 >> normalizationShift);
            outputArray[outputOffset + (NUM_FEATURES / 4) * 2 + i] = NtcPackInt8x4(x10 >> normalizationShift);
            outputArray[outputOffset + (NUM_FEATURES / 4) * 3 + i] = NtcPackInt8x4(x11 >> normalizationShift);
        }
        else
        {
            // Blend the features of the 4 pixels using integer weights.
            int4 d = (x00 + x01 + x10 + x11) >> normalizationShift;
            outputArray[outputOffset + i] = NtcPackInt8x4(d);
        }
    }
}

inline void EvaluateLayer_CoopVec_Shared<T_IN: __BuiltinArithmeticType, let T_IN_NUM: int, let IN_IS_PACKED: bool, let IN: int, let OUT: int, let ACT: bool>(
    int weightOffset, inout uint scaleOffset, inout uint biasOffset, in CoopVec<T_IN, T_IN_NUM> inputArray, out CoopVec<float, OUT> outputArray)
{
    NtcEvaluateLayerMatMulAdd_CoopVec_Int8<T_IN, T_IN_NUM, IN_IS_PACKED, IN, OUT>
        (t_WeightBuffer, weightOffset, biasOffset, inputArray, outputArray);

    let scale = CoopVec<float, OUT>.load(s_Scale, scaleOffset);

    outputArray = outputArray * scale;

    if (ACT)
    {
        NtcHGELUClamp_Forward_CoopVec(outputArray, true);
    }

    // Advance the input offset to point at the next layer.
    scaleOffset += OUT * sizeof(float);
    biasOffset += OUT * sizeof(float);
}

void DecompressPixel_CoopVec(uint2 globalIndex, uint2 threadIndex)
{
    const int2 pixelPosition = int2(globalIndex) + int2(g_Const.gridLeft, g_Const.gridTop);
    const int2 dstPosition = pixelPosition + int2(g_Const.dstLeft - g_Const.srcLeft, g_Const.dstTop - g_Const.srcTop);
    const NtcColorMipConstants colorMip = NtcUnpackColorMipConstants(g_Const.colorMip);
    const float2 colorMipSize = float2(g_Const.imageWidth, g_Const.imageHeight);
    const NtcNeuralMipConstants highResNeuralMip = NtcUnpackNeuralMipConstants(g_Const.highResNeuralMip);
    const NtcNeuralMipConstants lowResNeuralMip = NtcUnpackNeuralMipConstants(g_Const.lowResNeuralMip);

#if PRELOAD_LATENTS
    // Preload the block of latents needed to decompress all pixels in this thread group
    const float2 highResNeuralMipSize = float2(highResNeuralMip.imageWidth, highResNeuralMip.imageHeight);
    const float2 lowResNeuralMipSize = float2(lowResNeuralMip.imageWidth, lowResNeuralMip.imageHeight);
    const float2 highResNeuralScale = highResNeuralMipSize / colorMipSize;
    const float2 lowResNeuralScale = lowResNeuralMipSize / colorMipSize;

    const float2 groupBase = float2(pixelPosition - threadIndex) + 0.5;
    const int2 baseHighResLatentPos = int2(floor(groupBase * highResNeuralScale)) - 1;
    const int2 baseLowResLatentPos = int2(floor(groupBase * lowResNeuralScale)) - 1;

    PreloadLatents<Params::HR_FEATURES>(NtcUnpackLatentEncodingConstants(g_Const.highResEncoding),
        highResNeuralMip, highResNeuralScale,
        baseHighResLatentPos, 0, threadIndex);
    PreloadLatents<Params::LR_FEATURES>(NtcUnpackLatentEncodingConstants(g_Const.lowResEncoding),
        lowResNeuralMip, lowResNeuralScale,
        baseLowResLatentPos, Params::HR_FEATURES / 4, threadIndex);
#endif

    // Preload the scale values into shared memory.
    int linearThreadIndex = threadIndex.x + threadIndex.y * DECOMPRESS_CS_BLOCK_WIDTH;

    while (linearThreadIndex < TOTAL_BIAS_CHANNELS)
    {
        float scale = t_WeightBuffer.Load<float>(g_Const.networkScaleBiasOffset + linearThreadIndex * sizeof(float));

        s_Scale[linearThreadIndex] = scale;
        linearThreadIndex += (DECOMPRESS_CS_BLOCK_WIDTH * DECOMPRESS_CS_BLOCK_HEIGHT);
    }

    GroupMemoryBarrierWithGroupSync();

    uint networkInputs[Params::INPUT_CHANNELS / 4];

    // Zero init the array - in some cases, INPUT_CHANNELS is rounded up from the actual used size.
    [unroll]
    for (int i = 0; i < Params::INPUT_CHANNELS / 4; ++i)
        networkInputs[i] = 0;

    int inputOffset = 0;
    const float2 uv = (float2(pixelPosition)+0.5) / float2(g_Const.imageWidth, g_Const.imageHeight);

#if PRELOAD_LATENTS
    // Sample the latent grids from preloaded data
    SampleLatentGridShared<Params::HR_FEATURES, true>(NtcUnpackLatentEncodingConstants(g_Const.highResEncoding),
        highResNeuralMip, uv, baseHighResLatentPos, 0, inputOffset, networkInputs);
    inputOffset += Params::SAMPLED_FEATURES_HR / 4;

    SampleLatentGridShared<Params::LR_FEATURES, false>(NtcUnpackLatentEncodingConstants(g_Const.lowResEncoding),
        lowResNeuralMip, uv, baseLowResLatentPos, Params::HR_FEATURES / 4, inputOffset, networkInputs);
    inputOffset += Params::SAMPLED_FEATURES_LR / 4;
#else
    NtcSampleLatentGrid<Params::HR_FEATURES, true>(t_InputFile, 0, NtcUnpackLatentEncodingConstants(g_Const.highResEncoding),
        highResNeuralMip, uv, inputOffset, networkInputs);
    inputOffset += Params::SAMPLED_FEATURES_HR / 4;

    NtcSampleLatentGrid<Params::LR_FEATURES, false>(t_InputFile, 0, NtcUnpackLatentEncodingConstants(g_Const.lowResEncoding),
        lowResNeuralMip, uv, inputOffset, networkInputs);
    inputOffset += Params::SAMPLED_FEATURES_LR / 4;
#endif

    // Encode the sample position
    NtcEncodeSamplePosition(float2(pixelPosition) * colorMip.positionScale,
        colorMip.positionLod, inputOffset, networkInputs);

    CoopVec<uint32_t, Params::INPUT_CHANNELS / 4> networkInputsVec;
    [unroll]
    for (int i = 0; i < Params::INPUT_CHANNELS / 4; ++i)
        networkInputsVec[i] = networkInputs[i];

    int scaleOffset = 0;
    int biasOffset = g_Const.networkScaleBiasOffset + TOTAL_BIAS_CHANNELS * sizeof(float);

    // Evaluate the MLP layers:
    // Input layer.  Inputs are packed i.e. 4 int8s in a uint.
    CoopVec<float, Params::HIDDEN_LAYER_CHANNELS> hiddenOutput1;
    EvaluateLayer_CoopVec_Shared<uint32_t, Params::INPUT_CHANNELS/4, true, Params::INPUT_CHANNELS, Params::HIDDEN_LAYER_CHANNELS, true>
    (g_Const.networkWeightOffsets.x, scaleOffset, biasOffset, networkInputsVec, hiddenOutput1);
    
    // Hidden layer 1. Inputs are float and converted internally to int8 by rounding. 
    CoopVec<float, Params::HIDDEN_LAYER_CHANNELS> hiddenOutput2;
    EvaluateLayer_CoopVec_Shared<float, Params::HIDDEN_LAYER_CHANNELS, false, Params::HIDDEN_LAYER_CHANNELS, Params::HIDDEN_LAYER_CHANNELS, true>
    (g_Const.networkWeightOffsets.y, scaleOffset, biasOffset, hiddenOutput1, hiddenOutput2);
    
    // Hidden layer 2. Inputs are float and converted internally to int8 by rounding. 
    CoopVec<float, Params::HIDDEN_LAYER_CHANNELS> hiddenOutput3;
    EvaluateLayer_CoopVec_Shared<float, Params::HIDDEN_LAYER_CHANNELS, false, Params::HIDDEN_LAYER_CHANNELS, Params::HIDDEN_LAYER_CHANNELS, true>
    (g_Const.networkWeightOffsets.z, scaleOffset, biasOffset, hiddenOutput2, hiddenOutput3);
    
    // Output layer. Inputs are float and converted internally to int8 by rounding. 
    CoopVec<float, Params::OUTPUT_CHANNELS> networkOutputs;
    EvaluateLayer_CoopVec_Shared<float, Params::HIDDEN_LAYER_CHANNELS, false, Params::HIDDEN_LAYER_CHANNELS, Params::OUTPUT_CHANNELS, false>
    (g_Const.networkWeightOffsets.w, scaleOffset, biasOffset, hiddenOutput3, networkOutputs);

    // Store the outputs into shared memory for efficient indexed access later.
    // Note: there is no need for a barrier before or after this store because we're using a dedicated
    // shared memory array, and each thread only reads the data it's written - nothing from other threads.
    // Also note: the ordering of array dimensions and (+4) to the channel count are optimized
    // for efficient shared memory access patterns.
    int threadOffset = (threadIndex.y * DECOMPRESS_CS_BLOCK_WIDTH + threadIndex.x) * (Params::OUTPUT_CHANNELS + 4);
    networkOutputs.store(s_Outputs, threadOffset * sizeof(float));

    HashBasedRNG rng = HashBasedRNG::Create(pixelPosition.x + pixelPosition.y * g_Const.imageWidth, 0);

    // Exit if this pixel is outside of the specified rectangle
    if (pixelPosition.x < g_Const.srcLeft || pixelPosition.y < g_Const.srcTop ||
        pixelPosition.x >= g_Const.srcRight || pixelPosition.y >= g_Const.srcBottom)
        return;
    
    // Shuffle the output data into destination textures
    for (int outputIndex = 0; outputIndex < g_Const.numOutputs; ++outputIndex)
    {
        const NtcDecompressOutputDesc outputDesc = g_Const.outputs[outputIndex];
        
        // Read 4 channels from the network output
        float4 texelValue;
        int firstOffset = threadOffset + outputDesc.firstChannel;

        texelValue.r = s_Outputs[firstOffset];
        if (outputDesc.numChannels >= 2)
            texelValue.g = s_Outputs[++firstOffset];
        if (outputDesc.numChannels >= 3)
            texelValue.b = s_Outputs[++firstOffset];
        if (outputDesc.numChannels == 4)
            texelValue.a = s_Outputs[++firstOffset];

        // Perform color space conversion, if needed
        texelValue.rgb = NtcConvertColorSpace(texelValue.rgb, outputDesc.srcRgbColorSpace, outputDesc.dstRgbColorSpace);
        texelValue.a = NtcConvertColorSpace(texelValue.a, outputDesc.srcAlphaColorSpace, outputDesc.dstAlphaColorSpace);
        
        // Apply dithering. Making the loop conditional on ditherScale makes the shader slower,
        // so just multiply by 0 if dithering is not needed.
        float4 dither = (rng.Next4LowPrecisionFloats() - 0.5f) * outputDesc.ditherScale;
        texelValue += dither;

        // If fewer than 4 channels are requested, set the remaining ones to default values
        if (outputDesc.numChannels <= 1) texelValue.y = 0;
        if (outputDesc.numChannels <= 2) texelValue.z = 0;
        if (outputDesc.numChannels <= 3) texelValue.w = 1;

        // Write out the texel to the UAV
        u_Outputs[outputDesc.textureIndex][dstPosition] = texelValue;
    }
}

[numthreads(DECOMPRESS_CS_BLOCK_WIDTH, DECOMPRESS_CS_BLOCK_HEIGHT, 1)]
void main(uint2 globalIndex : SV_DispatchThreadID, uint2 threadIndex : SV_GroupThreadID)
{
    DecompressPixel_CoopVec(globalIndex, threadIndex);
}
