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


groupshared float16_t4 s_Latents[LATENTS_COUNT / 4][HR_LATENTS_HEIGHT][HR_LATENTS_WIDTH];

groupshared float s_Outputs[DECOMPRESS_CS_BLOCK_HEIGHT * DECOMPRESS_CS_BLOCK_WIDTH * Params::OUTPUT_CHANNELS];

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
            float16_t4 inp = NtcLoadFourInputQuantizedLatents_FP16(t_InputFile, 0, encoding, neuralMip, addr);
            addr += 4;
        
            s_Latents[latentOffset + i][renamedThreadIdx.y][renamedThreadIdx.x] = inp;
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
    inout CoopVec<float16_t, Params::INPUT_CHANNELS> outputArray)
{
    int2 topLeftPos;
    float4 weights;
    NtcSetupLatentBilinearFilter(neuralMip, uv, topLeftPos, weights);
    float16_t4 iweights = float16_t4(weights);
    
    // Shift right the interpolated weights by 8 to undo the 256 factor above
    const int normalizationShift = 8;

    const int2 sharedPos = topLeftPos - baseLatentPos;
    
    [unroll]
    for (int i = 0; i < NUM_FEATURES / 4; i++)
    {
        if (i >= encoding.numFeatures / 4)
            break;

        const float16_t4 u00 = s_Latents[latentOffset + i][sharedPos.y    ][sharedPos.x    ];
        const float16_t4 u01 = s_Latents[latentOffset + i][sharedPos.y    ][sharedPos.x + 1];
        const float16_t4 u10 = s_Latents[latentOffset + i][sharedPos.y + 1][sharedPos.x    ];
        const float16_t4 u11 = s_Latents[latentOffset + i][sharedPos.y + 1][sharedPos.x + 1];

        const float16_t4 x00 = u00 * iweights.x;
        const float16_t4 x01 = u01 * iweights.y;
        const float16_t4 x10 = u10 * iweights.z;
        const float16_t4 x11 = u11 * iweights.w;

        if (ALL_CORNERS)
        {
            // Copy the latents for the 4 pixels into the network inputs.
            NtcCoopVecStoreHalf4(outputArray, outputOffset + i * 4 + NUM_FEATURES * 0, x00);
            NtcCoopVecStoreHalf4(outputArray, outputOffset + i * 4 + NUM_FEATURES * 1, x01);
            NtcCoopVecStoreHalf4(outputArray, outputOffset + i * 4 + NUM_FEATURES * 2, x10);
            NtcCoopVecStoreHalf4(outputArray, outputOffset + i * 4 + NUM_FEATURES * 3, x11);
        }
        else
        {
            // Blend the features of the 4 pixels using integer weights.
            float16_t4 d = x00 + x01 + x10 + x11;
            NtcCoopVecStoreHalf4(outputArray, outputOffset + i * 4, d);
        }
    }
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

    GroupMemoryBarrierWithGroupSync();
    
    CoopVec<float16_t, Params::INPUT_CHANNELS> networkInputs;

    int inputOffset = 0;
    const float2 uv = (float2(pixelPosition)+0.5) / float2(g_Const.imageWidth, g_Const.imageHeight);

#if PRELOAD_LATENTS
    // Sample the latent grids from preloaded data
    SampleLatentGridShared<Params::HR_FEATURES, true>(NtcUnpackLatentEncodingConstants(g_Const.highResEncoding),
        highResNeuralMip, uv, baseHighResLatentPos, 0, inputOffset, networkInputs);
    inputOffset += Params::SAMPLED_FEATURES_HR;

    SampleLatentGridShared<Params::LR_FEATURES, false>(NtcUnpackLatentEncodingConstants(g_Const.lowResEncoding),
        lowResNeuralMip, uv, baseLowResLatentPos, Params::HR_FEATURES / 4, inputOffset, networkInputs);
    inputOffset += Params::SAMPLED_FEATURES_LR;
#else
    NtcSampleLatentGrid_FP16<Params::HR_FEATURES, true>(t_InputFile, 0, NtcUnpackLatentEncodingConstants(g_Const.highResEncoding),
        highResNeuralMip, uv, inputOffset, networkInputs);
    inputOffset += Params::SAMPLED_FEATURES_HR;

    NtcSampleLatentGrid_FP16<Params::LR_FEATURES, false>(t_InputFile, 0, NtcUnpackLatentEncodingConstants(g_Const.lowResEncoding),
        lowResNeuralMip, uv, inputOffset, networkInputs);
    inputOffset += Params::SAMPLED_FEATURES_LR;
#endif

    // Encode the sample position
    NtcEncodeSamplePosition_FP16(float2(pixelPosition) * colorMip.positionScale,
        colorMip.positionLod, inputOffset, networkInputs);

    int scaleBiasOffset = g_Const.networkScaleBiasOffset;

    // Evaluate the MLP layers:
    // Input layer
    CoopVec<float16_t, Params::HIDDEN_LAYER_CHANNELS> hiddenOutput1;
    NtcEvaluateLayer_CoopVec_FP8<Params::INPUT_CHANNELS, Params::HIDDEN_LAYER_CHANNELS, true>
    (t_WeightBuffer, g_Const.networkWeightOffsets.x, scaleBiasOffset, false, networkInputs, hiddenOutput1);
    scaleBiasOffset += Params::HIDDEN_LAYER_CHANNELS * sizeof(float16_t);
    
    // Hidden layer 1
    CoopVec<float16_t, Params::HIDDEN_LAYER_CHANNELS> hiddenOutput2;
    NtcEvaluateLayer_CoopVec_FP8<Params::HIDDEN_LAYER_CHANNELS, Params::HIDDEN_LAYER_CHANNELS, true>
    (t_WeightBuffer, g_Const.networkWeightOffsets.y, scaleBiasOffset, false, hiddenOutput1, hiddenOutput2);
    scaleBiasOffset += Params::HIDDEN_LAYER_CHANNELS * sizeof(float16_t);
    
    // Hidden layer 2
    CoopVec<float16_t, Params::HIDDEN_LAYER_CHANNELS> hiddenOutput3;
    NtcEvaluateLayer_CoopVec_FP8<Params::HIDDEN_LAYER_CHANNELS, Params::HIDDEN_LAYER_CHANNELS, true>
    (t_WeightBuffer, g_Const.networkWeightOffsets.z, scaleBiasOffset, true, hiddenOutput2, hiddenOutput3);
    scaleBiasOffset += Params::HIDDEN_LAYER_CHANNELS * sizeof(float16_t);
    
    // Output layer
    CoopVec<float, Params::OUTPUT_CHANNELS> networkOutputs;
    NtcEvaluateOutputLayer_CoopVec_FP8<Params::HIDDEN_LAYER_CHANNELS, Params::OUTPUT_CHANNELS>
    (t_WeightBuffer, g_Const.networkWeightOffsets.w, scaleBiasOffset, hiddenOutput3, networkOutputs);

    // Store the outputs into shared memory for efficient indexed access later.
    // Note: there is no need for a barrier before or after this store because we're using a dedicated
    // shared memory array, and each thread only reads the data it's written - nothing from other threads.
    int threadOffset = (threadIndex.y * DECOMPRESS_CS_BLOCK_WIDTH + threadIndex.x) * Params::OUTPUT_CHANNELS;
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
