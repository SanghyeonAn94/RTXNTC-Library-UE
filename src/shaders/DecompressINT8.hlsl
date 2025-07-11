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

#include "DecompressCommon.hlsli"
   
// Use manual allocation and addressing for shared memory to share the same memory
// between latent preloading and matrix/scale/bias preloading because these actions
// do not overlap.

// First phase - latent preloading
static const int HR_LATENT_BASE_ADDR = 0;
static const int HR_LATENT_MEM_SIZE = (Params::HR_FEATURES / 4) * HR_LATENTS_WIDTH * HR_LATENTS_HEIGHT;
static const int LR_LATENT_BASE_ADDR = HR_LATENT_MEM_SIZE;
static const int LR_LATENT_MEM_SIZE = (Params::LR_FEATURES / 4) * LR_LATENTS_WIDTH * LR_LATENTS_HEIGHT;

// Second phase - matrix and scale/bias preloading
static const int MATRIX_B_BASE_ADDR = 0;
static const int MATRIX_B_MEM_SIZE = MAX_OUTPUT_SIZE * (MAX_INPUT_SIZE / 4);
static const int SCALE_BIAS_SIZE = MAX_OUTPUT_SIZE;
static const int BIAS_BASE_ADDR = MATRIX_B_MEM_SIZE;
static const int SCALE_BASE_ADDR = BIAS_BASE_ADDR + SCALE_BIAS_SIZE;

// Third phase - output shuffling
static const int OUTPUT_BASE_ADDR = 0;
#if USE_FLOAT16
static const int OUTPUT_UINTS = Params::OUTPUT_CHANNELS/2;
#else
static const int OUTPUT_UINTS = Params::OUTPUT_CHANNELS;
#endif
static const int OUTPUT_SIZE = DECOMPRESS_CS_BLOCK_WIDTH * DECOMPRESS_CS_BLOCK_HEIGHT * (OUTPUT_UINTS+1);

// Calculate the total shared memory size and allocate it
static const int SHARED_MEMORY_SIZE = max(OUTPUT_SIZE, max(HR_LATENT_MEM_SIZE + LR_LATENT_MEM_SIZE, MATRIX_B_MEM_SIZE + SCALE_BIAS_SIZE * 2));
groupshared uint s_SharedMem[SHARED_MEMORY_SIZE];

// Shared memory address calculation functions

int GetHighResLatentAddress(int latentIdx, int x, int y)
{
    return HR_LATENT_BASE_ADDR + (latentIdx * HR_LATENTS_HEIGHT + y) * HR_LATENTS_WIDTH + x;
}

int GetLowResLatentAddress(int latentIdx, int x, int y)
{
    return LR_LATENT_BASE_ADDR + (latentIdx * LR_LATENTS_HEIGHT + y) * LR_LATENTS_WIDTH + x;
}

int GetMatrixBAddress(int col, int row)
{
    return MATRIX_B_BASE_ADDR + col * (MAX_INPUT_SIZE / 4) + row;
}

int GetBiasAddress(int index)
{
    return BIAS_BASE_ADDR + index;
}

int GetScaleAddress(int index)
{
    return SCALE_BASE_ADDR + index;
}

int GetOutputAddress(int ch, int2 threadIdx)
{
    return (threadIdx.y * DECOMPRESS_CS_BLOCK_WIDTH + threadIdx.x) * (OUTPUT_UINTS+1) + ch;
}

template<bool HIGH_RES>
void PreloadLatents(
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
        
        for (int i = 0; i < encoding.numFeatures / 4; i++)
        {
            int4 inp = NtcLoadFourInputQuantizedLatents(t_InputFile, 0, encoding, neuralMip, addr);
            addr += 4;
            
            int sharedAddr = HIGH_RES
                ? GetHighResLatentAddress(latentOffset + i, renamedThreadIdx.x, renamedThreadIdx.y)
                : GetLowResLatentAddress(latentOffset + i, renamedThreadIdx.x, renamedThreadIdx.y);

            s_SharedMem[sharedAddr] = NtcPackInt8x4(inp);
        }
        linearThreadIndex += DECOMPRESS_CS_BLOCK_WIDTH * DECOMPRESS_CS_BLOCK_HEIGHT;
    }
}

template<int NUM_FEATURES, bool ALL_CORNERS, bool HIGH_RES>
void SampleLatentGridShared(
    NtcLatentEncodingConstants encoding,
    NtcNeuralMipConstants neuralMip,
    float2 uv,
    int2 baseLatentPos,
    int latentOffset,
    int outputOffset,
    inout uint outputArray[Params::INPUT_CHANNELS / 4])
{
    int2 topLeftPos;
    float4 weights;
    NtcSetupLatentBilinearFilter(neuralMip, uv, topLeftPos, weights);
    int4 iweights = int4(weights * 256.f);
    
    // Shift right the interpolated weights by 8 to undo the 256 factor above
    const int normalizationShift = 8;
    
    const int2 sharedPos = topLeftPos - baseLatentPos;
    
    for (int i = 0; i < encoding.numFeatures / 4; i++)
    {
        const int sharedAddr00 = HIGH_RES
            ? GetHighResLatentAddress(latentOffset + i, sharedPos.x, sharedPos.y)
            : GetLowResLatentAddress(latentOffset + i, sharedPos.x, sharedPos.y);
        const int sharedAddr01 = HIGH_RES
            ? GetHighResLatentAddress(latentOffset + i, sharedPos.x + 1, sharedPos.y)
            : GetLowResLatentAddress(latentOffset + i, sharedPos.x + 1, sharedPos.y);
        const int sharedAddr10 = HIGH_RES
            ? GetHighResLatentAddress(latentOffset + i, sharedPos.x, sharedPos.y + 1)
            : GetLowResLatentAddress(latentOffset + i, sharedPos.x, sharedPos.y + 1);
        const int sharedAddr11 = HIGH_RES
            ? GetHighResLatentAddress(latentOffset + i, sharedPos.x + 1, sharedPos.y + 1)
            : GetLowResLatentAddress(latentOffset + i, sharedPos.x + 1, sharedPos.y + 1);

        const uint32_t u00 = s_SharedMem[sharedAddr00];
        const uint32_t u01 = s_SharedMem[sharedAddr01];
        const uint32_t u10 = s_SharedMem[sharedAddr10];
        const uint32_t u11 = s_SharedMem[sharedAddr11];

        // Unpack the latents into int4's for blending and multiply by weights.
        const int4 x00 = NtcUnpackInt8x4(u00) * iweights.x;
        const int4 x01 = NtcUnpackInt8x4(u01) * iweights.y;
        const int4 x10 = NtcUnpackInt8x4(u10) * iweights.z;
        const int4 x11 = NtcUnpackInt8x4(u11) * iweights.w;

        if (ALL_CORNERS)
        {
            // Copy the latents for the 4 pixels into the network inputs.
            outputArray[outputOffset + i + (NUM_FEATURES / 4) * 0] = NtcPackInt8x4(x00 >> normalizationShift);
            outputArray[outputOffset + i + (NUM_FEATURES / 4) * 1] = NtcPackInt8x4(x01 >> normalizationShift);
            outputArray[outputOffset + i + (NUM_FEATURES / 4) * 2] = NtcPackInt8x4(x10 >> normalizationShift);
            outputArray[outputOffset + i + (NUM_FEATURES / 4) * 3] = NtcPackInt8x4(x11 >> normalizationShift);
        }
        else
        {
            // Blend the features of the 4 pixels.
            int4 d = (x00 + x01 + x10 + x11) >> normalizationShift;
            outputArray[outputOffset + i] = NtcPackInt8x4(d);
        }
    }
}

template<int IN, int OUT, bool OUT_FLOAT>
void EvaluateLayerINT8_SharedMem(
    int weightOffset,
    inout int scaleBiasOffset,
    int totalChannels,
    bool activation,
    uint inputArray[IN / 4],
#if USE_FLOAT16
    out uint outputArray[OUT_FLOAT ? OUT / 2 : OUT / 4],
#else
    out uint outputArray[OUT_FLOAT ? OUT : OUT / 4],
#endif
    int2 threadIndex)
{
    GroupMemoryBarrierWithGroupSync();

    // Preload the bias values into shared memory.
    // Note: this 'if' assumes that there are enough threads in the group to load all bias values in one pass.
    // If that ever changes, use a loop like one used for the weights below.
    const int linearThreadIndex = threadIndex.x + threadIndex.y * DECOMPRESS_CS_BLOCK_WIDTH;
    if (linearThreadIndex < OUT)
    {
        float scale = t_WeightBuffer.Load<float>(scaleBiasOffset + linearThreadIndex * sizeof(float));
        int bias = t_WeightBuffer.Load<int>(scaleBiasOffset + (totalChannels + linearThreadIndex) * sizeof(int));

        s_SharedMem[GetScaleAddress(linearThreadIndex)] = asuint(scale);
        s_SharedMem[GetBiasAddress(linearThreadIndex)] = asuint(bias);
    }

    // Preload the weights into shared memory.
    // The weights form a matrix with IN rows and OUT columns, stored in a column-major layout
    // (i.e. elements of a column are continuous.)
    int preloadIndex = linearThreadIndex;
    while (preloadIndex < (IN * OUT) / 4)
    {
        int k = preloadIndex % (IN/4); // row
        int c = preloadIndex / (IN/4); // column
        
        s_SharedMem[GetMatrixBAddress(c, k)] = t_WeightBuffer.Load(weightOffset + preloadIndex * 4);

        preloadIndex += DECOMPRESS_CS_BLOCK_WIDTH * DECOMPRESS_CS_BLOCK_HEIGHT;
    }

    // Advance the input offsets to point at the next layer.
    scaleBiasOffset += OUT * sizeof(float);

    GroupMemoryBarrierWithGroupSync();

    // Note: not unrolling the outer loop.
    // If we do, DXC/SPIR-V crashes.
    // DXC/DXIL compiles the unrolled loop successfully, but then creating a pipeline with it takes seconds,
    // and the resulting code works slower than a regular loop.
    for (uint c = 0; c < OUT; c += 4)
    {
        int acc0 = asint(s_SharedMem[GetBiasAddress(c + 0)]);
        int acc1 = asint(s_SharedMem[GetBiasAddress(c + 1)]);
        int acc2 = asint(s_SharedMem[GetBiasAddress(c + 2)]);
        int acc3 = asint(s_SharedMem[GetBiasAddress(c + 3)]);
        
        [unroll]
        for (uint k = 0; k < IN / 4; k++)
        {
            const int matrixAddr0 = GetMatrixBAddress(c + 0, k);
            const int matrixAddr1 = GetMatrixBAddress(c + 1, k);
            const int matrixAddr2 = GetMatrixBAddress(c + 2, k);
            const int matrixAddr3 = GetMatrixBAddress(c + 3, k);

#if USE_DP4A
            acc0 = dot4add_i8packed(inputArray[k], s_SharedMem[matrixAddr0], acc0);
            acc1 = dot4add_i8packed(inputArray[k], s_SharedMem[matrixAddr1], acc1);
            acc2 = dot4add_i8packed(inputArray[k], s_SharedMem[matrixAddr2], acc2);
            acc3 = dot4add_i8packed(inputArray[k], s_SharedMem[matrixAddr3], acc3);
#else
            acc0 += NtcDotProductInt8x4(inputArray[k], s_SharedMem[matrixAddr0]);
            acc1 += NtcDotProductInt8x4(inputArray[k], s_SharedMem[matrixAddr1]);
            acc2 += NtcDotProductInt8x4(inputArray[k], s_SharedMem[matrixAddr2]);
            acc3 += NtcDotProductInt8x4(inputArray[k], s_SharedMem[matrixAddr3]);
#endif
        }

        float4 results = float4(acc0, acc1, acc2, acc3);

        float4 scales;
        scales.x = asfloat(s_SharedMem[GetScaleAddress(c + 0)]);
        scales.y = asfloat(s_SharedMem[GetScaleAddress(c + 1)]);
        scales.z = asfloat(s_SharedMem[GetScaleAddress(c + 2)]);
        scales.w = asfloat(s_SharedMem[GetScaleAddress(c + 3)]);

        results *= scales;

#if USE_FLOAT16
        float16_t4 hresults = float16_t4(results);
        
        if (activation)
        {
            hresults = NtcHGELUClamp_ForwardHalf(hresults);
        }

        if (OUT_FLOAT)
        {
            outputArray[c / 2 + 0] = NtcHalf2ToUint(hresults.xy);
            outputArray[c / 2 + 1] = NtcHalf2ToUint(hresults.zw);
        }
        else
        {
            int4 iresults = NtcHGELUClamp_QuantizeHalf(hresults);

            outputArray[c / 4] = NtcPackInt8x4(iresults);
        }
#else
        if (activation)
        {
            results = NtcHGELUClamp_ForwardFloat(results);
        }

        if (OUT_FLOAT)
        {
            outputArray[c + 0] = asuint(results.x);
            outputArray[c + 1] = asuint(results.y);
            outputArray[c + 2] = asuint(results.z);
            outputArray[c + 3] = asuint(results.w);
        }
        else
        {
            int4 iresults = NtcHGELUClamp_QuantizeFloat(results);

            outputArray[c / 4] = NtcPackInt8x4(iresults);
        }
#endif
    }
}

void DecompressPixel(uint2 globalIndex, uint2 threadIndex)
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

    PreloadLatents<true>(NtcUnpackLatentEncodingConstants(g_Const.highResEncoding),
        highResNeuralMip, highResNeuralScale, baseHighResLatentPos, 0, threadIndex);
    PreloadLatents<false>(NtcUnpackLatentEncodingConstants(g_Const.lowResEncoding),
        lowResNeuralMip, lowResNeuralScale, baseLowResLatentPos, Params::HR_FEATURES / 4, threadIndex);
    GroupMemoryBarrierWithGroupSync();
#endif

    uint networkInputs[Params::INPUT_CHANNELS / 4];

    // Zero init the array - in some cases, INPUT_CHANNELS is rounded up from the actual used size.
    // DXC rightfully complains about the use of uninitialized variables in this case.
    [unroll]
    for (int i = 0; i < Params::INPUT_CHANNELS / 4; ++i)
        networkInputs[i] = 0;

    int inputOffset = 0;
    const float2 uv = (float2(pixelPosition) + 0.5) / colorMipSize;

#if PRELOAD_LATENTS
    // Sample the latent grids from preloaded data
    SampleLatentGridShared<Params::HR_FEATURES, true, true>(NtcUnpackLatentEncodingConstants(g_Const.highResEncoding),
        highResNeuralMip, uv, baseHighResLatentPos, 0, inputOffset, networkInputs);
    inputOffset += Params::SAMPLED_FEATURES_HR / 4;

    SampleLatentGridShared<Params::LR_FEATURES, false, false>(NtcUnpackLatentEncodingConstants(g_Const.lowResEncoding),
        lowResNeuralMip, uv, baseLowResLatentPos, Params::HR_FEATURES / 4, inputOffset, networkInputs);
    inputOffset += Params::SAMPLED_FEATURES_LR / 4;
#else
    // Sample the latent grids
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
    
    int scaleBiasOffset = g_Const.networkScaleBiasOffset;

    // Evaluate the MLP layers:
    const int totalChannels = Params::HIDDEN_LAYER_CHANNELS * 3 + Params::OUTPUT_CHANNELS;
    // Input layer
    uint hiddenOutput1[Params::HIDDEN_LAYER_CHANNELS / 4];
    EvaluateLayerINT8_SharedMem<Params::INPUT_CHANNELS, Params::HIDDEN_LAYER_CHANNELS, false>
        (g_Const.networkWeightOffsets.x, scaleBiasOffset, totalChannels,  true,
        networkInputs, hiddenOutput1, threadIndex);

    // Hidden layer 1
    uint hiddenOutput2[Params::HIDDEN_LAYER_CHANNELS / 4];
    EvaluateLayerINT8_SharedMem<Params::HIDDEN_LAYER_CHANNELS, Params::HIDDEN_LAYER_CHANNELS, false>
        (g_Const.networkWeightOffsets.y, scaleBiasOffset, totalChannels, true,
        hiddenOutput1, hiddenOutput2, threadIndex);

    // Hidden layer 2
    EvaluateLayerINT8_SharedMem<Params::HIDDEN_LAYER_CHANNELS, Params::HIDDEN_LAYER_CHANNELS, false>
        (g_Const.networkWeightOffsets.z, scaleBiasOffset, totalChannels, true,
        hiddenOutput2, hiddenOutput1, threadIndex);

    // Output layer
    uint networkOutputs[OUTPUT_UINTS];
    EvaluateLayerINT8_SharedMem<Params::HIDDEN_LAYER_CHANNELS, Params::OUTPUT_CHANNELS, true>
        (g_Const.networkWeightOffsets.w, scaleBiasOffset, totalChannels, false,
        hiddenOutput1, networkOutputs, threadIndex);
    
    HashBasedRNG rng = HashBasedRNG::Create(pixelPosition.x + pixelPosition.y * g_Const.imageWidth, 0);

    // Store the outputs into shared memory for efficient indexed access later.
    // Note: there is no need for a barrier after this store because each thread only reads the data
    // it's written - nothing from other threads.
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (int ch = 0; ch < OUTPUT_UINTS; ++ch)
    {
        s_SharedMem[GetOutputAddress(ch, threadIndex)] = networkOutputs[ch];
    }

    // Exit if this pixel is outside of the specified rectangle
    if (pixelPosition.x < g_Const.srcLeft || pixelPosition.y < g_Const.srcTop ||
        pixelPosition.x >= g_Const.srcRight || pixelPosition.y >= g_Const.srcBottom)
        return;
    
    // Shuffle the output data into destination textures
    for (int outputIndex = 0; outputIndex < g_Const.numOutputs; ++outputIndex)
    {
        const NtcDecompressOutputDesc outputDesc = g_Const.outputs[outputIndex];
        
        // Read 4 channels from the shared buffer
        float4 texelValue;
        [unroll]
        for (int ch = 0; ch < 4; ++ch)
        {
            int srcChannel = min(outputDesc.firstChannel + ch, Params::OUTPUT_CHANNELS - 1);
#if USE_FLOAT16
            uint twoCh = s_SharedMem[GetOutputAddress(srcChannel/2, threadIndex)];
            if (srcChannel & 1)
                twoCh >>= 16;
            texelValue[ch] = asfloat16(uint16_t(twoCh));
#else
            texelValue[ch] = asfloat(s_SharedMem[GetOutputAddress(srcChannel, threadIndex)]);
#endif
        }

        // Perform color space conversion, if needed
        texelValue.rgb = NtcConvertColorSpace(texelValue.rgb, outputDesc.srcRgbColorSpace, outputDesc.dstRgbColorSpace);
        texelValue.a = NtcConvertColorSpace(texelValue.a, outputDesc.srcAlphaColorSpace, outputDesc.dstAlphaColorSpace);
        
        // Apply dithering
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
    DecompressPixel(globalIndex, threadIndex);
}
