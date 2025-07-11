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

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include "CudaUtils.h"
#include "FeatureGridMath.h"
#include "MlpDesc.h"
#include "RegressionCommon.h"
#include "LatentQuantization.h"
#include "tin/tin_matrix_host.h"
#include "tin/tin_activation.h"
#include "tin/tin_mlp.h"
#include <libntc/ntc.h>
#include <cuda_fp8.h>


namespace ntc::cuda
{

namespace th = tin::host;

struct AddressParams
{
    th::HMatrixB wtMat;
    int rows;
    int col = 0;
    int weightOffsetForLayer = 0;
    int channelOffsetForLayer = 0;
    int totalChannels = 0;
    int globalColumnIndex = 0;
    bool inputLayer = false;
    bool outputLayer = false;

    __device__ AddressParams(int rows, int cols)
        : wtMat(rows, cols)
        , rows(rows)
    { }
};

static __device__ AddressParams GetColumnAddressParams(
    int hiddenLayers,
    int inputChannels,
    int hiddenChannels,
    int outputChannels,
    int threadIdx)
{
    int lastLayerOffset = hiddenChannels * (hiddenLayers + 1);

    int colLast = threadIdx - (lastLayerOffset);
    int colFirst = threadIdx % hiddenChannels;

    bool outputLayer = colLast >= 0;

    bool inputLayer = (threadIdx - hiddenChannels) < 0;
    int rows = inputLayer ? inputChannels : hiddenChannels;
    int cols = outputLayer ? outputChannels : hiddenChannels;

    AddressParams params(rows, cols);
    params.col = (outputLayer ? colLast : colFirst);

    int hiddenLayer = (threadIdx - hiddenChannels) / hiddenChannels;
    params.weightOffsetForLayer = inputLayer ? 0 : inputChannels * hiddenChannels + hiddenChannels * hiddenChannels * hiddenLayer;
    params.channelOffsetForLayer = inputLayer ? 0 : hiddenChannels * (hiddenLayer + 1);
    params.totalChannels = lastLayerOffset + outputChannels;
    params.globalColumnIndex = threadIdx;
    params.inputLayer = inputLayer;
    params.outputLayer = outputLayer;
    return params;
}
extern __constant__ ChannelInfo g_ChannelInfo[NTC_MAX_CHANNELS];

__device__ void QuantizeColumnInt8(
    int weightCount,
    AddressParams params,
    half* __restrict__ halfWeights,
    int8_t* __restrict__ int8WeightsForLayer,
    float* __restrict__ scaleForLayer,
    int32_t* __restrict__ biasForLayer)
{
    half2* half2Weights = (half2*)(halfWeights + params.weightOffsetForLayer);

    float elemMin = std::numeric_limits<float>::max();
    float elemMax = std::numeric_limits<float>::min();

    for (int r = 0; r < params.rows; r += 2)
    {
        int elemOffset = params.wtMat.get_packed_offset(r, params.col);

        float2 elem = __half22float2(half2Weights[elemOffset]);

        elemMin = std::min(elemMin, elem.x);
        elemMax = std::max(elemMax, elem.x);
        elemMin = std::min(elemMin, elem.y);
        elemMax = std::max(elemMax, elem.y);
    }
    float limit = std::max(fabs(elemMax), fabs(elemMin));
    float ilimit = __frcp_rn(limit);

    // Quantize each column
    const float levels = 256;
    const float scale = (levels - 1) / 2;
    const float iscale = 1 / scale;
    const float qmin = -levels / 2 + 1;
    const float qmax =  levels / 2 - 1;

    int integerWeightSum = 0;

    for (int r = 0; r < params.rows; r += 2)
    {
        int elemOffset = params.wtMat.get_packed_offset(r, params.col);

        half2 helem = half2Weights[elemOffset];

        float2 elem = __half22float2(helem);
        elem.x = round(elem.x * scale * ilimit);
        elem.x = std::max(std::min(elem.x, qmax), qmin);
        int8_t qx = int8_t(elem.x);
        elem.x = elem.x * limit * iscale;

        elem.y = round(elem.y * (scale / limit));
        elem.y = std::max(std::min(elem.y, qmax), qmin);
        int8_t qy = int8_t(elem.y);
        elem.y = elem.y * limit * iscale;
        half2 res = __float22half2_rn(elem);

        half2Weights[elemOffset] = res;

        if (int8WeightsForLayer)
        {
            int addr = params.col * params.rows + r;
            int8WeightsForLayer[addr + 0] = qx;
            int8WeightsForLayer[addr + 1] = qy;
        }
        
        integerWeightSum += qx + qy;
    }

    if (scaleForLayer || biasForLayer)
    {
        float layerScale = limit * iscale;
        float layerBias = halfWeights[weightCount + params.globalColumnIndex];

        const float activationScale = tin::ActHGELUClamp::step;
        const int activationBias = tin::ActHGELUClamp::bias;

        if (params.inputLayer)
        {
            layerScale /= tin::InputQuant::scale;
        }
        else
        {
            layerScale *= activationScale;
            layerBias  -= float(integerWeightSum * activationBias) * layerScale;

            if (params.outputLayer)
            {
                layerScale *= g_ChannelInfo[params.col].optimalToLinearScale;
                layerBias  = layerBias * g_ChannelInfo[params.col].optimalToLinearScale + g_ChannelInfo[params.col].optimalToLinearBias;
            }
        }

        // Convert the float scale and bias from the (float(output) * scale + bias) form to the
        // (float(output + int(bias/scale)) * scale) form.
        // Special case when scale is zero, which can happen on the output layer if an image channel is constant.
        // See also LoadWeightsFromStream(...) in TextureSetMetadata.cpp which implements the same logic.
        int integerLayerBias = 0;
        if (layerScale == 0.f)
        {
            // Zero scale: use a predefined constant scale and express the constant bias using this scale.
            float const constantScale = 65536.f;
            layerScale = 1.f / constantScale;
            integerLayerBias = int(roundf(layerBias * constantScale));

            // Zero out the weights to produce the correct result.
            if (int8WeightsForLayer)
            {
                for (int r = 0; r < params.rows; r += 2)
                {
                    int addr = params.col * params.rows + r;
                    int8WeightsForLayer[addr + 0] = 0;
                    int8WeightsForLayer[addr + 1] = 0;
                }
            }
        }
        else
        {
            // Nonzero scale: simple conversion.
            integerLayerBias = int(roundf(layerBias / layerScale));
        }

        if (scaleForLayer) scaleForLayer[params.col] = layerScale;
        if (biasForLayer) biasForLayer[params.col] = integerLayerBias;
    }
}

__device__ void QuantizeColumnFP8(
    int weightCount,
    AddressParams params,
    half* __restrict__ halfWeights,
    int8_t* __restrict__ fp8WeightsForLayer,
    half* __restrict__ scaleForLayer,
    half* __restrict__ biasForLayer)
{
    half2* half2Weights = (half2*)(halfWeights + params.weightOffsetForLayer);

    for (int r = 0; r < params.rows; r += 2)
    {
        int elemOffset = params.wtMat.get_packed_offset(r, params.col);

        half2 helem = half2Weights[elemOffset];
        half2 res;

        if (fp8WeightsForLayer)
        {
            // When we need to actually convert the weights, use CUDA FP8 math
            __nv_fp8x2_e4m3 qelem = __nv_fp8x2_e4m3(__half2(helem));
            int8_t qx = int8_t(qelem.__x & 0xff);
            int8_t qy = int8_t(qelem.__x >> 8);
            res = half2(qelem);

            int addr = params.col * params.rows + r;
            fp8WeightsForLayer[addr + 0] = qx;
            fp8WeightsForLayer[addr + 1] = qy;
        }
        else
        {
            // When we don't need the FP8 weights, use the round function because it's faster on pre-SM8.9 GPUs
            res.x = tin::RoundHalfToFloatE4M3(helem.x);
            res.y = tin::RoundHalfToFloatE4M3(helem.y);
        }
        
        half2Weights[elemOffset] = res;
    }

    if (scaleForLayer || biasForLayer)
    {
        float layerScale = 1.f;
        float layerBias = halfWeights[weightCount + params.globalColumnIndex];

        if (params.outputLayer)
        {
            layerScale *= g_ChannelInfo[params.col].optimalToLinearScale;
            layerBias  = layerBias * g_ChannelInfo[params.col].optimalToLinearScale + g_ChannelInfo[params.col].optimalToLinearBias;
        }

        if (scaleForLayer) scaleForLayer[params.col] = layerScale;
        if (biasForLayer) biasForLayer[params.col] = layerBias;
    }
}

__global__ void QuantizeNetworkInt8Kernel(
    int weightCount,
    int hiddenLayers,
    int inputChannels,
    int hiddenChannels,
    int outputChannels,
    half* __restrict__ halfWeights,
    int8_t* __restrict__ int8Data)
{
    using namespace cooperative_groups;
    auto block = cooperative_groups::this_thread_block();

    int i = block.thread_rank();
    AddressParams params = GetColumnAddressParams(hiddenLayers, inputChannels, hiddenChannels, outputChannels, i);

    // See the comment block in the beginning of TextureSet.cpp for the weight layouts
    
    QuantizeColumnInt8(weightCount, params, halfWeights,
        int8Data ? int8Data + params.weightOffsetForLayer : nullptr,
        int8Data ? (float*)(int8Data + weightCount + params.channelOffsetForLayer * sizeof(float)) : nullptr,
        int8Data ? (int32_t*)(int8Data + weightCount + (params.totalChannels + params.channelOffsetForLayer) * sizeof(float)) : nullptr);
}

__global__ void QuantizeNetworkFP8Kernel(
    int weightCount,
    int hiddenLayers,
    int inputChannels,
    int hiddenChannels,
    int outputChannels,
    half* __restrict__ halfWeights,
    int8_t* __restrict__ fp8Data)
{
    using namespace cooperative_groups;
    auto block = cooperative_groups::this_thread_block();

    int i = block.thread_rank();
    AddressParams params = GetColumnAddressParams(hiddenLayers, inputChannels, hiddenChannels, outputChannels, i);

    // See the comment block in the beginning of TextureSet.cpp for the weight layouts

    if (params.outputLayer)
    {
        // Output layer scale and bias are packed together after the fp8 bias values
        QuantizeColumnInt8(weightCount, params, halfWeights,
            fp8Data ? fp8Data + params.weightOffsetForLayer : nullptr,
            fp8Data ? (float*)(fp8Data + weightCount + params.channelOffsetForLayer * sizeof(half)) : nullptr,
            fp8Data ? (int32_t*)(fp8Data + weightCount + params.channelOffsetForLayer * sizeof(half) + outputChannels * sizeof(float)) : nullptr);
    }
    else
    {
        // No scale values, just bias packed together for all layers
        QuantizeColumnFP8(weightCount, params, halfWeights,
            fp8Data ? fp8Data + params.weightOffsetForLayer : nullptr,
            nullptr,
            fp8Data ? (half*)(fp8Data + weightCount + params.channelOffsetForLayer * sizeof(half)) : nullptr);
    }
}

void QuantizeNetwork(
    MlpDesc const* mlpDesc,
    half* __restrict__ halfWeights,
    int8_t* __restrict__ outputData,
    bool useFP8)
{
    int const outputCount = mlpDesc->GetLayerOutputCount();
    int const weightCount = mlpDesc->GetWeightCount();
    
    int threadBlockSize = outputCount;
    if (useFP8)
    {
        QuantizeNetworkFP8Kernel <<< outputCount, threadBlockSize >>> (weightCount, mlpDesc->GetHiddenLayers(),
            mlpDesc->GetInputChannels(), mlpDesc->GetHiddenChannels(), mlpDesc->GetOutputChannels(),
            halfWeights, outputData);
    }
    else
    {
        QuantizeNetworkInt8Kernel <<< outputCount, threadBlockSize >>> (weightCount, mlpDesc->GetHiddenLayers(),
            mlpDesc->GetInputChannels(), mlpDesc->GetHiddenChannels(), mlpDesc->GetOutputChannels(),
            halfWeights, outputData);
    }
}

__device__ void UnquantizeColumnInt8(
    int weightCount,
    AddressParams params,
    half* __restrict__ halfWeights,
    int8_t const* __restrict__ int8WeightsForLayer,
    float const* __restrict__ scaleForLayer,
    int const* __restrict__ biasForLayer)
{
    half2* half2Weights = (half2*)(halfWeights + params.weightOffsetForLayer);

    float layerScale = scaleForLayer[params.col];
    int const integerLayerBias = biasForLayer[params.col];
    float layerBias = float(integerLayerBias) * layerScale;

    // This function reverses the effect of QuantizeNetworkInt8Kernel, except the optimalToLinear scale and bias
    
    // Undo the layerScale multiplication
    if (params.inputLayer)
    {
        layerScale *= tin::InputQuant::scale;
    }
    else
    {
        layerScale *= tin::ActHGELUClamp::invStep;
    }

    // Go over all weights in the column and multiply them by scale.
    // Also accumulate the sum of integer weights to undo the bias change.
    int integerWeightSum = 0;
    for (int r = 0; r < params.rows; r += 2)
    {
        // Read two int8 weights in colum major layout
        int addr = params.col * params.rows + r;
        int8_t qx = int8WeightsForLayer[addr + 0];
        int8_t qy = int8WeightsForLayer[addr + 1];

        float2 elem;
        elem.x = float(qx) * layerScale;
        elem.y = float(qy) * layerScale;

        // Write two fp16 weights in MMA layout
        int elemOffset = params.wtMat.get_packed_offset(r, params.col);
        half2Weights[elemOffset] = __float22half2_rn(elem);
    
        integerWeightSum += qx + qy;
    }

    // Undo the bias change
    if (!params.inputLayer)
    {
        const float activationScale = tin::ActHGELUClamp::step;
        const int activationBias = tin::ActHGELUClamp::bias;

        // Note: multiplying by activationScale here because that term was removed from layerScale earlier
        layerBias += float(integerWeightSum * activationBias) * layerScale * activationScale;
    }

    // Write the fp16 bias
    halfWeights[weightCount + params.globalColumnIndex] = layerBias;
}

__device__ void UnquantizeColumnFP8(
    int weightCount,
    AddressParams params,
    half* __restrict__ halfWeights,
    int8_t const* __restrict__ fp8WeightsForLayer,
    half const* __restrict__ scaleForLayer,
    half const* __restrict__ biasForLayer)
{
    half2* half2Weights = (half2*)(halfWeights + params.weightOffsetForLayer);

    // This function reverses the effect of QuantizeNetworkFP8Kernel, except the optimalToLinear scale and bias
    
    for (int r = 0; r < params.rows; r += 2)
    {
        // Read two fp8 weights in colum major layout
        int addr = params.col * params.rows + r;
        
        __nv_fp8x2_e4m3 qelem;
        qelem.__x = *reinterpret_cast<uint16_t const*>(fp8WeightsForLayer + addr);

        // Write two fp16 weights in MMA layout
        int elemOffset = params.wtMat.get_packed_offset(r, params.col);
        half2Weights[elemOffset] = half2(qelem);
    }

    // Write the fp16 bias
    float layerBias = biasForLayer ? float(biasForLayer[params.col]) : 0.f;
    halfWeights[weightCount + params.globalColumnIndex] = layerBias;
}

__global__ void ConvertNetworkFromInt8ToFP16Kernel(
    int weightCount,
    int hiddenLayers,
    int inputChannels,
    int hiddenChannels,
    int outputChannels,
    half* __restrict__ halfWeights,
    int8_t* __restrict__ int8Data)
{
    using namespace cooperative_groups;
    auto block = cooperative_groups::this_thread_block();
    
    int i = block.thread_rank();
    AddressParams params = GetColumnAddressParams(hiddenLayers, inputChannels, hiddenChannels, outputChannels, i);

    // See the comment block in the beginning of TextureSet.cpp for the weight layouts

    UnquantizeColumnInt8(weightCount, params, halfWeights,
        int8Data + params.weightOffsetForLayer,
        (float*)(int8Data + weightCount + params.channelOffsetForLayer * sizeof(float)),
        (int32_t*)(int8Data + weightCount + (params.totalChannels + params.channelOffsetForLayer) * sizeof(float)));
}

__global__ void ConvertNetworkFromFP8ToFP16Kernel(
    int weightCount,
    int hiddenLayers,
    int inputChannels,
    int hiddenChannels,
    int outputChannels,
    half* __restrict__ halfWeights,
    int8_t const* __restrict__ fp8Data)
{
    using namespace cooperative_groups;
    auto block = cooperative_groups::this_thread_block();
    
    int i = block.thread_rank();
    AddressParams params = GetColumnAddressParams(hiddenLayers, inputChannels, hiddenChannels, outputChannels, i);

    // See the comment block in the beginning of TextureSet.cpp for the weight layouts

    if (params.outputLayer)
    {
        // Output layer scale and bias are packed together after the fp8 bias values
        UnquantizeColumnInt8(weightCount, params, halfWeights,
            fp8Data + params.weightOffsetForLayer,
            (float*)(fp8Data + weightCount + params.channelOffsetForLayer * sizeof(half)),
            (int32_t*)(fp8Data + weightCount + params.channelOffsetForLayer * sizeof(half) + outputChannels * sizeof(float)));
    }
    else
    {
        // No scale values, just bias packed together for all layers
        UnquantizeColumnFP8(weightCount, params, halfWeights,
            fp8Data + params.weightOffsetForLayer,
            nullptr,
            (half*)(fp8Data + weightCount + params.channelOffsetForLayer * sizeof(half)));
    }
}

void ConvertNetworkFromQuantizedToFp16(
    MlpDesc const* mlpDesc,
    half* __restrict__ halfWeights,
    int8_t* __restrict__ inputData,
    bool useFP8)
{
    int const outputCount = mlpDesc->GetLayerOutputCount();
    int const weightCount = mlpDesc->GetWeightCount();

    int threadBlockSize = outputCount;
    if (useFP8)
    {
        ConvertNetworkFromFP8ToFP16Kernel <<< outputCount, threadBlockSize >>> (weightCount, mlpDesc->GetHiddenLayers(),
        mlpDesc->GetInputChannels(), mlpDesc->GetHiddenChannels(), mlpDesc->GetOutputChannels(), halfWeights, inputData);
    }
    else
    {
        ConvertNetworkFromInt8ToFP16Kernel <<< outputCount, threadBlockSize >>> (weightCount, mlpDesc->GetHiddenLayers(),
        mlpDesc->GetInputChannels(), mlpDesc->GetHiddenChannels(), mlpDesc->GetOutputChannels(), halfWeights, inputData);
    }
}

__global__ void ExportNetworkIntoRowMajorLayoutKernel(
    int weightCount,
    int hiddenLayers,
    int inputChannels,
    int hiddenChannels,
    int outputChannels,
    half* __restrict__ halfWeights,
    half* __restrict__ fp16Data)
{
    using namespace cooperative_groups;
    auto block = cooperative_groups::this_thread_block();

    int i = block.thread_rank();
    AddressParams params = GetColumnAddressParams(hiddenLayers, inputChannels, hiddenChannels, outputChannels, i);

    // See the comment block in the beginning of TextureSet.cpp for the weight layouts

    half2* half2Weights = (half2*)(halfWeights + params.weightOffsetForLayer);
    half* fp16WeightsForLayer = fp16Data + params.weightOffsetForLayer;
    half* biasForLayer = fp16Data + weightCount + params.channelOffsetForLayer;

    half layerScale = half(1.f);
    if (params.outputLayer)
        layerScale = g_ChannelInfo[params.col].optimalToLinearScale;

    for (int r = 0; r < params.rows; r += 2)
    {
        int elemOffset = params.wtMat.get_packed_offset(r, params.col);

        half2 helem = half2Weights[elemOffset];
        
        int addr = params.col * params.rows + r;
        fp16WeightsForLayer[addr + 0] = helem.x * layerScale;
        fp16WeightsForLayer[addr + 1] = helem.y * layerScale;
    }

    half layerBias = halfWeights[weightCount + params.globalColumnIndex];

    if (params.outputLayer)
    {
        layerBias = layerBias * layerScale + half(g_ChannelInfo[params.col].optimalToLinearBias);
    }

    biasForLayer[params.col] = layerBias;
}

void ExportNetworkIntoRowMajorLayout(
    MlpDesc const* mlpDesc,
    half* __restrict__ halfWeights,
    half* __restrict__ outputData)
{
    int const outputCount = mlpDesc->GetLayerOutputCount();
    int const weightCount = mlpDesc->GetWeightCount();

    int threadBlockSize = outputCount;
    ExportNetworkIntoRowMajorLayoutKernel <<< outputCount, threadBlockSize >>> (weightCount, mlpDesc->GetHiddenLayers(),
        mlpDesc->GetInputChannels(), mlpDesc->GetHiddenChannels(), mlpDesc->GetOutputChannels(),
        halfWeights, outputData);
}

__global__ void ImportNetworkFromRowMajorLayoutKernel(
    int weightCount,
    int hiddenLayers,
    int inputChannels,
    int hiddenChannels,
    int outputChannels,
    half* __restrict__ halfWeights,
    half* __restrict__ inputData)
{
    using namespace cooperative_groups;
    auto block = cooperative_groups::this_thread_block();

    int i = block.thread_rank();
    AddressParams params = GetColumnAddressParams(hiddenLayers, inputChannels, hiddenChannels, outputChannels, i);

    // See the comment block in the beginning of TextureSet.cpp for the weight layouts

    half2* half2Weights = (half2*)(halfWeights + params.weightOffsetForLayer);
    half* fp16WeightsForLayer = inputData + params.weightOffsetForLayer;
    half* biasForLayer = inputData + weightCount + params.channelOffsetForLayer;

    // This function reverses the effect of QuantizeNetworkFP8Kernel, except the optimalToLinear scale and bias

    for (int r = 0; r < params.rows; r += 2)
    {
        // Read two fp16 weights in column major layout
        int addr = params.col * params.rows + r;

        half2 values = *reinterpret_cast<half2 const*>(fp16WeightsForLayer + addr);

        // Write two fp16 weights in MMA layout
        int elemOffset = params.wtMat.get_packed_offset(r, params.col);
        half2Weights[elemOffset] = values;
    }

    // Write the fp16 bias
    float layerBias = biasForLayer ? float(biasForLayer[params.col]) : 0.f;
    halfWeights[weightCount + params.globalColumnIndex] = layerBias;
}

void ImportNetworkFromRowMajorLayout(
    MlpDesc const* mlpDesc,
    half* __restrict__ halfWeights,
    half* __restrict__ inputData)
{
    int const outputCount = mlpDesc->GetLayerOutputCount();
    int const weightCount = mlpDesc->GetWeightCount();

    int threadBlockSize = outputCount;
    ImportNetworkFromRowMajorLayoutKernel <<< outputCount, threadBlockSize >>> (weightCount, mlpDesc->GetHiddenLayers(),
        mlpDesc->GetInputChannels(), mlpDesc->GetHiddenChannels(), mlpDesc->GetOutputChannels(),
        halfWeights, inputData);
}

__device__ int WeightIndexToFeatureAddress(
    int width,
    int height,
    int numFeatures,
    int weightIdx)
{
    int feature = weightIdx % numFeatures;
    int pixel = weightIdx / numFeatures;
    int x = pixel % width;
    int y = pixel / width;

    //     [------------- plane -------------]   [----- pixel -----]   [- feature -]
    return (feature >> 1) * width * height * 2 + (y * width + x) * 2 + (feature & 1);
}

__global__ void QuantizeAndPackLatentsKernel(
    int width,
    int height,
    int numFeatures,
    int numWeights,
    int numQuantizedWords,
    int quantBits,
    const half* __restrict__ w_in,
    uint32_t* __restrict__ w_packed_out)
{
    using namespace cooperative_groups;

    grid_group gg = this_grid();
    int threadIdx = gg.thread_rank();

    if (threadIdx >= numQuantizedWords)
        return;
        
    QuantizationParameters const quantizationParams = GetLatentQuantization(quantBits);

    const int elementsPerThread = 32 / quantBits;
    const int elementMask = (1 << quantBits) - 1;

    uint32_t result = 0;
    for (int elemIdx = 0; elemIdx < elementsPerThread; ++elemIdx)
    {
        const int weightIdx = threadIdx * elementsPerThread + elemIdx;

        const int srcAddr = WeightIndexToFeatureAddress(width, height, numFeatures, weightIdx);
        
        if (srcAddr >= numWeights)
            break;

        // Load the weight
        float weight = w_in[srcAddr];

        // Quantize
        weight *= quantizationParams.scale;
        weight = std::min(std::max(weight, quantizationParams.qmin), quantizationParams.qmax);
        // Offset so that -1 maps to 0
        weight += quantizationParams.scale - 1.f;
        // Convert to integer
        const int w_i = int(floorf(weight));
        // Pack into the result
        result |= (w_i & elementMask) << (elemIdx * quantBits);
    }

    w_packed_out[threadIdx] = result;
}

void QuantizeAndPackLatents(
    int width,
    int height,
    int numFeatures,
    int quantBits,
    const half* __restrict__ w_in,
    uint32_t* __restrict__ w_packed_out)
{
    int numWeights = width * height * numFeatures;
    int numQuantizedWords = FeatureGridMath::GetQuantizedLatentSizeUints(numWeights, quantBits);

    int dim_tb = tin::WarpSize;
    int dim_grid = (numQuantizedWords + dim_tb - 1) / dim_tb;

    QuantizeAndPackLatentsKernel <<< dim_grid, dim_tb >>> (width, height, numFeatures, numWeights, numQuantizedWords, quantBits, w_in, w_packed_out);
}

__global__ void UnpackQuantizedLatentsKernel(
    int width,
    int height,
    int numFeatures,
    int numWeights,
    int numQuantizedWords,
    int quantBits,
    const uint32_t* __restrict__ w_packed_in,
    half* __restrict__ w_out)
{
    using namespace cooperative_groups;

    grid_group gg = this_grid();
    int threadIdx = gg.thread_rank();

    if (threadIdx >= numQuantizedWords)
        return;

    const int elementsPerThread = 32 / quantBits;

    QuantizationParameters const quantizationParams = GetLatentQuantization(quantBits);
    
    const uint32_t elementMask = (1 << quantBits) - 1;

    const uint32_t packed = w_packed_in[threadIdx];

    for (int elemIdx = 0; elemIdx < elementsPerThread; ++elemIdx)
    {
        const int weightIdx = threadIdx * elementsPerThread + elemIdx;

        const int dstAddr = WeightIndexToFeatureAddress(width, height, numFeatures, weightIdx);

        if (dstAddr >= numWeights)
            break;

        // Convert from [0..2^quant_bits-1] to (-1..1)
        const uint32_t w_i = (packed >> (elemIdx * quantBits)) & elementMask;
        float w = float(w_i) * quantizationParams.step + quantizationParams.bias;

        w_out[dstAddr] = half(w);
    }
}

void UnpackQuantizedLatents(
    int width,
    int height,
    int numFeatures,
    int quantBits,
    const uint32_t* __restrict__ w_packed_in,
    half* __restrict__ w_out)
{
    int numWeights = width * height * numFeatures;
    int numQuantizedWords = FeatureGridMath::GetQuantizedLatentSizeUints(numWeights, quantBits);

    int dim_tb = tin::WarpSize;
    int dim_grid = (numQuantizedWords + dim_tb - 1) / dim_tb;

    UnpackQuantizedLatentsKernel <<< dim_grid, dim_tb >>> (width, height, numFeatures, numWeights, numQuantizedWords, quantBits, w_packed_in, w_out);
}

} // namespace ntc::cuda