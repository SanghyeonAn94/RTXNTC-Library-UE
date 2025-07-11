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

#include "TextureSetMetadata.h"
#include "CoopVecWeightConverter.h"
#include "Errors.h"
#include "FeatureGridMath.h"
#include "JsonFileFormat.h"
#include "LatentQuantization.h"
#include "MathUtils.h"
#include "MlpDesc.h"
#include "TextureMetadata.h"
#include "Context.h"
#include <cassert>
#include <cinttypes>
#include <cstring>

#include <libntc/shaders/DecompressConstants.h>

namespace ntc
{

constexpr float c_ConstantBiasScale = 65536.f;

TextureSetMetadata::TextureSetMetadata(IAllocator* allocator, Context const* context,
    TextureSetDesc const& desc, LatentShape const& latentShape)
    : m_allocator(allocator)
    , m_context(context)
    , m_desc(desc)
    , m_latentShape(latentShape)
    , m_textureInfos(allocator)
    , m_rowMajorWeightDataInt8(allocator)
    , m_rowMajorWeightDataFP8(allocator)
    , m_latentImages(allocator)
{
    if (!m_latentShape.IsEmpty())
    {
        m_mlpDesc = MlpDesc::PickOptimalConfig(m_latentShape.highResFeatures, m_latentShape.lowResFeatures);
    }

    m_channelColorSpaces.fill(ntc::ColorSpace::Linear);

    // Initialize the shuffle pattern with an identity map (i -> i)
    for (int ch = 0; ch < NTC_MAX_CHANNELS; ++ch)
    {
        m_channelShuffleMapping[ch].type = ShuffleSourceType::Channel;
        m_channelShuffleMapping[ch].channelIndex = ch;
    }
}

ITextureMetadata* TextureSetMetadata::AddTexture()
{
    TextureMetadata* texture = new(m_allocator->Allocate(sizeof(TextureMetadata))) TextureMetadata(m_allocator, this);
    m_textureInfos.push_back(UniquePtr<TextureMetadata>(texture, m_allocator));
    return texture;
}

Status TextureSetMetadata::RemoveTexture(ITextureMetadata* texture)
{
    for (auto it = m_textureInfos.begin(); it != m_textureInfos.end(); ++it)
    {
        if (it->get() == texture)
        {
            m_textureInfos.erase(it);
            return Status::Ok;
        }
    }
    return Status::OutOfRange;
}

void TextureSetMetadata::ClearTextureMetadata()
{
    m_textureInfos.clear();
}

int TextureSetMetadata::GetTextureCount() const
{
    return int(m_textureInfos.size());
}

ITextureMetadata* TextureSetMetadata::GetTexture(int textureIndex)
{
    if (textureIndex < 0 || textureIndex >= int(m_textureInfos.size()))
        return nullptr;

    return m_textureInfos[textureIndex].get();
}

ITextureMetadata const* TextureSetMetadata::GetTexture(int textureIndex) const
{
    if (textureIndex < 0 || textureIndex >= int(m_textureInfos.size()))
        return nullptr;

    return m_textureInfos[textureIndex].get();
}

ColorSpace TextureSetMetadata::GetChannelStorageColorSpace(int channel) const
{
    if (channel < NTC_MAX_CHANNELS)
        return m_channelColorSpaces[channel];
    return ColorSpace::Linear;
}

int TextureSetMetadata::GetNetworkVersion() const
{
    return m_mlpDesc ? m_mlpDesc->networkVersion : NTC_NETWORK_UNKNOWN;
}

Status TextureSetMetadata::GetStreamRangeForLatents(int firstMip, int numMips, StreamRange& outRange) const
{
    if (firstMip < 0 || firstMip >= m_desc.mips)
    {
        SetErrorMessage("firstMip (%d) must be between 0 and %d", firstMip, m_desc.mips - 1);
        return Status::OutOfRange;
    }
    
    if (numMips < 1 || firstMip + numMips > m_desc.mips)
    {
        SetErrorMessage("numMips (%d) must be between 1 and %d", numMips, m_desc.mips - firstMip);
        return Status::OutOfRange;
    }

    int const firstNeuralLod = m_colorMips[firstMip].neuralLod;
    int const lastNeuralLod = m_colorMips[firstMip + numMips - 1].neuralLod;

    uint64_t rangeStart = 0;
    uint64_t rangeEnd = 0;

    auto addView = [&rangeStart, &rangeEnd, this](StreamRange view) {
        uint64_t const viewEnd = view.offset + view.size;
        if (rangeStart != rangeEnd)
        {
            rangeStart = std::min(rangeStart, view.offset);
            rangeEnd = std::max(rangeEnd, viewEnd);
        }
        else
        {
            rangeStart = view.offset;
            rangeEnd = viewEnd;
        }
    };

    for (int neuralLod = firstNeuralLod; neuralLod <= lastNeuralLod; ++neuralLod)
    {
        addView(m_latentImages[neuralLod].highResRange);
        addView(m_latentImages[neuralLod].lowResRange);
    }

    outRange.offset = rangeStart;
    outRange.size = rangeEnd - rangeStart;

    return Status::Ok;
}

Status TextureSetMetadata::GetFusedMipLevels(int mipLevel, int* pOutFirstFusedMip, int* pOutLastFusedMip) const
{
    if (mipLevel < 0 || mipLevel >= m_desc.mips)
    {
        SetErrorMessage("mipLevel (%d) must be between 0 and %d", mipLevel, m_desc.mips - 1);
        return Status::OutOfRange;
    }

    int const neuralLod = m_colorMips[mipLevel].neuralLod;

    if (pOutFirstFusedMip)
    {
        // Go down from mipLevel to find the first mismatching neural LOD index
        // Start with mip 0 in case we don't find a mismatching index.
        *pOutFirstFusedMip = 0;
        for (int i = mipLevel - 1; i >= 0; --i)
        {
            if (m_colorMips[i].neuralLod != neuralLod)
            {
                *pOutFirstFusedMip = i + 1;
                break;
            }
        }
    }

    if (pOutLastFusedMip)
    {
        // Go up from mipLevel to find the first mismatching neural LOD index.
        // Start with the last mip in case we don't find a mismatching index.
        *pOutLastFusedMip = m_desc.mips - 1;
        for (int i = mipLevel + 1; i < m_desc.mips; ++i)
        {
            if (m_colorMips[i].neuralLod != neuralLod)
            {
                *pOutLastFusedMip = i - 1;
                return Status::Ok;
            }
        }
    }

    return Status::Ok;
}

int TextureSetMetadata::GetNumLatentImages() const
{
    return int(m_latentImages.size());
}

Status TextureSetMetadata::GetMipLevelsForLatentImage(int latentImageIndex, int* pOutFirstColorMip, int* pOutLastColorMip) const
{
    if (latentImageIndex < 0 || latentImageIndex >= GetNumLatentImages())
    {
        SetErrorMessage("Invalid latentImageIndex (%d), it must be between 0 and %d",
            latentImageIndex, GetNumLatentImages());
        return Status::OutOfRange;
    }
    int first = -1;
    int last = -1;
    for (int mipLevel = 0; mipLevel < m_desc.mips; ++mipLevel)
    {
        auto const& colorMip = m_colorMips[mipLevel];
        if (colorMip.neuralLod == latentImageIndex)
        {
            if (first < 0)
                first = mipLevel;
            last = mipLevel;
        }
        else if (last >= 0)
        {
            break;
        }
    }
    
    if (first < 0)
    {
        SetErrorMessage("No color MIPs found that are represented by latent image %d", latentImageIndex);
        return Status::OutOfRange;
    }

    if (pOutFirstColorMip)
        *pOutFirstColorMip = first;
    if (pOutLastColorMip)
        *pOutLastColorMip = last;

    return Status::Ok;
}

InferenceWeightType TextureSetMetadata::GetBestSupportedWeightType() const
{
    for (auto type : { InferenceWeightType::CoopVecFP8,
                       InferenceWeightType::CoopVecInt8,
                       InferenceWeightType::GenericInt8 })
    {
        if (IsInferenceWeightTypeSupported(type))
            return type;
    }

    return InferenceWeightType::Unknown;
}

Status TextureSetMetadata::GetInferenceWeights(InferenceWeightType weightType,
    void const** pOutData, size_t* pOutSize, size_t* pOutConvertedSize)
{
    WeightLayout const* layout = m_context->GetWeightLayout(m_mlpDesc->networkVersion, weightType);

    if (!IsInferenceWeightTypeSupported(weightType) || layout == nullptr)
    {
        SetErrorMessage("No weights available for weightType = %s.",
            InferenceWeightTypeToString(weightType));
        return Status::Unsupported;
    }

    void const* pData = nullptr;
    size_t size = 0;
    size_t convertedSize = 0;
    
    switch(weightType)
    {
        case InferenceWeightType::GenericInt8:
            size = m_rowMajorWeightDataInt8.size();
            pData = m_rowMajorWeightDataInt8.data();
            break;
        case InferenceWeightType::GenericFP8:
            size = m_rowMajorWeightDataFP8.size();
            pData = m_rowMajorWeightDataFP8.data();
            break;
        case InferenceWeightType::CoopVecInt8:
            size = m_rowMajorWeightDataInt8.size();
            pData = m_rowMajorWeightDataInt8.data();
            convertedSize = layout->bufferSize;
            break;
        case InferenceWeightType::CoopVecFP8:
            size = m_rowMajorWeightDataFP8.size();
            pData = m_rowMajorWeightDataFP8.data();
            convertedSize = layout->bufferSize;
            break;
        default:
            assert(!"Unsupported value - should be verified by IsInferenceWeightTypeSupported above");
            break;
    }

    if (pOutData) *pOutData = pData;
    if (pOutSize) *pOutSize = size;
    if (pOutConvertedSize) *pOutConvertedSize = convertedSize;

    return Status::Ok;
}

bool TextureSetMetadata::IsInferenceWeightTypeSupported(InferenceWeightType weightType) const
{
    if (CoopVecWeightConverter::IsCoopVecWeightType(weightType))
    {
        // For CoopVec types, check if the source generic data is available and if the conversion is supported
        InferenceWeightType const genericType = CoopVecWeightConverter::GetGenericWeightType(weightType);

        return IsInferenceWeightTypeSupported(genericType)
            && CoopVecWeightConverter::IsConversionSupported(m_context->GetGraphicsResources(), weightType);
    }

    switch(weightType)
    {
    case InferenceWeightType::GenericInt8:
        return !m_rowMajorWeightDataInt8.empty();
    case InferenceWeightType::GenericFP8:
        return !m_rowMajorWeightDataFP8.empty();
    default:
        return false;
    }
}

Status TextureSetMetadata::ShuffleInferenceOutputs(ShuffleSource mapping[NTC_MAX_CHANNELS])
{
    if (mapping == nullptr)
    {
        SetErrorMessage("mapping is NULL");
        return Status::InvalidArgument;
    }

    // Validate the mapping
    for (int ch = 0; ch < NTC_MAX_CHANNELS; ++ch)
    {
        switch(mapping[ch].type)
        {
            case ShuffleSourceType::Undefined:
            case ShuffleSourceType::Channel:
            case ShuffleSourceType::Constant:
                break;
            default:
                SetErrorMessage("mapping[%d] has invalid type %d", ch, int(mapping[ch].type));
                return Status::InvalidArgument;
        }

        if (mapping[ch].type == ShuffleSourceType::Channel &&
            (mapping[ch].channelIndex < 0 || mapping[ch].channelIndex >= NTC_MAX_CHANNELS))
        {
            SetErrorMessage("mapping[%d] is using invalid channel %d, must be between 0 and %d",
                ch, mapping[ch].channelIndex, NTC_MAX_CHANNELS - 1);
            return Status::OutOfRange;
        }
    }

    auto shuffleWeights = [this, mapping](Vector<uint8_t>& data, InferenceWeightType weightType)
    {
        if (data.empty())
            return;

        WeightLayout const* layout = m_context->GetWeightLayout(m_mlpDesc->networkVersion, weightType);
        assert(layout);

        Span const& outputLayerWeights = layout->weights[NTC_MLP_LAYERS - 1];
        Span const& outputLayerScales = layout->scales[NTC_MLP_LAYERS - 1];
        Span const& outputLayerBiases = layout->biases[NTC_MLP_LAYERS - 1];

        int const inputChannels = m_mlpDesc->GetLayerInputChannels(NTC_MLP_LAYERS - 1);
        int const outputChannels = m_mlpDesc->GetLayerOutputChannels(NTC_MLP_LAYERS - 1);

        // Sanity check
        assert(inputChannels == NTC_MLP_HIDDEN_CHANNELS);
        assert(outputChannels == NTC_MLP_OUTPUT_CHANNELS);
        static_assert(NTC_MAX_CHANNELS == NTC_MLP_OUTPUT_CHANNELS,
            "This function assumes that NTC_MAX_CHANNELS == NTC_MLP_OUTPUT_CHANNELS");

        // Output layer for Int8 and FP8 modes has 8-bit weights, Float32 scale and Int32 bias
        assert(outputLayerWeights.type == DataType::FP8 || outputLayerWeights.type == DataType::Int8);
        assert(outputLayerScales.type == DataType::FP32);
        assert(outputLayerBiases.type == DataType::Int32);

        uint8_t* weights = data.data() + outputLayerWeights.offset;
        float* scale = (float*)(data.data() + outputLayerScales.offset);
        int32_t* bias = (int32_t*)(data.data() + outputLayerBiases.offset);
        std::array<uint8_t, NTC_MLP_HIDDEN_CHANNELS * NTC_MLP_OUTPUT_CHANNELS> tmpWeights;
        std::array<float, NTC_MLP_OUTPUT_CHANNELS> tmpScale;
        std::array<int32_t, NTC_MLP_OUTPUT_CHANNELS> tmpBias;

        // Shuffle the row data into 'tmpWeights', scale and bias into 'tmpScale' and 'tmpBias'
        for (int dstRow = 0; dstRow < outputChannels; ++dstRow)
        {
            ShuffleSource& src = mapping[dstRow];
            uint8_t* pDstRow = tmpWeights.data() + inputChannels * dstRow;
            if (src.type == ShuffleSourceType::Channel)
            {
                uint8_t const* pSrcRow = weights + inputChannels * src.channelIndex;
                memcpy(pDstRow, pSrcRow, inputChannels);
                tmpScale[dstRow] = scale[src.channelIndex];
                tmpBias[dstRow] = bias[src.channelIndex];
            }
            else
            {
                // The row will produce zero as the output of the matrix-vector multiplication, plus (bias * scale).
                // Set the scale and bias to produce the constant value.
                memset(pDstRow, 0, inputChannels);
                tmpScale[dstRow] = 1.f / c_ConstantBiasScale;
                tmpBias[dstRow] = (src.type == ShuffleSourceType::Constant) ? int(roundf(src.constantValue * c_ConstantBiasScale)) : 0;
            }
        }

        // Copy the shuffled data back into the MLP vector
        memcpy(weights, tmpWeights.data(), outputLayerWeights.size);
        memcpy(scale, tmpScale.data(), outputLayerScales.size);
        memcpy(bias, tmpBias.data(), outputLayerBiases.size);
    };

    shuffleWeights(m_rowMajorWeightDataInt8, InferenceWeightType::GenericInt8);
    shuffleWeights(m_rowMajorWeightDataFP8, InferenceWeightType::GenericFP8);

    std::array<ColorSpace, NTC_MAX_CHANNELS> newColorSpaces;
    std::array<ShuffleSource, NTC_MAX_CHANNELS> newMapping;

    for (int ch = 0; ch < NTC_MAX_CHANNELS; ++ch)
    {
        switch(mapping[ch].type)
        {
            case ShuffleSourceType::Undefined:
            case ShuffleSourceType::Constant:
                newColorSpaces[ch] = ColorSpace::Linear;
                newMapping[ch] = mapping[ch];
                break;
            case ShuffleSourceType::Channel: {
                int const srcChannel = mapping[ch].channelIndex;
                newColorSpaces[ch] = m_channelColorSpaces[srcChannel];
                newMapping[ch] = m_channelShuffleMapping[srcChannel];
                break;
            }
        }
    }

    m_channelColorSpaces = newColorSpaces;
    m_channelShuffleMapping = newMapping;

    return Status::Ok;
}

Status TextureSetMetadata::LoadMetadataFromStream(json::Document const& document, uint64_t binaryChunkOffset,
        uint64_t binaryChunkSize, LatentShape const& latentShape, IStream* inputStream)
{
    ClearTextureMetadata();

    m_sourceStreamSize = inputStream->Size();
    m_binaryChunkOffset = binaryChunkOffset;
    m_binaryChunkSize = binaryChunkSize;
    m_latentShape = latentShape;
    
    // MLP versions

    if (!document.mlp.has_value() && document.mlpVersions.empty())
    {
        SetErrorMessage("File doesn't contain MLP information");
        return Status::FileUnrecognized;
    }

    if (document.mlp.has_value() && !ValidateMLP(document, *document.mlp))
        return Status::FileUnrecognized;

    for (json::MLP const& mlp : document.mlpVersions)
    {
        if (!ValidateMLP(document, mlp))
            return Status::FileUnrecognized;
    }

    // Texture headers

    for (auto const& jsonTexture : document.textures)
    {    
        ITextureMetadata* texture = AddTexture();
        texture->SetName(jsonTexture.name.c_str());
        texture->SetChannels(jsonTexture.firstChannel, jsonTexture.numChannels);
        texture->SetChannelFormat(jsonTexture.channelFormat.value_or(ChannelFormat::UNORM8));
        texture->SetBlockCompressedFormat(jsonTexture.bcFormat.value_or(BlockCompressedFormat::None));
        texture->SetRgbColorSpace(jsonTexture.rgbColorSpace.value_or(ColorSpace::Linear));
        texture->SetAlphaColorSpace(jsonTexture.alphaColorSpace.value_or(ColorSpace::Linear));
        texture->SetBlockCompressionQuality(jsonTexture.bcQuality.value_or(BlockCompressionMaxQuality));
        if (jsonTexture.bcAccelerationDataView.has_value())
        {
            if (!ValidateBufferView(*jsonTexture.bcAccelerationDataView, 0, document))
                return Status::FileUnrecognized;

            json::BufferView const& accelDataView = document.views[*jsonTexture.bcAccelerationDataView];

            if (!inputStream->Seek(binaryChunkOffset + accelDataView.offset))
                return Status::IOError;

            Vector<uint8_t> bcData(m_allocator);
            bcData.resize(accelDataView.storedSize);

            if (!inputStream->Read(bcData.data(), accelDataView.storedSize))
                return Status::IOError;

            static_cast<TextureMetadata*>(texture)->SetBlockCompressionModeHistogram(bcData.data(), bcData.size());
        }
    }

    // Channel headers

    if (!document.channels.empty())
    {
        if (document.channels.size() != m_desc.channels)
        {
            SetErrorMessage("The file describes %d texture channels while %d are expected",
                document.numChannels, m_desc.channels);
            return Status::FileUnrecognized;
        }

        for (size_t i = 0; i < document.channels.size(); ++i)
        {
            json::Channel const& jsonChannel = document.channels[i];
            m_channelColorSpaces[i] = jsonChannel.colorSpace.value_or(ColorSpace::Linear);
        }
    }

    // Validate the neural LOD headers

    int neuralLod = 0;
    m_latentImages.clear();
    for (const auto& mipHeader : document.latents)
    {
        // Views

        uint64_t const highResSize = (uint64_t(mipHeader.highResWidth * mipHeader.highResHeight) 
            * uint32_t(mipHeader.highResBitsPerPixel) + 7) / 8;

        uint64_t const lowResSize = (uint64_t(mipHeader.lowResWidth * mipHeader.lowResHeight) 
            * uint32_t(mipHeader.lowResBitsPerPixel) + 7) / 8;

        if (!ValidateBufferView(mipHeader.highResView, highResSize, document))
            return Status::FileUnrecognized;
            
        if (!ValidateBufferView(mipHeader.lowResView, lowResSize, document))
            return Status::FileUnrecognized;

        // Packing
        int const highResBitsPerPixel = latentShape.highResFeatures * latentShape.highResQuantBits;
        int const lowResBitsPerPixel = latentShape.lowResFeatures * latentShape.lowResQuantBits;
        if (mipHeader.highResBitsPerPixel != highResBitsPerPixel ||
            mipHeader.lowResBitsPerPixel != lowResBitsPerPixel)
        {
            SetErrorMessage("Neural MIP %d packing strides (%d and %d bits) don't match "
                "the expected strides (%d and %d bits)",
                neuralLod,
                mipHeader.highResBitsPerPixel,
                mipHeader.lowResBitsPerPixel,
                highResBitsPerPixel,
                lowResBitsPerPixel);
            return Status::FileUnrecognized;
        }

        LatentImageDesc& imageDesc = m_latentImages.emplace_back();
        imageDesc.highResRange.offset = document.views[mipHeader.highResView].offset + m_binaryChunkOffset;
        imageDesc.highResRange.size = document.views[mipHeader.highResView].storedSize;
        imageDesc.lowResRange.offset = document.views[mipHeader.lowResView].offset + m_binaryChunkOffset;
        imageDesc.lowResRange.size = document.views[mipHeader.lowResView].storedSize;
        imageDesc.highResWidth = mipHeader.highResWidth;
        imageDesc.highResHeight = mipHeader.highResHeight;
        imageDesc.lowResWidth = mipHeader.lowResWidth;
        imageDesc.lowResHeight = mipHeader.lowResHeight;

        ++neuralLod;
    }

    // Validate the color MIP levels

    int mipLevel = 0;
    m_colorMips.fill(ColorMipDesc());
    for (auto const& colorMip : document.colorMips)
    {
        if (!colorMip.latentMip.has_value())
        {
            SetErrorMessage("Color MIP %d doesn't have a mapping to a latent image, "
                "which is currently unsupported.", mipLevel);
            return Status::FileUnrecognized;
        }

        m_colorMips[mipLevel].neuralLod = *colorMip.latentMip;
        if (colorMip.positionLod.has_value() && colorMip.positionScale.has_value())
        {
            m_colorMips[mipLevel].positionLod = *colorMip.positionLod;
            m_colorMips[mipLevel].positionScale = *colorMip.positionScale;
        }
        else
        {
            // [COMPAT]
            // These parameters are always provided now, but older versions of libntc didn't do that.
            // Reconstruct them using normal rules in this case.
            FeatureGridMath::GetPositionLodAndScale(*colorMip.latentMip, mipLevel,
                m_colorMips[mipLevel].positionLod,
                m_colorMips[mipLevel].positionScale);
        }

        if (colorMip.width.has_value() && colorMip.height.has_value())
        {
            int mipWidth = std::max(m_desc.width >> mipLevel, 1);
            int mipHeight = std::max(m_desc.height >> mipLevel, 1);
            if (*colorMip.width != mipWidth || *colorMip.height != mipHeight)
            {
                SetErrorMessage("Color MIP %d specifies dimensions of %dx%d, "
                    "but it is expected to be %dx%d.", mipLevel,
                    *colorMip.width, *colorMip.height,
                    mipWidth, mipHeight);
                return Status::FileUnrecognized;
            }
        }

        ++mipLevel;
    }

    return Status::Ok;
}

bool TextureSetMetadata::ReadViewFromStream(IStream* stream, json::Document const& document,
     uint32_t view, void* pData, uint64_t size)
{
    json::BufferView const& viewDesc = document.views[view];
    assert(size <= viewDesc.storedSize); // Should be validated by ValidateBufferView before
    return stream->Seek(m_binaryChunkOffset + viewDesc.offset) && stream->Read(pData, size);
}

LatentImageDesc const* TextureSetMetadata::GetLatentImageDesc(int neuralLod) const
{
    if (neuralLod < 0 || neuralLod >= int(m_latentImages.size()))
        return nullptr;

    return &m_latentImages[neuralLod];
}

int TextureSetMetadata::ColorMipToNeuralLod(int colorMip) const
{
    if (colorMip < 0 || colorMip >= m_desc.mips)
        return -1;

    return m_colorMips[colorMip].neuralLod;
}

Status TextureSetMetadata::ConvertInferenceWeights(InferenceWeightType weightType, void* commandList,
    void* srcBuffer, uint64_t srcOffset, void* dstBuffer, uint64_t dstOffset)
{
    if (!commandList)
    {
        SetErrorMessage("commandList is NULL");
        return Status::InvalidArgument;
    }

    if (!srcBuffer)
    {
        SetErrorMessage("srcBuffer is NULL");
        return Status::InvalidArgument;
    }
    
    if (!dstBuffer)
    {
        SetErrorMessage("dstBuffer is NULL");
        return Status::InvalidArgument;
    }

    if (dstBuffer == srcBuffer)
    {
        SetErrorMessage("dstBuffer must not be the same as srcBuffer");
        return Status::InvalidArgument;
    }

    if (!CoopVecWeightConverter::IsCoopVecWeightType(weightType))
    {
        SetErrorMessage("Unsupported weightType (%s), must be one of the CoopVec types",
            InferenceWeightTypeToString(weightType));
        return Status::InvalidArgument;
    }
    
    WeightLayout const* srcLayout = m_context->GetWeightLayout(m_mlpDesc->networkVersion,
        CoopVecWeightConverter::GetGenericWeightType(weightType));

    assert(srcLayout); // Row-major layouts are always available

    WeightLayout const* dstLayout = m_context->GetWeightLayout(m_mlpDesc->networkVersion, weightType);

    if (dstLayout == nullptr)
    {
        SetErrorMessage("The requested conversion operation is not supported by the graphics device or "
            "disabled by the context settings.");
        return Status::Unsupported;
    }

    CoopVecWeightConverter::ConvertWeights(m_context->GetGraphicsResources(), *m_mlpDesc,
        *srcLayout, srcBuffer, srcOffset,
        *dstLayout, dstBuffer, dstOffset, commandList);

    return Status::Ok;
}

// Get the MLP weight type from the legacy descriptor in the MLP or the new descriptor in its first layer
static std::optional<json::MlpDataType> GetRepresentativeMlpWeightType(std::optional<json::MLP> const& mlp)
{
    if (!mlp.has_value())
        return std::nullopt;

    if (!mlp->layers.empty() && mlp->layers[0].weightType.has_value())
        return mlp->layers[0].weightType;

    return mlp->weightType;
}

Status TextureSetMetadata::LoadWeightsFromStream(json::Document const& document, IStream* inputStream)
{
    auto readMlpData = [this, &inputStream, &document]
    (json::MLP const& mlp, Vector<uint8_t>& dst, InferenceWeightType weightType)
    {
        WeightLayout const* weightLayout = m_context->GetWeightLayout(m_mlpDesc->networkVersion, weightType);
        dst.resize(weightLayout->bufferSize);
        
        for (int layerIndex = 0; layerIndex < NTC_MLP_LAYERS; ++layerIndex)
        {
            json::MLPLayer const& layer = mlp.layers[layerIndex];

            Span const& layerWeights = weightLayout->weights[layerIndex];
            Span const& layerScale = weightLayout->scales[layerIndex];
            Span const& layerBias = weightLayout->biases[layerIndex];

            if (!ReadViewFromStream(inputStream, document, layer.weightView,
                dst.data() + layerWeights.offset, layerWeights.size))
                return Status::IOError;

            if (layerScale.size != 0)
            {
                if (layer.scaleView.has_value())
                {
                    if (!ReadViewFromStream(inputStream, document, *layer.scaleView,
                        dst.data() + layerScale.offset, layerScale.size))
                        return Status::IOError;
                }
                else
                {
                    // No scale vectors provided - fill the memory with 1.0
                    assert(layerScale.type == DataType::FP32);

                    float* scales = (float*)(dst.data() + layerScale.offset);
                    for (uint32_t i = 0; i < layer.outputChannels; ++i)
                        scales[i] = 1.0f;
                }
            }

            if (!ReadViewFromStream(inputStream, document, layer.biasView,
                dst.data() + layerBias.offset, layerBias.size))
                return Status::IOError;

            // For INT8 layers in legacy files, convert the float scale and bias from the (float(output) * scale + bias) form
            // to the (float(output + int(bias/scale)) * scale) form
            std::optional<json::MlpDataType> storedBiasType;
            if (layer.biasType.has_value())
                storedBiasType = layer.biasType;
            else if (layer.scaleBiasType.has_value())
                storedBiasType = layer.scaleBiasType;
            else
                storedBiasType = mlp.scaleBiasType;

            if (layerWeights.type == DataType::Int8 &&
                layerBias.type == DataType::Int32 && storedBiasType == json::MlpDataType::Float32 &&
                layerScale.type == DataType::FP32)
            {
                int const inputChannels = m_mlpDesc->GetLayerInputChannels(layerIndex);
                for (uint32_t i = 0; i < layer.outputChannels; ++i)
                {
                    float* const pScale = (float*)(dst.data() + layerScale.offset) + i;
                    
                    // Use void* because we're converting from a float to an int
                    void* pBias = dst.data() + layerBias.offset + i * sizeof(float);

                    // Special case when scale is zero, which can happen on the output layer if an image channel is constant.
                    // See also QuantizeColumnInt8(...) in Quantizer.cu which implements the same logic.
                    if (*pScale == 0.f)
                    {
                        // Zero scale: use a predefined constant scale and express the constant bias using this scale.
                        *pScale = 1.f / c_ConstantBiasScale;
                        *(int*)pBias = int(roundf(*(float*)pBias * c_ConstantBiasScale));

                        // Zero out the weights to produce the correct result.
                        uint8_t* pWeightRow = dst.data() + layerWeights.offset + inputChannels * i;
                        memset(pWeightRow, 0, inputChannels);
                    }
                    else
                    {
                        // Nonzero scale: simple conversion.
                        *(int*)pBias = int(roundf(*(float*)pBias / *pScale));
                    }
                }
            }
        }

        return Status::Ok;
    };

    Status status;

    std::optional<json::MlpDataType> const documentMlpWeightType = GetRepresentativeMlpWeightType(document.mlp);

    if (documentMlpWeightType == json::MlpDataType::Int8)
    {
        status = readMlpData(*document.mlp, m_rowMajorWeightDataInt8, InferenceWeightType::GenericInt8);
        if (status != Status::Ok)
            return status;
    }
    else if (documentMlpWeightType == json::MlpDataType::FloatE4M3)
    {
        status = readMlpData(*document.mlp, m_rowMajorWeightDataFP8, InferenceWeightType::GenericFP8);
        if (status != Status::Ok)
            return status;
    }

    for (json::MLP const& mlp : document.mlpVersions)
    {
        std::optional<json::MlpDataType> const mlpWeightType = GetRepresentativeMlpWeightType(mlp);

        if (mlpWeightType == json::MlpDataType::Int8)
        {
            status = readMlpData(mlp, m_rowMajorWeightDataInt8, InferenceWeightType::GenericInt8);
            if (status != Status::Ok)
                return status;
        }
        else if (mlpWeightType == json::MlpDataType::FloatE4M3)
        {
            status = readMlpData(mlp, m_rowMajorWeightDataFP8, InferenceWeightType::GenericFP8);
            if (status != Status::Ok)
                return status;
        }
    }

    return Status::Ok;
}

void TextureSetMetadata::GetWeightOffsets(InferenceWeightType weightType,
    int weightOffsets[NTC_MLP_LAYERS], int& scaleBiasOffset)
{
    WeightLayout const* layout = m_context->GetWeightLayout(m_mlpDesc->networkVersion, weightType);
    assert(layout); // Support should be validated by the caller

    for (int layer = 0; layer < NTC_MLP_LAYERS; ++layer)
        weightOffsets[layer] = int(layout->weights[layer].offset);
    scaleBiasOffset = int(layout->combinedScaleBias.offset);
}

static uint32_t GetChannelMask(int firstChannel, int numChannels)
{
    return ((1u << numChannels) - 1u) << firstChannel;
}

uint32_t TextureSetMetadata::GetValidChannelMask() const
{
    uint32_t validMask = 0;
    for (auto& texture : m_textureInfos)
    {
        int firstChannel, numChannels;
        texture->GetChannels(firstChannel, numChannels);

        validMask |= GetChannelMask(firstChannel, numChannels);
    }
    
    // Textures are defined in un-shuffled space, but the channels might be shuffled,
    // so shuffle the valid mask as well. Also consider constant outputs as valid.
    uint32_t shuffledMask = 0;
    for (int ch = 0; ch < NTC_MAX_CHANNELS; ++ch)
    {
        ShuffleSource const& src = m_channelShuffleMapping[ch];
        if (src.type == ShuffleSourceType::Channel && (validMask & (1 << src.channelIndex)) != 0 ||
            src.type == ShuffleSourceType::Constant)
        {
            shuffledMask |= (1 << ch);
        }
    }
        
    return shuffledMask;
}

uint32_t TextureSetMetadata::GetPackedColorSpaces() const
{
    // Pack the color space data into the constant, 2 bits per channel.
    // The packing is somewhat fragile; if we add more channels or have more than 4 color spaces, it will have to change.
    static_assert(NTC_MAX_CHANNELS <= 16);
    uint32_t packed = 0;
    for (int channel = 0; channel < NTC_MAX_CHANNELS; ++channel)
    {
        ColorSpace const colorSpace = GetChannelStorageColorSpace(channel);
        assert(int(colorSpace) <= 3);

        packed |= uint32_t(colorSpace) << (2 * channel);
    }
    return packed;
}

void TextureSetMetadata::FillNeuralMipConstants(
    NtcNeuralMipConstants& highResLatents,
    NtcNeuralMipConstants& lowResLatents,
    int neuralLod,
    uint64_t latentBufferOffset)
{
    if (m_latentImages.empty())
        return;

    LatentImageDesc const& latentImage = m_latentImages[neuralLod];

    if (latentImage.highResRange.offset < latentBufferOffset || 
        latentImage.lowResRange.offset < latentBufferOffset)
    {
        // If the neural mip data is out of the available range, silently do nothing.
        // The decompression function validates the necessary range, and the inference function will ignore the issue.
        // TODO: implement binding a subset of the mip chain for inference.
        return;
    }

    highResLatents.dataOffset = uint32_t(latentImage.highResRange.offset - latentBufferOffset);
    highResLatents.imageWidth = uint16_t(latentImage.highResWidth);
    highResLatents.imageHeight = uint16_t(latentImage.highResHeight);
    highResLatents.sliceLeft = 0;
    highResLatents.sliceTop = 0;
    highResLatents.sliceWidth = highResLatents.imageWidth;
    highResLatents.sliceHeight = highResLatents.imageHeight;

    lowResLatents.dataOffset = uint32_t(latentImage.lowResRange.offset - latentBufferOffset);
    lowResLatents.imageWidth = uint16_t(latentImage.lowResWidth);
    lowResLatents.imageHeight = uint16_t(latentImage.lowResHeight);
    lowResLatents.sliceLeft = 0;
    lowResLatents.sliceTop = 0;
    lowResLatents.sliceWidth = lowResLatents.imageWidth;
    lowResLatents.sliceHeight = lowResLatents.imageHeight;
}

void TextureSetMetadata::FillColorMipConstants(
    NtcColorMipConstants& colorMip,
    int mipLevel)
{
    colorMip.neuralMip = m_colorMips[mipLevel].neuralLod;
    colorMip.positionLod = m_colorMips[mipLevel].positionLod;
    colorMip.positionScale = m_colorMips[mipLevel].positionScale;
    colorMip.pad = 0;
}

void TextureSetMetadata::FillDecompressionConstants(
    NtcDecompressConstants& output,
    InferenceWeightType weightType,
    int weightOffset,
    int mipLevel,
    Rect srcRect,
    Point dstOffset,
    OutputTextureDesc const* pOutputTextures,
    int numOutputTextures,
    int firstOutputDescriptorIndex,
    uint64_t latentBufferOffset)
{
    int const mipWidth = std::max(m_desc.width >> mipLevel, 1);
    int const mipHeight = std::max(m_desc.height >> mipLevel, 1);

    output.srcLeft = srcRect.left;
    output.srcTop = srcRect.top;
    output.srcRight = srcRect.left + srcRect.width;
    output.srcBottom = srcRect.top + srcRect.height;
    output.dstLeft = dstOffset.x;
    output.dstTop = dstOffset.y;
    
    // Round down the grid origin to a multiple of 8 to make sure each thread group preloads all the latents
    output.gridLeft = srcRect.left & ~7;
    output.gridTop = srcRect.top & ~7;
    output.imageWidth = mipWidth;
    output.imageHeight = mipHeight;
    
    GetWeightOffsets(weightType, output.networkWeightOffsets, output.networkScaleBiasOffset);
    for (int layer = 0; layer < NTC_MLP_LAYERS; ++layer)
        output.networkWeightOffsets[layer] += weightOffset;
    output.networkScaleBiasOffset += weightOffset;

    // If there are no user-specified outputs, fill out the output descriptors from metadata
    OutputTextureDesc defaultOutputs[DECOMPRESS_CS_MAX_OUTPUTS] {};
    if (!numOutputTextures)
    {
        pOutputTextures = defaultOutputs;
        numOutputTextures = GetTextureCount();

        for (int index = 0; index < numOutputTextures; ++index)
        {
            ITextureMetadata const* src = GetTexture(index);
            OutputTextureDesc& dst = defaultOutputs[index];

            dst.descriptorIndex = index;
            src->GetChannels(dst.firstChannel, dst.numChannels);
            dst.rgbColorSpace = src->GetRgbColorSpace();
            dst.alphaColorSpace = src->GetAlphaColorSpace();

            // Apply dithering to all UNORM8 textures
            dst.ditherScale = src->GetChannelFormat() == ChannelFormat::UNORM8 ? 1.f / 255.f : 0.f;
        }
    }

    // Fill the output constants from either user-specified pOutputTextures or the automatic values
    output.numOutputs = numOutputTextures;
    for (int index = 0; index < numOutputTextures; ++index)
    {
        OutputTextureDesc const& src = pOutputTextures[index];
        NtcDecompressOutputDesc& dst = output.outputs[index];
        
        dst.firstChannel = src.firstChannel;
        dst.numChannels = src.numChannels;
        dst.dstRgbColorSpace = int(src.rgbColorSpace);
        dst.dstAlphaColorSpace = int(src.alphaColorSpace);
        dst.ditherScale = src.ditherScale;
        dst.textureIndex = firstOutputDescriptorIndex + src.descriptorIndex;
        
        int alphaChannel = dst.firstChannel + 3;
        // TODO: validate that all 3 RGB channels have the same storage color space, or support them being different.
        // It would be really weird if they were different, but still valid through the WriteChannels API
        // and that would lead to incorrect output data in the decompression pass.
        dst.srcRgbColorSpace = int(m_channelColorSpaces[dst.firstChannel]);
        dst.srcAlphaColorSpace = (alphaChannel < NTC_MAX_CHANNELS)
            ? int(m_channelColorSpaces[alphaChannel])
            : int(ColorSpace::Linear);
    }

    FillLatentEncodingConstants(output.highResEncoding,
        m_latentShape.highResFeatures, m_latentShape.highResQuantBits, weightType);
    FillLatentEncodingConstants(output.lowResEncoding,
        m_latentShape.lowResFeatures, m_latentShape.lowResQuantBits, weightType);

    int const neuralLod = m_colorMips[mipLevel].neuralLod;
    FillNeuralMipConstants(output.highResNeuralMip, output.lowResNeuralMip, neuralLod, latentBufferOffset);

    FillColorMipConstants(output.colorMip, mipLevel);
}

bool TextureSetMetadata::ValidateBufferView(uint32_t view, uint64_t minSize,
    json::Document const& document)
{
    if (view >= document.views.size())
    {
        SetErrorMessage("Invalid view index %u", view);
        return false;
    }

    json::BufferView const& viewDesc = document.views[view];
    if ((viewDesc.offset & 3) != 0)
    {
        SetErrorMessage("View %u offset %" PRIu64 " is not 4-byte aligned", view, viewDesc.offset);
        return false;
    }

    if (viewDesc.storedSize < minSize)
    {
        SetErrorMessage("View %u size %" PRIu64 " is less than minimum expected size %" PRIu64 ".",
            view, viewDesc.storedSize, minSize);
        return false;
    }

    if (viewDesc.offset + viewDesc.storedSize > m_binaryChunkSize)
    {
        SetErrorMessage("View %u ends at byte offset %" PRIu64 " from the binary chunk start, "
            "which is outside of the chunk size %" PRIu64 ".",
            view, viewDesc.offset + viewDesc.storedSize, m_binaryChunkSize);
        return false;
    }

    return true;
}

static size_t GetMlpDataTypeSize(json::MlpDataType dataType)
{
    switch(dataType)
    {
    case json::MlpDataType::Int8:       return sizeof(uint8_t);
    case json::MlpDataType::FloatE4M3:  return sizeof(uint8_t);
    case json::MlpDataType::FloatE5M2:  return sizeof(uint8_t);
    case json::MlpDataType::Float16:    return sizeof(uint16_t);
    case json::MlpDataType::Float32:    return sizeof(float);
    default: return 0;
    }
}

bool TextureSetMetadata::ValidateMLP(json::Document const& document, json::MLP const& mlp)
{
    if (mlp.layers.size() != NTC_MLP_LAYERS)
    {
        SetErrorMessage("File describes an MLP with %d layers, while only %d layers are supported",
            int(mlp.layers.size()), NTC_MLP_LAYERS);
        return false;
    }

    // Derive the NetworkVersion from the MLP input count

    m_mlpDesc = MlpDesc::FromInputChannels(mlp.layers[0].inputChannels);
    if (!m_mlpDesc)
    {
        SetErrorMessage("File contains MLP weights for %d input channels, "
            "which is an unsupported configuration.",
            mlp.layers[0].inputChannels);
        return false;
    }

    // Validate the MLP geometry

    if (mlp.weightLayout != json::MatrixLayout::RowMajor)
    {
        SetErrorMessage("Only row-major MLP weight layout is supported at this time.");
        return false;
    }

    // We support two MLP configurations:
    // 1. All layers have Int8 weights, Float32 scale, Int32 or Float32 (legacy) bias
    // 2. Layers 0-2 have FP8 weights and Float16 bias;
    //    Layer 3 has Int8 weights, Float32 scale, Int32 or Float32 (legacy) bias
    // Validate that the provided config matches one of these.
    static std::array<json::MlpDataType, NTC_MLP_LAYERS> const c_Int8WeightTypes = {
        json::MlpDataType::Int8, json::MlpDataType::Int8, json::MlpDataType::Int8, json::MlpDataType::Int8 };
    static std::array<std::optional<json::MlpDataType>, NTC_MLP_LAYERS> const c_Int8ScaleTypes = {
        json::MlpDataType::Float32, json::MlpDataType::Float32, json::MlpDataType::Float32, json::MlpDataType::Float32 };
    static std::array<json::MlpDataType, NTC_MLP_LAYERS> const c_Int8BiasTypes = {
        json::MlpDataType::Int32, json::MlpDataType::Int32, json::MlpDataType::Int32, json::MlpDataType::Int32 };
    static std::array<json::MlpDataType, NTC_MLP_LAYERS> const c_Int8LegacyBiasTypes = {
        json::MlpDataType::Float32, json::MlpDataType::Float32, json::MlpDataType::Float32, json::MlpDataType::Float32 };
    static std::array<json::MlpDataType, NTC_MLP_LAYERS> const c_FP8WeightTypes = {
        json::MlpDataType::FloatE4M3, json::MlpDataType::FloatE4M3, json::MlpDataType::FloatE4M3, json::MlpDataType::Int8 };
    static std::array<std::optional<json::MlpDataType>, NTC_MLP_LAYERS> const c_FP8ScaleTypes = {
        std::nullopt, std::nullopt, std::nullopt, json::MlpDataType::Float32 };
    static std::array<json::MlpDataType, NTC_MLP_LAYERS> const c_FP8BiasTypes = {
        json::MlpDataType::Float16, json::MlpDataType::Float16, json::MlpDataType::Float16, json::MlpDataType::Float32 };
    static std::array<json::MlpDataType, NTC_MLP_LAYERS> const c_FP8LegacyBiasTypes = {
        json::MlpDataType::Float16, json::MlpDataType::Float16, json::MlpDataType::Float16, json::MlpDataType::Int32 };

    std::array<json::MlpDataType, NTC_MLP_LAYERS> weightTypes;
    std::array<std::optional<json::MlpDataType>, NTC_MLP_LAYERS> scaleTypes;
    std::array<json::MlpDataType, NTC_MLP_LAYERS> biasTypes;
    for (int layerIndex = 0; layerIndex < NTC_MLP_LAYERS; ++layerIndex)
    {
        json::MLPLayer const& layer = mlp.layers[layerIndex];
        if (layer.weightType.has_value())
            weightTypes[layerIndex] = *layer.weightType;
        else if (mlp.weightType.has_value())
            weightTypes[layerIndex] = *mlp.weightType;
        else
        {
            SetErrorMessage("MLP layer %d doesn't have a weight type defined, "
                "and the MLP doesn't have a default weight type either.", layerIndex);
            return false;
        }

        if (layer.scaleType.has_value())
            scaleTypes[layerIndex] = layer.scaleType;
        else if (layer.scaleBiasType.has_value() && layer.scaleView.has_value())
            scaleTypes[layerIndex] = layer.scaleBiasType;
        else if (mlp.scaleBiasType.has_value() && layer.scaleView.has_value())
            scaleTypes[layerIndex] = mlp.scaleBiasType;

        if (layer.biasType.has_value())
            biasTypes[layerIndex] = *layer.biasType;
        else if (layer.scaleBiasType.has_value())
            biasTypes[layerIndex] = *layer.scaleBiasType;
        else if (mlp.scaleBiasType.has_value())
            biasTypes[layerIndex] = *mlp.scaleBiasType;
        else
        {
            SetErrorMessage("MLP layer %d doesn't have a bias type defined, "
                "and the MLP doesn't have a default bias type either.", layerIndex);
            return false;
        }
    }

    bool const isValidInt8MLP =
        weightTypes == c_Int8WeightTypes &&
        scaleTypes == c_Int8ScaleTypes &&
        (biasTypes == c_Int8BiasTypes || biasTypes == c_Int8LegacyBiasTypes);

    bool const isValidFP8MLP =
        weightTypes == c_FP8WeightTypes &&
        scaleTypes == c_FP8ScaleTypes &&
        (biasTypes == c_FP8BiasTypes || biasTypes == c_FP8LegacyBiasTypes);

    if (!isValidInt8MLP && !isValidFP8MLP)
    {
        SetErrorMessage("Only Int8 weights + Float32 scale + Float32/Int32 bias or FloatE4M3 + Float16 bias"
            " MLP configurations are supported at this time.");
        return false;
    }

    for (int layerIndex = 0; layerIndex < NTC_MLP_LAYERS; ++layerIndex)
    {
        json::MLPLayer const& layer = mlp.layers[layerIndex];

        if (layer.inputChannels != m_mlpDesc->GetLayerInputChannels(layerIndex) ||
            layer.outputChannels != m_mlpDesc->GetLayerOutputChannels(layerIndex))
        {   
            SetErrorMessage("File describes MLP layer %d with %d inputs and %d outputs, "
                "but that layer should have %d inputs and %d outputs.",
                layerIndex,
                layer.inputChannels,
                layer.outputChannels,
                m_mlpDesc->GetLayerInputChannels(layerIndex),
                m_mlpDesc->GetLayerOutputChannels(layerIndex));
            return false;
        }

        size_t const expectedWeightSize = layer.inputChannels * layer.outputChannels *
            GetMlpDataTypeSize(weightTypes[layerIndex]);

        if (!ValidateBufferView(layer.weightView, expectedWeightSize, document))
            return false;

        if (scaleTypes[layerIndex].has_value())
        {
            size_t const expectedScaleSize = layer.outputChannels * GetMlpDataTypeSize(*scaleTypes[layerIndex]);

            if (layer.scaleView.has_value())
            {
                if (!ValidateBufferView(*layer.scaleView, expectedScaleSize, document))
                    return false;
            }
        }

        size_t const expectedBiasSize = layer.outputChannels * GetMlpDataTypeSize(biasTypes[layerIndex]);

        if (!ValidateBufferView(layer.biasView, expectedBiasSize, document))
            return false;
    }

    return true;
}

void TextureSetMetadata::FillLatentEncodingConstants(NtcLatentEncodingConstants& encoding,
    int numFeatures, int quantBits, InferenceWeightType weightType)
{
    QuantizationParameters const quantizationParams = GetLatentQuantization(quantBits);
    
    encoding.numFeatures = numFeatures;
    encoding.quantBits = quantBits;
    encoding.logElementsPerUint = 5 - Log2i(quantBits);
    encoding.pad = 0;

    encoding.addressMask = (1u << encoding.logElementsPerUint) - 1u;
    encoding.dataMask = (1u << encoding.quantBits) - 1u;

    if (weightType == InferenceWeightType::GenericFP8 || weightType == InferenceWeightType::CoopVecFP8)
    {
        // Copy the step and bias parameters (floats) into int variables
        std::memcpy(&encoding.quantizedScale, &quantizationParams.step, sizeof(float));
        std::memcpy(&encoding.quantizedBias, &quantizationParams.bias, sizeof(float));
    }
    else
    {
        // These scale and bias values convert latents from their low-bit quantized representation
        // into the same quantized representation used by the Int8 network inputs.
        // The reason this works at all is a numeric relationship between the quantization scales:
        // - 1-bit latents have scale = 2/3, i.e. we use 3 bins and drop the lowest one
        // - 2-bit latents have scale = 2/5
        // - 4-bit latents have scale = 2/17
        // - 8-bit latents have scale = 2/255 as a special case to be compatible with inputs
        // - network inputs have scale = 2/255 = 2/(3*5*17)
        // So, we just multiply the quantized latents by the product of the remaining 2 factors 
        // to get it to the right scale. The bias is just negative index of the latent bin that contains zero,
        // multiplied by the scale.
        switch(quantBits)
        {
            case 1:
                encoding.quantizedScale = 5 * 17;
                encoding.quantizedBias = 0;
                break;
            case 2:
                encoding.quantizedScale = 3 * 17;
                encoding.quantizedBias = -encoding.quantizedScale;
                break;
            case 4:
                encoding.quantizedScale = 3 * 5;
                encoding.quantizedBias = -7 * encoding.quantizedScale;
                break;
            case 8:
                encoding.quantizedScale = 1;
                encoding.quantizedBias = -126;
                break;
            default:
                assert(!"Unsupported latent quantization bits!");
        }
    }
}

Status TextureSetMetadata::ValidateLatentShape(LatentShape const& latentShape, int networkVersion)
{
    if (latentShape.IsEmpty())
        return Status::Ok;
        
    if (latentShape.highResFeatures < 4 ||
        latentShape.highResFeatures > 16 ||
        (latentShape.highResFeatures & 3) != 0 ||
        latentShape.lowResFeatures < 4 ||
        latentShape.lowResFeatures > 16 ||
        (latentShape.lowResFeatures & 3) != 0)
    {
        SetErrorMessage("Invalid LatentShape: highResFeatures (%d) and lowResFeatures (%d) "
            "must be multiples of 4 between 4 and 16.",
            latentShape.highResFeatures, latentShape.lowResFeatures);
        return Status::OutOfRange;
    }

    if (latentShape.gridSizeScale < 2 || latentShape.gridSizeScale > 8)
    {
        SetErrorMessage("Invalid LatentShape: gridSizeScale (%d) must be between 2 and 8.",
            latentShape.gridSizeScale);
        return Status::OutOfRange;
    }

    constexpr int maxQuantBits = 8;
    if (latentShape.highResQuantBits <= 0 ||
        latentShape.highResQuantBits > maxQuantBits ||
        !IsPowerOf2(latentShape.highResQuantBits) ||
        latentShape.lowResQuantBits <= 0 ||
        latentShape.lowResQuantBits > maxQuantBits ||
        !IsPowerOf2(latentShape.lowResQuantBits))
    {
        SetErrorMessage("Invalid LatentShape: highResQuantBits (%d) and lowResQuantBits (%d) "
            "must be powers of 2 between 1 and %d.",
            latentShape.highResQuantBits, latentShape.lowResQuantBits, maxQuantBits);
        return Status::OutOfRange;
    }

    if (networkVersion != NTC_NETWORK_UNKNOWN)
    {
        MlpDesc const* mlpDesc = MlpDesc::FromNetworkVersion(networkVersion);

        if (!mlpDesc)
        {
            SetErrorMessage("NetworkVersion (%d) must be between %d and %d",
                networkVersion, NTC_NETWORK_UNKNOWN, NTC_NETWORK_XLARGE);
            return Status::OutOfRange;
        }

        if (mlpDesc->highResFeatures < latentShape.highResFeatures ||
            mlpDesc->lowResFeatures < latentShape.lowResFeatures)
        {
            SetErrorMessage("NetworkVersion (%s) is too small for the provided LatentShape: it supports "
                "up to %d high-res and %d low-res features, while the latent shape specifies %d high-res and "
                "%d low-res features.",
                NetworkVersionToString(networkVersion),
                mlpDesc->highResFeatures, mlpDesc->lowResFeatures,
                latentShape.highResFeatures, latentShape.lowResFeatures);
            return Status::OutOfRange;
        }
    }

    return Status::Ok;
}

Status TextureSetMetadata::ValidateTextureSetDesc(const TextureSetDesc& desc)
{
    if (desc.width <= 0 || desc.height <= 0 || desc.channels <= 0 || desc.mips <= 0)
    {
        SetErrorMessage("Invalid TextureSetDesc: width (%d), height (%d), channels (%d) and mips (%d) "
            "must be positive numbers.", desc.width, desc.height, desc.channels, desc.mips);
        return Status::OutOfRange;
    }

    if (desc.channels > NTC_MAX_CHANNELS)
    {
        SetErrorMessage("Invalid TextureSetDesc: too many channels (%d). "
            "Only up to %d channels are supported.", desc.channels, NTC_MAX_CHANNELS);
        return Status::OutOfRange;
    }

    return Status::Ok;
}

Status TextureSetMetadata::DeserializeTextureSetDesc(json::Document const& document, TextureSetDesc& desc,
    LatentShape& latentShape)
{
    desc.width = document.width;
    desc.height = document.height;
    desc.channels = document.numChannels;
    desc.mips = document.numColorMips.value_or(1);
    if (document.latentShape.has_value() && !document.latents.empty())
    {
        latentShape.highResQuantBits = document.latentShape->highResQuantBits;
        latentShape.lowResQuantBits = document.latentShape->lowResQuantBits;
        latentShape.highResFeatures = document.latentShape->highResFeatures;
        latentShape.lowResFeatures = document.latentShape->lowResFeatures;
        
        json::LatentImage const& firstLatents = document.latents[0];
        float const widthRatio = float(desc.width) / float(std::max(firstLatents.highResWidth, 1u));
        float const heightRatio = float(desc.height) / float(std::max(firstLatents.highResHeight, 1u));
        latentShape.gridSizeScale = int(roundf(std::min(widthRatio, heightRatio)));
    }
    else
        latentShape = LatentShape::Empty();

    ntc::Status status = ValidateTextureSetDesc(desc);
    if (status != Status::Ok)
        return status;

    status = ValidateLatentShape(latentShape, NTC_NETWORK_UNKNOWN);
    if (status != Status::Ok)
        return status;

    return Status::Ok;
}

Status TextureSetMetadata::LoadFileHeadersFromStream(IAllocator* allocator, IStream* inputStream,
    json::Document& outDocument, uint64_t& outBinaryChunkOffset, uint64_t& outBinaryChunkSize)
{
    json::FileHeader fileHeader;
    if (!inputStream->Read(&fileHeader, sizeof fileHeader))
    {
        SetErrorMessage("Failed to read the file header - file smaller than %zu bytes?", sizeof(fileHeader));
        return Status::IOError;
    }

    if (fileHeader.signature != json::FileHeader::SignatureValue)
    {
        SetErrorMessage("Unrecognized file format.");
        return Status::FileUnrecognized;
    }

    if (fileHeader.version != json::FileHeader::CurrentVersion)
    {
        SetErrorMessage("Incompatible file format version: expected %d, got %d.",
            json::FileHeader::CurrentVersion, fileHeader.version);
        return Status::FileUnrecognized;
    }

    uint64_t const streamSize = inputStream->Size();
    uint64_t const expectedStreamSize = std::max(
        fileHeader.jsonChunkOffset + fileHeader.jsonChunkSize,
        fileHeader.binaryChunkOffset + fileHeader.binaryChunkSize);
    if (streamSize < expectedStreamSize)
    {
        SetErrorMessage("File incomplete: expected at least %" PRIu64 " bytes, actual size %" PRIu64 " bytes.",
            expectedStreamSize, streamSize);
        return Status::FileUnrecognized;
    }

    if (!inputStream->Seek(fileHeader.jsonChunkOffset))
        return Status::IOError;
    
    Vector<char> jsonData(allocator);
    jsonData.resize(fileHeader.jsonChunkSize + 1);
    if (!inputStream->Read(jsonData.data(), fileHeader.jsonChunkSize))
        return Status::IOError;
    jsonData[fileHeader.jsonChunkSize] = 0;

    String errorMessage(allocator);
    if (!json::ParseDocument(outDocument, jsonData.data(), errorMessage))
    {
        SetUnformattedErrorMessage(errorMessage.c_str());
        return Status::FileUnrecognized;
    }

    if (outDocument.schemaVersion != json::Document::SchemaVersion)
    {
        SetErrorMessage("Incompatible file schema version: expected %u, got %u.",
            json::Document::SchemaVersion, outDocument.schemaVersion);
        return Status::FileUnrecognized;
    }

    outBinaryChunkOffset = fileHeader.binaryChunkOffset;
    outBinaryChunkSize = fileHeader.binaryChunkSize;

    return Status::Ok;
}

}
