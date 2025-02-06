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

#include "TextureMetadata.h"
#include "TextureSetMetadata.h"
#include "Errors.h"
#include <libntc/shaders/BlockCompressConstants.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

namespace ntc
{

TextureMetadata::TextureMetadata(IAllocator* allocator, TextureSetMetadata* parent)
    : m_allocator(allocator)
    , m_parent(parent)
    , m_name(allocator)
    , m_bcHistogram(allocator)
{
}

void TextureMetadata::SetName(const char* name)
{
    m_name = String(name, m_allocator);
}

const char* TextureMetadata::GetName() const
{
    return m_name.c_str();
}

Status TextureMetadata::SetChannels(int firstChannel, int numChannels)
{
    const TextureSetDesc textureSetDesc = m_parent->GetDesc();

    if (firstChannel < 0 || firstChannel >= textureSetDesc.channels)
    {
        SetErrorMessage("firstChannel (%d) must be between 0 and %d.", firstChannel, textureSetDesc.channels - 1);
        return Status::OutOfRange;
    }

    if (numChannels < 1 || numChannels + firstChannel > textureSetDesc.channels)
    {
        SetErrorMessage("For the provided firstChannel (%d), numChannels (%d) must be between 1 and %d.",
            firstChannel, numChannels, textureSetDesc.channels - firstChannel);
        return Status::OutOfRange;
    }

    m_firstChannel = firstChannel;
    m_numChannels = numChannels;
    return Status::Ok;
}

void TextureMetadata::GetChannels(int& outFirstChannel, int& outNumChannels) const
{
    outFirstChannel = m_firstChannel;
    outNumChannels = m_numChannels;
}

void TextureMetadata::SetRgbColorSpace(ColorSpace colorSpace)
{
    m_rgbColorSpace = colorSpace;
}

ColorSpace TextureMetadata::GetRgbColorSpace() const
{
    return m_rgbColorSpace;
}

void TextureMetadata::SetAlphaColorSpace(ColorSpace colorSpace)
{
    m_alphaColorSpace = colorSpace;
}

ColorSpace TextureMetadata::GetAlphaColorSpace() const
{
    return m_alphaColorSpace;
}

Status TextureMetadata::SetBlockCompressionAccelerationData(void const* pData, size_t size)
{
    if (!pData)
    {
        SetErrorMessage("pData is NULL");
        return Status::InvalidArgument;
    }

    if (size != BlockCompressionAccelerationBufferSize)
    {
        SetErrorMessage("Invalid size of acceleration data (%zu), must be %zu", size, BlockCompressionAccelerationBufferSize);
        return Status::InvalidArgument;
    }

    uint32_t const* pUintData = static_cast<uint32_t const*>(pData);
    uint32_t nonzeroCount = 0;
    uint32_t sumOfStats = 0;
    for (uint32_t index = 0; index < BlockCompressionAccelerationBufferSize / sizeof(uint32_t); ++index)
    {
        if (!pUintData[index])
            continue;

        ++nonzeroCount;
        sumOfStats += pUintData[index];
    }

    if (nonzeroCount == 0)
    {
        SetErrorMessage("Acceleration data buffer is empty (all zeros)");
        return Status::InvalidArgument;
    }

    m_bcHistogram.resize(nonzeroCount);
    uint32_t histogramIndex = 0;
    for (uint32_t index = 0; index < BlockCompressionAccelerationBufferSize / sizeof(uint32_t); ++index)
    {
        if (!pUintData[index])
            continue;

        BCHistogramEntry& entry = m_bcHistogram[histogramIndex];
        entry.mode = uint8_t(index >> 6);
        entry.partition = uint8_t(index & 63);
        entry.frequency = uint16_t(std::clamp(int(65535.f * float(pUintData[index]) / float(sumOfStats)), 1, 0xffff));
        ++histogramIndex;
    }

    std::sort(m_bcHistogram.begin(), m_bcHistogram.end(), [](BCHistogramEntry const& a, BCHistogramEntry const& b) { return a.frequency > b.frequency; });
    
    return Status::Ok;
}

bool TextureMetadata::HasBlockCompressionAccelerationData() const
{
    return !m_bcHistogram.empty();
}

void TextureMetadata::SetBlockCompressionQuality(uint8_t quality)
{
    m_bcQuality = quality;
}

uint8_t TextureMetadata::GetBlockCompressionQuality() const
{
    return m_bcQuality;
}

void TextureMetadata::GetBlockCompressionModeHistogram(void const** ppData, size_t* pSize) const
{
    if (ppData) *ppData = m_bcHistogram.data();
    if (pSize) *pSize = m_bcHistogram.size() * sizeof(BCHistogramEntry);
}

void TextureMetadata::SetBlockCompressionModeHistogram(void const* pData, size_t size)
{
    if ((size % sizeof(BCHistogramEntry)) != 0)
        return;

    m_bcHistogram.resize(size / sizeof(BCHistogramEntry));

    if (size != 0)
        memcpy(m_bcHistogram.data(), pData, size);
}

bool TextureMetadata::GetAllowedBCModes(uint32_t* pData, size_t size, uint8_t quality) const
{
    if (m_bcHistogram.empty())
        return false;

    if (size != BLOCK_COMPRESS_MODE_MASK_UINTS * sizeof(uint32_t))
        return false;
    
    uint32_t count = std::max(1u, uint32_t(roundf(float(m_bcHistogram.size()) * float(quality) / 255.f)));
    assert(count <= m_bcHistogram.size());

    memset(pData, 0, size);
    for (uint32_t index = 0; index < count; ++index)
    {
        BCHistogramEntry const& entry = m_bcHistogram[index];
        uint32_t modePartition = (uint32_t(entry.mode) << 6) | uint32_t(entry.partition);
        pData[modePartition >> 5] |= (1 << (modePartition & 31));
    }
    return true;
}

}