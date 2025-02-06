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

#include <libntc/ntc.h>
#include "StdTypes.h"

namespace ntc
{

class TextureSetMetadata;

class TextureMetadata : public ITextureMetadata
{
public:
    TextureMetadata(IAllocator* allocator, TextureSetMetadata* parent);

    void SetName(const char* name) override;
    char const* GetName() const override;
    String const& GetNameString() const { return m_name; }

    Status SetChannels(int firstChannel, int numChannels) override;
    void GetChannels(int& outFirstChannel, int& outNumChannels) const override;
    int GetFirstChannel() const override { return m_firstChannel; }
    int GetNumChannels() const override { return m_numChannels; }

    void SetChannelFormat(ChannelFormat format) override { m_channelFormat = format; }
    ChannelFormat GetChannelFormat() const override { return m_channelFormat; }

    void SetBlockCompressedFormat(BlockCompressedFormat format) override { m_bcFormat = format; }
    BlockCompressedFormat GetBlockCompressedFormat() const override { return m_bcFormat; }

    void SetRgbColorSpace(ColorSpace colorSpace) override;
    ColorSpace GetRgbColorSpace() const override;
    
    void SetAlphaColorSpace(ColorSpace colorSpace) override;
    ColorSpace GetAlphaColorSpace() const override;

    Status SetBlockCompressionAccelerationData(void const* pData, size_t size) override;
    bool HasBlockCompressionAccelerationData() const override;
    void SetBlockCompressionQuality(uint8_t quality) override;
    uint8_t GetBlockCompressionQuality() const override;

    void GetBlockCompressionModeHistogram(void const** ppData, size_t* pSize) const;
    void SetBlockCompressionModeHistogram(void const* pData, size_t size);
    bool GetAllowedBCModes(uint32_t* pData, size_t size, uint8_t quality) const;
    
private:
    struct BCHistogramEntry
    {
        uint8_t mode;
        uint8_t partition;
        uint16_t frequency;
    };

    static_assert(sizeof(BCHistogramEntry) == 4, "BCHistogramEntry is supposed to pack into a uint32_t");

    IAllocator* m_allocator;
    TextureSetMetadata* m_parent;
    String m_name;
    int m_firstChannel = 0;
    int m_numChannels = 0;
    ChannelFormat m_channelFormat = ChannelFormat::UNORM8;
    BlockCompressedFormat m_bcFormat = BlockCompressedFormat::None;
    ColorSpace m_rgbColorSpace = ColorSpace::Linear;
    ColorSpace m_alphaColorSpace = ColorSpace::Linear;
    Vector<BCHistogramEntry> m_bcHistogram;
    uint8_t m_bcQuality = BlockCompressionMaxQuality;
};


}