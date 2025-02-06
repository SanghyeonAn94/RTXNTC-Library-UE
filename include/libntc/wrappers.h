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

namespace ntc::utils
{

// Universal RAII-style wrapper NTC resources, closes or deletes the resource on destruction using the provided context.
// See below for some useful typedef's on this class like FileStreamWrapper.
template<typename Resource, typename Deleter>
class UniversalWrapper
{
public:
    UniversalWrapper(IContext const* context)
        : m_context(context)
    { }

    ~UniversalWrapper()
    {
        Close();
    }

    operator Resource* () const
    {
        return m_resource;
    }

    Resource* Get() const
    {
        return m_resource;
    }

    Resource** ptr()
    {
        return &m_resource;
    }

    Resource* operator->() const
    {
        return m_resource;
    }

    void Close()
    {
        if (m_resource)
        {
            Deleter::Delete(m_context, m_resource);
            m_resource = nullptr;
        }
    }

    void Detach()
    {
        m_resource = nullptr;
    }

    UniversalWrapper(const UniversalWrapper&) = delete;
    UniversalWrapper operator=(const UniversalWrapper&) = delete;

    UniversalWrapper(UniversalWrapper&& other)
    {
        m_context = other.m_context;
        m_resource = other.m_resource;
        other.m_resource = nullptr;
    }

    UniversalWrapper& operator=(UniversalWrapper&& other)
    {
        m_context = other.m_context;
        m_resource = other.m_resource;
        other.m_resource = nullptr;
        return *this;
    }

private:
    IContext const* m_context = nullptr;
    Resource* m_resource = nullptr;
};

struct FileStreamDeleter
{
    static void Delete(IContext const* context, IStream* stream)
    {
        context->CloseFile(stream);
    }
};

struct MemoryStreamDeleter
{
    static void Delete(IContext const* context, IStream* stream)
    {
        context->CloseMemory(stream);
    }
};

struct TextureSetMetadataDeleter
{
    static void Delete(IContext const* context, ITextureSetMetadata* textureSetMetadata)
    {
        context->DestroyTextureSetMetadata(textureSetMetadata);
    }
};

struct TextureSetDeleter
{
    static void Delete(IContext const* context, ITextureSet* textureSet)
    {
        context->DestroyTextureSet(textureSet);
    }
};

struct SharedTextureDeleter
{
    static void Delete(IContext const* context, ISharedTexture* sharedTexture)
    {
        context->ReleaseSharedTexture(sharedTexture);
    }
};

struct AdaptiveCompressionSessionDeleter
{
    static void Delete(IContext const* context, IAdaptiveCompressionSession* session)
    {
        context->DestroyAdaptiveCompressionSession(session);
    }
};

} // namespace ntc::utils

namespace ntc
{
typedef utils::UniversalWrapper<IStream, utils::FileStreamDeleter> FileStreamWrapper;
typedef utils::UniversalWrapper<IStream, utils::MemoryStreamDeleter> MemoryStreamWrapper;
typedef utils::UniversalWrapper<ITextureSetMetadata, utils::TextureSetMetadataDeleter> TextureSetMetadataWrapper;
typedef utils::UniversalWrapper<ITextureSet, utils::TextureSetDeleter> TextureSetWrapper;
typedef utils::UniversalWrapper<ISharedTexture, utils::SharedTextureDeleter> SharedTextureWrapper;
typedef utils::UniversalWrapper<IAdaptiveCompressionSession, utils::AdaptiveCompressionSessionDeleter>
    AdaptiveCompressionSessionWrapper;

// RAII-style wrapper for IContext 
class ContextWrapper
{
public:
    ContextWrapper() = default;

    ~ContextWrapper()
    {
        Release();
    }

    operator IContext* () const
    {
        return m_context;
    }

    IContext* Get() const
    {
        return m_context;
    }

    IContext** ptr()
    {
        return &m_context;
    }

    IContext* operator->() const
    {
        return m_context;
    }

    void Release()
    {
        if (m_context)
        {
            ntc::DestroyContext(m_context);
            m_context = nullptr;
        }
    }

    void Detach()
    {
        m_context = nullptr;
    }

    ContextWrapper(const ContextWrapper&) = delete;
    ContextWrapper operator=(const ContextWrapper&) = delete;

    ContextWrapper(ContextWrapper&& other)
    {
        m_context = other.m_context;
        other.m_context = nullptr;
    }

    ContextWrapper& operator=(ContextWrapper&& other)
    {
        m_context = other.m_context;
        other.m_context = nullptr;
        return *this;
    }

private:
    IContext* m_context = nullptr;
};
}
