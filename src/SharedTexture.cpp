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

#include "SharedTexture.h"
#include "Errors.h"

namespace ntc
{

SharedTexture::SharedTexture(const SharedTextureDesc& desc)
    : m_desc(desc)
{
}

SharedTexture::~SharedTexture()
{
    for (int mip = 0; mip < m_desc.mips; ++mip)
    {
        if (m_mipSurfaces[mip])
        {
            cudaDestroySurfaceObject(m_mipSurfaces[mip]);
            m_mipSurfaces[mip] = 0;
        }

        // No function to release cudaArray's, is that right?
    }

    // No function to release mipmappedArray either

    if (m_externalMemory)
    {
        cudaDestroyExternalMemory(m_externalMemory);
        m_externalMemory = nullptr;
    }
}

const SharedTextureDesc& SharedTexture::GetDesc() const
{
    return m_desc;
}

cudaSurfaceObject_t SharedTexture::GetSurfaceObject(int mip) const
{
    if (mip < 0 || mip > NTC_MAX_MIPS)
        return 0;

    return m_mipSurfaces[mip];
}

int SharedTexture::GetPixelStride() const
{
    return m_pixelStride;
}

Status SharedTexture::Initialize()
{
    if (!m_desc.sharedHandle)
    {
        SetErrorMessage("SharedTextureDesc.sharedHandle is NULL.");
        return Status::InvalidArgument;
    }

    if (m_desc.width <= 0 || m_desc.height <= 0)
    {
        SetErrorMessage("SharedTextureDesc.width (%d) and height (%d) must be positive.", m_desc.width, m_desc.height);
        return Status::OutOfRange;
    }

    if (m_desc.channels <= 0 || m_desc.channels > 4)
    {
        SetErrorMessage("SharedTextureDesc.channels (%d) must be between 1 and 4.", m_desc.channels);
        return Status::OutOfRange;
    }

    if (m_desc.mips <= 0 || m_desc.mips > NTC_MAX_MIPS)
    {
        SetErrorMessage("SharedTextureDesc.mips (%d) must be between 1 and NTC_MAX_MIPS (%d).", m_desc.mips, NTC_MAX_MIPS);
        return Status::OutOfRange;
    }
    
    if (m_desc.sizeInBytes == 0)
    {
        SetErrorMessage("SharedTextureDesc.sizeInBytes (%zu) must be positive.", m_desc.sizeInBytes);
        return Status::OutOfRange;
    }

    cudaExternalMemoryHandleDesc emhd{};
    switch (m_desc.handleType)
    {
        case SharedHandleType::D3D12Resource:
            emhd.type = cudaExternalMemoryHandleTypeD3D12Resource;
            emhd.handle.win32.handle = reinterpret_cast<void*>(m_desc.sharedHandle);
            break;
        case SharedHandleType::OpaqueWin32:
            emhd.type = cudaExternalMemoryHandleTypeOpaqueWin32;
            emhd.handle.win32.handle = reinterpret_cast<void*>(m_desc.sharedHandle);
            break;
        case SharedHandleType::OpaqueFd:
            emhd.type = cudaExternalMemoryHandleTypeOpaqueFd;
            emhd.handle.fd = static_cast<int>(m_desc.sharedHandle);
            break;
        default:
            return Status::InvalidArgument;
    }
    emhd.size = m_desc.sizeInBytes;
    emhd.flags = m_desc.dedicatedResource ? cudaExternalMemoryDedicated : 0;

    cudaError_t err = cudaImportExternalMemory(&m_externalMemory, &emhd);
    if (err != cudaSuccess)
    {
        SetCudaErrorMessage("cudaImportExternalMemory", err);
        return Status::CudaError;
    }


    cudaExternalMemoryMipmappedArrayDesc emmad{};
    emmad.extent.width = m_desc.width;
    emmad.extent.height = m_desc.height;
    emmad.extent.depth = 0; // Note: 3D textures are not supported, and 0 means it's a 2D texture in block-linear layout (on Vulkan)
    int bits = 0;
    switch (m_desc.format)
    {
        case ChannelFormat::UNORM8:
            switch (m_desc.channels)
            {
                case 1: emmad.formatDesc.f = cudaChannelFormatKindUnsignedNormalized8X1; break;
                case 2: emmad.formatDesc.f = cudaChannelFormatKindUnsignedNormalized8X2; break;
                case 4: emmad.formatDesc.f = cudaChannelFormatKindUnsignedNormalized8X4; break;
                default: return Status::OutOfRange;
            }
            bits = 8;
            break;
        case ChannelFormat::UNORM16:
            emmad.formatDesc.f = cudaChannelFormatKindUnsigned;
            bits = 16;
            break;
        case ChannelFormat::FLOAT16:
            emmad.formatDesc.f = cudaChannelFormatKindFloat;
            bits = 16;
            break;
        case ChannelFormat::FLOAT32:
            emmad.formatDesc.f = cudaChannelFormatKindFloat;
            bits = 32;
            break;
        case ChannelFormat::UINT32:
            emmad.formatDesc.f = cudaChannelFormatKindUnsigned;
            bits = 32;
            break;
        default:
            SetErrorMessage("SharedTextureDesc.format (%d) is an unsupported value.", m_desc.format);
            return Status::OutOfRange;
    }
    emmad.formatDesc.x = bits;
    if (m_desc.channels >= 2)
        emmad.formatDesc.y = bits;
    if (m_desc.channels >= 3)
        emmad.formatDesc.z = bits;
    if (m_desc.channels >= 4)
        emmad.formatDesc.w = bits;
    emmad.numLevels = m_desc.mips;
    emmad.flags = cudaArraySurfaceLoadStore;

    err = cudaExternalMemoryGetMappedMipmappedArray(&m_mipmappedArray, m_externalMemory, &emmad);
    if (err != cudaSuccess)
    {
        SetCudaErrorMessage("cudaExternalMemoryGetMappedMipmappedArray", err);
        return Status::CudaError;
    }

    m_pixelStride = (bits * m_desc.channels) / 8;


    for (int mip = 0; mip < m_desc.mips; ++mip)
    {
        err = cudaGetMipmappedArrayLevel(&m_mipArrays[mip], m_mipmappedArray, mip);
        if (err != cudaSuccess)
        {
            SetCudaErrorMessage("cudaGetMipmappedArrayLevel", err);
            return Status::CudaError;
        }

        cudaResourceDesc rd{};
        rd.resType = cudaResourceTypeArray;
        rd.res.array.array = m_mipArrays[mip];
        err = cudaCreateSurfaceObject(&m_mipSurfaces[mip], &rd);
        if (err != cudaSuccess)
        {
            SetCudaErrorMessage("cudaCreateSurfaceObject", err);
            return Status::CudaError;
        }
    }

    ClearErrorMessage();
    return Status::Ok;
}

}