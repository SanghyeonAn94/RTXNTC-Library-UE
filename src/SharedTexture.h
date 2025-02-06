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
#include <array>
#include <cuda_runtime_api.h>

namespace ntc
{

class SharedTexture : public ISharedTexture
{
public:
    SharedTexture(const SharedTextureDesc& desc);
    ~SharedTexture() override;

    const SharedTextureDesc& GetDesc() const override;
    cudaSurfaceObject_t GetSurfaceObject(int mip) const;
    int GetPixelStride() const;
    Status Initialize();

private:
    SharedTextureDesc m_desc;
    int m_pixelStride = 0;
    cudaExternalMemory_t m_externalMemory = nullptr;
    cudaMipmappedArray_t m_mipmappedArray = nullptr;
    std::array<cudaArray_t, NTC_MAX_MIPS> m_mipArrays{};
    std::array<cudaSurfaceObject_t, NTC_MAX_MIPS> m_mipSurfaces{};
};

}
