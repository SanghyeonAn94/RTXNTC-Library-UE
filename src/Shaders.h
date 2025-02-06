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

namespace ntc
{

enum class InferenceMath
{
    Legacy,
    DP4aNoFloat16,
    DP4aWithFloat16,
    CoopVecInt8,
    CoopVecFP8
};

struct MlpDesc;

#if NTC_WITH_PREBUILT_SHADERS
#if NTC_WITH_DX12
void GetDecompressDxilShaderBytecode(MlpDesc const* mlpDesc, InferenceMath mathVersion, bool preloadLatents, const void** pOutData, size_t* pOutSize);
void GetBlockCompressDxilShaderBytecode(BlockCompressedFormat format, bool writeAccelerationData, const void** pOutData, size_t* pOutSize);
void GetImageDifferenceDxilShaderBytecode(const void** pOutData, size_t* pOutSize);
#endif
#if NTC_WITH_VULKAN
void GetDecompressSpirvShaderBytecode(MlpDesc const* mlpDesc, InferenceMath mathVersion, bool preloadLatents, const void** pOutData, size_t* pOutSize);
void GetBlockCompressSpirvShaderBytecode(BlockCompressedFormat format, bool writeAccelerationData, const void** pOutData, size_t* pOutSize);
void GetImageDifferenceSpirvShaderBytecode(const void** pOutData, size_t* pOutSize);
#endif
#endif

}
