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

#include "Shaders.h"
#include "MlpDesc.h"
#include <ShaderMake/ShaderBlob.h>

#if NTC_WITH_DX12
#include <DecompressINT8.dxil.h>
#include <DecompressCoopVecInt8.dxil.h>
#include <DecompressCoopVecFP8.dxil.h>
#include <CompressBC1.dxil.h>
#include <CompressBC2.dxil.h>
#include <CompressBC3.dxil.h>
#include <CompressBC4.dxil.h>
#include <CompressBC5.dxil.h>
#include <CompressBC6.dxil.h>
#include <CompressBC7.dxil.h>
#include <ImageDifference.dxil.h>
#endif
#if NTC_WITH_VULKAN
#include <DecompressINT8.spirv.h>
#include <DecompressCoopVecInt8.spirv.h>
#include <DecompressCoopVecFP8.spirv.h>
#include <CompressBC1.spirv.h>
#include <CompressBC2.spirv.h>
#include <CompressBC3.spirv.h>
#include <CompressBC4.spirv.h>
#include <CompressBC5.spirv.h>
#include <CompressBC6.spirv.h>
#include <CompressBC7.spirv.h>
#include <ImageDifference.spirv.h>
#endif

namespace ntc
{
void GetDecompressShaderBytecode(const uint8_t* blobData, size_t blobSize, MlpDesc const* mlpDesc,
    InferenceMath mathVersion, bool preloadLatents, const void** pOutData, size_t* pOutSize)
{
    ShaderMake::ShaderConstant constants[] = {
        { "NETWORK_VERSION", NetworkVersionToString(mlpDesc->networkVersion) },
        { "PRELOAD_LATENTS", preloadLatents ? "1" : "0" },
        { "USE_DP4A", (mathVersion == InferenceMath::DP4aWithFloat16 || mathVersion == InferenceMath::DP4aNoFloat16) ? "1": "0" },
        { "USE_FLOAT16", (mathVersion == InferenceMath::DP4aWithFloat16) ? "1": "0" },
    };

    // No USE_DP4A or USE_FLOAT16 in the CoopVec shader
    int const numConstants = (mathVersion == InferenceMath::CoopVecInt8 || mathVersion == InferenceMath::CoopVecFP8) ? 2 : 4;

    ShaderMake::FindPermutationInBlob(blobData, blobSize, constants, numConstants, pOutData, pOutSize);
}

void GetBC7ShaderBytecode(const uint8_t* blobData, size_t blobSize, bool writeAccelerationData,
    const void** pOutData, size_t* pOutSize)
{
    ShaderMake::ShaderConstant constants[] = {
        { "WRITE_ACCELERATION", writeAccelerationData ? "1" : "0" }
    };

    ShaderMake::FindPermutationInBlob(blobData, blobSize,
        constants, sizeof(constants) / sizeof(constants[0]), pOutData, pOutSize);
}

#define SET_SHADER_BYTECODE(symbol) \
    *pOutData = symbol; \
    *pOutSize = sizeof(symbol);

#if NTC_WITH_DX12
void GetDecompressDxilShaderBytecode(MlpDesc const* mlpDesc, InferenceMath mathVersion, bool preloadLatents, const void** pOutData, size_t* pOutSize)
{
    if (mathVersion == InferenceMath::CoopVecInt8)
        GetDecompressShaderBytecode(g_DecompressCoopVecInt8_dxil, sizeof(g_DecompressCoopVecInt8_dxil), mlpDesc, mathVersion, preloadLatents, pOutData, pOutSize);
    else if (mathVersion == InferenceMath::CoopVecFP8)
        GetDecompressShaderBytecode(g_DecompressCoopVecFP8_dxil, sizeof(g_DecompressCoopVecFP8_dxil), mlpDesc, mathVersion, preloadLatents, pOutData, pOutSize);
    else
        GetDecompressShaderBytecode(g_DecompressINT8_dxil, sizeof(g_DecompressINT8_dxil), mlpDesc, mathVersion, preloadLatents, pOutData, pOutSize);
}

void GetBlockCompressDxilShaderBytecode(BlockCompressedFormat format, bool writeAccelerationData, const void** pOutData, size_t* pOutSize)
{
    switch(format)
    {
        case BlockCompressedFormat::BC1:
            SET_SHADER_BYTECODE(g_CompressBC1_dxil);
            break;
        case BlockCompressedFormat::BC2:
            SET_SHADER_BYTECODE(g_CompressBC2_dxil);
            break;
        case BlockCompressedFormat::BC3:
            SET_SHADER_BYTECODE(g_CompressBC3_dxil);
            break;
        case BlockCompressedFormat::BC4:
            SET_SHADER_BYTECODE(g_CompressBC4_dxil);
            break;
        case BlockCompressedFormat::BC5:
            SET_SHADER_BYTECODE(g_CompressBC5_dxil);
            break;
        case BlockCompressedFormat::BC6:
            SET_SHADER_BYTECODE(g_CompressBC6_dxil);
            break;
        case BlockCompressedFormat::BC7:
            GetBC7ShaderBytecode(g_CompressBC7_dxil, sizeof(g_CompressBC7_dxil), writeAccelerationData, pOutData, pOutSize);
            break;
    }
}
void GetImageDifferenceDxilShaderBytecode(const void** pOutData, size_t* pOutSize)
{
    SET_SHADER_BYTECODE(g_ImageDifference_dxil);
}
#endif

#if NTC_WITH_VULKAN
void GetDecompressSpirvShaderBytecode(MlpDesc const* mlpDesc, InferenceMath mathVersion, bool preloadLatents, const void** pOutData, size_t* pOutSize)
{
    if (mathVersion == InferenceMath::CoopVecInt8)
        GetDecompressShaderBytecode(g_DecompressCoopVecInt8_spirv, sizeof(g_DecompressCoopVecInt8_spirv), mlpDesc, mathVersion, preloadLatents, pOutData, pOutSize);
    else if (mathVersion == InferenceMath::CoopVecFP8)
        GetDecompressShaderBytecode(g_DecompressCoopVecFP8_spirv, sizeof(g_DecompressCoopVecFP8_spirv), mlpDesc, mathVersion, preloadLatents, pOutData, pOutSize);
    else
        GetDecompressShaderBytecode(g_DecompressINT8_spirv, sizeof(g_DecompressINT8_spirv), mlpDesc, mathVersion, preloadLatents, pOutData, pOutSize);
}

void GetBlockCompressSpirvShaderBytecode(BlockCompressedFormat format, bool writeAccelerationData, const void** pOutData, size_t* pOutSize)
{
    switch (format)
    {
        case BlockCompressedFormat::BC1:
            SET_SHADER_BYTECODE(g_CompressBC1_spirv);
            break;
        case BlockCompressedFormat::BC2:
            SET_SHADER_BYTECODE(g_CompressBC2_spirv);
            break;
        case BlockCompressedFormat::BC3:
            SET_SHADER_BYTECODE(g_CompressBC3_spirv);
            break;
        case BlockCompressedFormat::BC4:
            SET_SHADER_BYTECODE(g_CompressBC4_spirv);
            break;
        case BlockCompressedFormat::BC5:
            SET_SHADER_BYTECODE(g_CompressBC5_spirv);
            break;
        case BlockCompressedFormat::BC6:
            SET_SHADER_BYTECODE(g_CompressBC6_spirv);
            break;
        case BlockCompressedFormat::BC7:
            GetBC7ShaderBytecode(g_CompressBC7_spirv, sizeof(g_CompressBC7_spirv), writeAccelerationData, pOutData, pOutSize);
            break;
    }
}
void GetImageDifferenceSpirvShaderBytecode(const void** pOutData, size_t* pOutSize)
{
    SET_SHADER_BYTECODE(g_ImageDifference_spirv);
}
#endif
}