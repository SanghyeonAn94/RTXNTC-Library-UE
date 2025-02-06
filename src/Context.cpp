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

#include "AdaptiveCompressionSession.h"
#include "Context.h"
#include "CoopVecWeightConverter.h"
#include "CudaDeviceGuard.h"
#include "Errors.h"
#include "GraphicsResources.h"
#include "JsonFileFormat.h"
#include "MathUtils.h"
#include "Shaders.h"
#include "Stream.h"
#include "TextureMetadata.h"
#include "TextureSetMetadata.h"

#if NTC_WITH_CUDA
#include "Regression.h"
#include "SharedTexture.h"
#include "TextureSet.h"
#endif

#include <cinttypes>
#include <cassert>
#include <cmath>
#include <cstring>

#include <libntc/shaders/DecompressConstants.h>
#include <libntc/shaders/BlockCompressConstants.h>
#include <libntc/shaders/ImageDifferenceConstants.h>

namespace ntc
{

Context::Context(ContextParameters const& params)
    : m_allocator(params.pAllocator)
    , m_cudaDevice(params.cudaDevice)
{
    if (params.graphicsApi != GraphicsAPI::None)
    {
        m_graphicsResources = new (m_allocator->Allocate(sizeof(GraphicsResources))) GraphicsResources(params);
    }
}

Context::~Context()
{
    if (m_graphicsResources)
    {
        m_graphicsResources->~GraphicsResources();
        m_allocator->Deallocate(m_graphicsResources, sizeof(GraphicsResources));
        m_graphicsResources = nullptr;
    }
}

Status Context::OpenFile(const char* fileName, bool write, IStream** pOutStream) const
{
    if (!fileName)
    {
        SetErrorMessage("fileName is NULL.");
        return Status::InvalidArgument;
    }

    if (!pOutStream)
    {
        SetErrorMessage("pOutStream is NULL.");
        return Status::InvalidArgument;
    }
    
    FILE* file = fopen(fileName, write ? "wb" : "rb");

    if (!file)
    {
        SetErrorMessage("Cannot open file '%s': %s.", fileName, strerror(errno));
        return Status::FileUnavailable;
    }

    *pOutStream = new(m_allocator->Allocate(sizeof(FileStream)))FileStream(file);

    return Status::Ok;
}

void Context::CloseFile(IStream* stream) const
{
    if (!stream)
        return;

    stream->~IStream();
    m_allocator->Deallocate(stream, sizeof(FileStream));
}

Status Context::OpenMemory(void* pData, size_t size, IStream** pOutStream) const
{
    if (!pData)
    {
        SetErrorMessage("pData is NULL.");
        return Status::InvalidArgument;
    }

    if (size == 0)
    {
        SetErrorMessage("size is 0.");
        return Status::InvalidArgument;
    }

    if (!pOutStream)
    {
        SetErrorMessage("pOutStream is NULL.");
        return Status::InvalidArgument;
    }

    *pOutStream = new(m_allocator->Allocate(sizeof(MemoryStream)))MemoryStream(
        static_cast<uint8_t*>(pData), size, false);
    
    return Status::Ok;
}

Status Context::OpenReadOnlyMemory(void const* pData, size_t size, IStream** pOutStream) const
{
    if (!pData)
    {
        SetErrorMessage("pData is NULL.");
        return Status::InvalidArgument;
    }

    if (size == 0)
    {
        SetErrorMessage("size is 0.");
        return Status::InvalidArgument;
    }

    if (!pOutStream)
    {
        SetErrorMessage("pOutStream is NULL.");
        return Status::InvalidArgument;
    }

    *pOutStream = new(m_allocator->Allocate(sizeof(MemoryStream)))MemoryStream(
        const_cast<uint8_t*>(static_cast<uint8_t const*>(pData)), size, true);
    
    return Status::Ok;
}

void Context::CloseMemory(IStream* stream) const
{
    if (!stream)
        return;

    stream->~IStream();
    m_allocator->Deallocate(stream, sizeof(MemoryStream));
}

Status Context::CreateTextureSet(const TextureSetDesc& desc,
                                 const TextureSetFeatures& features, ITextureSet** pOutTextureSet) const
{
#if NTC_WITH_CUDA
    if (!pOutTextureSet)
    {
        SetErrorMessage("pOutTextureSet is NULL.");
        return Status::InvalidArgument;
    }

    if (!IsCudaAvailable())
    {
        SetErrorMessage("Cannot create a TextureSet object when no suitable CUDA device is available.");
        return Status::CudaUnavailable;
    }

    CudaDeviceGuard cudaGuard(this);
    if (!cudaGuard.Success())
        return Status::CudaError;

    Status status = TextureSetMetadata::ValidateTextureSetDesc(desc);
    if (status != Status::Ok)
        return status;

    TextureSet* textureSet = new(m_allocator->Allocate(sizeof(TextureSet)))
        TextureSet(m_allocator, this, desc);
    
    status = textureSet->Initialize(features);
    if (status != Status::Ok)
    {
        DestroyTextureSet(textureSet);
        return status;
    }
    
    *pOutTextureSet = textureSet;
    ClearErrorMessage();
    return Status::Ok;
#else
    SetErrorMessage("Cannot create TextureSet objects when LibNTC was compiled without CUDA support.");
    return Status::CudaUnavailable;
#endif
}

void Context::DestroyTextureSet(ITextureSet* textureSet) const
{
#if NTC_WITH_CUDA
    if (!textureSet)
        return;

    CudaDeviceGuard cudaGuard(this);
    if (!cudaGuard.Success())
        return;

    TextureSet* implementation = dynamic_cast<TextureSet*>(textureSet);
    textureSet->~ITextureSet();
    m_allocator->Deallocate(implementation, sizeof(TextureSet));
#endif
}

Status Context::CreateTextureSetMetadataFromStream(IStream* inputStream, ITextureSetMetadata** pOutMetadata) const
{
    if (!pOutMetadata)
    {
        SetErrorMessage("pOutMetadata is NULL.");
        return Status::InvalidArgument;
    }

    if (!inputStream)
    {
        SetErrorMessage("inputStream is NULL.");
        return Status::InvalidArgument;
    }

    json::Document document(m_allocator);
    uint64_t binaryChunkOffset, binaryChunkSize;
    Status status = TextureSetMetadata::LoadFileHeadersFromStream(m_allocator, inputStream, document,
        binaryChunkOffset, binaryChunkSize);
    if (status != Status::Ok)
        return status;

    TextureSetDesc desc;
    LatentShape latentShape;
    status = TextureSetMetadata::DeserializeTextureSetDesc(document, desc, latentShape);
    if (status != Status::Ok)
        return status;

    TextureSetMetadata* textureSetMetadata = new(m_allocator->Allocate(sizeof(TextureSetMetadata)))
        TextureSetMetadata(m_allocator, desc, latentShape);

    status = textureSetMetadata->LoadMetadataFromStream(document, binaryChunkOffset, binaryChunkSize,
        latentShape, inputStream);

    if (status == Status::Ok)
    {
        status = textureSetMetadata->LoadWeightsFromStream(document, inputStream, m_graphicsResources);
    }

    if (status != Status::Ok)
    {
        DestroyTextureSetMetadata(textureSetMetadata);
        return status;
    }

    ClearErrorMessage();
    *pOutMetadata = textureSetMetadata;
    return Status::Ok;
}

void Context::DestroyTextureSetMetadata(ITextureSetMetadata* textureSetMetadata) const
{
    if (!textureSetMetadata)
        return;

    TextureSetMetadata* implementation = dynamic_cast<TextureSetMetadata*>(textureSetMetadata);
    textureSetMetadata->~ITextureSetMetadata();
    m_allocator->Deallocate(implementation, sizeof(TextureSetMetadata));
}

Status Context::CreateCompressedTextureSetFromStream(IStream* inputStream,
    const TextureSetFeatures& features, ITextureSet** pOutTextureSet) const
{
#if NTC_WITH_CUDA
    if (!pOutTextureSet)
    {
        SetErrorMessage("pOutTextureSet is NULL.");
        return Status::InvalidArgument;
    }

    if (!inputStream)
    {
        SetErrorMessage("inputStream is NULL.");
        return Status::InvalidArgument;
    }

    if (!IsCudaAvailable())
    {
        SetErrorMessage("Cannot create a TextureSet object when no suitable CUDA device is available.");
        return Status::CudaUnavailable;
    }

    CudaDeviceGuard cudaGuard(this);
    if (!cudaGuard.Success())
        return Status::CudaError;
        
    json::Document document(m_allocator);
    uint64_t binaryChunkOffset, binaryChunkSize;
    Status status = TextureSetMetadata::LoadFileHeadersFromStream(m_allocator, inputStream, document,
        binaryChunkOffset, binaryChunkSize);
    if (status != Status::Ok)
        return status;

    TextureSetDesc desc{};
    LatentShape latentShape;
    status = TextureSetMetadata::DeserializeTextureSetDesc(document, desc, latentShape);
    if (status != Status::Ok)
        return status;

    TextureSet* textureSet = nullptr;
    status = CreateTextureSet(desc, features, (ITextureSet**)&textureSet);
    if (status != Status::Ok)
        return status;

    status = textureSet->LoadFromStreamPostHeader(document, binaryChunkOffset, binaryChunkSize, inputStream, latentShape);
    if (status != Status::Ok)
    {
        DestroyTextureSet(textureSet);
        return status;
    }

    *pOutTextureSet = textureSet;
    
    ClearErrorMessage();
    return Status::Ok;
#else
    SetErrorMessage("Cannot create TextureSet objects when LibNTC was compiled without CUDA support.");
    return Status::CudaUnavailable;
#endif
}

Status Context::CreateCompressedTextureSetFromMemory(void const* pData, size_t size,
    const TextureSetFeatures& features, ITextureSet** pOutTextureSet) const
{
    MemoryStreamWrapper stream(this);

    Status status = OpenReadOnlyMemory(pData, size, stream.ptr());
    if (status != Status::Ok)
        return status;
        
    return CreateCompressedTextureSetFromStream(stream, features, pOutTextureSet);
}

Status Context::CreateCompressedTextureSetFromFile(char const* fileName,
    const TextureSetFeatures& features, ITextureSet** pOutTextureSet) const
{
    FileStreamWrapper stream(this);

    Status status = OpenFile(fileName, false, stream.ptr());
    if (status != Status::Ok)
        return status;
        
    return CreateCompressedTextureSetFromStream(stream, features, pOutTextureSet);
}

Status Context::RegisterSharedTexture(const SharedTextureDesc& desc, ISharedTexture** pOutTexture) const
{
#if NTC_WITH_CUDA
    if (!pOutTexture)
    {
        SetErrorMessage("pOutTexture is NULL.");
        return Status::InvalidArgument;
    }

    if (!IsCudaAvailable())
    {
        SetErrorMessage("Cannot create a SharedTexture object when no suitable CUDA device is available.");
        return Status::CudaUnavailable;
    }
    
    CudaDeviceGuard cudaGuard(this);
    if (!cudaGuard.Success())
        return Status::CudaError;

    SharedTexture* sharedTexture = new(m_allocator->Allocate(sizeof(SharedTexture))) SharedTexture(desc);
    
    Status status = sharedTexture->Initialize();
    if (status != Status::Ok)
    {
        ReleaseSharedTexture(sharedTexture);
        return status;
    }

    *pOutTexture = sharedTexture;
    ClearErrorMessage();
    return Status::Ok;
#else
    SetErrorMessage("Cannot create SharedTexture objects when LibNTC was compiled without CUDA support.");
    return Status::CudaUnavailable;
#endif
}

void Context::ReleaseSharedTexture(ISharedTexture* texture) const
{
#if NTC_WITH_CUDA
    if (!texture)
        return;

    CudaDeviceGuard cudaGuard(this);
    if (!cudaGuard.Success())
        return;

    texture->~ISharedTexture();
    m_allocator->Deallocate(texture, sizeof(SharedTexture));
#endif
}

Status Context::CreateAdaptiveCompressionSession(IAdaptiveCompressionSession** pOutSession) const
{
    if (!pOutSession)
    {
        SetErrorMessage("pOutSession is NULL");
        return Status::InvalidArgument;
    }

    *pOutSession = new (m_allocator->Allocate(sizeof(AdaptiveCompressionSession))) AdaptiveCompressionSession();
    ClearErrorMessage();
    return Status::Ok;
}

void Context::DestroyAdaptiveCompressionSession(IAdaptiveCompressionSession *session) const
{
    if (!session)
        return;

    m_allocator->Deallocate(session, sizeof(AdaptiveCompressionSession));
}

Status Context::MakeDecompressionComputePass(MakeDecompressionComputePassParameters const& params, ComputePassDesc* pOutComputePass) const
{
    if (!pOutComputePass)
    {
        SetErrorMessage("pOutComputePass is NULL");
        return Status::InvalidArgument;
    }

    if (!m_graphicsResources)
    {
        SetErrorMessage("The context was initialized with graphicsApi == None");
        return Status::InvalidArgument;
    }

    if (!params.pOutputTextures && params.numOutputTextures != 0)
    {
        SetErrorMessage("pOutputTextures is NULL while numOutputTextures (%d) is nonzero", params.numOutputTextures);
        return Status::InvalidArgument;
    }

    if (params.numOutputTextures > DECOMPRESS_CS_MAX_OUTPUTS)
    {
        SetErrorMessage("numOutputTextures (%d) is too large, must be %d or less", params.numOutputTextures, DECOMPRESS_CS_MAX_OUTPUTS);
        return Status::InvalidArgument;
    }

    // Validate the output textures
    for (int textureIndex = 0; textureIndex < params.numOutputTextures; ++textureIndex)
    {
        OutputTextureDesc const& desc = params.pOutputTextures[textureIndex];

        if (desc.firstChannel < 0 || desc.numChannels <= 0 || desc.numChannels > 4 ||
            desc.firstChannel + desc.numChannels >= NTC_MLP_OUTPUT_CHANNELS)
        {
            SetErrorMessage("pOutputTextures[%d] has invalid channel configuration: firstChannel = %d, numChannels = %d",
                textureIndex, desc.firstChannel, desc.numChannels);
            return Status::InvalidArgument;
        }

        if (desc.descriptorIndex < 0)
        {
            SetErrorMessage("pOutputTextures[%d] has invalid descriptorOffset (%d)",
                textureIndex, desc.descriptorIndex);
            return Status::InvalidArgument;
        }
    }

    TextureSetMetadata* textureSetMetadata = dynamic_cast<TextureSetMetadata*>(params.textureSetMetadata);
    if (!textureSetMetadata)
    {
        SetErrorMessage("textureSetMetadata is NULL or points at a wrong object type");
        return Status::InvalidArgument;
    }

    TextureSetDesc const& textureSetDesc = textureSetMetadata->GetDesc();
    LatentShape const& latentShape = textureSetMetadata->GetLatentShape();
    MlpDesc const* mlpDesc = textureSetMetadata->GetMlpDesc();

    if (!mlpDesc)
    {
        SetErrorMessage("The texture set metadata doesn't have MLP information.");
        return Status::InvalidArgument;
    }
    
    if (params.mipLevel < 0 || params.mipLevel >= textureSetDesc.mips)
    {
        SetErrorMessage("mipLevel (%d) must be between 0 and %d.", params.mipLevel, textureSetDesc.mips - 1);
        return Status::OutOfRange;
    }

    // Pre-compute some parameters needed to pick the right shader
    int const mipWidth = std::max(textureSetDesc.width >> params.mipLevel, 1);
    int const mipHeight = std::max(textureSetDesc.height >> params.mipLevel, 1);
    int const neuralLod = textureSetMetadata->ColorMipToNeuralLod(params.mipLevel);
    LatentImageDesc const* latentImage = textureSetMetadata->GetLatentImageDesc(neuralLod);
    if (!latentImage)
    {
        // This shouldn't happen with all the validation above, but let's be sure
        SetErrorMessage("latentImage is NULL");
        return Status::InternalError;
    }

    // The preload code only works when the resolution of high-res latents is <= the resolution of color pixels,
    // otherwise there's not enough shared memory allocated to store all the latents.
    bool const preloadLatents = latentImage->highResWidth <= mipWidth && latentImage->highResHeight <= mipHeight;

    // Use the weight vectors from the texture set as support flags - if coopvec int8 or fp8 is not supported
    // or disabled, the vector is empty. If there are no fp8 weights provided for example, the vector is also empty.
    bool const legacyInt8 = !textureSetMetadata->GetInferenceWeightVector(InferenceWeightType::GenericInt8)->empty();
    bool const coopVecInt8 = !textureSetMetadata->GetInferenceWeightVector(InferenceWeightType::CoopVecInt8)->empty();
    bool const coopVecFP8 = params.enableFP8 && !textureSetMetadata->GetInferenceWeightVector(InferenceWeightType::CoopVecFP8)->empty();
    bool const useCoopVec = coopVecInt8 || coopVecFP8;

    // Select the shader version based on which math modes are supported and enabled.
    InferenceMath mathVersion;
    InferenceWeightType weightType;
    if (coopVecFP8)
    {
        mathVersion = InferenceMath::CoopVecFP8;
        weightType = InferenceWeightType::CoopVecFP8;
    }
    else if (coopVecInt8)
    {
        mathVersion = InferenceMath::CoopVecInt8;
        weightType = InferenceWeightType::CoopVecInt8;
    }
    else if (legacyInt8)
    {
        if (m_graphicsResources->IsDP4aSupported() && m_graphicsResources->IsFloat16Supported())
            mathVersion = InferenceMath::DP4aWithFloat16;
        else if (m_graphicsResources->IsDP4aSupported())
            mathVersion = InferenceMath::DP4aNoFloat16;
        else
            mathVersion = InferenceMath::Legacy;
        weightType = InferenceWeightType::GenericInt8;
    }
    else
    {
        SetErrorMessage("No weights for the supported inference modes found in the texture set");
        return Status::Unsupported;
    }

    pOutComputePass->computeShader = nullptr;
    pOutComputePass->computeShaderSize = 0;
    
#if NTC_WITH_PREBUILT_SHADERS
    switch (m_graphicsResources->GetGraphicsApi())
    {
        case GraphicsAPI::D3D12:
#if NTC_WITH_DX12
            GetDecompressDxilShaderBytecode(mlpDesc, mathVersion, preloadLatents,
                &pOutComputePass->computeShader, &pOutComputePass->computeShaderSize);
#endif
            break;
        case GraphicsAPI::Vulkan:
#if NTC_WITH_VULKAN
            GetDecompressSpirvShaderBytecode(mlpDesc, mathVersion, preloadLatents,
                &pOutComputePass->computeShader, &pOutComputePass->computeShaderSize);
#endif
            break;
        default:
            // Shouldn't happen - None is handled above and invalid values are handled at context initialization
            return Status::InvalidArgument;
    }
#endif

    Rect srcRect { 0, 0, mipWidth, mipHeight };
    if (params.pSrcRect)
    {
        int const left = params.pSrcRect->left;
        int const top = params.pSrcRect->top;
        int const width = params.pSrcRect->width;
        int const height = params.pSrcRect->height;
        int const right = left + width;
        int const bottom = top + height;
        if (left < 0 || top < 0 || width <= 0 || height <= 0 || right > mipWidth || bottom > mipHeight)
        {
            SetErrorMessage("Invalid rectangle specified. For mip %d, left (%d) and top (%d) must be >= 0; "
                "width (%d) and height (%d) must be > 0; (left + width) (%d) must be <= %d; "
                "(top + height) must be <= %d.", 
                params.mipLevel, left, top, width, height, right, bottom, mipHeight);
            return Status::OutOfRange;
        }
        srcRect = *params.pSrcRect;
    }

    Point dstOffset { srcRect.left, srcRect.top };
    if (params.pDstOffset)
    {
        dstOffset = *params.pDstOffset;
    }

    // Get the stream range for this mip level
    StreamRange requiredRange;
    Status status = textureSetMetadata->GetStreamRangeForLatents(params.mipLevel, 1, requiredRange);
    if (status != Status::Ok)
        return status;
    
    StreamRange latentStreamRange = params.latentStreamRange;

    // Convert the "entire stream" range into the actual stream size
    if (latentStreamRange.size + 1 == 0)
    {
        latentStreamRange.size = std::max(textureSetMetadata->GetSourceStreamSize(), latentStreamRange.offset)
            - latentStreamRange.offset;
    }
        
    // Validate that the provided stream range contains the required range
    if (requiredRange.offset < latentStreamRange.offset ||
        requiredRange.offset + requiredRange.size > latentStreamRange.offset + latentStreamRange.size)
    {
        SetErrorMessage("Decompression of mip level %d requires input stream range %" PRId64 "-%" PRId64", "
            "which is not contained in the provided range %" PRId64 "-%" PRId64 ".",
            params.mipLevel,
            requiredRange.offset, requiredRange.offset + requiredRange.size,
            latentStreamRange.offset, latentStreamRange.offset + latentStreamRange.size);
        return Status::OutOfRange;
    }

    NtcDecompressConstants& constants = reinterpret_cast<NtcDecompressConstants&>(pOutComputePass->constantBufferData);
    static_assert(sizeof(NtcDecompressConstants) <= sizeof(ComputePassDesc::constantBufferData),
        "NtcDecompressConstants don't fit into constantBufferData. Increase MaxComputePassConstantSize.");
    pOutComputePass->constantBufferSize = sizeof(NtcDecompressConstants);

    // Fill the constant buffer using that latent buffer offset
    textureSetMetadata->FillDecompressionConstants(constants, weightType, params.mipLevel, srcRect, dstOffset,
        params.pOutputTextures, params.numOutputTextures, params.firstOutputDescriptorIndex, latentStreamRange.offset);

    int const gridWidth = constants.srcRight - constants.gridLeft;
    int const gridHeight = constants.srcBottom - constants.gridTop;

    pOutComputePass->weightBufferData = textureSetMetadata->GetInferenceWeightVector(weightType)->data();
    pOutComputePass->weightBufferSize = textureSetMetadata->GetInferenceWeightVector(weightType)->size();
    pOutComputePass->dispatchWidth = (gridWidth + DECOMPRESS_CS_BLOCK_WIDTH - 1) / DECOMPRESS_CS_BLOCK_WIDTH;
    pOutComputePass->dispatchHeight = (gridHeight + DECOMPRESS_CS_BLOCK_HEIGHT - 1) / DECOMPRESS_CS_BLOCK_HEIGHT;

    if (!pOutComputePass->computeShader)
    {
        SetErrorMessage("The requested shader binary is unavailable.");
        return Status::ShaderUnavailable;
    }

    return Status::Ok;
}

Status Context::MakeBlockCompressionComputePass(MakeBlockCompressionComputePassParameters const& params,
    ComputePassDesc* pOutComputePass) const
{
    if (!pOutComputePass)
    {
        SetErrorMessage("pOutComputePass is NULL");
        return Status::InvalidArgument;
    }

    if (!m_graphicsResources)
    {
        SetErrorMessage("The context was initialized with graphicsApi == None");
        return Status::InvalidArgument;
    }

    if (params.srcRect.left < 0 || params.srcRect.top < 0 || params.srcRect.width <= 0 || params.srcRect.height <= 0)
    {
        SetErrorMessage("srcRect.left (%d) and srcRect.top (%d) must be >= 0; srcRect.width (%d) and "
            "srcRect.height (%d) must be > 0",
            params.srcRect.left, params.srcRect.top, params.srcRect.width, params.srcRect.height);
        return Status::OutOfRange;
    }

    if (params.dstOffsetInBlocks.x < 0 || params.dstOffsetInBlocks.y < 0)
    {
        SetErrorMessage("dstOffsetInBlocks.x (%d) and dstOffsetInBlocks.y (%d) must be >= 0",
            params.dstOffsetInBlocks.x, params.dstOffsetInBlocks.y);
        return Status::OutOfRange;
    }

    if (params.dstFormat < BlockCompressedFormat::BC1 || params.dstFormat > BlockCompressedFormat::BC7)
    {
        SetErrorMessage("dstFormat (%s) has invalid value", BlockCompressedFormatToString(params.dstFormat));
        return Status::OutOfRange;
    }

    pOutComputePass->computeShader = nullptr;
    pOutComputePass->computeShaderSize = 0;
    
#if NTC_WITH_PREBUILT_SHADERS
    switch (m_graphicsResources->GetGraphicsApi())
    {
        case GraphicsAPI::D3D12:
#if NTC_WITH_DX12
            GetBlockCompressDxilShaderBytecode(params.dstFormat, params.writeAccelerationData,
                &pOutComputePass->computeShader, &pOutComputePass->computeShaderSize);
#endif
            break;
        case GraphicsAPI::Vulkan:
#if NTC_WITH_VULKAN
            GetBlockCompressSpirvShaderBytecode(params.dstFormat, params.writeAccelerationData,
                &pOutComputePass->computeShader, &pOutComputePass->computeShaderSize);
#endif
            break;
        default:
            // Shouldn't happen - None is handled above and invalid values are handled at context initialization
            return Status::InvalidArgument;
    }
#endif

    NtcBlockCompressConstants& constants = reinterpret_cast<NtcBlockCompressConstants&>(pOutComputePass->constantBufferData);
    static_assert(sizeof(NtcBlockCompressConstants) <= sizeof(ComputePassDesc::constantBufferData),
        "NtcBlockCompressConstants don't fit into constantBufferData. Increase MaxComputePassConstantSize.");
    pOutComputePass->constantBufferSize = sizeof(NtcBlockCompressConstants);

    constants.srcLeft = params.srcRect.left;
    constants.srcTop = params.srcRect.top;
    constants.dstOffsetX = params.dstOffsetInBlocks.x;
    constants.dstOffsetY = params.dstOffsetInBlocks.y;
    constants.widthInBlocks = (params.srcRect.width + 3) / 4;
    constants.heightInBlocks = (params.srcRect.height + 3) / 4;
    constants.alphaThreshold = params.alphaThreshold;

    memset(constants.allowedModes, 0xff, sizeof(constants.allowedModes));
    if (params.texture && params.dstFormat == BlockCompressedFormat::BC7 && !params.writeAccelerationData)
    {
        static_cast<TextureMetadata const*>(params.texture)->GetAllowedBCModes(constants.allowedModes,
            sizeof(constants.allowedModes), params.quality);
    }
    
    pOutComputePass->dispatchWidth = (constants.widthInBlocks + BLOCK_COMPRESS_CS_ST_GROUP_WIDTH - 1)
        / BLOCK_COMPRESS_CS_ST_GROUP_WIDTH;
    pOutComputePass->dispatchHeight = (constants.heightInBlocks + BLOCK_COMPRESS_CS_ST_GROUP_HEIGHT - 1)
        / BLOCK_COMPRESS_CS_ST_GROUP_HEIGHT;

    if (!pOutComputePass->computeShader)
    {
        SetErrorMessage("The requested shader binary is unavailable.");
        return Status::ShaderUnavailable;
    }

    return Status::Ok;
}

Status Context::MakeImageDifferenceComputePass(MakeImageDifferenceComputePassParameters const& params,
    ComputePassDesc* pOutComputePass) const
{
    if (!pOutComputePass)
    {
        SetErrorMessage("pOutComputePass is NULL");
        return Status::InvalidArgument;
    }

    if (!m_graphicsResources)
    {
        SetErrorMessage("The context was initialized with graphicsApi == None");
        return Status::InvalidArgument;
    }

    if (params.extent.left < 0 || params.extent.top < 0)
    {
        SetErrorMessage("Left (%d) and top (%d) must be non-negative", params.extent.left, params.extent.top);
        return Status::OutOfRange;
    }
    
    if (params.extent.width <= 0 || params.extent.height <= 0)
    {
        SetErrorMessage("Width (%d) and height (%d) must be positive", params.extent.width, params.extent.height);
        return Status::OutOfRange;
    }

    if ((params.outputOffset & 3) != 0)
    {
        SetErrorMessage("outputOffset (%d) must be aligned to 4 bytes", params.outputOffset);
        return Status::OutOfRange;
    }

    pOutComputePass->computeShader = nullptr;
    pOutComputePass->computeShaderSize = 0;

#if NTC_WITH_PREBUILT_SHADERS
    switch (m_graphicsResources->GetGraphicsApi())
    {
        case GraphicsAPI::D3D12:
#if NTC_WITH_DX12
            GetImageDifferenceDxilShaderBytecode(&pOutComputePass->computeShader, &pOutComputePass->computeShaderSize);
#endif
            break;
        case GraphicsAPI::Vulkan:
#if NTC_WITH_VULKAN
            GetImageDifferenceSpirvShaderBytecode(&pOutComputePass->computeShader, &pOutComputePass->computeShaderSize);
#endif
            break;
        default:
            // Shouldn't happen - None is handled above and invalid values are handled at context initialization
            return Status::InvalidArgument;
    }
#endif

    NtcImageDifferenceConstants& constants = reinterpret_cast<NtcImageDifferenceConstants&>(pOutComputePass->constantBufferData);
    static_assert(sizeof(NtcImageDifferenceConstants) <= sizeof(ComputePassDesc::constantBufferData),
        "NtcImageDifferenceConstants don't fit into constantBufferData. Increase MaxComputePassConstantSize.");
    pOutComputePass->constantBufferSize = sizeof(NtcImageDifferenceConstants);

    constants.left = params.extent.left;
    constants.top = params.extent.top;
    constants.width = params.extent.width;
    constants.height = params.extent.height;
    constants.alphaThreshold = params.alphaThreshold;
    constants.useAlphaThreshold = params.useAlphaThreshold ? 1 : 0;
    constants.useMSLE = params.useMSLE ? 1 : 0;
    constants.outputOffset = params.outputOffset;
    
    pOutComputePass->dispatchWidth = (params.extent.width + IMAGE_DIFFERENCE_CS_PIXELS_PER_BLOCK_X - 1)
        / IMAGE_DIFFERENCE_CS_PIXELS_PER_BLOCK_X;
    pOutComputePass->dispatchHeight = (params.extent.height + IMAGE_DIFFERENCE_CS_PIXELS_PER_BLOCK_Y - 1)
        / IMAGE_DIFFERENCE_CS_PIXELS_PER_BLOCK_Y;

    if (!pOutComputePass->computeShader)
    {
        SetErrorMessage("The requested shader binary is unavailable.");
        return Status::ShaderUnavailable;
    }

    return Status::Ok;
}

Status Context::MakeInferenceData(ITextureSetMetadata* _textureSetMetadata, StreamRange latentStreamRange,
    InferenceWeightType weightType, InferenceData* pOutInferenceData) const
{
    if (!pOutInferenceData)
    {
        SetErrorMessage("pOutInferenceData is NULL");
        return Status::InvalidArgument;
    }

    TextureSetMetadata* textureSetMetadata = dynamic_cast<TextureSetMetadata*>(_textureSetMetadata);
    if (!textureSetMetadata)
    {
        SetErrorMessage("textureSetMetadata is NULL or points at a wrong object type");
        return Status::InvalidArgument;
    }

    switch(weightType)
    {
        case InferenceWeightType::GenericInt8:
        case InferenceWeightType::CoopVecInt8:
        case InferenceWeightType::GenericFP8:
        case InferenceWeightType::CoopVecFP8:
            break;
        default:
            SetErrorMessage("Unsupported weightType (%s)", InferenceWeightTypeToString(weightType));
    }

    if (!textureSetMetadata->IsInferenceWeightTypeSupported(weightType))
    {
        SetErrorMessage("The texture set does not provide %s weights", InferenceWeightTypeToString(weightType));
        return Status::Unsupported;
    }
    
    memset(pOutInferenceData, 0, sizeof(InferenceData));

    TextureSetDesc const& textureSetDesc = textureSetMetadata->GetDesc();
    LatentShape const& latentShape = textureSetMetadata->GetLatentShape();

    TextureSetMetadata::FillLatentEncodingConstants(pOutInferenceData->constants.highResEncoding,
        latentShape.highResFeatures, latentShape.highResQuantBits, weightType);
    TextureSetMetadata::FillLatentEncodingConstants(pOutInferenceData->constants.lowResEncoding,
        latentShape.lowResFeatures, latentShape.lowResQuantBits, weightType);

    for (int neuralLod = 0; neuralLod < textureSetMetadata->GetNumLatentImages(); ++neuralLod)
    {
        assert(neuralLod < NTC_MAX_NEURAL_MIPS);

        textureSetMetadata->FillNeuralMipConstants(
            pOutInferenceData->constants.highResNeuralMips[neuralLod],
            pOutInferenceData->constants.lowResNeuralMips[neuralLod],
            neuralLod, latentStreamRange.offset);
    }

    for (int mipLevel = 0; mipLevel < textureSetDesc.mips; ++mipLevel)
    {
        assert(mipLevel < NTC_MAX_MIPS);
        
        textureSetMetadata->FillColorMipConstants(pOutInferenceData->constants.colorMips[mipLevel], mipLevel);
    }

    pOutInferenceData->constants.imageWidth = textureSetDesc.width;
    pOutInferenceData->constants.imageHeight = textureSetDesc.height;
    pOutInferenceData->constants.imageMips = textureSetDesc.mips;
    textureSetMetadata->GetWeightOffsets(weightType, pOutInferenceData->constants.networkWeightOffsets,
        pOutInferenceData->constants.networkScaleBiasOffset);
    pOutInferenceData->constants.validChannelMask = textureSetMetadata->GetValidChannelMask();
    pOutInferenceData->constants.channelColorSpaces = textureSetMetadata->GetPackedColorSpaces();
 
    return Status::Ok;
}

Status Context::MakePartialInferenceData(ITextureSetMetadata* _textureSetMetadata, IStream* inputStream,
    int firstMipLevel, int numMipLevels, Rect firstMipSlice, InferenceWeightType weightType,
    InferenceData* pOutInferenceData, void* pOutLatentData, size_t* pInOutLatentSize) const
{
    if (!pInOutLatentSize)
    {
        SetErrorMessage("pInOutLatentSize is NULL");
        return Status::InvalidArgument;
    }

    if (!pOutInferenceData && pOutLatentData)
    {
        SetErrorMessage("pOutInferenceData is NULL");
        return Status::InvalidArgument;
    }

    TextureSetMetadata* textureSetMetadata = dynamic_cast<TextureSetMetadata*>(_textureSetMetadata);
    if (!textureSetMetadata)
    {
        SetErrorMessage("textureSetMetadata is NULL or points at a wrong object type");
        return Status::InvalidArgument;
    }

    switch(weightType)
    {
        case InferenceWeightType::GenericInt8:
        case InferenceWeightType::CoopVecInt8:
        case InferenceWeightType::GenericFP8:
        case InferenceWeightType::CoopVecFP8:
            break;
        default:
            SetErrorMessage("Unsupported weightType (%s)", InferenceWeightTypeToString(weightType));
            return Status::InvalidArgument;
    }

    if (!textureSetMetadata->IsInferenceWeightTypeSupported(weightType))
    {
        SetErrorMessage("The texture set does not provide %s weights", InferenceWeightTypeToString(weightType));
        return Status::Unsupported;
    }
    
    TextureSetDesc const& textureSetDesc = textureSetMetadata->GetDesc();
    LatentShape const& latentShape = textureSetMetadata->GetLatentShape();

    if (firstMipLevel < 0 || numMipLevels <= 0 || firstMipLevel + numMipLevels > textureSetDesc.mips)
    {
        SetErrorMessage("firstMipLevel (%d) and numMipLevels (%d) must describe a valid range within 0-%d",
            firstMipLevel, numMipLevels, textureSetDesc.mips - 1);
        return Status::OutOfRange;
    }
    
    int colorMipWidth = std::max(textureSetDesc.width >> firstMipLevel, 1);
    int colorMipHeight = std::max(textureSetDesc.height >> firstMipLevel, 1);
    
    if (firstMipSlice.left < 0 || firstMipSlice.top < 0 || firstMipSlice.width <= 0 || firstMipSlice.height <= 0 ||
        firstMipSlice.left + firstMipSlice.width > colorMipWidth ||
        firstMipSlice.top + firstMipSlice.height > colorMipWidth)
    {
        SetErrorMessage("firstMipSlice (%dx%d starting at %d, %d) must be within the bounds of MIP %d (%dx%d)",
            firstMipSlice.left + firstMipSlice.width, firstMipSlice.top + firstMipSlice.height,
            firstMipSlice.left, firstMipSlice.top,
            firstMipLevel, colorMipWidth, colorMipHeight);
        return Status::OutOfRange;
    }

    int const highResBitsPerLatentPixel = latentShape.highResFeatures * latentShape.highResQuantBits;
    int const lowResBitsPerLatentPixel = latentShape.lowResFeatures * latentShape.lowResQuantBits;

    // Validate that the latent pixels occupy a multiple of 4 bits. That should always be true because we enforce
    // that the number of latents is a multiple of 4 (see TextureSetMetadata::ValidateLatentShape)
    assert((highResBitsPerLatentPixel & 3) == 0);
    assert((lowResBitsPerLatentPixel & 3) == 0);

    size_t const latentBufferSize = pOutLatentData ? *pInOutLatentSize : 0;
    size_t totalLatentSize = 0;

    NtcTextureSetConstants* constants = pOutInferenceData ? &pOutInferenceData->constants : nullptr;

    if (constants)
    {
        memset(constants, 0, sizeof(NtcTextureSetConstants));

        TextureSetMetadata::FillLatentEncodingConstants(constants->highResEncoding,
            latentShape.highResFeatures, latentShape.highResQuantBits, weightType);
        TextureSetMetadata::FillLatentEncodingConstants(constants->lowResEncoding,
            latentShape.lowResFeatures, latentShape.lowResQuantBits, weightType);
        textureSetMetadata->GetWeightOffsets(weightType, constants->networkWeightOffsets,
            constants->networkScaleBiasOffset);

        constants->imageWidth = textureSetDesc.width;
        constants->imageHeight = textureSetDesc.height;
        constants->imageMips = textureSetDesc.mips;
        constants->validChannelMask = textureSetMetadata->GetValidChannelMask();
        constants->channelColorSpaces = textureSetMetadata->GetPackedColorSpaces();
    }

    for (int colorMip = 0; colorMip < textureSetDesc.mips; ++colorMip)
    {
        bool const validColorMip = colorMip >= firstMipLevel && colorMip < firstMipLevel + numMipLevels;

        if (constants)
        {
            textureSetMetadata->FillColorMipConstants(constants->colorMips[colorMip], colorMip);
        }

        if (!validColorMip)
            continue;

        // Process neural LODs only for the last mip level in a fused mip range, to stay conservative.
        // The condition below detects either a change in neural LOD on the next mip or that it's the last mip.
        int const neuralLod = textureSetMetadata->ColorMipToNeuralLod(colorMip);
        if (colorMip < firstMipLevel + numMipLevels - 1 && 
            neuralLod == textureSetMetadata->ColorMipToNeuralLod(colorMip + 1))
            continue;

        colorMipWidth = std::max(textureSetDesc.width >> colorMip, 1);
        colorMipHeight = std::max(textureSetDesc.height >> colorMip, 1);

        LatentImageDesc const* latentImage = textureSetMetadata->GetLatentImageDesc(neuralLod);
        if (!latentImage)
        {
            // This shouldn't happen, but let's be sure
            SetErrorMessage("latentImage is NULL");
            return Status::InternalError;
        }

        int const deltaMip = colorMip - firstMipLevel;
        int const colorSliceLeft = firstMipSlice.left >> deltaMip;
        int const colorSliceTop = firstMipSlice.top >> deltaMip;
        
        // Calculate the right and bottom pixel positions for the mip slice, rounding up the division by power of 2.
        int const colorSliceRight = ShiftRightRoundUp(firstMipSlice.left + firstMipSlice.width, deltaMip) - 1;
        int const colorSliceBottom = ShiftRightRoundUp(firstMipSlice.top + firstMipSlice.height, deltaMip) - 1;

        // Calculates the range of latents (min, size) in one dimension for the given range of color pixels.
        auto calculateLatentRange = [](int colorMin, int colorMax, int colorSize, int latentSize, int bitsPerElement,
            int& outMin, int& outSize)
        {
            // Instead of adding 0.5 on both ends like the sampling math does, use the left and right extents.
            // This allows us to calculate a conservative extent that covers this mip level and more detailed ones too.
            // When downsampling extents, sometimes the center positions go up and sometimes they go down.
            float umin = float(colorMin) / float(colorSize);
            float umax = float(colorMax + 1) / float(colorSize);
            int lmin = std::max(int(umin * float(latentSize) - 0.5f), 0);
            int lmax = std::min(int(umax * float(latentSize) - 0.5f) + 1, latentSize - 1);
            outMin = lmin;
            outSize = lmax - lmin + 1;
            
            // If the elements (pixels or rows) are occupying an odd number of 4-bit nibbles, make sure that we cut out
            // an even number of those elements at an even offset, so that we're operating on whole bytes.
            if ((bitsPerElement & 4) != 0)
            {
                if ((outMin & 1) != 0)
                {
                    outMin -= 1;
                    outSize += 1;
                }
                outSize = std::min((outSize + 1) & ~1, latentSize);
            }
        };

        Rect highResLatentSlice;
        Rect lowResLatentSlice;
        calculateLatentRange(colorSliceLeft, colorSliceRight, colorMipWidth, latentImage->highResWidth,
            highResBitsPerLatentPixel, highResLatentSlice.left, highResLatentSlice.width);
        calculateLatentRange(colorSliceTop, colorSliceBottom, colorMipHeight, latentImage->highResHeight,
            highResBitsPerLatentPixel * latentImage->highResWidth, highResLatentSlice.top, highResLatentSlice.height);
        calculateLatentRange(colorSliceLeft, colorSliceRight, colorMipWidth, latentImage->lowResWidth,
            lowResBitsPerLatentPixel, lowResLatentSlice.left, lowResLatentSlice.width);
        calculateLatentRange(colorSliceTop, colorSliceBottom, colorMipHeight, latentImage->lowResHeight,
            lowResBitsPerLatentPixel * latentImage->lowResWidth, lowResLatentSlice.top, lowResLatentSlice.height);

        size_t const highResLatentPixels = size_t(highResLatentSlice.width) * size_t(highResLatentSlice.height);
        size_t const lowResLatentPixels = size_t(lowResLatentSlice.width) * size_t(lowResLatentSlice.height);
        size_t const highResBytes = RoundUp4((highResLatentPixels * highResBitsPerLatentPixel) >> 3);
        size_t const lowResBytes = RoundUp4((lowResLatentPixels * lowResBitsPerLatentPixel) >> 3);
        size_t const currentOffset = totalLatentSize;
        totalLatentSize += highResBytes + lowResBytes;

        if (totalLatentSize <= latentBufferSize)
        {
            auto readLatentSlice = [this, inputStream, pOutLatentData, textureSetMetadata, latentImage, latentBufferSize]
                (bool highRes, Rect slice, int bitsPerPixel, size_t destOffset)
            {
                StreamRange const location = highRes
                    ? latentImage->highResRange
                    : latentImage->lowResRange;
                int const sourceWidth = highRes
                    ? latentImage->highResWidth
                    : latentImage->lowResWidth;

                size_t const bytesPerDstRow = (size_t(slice.width) * size_t(bitsPerPixel)) >> 3;

                for (int row = slice.top; row < slice.top + slice.height; ++row)
                {
                    uint64_t const firstPixel = uint64_t(row) * uint64_t(sourceWidth) + uint64_t(slice.left);
                    uint64_t const sourceOffset = location.offset + ((firstPixel * uint64_t(bitsPerPixel)) >> 3);

                    if (!inputStream->Seek(sourceOffset))
                        return false;

                    size_t const destRowOffset = destOffset + size_t(row - slice.top) * bytesPerDstRow;
                    
                    void* pDest = reinterpret_cast<uint8_t*>(pOutLatentData) + destRowOffset;

                    assert(destRowOffset + bytesPerDstRow <= latentBufferSize);

                    if (!inputStream->Read(pDest, bytesPerDstRow))
                        return false;
                }

                return true;
            };

            // Read the high-res latent slice
            if (!readLatentSlice(true, highResLatentSlice, highResBitsPerLatentPixel, currentOffset))
                return Status::IOError;

            // Read the low-res latent slice
            if (!readLatentSlice(false, lowResLatentSlice, lowResBitsPerLatentPixel, currentOffset + highResBytes))
                return Status::IOError;
        }

        if (constants)
        {
            NtcNeuralMipConstants& highResLatents = constants->highResNeuralMips[neuralLod];
            highResLatents.dataOffset = uint32_t(currentOffset);
            highResLatents.imageWidth = uint16_t(latentImage->highResWidth);
            highResLatents.imageHeight = uint16_t(latentImage->highResHeight);
            highResLatents.sliceWidth = uint16_t(highResLatentSlice.width);
            highResLatents.sliceHeight = uint16_t(highResLatentSlice.height);
            highResLatents.sliceLeft = uint16_t(highResLatentSlice.left);
            highResLatents.sliceTop = uint16_t(highResLatentSlice.top);

            NtcNeuralMipConstants& lowResLatents = constants->lowResNeuralMips[neuralLod];
            lowResLatents.dataOffset = uint32_t(currentOffset + highResBytes);
            lowResLatents.imageWidth = uint16_t(latentImage->lowResWidth);
            lowResLatents.imageHeight = uint16_t(latentImage->lowResHeight);
            lowResLatents.sliceWidth = uint16_t(lowResLatentSlice.width);
            lowResLatents.sliceHeight = uint16_t(lowResLatentSlice.height);
            lowResLatents.sliceLeft = uint16_t(lowResLatentSlice.left);
            lowResLatents.sliceTop = uint16_t(lowResLatentSlice.top);
        }
    }

    if (pOutInferenceData && totalLatentSize > latentBufferSize)
    {
        SetErrorMessage("The provided buffer (%zu bytes) is too small for the requested range of latents. "
            "It need to be at least %zu bytes large.", latentBufferSize, totalLatentSize);
        return Status::OutOfRange;
    }

    // Return the buffer size required or used for this operation
    *pInOutLatentSize = totalLatentSize;
    if (!pOutInferenceData)
        return Status::Incomplete;


    return Status::Ok;
}

Status Context::GetConservativeLatentBufferSize(ITextureSetMetadata* _textureSetMetadata,
    int firstMipLevel, int numMipLevels, int firstMipSliceWidth, int firstMipSliceHeight, int sliceAlignment,
    size_t* pOutLatentSize) const
{
    if (!pOutLatentSize)
    {
        SetErrorMessage("pOutLatentSize is NULL");
        return Status::InvalidArgument;
    }

    TextureSetMetadata* textureSetMetadata = dynamic_cast<TextureSetMetadata*>(_textureSetMetadata);
    if (!textureSetMetadata)
    {
        SetErrorMessage("textureSetMetadata is NULL or points at a wrong object type");
        return Status::InvalidArgument;
    }

    TextureSetDesc const& textureSetDesc = textureSetMetadata->GetDesc();
    LatentShape const& latentShape = textureSetMetadata->GetLatentShape();

    if (firstMipLevel < 0 || numMipLevels <= 0 || firstMipLevel + numMipLevels > textureSetDesc.mips)
    {
        SetErrorMessage("firstMipLevel (%d) and numMipLevels (%d) must describe a valid range within 0-%d",
            firstMipLevel, numMipLevels, textureSetDesc.mips - 1);
        return Status::OutOfRange;
    }
    
    int colorMipWidth = std::max(textureSetDesc.width >> firstMipLevel, 1);
    int colorMipHeight = std::max(textureSetDesc.height >> firstMipLevel, 1);
    
    if (firstMipSliceWidth <= 0 || firstMipSliceWidth > colorMipWidth ||
        firstMipSliceHeight <= 0 || firstMipSliceHeight > colorMipHeight)
    {
        SetErrorMessage("firstMipSliceWidth (%d) and firstMipSliceHeight (%d) must be between 1 and (%d, %d)",
            firstMipSliceWidth, firstMipSliceHeight, colorMipWidth, colorMipHeight);
        return Status::OutOfRange;
    }

    if (sliceAlignment < 1)
    {
        SetErrorMessage("sliceAlignment (%d) must be 1 or more", sliceAlignment);
        return Status::OutOfRange;
    }

    // We can only use power-of-2 alignments productively.
    if (!IsPowerOf2(sliceAlignment))
    {
        sliceAlignment = 1;
    }

    int const highResBitsPerLatentPixel = latentShape.highResFeatures * latentShape.highResQuantBits;
    int const lowResBitsPerLatentPixel = latentShape.lowResFeatures * latentShape.lowResQuantBits;

    size_t totalLatentSize = 0;

    // Mirroring the logic in MakePartialInferenceData, we go over the color mips to find the last requested color mip
    // in every neural LOD.
    for (int colorMip = firstMipLevel; colorMip < firstMipLevel + numMipLevels; ++colorMip)
    {
        int const neuralLod = textureSetMetadata->ColorMipToNeuralLod(colorMip);
        if (colorMip < firstMipLevel + numMipLevels - 1 && 
            neuralLod == textureSetMetadata->ColorMipToNeuralLod(colorMip + 1))
            continue;

        colorMipWidth = std::max(textureSetDesc.width >> colorMip, 1);
        colorMipHeight = std::max(textureSetDesc.height >> colorMip, 1);

        LatentImageDesc const* latentImage = textureSetMetadata->GetLatentImageDesc(neuralLod);
        if (!latentImage)
        {
            // This shouldn't happen, but let's be sure
            SetErrorMessage("latentImage is NULL");
            return Status::InternalError;
        }

        // To calculate the size of the color slice, we assume the worst case positioning.
        // When generating 'deltaMip' mip levels, the top level pixel has size of (1 << deltaMip).
        // Worst case positioning is when the slice starts at (1 << deltaMip) - 1 because then it touches
        // the most pixels in coarser mips. Since the caller provides us with the alignment constraint,
        // we can use that to make the worst case better.
        int const deltaMip = colorMip - firstMipLevel;        
        int const mipSliceAlignment = ShiftRightRoundUp(sliceAlignment, deltaMip);
        int const worstCaseOffset = std::max((1 << deltaMip) - mipSliceAlignment, 0);
        int const colorSliceWidth = ShiftRightRoundUp(worstCaseOffset + firstMipSliceWidth, deltaMip);
        int const colorSliceHeight = ShiftRightRoundUp(worstCaseOffset + firstMipSliceHeight, deltaMip);

        // This function calculates the maximum size of latent slice in one dimension (width or height)
        // for a given dimension of the color slice.
        auto calculateLatentSlice = [mipSliceAlignment](int colorSlice, int colorSize,
            int latentSize, int bitsPerElement)
        {
            // Check if the slice will be aligned to integer latent pixels.
            // If not, an extra pixel will be added to 'colorSlice' below to account for different positions.
            float const latentAlignment = float(mipSliceAlignment) * float(latentSize) / float(colorSize);
            bool const aligned = floorf(latentAlignment) == latentAlignment;

            // Calculate the size of the latent image slice matching the given color slice.
            float uslice = (float(colorSlice + int(!aligned))) / float(colorSize);
            int lslice = int(ceilf(uslice * float(latentSize))) + 2; // +2 for the linear filtering margin
            
            // If the latent pixels occupy an odd number of 4-bit nibbles, add 1 latent pixel of padding
            // and then round up to even. The padding accounts for the offset down (outMin -= 1)
            // in MakePartialInferenceData.
            if ((bitsPerElement & 4) != 0)
            {
                lslice = (lslice + 2) & ~1;
            }

            // Clamp to the latent image dimensions.
            lslice = std::min(lslice, latentSize);

            return lslice;
        };

        int const highResLatentSliceWidth = calculateLatentSlice(colorSliceWidth, colorMipWidth,
            latentImage->highResWidth, highResBitsPerLatentPixel);
        int const highResLatentSliceHeight = calculateLatentSlice(colorSliceHeight, colorMipHeight,
            latentImage->highResHeight, 0);
        int const lowResLatentSliceWidth = calculateLatentSlice(colorSliceWidth, colorMipWidth,
            latentImage->lowResWidth, lowResBitsPerLatentPixel);
        int const lowResLatentSliceHeight = calculateLatentSlice(colorSliceHeight, colorMipHeight,
            latentImage->lowResHeight, 0);
        
        size_t const highResLatentPixels = size_t(highResLatentSliceWidth) * size_t(highResLatentSliceHeight);
        size_t const lowResLatentPixels = size_t(lowResLatentSliceWidth) * size_t(lowResLatentSliceHeight);
        size_t const highResBytes = RoundUp4((highResLatentPixels * highResBitsPerLatentPixel) >> 3);
        size_t const lowResBytes = RoundUp4((lowResLatentPixels * lowResBitsPerLatentPixel) >> 3);

        totalLatentSize += highResBytes + lowResBytes;
    }

    *pOutLatentSize = totalLatentSize;
    return Status::Ok;
}

bool Context::IsCooperativeVectorInt8Supported() const
{
    if (m_graphicsResources)
        return m_graphicsResources->IsCoopVecInt8Supported();

    return false;
}

bool Context::IsCooperativeVectorFP8Supported() const
{
    if (m_graphicsResources)
        return m_graphicsResources->IsCoopVecFP8Supported();

    return false;
}

}
