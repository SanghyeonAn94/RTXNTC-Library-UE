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

#include <cstdint>
#include <cstddef>

#ifdef NTC_BUILD_SHARED
    #ifdef _MSC_VER
        #define NTC_API __declspec(dllexport)
    #else
        #define NTC_API
    #endif
#elif NTC_STATIC
    #define NTC_API extern
#else
    #ifdef _MSC_VER
        #define NTC_API __declspec(dllimport)
    #else
        #define NTC_API
    #endif
#endif

#include "shaders/InferenceConstants.h"

namespace ntc
{

// Update the interface version whenever changes to the LibNTC API are made.
constexpr uint32_t InterfaceVersion = 0x25'01'24'01; // Year, month, day, ordinal

// Version information for the library, see GetLibraryVersion()
struct VersionInfo
{
    int major;
    int minor;
    int point;

    char const* branch;
    char const* commitHash;
};

enum class Status
{ 
    Ok                = 0,
    InterfaceMismatch = 1,
    CudaError         = 2,
    CudaUnavailable   = 3,
    FileUnavailable   = 4,
    FileUnrecognized  = 5,
    FileIncompatible  = 6,
    Incomplete        = 7,
    Unsupported       = 8,
    InternalError     = 9,
    InvalidArgument   = 10,
    InvalidState      = 11,
    IOError           = 12,
    NotImplemented    = 13,
    OutOfMemory       = 14,
    OutOfRange        = 15,
    ShaderUnavailable = 16,
    UnknownError      = 17,
};

// This interface declares functions of a custom CPU memory allocator for the NTC library.
class IAllocator
{
public:
    // Allocates the specified amount of memory.
    virtual void* Allocate(size_t size) = 0;

    // Releases the memory previously allocated with the same allocator.
    virtual void Deallocate(void* ptr, size_t size) = 0;

    virtual ~IAllocator() = default;
};


// This interface declares the functions needed by the NTC library to access application-provided streams,
// such as files and memory buffers.
// Predefined implementations can be obtained with OpenFile and OpenMemory functions in IContext.
class IStream
{
public:
    // Read 'size' bytes from the current position in the stream, returns 'true' if successful.
    // On error, returns 'false'; contents of 'dst' don't matter in this case.
    virtual bool Read(void* dst, size_t size) = 0;

    // Write 'size' bytes at the current position in the stream, returns 'true' if successful.
    virtual bool Write(const void* src, size_t size) = 0;

    // Navigate to the provided offset from the beginning of the stream, returns 'true' if successful.
    // Only used in streams open for reading.
    virtual bool Seek(uint64_t offset) = 0;

    // Returns the current position in the stream.
    virtual uint64_t Tell() = 0;

    // Returns the size of the stream in bytes for streams opened for reading.
    virtual uint64_t Size() = 0;

    virtual ~IStream() = default;
};

// A structure that describes the range of data in a stream.
// To describe a complete stream or file, use 'EntireStream'.
struct StreamRange
{
    uint64_t offset = 0;
    uint64_t size = 0;
};

constexpr StreamRange EntireStream = StreamRange{0, ~0ull};

constexpr int DisableCudaDevice = -1; // Use for ContextParameters::cudaDevice
constexpr size_t BlockCompressionAccelerationBufferSize = 8*64*4; // 8 BC7 modes, 64 uints per mode
constexpr uint8_t BlockCompressionMaxQuality = 255;

// *** Texture sets ***

// This struct defines the shape of the latent space, i.e. compressed representation of the textures.
struct LatentShape
{
    int gridSizeScale = 4;
    int highResFeatures = 16;
    int lowResFeatures = 16;
    int highResQuantBits = 2;
    int lowResQuantBits = 4;

    bool operator==(const LatentShape& other) const
    {
        return gridSizeScale == other.gridSizeScale
            && highResFeatures == other.highResFeatures
            && lowResFeatures == other.lowResFeatures
            && highResQuantBits == other.highResQuantBits
            && lowResQuantBits == other.lowResQuantBits;
    }

    bool operator!=(const LatentShape& other) const
    {
        return !(*this == other);
    }

    static constexpr LatentShape Empty()
    {
        return LatentShape{ 0, 0, 0, 0, 0 };
    }

    bool IsEmpty() const
    {
        return *this == Empty();
    }
};

struct TextureSetDesc
{
    int width = 0;
    int height = 0;
    int channels = 0;
    int mips = 1;

    bool operator==(const TextureSetDesc& other) const
    {
        return width == other.width
            && height == other.height
            && channels == other.channels
            && mips == other.mips;
    }

    bool operator!=(const TextureSetDesc& other) const
    {
        return !(*this == other);
    }
};

struct TextureSetFeatures
{
    // Number of bytes allocated in the staging buffer for ITextureSet::Read/WriteChannels with Host memory.
    // Set this to accommodate for the largest pixel format that is going to be used in these operations.
    // This parameter can be zero, which means Read/WriteChannels with Host memory will not be available.
    int stagingBytesPerPixel = 4;

    // Custom width of the staging buffer. 0 means it matches the width of the texture set.
    int stagingWidth = 0;

    // Custom height of the staging buffer. 0 means it matches the height of the texture set.
    int stagingHeight = 0;

    // Enables the allocation of resources for compression operations.
    bool enableCompression = true;

    // Toggles the memory allocation for color data between single buffer (shared reference and output) and two buffers.
    // When a single buffer is used, the ITextureSet::Decompress function will overwrite the reference data,
    // so this mode is not suitable for repeated compression runs.
    bool separateRefOutData = false;
};

struct CompressionSettings
{
    int trainingSteps = 100000;
    int stepsPerIteration = 1000;
    float networkLearningRate = 0.005f;
    float gridLearningRate = 0.1f;
    int kPixelsPerBatch = 64;

    // Controls whether a separate MLP will be trained for FP8 inference.
    // Enabling it repurposes a small portion of the training steps, thereby slightly reducing image quality,
    // and stores a second version of the MLP in the NTC files, making them larger by 16 kB or so.
    bool trainFP8Weights = true;

    // Set to a nonzero value to get more stable compression results
    uint32_t randomSeed = 0;
    bool stableTraining = false;
};

#define NTC_MAX_KPIXELS_PER_BATCH 128

enum class ChannelFormat
{
    UNKNOWN = 0,
    
    UNORM8 = 1,
    UNORM16 = 2,
    FLOAT16 = 3,
    FLOAT32 = 4,
    UINT32 = 5
};

enum class BlockCompressedFormat
{
    None = 0,

    BC1 = 1,
    BC2 = 2,
    BC3 = 3,
    BC4 = 4, // Unsigned
    BC5 = 5, // Unsigned
    BC6 = 6, // Unsigned
    BC7 = 7
};

enum class ColorSpace
{
    Linear = 0,
    sRGB = 1,
    HLG = 2
};

enum class TextureDataPage
{
    Reference,
    Output
};

// Identifies which type of shared handle is used in SharedTextureDesc, see cudaExternalMemoryHandleType.
enum class SharedHandleType
{
    D3D12Resource,
    OpaqueWin32,
    OpaqueFd
};

enum class AddressSpace
{
    Host,
    Device
};

enum class GraphicsAPI
{
    None,
    D3D12,
    Vulkan
};

enum class InferenceWeightType
{
    GenericInt8,
    GenericFP8,
    CoopVecInt8,
    CoopVecFP8
};

struct SharedTextureDesc
{
    int width = 0;
    int height = 0;
    int channels = 0;
    int mips = 1;
    ChannelFormat format = ChannelFormat::UNORM8;

    // This flag must match how the texture was created on the graphics API side to avoid failures and image corruption.
    // D3D12: set dedicatedResource = true when CreateCommittedResource was used (and not CreatePlacedResource)
    // Vulkan: set dedicatedResource = true when VkMemoryDedicatedAllocateInfo was used in the vkAllocateMemory call
    bool dedicatedResource = false;

    size_t sizeInBytes = 0;
    SharedHandleType handleType = SharedHandleType::D3D12Resource;
    uint64_t sharedHandle = 0;
};

struct Point
{
    int x = 0;
    int y = 0;

    Point()
    { }

    Point(int _x, int _y)
        : x(_x), y(_y)
    { }
};

struct Rect
{
    int left = 0;
    int top = 0;
    int width = 0;
    int height = 0;

    Rect()
    { }
    
    Rect(int _width, int _height)
        : left(0), top(0), width(_width), height(_height)
    { }

    Rect(int _left, int _top, int _width, int _height)
        : left(_left), top(_top), width(_width), height(_height)
    { }
};

class ISharedTexture
{
public:
    // Returns the descriptor of the shared texture provided at its registration time.
    virtual const SharedTextureDesc& GetDesc() const = 0;

    virtual ~ISharedTexture() = default;
};

// Maximum size of all constant buffer structures used by precompiled compute shaders.
// Whether the structures actually fit is verified inside LibNTC using static_assert.
static constexpr size_t MaxComputePassConstantSize = 43*16;

// The ComputePassDesc structure holds information necessary to create and run a compute pass using a graphics API (DX12 or Vulkan).
struct ComputePassDesc
{
    // Bytecode for the compute shader. 
    // The pointer references a static array inside the NTC DLL which is therefore persistent.
    // For the resources that need to be bound, see the comments to methods that produce ComputePassDesc.
    const void* computeShader = nullptr;
    size_t computeShaderSize = 0;

    // Data for the constant buffer to be used by the compute shader.
    uint8_t constantBufferData[MaxComputePassConstantSize];
    size_t constantBufferSize = 0;

    // Data for the weight buffer to be used by the decompression compute shader. Will be NULL for other shaders.
    // The pointer references a buffer owned by the ITextureSetMetadata object and will become invalid if that object is destroyed.
    // The buffer contents are static during the lifetime of ITextureSetMetadata.
    const void* weightBufferData = nullptr;
    size_t weightBufferSize = 0;

    // Dispatch dimensions for the compute shader.
    int dispatchWidth = 0;
    int dispatchHeight = 0;
};

// The InferenceData structure holds information necessary to use Inference on Sample in a graphics shader
// for a specific texture set.
struct InferenceData
{
    // Constant data to be passed to 'NtcSampleTextureSet' and other shader functions.
    NtcTextureSetConstants constants;
};


// The ITextureMetadata interface is used to provide information about a specific texture in a texture set.
class ITextureMetadata
{
public:
    // Sets a descriptive name for the texture in the texture set.
    virtual void SetName(const char* name) = 0;

    // Returns the name previously set with 'SetName'.
    virtual const char* GetName() const = 0;

    // Sets the range of channels in the texture set that correspond to this texture.
    virtual Status SetChannels(int firstChannel, int numChannels) = 0;
    
    // Returns the channel range previously set with 'SetChannels'.
    virtual void GetChannels(int& outFirstChannel, int& outNumChannels) const = 0;

    // Returns the first channel in the texture set occupied by this texture.
    // Shortcut for GetChannels.
    virtual int GetFirstChannel() const = 0;

    // Returns the number of channels in the texture set occupied by this texture.
    // Shortcut for GetChannels.
    virtual int GetNumChannels() const = 0;

    // Sets the format that should be used for creating the texture for decompression with 'MakeDecompressionComputePass'.
    virtual void SetChannelFormat(ChannelFormat format) = 0;

    // Returns the value previously set with 'SetChannelFormat'.
    virtual ChannelFormat GetChannelFormat() const = 0;

    // Sets the BC format that should be used for encoding this texture after decompression.
    virtual void SetBlockCompressedFormat(BlockCompressedFormat format) = 0;

    // Returns the value previously set with 'SetBlockCompressedFormat'.
    virtual BlockCompressedFormat GetBlockCompressedFormat() const = 0;

    // Sets the color space for RGB data when the texture is decompressed using 'MakeDecompressionComputePass'.
    virtual void SetRgbColorSpace(ColorSpace colorSpace) = 0;

    // Returns the value previously set with 'SetRgbColorSpace'.
    virtual ColorSpace GetRgbColorSpace() const = 0;

    // Sets the color space for Alpha channel data when the texture is decompressed using 'MakeDecompressionComputePass'.
    virtual void SetAlphaColorSpace(ColorSpace colorSpace) = 0;

    // Returns the value previously set with 'SetAlphaColorSpace'.
    virtual ColorSpace GetAlphaColorSpace() const = 0;

    // Sets the acceleration data for BC compression obtained by running a compression pass.
    // The data should be read from the buffer used in the compression pass, see 'MakeBlockCompressionComputePass'.
    virtual Status SetBlockCompressionAccelerationData(void const* pData, size_t size) = 0;

    // Returns 'true' if 'SetBlockCompressionAccelerationData' was previously called with valid data.
    virtual bool HasBlockCompressionAccelerationData() const = 0;

    // Sets the default quality value for block compression of this texture.
    virtual void SetBlockCompressionQuality(uint8_t quality) = 0;

    // Returns the value previously set with 'SetBlockCompressionQuality'.
    // This value should be passed to 'MakeBlockCompressionComputePass'.
    // The default value is 'BlockCompressionMaxQuality'.
    virtual uint8_t GetBlockCompressionQuality() const = 0;

    virtual ~ITextureMetadata() = default;
};


// The ITextureSetMetadata interface is used to provide information about texture set contents without allocating memory
// for the actual texture data. This should be useful when decompressing textures at game load time,
// so that the engine can pre-allocate the textures and then launch decompression kernels.
class ITextureSetMetadata
{
public:
    // Returns the descriptor of the texture set.
    virtual TextureSetDesc const& GetDesc() const = 0;

    // Returns the current latent shape of the texture set.
    virtual LatentShape const& GetLatentShape() const = 0;

    // Creates a new ITextureMetadata object describing a texture in the set, and returns it.
    virtual ITextureMetadata* AddTexture() = 0;

    // Removes and deletes a previously added ITextureMetadata object. If no such object exists, returns OutOfRange.
    virtual Status RemoveTexture(ITextureMetadata* texture) = 0;

    // Removes and deletes all ITextureMetadata objects.
    virtual void ClearTextureMetadata() = 0;

    // Returns the count of the ITextureSetMetadata objects.
    virtual int GetTextureCount() const = 0;

    // Returns the ITextureSetMetadata object by index.
    virtual ITextureMetadata* GetTexture(int textureIndex) = 0;

    // Returns the ITextureSetMetadata object by index (const version).
    virtual ITextureMetadata const* GetTexture(int textureIndex) const = 0;

    // Returns the color space that is used in the internal encoding for a specific channel.
    virtual ColorSpace GetChannelStorageColorSpace(int channel) const = 0;

    // Returns one of the NTC_NETWORK_{SMALL,MEDIUM,LARGE,XLARGE} constants describing
    // which version of the neural network (MLP) should be used to decode this texture set.
    virtual int GetNetworkVersion() const = 0;

    // Returns the range of data from the compressed stream or file that contains all latents needed
    // to decompress the specified range of mip levels.
    virtual Status GetStreamRangeForLatents(int firstMip, int numMips, StreamRange& outRange) const = 0;

    // Returns the range of mip levels including 'mipLevel' that are represented by the same latent image.
    // If a slice of any of these mips is requested for partial inference, scaled slices of all of them
    // will also be available for free.
    virtual Status GetFusedMipLevels(int mipLevel, int* pOutFirstFusedMip, int* pOutLastFusedMip) const = 0;

    // Returns the number of latent images (neural MIPs) in this texture set.
    virtual int GetNumLatentImages() const = 0;

    // Returns the range of color MIP levels that are encoded by the same latent image.
    // When no such MIP levels exist, returns OutOfRange and the output parameters are unchanged.
    virtual Status GetMipLevelsForLatentImage(int latentImageIndex, int* pOutFirstColorMip, int* pOutLastColorMip) const = 0;
    
    // Returns a pointer and size for the data that should be uploaded to the weight buffer
    // that is passed to NtcSampleTextureSet(...) in shaders using this texture set.
    virtual Status GetInferenceWeights(InferenceWeightType weightType, void const** pOutData, size_t* pOutSize) const = 0;

    // Returns true if the given inference weight type is included with this texture set
    // and available on the current graphics device.
    virtual bool IsInferenceWeightTypeSupported(InferenceWeightType weightType) const = 0;

    virtual ~ITextureSetMetadata() = default;
};

// Parameters for ITextureSet::WriteChannels(...)
struct WriteChannelsParameters
{
    // Destination mip level in the NTC internal representation
    int mipLevel = 0;

    // First channel in the NTC internal representation
    int firstChannel = 0;

    // Number of channels to process, 1..NTC_MAX_CHANNELS
    int numChannels = 0;

    // Texture data in CPU or GPU memory (see addressSpace)
    unsigned char const* pData = nullptr;

    // Defines which address space the texture data is located in (CPU or GPU)
    AddressSpace addressSpace = AddressSpace::Host;

    // Width of the texture data
    int width = 0;

    // Height of the texture data
    int height = 0;

    // Distance in bytes between adjacent pixels
    size_t pixelStride = 0;

    // Distance in bytes between adjacent image rows
    size_t rowPitch = 0;

    // Storage format of the color components
    ChannelFormat channelFormat = ChannelFormat::UNKNOWN;

    // Points at an array of 'numChannels' items. NULL means all color spaces are assumed to be Linear.
    ColorSpace const* srcColorSpaces = nullptr;

    // Points at an array of 'numChannels' items. NULL means all color spaces are assumed to be Linear.
    ColorSpace const* dstColorSpaces = nullptr;

    // Flip the texture along the Y axis
    bool verticalFlip = false;
};

// Parameters for ITextureSet::ReadChannels(...)
struct ReadChannelsParameters
{
    // Source data page in the NTC internal representation.
    // If the texture set was created with 'separateRefOutData = false', both pages map to the same memory.
    TextureDataPage page = TextureDataPage::Output;

    // Source mip level in the NTC internal representation
    int mipLevel = 0;

    // First channel in the NTC internal representation
    int firstChannel = 0;

    // Number of channels to process, 1..NTC_MAX_CHANNELS
    int numChannels = 0;

    // Memory for the texture data in CPU or GPU memory (see addressSpace)
    unsigned char* pOutData = nullptr;

    // Defines which address space the texture data is located in (CPU or GPU)
    AddressSpace addressSpace = AddressSpace::Host;

    // Width of the texture data
    int width = 0;
    
    // Height of the texture data
    int height = 0;

    // Distance in bytes between adjacent pixels
    size_t pixelStride = 0;

    // Distance in bytes between adjacent image rows
    size_t rowPitch = 0;

    // Storage format of the color components
    ChannelFormat channelFormat = ChannelFormat::UNKNOWN;

    // Points at an array of 'numChannels' items. NULL means all color spaces are assumed to be Linear.
    ColorSpace const* dstColorSpaces = nullptr;

    // Controls whether dithering is applied before rounding the color data to the output format
    bool useDithering = false;
};

// Parameters for ITextureSet::WriteChannelsFromTexture(...)
struct WriteChannelsFromTextureParameters
{
    // Destination mip level in the NTC internal representation
    int mipLevel = 0;

    // First channel in the NTC internal representation
    int firstChannel = 0;

    // Number of channels to process, 1..4
    int numChannels = 0;

    // Shared texture object
    ISharedTexture* texture = nullptr;

    // Source mip level in the texture object
    int textureMipLevel = 0;

    // Color space for the RGB channels in the texture
    ColorSpace srcRgbColorSpace = ColorSpace::Linear;

    // Color space for the Alpha channel in the texture
    ColorSpace srcAlphaColorSpace = ColorSpace::Linear;

    // Color space for the RGB channels in the NTC internal representation
    ColorSpace dstRgbColorSpace = ColorSpace::Linear;

    // Color space for the Alpha channel in the NTC internal representation
    ColorSpace dstAlphaColorSpace = ColorSpace::Linear;

    // Flip the texture along the Y axis
    bool verticalFlip = false;
};

// Parameters for ITextureSet::ReadChannelsIntoTexture(...)
struct ReadChannelsIntoTextureParameters
{
    // Source data page in the NTC internal representation.
    // If the texture set was created with 'separateRefOutData = false', both pages map to the same memory.
    TextureDataPage page = TextureDataPage::Output;

    // Source mip level in the NTC internal representation
    int mipLevel = 0;
    
    // First channel in the NTC internal representation
    int firstChannel = 0;

    // Number of channels to process, 1..4
    int numChannels = 0;

    // Shared texture object
    ISharedTexture* texture = nullptr;

    // Destination mip level in the texture object
    int textureMipLevel = 0;

    // Color space for the RGB channels in the texture
    ColorSpace dstRgbColorSpace = ColorSpace::Linear;

    // Color space for the Alpha channel in the texture
    ColorSpace dstAlphaColorSpace = ColorSpace::Linear;

    // Controls whether dithering is applied before rounding the color data to the output format
    bool useDithering = false;
};

// Statistics provided by ITextureSet::Compress(...)
struct CompressionStats
{
    int currentStep = 0;
    float learningRate = 0.f;
    float millisecondsPerStep = 0.f;
    float loss = 0.f;
    float lossScale = 0.f;
};

// Statistics provided by ITextureSet::Decompress(...)
// Loss values are Mean Square Error, use LossToPSNR(...) to convert them to dB when the signal range is 0-1.
struct DecompressionStats
{
    float perChannelLoss[NTC_MAX_CHANNELS];
    float perMipLoss[NTC_MAX_MIPS];
    float overallLoss = 0;
    float gpuTimeMilliseconds = 0;
};

// The ITextureSet interface extends ITextureSetMetadata and provides APIs for compressing and decompressing texture sets
// using CUDA. Unlike metadata objects, ITextureSet allocates GPU memory to hold all texture data and extra buffers for network training.
class ITextureSet : virtual public ITextureSetMetadata
{
public:
    // Changes the latent shape of this texture set.
    // All compressed data is lost.
    // This function cannot be used while compression is in progress.
    // The networkVersion parameter can be used to specify the network version used for this texture set.
    // If networkVersion is set to NTC_NETWORK_UNKNOWN, the smallest compatible version is selected.
    virtual Status SetLatentShape(LatentShape const& newShape, int networkVersion = NTC_NETWORK_UNKNOWN) = 0;
    
    // Returns a conservative estimate for the size of memory buffer needed to save the texture set into a memory stream.
    virtual uint64_t GetOutputStreamSize() = 0;

    // Saves the compressed texture set into a stream.
    virtual Status SaveToStream(IStream* stream) = 0;

    // Loads the compressed texture set from a stream if the stored data has the same dimensions as this texture set.
    // If the stored data has different dimensions, returns FileIncompatible, and the data in memory is unchanged.
    virtual Status LoadFromStream(IStream* stream) = 0;

    // Saves the compressed texture set into a buffer in memory.
    // The pSize parameter must point to a variable that contains the buffer size on input
    // and will contain the actual written size after the function executes.
    // Shortcut for IContext::OpenMemory and SaveToStream.
    virtual Status SaveToMemory(void* pData, size_t* pSize) = 0;

    // Loads the compressed texture set from a buffer in memory.
    // Shortcut for IContext::OpenReadOnly and LoadFromStream.
    virtual Status LoadFromMemory(void const* pData, size_t size) = 0;

    // Saves the compressed texture set into a file.
    // Shortcut for IContext::OpenFile and SaveToStream.
    virtual Status SaveToFile(char const* fileName) = 0;

    // Loads the compressed texture set from a file.
    // Shortcut for IContext::OpenFile and LoadFromStream.
    virtual Status LoadFromFile(char const* fileName) = 0;

    // *** Raw texture data access ***

    // Stores texture data into the internal multichannel texture set representation.
    virtual Status WriteChannels(WriteChannelsParameters const& params) = 0;

    // Extracts texture data from the internal multichannel texture set representation.
    virtual Status ReadChannels(ReadChannelsParameters const& params) = 0;

    // Stores texture data from a shared texture object into the internal representation.
    virtual Status WriteChannelsFromTexture(WriteChannelsFromTextureParameters const& params) = 0;

    // Extracts texture data from the internal representation into a shared texture object.
    virtual Status ReadChannelsIntoTexture(ReadChannelsIntoTextureParameters const& params) = 0;

    // Generates all mip levels starting from 1 until the last one, using the data
    // from mip level 0 and a 2x2 box downsampling filter.
    virtual Status GenerateMips() = 0;

    // *** Compression ***

    // Copies the texture data to the GPU and prepares other GPU resources.
    virtual Status BeginCompression(const CompressionSettings& settings) = 0;

    // Runs the training steps and provides the current loss function values.
    // When all training steps are done, this function will return Status::Ok, and before that Status::Incomplete.
    virtual Status RunCompressionSteps(CompressionStats* pOutStats) = 0;

    // Runs the quantization passes and copies the final compressed data back to the CPU.
    virtual Status FinalizeCompression() = 0;

    // Cancels the compression process and returns the texture to the original uncompressed state.
    virtual void AbortCompression() = 0;

    // *** Decompression ***

    // Decompresses the previously compressed and loaded texture set into its internal representation.
    // Use ReadChannels to extract the decompressed data.
    virtual Status Decompress(DecompressionStats* pOutStats, bool useFP8Weights = false) = 0;
    
    // *** Misc ***

    // Sets the index of a channel that contains an alpha mask. Negative values mean no mask.
    // The 0.0 and 1.0 values in the mask channel will be compressed with greater accuracy than in other channels.
    // When discardMaskedOutPixels = true, pixels where the mask is zero are assumed to have undefined data in all other channels.
    virtual Status SetMaskChannelIndex(int index, bool discardMaskedOutPixels) = 0;

    // Sets a parameter that can be used for experimental code changes in the library.
    // Has no effect on release builds.
    virtual void SetExperimentalKnob(float value) = 0;
};

// This interface implements a fast search algorithm to experimentally identify the optimal compression preset
// that reaches a target PSNR for one texture set.
// See the integration guide for a usage example.
class IAdaptiveCompressionSession
{
public:
    // Resets the session and defines a new target PSNR value.
    // Call this first before any other methods.
    // If maxBitsPerPixel is greater than 0, the search is constrained to not exceed that value.
    // If networkVersion is not UNKNOWN, the search is constrained to be compatible with that network.
    virtual Status Reset(float targetPsnr, float maxBitsPerPixel = 0.f, int networkVersion = NTC_NETWORK_UNKNOWN) = 0;

    // Returns 'true' if no more compression runs are needed.
    virtual bool Finished() = 0;

    // Returns the current BPP and latent shape for the next compression run if Finished() == false,
    // and the final BPP and latent shape if Finished() == true.
    virtual void GetCurrentPreset(float*  pOutBitsPerPixel, LatentShape* pOutLatentShape) = 0;

    // Submits the result of the latest compression run and moves on to the next experiment.
    virtual void Next(float currentPsnr) = 0;

    // Returns the index of the compression run since Reset(...) that is selected as the final result, starting at 0.
    // Only valid when Finished() returns true, otherwise GetIndexOfFinalRun() returns -1.
    virtual int GetIndexOfFinalRun() = 0;

    virtual ~IAdaptiveCompressionSession() = default;
};


// This structure describes which NTC channels should be decompressed into a single texture object,
// where that object is located, and how the data should be converted. Used with 'IContext::MakeDecompressionComputePass'.
struct OutputTextureDesc
{
    // Index of the texture UAV descriptor in the descriptor table, starting from 'firstOutputDescriptorIndex'
    int descriptorIndex = 0;

    // First channel in the NTC texture set to put into this texture's Red channel
    int firstChannel = 0;

    // Number of channels from the NTC texture set to put into this texture
    int numChannels = 0;

    // Color space for the RGB channels stored in this texture (normally Linear or sRGB)
    ColorSpace rgbColorSpace = ColorSpace::Linear;

    // Color space for the Alpha channel stored in this texture (normally Linear)
    ColorSpace alphaColorSpace = ColorSpace::Linear;

    // Multiplier for the dithering noise that is added before storing the output value.
    // For 8-bit textures, this should be 1/255.
    float ditherScale = 0.f;
};

struct MakeDecompressionComputePassParameters
{
    // The metadata for the NTC texture set to decompress.
    ITextureSetMetadata* textureSetMetadata = nullptr;

    // Specifies which part of the input stream or file is available as a latent buffer.
    StreamRange latentStreamRange = EntireStream;

    // Mip level from the NTC texture set.
    int mipLevel = 0;

    // Offset for the descriptor indices in the descriptor table bound as the array of output textures.
    int firstOutputDescriptorIndex = 0;

    // Optional, specifies a rectangle within the source textures to decompress.
    Rect const* pSrcRect = nullptr;

    // Optional, specifies an offset in the destination texture to place the results into.
    // If not specified, the offset is assumed to be equal to the origin of 'pSrcRect'.
    Point const* pDstOffset = nullptr;

    // Optional, describes the mapping of NTC channels into output textures.
    // If not specified, the mapping is derived from the metadata.
    OutputTextureDesc const* pOutputTextures = nullptr;
    int numOutputTextures = 0;

    // Controls whether the FP8 version of decompression math should be used, if supported by the device
    // and if the weights are present in the texture set. FP8 decompression is somewhat faster but has higher error.
    bool enableFP8 = false;
};

struct MakeBlockCompressionComputePassParameters
{
    // Extent of the texture data to compress from the source texture.
    Rect srcRect;

    // Offset of the top left output block in the destination texture.
    Point dstOffsetInBlocks;

    // BCn format to encode into, must not be None.
    BlockCompressedFormat dstFormat = BlockCompressedFormat::None;

    // Alpha threshold for transparent pixels in BC1, has no effect on other formats.
    float alphaThreshold = 1.f / 255.f;

    // Controls whether BC7 acceleration data is written by the compression pass.
    // When true, the output buffer for this data must be bound to the shader.
    bool writeAccelerationData = false;
    
    // Optional, texture object to get BC7 acceleration data from.
    ITextureMetadata const* texture = nullptr;

    // BC7 compression quality, only effective when 'texture' is provided and if it has the acceleration data.
    uint8_t quality = BlockCompressionMaxQuality;
};

struct MakeImageDifferenceComputePassParameters
{
    // Extent of the texture data to compare, in both textures.
    Rect extent;

    // Controls whether pixels whose alpha is lower than a threshold are considered transparent,
    // and therefore their RGB difference is ignored.
    bool useAlphaThreshold = false;

    // Opacity threshold for transparent pixels.
    float alphaThreshold = 1.f / 255.f;

    // If true, the difference pass will produce a Mean Square Log Error metric instead of regular MSE.
    // In that case, the absolute value of inputs is taken before computing the error.
    bool useMSLE = false;

    // Offset in the output buffer, in bytes. Must be a multiple of 4.
    uint32_t outputOffset = 0;
};

class IContext
{
public:

    // *** Stream (file-like) objects ***

    // Opens a regular filesystem file for reading "rb" or writing "wb", returns the stream object into pOutStream.
    virtual Status OpenFile(const char* fileName, bool write, IStream** pOutStream) const = 0;

    // Closes the file previously opened with OpenFile.
    virtual void CloseFile(IStream* stream) const = 0;

    // Opens a memory buffer for reading and writing, returns the stream object into pOutStream.
    virtual Status OpenMemory(void* pData, size_t size, IStream** pOutStream) const = 0;

    // Opens a memory buffer for reading, returns the stream object into pOutStream.
    virtual Status OpenReadOnlyMemory(void const* pData, size_t size, IStream** pOutStream) const = 0;

    // Closes the memory buffer previously opened with OpenMemory or OpenReadOnlyMemory.
    virtual void CloseMemory(IStream* stream) const = 0;


    // *** Texture set creation and destruction ***

    // Creates a context for working on a single texture set, including CPU backing store for the texture data and all
    // necessary GPU resources for the specified operations (compression, decompression)
    virtual Status CreateTextureSet(const TextureSetDesc& desc,
        const TextureSetFeatures& features, ITextureSet** pOutTextureSet) const = 0;

    // Destroys the previously created texture set.
    virtual void DestroyTextureSet(ITextureSet* textureSet) const = 0;


    // *** Construction of metadata or texture sets from stream objects ***

    // Reads the compressed texture header from a stream and creates a metadata object describing the texture set.
    virtual Status CreateTextureSetMetadataFromStream(IStream* inputStream,
        ITextureSetMetadata** pOutMetadata) const = 0;

    // Destroys the previously created metadata object.
    virtual void DestroyTextureSetMetadata(ITextureSetMetadata* textureSetMetadata) const = 0;

    // Loads the compressed texture set from a stream.
    virtual Status CreateCompressedTextureSetFromStream(IStream* inputStream,
        const TextureSetFeatures& features, ITextureSet** pOutTextureSet) const = 0;

    // Loads the compressed texture set from a memory buffer.
    // Shortcut for OpenReadOnlyMemory and CreateCompressedTextureSetFromStream.
    virtual Status CreateCompressedTextureSetFromMemory(void const* pData, size_t size,
        const TextureSetFeatures& features, ITextureSet** pOutTextureSet) const = 0;

    // Loads the compressed texture set from a file.
    // Shortcut for OpenFile and CreateCompressedTextureSetFromStream.
    virtual Status CreateCompressedTextureSetFromFile(char const* fileName,
        const TextureSetFeatures& features, ITextureSet** pOutTextureSet) const = 0;
    

    // *** Shared (DX12 or Vulkan) texture interface ***

    // Registers a shared texture with NTC and returns an object representing it.
    virtual Status RegisterSharedTexture(const SharedTextureDesc& desc, ISharedTexture** pOutTexture) const = 0;

    // Releases the object representing a shared texture.
    virtual void ReleaseSharedTexture(ISharedTexture* texture) const = 0;


    // *** Adaptive compression ***

    // Creates an adaptive compression session object.
    virtual Status CreateAdaptiveCompressionSession(IAdaptiveCompressionSession** pOutSession) const = 0;

    // Destroys the adaptive compression session object.
    virtual void DestroyAdaptiveCompressionSession(IAdaptiveCompressionSession* session) const = 0;


    // *** Graphics API passes ***

    // Describes a compute pass that decompresses a certain mip level of an NTC texture set into texture objects.
    // The following resources need to be bound to the pipeline:
    // - ConstantBuffer at b0 containing the CB data (ComputePassDecs::constantBufferData) (Vulkan: dset 0, binding 0)
    // - ByteAddressBuffer at t1 containing the latent data (the latentStreamRange portion of the compressed stream) (Vulkan: dset 0, binding 1)
    // - ByteAddressBuffer at t2 containing the weight data (ComputePassDecs::weightBufferData) (Vulkan: dset 0, binding 2)
    // - Unsized array of RWTexture2D<float4> at u0... containing the UAVs for the destination textures (Vulkan: dset 1, binding 0)
    // If this function returns Status::ShaderUnavailable, the rest of the data in 'pOutComputePass' is still valid.
    virtual Status MakeDecompressionComputePass(MakeDecompressionComputePassParameters const& params,
        ComputePassDesc* pOutComputePass) const = 0;

    // Describes a compute pass that will compress a 2D texture (single mip level) into a BCn format and store it into a buffer.
    // The following resources need to be bound to the pipeline:
    // - ConstantBuffer at b0 containing the CB data (ComputePassDecs::constantBufferData) (Vulkan: dset 0, binding 0)
    // - Texture2D<float4> at t1 containing the source texture (Vulkan: dset 0, binding 1)
    // - RWTexture2D<uint2|uint4> at u2 containing the destination texture (Vulkan: dset 0, binding 2) - use uint2 for BC1 and BC4, and uint4 for all other BC modes
    // - RWByteAddressBuffer at u3 containing the buffer for acceleration data (Vulkan: dset 0, binding 3) - only if writeAccelerationData is true and only for BC7
    // If this function returns Status::ShaderUnavailable, the rest of the data in 'pOutComputePass' is still valid.
    virtual Status MakeBlockCompressionComputePass(MakeBlockCompressionComputePassParameters const& params,
        ComputePassDesc* pOutComputePass) const = 0;
    
    // Describes a compute pass that compares two images and writes per-channel MSE values into a provided UAV buffer.
    // The following resources need to be bound to the pipeline:
    // - ConstantBuffer at b0 containing the CB data (ComputePassDecs::constantBufferData) (Vulkan: dset 0, binding 0)
    // - Texture2D<float4> at t1, t2 containing the source textures (Vulkan: dset 0, binding 1, 2)
    // - RWByteAddressBuffer at u3 containing the destination buffer (Vulkan: dset 0, binding 3)
    // * Note: the destination buffer needs to be at least 32 bytes large, cleared with zeros before executing the pass.
    //   On completion, the buffer will contain 4 uint64's with per-channel MSE values in fixed-point 16.48 format.
    //   Use ntc::DecodeImageDifferenceResult to convert those values into regular double numbers.
    // If this function returns Status::ShaderUnavailable, the rest of the data in 'pOutComputePass' is still valid.
    virtual Status MakeImageDifferenceComputePass(MakeImageDifferenceComputePassParameters const& params,
        ComputePassDesc* pOutComputePass) const = 0;

    // Populates the data necessary to run inference on sample with this texture set: the constant buffer
    // and the weights buffer. To sample a neural texture set, include "libntc/shaders/Inference.hlsli"
    // and call the NtcSampleTextureSet(...) function, providing the constants and weights returned by this function,
    // as well a portion of the NTC file as the 'latentsBuffer' parameter.
    // The 'latentStreamRange' parameter specifies which part of the input stream or file is available as the latent buffer.
    // The 'weightType' parameter indicates which math version the inference pass will use.
    // If this parameter is true but the extensions are unavailable, the function will return Status::Unsuppported.
    // Note that this function does not validate that the provided latent stream range contains the mip levels
    // that will be sampled; failure to provide necessary data will lead to silent corruption.
    virtual Status MakeInferenceData(ITextureSetMetadata* textureSetMetadata, StreamRange latentStreamRange,
        InferenceWeightType weightType, InferenceData* pOutInferenceData) const = 0;

    // Populates the data necessary to run inference on sample with the given texture set for a subset of the pixels.
    // The subset of latent data is read from the stream and placed into the provided buffer at 'pOutLatentData',
    // whose size is provided in 'pInOutLatentSize', and the actual written size is returned through the same parameter.
    // When 'pOutLatentData' is NULL, the required buffer size is returned in 'pInOutLatentSize', and the function
    // returns Status::Incomplete. No stream reading is done in that case, and 'inputStream' may be NULL.
    virtual Status MakePartialInferenceData(ITextureSetMetadata* textureSetMetadata, IStream* inputStream,
        int firstMipLevel, int numMipLevels, Rect firstMipSlice,
        InferenceWeightType weightType, InferenceData* pOutInferenceData, void* pOutLatentData, size_t* pInOutLatentSize) const = 0;

    // Calculates the buffer size that will be sufficient to extract a portion of latent data for any
    // pixel rectangle of given size by MakePartialInferenceData(...). The size is returned through 'pOutLatentSize'.
    // The 'sliceAlignment' parameter specifies that the slice origins will be placed at a multiple of some number,
    // which may reduce the memory requirements.
    virtual Status GetConservativeLatentBufferSize(ITextureSetMetadata* textureSetMetadata,
        int firstMipLevel, int numMipLevels, int firstMipSliceWidth, int firstMipSliceHeight, int sliceAlignment,
        size_t* pOutLatentSize) const = 0;
    
    // Returns true if the graphics device supports all the required features and extensions for cooperative vector
    // based decompression of NTC texture sets using Int8 math.
    virtual bool IsCooperativeVectorInt8Supported() const = 0;

    // Returns true if the graphics device supports all the required features and extensions for cooperative vector
    // based decompression of NTC texture sets using FP8 math.
    virtual bool IsCooperativeVectorFP8Supported() const = 0;

    virtual ~IContext() = default;
};

struct ContextParameters
{
    // A constant needed to validate that the library was built with the same header version
    // as the application. Do not modify the default value.
    uint32_t interfaceVersion = InterfaceVersion;
    
    // Custom allocator interface, optional.
    // If pAllocator is NULL, a built-in runtime allocator is used.
    // Application-provided allocators *must* be kept alive for the entire lifetime of the context.
    IAllocator* pAllocator = nullptr;

    // Index of the CUDA device to use, or ntc::DisableCudaDevice to skip initializing CUDA.
    int cudaDevice = 0;

    // Graphics device interface, optional.
    GraphicsAPI graphicsApi = GraphicsAPI::None;

    // Note: the context will *NOT* call AddRef on the D3D12 device to prevent memory leaks.
    void* d3d12Device = nullptr; // Valid when graphicsApi == D3D12

    void* vkInstance = nullptr; // These three are valid when graphicsApi == Vulkan
    void* vkPhysicalDevice = nullptr;
    void* vkDevice = nullptr;

    // Informs the library whether the graphics device supports DP4a instructions.
    // All modern GPUs support it, so the parameter is initialized to 'true'.
    // On Vulkan, check and enable VkPhysicalDeviceVulkan13Features::shaderIntegerDotProduct
    // On DX12, call ID3D12Device::CheckFeatureSupport with D3D12_FEATURE_SHADER_MODEL
    //   and check if the shader model is >= 6.4
    bool graphicsDeviceSupportsDP4a = true;

    // Informs the library whether the graphics device supports FP16 instructions.
    // All modern GPUs support it, so the parameter is initialized to 'true'.
    // On Vulkan, check and enable VkPhysicalDeviceVulkan12Features::shaderFloat16
    // On DX12, call ID3D12Device::CheckFeatureSupport with D3D12_FEATURE_D3D12_OPTIONS4
    //   and check the Native16BitShaderOpsSupported field.
    bool graphicsDeviceSupportsFloat16 = true;

    // Controls whether cooperative vector based decompression using Int8 math should be attempted, if supported.
    // Also controls whether the CoopVecInt8 weight types are available for texture sets.
    bool enableCooperativeVectorInt8 = true;

    // Controls whether cooperative vector based decompression using FP8 math should be attempted, if supported.
    // Also controls whether the CoopVecFP8 weight types are available for texture sets.
    bool enableCooperativeVectorFP8 = true;
};

extern "C"
{
// *** Context management ***

// Returns the value of ntc::InterfaceVersion that the library was built with.
NTC_API uint32_t GetInterfaceVersion();

// Returns a structure with LibNTC version info.
NTC_API VersionInfo GetLibraryVersion();

// Creates a context for all NTC operations.
// When the context is fully initialized,
//      returns Ok.
// When CUDA is unavailable or device is incompatible with NTC operations,
//      returns CudaUnavailable, and TextureSet objects cannot be created.
// When the library interface version doesn't match the version provided in params.interfaceVersion,
//      returns InterfaceMismatch.
NTC_API Status CreateContext(IContext** pOutContext, ContextParameters const& params);

// Releases all resources owned by the context.
NTC_API void DestroyContext(IContext* context);

// Returns a pointer to the buffer with the error message associated with the latest function call *on this thread*.
// The result is never NULL but it can be an empty string.
NTC_API const char* GetLastErrorMessage();

// Provides a textual representation of the status code.
// If the code is unknown, it is printed as a number into a temporary static string.
NTC_API const char* StatusToString(Status status);

// Provides a textual representation of the channel format.
NTC_API const char* ChannelFormatToString(ChannelFormat format);

// Provides a textual representation of the BCn format.
NTC_API const char* BlockCompressedFormatToString(BlockCompressedFormat format);

// Provides a textual representation of the color space.
NTC_API const char* ColorSpaceToString(ColorSpace colorSpace);

// Provides a textual representation of the network version (NTC_NETWORK_... constants).
NTC_API const char* NetworkVersionToString(int networkVersion);

// Provides a textual representation of the weight type.
NTC_API const char* InferenceWeightTypeToString(InferenceWeightType weightType);

// Returns the size of one color component in a pixel for the given format.
NTC_API size_t GetBytesPerPixelComponent(ChannelFormat format);

// Converts L2 loss from training or decompression passes into PSNR.
NTC_API float LossToPSNR(float loss);

// Calculates the size of the file that will be produced by compressing a texture set
// with the provided description. The estimate does not include the file headers (typically less than 1 KB).
NTC_API Status EstimateCompressedTextureSetSize(TextureSetDesc const& textureSetDesc,
    LatentShape const& latentShape, size_t& outSize);

// Returns the number of known latent shapes, to be used with EnumerateKnownLatentShapes(...)
NTC_API int GetKnownLatentShapeCount(int networkVersion);

// Returns one known latent shape by index. Known means it performs well for the given BPP value compared
// to other latent shapes that result in the same BPP value.
NTC_API Status EnumerateKnownLatentShapes(int index, int networkVersion, float& outBitsPerPixel, LatentShape& outShape);

// Selects a known-good latent shape for the provided bitsPerPixel value, with 25% tolerance.
// When no fitting latent shape can be found, returns Status::OutOfRange.
NTC_API Status PickLatentShape(float requestedBitsPerPixel, int networkVersion, float& outBitsPerPixel, LatentShape& outShape);

// Returns the bits-per-pixel value that corresponds to the provided latent shape.
NTC_API float GetLatentShapeBitsPerPixel(LatentShape const& shape);

// Converts the results placed in the output buffer by the IContext::MakeImageDifferenceComputePass pass
// into a regular double representing the MSE value.
NTC_API double DecodeImageDifferenceResult(uint64_t value);

} // extern "C"


}

#include "wrappers.h"
