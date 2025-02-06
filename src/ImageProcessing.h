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

/* Internal texture data is stored in an unusual layout that is optimized for coalesced access
*  from the training and inference kernels. It's best viewed as a multidimensional array
*  with the following dimensions:
*
*  - Mip levels are the outermost dimension, all data for mip 1 immediately follows mip 0, and so on.
*  - Rows are the next dimension, row 1 follows row 0, and so on. Row pitch is (sizeof(half) * width * channels).
*  - Groups of 2 channels. Data for one row's channels 2-3 follows channels 0-1 for the same row.
*  - Pixels in the row.
*  - Channels 0 and 1 in the channel group for one pixel. These are stored as (half), so when a warp accesses 2 channels
*    for 32 consecutive pixels, the access is perfectly coalesced with 32 bits per thread.
*
*  Channel groups are generalized in the PitchLinearImageSlice structure as 'logChannelGroupSize' and 'channelGroupStride' members,
*  which allows us using the same code for accessing internal texture data and regular pitch linear images.
*/
struct PitchLinearImageSlice
{
    uint8_t* __restrict__ pData;
    int width;
    int height;
    int pixelStride;
    int rowPitch;
    int channels;
    int firstChannel;
    int logChannelGroupSize;
    int channelGroupStride;
    ChannelFormat format;
    
    // Color spaces for channels in the slice, i.e. [0] maps to firstChannel.
    ColorSpace channelColorSpaces[NTC_MAX_CHANNELS];

    // Use this constant for logChannelGroupSize when the image has simple layout without strided channels
    static constexpr int AllChannelsTogether = 5;
};

struct SurfaceInfo
{
    cudaSurfaceObject_t surface;
    int width;
    int height;
    int pixelStride;
    int channels;
    ChannelFormat format;
    ColorSpace rgbColorSpace;
    ColorSpace alphaColorSpace;
};

} // namespace ntc

namespace ntc::cuda
{

void ResizeMultichannelImage(
    PitchLinearImageSlice src,
    PitchLinearImageSlice dst,
    ColorSpace channelColorSpaces[NTC_MAX_CHANNELS]);

void CopyImage(
    PitchLinearImageSlice src,
    PitchLinearImageSlice dst,
    bool useDithering,
    bool verticalFlip);

void CopyImageToSurface(
    PitchLinearImageSlice src,
    SurfaceInfo dst,
    bool useDithering,
    bool verticalFlip);

void CopySurfaceToImage(
    SurfaceInfo src,
    PitchLinearImageSlice dst,
    bool verticalFlip);

cudaError_t ComputeMinMaxChannelValues(
    PitchLinearImageSlice image,
    // This should point to NTC_MAX_CHANNELS*2 int's in device memory
    int* scratchMemory,
    float outMinimums[NTC_MAX_CHANNELS],
    float outMaximums[NTC_MAX_CHANNELS]);

} // namespace ntc::cuda