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

#ifndef IMAGE_DIFFERENCE_CONSTANTS_H
#define IMAGE_DIFFERENCE_CONSTANTS_H

#define IMAGE_DIFFERENCE_CS_BLOCK_WIDTH 16
#define IMAGE_DIFFERENCE_CS_BLOCK_HEIGHT 16
#define IMAGE_DIFFERENCE_CS_PIXELS_PER_THREAD_X 4
#define IMAGE_DIFFERENCE_CS_PIXELS_PER_THREAD_Y 4
#define IMAGE_DIFFERENCE_CS_PIXELS_PER_BLOCK_X (IMAGE_DIFFERENCE_CS_BLOCK_WIDTH * IMAGE_DIFFERENCE_CS_PIXELS_PER_THREAD_X)
#define IMAGE_DIFFERENCE_CS_PIXELS_PER_BLOCK_Y (IMAGE_DIFFERENCE_CS_BLOCK_HEIGHT * IMAGE_DIFFERENCE_CS_PIXELS_PER_THREAD_Y)

struct NtcImageDifferenceConstants
{
    int left;
    int top;
    int width;
    int height;

    float alphaThreshold;
    int useAlphaThreshold;
    int useMSLE;
    uint32_t outputOffset;
};

#endif