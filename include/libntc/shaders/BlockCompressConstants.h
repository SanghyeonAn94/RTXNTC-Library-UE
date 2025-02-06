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

#ifndef BLOCK_COMPRESS_CONSTANTS_H
#define BLOCK_COMPRESS_CONSTANTS_H

#define BLOCK_COMPRESS_CS_ST_GROUP_WIDTH 16
#define BLOCK_COMPRESS_CS_ST_GROUP_HEIGHT 8

#define BLOCK_COMPRESS_MODE_MASK_UINTS 16

struct NtcBlockCompressConstants
{
    int srcLeft;
    int srcTop;
    int dstOffsetX;
    int dstOffsetY;

    int widthInBlocks;
    int heightInBlocks;
    float alphaThreshold;
    int padding;

#ifdef __cplusplus
    uint32_t allowedModes[BLOCK_COMPRESS_MODE_MASK_UINTS];
#else
    uint4 allowedModes[BLOCK_COMPRESS_MODE_MASK_UINTS / 4];
#endif
};

#endif
