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

#include "libntc/shaders/ColorSpaces.hlsli"
#include "libntc/shaders/DecompressConstants.h"
#include "libntc/shaders/Inference.hlsli"
#include "HashBasedRNG.hlsli"
#include "Vulkan.hlsli"

#ifdef __cplusplus
static const NtcDecompressConstants g_Const;
#else
VK_BINDING(0, 0) ConstantBuffer<NtcDecompressConstants> g_Const : register(b0);
#endif
VK_BINDING(1, 0) ByteAddressBuffer t_InputFile : register(t1);
VK_BINDING(2, 0) ByteAddressBuffer t_WeightBuffer : register(t2);
VK_BINDING(0, 1) RWTexture2D<float4> u_Outputs[] : register(u0);

typedef NtcNetworkParams<NETWORK_VERSION> Params;

static const int LATENTS_COUNT = Params::HR_FEATURES + Params::LR_FEATURES;

// Normally a margin of 2 latent pixels (one on each side) would be enough for bilinear filtering,
// but in some cases with odd resolutions, 3 is required.
static const int PRELOAD_MARGIN = 3;

static const int HR_LATENTS_WIDTH = DECOMPRESS_CS_BLOCK_WIDTH + PRELOAD_MARGIN;
static const int HR_LATENTS_HEIGHT = DECOMPRESS_CS_BLOCK_HEIGHT + PRELOAD_MARGIN;
static const int LR_LATENTS_WIDTH = DECOMPRESS_CS_BLOCK_WIDTH / 2 + PRELOAD_MARGIN;
static const int LR_LATENTS_HEIGHT = DECOMPRESS_CS_BLOCK_HEIGHT / 2 + PRELOAD_MARGIN;
static const int MAX_INPUT_SIZE = Params::INPUT_CHANNELS > Params::HIDDEN_LAYER_CHANNELS ? Params::INPUT_CHANNELS : Params::HIDDEN_LAYER_CHANNELS;
static const int MAX_OUTPUT_SIZE = Params::HIDDEN_LAYER_CHANNELS > Params::OUTPUT_CHANNELS ? Params::HIDDEN_LAYER_CHANNELS : Params::OUTPUT_CHANNELS;
