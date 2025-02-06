/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

// This file is intended to test whether the CoopVec inference header compiles with DXC (not Slang).
// It does not contain any useful functionality.

#include "libntc/shaders/InferenceCoopVec.hlsli"

#define NETWORK_VERSION NTC_NETWORK_MEDIUM

typedef NtcNetworkParams<NETWORK_VERSION> NtcParams;

ConstantBuffer<NtcTextureSetConstants> g_NtcMaterial : register(b0);
ByteAddressBuffer t_InputFile : register(t0);
ByteAddressBuffer t_WeightBuffer : register(t1);
RWTexture2D<float4> u_Output : register(u0);

[numthreads(1,1,1)]
void main()
{
    float channels[NtcParams::OUTPUT_CHANNELS];

    #if USE_FP8
        NtcSampleTextureSet_CoopVec_FP8<NETWORK_VERSION>(g_NtcMaterial, t_InputFile, 0,
            t_WeightBuffer, 0, 0, 0, true, channels);
    #else
        NtcSampleTextureSet_CoopVec_Int8<NETWORK_VERSION>(g_NtcMaterial, t_InputFile, 0,
            t_WeightBuffer, 0, 0, 0, true, channels);
    #endif

    u_Output[int2(0,0)] = float4(channels[0], channels[1], channels[2], channels[3]);
}