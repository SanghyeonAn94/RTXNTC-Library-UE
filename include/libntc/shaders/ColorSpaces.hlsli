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

 #ifndef NTC_COLOR_SPACES_HLSLI
 #define NTC_COLOR_SPACES_HLSLI

// These constants must match the values of ntc::ColorSpace in ntc.h
static const uint NtcColorSpace_Linear = 0;
static const uint NtcColorSpace_sRGB = 1;
static const uint NtcColorSpace_HLG = 2;


// Hybrid Log-Gamma (HLG) encoding and decoding functions.
// https://en.wikipedia.org/wiki/Hybrid_log%E2%80%93gamma
struct NtcHybridLogGammaColorSpace
{
    static float Encode(float e)
    {
        const float a = 0.17883277f;
        const float b = 0.28466892f;
        const float c = 0.55991073f;

        float sign = (e < 0.f) ? -1.f : 1.f;
        e = abs(e);
        return (e < 1.f)
            ? 0.5f * sqrt(e) * sign
            : (a * log(e - b) + c) * sign;
    }

    static float Decode(float h)
    {
        const float a = 0.17883277f;
        const float b = 0.28466892f;
        const float c = 0.55991073f;
        
        float sign = (h < 0.f) ? -1.f : 1.f;
        h = abs(h);
        return (h < 0.5f)
            ? 4.f * h * h * sign
            : (exp((h - c) / a) + b) * sign;
    }

    static float3 Encode(float3 e)
    {
        return float3(Encode(e.x), Encode(e.y), Encode(e.z));
    }

    static float3 Decode(float3 h)
    {
        return float3(Decode(h.x), Decode(h.y), Decode(h.z));
    }

    static float4 Encode(float4 e)
    {
        return float4(Encode(e.x), Encode(e.y), Encode(e.z), Encode(e.w));
    }

    static float4 Decode(float4 h)
    {
        return float4(Decode(h.x), Decode(h.y), Decode(h.z), Decode(h.w));
    }
};


// sRGB encoding and decoding functions.
struct NtcSrgbColorSpace
{
    static float Encode(float lin)
    {
        return (lin <= 0.0031308f)
            ? lin * 12.92f
            : 1.055f * pow(lin, 1.0f / 2.4f) - 0.055f;
    }

    static float Decode(float encoded)
    {
        return (encoded <= 0.04045f)
            ? encoded / 12.92f
            : pow((encoded + 0.055f) / 1.055f, 2.4f);
    }

    static float3 Encode(float3 e)
    {
        return float3(Encode(e.x), Encode(e.y), Encode(e.z));
    }

    static float4 Encode(float4 e)
    {
        return float4(Encode(e.x), Encode(e.y), Encode(e.z), Encode(e.w));
    }

    static float3 Decode(float3 h)
    {
        return float3(Decode(h.x), Decode(h.y), Decode(h.z));
    }

    static float4 Decode(float4 h)
    {
        return float4(Decode(h.x), Decode(h.y), Decode(h.z), Decode(h.w));
    }
};

float NtcConvertColorSpace(float texelValue, int srcColorSpace, int dstColorSpace)
{
    if (srcColorSpace == dstColorSpace)
        return texelValue;
    
    switch (srcColorSpace)
    {
        case NtcColorSpace_sRGB:
            texelValue = NtcSrgbColorSpace::Decode(texelValue);
            break;
        case NtcColorSpace_HLG:
            texelValue = NtcHybridLogGammaColorSpace::Decode(texelValue);
            break;
    }

    switch (dstColorSpace)
    {
        case NtcColorSpace_sRGB:
            texelValue = NtcSrgbColorSpace::Encode(texelValue);
            break;
        case NtcColorSpace_HLG:
            texelValue = NtcHybridLogGammaColorSpace::Encode(texelValue);
            break;
    }

    return texelValue;
}

float3 NtcConvertColorSpace(float3 texelValue, int srcColorSpace, int dstColorSpace)
{
    if (srcColorSpace == dstColorSpace)
        return texelValue;
    
    switch (srcColorSpace)
    {
        case NtcColorSpace_sRGB:
            texelValue = NtcSrgbColorSpace::Decode(texelValue);
            break;
        case NtcColorSpace_HLG:
            texelValue = NtcHybridLogGammaColorSpace::Decode(texelValue);
            break;
    }

    switch (dstColorSpace)
    {
        case NtcColorSpace_sRGB:
            texelValue = NtcSrgbColorSpace::Encode(texelValue);
            break;
        case NtcColorSpace_HLG:
            texelValue = NtcHybridLogGammaColorSpace::Encode(texelValue);
            break;
    }

    return texelValue;
}

#endif