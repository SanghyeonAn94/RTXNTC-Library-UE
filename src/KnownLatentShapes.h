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

#include <libntc/ntc.h>

namespace ntc
{
    
struct KnownLatentShape
{
    float bitsPerPixel = 0.f;
    LatentShape shapes[NTC_NETWORK_COUNT];
};

constexpr int KnownLatentShapeCount = 26;
extern KnownLatentShape const g_KnownLatentShapes[KnownLatentShapeCount];

}