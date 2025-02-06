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

namespace ntc
{

class Context;

inline int Log2i(uint32_t u)
{
    for (int i = 0; i < 32; ++i)
        if (u <= (1u << i))
            return i;
    return -1;
}

inline bool IsPowerOf2(uint32_t u)
{
    return (u & (u - 1)) == 0;
}

// Rounds the input value up to the nearest multiple of 4.
template<typename T>
inline T RoundUp4(T x)
{
    return (x + 3) & ~T(3);
}

// Shifts the input value right by a given number of bits, rounding up.
template<typename T>
inline T ShiftRightRoundUp(T value, int shift)
{
    return (value + (T(1) << shift) - T(1)) >> shift;
}


}
