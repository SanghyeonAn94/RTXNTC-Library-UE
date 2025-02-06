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

#include <cuda_fp16.h>

namespace tin {

    template <class T>
    constexpr T numeric_max() {
        if constexpr (std::is_same<T, half>::value) {
            return 65504.f;
        }
        else {
            return std::numeric_limits<T>::max();
        }
    }

    template<int C>
    constexpr int last_param() {
        return C;
    }

    template<int C, int... Cs>
    constexpr int last_param() {
        return last_param<Cs...>();
    }

    template<int idx, int C>
    constexpr int get_param() {
        return C;
    }

    template<int idx, int C, int... Cs>
    constexpr int get_param() {
        if constexpr(idx == 0) {
            return C;
        }
        return get_param<idx - 1, Cs...>();
    }

    template<int... Cs>
    constexpr int first_param() { return get_param<0, Cs...>(); }


    template<int C>
    constexpr int num_param() {
        return 1;
    }

    template<int C, int... Cs>
    constexpr int num_param() {
        return num_param<Cs...>() + 1;
    }

}