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
#include <cuda_runtime.h>
#include <stdexcept>

namespace tin {

    inline void check_cuda_error(cudaError_t err, const std::string& description="") {
        if (err != cudaSuccess)
            throw std::runtime_error{ std::string(cudaGetErrorString(err)) + ": " + description };
    }

    inline void throw_error(const std::string& description) {
        throw std::runtime_error{ description };
    }
}