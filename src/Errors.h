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

#pragma once

#if NTC_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace ntc
{

void SetUnformattedErrorMessage(const char* message);
void SetErrorMessage(const char* fmt...);
#if NTC_WITH_CUDA
void SetCudaErrorMessage(const char* functionName, cudaError_t err);
#endif
void ClearErrorMessage();
char const* GetErrorBuffer();

}