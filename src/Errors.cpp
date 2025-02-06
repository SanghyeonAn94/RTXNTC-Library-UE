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

#include "Errors.h"
#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace ntc
{

static thread_local char stl_ErrorBuffer[1024];

void SetUnformattedErrorMessage(char const* message)
{
    strncpy(stl_ErrorBuffer, message, sizeof stl_ErrorBuffer - 1);
    stl_ErrorBuffer[sizeof stl_ErrorBuffer - 1] = 0;
}

void SetErrorMessage(char const* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vsnprintf(stl_ErrorBuffer, sizeof stl_ErrorBuffer, fmt, args);  // NOLINT(clang-diagnostic-format-nonliteral)
    va_end(args);
}

void ClearErrorMessage()
{
    stl_ErrorBuffer[0] = 0;
}

#if NTC_WITH_CUDA
void SetCudaErrorMessage(char const* functionName, cudaError_t err)
{
    SetErrorMessage("Call to %s failed, error code = %s.", functionName, cudaGetErrorName(err));
}
#endif

char const* GetErrorBuffer(void)
{
    return stl_ErrorBuffer;
}

}