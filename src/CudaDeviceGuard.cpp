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

#include "CudaDeviceGuard.h"
#include "Context.h"
#include "Errors.h"
#include <cuda_runtime_api.h>

namespace ntc
{

CudaDeviceGuard::CudaDeviceGuard(Context const* context)
{
    int device = context->GetCudaDevice();
    if (device >= 0)
    {
        int originalDevice = -1;
        cudaError_t err = cudaGetDevice(&originalDevice);
        if (err != cudaSuccess)
        {
            SetCudaErrorMessage("cudaGetDevice", err);
            return;
        }

        if (originalDevice == device)
        {
            // Current device is the requested one: do nothing and don't need to restore it later
            m_success = true;
            return;
        }

        // Remember the original device ID to restore in the destructor
        m_originalDevice = originalDevice;
    
        err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            SetCudaErrorMessage("cudaSetDevice", err);
            return;
        }
        
        m_success = true;
    }
}

CudaDeviceGuard::~CudaDeviceGuard()
{
    if (m_originalDevice >= 0)
    {
        cudaSetDevice(m_originalDevice);
    }
}

}