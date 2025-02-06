/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef NTC_VULKAN_HLSLI
#define NTC_VULKAN_HLSLI

#ifdef SPIRV
#define VK_BINDING(reg,dset) [[vk::binding(reg,dset)]]
#else
#define VK_BINDING(reg,dset) 
#endif

#endif // NTC_VULKAN_HLSLI