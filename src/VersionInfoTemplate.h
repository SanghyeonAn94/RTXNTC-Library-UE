/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

// The VersionInfoTemplate.h file is used as a template to generate the real VersionInfo.h by CMake
// using configure_file - see src/CMakeLists.txt

#define NTC_VERSION_MAJOR @NTC_VERSION_MAJOR@
#define NTC_VERSION_MINOR @NTC_VERSION_MINOR@
#define NTC_VERSION_POINT @NTC_VERSION_POINT@
#define NTC_VERSION_BRANCH "@NTC_VERSION_BRANCH@"
#define NTC_VERSION_HASH "@NTC_VERSION_HASH@"
#define NTC_VERSION_FULL "@NTC_VERSION_MAJOR@.@NTC_VERSION_MINOR@.@NTC_VERSION_POINT@ @NTC_VERSION_BRANCH@-@NTC_VERSION_HASH@"