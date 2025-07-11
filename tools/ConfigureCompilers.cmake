# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

function(LibNTC_ConfigureShaderCompilerVersions)
    option(SHADERMAKE_FIND_COMPILERS "" ON)
    option(SHADERMAKE_FIND_SLANG "" ON)
    option(SHADERMAKE_FIND_DXC "" ON)
    option(SHADERMAKE_FIND_DXC_VK "" OFF)
    set(SHADERMAKE_DXC_VERSION "v1.8.2505" CACHE STRING "")
    set(SHADERMAKE_DXC_DATE "2025_05_24" CACHE STRING "")
    set(SHADERMAKE_SLANG_VERSION "2025.12.1" CACHE STRING "")
endfunction()

function(LibNTC_CopyDXCompilerDLLs)
    # Copy the DXC DLLs next to Slang because the -dxc-path option doesn't seem to work.
    if (WIN32 AND EXISTS "${SHADERMAKE_SLANG_PATH}" AND EXISTS "${SHADERMAKE_DXC_PATH}")
        cmake_path(REMOVE_FILENAME SHADERMAKE_SLANG_PATH OUTPUT_VARIABLE SLANG_DIRECTORY)
        cmake_path(REPLACE_FILENAME SHADERMAKE_DXC_PATH "dxcompiler.dll" OUTPUT_VARIABLE DXCOMPILER_DLL)
        cmake_path(REPLACE_FILENAME SHADERMAKE_DXC_PATH "dxil.dll" OUTPUT_VARIABLE DXIL_DLL)
        file(COPY "${DXCOMPILER_DLL}" "${DXIL_DLL}" DESTINATION "${SLANG_DIRECTORY}/")
    endif()
endfunction()