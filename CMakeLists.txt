# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

cmake_minimum_required(VERSION 3.18)
include(FetchContent)

set(LANGUAGES CXX)

option(NTC_BUILD_SHARED "Build LibNTC as a shared library (DLL or .so)" ON)
option(NTC_WITH_CUDA "Compile NTC with CUDA support" ON)
option(NTC_WITH_DX12 "Compile NTC with DX12 support" "${WIN32}")
option(NTC_WITH_VULKAN "Compile NTC with Vulkan support" ON)
option(NTC_WITH_PREBUILT_SHADERS "Compile NTC with pre-built shaders for decompression etc." ON)

if (NTC_WITH_CUDA)
    list(APPEND LANGUAGES CUDA)
endif()

project(
    libntc
    DESCRIPTION "Neural Texture Compression Library"
    LANGUAGES ${LANGUAGES}
)

# Version info

set(NTC_VERSION_MAJOR 0)
set(NTC_VERSION_MINOR 5)
set(NTC_VERSION_POINT 0)

# Read Git branch name into NTC_VERSION_BRANCH
execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE NTC_VERSION_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Read Git short hash into NTC_VERSION_HASH
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE NTC_VERSION_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "LibNTC version: ${NTC_VERSION_MAJOR}.${NTC_VERSION_MINOR}.${NTC_VERSION_POINT} "
                "${NTC_VERSION_BRANCH}-${NTC_VERSION_HASH}")


if ("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
    set (LIBNTC_IS_SUBMODULE OFF)
    set (LIBNTC_BIN_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
else()
    set (LIBNTC_IS_SUBMODULE ON)
endif()

# Include Vulkan-Headers if it hasn't been included before
if (NOT TARGET Vulkan-Headers AND NOT TARGET Vulkan::Headers AND NTC_WITH_VULKAN)
    FetchContent_Declare(
        Vulkan-Headers
        GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
        GIT_TAG v1.4.307
        SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/Vulkan-Headers"
    )
    FetchContent_MakeAvailable(Vulkan-Headers)
endif()

# Include ShaderMake if it hasn't been included before and if it's needed
if (NOT TARGET ShaderMake AND NTC_WITH_PREBUILT_SHADERS AND (NTC_WITH_DX12 OR NTC_WITH_VULKAN))
    set(SHADERMAKE_SEARCH_FOR_COMPILERS OFF CACHE BOOL "")
    set(SHADERMAKE_BIN_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/ShaderMake" CACHE STRING "")
    FetchContent_Declare(
        ShaderMake
        GIT_REPOSITORY https://github.com/NVIDIAGameWorks/ShaderMake.git
        GIT_TAG 94406cff724a967b9e8bea308367e14e54a6d0da
        SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/ShaderMake"
    )
    FetchContent_MakeAvailable(ShaderMake)
endif()

# Include NVAPI if it hasn't been included before and if we're building for Windows with DX12 support.
# If the upper level project provides its own copy of NVAPI, make sure it's declared as a CMake target 'nvapi'
# and it's using NVAPI SDK R570 or newer.
if (NOT TARGET nvapi AND WIN32 AND NTC_WITH_DX12)
    set(NTC_NVAPI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/nvapi")

    FetchContent_Declare(
        nvapi
        GIT_REPOSITORY https://github.com/NVIDIA/nvapi.git
        GIT_TAG d08488fcc82eef313b0464db37d2955709691e94
        SOURCE_DIR "${NTC_NVAPI_DIR}"
    )
    FetchContent_MakeAvailable(nvapi)
    
    add_library(nvapi STATIC IMPORTED GLOBAL)
    target_include_directories(nvapi INTERFACE "${NTC_NVAPI_DIR}")
    set_property(TARGET nvapi PROPERTY IMPORTED_LOCATION "${NTC_NVAPI_DIR}/amd64/nvapi64.lib")
endif()

if (MSVC AND NOT LIBNTC_IS_SUBMODULE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /MP")
endif()

add_subdirectory(external/rapidjson)
add_subdirectory(src)
