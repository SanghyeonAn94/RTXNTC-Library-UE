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

#include <libntc/ntc.h>
#include <cstdio>

namespace ntc
{

class FileStream : public IStream
{
public:
    FileStream(FILE* file);
    ~FileStream() override;

    bool Read(void* dst, size_t size) override;
    bool Write(const void* src, size_t size) override;
    bool Seek(uint64_t offset) override;
    uint64_t Tell() override;
    uint64_t Size() override;

private:
    FILE* m_file;
};

class MemoryStream : public IStream
{
public:
    MemoryStream(uint8_t* data, size_t size, bool readonly);

    bool Read(void* dst, size_t size) override;
    bool Write(const void* src, size_t size) override;
    bool Seek(uint64_t offset) override;
    uint64_t Tell() override;
    uint64_t Size() override;

private:
    uint8_t* m_data;
    size_t m_size;
    size_t m_ptr;
    bool m_readonly;
};

}