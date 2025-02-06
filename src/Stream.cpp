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

#include "Stream.h"
#include <cstring>

#ifdef _WIN32
#define fseeko64 _fseeki64
#define ftello64 _ftelli64
#endif

namespace ntc
{

FileStream::FileStream(FILE* file)
    : m_file(file)
{ }

FileStream::~FileStream()
{
    if (!m_file)
        return;

    fclose(m_file);
    m_file = nullptr;
}

bool FileStream::Read(void* dst, size_t size)
{
    return fread(dst, size, 1, m_file) == 1;
}

bool FileStream::Write(const void* src, size_t size)
{
    return fwrite(src, size, 1, m_file) == 1;
}

bool FileStream::Seek(uint64_t offset)
{
    return fseeko64(m_file, offset, SEEK_SET) == 0;
}

uint64_t FileStream::Tell()
{
    return ftello64(m_file);
}

uint64_t FileStream::Size()
{
    uint64_t current = ftello64(m_file);
    fseeko64(m_file, 0, SEEK_END);
    uint64_t size = ftello64(m_file);
    fseeko64(m_file, current, SEEK_SET);
    return size;
}

MemoryStream::MemoryStream(uint8_t* data, size_t size, bool readonly)
    : m_data(data)
    , m_size(size)
    , m_ptr(0)
    , m_readonly(readonly)
{ }

bool MemoryStream::Read(void* dst, size_t size)
{
    if (!m_data)
        return false;

    if (size + m_ptr > m_size)
        return false;

    memcpy(dst, m_data + m_ptr, size);
    m_ptr += size;

    return true;
}

bool MemoryStream::Write(const void* src, size_t size)
{
    if (m_readonly)
        return false;

    if (!m_data)
        return false;

    if (size + m_ptr > m_size)
        return false;

    memcpy(m_data + m_ptr, src, size);
    m_ptr += size;

    return true;
}

bool MemoryStream::Seek(uint64_t offset)
{
    if (offset >= m_size)
        return false;

    m_ptr = offset;

    return true;
}

uint64_t MemoryStream::Tell()
{
    return m_ptr;
}

uint64_t MemoryStream::Size()
{
    return m_size;
}

}
