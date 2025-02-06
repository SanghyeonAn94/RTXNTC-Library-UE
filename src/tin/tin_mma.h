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

#include <cuda.h>

#include <mma.h>
#include "tin_ptx.h"
#include "tin_common.h"

namespace tin {

    // MMAMatrix is a warp wide storage for an 16x8 dword matrix (row major) or a 8x16 dword matrix (col major).
    // For 16-bit datatypes MMAMatrix size is 16x16 i.e. 16x8*2 (row-major) and 8*2x16 (col major).
    // For 8-bit datatypes it is 16x32 (row major) or 32x16 matrix (col major).
    // MMAMatrix operations are mapped to tensor core intrinsics
    template <class T, bool ROW_MAJOR>
    struct alignas(16) MMAMatrix {

        // MMAReg is a single warp-wide register. It stores an 8x4 dword matrix (row major) or a 4x8 dword matrix (col major).
        // For 16-bit datatypes, it has 8x8 elements i.e. 8x4*2 (row major) or a 4*2x8 (col major).
        // For 8-bit datatypes, it has 8x4*2 = 8x32 elements (row major) or 32x8 elements (col major).
        struct MMAReg {

            TIN_DEVICE_HOST static uint32_t rc_to_lin_packed(uint32_t row_packed, uint32_t col_packed) {
                if (ROW_MAJOR) {
                    return row_packed * ColsPacked + col_packed;
                }
                else {
                    return col_packed * RowsPacked + row_packed;
                }
            }

            TIN_DEVICE_HOST static uint2 lin_to_rc_packed(uint32_t idx) {
                uint2 rc_packed;
                if (ROW_MAJOR) {
                    rc_packed.x = idx / ColsPacked;
                    rc_packed.y = idx % ColsPacked;
                }
                else {
                    rc_packed.y = idx / RowsPacked;
                    rc_packed.x = idx % RowsPacked;
                }
                return rc_packed;
            }

            static const uint32_t NumOuter = 8U;
            static const uint32_t InnerPacked = WarpSize / NumOuter;
            static const uint32_t RowsPacked = ROW_MAJOR ? NumOuter : InnerPacked;
            static const uint32_t ColsPacked = ROW_MAJOR ? InnerPacked : NumOuter;

            static const uint32_t NumInner = InnerPacked * num_packed<T>();
            static const uint32_t NumElements = NumInner * NumOuter;
            static const uint32_t Rows = ROW_MAJOR ? NumOuter : NumInner;
            static const uint32_t Cols = ROW_MAJOR ? NumInner : NumOuter;

            uint32_t m_reg;
        };

#if defined(__CUDACC__)

        TIN_DEVICE void clear() {
TIN_UNROLL
            for (uint32_t i = 0; i < RegCols; i++) {
TIN_UNROLL
                for (uint32_t j = 0; j < RegRows; j++) {
                    uint32_t reg_idx = reg_rc_to_idx(j, i);
                    m_mma_reg[reg_idx].m_reg = 0U;
                }
            }
        }

        TIN_DEVICE MMAMatrix<T, !ROW_MAJOR> transpose() const {
            MMAMatrix<T, !ROW_MAJOR> r;

TIN_UNROLL
            for (uint32_t i = 0; i < RegCols; i++) {
TIN_UNROLL
                for (uint32_t j = 0; j < RegRows; j++) {
                    uint32_t src_reg_idx = reg_rc_to_idx(j, i);
                    uint32_t dst_reg_idx = r.reg_rc_to_idx(i, j);
                    r.m_mma_reg[dst_reg_idx].m_reg = m_mma_reg[src_reg_idx].m_reg;
                }
            }
            return r;
        }

#endif

        static const uint32_t RegRows = 2;
        static const uint32_t RegCols = 2;
        static const uint32_t NumRegs = RegRows * RegCols;

        static const uint32_t Rows = MMAReg::Rows * RegRows;
        static const uint32_t Cols = MMAReg::Cols * RegCols;
        static const uint32_t RowsPacked = MMAReg::RowsPacked * RegRows;
        static const uint32_t ColsPacked = MMAReg::ColsPacked * RegCols;

        MMAReg m_mma_reg[NumRegs];

        static TIN_DEVICE_HOST uint32_t get_reg_row(uint32_t row_packed) { return row_packed / MMAReg::RowsPacked; }
        static TIN_DEVICE_HOST uint32_t get_reg_col(uint32_t col_packed) { return col_packed / MMAReg::ColsPacked; }

        static TIN_DEVICE_HOST uint32_t get_subreg_row(uint32_t row_packed) { return row_packed % MMAReg::RowsPacked; }
        static TIN_DEVICE_HOST uint32_t get_subreg_col(uint32_t col_packed) { return col_packed % MMAReg::ColsPacked; }

        TIN_DEVICE_HOST static uint32_t reg_rc_to_idx(uint32_t reg_row, uint32_t reg_col) {
            return reg_col * RegRows + reg_row;
        }
    };

    template <class T> using MMAMatrixA = MMAMatrix<T, true >;
    template <class T> using MMAMatrixB = MMAMatrix<T, false>;

#if defined(__CUDACC__)
    template <class T> using MMAFragAcc = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, MMAMatrixA<T>::Rows, MMAMatrixA<T>::Cols, MMAMatrixB<T>::Cols, T>;
    template <class T> using MMAFragA   = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a   , MMAMatrixA<T>::Rows, MMAMatrixA<T>::Cols, MMAMatrixB<T>::Cols, T, nvcuda::wmma::row_major>;
    template <class T> using MMAFragB   = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b   , MMAMatrixA<T>::Rows, MMAMatrixA<T>::Cols, MMAMatrixB<T>::Cols, T, nvcuda::wmma::col_major>;

    template <bool ROW_MAJOR>
    TIN_DEVICE MMAMatrix<half, !ROW_MAJOR>
    change_major_axis(
        const MMAMatrix<half, ROW_MAJOR>& x) {

        MMAMatrix<half, !ROW_MAJOR> r;

        using MMAMat = MMAMatrix<half, ROW_MAJOR>;
        const uint32_t RegRows = MMAMat::RegRows;
        const uint32_t RegCols = MMAMat::RegCols;

TIN_UNROLL
        for (uint32_t i = 0; i < RegCols; i++) {
TIN_UNROLL
            for (uint32_t j = 0; j < RegRows; j++) {
                uint32_t reg_idx = x.reg_rc_to_idx(j, i);
                r.m_mma_reg[reg_idx].m_reg = _transpose(x.m_mma_reg[reg_idx].m_reg);
            }
        }

        return r;
    };

    template <class T>
    TIN_DEVICE MMAMatrixA<T>
    mad(const MMAMatrixA<T>& a, const MMAMatrixB<T>& b, const MMAMatrixA<T>& c) {
        MMAFragAcc<T> r;
        MMAFragA  <T> am = *((MMAFragA  <T>*)(&a));
        MMAFragB  <T> bm = *((MMAFragB  <T>*)(&b));
        MMAFragAcc<T> cm = *((MMAFragAcc<T>*)(&c));

        nvcuda::wmma::mma_sync(r, am, bm, cm);
        return *((MMAMatrixA<T>*)(&r));
    };

    template <class T, bool ROW_MAJOR>
    TIN_DEVICE MMAMatrix<T, ROW_MAJOR> operator +(const MMAMatrix<T, ROW_MAJOR>& a, const MMAMatrix<T, ROW_MAJOR>& b) {
        MMAMatrix<T, ROW_MAJOR> r;

        const uint32_t RegRows = r.RegRows;
        const uint32_t RegCols = r.RegCols;

TIN_UNROLL
        for (uint32_t i = 0; i < RegCols; i++) {
TIN_UNROLL
            for (uint32_t j = 0; j < RegRows; j++) {
                uint32_t reg_idx = r.reg_rc_to_idx(j, i);

                const auto& x = *((PackedType<T>*) & (a.m_mma_reg[reg_idx].m_reg));
                const auto& y = *((PackedType<T>*) & (b.m_mma_reg[reg_idx].m_reg));
                auto& z = *((PackedType<T>*) & (r.m_mma_reg[reg_idx].m_reg));
                z = x + y;
            }
        }

        return r;
    };

    TIN_DEVICE inline void reduce_sum(const MMAMatrixA<half>& x, half* dest) {

        auto dest_v = (half2*)dest;

        const uint32_t RegCols = x.RegCols;
        const uint32_t RegRows = x.RegRows;
        using MMAReg = MMAMatrixA<half>::MMAReg;

        uint32_t lane_id = _lane_id();
        uint2 subreg_rc = MMAReg::lin_to_rc_packed(lane_id);

        // sum row registers
        MMAMatrixA<half> r_mat = x;

TIN_UNROLL
        for (uint32_t reg_col = 0; reg_col < RegCols; reg_col++)
        {
TIN_UNROLL
            for (uint32_t j = 2; j <= RegRows; j <<= 1)
            {
TIN_UNROLL
                for (uint32_t reg_row = 0; reg_row < RegRows; reg_row += j)
                {
                    uint32_t src_reg_idx = x.reg_rc_to_idx(reg_row + j / 2, reg_col);
                    uint32_t dst_reg_idx = x.reg_rc_to_idx(reg_row, reg_col);

                    auto& src_reg = *((half2*) & (r_mat.m_mma_reg[src_reg_idx]));
                    auto& dst_reg = *((half2*) & (r_mat.m_mma_reg[dst_reg_idx]));
                    dst_reg += src_reg;
                }
            }

            // Binary reduce rows inside a register
            uint32_t reg_idx = x.reg_rc_to_idx(0, reg_col);
            auto& reg = *((half2*) & (r_mat.m_mma_reg[reg_idx]));

TIN_UNROLL
            for (uint32_t j = 1; j < MMAReg::RowsPacked; j <<= 1)
            {
                half2 regs_shfl = __shfl_down_sync(0xFFFFFFFF, reg, MMAReg::ColsPacked * j);
                reg += regs_shfl;
            }

            // Write to memory
            uint32_t col = reg_col * MMAReg::ColsPacked + subreg_rc.y;

            if (subreg_rc.x == 0)
            {
                dest_v[col] = reg;
            }
        }

    }

#endif
}
