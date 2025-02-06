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
#include "tin_mma.h"
#include <iostream>
#include <iomanip>

namespace tin {

    namespace host {

        template <class T, bool ROW_MAJOR>
        class Matrix {
        public:
            TIN_DEVICE_HOST Matrix(uint32_t rows, uint32_t cols) : n_rows(rows), n_cols(cols)
            {
                mma_rows = n_rows / MMAMat::Rows;
                mma_cols = n_cols / MMAMat::Cols;
            }

            TIN_DEVICE_HOST uint32_t get_packed_offset(uint32_t row, uint32_t col)
            {
                const uint32_t NumPacked = num_packed<T>();
                uint32_t row_packed = ROW_MAJOR ? row : row / NumPacked;
                uint32_t col_packed = ROW_MAJOR ? col / NumPacked : col;

                uint32_t mma_row = get_mma_row(row_packed);
                uint32_t mma_col = get_mma_col(col_packed);
                uint32_t mma_idx = mma_rc_to_idx(mma_row, mma_col);

                uint32_t submma_row = get_submma_row(row_packed);
                uint32_t submma_col = get_submma_col(col_packed);

                uint32_t reg_row = MMAMat::get_reg_row(submma_row);
                uint32_t reg_col = MMAMat::get_reg_col(submma_col);
                uint32_t reg_idx = MMAMat::reg_rc_to_idx(reg_row, reg_col);

                uint32_t subreg_row = MMAMat::get_subreg_row(submma_row);
                uint32_t subreg_col = MMAMat::get_subreg_col(submma_col);

                uint32_t lane_idx = MMAReg::rc_to_lin_packed(subreg_row, subreg_col);
                uint32_t dw_offset = (WarpSize * mma_idx + lane_idx) * MMAMat::NumRegs + reg_idx;
                return dw_offset;
            }

        protected:
            uint32_t n_rows;
            uint32_t n_cols;
            uint32_t mma_rows;
            uint32_t mma_cols;

            using MMAMat = MMAMatrix<T, ROW_MAJOR>;
            using MMAReg = typename MMAMat::MMAReg;

            TIN_DEVICE_HOST static uint32_t get_mma_row(uint32_t row_packed) { return row_packed / MMAMat::RowsPacked; }
            TIN_DEVICE_HOST static uint32_t get_mma_col(uint32_t col_packed) { return col_packed / MMAMat::ColsPacked; }

            TIN_DEVICE_HOST static uint32_t get_submma_row(uint32_t row_packed) { return row_packed % MMAMat::RowsPacked; }
            TIN_DEVICE_HOST static uint32_t get_submma_col(uint32_t col_packed) { return col_packed % MMAMat::ColsPacked; }

            TIN_DEVICE_HOST uint32_t mma_rc_to_idx(uint32_t mma_row, uint32_t mma_col) {
                return mma_col * mma_rows + mma_row;
            }
        };

        template<bool ROW_MAJOR> using HMatrix = Matrix<half, ROW_MAJOR>;
        using HMatrixA = HMatrix<true>;
        using HMatrixB = HMatrix<false>;
    }
}
