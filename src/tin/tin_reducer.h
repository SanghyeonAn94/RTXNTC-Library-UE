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

#include "tin_common.h"
#include "tin_utils.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace tin {

    template<typename T, int WARPS_PER_WG>
    class Reducer {
    public:
        static constexpr size_t sharedmem_size() {
            return WARPS_PER_WG > 1 ? WARPS_PER_WG : 0;
        }

        template<typename Func>
        static __device__ T reduce(T* mem, T data) {
            namespace cg = cooperative_groups;
            auto g = cg::this_thread_block();
            auto tile32 = cg::tiled_partition<WarpSize>(g);
            int warp_id = tile32.meta_group_rank();
            int thread_in_warp = tile32.thread_rank();

            static_assert((WARPS_PER_WG & (WARPS_PER_WG - 1)) == 0, "WARPS_PER_WG must be a power of 2");
            static_assert(WARPS_PER_WG <= 32, "WARPS_PER_WG must be <= 32");

            auto f = Func();

            // Horizontal reduction within each warp
            data = cg::reduce(tile32, data, f);

            if constexpr (WARPS_PER_WG == 1)
                return data;
            
            // Transpose the reduction results - thread 0 from each warps stores into a unique smem location
            if (thread_in_warp == 0)
                mem[warp_id] = data;

            g.sync();

            // Finish the transpose and do vertical reduction.
            // This block is only executed by WARPS_PER_WG threads.
            auto tilew = cg::tiled_partition<WARPS_PER_WG>(g);
            if (tilew.meta_group_rank() == 0)
            {
                // Each thread loads the result of previous reduction from another warp.
                data = mem[tilew.thread_rank()];
                
                // Reduce within the small group of WARPS_PER_WG threads.
                data = cg::reduce(tilew, data, f);

                // Store the reduction results into smem location 0 to distribute it to every thread later
                if (tilew.thread_rank() == 0)
                    mem[0] = data;
            }
            g.sync();

            return mem[0];
        }
        
        static __device__ T sum(T* mem, T data) {
            return reduce<cooperative_groups::plus<T>>(mem, data);
        }

        static __device__ T min(T* mem, T data) {
            return reduce<cooperative_groups::less<T>>(mem, data);
        }

        static __device__ T max(T* mem, T data) {
            return reduce<cooperative_groups::greater<T>>(mem, data);
        }
    };


}