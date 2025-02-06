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

namespace ntc::cuda
{

template <int MAX_NUM_FEATURES, bool ALL_CORNERS>
class FeatureGrid
{
public:
    __device__ FeatureGrid(int numFeatures, int width, int height)
        : m_latentWidth(width)
        , m_latentHeight(height)
        , m_numFeatures(numFeatures)
    { }

    template<int INPUT_CHANNELS>
    __device__ void Sample(float u, float v, int offset, const half* data, tin::HArray<INPUT_CHANNELS>& outputArray) {

        float x = u * m_latentWidth  - 0.5f;
        float y = v * m_latentHeight - 0.5f;

        int x_b = floor(x);
        int y_b = floor(y);

        float dx = x - x_b;
        float dy = y - y_b;
        float dxn = 1 - dx;
        float dyn = 1 - dy;

        half2 w00 = __float2half2_rn(dxn * dyn);
        half2 w01 = __float2half2_rn(dx * dyn);
        half2 w10 = __float2half2_rn(dxn * dy);
        half2 w11 = __float2half2_rn(dx * dy);

        int x0 = std::max(0, std::min(m_latentWidth - 1, x_b));
        int y0 = std::max(0, std::min(m_latentHeight - 1, y_b));
        int x1 = std::max(0, std::min(m_latentWidth - 1, x_b + 1));
        int y1 = std::max(0, std::min(m_latentHeight - 1, y_b + 1));

        int a00 = (y0 * m_latentWidth + x0);
        int a01 = (y0 * m_latentWidth + x1);
        int a10 = (y1 * m_latentWidth + x0);
        int a11 = (y1 * m_latentWidth + x1);

        const half2* l2 = (half2*)data;

#pragma unroll
        for (int i = 0; i < MAX_NUM_FEATURES / 2; i++)
        {
            bool active = i < m_numFeatures / 2;
            const half2 zero { 0.f, 0.f };

            half2 x00 = active ? l2[a00] : zero; a00 += (m_latentWidth * m_latentHeight);
            half2 x01 = active ? l2[a01] : zero; a01 += (m_latentWidth * m_latentHeight);
            half2 x10 = active ? l2[a10] : zero; a10 += (m_latentWidth * m_latentHeight);
            half2 x11 = active ? l2[a11] : zero; a11 += (m_latentWidth * m_latentHeight);

            if constexpr (ALL_CORNERS)
            {
                outputArray.set_packed_element(x00 * w00, offset + i);
                outputArray.set_packed_element(x01 * w01, offset + i + 1 * (MAX_NUM_FEATURES / 2));
                outputArray.set_packed_element(x10 * w10, offset + i + 2 * (MAX_NUM_FEATURES / 2));
                outputArray.set_packed_element(x11 * w11, offset + i + 3 * (MAX_NUM_FEATURES / 2));
            }
            else
            {
                half2 d = x00 * w00 + x01 * w01 + x10 * w10 + x11 * w11;
                outputArray.set_packed_element(d, offset + i);
            }
        }
    }

    template<int INPUT_CHANNELS, typename GRID_GRAD_TYPE>
    __device__ void SampleBackward(float u, float v, int offset, const tin::HArray<INPUT_CHANNELS>& inputArray,
        GRID_GRAD_TYPE* gradients, uint32_t* gradientMask)
    {
        float x = u * m_latentWidth  - 0.5f;
        float y = v * m_latentHeight - 0.5f;

        int x_b = floor(x);
        int y_b = floor(y);

        float dx  = x - x_b;
        float dy  = y - y_b;
        float dxn = 1 - dx;
        float dyn = 1 - dy;

        half2 w00 = __float2half2_rn(dxn * dyn);
        half2 w01 = __float2half2_rn(dx * dyn);
        half2 w10 = __float2half2_rn(dxn * dy);
        half2 w11 = __float2half2_rn(dx * dy);

        int x0 = max(0, min(m_latentWidth - 1, x_b));
        int y0 = max(0, min(m_latentHeight - 1, y_b));
        int x1 = max(0, min(m_latentWidth - 1, x_b + 1));
        int y1 = max(0, min(m_latentHeight - 1, y_b + 1));

        bool mask = x0 < m_latentWidth && y0 < m_latentHeight;

        int a00 = (y0 * m_latentWidth + x0);
        int a01 = (y0 * m_latentWidth + x1);
        int a10 = (y1 * m_latentWidth + x0);
        int a11 = (y1 * m_latentWidth + x1);

        // Mark the gradient mask for the touched pixels
        if (mask)
        {
            atomicOr(gradientMask + (a00 >> 5), 1u << (a00 & 31));
            atomicOr(gradientMask + (a01 >> 5), 1u << (a01 & 31));
            atomicOr(gradientMask + (a10 >> 5), 1u << (a10 & 31));
            atomicOr(gradientMask + (a11 >> 5), 1u << (a11 & 31));
        }

#pragma unroll
        for (int i = 0; i < MAX_NUM_FEATURES / 2; i++)
        {
            if (i >= m_numFeatures / 2)
                mask = false;

            half2 lat00, lat01, lat10, lat11;

            if constexpr (ALL_CORNERS)
            {
                lat00 = w00 * inputArray.get_packed_element(offset + i    );
                lat01 = w01 * inputArray.get_packed_element(offset + i + 1 * (MAX_NUM_FEATURES / 2));
                lat10 = w10 * inputArray.get_packed_element(offset + i + 2 * (MAX_NUM_FEATURES / 2));
                lat11 = w11 * inputArray.get_packed_element(offset + i + 3 * (MAX_NUM_FEATURES / 2));
            } 
            else
            {
                half2 lat = inputArray.get_packed_element(offset + i);
                lat00 = lat * w00;
                lat01 = lat * w01;
                lat10 = lat * w10;
                lat11 = lat * w11;
            }

            if (std::is_same<GRID_GRAD_TYPE, float>::value)
            {
                auto l2 = (float2*)gradients;
                float2* x00 = l2 + a00; a00 += (m_latentWidth * m_latentHeight);
                float2* x01 = l2 + a01; a01 += (m_latentWidth * m_latentHeight);
                float2* x10 = l2 + a10; a10 += (m_latentWidth * m_latentHeight);
                float2* x11 = l2 + a11; a11 += (m_latentWidth * m_latentHeight);

                if (mask)
                {
                    tin::_atomic_addf(&(x00->x), float(lat00.x));
                    tin::_atomic_addf(&(x00->y), float(lat00.y));

                    tin::_atomic_addf(&(x01->x), float(lat01.x));
                    tin::_atomic_addf(&(x01->y), float(lat01.y));

                    tin::_atomic_addf(&(x10->x), float(lat10.x));
                    tin::_atomic_addf(&(x10->y), float(lat10.y));

                    tin::_atomic_addf(&(x11->x), float(lat11.x));
                    tin::_atomic_addf(&(x11->y), float(lat11.y));
                }
            }
            else
            {
                auto l2 = (half2*)gradients;
                half2* x00 = l2 + a00; a00 += (m_latentWidth * m_latentHeight);
                half2* x01 = l2 + a01; a01 += (m_latentWidth * m_latentHeight);
                half2* x10 = l2 + a10; a10 += (m_latentWidth * m_latentHeight);
                half2* x11 = l2 + a11; a11 += (m_latentWidth * m_latentHeight);

                if (mask)
                {
                    tin::_atomic_addh2(x00, lat00);
                    tin::_atomic_addh2(x01, lat01);
                    tin::_atomic_addh2(x10, lat10);
                    tin::_atomic_addh2(x11, lat11);
                }
            }
        }
    }

    __device__ int size() {
        return m_latentWidth * m_latentHeight * m_numFeatures;
    }

private:
    int m_latentWidth;
    int m_latentHeight;
    int m_numFeatures;
};

} // namespace ntc::cuda