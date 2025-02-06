/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "AdaptiveCompressionSession.h"
#include "Errors.h"
#include "KnownLatentShapes.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <algorithm>

#define RUN_TEST 0

namespace ntc
{

Status AdaptiveCompressionSession::Reset(float targetPsnr, float maxBitsPerPixel, int networkVersion)
{
    if (networkVersion == NTC_NETWORK_UNKNOWN)
      networkVersion = NTC_NETWORK_XLARGE;
    
    int const presetCount = GetKnownLatentShapeCount(networkVersion);
    if (presetCount == 0)
    {
        SetErrorMessage("Unsupported networkVersion (%d).", networkVersion);
        return Status::OutOfRange;
    }

    m_targetPsnr = targetPsnr;
    m_currentRunIndex = 0;
    m_presetCount = presetCount;
    m_networkVersion = networkVersion;

    if (maxBitsPerPixel > 0.f)
    {
        // If a maximum bitrate is provided, convert that to a preset index.
        m_maxPreset = -1;
        for (int index = presetCount - 1; index >= 0; --index)
        {
            if (g_KnownLatentShapes[index].bitsPerPixel <= maxBitsPerPixel)
            {
                m_maxPreset = index;
                break;
            }
        }

        // Handle out-of-range bpp values.
        if (m_maxPreset < 0)
        {
            SetErrorMessage("Provided maxBitsPerPixel value (%.2f) is out of the supported range "
                "for %s network (%.1f-%.0f).",
                maxBitsPerPixel,
                NetworkVersionToString(networkVersion),
                g_KnownLatentShapes[0].bitsPerPixel,
                g_KnownLatentShapes[presetCount - 1].bitsPerPixel);
            return Status::InvalidArgument;
        }
    }
    else
    {
        // No maximum bitrate is provided.
        m_maxPreset = presetCount - 1;
    }

    // Start the full-range search at some midpoint.
    // Constrain the index to 12 becaues that maps to 3.5 bpp, which happens to be noticeably faster than 4.0.
    m_currentPreset = std::min(m_maxPreset / 2, 12);
    assert(g_KnownLatentShapes[m_currentPreset].bitsPerPixel <= 3.5f);

    // Reset the boundaries.
    m_leftPreset = -1;
    m_rightPreset = -1;
    m_leftPsnr = -1;
    m_rightPsnr = -1;

    ClearErrorMessage();
    return Status::Ok;
}

bool AdaptiveCompressionSession::Finished()
{
    return m_leftPreset == m_rightPreset && m_leftPreset >= 0;
}

void AdaptiveCompressionSession::GetCurrentPreset(float *pOutBitsPerPixel, LatentShape *pOutLatentShape)
{
    KnownLatentShape const& preset = g_KnownLatentShapes[m_currentPreset];
    if (pOutBitsPerPixel)
        *pOutBitsPerPixel = preset.bitsPerPixel;
    if (pOutLatentShape)
        *pOutLatentShape = preset.shapes[m_networkVersion - NTC_NETWORK_SMALL];
}

/* This is the function used for interpolation search - a logarithmic curve of PSNR vs. BPP
   The forward version is not used here, just for reference.
static float Model(float x, float a, float b)
{
    return a * logf(x) + b;
}*/

// The inverse of Model(...)
static float InverseModel(float y, float a, float b)
{
    return expf((y - b) / a);
}

// Calculates the (a,b) parameters of the Model function using two points (x1,y1) and (x2,y2)
static void GetModelParams(float x1, float x2, float y1, float y2, float& a, float& b)
{
    x1 = logf(x1);
    x2 = logf(x2);
    a = (y2 - y1) / (x2 - x1);
    b = y1 - a * x1;
}

// Finds the index of the preset with BPP most closely matching the given one.
// Only searches within the (excludeLeft, excludeRight) index range, non-inclusively.
// When no matching preset found, returns -1.
static int FindClosestPreset(float targetBpp, int excludeLeft, int excludeRight, int presetCount)
{
    // If the requested BPP is clearly out of range, early out.
    if (excludeLeft >= 0 && targetBpp <= g_KnownLatentShapes[excludeLeft].bitsPerPixel ||
        excludeRight < presetCount && targetBpp >= g_KnownLatentShapes[excludeRight].bitsPerPixel)
        return -1;

    int bestIndex = -1;
    float bestBpp = -1;
    for (int index = excludeLeft + 1; index < excludeRight; ++index)
    {
        float currentBpp = g_KnownLatentShapes[index].bitsPerPixel;
        if ((bestIndex < 0) || (fabsf(currentBpp - targetBpp) < fabsf(bestBpp - targetBpp)))
        {
            bestIndex = index;
            bestBpp = currentBpp;
        }
    }
    return bestIndex;
}

void AdaptiveCompressionSession::Next(float currentPsnr)
{
    // Store the current preset in the history array.
    assert(m_currentRunIndex < MaxRuns);
    m_presetHistory[m_currentRunIndex] = m_currentPreset;
    ++m_currentRunIndex;

    // This is the first experiment - we got the midpoint result.
    // Choose if we test the lowest or highest preset next.
    if (m_leftPreset < 0 && m_rightPreset < 0)
    {
        if (currentPsnr >= m_targetPsnr)
        {
            m_rightPreset = m_currentPreset;
            m_rightPsnr = currentPsnr;
            m_leftPreset = 0;
            m_leftPsnr = -1;
            m_currentPreset = m_leftPreset;
        }
        else
        {
            m_leftPreset = m_currentPreset;
            m_leftPsnr = currentPsnr;
            m_rightPreset = m_maxPreset;
            m_rightPsnr = -1;
            m_currentPreset = m_rightPreset;
        }
        return;
    }

    // Maybe this is the second experiment - we got either the left or right result.
    if (m_currentPreset == m_leftPreset)
        m_leftPsnr = currentPsnr;
    else if (m_currentPreset == m_rightPreset)
        m_rightPsnr = currentPsnr;
    else
    {
        // No, it's some midpoint.
        // Update the boundaries according to the experiment result.
        if (currentPsnr >= m_targetPsnr)
        {
            m_rightPreset = m_currentPreset;
            m_rightPsnr = currentPsnr;
        }
        else
        {
            m_leftPreset = m_currentPreset;
            m_leftPsnr = currentPsnr;
        }
    }

    // Early out if the search range is empty after updating the boundaries.
    if (m_leftPreset == m_rightPreset)
        return;

    // Fit a model curve to the current boundaries
    float a, b;
    float const leftBpp = g_KnownLatentShapes[m_leftPreset].bitsPerPixel;
    float const rightBpp = g_KnownLatentShapes[m_rightPreset].bitsPerPixel;
    GetModelParams(leftBpp, rightBpp, m_leftPsnr, m_rightPsnr, a, b);

    // Predict the optimal BPP using the fitted model
    float const expectedBpp = InverseModel(m_targetPsnr, a, b);

    // Find a real BPP value most closely matching the predicted BPP, but excluding the left and right points.
    int const expectedPreset = FindClosestPreset(expectedBpp, m_leftPreset, m_rightPreset, m_presetCount);

    // If the prediction is not matching any real point between left and right, stop the search.
    if (expectedPreset < 0)
    {
        if (m_leftPsnr >= m_targetPsnr)
        {
            m_rightPreset = m_leftPreset;
            m_rightPsnr = m_leftPsnr;
        }
        else
        {
            m_leftPreset = m_rightPreset;
            m_leftPsnr = m_rightPsnr;
        }
        m_currentPreset = m_leftPreset;
    }
    else
    {
        m_currentPreset = expectedPreset;
    }
}

int AdaptiveCompressionSession::GetIndexOfFinalRun()
{
    if (!Finished())
        return -1;

    for (int index = 0; index < m_currentRunIndex; ++index)
    {
        if (m_presetHistory[index] == m_currentPreset)
            return index;
    }

    assert(false);
    return -1;
}

#if RUN_TEST

// Experimental PSNR vs. BPP curves from some materials.
constexpr int TestMaterialCount = 27;
static const float TestData[TestMaterialCount][KnownLatentShapeCount] = {
    { 30.59f, 31.19f, 31.66f, 32.30f, 32.79f, 33.43f, 34.00f, 35.47f, 35.93f, 36.73f, 37.11f, 38.07f, 38.63f,
      39.32f, 40.18f, 40.85f, 41.04f, 41.84f, 42.41f, 43.26f, 43.71f, 44.74f, 46.88f, 47.51f, 48.22f, 48.72f },
    { 24.76f, 25.34f, 25.60f, 25.92f, 26.21f, 26.65f, 27.09f, 27.71f, 28.08f, 28.57f, 28.88f, 29.52f, 29.50f,
      30.13f, 30.40f, 30.96f, 31.95f, 32.88f, 33.50f, 34.39f, 34.98f, 36.29f, 36.59f, 37.09f, 38.95f, 40.14f },
    { 27.33f, 27.79f, 28.12f, 28.62f, 28.91f, 29.37f, 29.72f, 30.46f, 30.71f, 31.46f, 31.80f, 32.74f, 32.62f,
      33.26f, 34.22f, 34.82f, 36.35f, 37.16f, 37.86f, 39.33f, 39.91f, 41.44f, 44.36f, 45.09f, 46.95f, 47.90f },
    { 32.74f, 33.23f, 33.69f, 34.08f, 34.55f, 35.08f, 35.62f, 36.42f, 36.87f, 37.50f, 37.83f, 38.54f, 38.88f,
      39.57f, 39.91f, 40.42f, 41.23f, 41.93f, 42.51f, 43.52f, 44.10f, 45.36f, 46.15f, 46.53f, 47.87f, 48.72f },
    { 44.53f, 44.81f, 47.60f, 46.40f, 48.69f, 49.88f, 50.28f, 50.50f, 51.04f, 51.01f, 51.46f, 51.81f, 51.84f,
      52.26f, 52.34f, 52.28f, 52.19f, 52.94f, 53.32f, 53.35f, 53.65f, 53.92f, 53.71f, 55.65f, 54.01f, 54.19f },
    { 35.52f, 36.03f, 37.27f, 37.81f, 38.59f, 39.51f, 40.35f, 41.31f, 41.90f, 42.80f, 43.27f, 44.32f, 45.89f,
      47.02f, 47.12f, 47.89f, 48.36f, 49.36f, 49.92f, 50.80f, 51.46f, 52.61f, 55.02f, 55.68f, 56.20f, 57.31f },
    { 26.38f, 26.93f, 27.27f, 27.98f, 28.30f, 28.89f, 29.59f, 30.73f, 31.38f, 33.07f, 33.54f, 35.40f, 34.76f,
      36.22f, 38.06f, 39.04f, 39.29f, 40.22f, 40.85f, 41.96f, 42.50f, 43.74f, 46.33f, 47.09f, 48.22f, 48.99f },
    { 30.78f, 31.97f, 32.71f, 33.13f, 33.81f, 34.57f, 35.17f, 35.85f, 36.19f, 36.77f, 37.18f, 37.86f, 38.01f,
      38.78f, 39.00f, 39.80f, 39.97f, 40.96f, 41.61f, 42.53f, 43.07f, 44.09f, 44.61f, 44.80f, 46.23f, 46.83f },
    { 26.29f, 26.93f, 27.33f, 27.76f, 28.03f, 28.63f, 29.08f, 29.75f, 30.14f, 30.63f, 31.04f, 31.74f, 31.96f,
      32.64f, 33.11f, 33.70f, 34.86f, 35.79f, 36.50f, 37.55f, 38.15f, 39.54f, 40.88f, 41.48f, 43.59f, 44.80f },
    { 35.10f, 35.24f, 35.62f, 35.99f, 36.30f, 36.75f, 36.96f, 37.67f, 38.10f, 38.68f, 39.02f, 39.79f, 39.98f,
      40.85f, 41.65f, 42.12f, 44.17f, 44.99f, 45.50f, 47.24f, 47.58f, 48.99f, 50.97f, 51.65f, 52.16f, 52.95f },
    { 30.88f, 31.42f, 31.89f, 32.27f, 32.68f, 33.18f, 33.59f, 34.41f, 34.72f, 35.45f, 35.71f, 36.63f, 36.65f,
      37.43f, 38.17f, 38.71f, 40.37f, 41.15f, 41.79f, 43.19f, 43.75f, 45.16f, 47.40f, 47.92f, 49.52f, 50.56f },
    { 40.18f, 40.55f, 41.45f, 41.78f, 42.42f, 43.09f, 43.70f, 44.63f, 45.05f, 45.84f, 46.26f, 47.28f, 47.93f,
      48.79f, 49.55f, 50.12f, 50.73f, 51.49f, 51.94f, 52.59f, 53.01f, 53.93f, 55.52f, 56.43f, 56.73f, 57.06f },
    { 36.62f, 36.79f, 37.58f, 37.71f, 38.38f, 38.86f, 39.34f, 39.96f, 40.31f, 40.87f, 41.18f, 41.94f, 42.11f,
      42.73f, 42.99f, 43.65f, 44.85f, 45.40f, 45.98f, 47.15f, 47.44f, 48.55f, 48.82f, 49.38f, 50.46f, 50.88f },
    { 40.80f, 42.00f, 42.32f, 43.54f, 43.95f, 45.24f, 46.29f, 47.83f, 48.79f, 49.54f, 50.15f, 51.11f, 53.14f,
      53.60f, 53.98f, 54.34f, 53.46f, 54.18f, 54.84f, 55.01f, 55.58f, 56.28f, 57.48f, 58.06f, 58.73f, 59.13f },
    { 34.36f, 34.92f, 35.55f, 36.33f, 36.85f, 37.52f, 38.35f, 39.53f, 40.21f, 41.24f, 41.69f, 42.96f, 44.35f,
      45.46f, 46.20f, 46.94f, 47.13f, 48.07f, 48.78f, 49.58f, 50.02f, 50.92f, 53.34f, 54.72f, 54.55f, 54.93f },
    { 29.80f, 30.49f, 30.96f, 31.30f, 31.76f, 32.35f, 32.83f, 33.46f, 33.86f, 34.44f, 34.88f, 35.66f, 35.89f,
      36.59f, 37.03f, 37.61f, 38.41f, 39.62f, 40.26f, 41.19f, 41.78f, 42.98f, 44.13f, 44.59f, 46.09f, 47.19f },
    { 34.22f, 34.79f, 35.38f, 35.96f, 36.46f, 37.16f, 37.73f, 38.63f, 39.08f, 39.76f, 40.12f, 40.91f, 41.22f,
      42.00f, 42.50f, 43.16f, 43.62f, 44.52f, 45.29f, 46.18f, 46.73f, 47.89f, 49.18f, 49.89f, 51.09f, 51.89f },
    { 33.08f, 33.92f, 34.54f, 34.94f, 35.42f, 36.28f, 36.87f, 37.46f, 37.96f, 38.40f, 38.85f, 39.58f, 39.87f,
      40.65f, 41.11f, 41.74f, 42.07f, 43.19f, 44.04f, 44.81f, 45.50f, 46.79f, 47.68f, 48.41f, 49.92f, 51.04f },
    { 33.21f, 33.80f, 34.72f, 34.97f, 35.76f, 36.35f, 36.94f, 37.84f, 38.20f, 38.89f, 39.24f, 40.04f, 40.31f,
      40.98f, 41.30f, 41.95f, 42.87f, 43.84f, 44.40f, 45.39f, 46.09f, 47.02f, 47.78f, 48.75f, 49.87f, 50.36f },
    { 32.23f, 32.86f, 33.42f, 33.69f, 34.10f, 34.79f, 35.33f, 35.92f, 36.36f, 36.86f, 37.23f, 38.02f, 38.14f,
      38.86f, 39.25f, 39.97f, 40.43f, 41.33f, 42.05f, 42.88f, 43.47f, 44.78f, 45.95f, 46.45f, 48.28f, 49.28f },
    { 30.34f, 30.87f, 31.08f, 31.57f, 31.78f, 32.26f, 32.67f, 33.33f, 33.69f, 34.33f, 34.65f, 35.52f, 35.64f,
      36.24f, 36.85f, 37.50f, 38.96f, 39.83f, 40.71f, 41.90f, 42.72f, 44.08f, 46.54f, 47.14f, 49.21f, 50.14f },
    { 30.24f, 30.95f, 31.51f, 32.08f, 32.60f, 33.33f, 33.86f, 34.72f, 35.15f, 35.98f, 36.38f, 37.38f, 37.94f,
      38.81f, 39.41f, 40.27f, 41.37f, 42.43f, 43.15f, 44.44f, 44.98f, 46.31f, 49.14f, 50.36f, 51.33f, 52.12f },
    { 32.89f, 34.04f, 34.63f, 35.26f, 35.80f, 36.73f, 37.45f, 38.15f, 38.67f, 39.20f, 39.73f, 40.54f, 40.96f,
      41.98f, 42.20f, 43.10f, 43.02f, 44.41f, 45.27f, 45.81f, 46.74f, 47.92f, 49.43f, 50.13f, 51.55f, 52.66f },
    { 29.64f, 30.78f, 31.10f, 32.04f, 32.29f, 33.11f, 33.77f, 34.69f, 35.24f, 35.93f, 36.30f, 37.19f, 38.17f,
      38.81f, 39.57f, 40.13f, 40.52f, 41.63f, 42.60f, 43.52f, 44.17f, 45.56f, 48.18f, 48.92f, 50.59f, 51.24f },
    { 39.08f, 39.85f, 40.56f, 40.83f, 41.49f, 42.22f, 42.86f, 43.49f, 44.00f, 44.50f, 44.92f, 45.70f, 45.58f,
      46.45f, 46.75f, 47.38f, 47.99f, 49.00f, 49.64f, 50.18f, 50.86f, 51.79f, 52.21f, 53.09f, 53.60f, 54.19f },
    { 36.52f, 37.47f, 37.96f, 39.07f, 39.56f, 40.34f, 41.07f, 42.55f, 43.18f, 43.98f, 44.39f, 45.36f, 46.37f,
      47.20f, 47.87f, 48.49f, 48.11f, 49.01f, 49.44f, 50.41f, 50.94f, 51.93f, 54.33f, 55.15f, 55.54f, 56.02f },
    { 37.37f, 37.94f, 38.57f, 39.11f, 39.65f, 40.26f, 40.69f, 41.67f, 42.11f, 42.82f, 43.20f, 44.14f, 44.49f,
      45.16f, 45.92f, 46.50f, 47.42f, 48.18f, 48.81f, 49.88f, 50.28f, 51.24f, 53.02f, 54.02f, 54.63f, 55.03f },
};

bool AdaptiveCompressionSession::Test()
{
    bool testPassed = true;
    float const maxBitsPerPixel = 12.f;
    int const networkVersion = NTC_NETWORK_LARGE;
    int const presetCount = GetKnownLatentShapeCount(networkVersion);

    AdaptiveCompressionSession session(nullptr);
    for (int materialIndex = 0; materialIndex < TestMaterialCount; ++materialIndex)
    {
        for (float targetPsnr = 30.f; targetPsnr <= 50.f; targetPsnr += 5.f)
        {
            float bppHistory[MaxRuns];

            // Find the optimal BPP using the adaptive compression session
            int experimentCount = 0;
            session.Reset(targetPsnr, maxBitsPerPixel, networkVersion);
            while (!session.Finished())
            {
                float bpp;
                session.GetCurrentPreset(&bpp, nullptr);
                bppHistory[experimentCount] = bpp;
                
                int preset = FindClosestPreset(bpp, -1, presetCount + 1, presetCount);
                assert(preset >= 0);

                float psnr = TestData[materialIndex][preset];
                ++experimentCount;
                session.Next(psnr);
            }

            float finalBpp;
            session.GetCurrentPreset(&finalBpp, nullptr);

            // Verify that GetIndexOfFinalRun() returns the correct index
            int finalIndex = session.GetIndexOfFinalRun();
            assert(bppHistory[finalIndex] == finalBpp);

            // Find the optimal BPP using linear search.
            // This loop will produce the lowest BPP that results in the target PSNR or more,
            // unless the target PSNR cannot be reached within 'maxBitsPerPixel', in which case the highest
            // supported BPP will be returned.
            float idealBpp = -1;
            for (int presetIndex = 0;
                presetIndex < presetCount &&
                    (maxBitsPerPixel <= 0.f || g_KnownLatentShapes[presetIndex].bitsPerPixel <= maxBitsPerPixel);
                ++presetIndex)
            {
                idealBpp = g_KnownLatentShapes[presetIndex].bitsPerPixel;

                if (TestData[materialIndex][presetIndex] >= targetPsnr)
                    break;
            }

            // Compare the results
            char const* testResult;
            if (idealBpp == finalBpp)
                testResult = "OK";
            else if (idealBpp < finalBpp)
                testResult = "SUBOPT";
            else
            {
                testResult = "FAIL";
                testPassed = false;
            }

            printf("Material %d: target = %.2f dB, found %5.2f bpp, ideal %5.2f bpp, %d experiments, final #%d - %s\n",
                materialIndex, targetPsnr, finalBpp, idealBpp, experimentCount, finalIndex, testResult);
        }
    }

    return testPassed;
}

// Poor man's test framework: call a function at module init.
static bool g_TestPassed = AdaptiveCompressionSession::Test();

#endif

}