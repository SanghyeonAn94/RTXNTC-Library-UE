// GENERATED FILE - DO NOT MODIFY - USE ../generate_kernels.py TO UPDATE
#pragma once
namespace ntc::cuda {

__global__ void InferenceKernel_small(InferenceKernelParams const params);
__global__ void InferenceKernel_medium(InferenceKernelParams const params);
__global__ void InferenceKernel_large(InferenceKernelParams const params);
__global__ void InferenceKernel_xlarge(InferenceKernelParams const params);
__global__ void RegressionKernel_small_stable0(RegressionKernelParams const params);
__global__ void RegressionKernel_small_stable1(RegressionKernelParams const params);
__global__ void RegressionKernel_medium_stable0(RegressionKernelParams const params);
__global__ void RegressionKernel_medium_stable1(RegressionKernelParams const params);
__global__ void RegressionKernel_large_stable0(RegressionKernelParams const params);
__global__ void RegressionKernel_large_stable1(RegressionKernelParams const params);
__global__ void RegressionKernel_xlarge_stable0(RegressionKernelParams const params);
__global__ void RegressionKernel_xlarge_stable1(RegressionKernelParams const params);

} // namespace ntc::cuda

