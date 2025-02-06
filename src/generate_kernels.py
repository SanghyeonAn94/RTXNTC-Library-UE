# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


# This file generates code for CUDA kernel permutations with various template parameter values.
# The purpose of this process is to make many .cu files with one kernel version in each file,
# which reduces the time that it takes to build the project by compiling many kernels in parallel.

# The script is not invoked by the build process and is supposed to be executed manually 
# whenever the set of necessary kernels changes. Check in the generated files into git.

regression_versions = [
    ("small", "NTC_NETWORK_SMALL"),
    ("medium", "NTC_NETWORK_MEDIUM"),
    ("large", "NTC_NETWORK_LARGE"),
    ("xlarge", "NTC_NETWORK_XLARGE")
]


generated_notice = "// GENERATED FILE - DO NOT MODIFY - USE ../generate_kernels.py TO UPDATE\n"

# Permutations header - used on the host side to select the right kernel version 
# given the template parameters, and to validate the parameters.
with open("kernels/kernel_permutations.h", "w") as file:
    file.write(generated_notice)
    file.write("#pragma once\n")
    file.write("namespace ntc::cuda {\n")
    file.write("\n")

    for (name, symbol) in regression_versions:
        file.write(f"__global__ void InferenceKernel_{name}(InferenceKernelParams const params);\n")

    for (name, symbol) in regression_versions:
        for stable in range(2):
            file.write(f"__global__ void RegressionKernel_{name}_stable{stable}(RegressionKernelParams const params);\n")

    file.write("\n")
    file.write("} // namespace ntc::cuda\n")
    file.write("\n")


# Individual kernel files
for (name, symbol) in regression_versions:

    # Inference kernel
    with open(f"kernels/inference_{name}.cu", "w") as file:
        file.write(generated_notice)
        file.write('#include "../RegressionKernels.h"\n')
        file.write(f"INFERENCE_KERNEL_IMPL({name}, {symbol})\n")

    # Regression kernel
    for stable in range(2):
        with open(f"kernels/regression_{name}_stable{stable}.cu", "w") as file:
            file.write(generated_notice)
            file.write('#include "../RegressionKernels.h"\n')
            file.write(f"REGRESSION_KERNEL_IMPL({name}, {symbol}, {stable})\n")
