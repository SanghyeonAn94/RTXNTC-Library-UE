# RTX Neural Texture Compression Library (LibNTC)

This repository (or folder, depending on how you got here) contains the source code for the NTC library.

LibNTC can be built separately following the same instructions as the full [RTXNTC SDK](https://github.com/NVIDIA-RTX/RTXNTC), and does not depend on anything else in the SDK. This means it can be included into a larger project by copying only this folder or adding a submodule.

For the integration guide, please see the RTXNTC SDK.

## CMake Configuration options

- `NTC_BUILD_SHARED`: Configures whether LibNTC should be built as a static library (`OFF`) or a dynamic one (`ON`). Default is `ON`.
- `NTC_WITH_CUDA`: Enables the CUDA-based functionality like compression. Set this to `OFF` to build a compact version of LibNTC for integration into game engines.
- `NTC_CUDA_ARCHITECTURES`: List of CUDA architectures in the [CMake format](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html) for which the kernels should be compiled. A reduced list can make builds faster for development purposes.
- `NTC_WITH_DX12`: Enables building the DX12 shaders and weight conversion features (Windows only). Default is `ON`.
- `NTC_WITH_VULKAN`: Enables building the Vulkan shaders and weight conversion features. Default is `ON`.
- `NTC_WITH_PREBUILT_SHADERS`: Enables building the shaders for decompression on load, BCn compression, and image difference passes. Default is `ON`.
- `LIBNTC_BIN_DIRECTORY`: Location for the build LibNTC binaries on all build configurations. When unset, CMake default locations are used.

## Directory structure

- [`external/dxc`](external/dxc): Binary builds of a regular recent version of the DirectX Shader Compiler
- [`external/slang`](external/slang): Binary builds of the [Slang compiler](https://github.com/shader-slang/slang) and a [custom version of DXC](https://github.com/NVIDIA-RTX/DirectXShaderCompiler/tree/CooperativeVector) with Cooperative Vector support
- [`include/libntc`](include/libntc): Public C++ headers for LibNTC
- [`include/libntc/shaders`](include/libntc/shaders): HLSL/Slang shader headers for things like Inference on Sample
- [`src`](src): Source code for LibNTC
- [`src/kernels`](src/kernels): Generated code for permutations of CUDA kernels doing NTC compression and decompression
- [`src/RegressionKernels.h`](src/RegressionKernels.h) Actual source code for the CUDA compression and decompression kernels
- [`src/shaders`](src/shaders): Source code for the prebuilt shaders used by LibNTC, such as decompression or BCn encoding
- [`src/tin`](src/tin): Source code for the Tiny Inline Networks (TIN) library, customized for NTC

## License

[NVIDIA RTX SDKs LICENSE](LICENSE.txt)

This project includes NVAPI software. All uses of NVAPI software are governed by the license terms specified here: https://github.com/NVIDIA/nvapi/blob/main/License.txt