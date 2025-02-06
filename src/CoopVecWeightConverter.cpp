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

#include "CoopVecWeightConverter.h"
#include "Context.h"
#include "GraphicsResources.h"
#include "StdTypes.h"

#if NTC_WITH_DX12 && NTC_WITH_DX12_COOPVEC
#include <dxgi1_4.h>
#include <nvapi.h>
#include <wrl.h>
#if NVAPI_SDK_VERSION < 572'18
#error "LibNTC requires NVAPI SDK version R570 or newer."
#endif
#endif

namespace ntc
{

static const uint32_t g_NvidiaVendorID = 0x10DE;

#if NTC_WITH_VULKAN
static VkConvertCooperativeVectorMatrixInfoNV GetVkConvertLayerDesc(bool useFP8, int inputChannels, int outputChannels,
    size_t* pDstSize, void const* srcData = nullptr, void* dstData = nullptr)
{
    VkConvertCooperativeVectorMatrixInfoNV layoutInfo;
    layoutInfo.sType = VK_STRUCTURE_TYPE_CONVERT_COOPERATIVE_VECTOR_MATRIX_INFO_NV;
    layoutInfo.pNext = nullptr;
    layoutInfo.srcSize = inputChannels * outputChannels;
    layoutInfo.srcData.hostAddress = srcData;
    layoutInfo.pDstSize = pDstSize;
    layoutInfo.dstData.hostAddress = dstData;
    layoutInfo.srcComponentType = useFP8
        ? VK_COMPONENT_TYPE_FLOAT_E4M3_NV
        : VK_COMPONENT_TYPE_SINT8_KHR;
    layoutInfo.dstComponentType = layoutInfo.srcComponentType;
    layoutInfo.numRows = outputChannels;
    layoutInfo.numColumns = inputChannels;
    layoutInfo.srcLayout = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV;
    layoutInfo.srcStride = inputChannels;
    layoutInfo.dstLayout = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_NV;
    layoutInfo.dstStride = 0;
    return layoutInfo;
}
#endif

#if NTC_WITH_DX12 && NTC_WITH_DX12_COOPVEC
static NVAPI_CONVERT_COOPERATIVE_VECTOR_MATRIX_DESC GetDX12ConvertLayerDesc(bool useFP8, int inputChannels, int outputChannels,
    size_t* pDstSize, void const* srcData = nullptr, void* dstData = nullptr)
{
    NVAPI_CONVERT_COOPERATIVE_VECTOR_MATRIX_DESC convertDesc{};
    convertDesc.version = NVAPI_CONVERT_COOPERATIVE_VECTOR_MATRIX_DESC_VER;
    convertDesc.srcSize = inputChannels * outputChannels;
    convertDesc.srcData.pHostAddress = const_cast<void*>(srcData);
    convertDesc.pDstSize = pDstSize;
    convertDesc.dstData.pHostAddress = dstData;
    convertDesc.srcComponentType = useFP8
        ? NVAPI_COOPERATIVE_VECTOR_COMPONENT_TYPE_SINT8
        : NVAPI_COOPERATIVE_VECTOR_COMPONENT_TYPE_FLOAT_E4M3;
    convertDesc.dstComponentType = convertDesc.srcComponentType;
    convertDesc.numRows = outputChannels;
    convertDesc.numColumns = inputChannels;
    convertDesc.srcLayout = NVAPI_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR;
    convertDesc.srcStride = inputChannels;
    convertDesc.dstLayout = NVAPI_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL;
    convertDesc.dstStride = 0;
    return convertDesc;
}
#endif

CoopVecWeightConverter::CoopVecWeightConverter(GraphicsResources const* resources, bool useFP8,
    int inputChannels, int hiddenChannels, int outputChannels, int numHiddenLayers)
    : m_resources(resources)
    , m_useFP8(useFP8)
    , m_inputChannels(inputChannels)
    , m_hiddenChannels(hiddenChannels)
    , m_outputChannels(outputChannels)
    , m_numHiddenLayers(numHiddenLayers)
{
    if (m_useFP8)
        m_isSupported = resources->IsCoopVecFP8Supported();
    else
        m_isSupported = resources->IsCoopVecInt8Supported();

    if (m_isSupported)
        CalculateOutputSizes();
}

void CoopVecWeightConverter::CalculateOutputSizes()
{
    m_dstTotalSize = 0;
    m_srcTotalSize = 0;
    
    if (m_useFP8 && !m_resources->IsCoopVecFP8Supported() || !m_useFP8 && !m_resources->IsCoopVecInt8Supported())
        return;
    
#if NTC_WITH_VULKAN
    if (m_resources->GetGraphicsApi() == GraphicsAPI::Vulkan)
    {
        VkDevice const vkDevice = m_resources->GetVulkanDevice();
        PFN_vkConvertCooperativeVectorMatrixNV const vkConvertCooperativeVectorMatrixNV =
            m_resources->GetConvertCooperativeVectorMatrixNV();

        assert(vkDevice);
        assert(vkConvertCooperativeVectorMatrixNV);
     
        {
            // Compute converted size for input layer
            VkConvertCooperativeVectorMatrixInfoNV convertInput =
                GetVkConvertLayerDesc(m_useFP8, m_inputChannels, m_hiddenChannels, &m_dstWeightSizeInput);
            vkConvertCooperativeVectorMatrixNV(vkDevice, &convertInput);
            m_dstTotalSize += m_dstWeightSizeInput;
            m_srcTotalSize += convertInput.srcSize;
        }
        {
            // Compute converted size for hidden weights
            VkConvertCooperativeVectorMatrixInfoNV convertHidden =
                GetVkConvertLayerDesc(m_useFP8, m_hiddenChannels, m_hiddenChannels, &m_dstWeightSizeHidden);
            vkConvertCooperativeVectorMatrixNV(vkDevice, &convertHidden);
            m_dstTotalSize += m_dstWeightSizeHidden * m_numHiddenLayers;
            m_srcTotalSize += convertHidden.srcSize * m_numHiddenLayers;
        }
        {
            // Compute converted size for output layer weights
            // Note: the output layer in FP8 mode uses Int8 math
            VkConvertCooperativeVectorMatrixInfoNV convertOutput =
                GetVkConvertLayerDesc(false, m_hiddenChannels, m_outputChannels, &m_dstWeightSizeOutput);
            vkConvertCooperativeVectorMatrixNV(vkDevice, &convertOutput);
            m_dstTotalSize += m_dstWeightSizeOutput;
            m_srcTotalSize += convertOutput.srcSize;
        }

        return;
    }
#endif
#if NTC_WITH_DX12 && NTC_WITH_DX12_COOPVEC
    if (m_resources->GetGraphicsApi() == GraphicsAPI::D3D12)
    {
        ID3D12Device* d3dDevice = m_resources->GetD3D12Device();

        NVAPI_CONVERT_COOPERATIVE_VECTOR_MATRIX_DESC convertInput =
            GetDX12ConvertLayerDesc(m_useFP8, m_inputChannels, m_hiddenChannels, &m_dstWeightSizeInput);
        NvAPI_D3D12_ConvertCooperativeVectorMatrix(d3dDevice, nullptr, &convertInput);
        m_srcTotalSize += convertInput.srcSize;
        m_dstTotalSize += m_dstWeightSizeInput;
        
        NVAPI_CONVERT_COOPERATIVE_VECTOR_MATRIX_DESC convertHidden =
            GetDX12ConvertLayerDesc(m_useFP8, m_hiddenChannels, m_hiddenChannels, &m_dstWeightSizeHidden);
        NvAPI_D3D12_ConvertCooperativeVectorMatrix(d3dDevice, nullptr, &convertHidden);
        m_srcTotalSize += convertHidden.srcSize * m_numHiddenLayers;
        m_dstTotalSize += m_dstWeightSizeHidden * m_numHiddenLayers;

        // Note: the output layer in FP8 mode uses Int8 math
        NVAPI_CONVERT_COOPERATIVE_VECTOR_MATRIX_DESC convertOutput =
            GetDX12ConvertLayerDesc(false, m_hiddenChannels, m_outputChannels, &m_dstWeightSizeOutput);
        NvAPI_D3D12_ConvertCooperativeVectorMatrix(d3dDevice, nullptr, &convertOutput);
        m_srcTotalSize += convertOutput.srcSize;
        m_dstTotalSize += m_dstWeightSizeOutput;
    }
#endif
}

void CoopVecWeightConverter::ConvertWeights(const uint8_t* src, uint8_t* dst)
{
#if NTC_WITH_VULKAN
    if (m_resources->GetGraphicsApi() == GraphicsAPI::Vulkan)
    {
        VkDevice const vkDevice = m_resources->GetVulkanDevice();
        PFN_vkConvertCooperativeVectorMatrixNV const vkConvertCooperativeVectorMatrixNV =
            m_resources->GetConvertCooperativeVectorMatrixNV();

        assert(vkDevice);
        assert(vkConvertCooperativeVectorMatrixNV);

        {
            // Convert input layer weights
            VkConvertCooperativeVectorMatrixInfoNV convertInput =
                GetVkConvertLayerDesc(m_useFP8, m_inputChannels, m_hiddenChannels, &m_dstWeightSizeInput, src, dst);
            vkConvertCooperativeVectorMatrixNV(vkDevice, &convertInput);
            src += convertInput.srcSize;
            dst += m_dstWeightSizeInput;
        }
        for (int hl = 0; hl < m_numHiddenLayers; hl++)
        {
            // Convert hidden layer weights
            VkConvertCooperativeVectorMatrixInfoNV convertHidden =
                GetVkConvertLayerDesc(m_useFP8, m_hiddenChannels, m_hiddenChannels, &m_dstWeightSizeHidden, src, dst);
            vkConvertCooperativeVectorMatrixNV(vkDevice, &convertHidden);
            src += convertHidden.srcSize;
            dst += m_dstWeightSizeHidden;
        }
        {
            // Convert output layer weights
            // Note: the output layer in FP8 mode uses Int8 math
            VkConvertCooperativeVectorMatrixInfoNV convertOutput =
                GetVkConvertLayerDesc(false, m_hiddenChannels, m_outputChannels, &m_dstWeightSizeOutput, src, dst);
            vkConvertCooperativeVectorMatrixNV(vkDevice, &convertOutput);
        }

        return;
    }
#endif
#if NTC_WITH_DX12 && NTC_WITH_DX12_COOPVEC
    if (m_resources->GetGraphicsApi() == GraphicsAPI::D3D12)
    {
        ID3D12Device* d3dDevice = m_resources->GetD3D12Device();

        // Convert input layer weights
        NVAPI_CONVERT_COOPERATIVE_VECTOR_MATRIX_DESC convertInput = GetDX12ConvertLayerDesc(
            m_useFP8, m_inputChannels, m_hiddenChannels, &m_dstWeightSizeInput, src, dst);
        NvAPI_D3D12_ConvertCooperativeVectorMatrix(d3dDevice, nullptr, &convertInput);
        src += convertInput.srcSize;
        dst += m_dstWeightSizeInput;
        
        // Convert hidden layer weights
        for (int hl = 0; hl < m_numHiddenLayers; hl++)
        {
            NVAPI_CONVERT_COOPERATIVE_VECTOR_MATRIX_DESC convertHidden = GetDX12ConvertLayerDesc(
                m_useFP8, m_hiddenChannels, m_hiddenChannels, &m_dstWeightSizeHidden, src, dst);
            NvAPI_D3D12_ConvertCooperativeVectorMatrix(d3dDevice, nullptr, &convertHidden);
            src += convertHidden.srcSize;
            dst += m_dstWeightSizeHidden;
        }

        // Convert output layer weights
        // Note: the output layer in FP8 mode uses Int8 math
        NVAPI_CONVERT_COOPERATIVE_VECTOR_MATRIX_DESC convertOutput = GetDX12ConvertLayerDesc(
            false, m_hiddenChannels, m_outputChannels, &m_dstWeightSizeOutput, src, dst);
        NvAPI_D3D12_ConvertCooperativeVectorMatrix(d3dDevice, nullptr, &convertOutput);
    }
#endif
}

void CoopVecWeightConverter::GetConvertedWeightOffsets(int weightOffsets[NTC_MLP_LAYERS])
{
    weightOffsets[0] = 0;
    weightOffsets[1] = weightOffsets[0] + int(m_dstWeightSizeInput);
    weightOffsets[2] = weightOffsets[1] + int(m_dstWeightSizeHidden);
    weightOffsets[3] = weightOffsets[2] + int(m_dstWeightSizeHidden);
}

#if NTC_WITH_DX12
bool CoopVecWeightConverter::InitializeNVAPI()
{
#if NTC_WITH_DX12_COOPVEC
    return NvAPI_Initialize() == NVAPI_OK;
#else
    return false;
#endif
}

#if NTC_WITH_DX12_COOPVEC
static bool GetNvidiaGpuArchitectureAndDriverVersion(ID3D12Device* pD3DDevice,
    NV_GPU_ARCHITECTURE_ID& outArchitectureID, NvU32& outDriverVersion)
{
    NvAPI_ShortString driverBranch;
    if (NvAPI_SYS_GetDriverAndBranchVersion(&outDriverVersion, driverBranch) != NVAPI_OK)
        return false;

    NvLogicalGpuHandle logicalGPUs[NVAPI_MAX_LOGICAL_GPUS]{};
    NvU32 logicalGpuCount = 0;
    if (NvAPI_EnumLogicalGPUs(logicalGPUs, &logicalGpuCount) != NVAPI_OK)
        return false;

    LUID const adapterLuid = pD3DDevice->GetAdapterLuid();
    for (NvU32 gpu = 0; gpu < logicalGpuCount; ++gpu)
    {
        LUID gpuLuid{};
        NV_LOGICAL_GPU_DATA logicalGpuData{};
        logicalGpuData.version = NV_LOGICAL_GPU_DATA_VER;
        logicalGpuData.pOSAdapterId = &gpuLuid;

        if (NvAPI_GPU_GetLogicalGpuInfo(logicalGPUs[gpu], &logicalGpuData) != NVAPI_OK)
            continue;

        if (logicalGpuData.physicalGpuCount == 0)
            continue;

        if (adapterLuid.LowPart == gpuLuid.LowPart && adapterLuid.HighPart == gpuLuid.HighPart)
        {
            NV_GPU_ARCH_INFO archInfo{};
            archInfo.version = NV_GPU_ARCH_INFO_VER;

            if (NvAPI_GPU_GetArchInfo(logicalGpuData.physicalGpuHandles[0], &archInfo) != NVAPI_OK)
                return false;

            outArchitectureID = archInfo.architecture_id;

            return true;
        }
    }

    return false;
}
#endif

void CoopVecWeightConverter::IsDX12CoopVecSupported(GraphicsResources const* resources,
    bool& outInt8Supported, bool& outFP8Supported)
{
    outInt8Supported = false;
    outFP8Supported = false;

#if NTC_WITH_DX12_COOPVEC
    NV_GPU_ARCHITECTURE_ID architectureID{};
    NvU32 driverVersion = 0;
    if (!GetNvidiaGpuArchitectureAndDriverVersion(resources->GetD3D12Device(), architectureID, driverVersion))
        return;

    // Verify that we're running on NVIDIA driver version 570 or above.
    // R565 reports cooperative vector support through NVAPI but it doesn't actually work.
    if (driverVersion < 570'00)
        return;

    // Verify that we're running on NVIDIA Ada or newer GPU.
    // There are driver bugs (4959641) with DX12 CoopVec on Ampere.
    if (architectureID < NV_GPU_ARCHITECTURE_AD100)
        return;

    // Get the property count
    NvU32 propertyCount = 0;
    if (NvAPI_D3D12_GetPhysicalDeviceCooperativeVectorProperties(resources->GetD3D12Device(),
        &propertyCount, nullptr) != NVAPI_OK)
        return;

    Vector<NVAPI_COOPERATIVE_VECTOR_PROPERTIES> properties(propertyCount, resources->GetAllocator());

    // Get the actual properties
    if (NvAPI_D3D12_GetPhysicalDeviceCooperativeVectorProperties(resources->GetD3D12Device(),
        &propertyCount, properties.data()) != NVAPI_OK)
        return;

    // Go over the properties and see if there are formats that we need in the list
    bool int8InputLayerSupported = false;
    bool int8OtherLayersSupported = false;
    bool fp8LayersSupported = false;
    for (const auto& prop : properties)
    {
        if (prop.version == NVAPI_COOPERATIVE_VECTOR_PROPERTIES_VER &&
            prop.inputType == NVAPI_COOPERATIVE_VECTOR_COMPONENT_TYPE_UINT32 &&
            prop.inputInterpretation == NVAPI_COOPERATIVE_VECTOR_COMPONENT_TYPE_SINT8_PACKED &&
            prop.matrixInterpretation == NVAPI_COOPERATIVE_VECTOR_COMPONENT_TYPE_SINT8 &&
            prop.resultType == NVAPI_COOPERATIVE_VECTOR_COMPONENT_TYPE_SINT32)
            int8InputLayerSupported = true;
            
        if (prop.version == NVAPI_COOPERATIVE_VECTOR_PROPERTIES_VER &&
            prop.inputType == NVAPI_COOPERATIVE_VECTOR_COMPONENT_TYPE_FLOAT32 &&
            prop.inputInterpretation == NVAPI_COOPERATIVE_VECTOR_COMPONENT_TYPE_SINT8 &&
            prop.matrixInterpretation == NVAPI_COOPERATIVE_VECTOR_COMPONENT_TYPE_SINT8 &&
            prop.resultType == NVAPI_COOPERATIVE_VECTOR_COMPONENT_TYPE_SINT32)
            int8OtherLayersSupported = true;

        if (prop.version == NVAPI_COOPERATIVE_VECTOR_PROPERTIES_VER &&
            prop.inputType == NVAPI_COOPERATIVE_VECTOR_COMPONENT_TYPE_FLOAT16 &&
            prop.inputInterpretation == NVAPI_COOPERATIVE_VECTOR_COMPONENT_TYPE_FLOAT_E4M3 &&
            prop.matrixInterpretation == NVAPI_COOPERATIVE_VECTOR_COMPONENT_TYPE_FLOAT_E4M3 &&
            prop.resultType == NVAPI_COOPERATIVE_VECTOR_COMPONENT_TYPE_FLOAT16)
            fp8LayersSupported = true;
    }

    outInt8Supported = int8InputLayerSupported && int8OtherLayersSupported;
    outFP8Supported = int8OtherLayersSupported && fp8LayersSupported;
#endif
}

void CoopVecWeightConverter::UnloadNVAPI()
{
#if NTC_WITH_DX12_COOPVEC
    // TODO: Enable the NvAPI_Unload call.
    // It is disabled as a workaround for bug 5057998: Calling Unload before destroying the DX12 device
    // results in a crash inside the device destructor.
    
    // NvAPI_Unload();
#endif
}
#endif

#if NTC_WITH_VULKAN
void CoopVecWeightConverter::IsVkCoopVecSupported(GraphicsResources const* resources,
    bool& outInt8Supported, bool& outFP8Supported)
{
    outInt8Supported = false;
    outFP8Supported = false;

    PFN_vkGetPhysicalDeviceCooperativeVectorPropertiesNV const vkGetPhysicalDeviceCooperativeVectorPropertiesNV = 
        resources->GetGetPhysicalDeviceCooperativeVectorPropertiesNV();

    VkPhysicalDevice vkPhysicalDevice = resources->GetVulkanPhysicalDevice();

    if (!vkGetPhysicalDeviceCooperativeVectorPropertiesNV || !vkPhysicalDevice)
        return;

    // Verify that the physical device we're on was made by NVIDIA.
    // When running on the integrated GPU in an Optimus system, the vkGetPhysicalDeviceCooperativeVectorPropertiesNV
    // function pointer is nonzero, but calling the function results in a crash.
    if (resources->GetVulkanPhysicalDeviceProperties().vendorID != g_NvidiaVendorID)
        return;
    
    // Get the property count
    uint32_t propertyCount = 0;
    if (vkGetPhysicalDeviceCooperativeVectorPropertiesNV(vkPhysicalDevice,
        &propertyCount, nullptr) != VK_SUCCESS)
        return;
        
    Vector<VkCooperativeVectorPropertiesNV> properties(propertyCount, resources->GetAllocator());
    // Init the sType fields
    for (auto& prop : properties)
        prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_VECTOR_PROPERTIES_NV;

    // Get the actual properties
    if (vkGetPhysicalDeviceCooperativeVectorPropertiesNV(vkPhysicalDevice,
        &propertyCount, properties.data()) != VK_SUCCESS)
        return;

    // Go over the properties and see if there are formats that we need in the list
    bool int8InputLayerSupported = false;
    bool int8OtherLayersSupported = false;
    bool fp8LayersSupported = false;
    for (const auto& prop : properties)
    {
        if (prop.sType == VK_STRUCTURE_TYPE_COOPERATIVE_VECTOR_PROPERTIES_NV &&
            prop.inputType == VK_COMPONENT_TYPE_UINT32_KHR &&
            prop.inputInterpretation == VK_COMPONENT_TYPE_SINT8_PACKED_NV &&
            prop.matrixInterpretation == VK_COMPONENT_TYPE_SINT8_KHR &&
            prop.resultType == VK_COMPONENT_TYPE_SINT32_KHR)
            int8InputLayerSupported = true;
            
        if (prop.sType == VK_STRUCTURE_TYPE_COOPERATIVE_VECTOR_PROPERTIES_NV &&
            prop.inputType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
            prop.inputInterpretation == VK_COMPONENT_TYPE_SINT8_KHR &&
            prop.matrixInterpretation == VK_COMPONENT_TYPE_SINT8_KHR &&
            prop.resultType == VK_COMPONENT_TYPE_SINT32_KHR)
            int8OtherLayersSupported = true;

        if (prop.sType == VK_STRUCTURE_TYPE_COOPERATIVE_VECTOR_PROPERTIES_NV &&
            prop.inputType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
            prop.inputInterpretation == VK_COMPONENT_TYPE_FLOAT_E4M3_NV &&
            prop.matrixInterpretation == VK_COMPONENT_TYPE_FLOAT_E4M3_NV &&
            prop.resultType == VK_COMPONENT_TYPE_FLOAT16_KHR)
            fp8LayersSupported = true;
    }

    outInt8Supported = int8InputLayerSupported && int8OtherLayersSupported;
    outFP8Supported = int8OtherLayersSupported && fp8LayersSupported;
}
#endif

}