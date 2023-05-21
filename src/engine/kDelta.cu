#include "GpuTypes.h"
#include "NNTypes.h"
#include <limits>
#include <cuda_fp16.h>

static __constant__ GpuData cData;

void SetKDeltaGpuData()
{
    cudaError_t status;
    status = cudaMemcpyToSymbol(cData, &(getGpu()._data), sizeof(GpuData));     
    RTERROR(status, "cudaMemcpyToSymbol: SetKDeltaGpuData copy to cData failed");
}

void GetKDeltaGpuData()
{
    cudaError_t status;
    status = cudaMemcpyFromSymbol(&(getGpu()._data), cData, sizeof(GpuData));     
    RTERROR(status, "cudaMemcpyFromSymbol: GetKDeltaGpuData copy From cData failed");
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * static_cast<NNFloat>(1.0 / (T(1) << 8 * sizeof(T)));
        pDelta[uOffset + pos] = w * (a - t) * a * (static_cast<NNFloat>(1.0) - a);
    }
}

template <>
__global__ void LAUNCH_BOUNDS() kCalculateSigmoidOutputDelta_kernel<uint8_t>(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint8_t* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (static_cast<NNFloat>(1.0) / static_cast<NNFloat>(UINT8_MAX));
        pDelta[uOffset + pos] = w * (a - t) * a * (static_cast<NNFloat>(1.0) - a);
    }
}

template <>
__global__ void LAUNCH_BOUNDS() kCalculateSigmoidOutputDelta_kernel<int8_t>(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, int8_t* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (static_cast<NNFloat>(1.0) / static_cast<NNFloat>(INT8_MAX));
        pDelta[uOffset + pos] = w * (a - t) * a * (static_cast<NNFloat>(1.0) - a);
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = pData[dOffset + pos];
        pDelta[uOffset + pos] = w * (a - t) * (static_cast<NNFloat>(1.0) - a * a);
    }
}


#include <cuda_fp16.h>

template <>
__global__ void LAUNCH_BOUNDS() kCalculateTanhOutputDelta_kernel<uint8_t>(uint32_t position, uint32_t batch, uint32_t stride, const NNFloat* pUnit, NNFloat* pDelta, const uint8_t* pData, const NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (static_cast<NNFloat>(1.0) / static_cast<NNFloat>(UINT8_MAX));
        pDelta[uOffset + pos] = w * (a - t) * (static_cast<NNFloat>(1.0) - a * a);
    }
}

template <>
__global__ void LAUNCH_BOUNDS() kCalculateTanhOutputDelta_kernel<int8_t>(uint32_t position, uint32_t batch, uint32_t stride, const NNFloat* pUnit, NNFloat* pDelta, const int8_t* pData, const NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (static_cast<NNFloat>(1.0) / static_cast<NNFloat>(INT8_MAX));
        pDelta[uOffset + pos] = w * (a - t) * (static_cast<NNFloat>(1.0) - a * a);
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, const NNFloat* pUnit, NNFloat* pDelta, const T* pData, const NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = pData[dOffset + pos];
        pDelta[uOffset + pos] = w * (a - t);
    }
}

template <>
__global__ void LAUNCH_BOUNDS() kCalculateLinearOutputDelta_kernel<uint8_t>(uint32_t position, uint32_t batch, uint32_t stride, const NNFloat* pUnit, NNFloat* pDelta, const uint8_t* pData, const NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (static_cast<NNFloat>(1.0) / static_cast<NNFloat>(UINT8_MAX));
        pDelta[uOffset + pos] = w * (a - t);
    }
}

template <>
__global__ void LAUNCH_BOUNDS() kCalculateLinearOutputDelta_kernel<int8_t>(uint32_t position, uint32_t batch, uint32_t stride, const NNFloat* pUnit, NNFloat* pDelta, const int8_t* pData, const NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (static_cast<NNFloat>(1.0) / static_cast<NNFloat>(INT8_MAX));
        pDelta[uOffset + pos] = w * (a - t);
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, const NNFloat* pUnit, NNFloat* pDelta, const T* pData, const NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = pData[dOffset + pos];
        pDelta[uOffset + pos] = w * (a - t) * (a > static_cast<NNFloat>(0.0));
    }
}

template <>
__global__ void LAUNCH_BOUNDS() kCalculateRELUOutputDelta_kernel<uint8_t>(uint32_t position, uint32_t batch, uint32_t stride, const NNFloat* pUnit, NNFloat* pDelta, const uint8_t* pData, const NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (static_cast<NNFloat>(1.0) / static_cast<NNFloat>(UINT8_MAX));
        pDelta[uOffset + pos] = w * (a - t) * (a > static_cast<NNFloat>(0.0));
    }
}

template <>
__global__ void LAUNCH_BOUNDS() kCalculateRELUOutputDelta_kernel<char>(uint32_t position, uint32_t batch, uint32_t stride, const NNFloat* pUnit, NNFloat* pDelta, const char* pData, const NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (static_cast<NNFloat>(1.0) / static_cast<NNFloat>(128.0));
        pDelta[uOffset + pos] = w * (a - t) * (a > static_cast<NNFloat>(0.0));
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, const NNFloat* pUnit, NNFloat* pDelta, const T* pData, const NNFloat* pDataWeight, NNFloat slope)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = pData[dOffset + pos];
        pDelta[uOffset + pos] = w * (a - t) * (a > static_cast<NNFloat>(0.0)) + (a <= static_cast<NNFloat>(0.0)) * slope;
    }
}

template <>
__global__ void LAUNCH_BOUNDS() kCalculateLRELUOutputDelta_kernel<uint8_t>(uint32_t position, uint32_t batch, uint32_t stride, const NNFloat* pUnit, NNFloat* pDelta, const uint8_t* pData, const NNFloat* pDataWeight, NNFloat slope)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (static_cast<NNFloat>(1.0) / static_cast<NNFloat>(256.0));
        pDelta[uOffset + pos] = w * (a - t) * (a > static_cast<NNFloat>(0.0)) + (a <= static_cast<NNFloat>(0.0)) * slope;
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, const NNFloat* pUnit, NNFloat* pDelta, const T* pData, const NNFloat* pDataWeight, NNFloat alpha)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = pData[dOffset + pos];
        pDelta[uOffset + pos] = w * (a - t) * (a >= static_cast<NNFloat>(0.0)) + (a < static_cast<NNFloat>(0.0)) * (a + alpha);
    }
}

template <>
__global__ void LAUNCH_BOUNDS() kCalculateELUOutputDelta_kernel<uint8_t>(uint32_t position, uint32_t batch, uint32_t stride, const NNFloat* pUnit, NNFloat* pDelta, const uint8_t* pData, const NNFloat* pDataWeight, NNFloat alpha)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (static_cast<NNFloat>(1.0) / static_cast<NNFloat>(256.0));
        pDelta[uOffset + pos] = w * (a - t) * (a >= static_cast<NNFloat>(0.0)) + (a < static_cast<NNFloat>(0.0)) * (a + alpha);
    }
}

#include <cuda_fp16.h>

template <>
__global__ void LAUNCH_BOUNDS() kCalculateELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, const NNFloat* pUnit, NNFloat* pDelta, const char* pData, const NNFloat* pDataWeight, NNFloat alpha)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (static_cast<NNFloat>(1.0) / static_cast<NNFloat>(128.0));
        pDelta[uOffset + pos] = w * (a - t) * ((a >= static_cast<NNFloat>(0.0)) + (a < static_cast<NNFloat>(0.0)) * (a + alpha));
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, const NNFloat* pUnit, NNFloat* pDelta, const T* pData, const NNFloat* pDataWeight, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = pData[dOffset + pos];
        pDelta[uOffset + pos] = w * (a - t) * ((a >= static_cast<NNFloat>(0.0)) * lambda + (a < static_cast<NNFloat>(0.0)) * (lambda * alpha * expf(a)));
    }
}

template <>
__global__ void LAUNCH_BOUNDS() kCalculateSELUOutputDelta_kernel<uint8_t>(uint32_t position, uint32_t batch, uint32_t stride, const NNFloat* pUnit, NNFloat* pDelta, const uint8_t* pData, const NNFloat* pDataWeight, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (static_cast<NNFloat>(1.0) / static_cast<NNFloat>(256.0));
        pDelta[uOffset + pos] = w * (a - t) * ((a >= static_cast<NNFloat>(0.0)) * lambda + (a < static_cast<NNFloat>(0.0)) * (lambda * alpha * expf(a)));
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, const NNFloat* pUnit, NNFloat* pDelta, const T* pData, const NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = pData[dOffset + pos];
        pDelta[uOffset + pos] = w * (a - t);
    }
}

template <>
__global__ void LAUNCH_BOUNDS() kCalculateSoftMaxOutputDelta_kernel<uint8_t>(uint32_t position, uint32_t batch, uint32_t stride, const NNFloat* pUnit, NNFloat* pDelta, const uint8_t* pData, const NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (static_cast<NNFloat>(1.0) / static_cast<NNFloat>(256.0));
        pDelta[uOffset + pos] = w * (a - t);
    }
}


