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
__global__ void LAUNCH_BOUNDS() kCalculateSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, T* pData, Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<Float>(1.0);
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * static_cast<Float>(1.0 / (T(1) << 8 * sizeof(T)));
        pDelta[uOffset + pos] = w * (a - t) * a * (static_cast<Float>(1.0) - a);
    }
}

template <>
__global__ void LAUNCH_BOUNDS() kCalculateSigmoidOutputDelta_kernel<uint8_t>(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint8_t* pData, Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<Float>(1.0);
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (static_cast<Float>(1.0) / static_cast<Float>(UINT8_MAX));
        pDelta[uOffset + pos] = w * (a - t) * a * (static_cast<Float>(1.0) - a);
    }
}

template <>
__global__ void LAUNCH_BOUNDS() kCalculateSigmoidOutputDelta_kernel<int8_t>(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, int8_t* pData, Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<Float>(1.0);
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (static_cast<Float>(1.0) / static_cast<Float>(INT8_MAX));
        pDelta[uOffset + pos] = w * (a - t) * a * (static_cast<Float>(1.0) - a);
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, T* pData, Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<Float>(1.0);
        Float a = pUnit[uOffset + pos];
        Float t = pData[dOffset + pos];
        pDelta[uOffset + pos] = w * (a - t) * (static_cast<Float>(1.0) - a * a);
    }
}


#include <cuda_fp16.h>

template <>
__global__ void LAUNCH_BOUNDS() kCalculateTanhOutputDelta_kernel<uint8_t>(uint32_t position, uint32_t batch, uint32_t stride, const Float* pUnit, Float* pDelta, const uint8_t* pData, const Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<Float>(1.0);
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (static_cast<Float>(1.0) / static_cast<Float>(UINT8_MAX));
        pDelta[uOffset + pos] = w * (a - t) * (static_cast<Float>(1.0) - a * a);
    }
}

template <>
__global__ void LAUNCH_BOUNDS() kCalculateTanhOutputDelta_kernel<int8_t>(uint32_t position, uint32_t batch, uint32_t stride, const Float* pUnit, Float* pDelta, const int8_t* pData, const Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<Float>(1.0);
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (static_cast<Float>(1.0) / static_cast<Float>(INT8_MAX));
        pDelta[uOffset + pos] = w * (a - t) * (static_cast<Float>(1.0) - a * a);
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, const Float* pUnit, Float* pDelta, const T* pData, const Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<Float>(1.0);
        Float a = pUnit[uOffset + pos];
        Float t = pData[dOffset + pos];
        pDelta[uOffset + pos] = w * (a - t);
    }
}

template <>
__global__ void LAUNCH_BOUNDS() kCalculateLinearOutputDelta_kernel<uint8_t>(uint32_t position, uint32_t batch, uint32_t stride, const Float* pUnit, Float* pDelta, const uint8_t* pData, const Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<Float>(1.0);
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (static_cast<Float>(1.0) / static_cast<Float>(UINT8_MAX));
        pDelta[uOffset + pos] = w * (a - t);
    }
}

template <>
__global__ void LAUNCH_BOUNDS() kCalculateLinearOutputDelta_kernel<int8_t>(uint32_t position, uint32_t batch, uint32_t stride, const Float* pUnit, Float* pDelta, const int8_t* pData, const Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<Float>(1.0);
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (static_cast<Float>(1.0) / static_cast<Float>(INT8_MAX));
        pDelta[uOffset + pos] = w * (a - t);
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, const Float* pUnit, Float* pDelta, const T* pData, const Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<Float>(1.0);
        Float a = pUnit[uOffset + pos];
        Float t = pData[dOffset + pos];
        pDelta[uOffset + pos] = w * (a - t) * (a > static_cast<Float>(0.0));
    }
}

template <>
__global__ void LAUNCH_BOUNDS() kCalculateRELUOutputDelta_kernel<uint8_t>(uint32_t position, uint32_t batch, uint32_t stride, const Float* pUnit, Float* pDelta, const uint8_t* pData, const Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<Float>(1.0);
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (static_cast<Float>(1.0) / static_cast<Float>(UINT8_MAX));
        pDelta[uOffset + pos] = w * (a - t) * (a > static_cast<Float>(0.0));
    }
}

template <>
__global__ void LAUNCH_BOUNDS() kCalculateRELUOutputDelta_kernel<char>(uint32_t position, uint32_t batch, uint32_t stride, const Float* pUnit, Float* pDelta, const char* pData, const Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<Float>(1.0);
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (static_cast<Float>(1.0) / static_cast<Float>(128.0));
        pDelta[uOffset + pos] = w * (a - t) * (a > static_cast<Float>(0.0));
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, const Float* pUnit, Float* pDelta, const T* pData, const Float* pDataWeight, Float slope)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<Float>(1.0);
        Float a = pUnit[uOffset + pos];
        Float t = pData[dOffset + pos];
        pDelta[uOffset + pos] = w * (a - t) * (a > static_cast<Float>(0.0)) + (a <= static_cast<Float>(0.0)) * slope;
    }
}

template <>
__global__ void LAUNCH_BOUNDS() kCalculateLRELUOutputDelta_kernel<uint8_t>(uint32_t position, uint32_t batch, uint32_t stride, const Float* pUnit, Float* pDelta, const uint8_t* pData, const Float* pDataWeight, Float slope)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<Float>(1.0);
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (static_cast<Float>(1.0) / static_cast<Float>(256.0));
        pDelta[uOffset + pos] = w * (a - t) * (a > static_cast<Float>(0.0)) + (a <= static_cast<Float>(0.0)) * slope;
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, const Float* pUnit, Float* pDelta, const T* pData, const Float* pDataWeight, Float alpha)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<Float>(1.0);
        Float a = pUnit[uOffset + pos];
        Float t = pData[dOffset + pos];
        pDelta[uOffset + pos] = w * (a - t) * (a >= static_cast<Float>(0.0)) + (a < static_cast<Float>(0.0)) * (a + alpha);
    }
}

template <>
__global__ void LAUNCH_BOUNDS() kCalculateELUOutputDelta_kernel<uint8_t>(uint32_t position, uint32_t batch, uint32_t stride, const Float* pUnit, Float* pDelta, const uint8_t* pData, const Float* pDataWeight, Float alpha)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<Float>(1.0);
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (static_cast<Float>(1.0) / static_cast<Float>(256.0));
        pDelta[uOffset + pos] = w * (a - t) * (a >= static_cast<Float>(0.0)) + (a < static_cast<Float>(0.0)) * (a + alpha);
    }
}

#include <cuda_fp16.h>

template <>
__global__ void LAUNCH_BOUNDS() kCalculateELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, const Float* pUnit, Float* pDelta, const char* pData, const Float* pDataWeight, Float alpha)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<Float>(1.0);
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (static_cast<Float>(1.0) / static_cast<Float>(128.0));
        pDelta[uOffset + pos] = w * (a - t) * ((a >= static_cast<Float>(0.0)) + (a < static_cast<Float>(0.0)) * (a + alpha));
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, const Float* pUnit, Float* pDelta, const T* pData, const Float* pDataWeight, Float alpha, Float lambda)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<Float>(1.0);
        Float a = pUnit[uOffset + pos];
        Float t = pData[dOffset + pos];
        pDelta[uOffset + pos] = w * (a - t) * ((a >= static_cast<Float>(0.0)) * lambda + (a < static_cast<Float>(0.0)) * (lambda * alpha * expf(a)));
    }
}

template <>
__global__ void LAUNCH_BOUNDS() kCalculateSELUOutputDelta_kernel<uint8_t>(uint32_t position, uint32_t batch, uint32_t stride, const Float* pUnit, Float* pDelta, const uint8_t* pData, const Float* pDataWeight, Float alpha, Float lambda)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<Float>(1.0);
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (static_cast<Float>(1.0) / static_cast<Float>(256.0));
        pDelta[uOffset + pos] = w * (a - t) * ((a >= static_cast<Float>(0.0)) * lambda + (a < static_cast<Float>(0.0)) * (lambda * alpha * expf(a)));
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, const Float* pUnit, Float* pDelta, const T* pData, const Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<Float>(1.0);
        Float a = pUnit[uOffset + pos];
        Float t = pData[dOffset + pos];
        pDelta[uOffset + pos] = w * (a - t);
    }
}

template <>
__global__ void LAUNCH_BOUNDS() kCalculateSoftMaxOutputDelta_kernel<uint8_t>(uint32_t position, uint32_t batch, uint32_t stride, const Float* pUnit, Float* pDelta, const uint8_t* pData, const Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<Float>(1.0);
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (static_cast<Float>(1.0) / static_cast<Float>(256.0));
        pDelta[uOffset + pos] = w * (a - t);
    }
}

template<typename T>
void kCalculateOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, T* pData, Float* pDataWeight, Float slope, Float alpha, Float lambda)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    switch (activation)
    {
        case Sigmoid:
            kCalculateSigmoidOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            LAUNCHERROR("kCalculateSigmoidOutputDelta_kernel");
            break;

        case Tanh:
            if (sizeof(T) == sizeof(unsigned char))
            {
                kCalculateTanhOutputDelta_kernel<uint8_t><<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, reinterpret_cast<uint8_t*>(pData), pDataWeight);
            }
            else if (sizeof(T) == sizeof(char))
            {
                kCalculateTanhOutputDelta_kernel<char><<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, reinterpret_cast<char*>(pData), pDataWeight);
            }
            else
            {
                kCalculateTanhOutputDelta_kernel<T><<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            }
            LAUNCHERROR("kCalculateTanhOutputDelta_kernel");
            break;

        case Linear:
            if (sizeof(T) == sizeof(unsigned char))
            {
                kCalculateLinearOutputDelta_kernel<uint8_t><<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, reinterpret_cast<uint8_t*>(pData), pDataWeight);
            }
            else if (sizeof(T) == sizeof(char))
            {
                kCalculateLinearOutputDelta_kernel<char><<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, reinterpret_cast<char*>(pData), pDataWeight);
            }
            else
            {
                kCalculateLinearOutputDelta_kernel<T><<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            }
            LAUNCHERROR("kCalculateLinearOutputDelta_kernel");
            break;

        case RectifiedLinear:
            if (sizeof(T) == sizeof(unsigned char))
            {
                kCalculateRELUOutputDelta_kernel<uint8_t><<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, reinterpret_cast<uint8_t*>(pData), pDataWeight);
            }
            else if (sizeof(T) == sizeof(char))
            {
                kCalculateRELUOutputDelta_kernel<char><<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, reinterpret_cast<char*>(pData), pDataWeight);
            }
            else
            {
                kCalculateRELUOutputDelta_kernel<T><<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            }
            LAUNCHERROR("kCalculateRELUOutputDelta_kernel");
            break;

        case LeakyRectifiedLinear:
            if (sizeof(T) == sizeof(unsigned char))
            {
                kCalculateLRELUOutputDelta_kernel<uint8_t><<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, reinterpret_cast<uint8_t*>(pData), pDataWeight, slope);
            }
            else if (sizeof(T) == sizeof(char))
            {
                kCalculateLRELUOutputDelta_kernel<char><<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, reinterpret_cast<char*>(pData), pDataWeight, slope);
            }
            else
            {
                kCalculateLRELUOutputDelta_kernel<T><<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight, slope);
            }
            LAUNCHERROR("kCalculateLRELUOutputDelta_kernel");
            break;

        case ExponentialLinear:
            if (sizeof(T) == sizeof(unsigned char))
            {
                kCalculateELUOutputDelta_kernel<uint8_t><<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, reinterpret_cast<uint8_t*>(pData), pDataWeight, alpha);
            }
            else if (sizeof(T) == sizeof(char))
            {
                kCalculateELUOutputDelta_kernel<char><<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, reinterpret_cast<char*>(pData), pDataWeight, alpha);
            }
            else
            {
                kCalculateELUOutputDelta_kernel<T><<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight, alpha);
            }
            LAUNCHERROR("kCalculateELUOutputDelta_kernel");
            break;

        case ScaledExponentialLinear:
            if (sizeof(T) == sizeof(unsigned char))
            {
                kCalculateSELUOutputDelta_kernel<uint8_t><<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, reinterpret_cast<uint8_t*>(pData), pDataWeight, alpha, lambda);
            }
            else if (sizeof(T) == sizeof(char))
            {
                kCalculateSELUOutputDelta_kernel<char><<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, reinterpret_cast<char*>(pData), pDataWeight, alpha, lambda);
            }
            else
            {
                kCalculateSELUOutputDelta_kernel<T><<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight, alpha, lambda);
            }
            LAUNCHERROR("kCalculateSELUOutputDelta_kernel");
            break;

        case SoftMax:
            if (sizeof(T) == sizeof(unsigned char))
            {
                kCalculateSoftMaxOutputDelta_kernel<uint8_t><<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, reinterpret_cast<uint8_t*>(pData), pDataWeight);
            }
            else if (sizeof(T) == sizeof(char))
            {
                kCalculateSoftMaxOutputDelta_kernel<char><<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, reinterpret_cast<char*>(pData), pDataWeight);
            }
            else
            {
                kCalculateSoftMaxOutputDelta_kernel<T><<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            }
            LAUNCHERROR("kCalculateSoftMaxOutputDelta_kernel");
            break;
    }
}
template <typename T>
__global__ void kCalculateIndexedSigmoidOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
    Float* pDelta, uint32_t* pIndex, T* pData, Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos =
            pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (1.0f / 256.0f);
        pDelta[uOffset + pos] = w * (a - t) * a * (1.0f - a);
    }
}

template <>
__global__ void kCalculateIndexedSigmoidOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
    Float* pDelta, uint32_t* pIndex, unsigned char* pData, Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos =
            pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (1.0f / 256.0f);
        pDelta[uOffset + pos] = w * (a - t) * a * (1.0f - a);
    }
}

template <>
__global__ void kCalculateIndexedSigmoidOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
    Float* pDelta, uint32_t* pIndex, char* pData, Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos =
            pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (1.0f / 128.0f);
        pDelta[uOffset + pos] = w * (a - t) * a * (1.0f - a);
    }
}

template <typename T>
__global__ void kCalculateIndexedTanhOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
    Float* pDelta, uint32_t* pIndex, T* pData, Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos =
            pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        Float a = pUnit[uOffset + pos];
        Float t = pData[dOffset + pos];
        pDelta[uOffset + pos] = w * (a - t) * (1.0f - a * a);
    }
}

template <>
__global__ void kCalculateIndexedTanhOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
    Float* pDelta, uint32_t* pIndex, unsigned char* pData, Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos =
            pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (1.0f / 256.0f);
        pDelta[uOffset + pos] = w * (a - t) * (1.0f - a * a);
    }
}
template <>
__global__ void kCalculateIndexedTanhOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
    Float* pDelta, uint32_t* pIndex, char* pData, Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos =
            pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (1.0f / 128.0f);
        pDelta[uOffset + pos] = w * (a - t) * (1.0f - a * a);
    }
}

template <typename T>
__global__ void kCalculateIndexedLinearOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
    Float* pDelta, uint32_t* pIndex, T* pData, Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos =
            pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        Float a = pUnit[uOffset + pos];
        Float t = pData[dOffset + pos];
        pDelta[uOffset + pos] = w * (a - t);
    }
}

template <>
__global__ void kCalculateIndexedLinearOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
    Float* pDelta, uint32_t* pIndex, unsigned char* pData, Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos =
            pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (1.0f / 256.0f);
        pDelta[uOffset + pos] = w * (a - t);
    }
}

template <>
__global__ void kCalculateIndexedLinearOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
    Float* pDelta, uint32_t* pIndex, char* pData, Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos =
            pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (1.0f / 128.0f);
        pDelta[uOffset + pos] = w * (a - t);
    }
}

template <typename T>
__global__ void kCalculateIndexedRELUOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
    Float* pDelta, uint32_t* pIndex, T* pData, Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos =
            pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        Float a = pUnit[uOffset + pos];
        Float t = pData[dOffset + pos];
        pDelta[uOffset + pos] = w * (a - t) * (a > 0.0f);
    }
}

template <>
__global__ void kCalculateIndexedRELUOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
    Float* pDelta, uint32_t* pIndex, unsigned char* pData, Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos =
            pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (1.0f / 256.0f);
        pDelta[uOffset + pos] = w * (a - t) * (a > 0.0f);
    }
}

template <>
__global__ void kCalculateIndexedRELUOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
    Float* pDelta, uint32_t* pIndex, char* pData, Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos =
            pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (1.0f / 128.0f);
        pDelta[uOffset + pos] = w * (a - t) * (a > 0.0f);
    }
}

template <typename T>
__global__ void kCalculateIndexedLRELUOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
    Float* pDelta, uint32_t* pIndex, T* pData, Float* pDataWeight, Float slope)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos =
            pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        Float a = pUnit[uOffset + pos];
        Float t = pData[dOffset + pos];
        pDelta[uOffset + pos] = w * (a - t) * ((a > 0.0f) + (a <= 0.0f) * slope);
    }
}

template <>
__global__ void kCalculateIndexedLRELUOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
    Float* pDelta, uint32_t* pIndex, unsigned char* pData, Float* pDataWeight, Float slope)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos =
            pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (1.0f / 256.0f);
        pDelta[uOffset + pos] = w * (a - t) * ((a > 0.0f) + (a <= 0.0f) * slope);
    }
}

template <>
__global__ void kCalculateIndexedLRELUOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
    Float* pDelta, uint32_t* pIndex, char* pData, Float* pDataWeight, Float slope)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos =
            pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (1.0f / 128.0f);
        pDelta[uOffset + pos] = w * (a - t) * ((a > 0.0f) + (a <= 0.0f) * slope);
    }
}
template <typename T>
__global__ void kCalculateIndexedELUOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
    Float* pDelta, uint32_t* pIndex, T* pData, Float* pDataWeight, Float alpha)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos =
            pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        Float a = pUnit[uOffset + pos];
        Float t = pData[dOffset + pos];
        pDelta[uOffset + pos] = w * (a - t) * ((a >= 0.0f) + (a < 0.0f) * (a + alpha));
    }
}

template <>
__global__ void kCalculateIndexedELUOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
    Float* pDelta, uint32_t* pIndex, unsigned char* pData, Float* pDataWeight, Float alpha)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos =
            pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (1.0f / 256.0f);
        pDelta[uOffset + pos] = w * (a - t) * ((a >= 0.0f) + (a < 0.0f) * (a + alpha));
    }
}

template <>
__global__ void kCalculateIndexedELUOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
    Float* pDelta, uint32_t* pIndex, char* pData, Float* pDataWeight, Float alpha)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos =
            pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (1.0f / 128.0f);
        pDelta[uOffset + pos] = w * (a - t) * ((a >= 0.0f) + (a < 0.0f) * (a + alpha));
    }
}

template <typename T>
__global__ void kCalculateIndexedSELUOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
    Float* pDelta, uint32_t* pIndex, T* pData, Float* pDataWeight, Float alpha, Float lambda)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos =
            pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        Float a = pUnit[uOffset + pos];
        Float t = pData[dOffset + pos];
        pDelta[uOffset + pos] = w * (a - t) * ((a >= 0.0f) * lambda + (a < 0.0f) * (lambda * alpha * expf(a)));
    }
}

template <>
__global__ void kCalculateIndexedSELUOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
    Float* pDelta, uint32_t* pIndex, unsigned char* pData, Float* pDataWeight, Float alpha, Float lambda)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos =
            pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (1.0f / 256.0f);
        pDelta[uOffset + pos] = w * (a - t) * ((a >= 0.0f) * lambda + (a < 0.0f) * (lambda * alpha * expf(a)));
    }
}

template <>
__global__ void kCalculateIndexedSELUOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
    Float* pDelta, uint32_t* pIndex, char* pData, Float* pDataWeight, Float alpha, Float lambda)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos =
            pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (1.0f / 128.0f);
        pDelta[uOffset + pos] = w * (a - t) * ((a >= 0.0f) * lambda + (a < 0.0f) * (lambda * alpha * expf(a)));
    }
}

template <typename T>
__global__ void kCalculateIndexedSoftMaxOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
    Float* pDelta, uint32_t* pIndex, T* pData, Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos =
            pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        Float a = pUnit[uOffset + pos];
        Float t = pData[dOffset + pos];
        pDelta[uOffset + pos] = w * (a - t);
    }
}

template <>
__global__ void kCalculateIndexedSoftMaxOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
    Float* pDelta, uint32_t* pIndex, unsigned char* pData, Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos =
            pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (1.0f / 256.0f);
        pDelta[uOffset + pos] = w * (a - t);
    }
}

template <>
__global__ void kCalculateIndexedSoftMaxOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
    Float* pDelta, uint32_t* pIndex, char* pData, Float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos =
            pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (1.0f / 128.0f);
        pDelta[uOffset + pos] = w * (a - t);
    }
}
template <typename T>
void kCalculateIndexedOutputDelta(
    Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
    Float* pDelta, uint32_t* pIndex, T* pData, Float* pDataWeight, Float slope, Float alpha, Float lambda)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    switch (activation)
    {
        case Sigmoid:
            kCalculateIndexedSigmoidOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(
                position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            LAUNCHERROR("kCalculateIndexedSigmoidOutputDelta_kernel");
            break;

        case Tanh:
            kCalculateIndexedTanhOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(
                position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            LAUNCHERROR("kCalculateIndexedTanhOutputDelta_kernel");
            break;

        case Linear:
            kCalculateIndexedLinearOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(
                position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            LAUNCHERROR("kCalculateIndexedLinearOutputDelta_kernel");
            break;

        case RectifiedLinear:
            kCalculateIndexedRELUOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(
                position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            LAUNCHERROR("kCalculateIndexedRELUOutputDelta_kernel");
            break;

        case LeakyRectifiedLinear:
            kCalculateIndexedLRELUOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(
                position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight, slope);
            LAUNCHERROR("kCalculateIndexedLRELUOutputDelta_kernel");
            break;

        case ExponentialLinear:
            kCalculateIndexedELUOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(
                position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight, alpha);
            LAUNCHERROR("kCalculateIndexedELUOutputDelta_kernel");
            break;

        case ScaledExponentialLinear:
            kCalculateIndexedSELUOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(
                position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight, alpha, lambda);
            LAUNCHERROR("kCalculateIndexedSELUOutputDelta_kernel");
            break;

        case SoftMax:
            kCalculateIndexedSoftMaxOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(
                position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            LAUNCHERROR("kCalculateIndexedSoftMaxOutputDelta_kernel");
            break;
    }
}
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSigmoidL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, T* pData, Float* pDataWeight)
{
    constexpr Float One = static_cast<Float>(1.0);
    constexpr Float Zero = static_cast<Float>(0.0);

    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : One;
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (One / static_cast<Float>(sizeof(T) * 8));
        Float diff = a - fabsf(t);
        diff = (t > Zero) ? min(Zero, diff) : max(Zero, diff);
        pDelta[uOffset + pos] = w * diff * a * (One - a);
    }
}

template<>
__global__ void LAUNCH_BOUNDS() kCalculateSigmoidL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, unsigned char* pData, Float* pDataWeight)
{
    constexpr Float One = static_cast<Float>(1.0);
    constexpr Float Zero = static_cast<Float>(0.0);

    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : One;
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (One / static_cast<Float>(256.0));
        Float diff = a - fabsf(t);
        diff = (t > Zero) ? min(Zero, diff) : max(Zero, diff);
        pDelta[uOffset + pos] = w * diff * a * (One - a);
    }
}

template<>
__global__ void LAUNCH_BOUNDS() kCalculateSigmoidL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, char* pData, Float* pDataWeight)
{
    constexpr Float One = static_cast<Float>(1.0);
    constexpr Float Zero = static_cast<Float>(0.0);

    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : One;
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (One / static_cast<Float>(128.0));
        Float diff = a - fabsf(t);
        diff = (t > Zero) ? min(Zero, diff) : max(Zero, diff);
        pDelta[uOffset + pos] = w * diff * a * (One - a);
    }
}

template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateTanhL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, T* pData, Float* pDataWeight)
{
    constexpr Float One = static_cast<Float>(1.0);
    constexpr Float Zero = static_cast<Float>(0.0);

    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : One;
        Float a = pUnit[uOffset + pos];
        Float t = pData[dOffset + pos];
        Float diff = a - fabsf(t);
        diff = (t > Zero) ? min(Zero, diff) : max(Zero, diff);
        pDelta[uOffset + pos] = w * diff * (One - a * a);
    }
}

template<>
__global__ void LAUNCH_BOUNDS() kCalculateTanhL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, unsigned char* pData, Float* pDataWeight)
{
    constexpr Float One = static_cast<Float>(1.0);
    constexpr Float Zero = static_cast<Float>(0.0);

    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : One;
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (One / static_cast<Float>(256.0));
        Float diff = a - fabsf(t);
        diff = (t > Zero) ? min(Zero, diff) : max(Zero, diff);
        pDelta[uOffset + pos] = w * diff * (One - a * a);
    }
}

template<>
__global__ void LAUNCH_BOUNDS() kCalculateTanhL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, char* pData, Float* pDataWeight)
{
    constexpr Float One = static_cast<Float>(1.0);
    constexpr Float Zero = static_cast<Float>(0.0);

    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : One;
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (One / static_cast<Float>(128.0));
        Float diff = a - fabsf(t);
        diff = (t > Zero) ? min(Zero, diff) : max(Zero, diff);
        pDelta[uOffset + pos] = w * diff * (One - a * a);
    }
}

template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateLinearL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, T* pData, Float* pDataWeight)
{
    constexpr Float Zero = static_cast<Float>(0.0);

    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<Float>(1.0);
        Float a = pUnit[uOffset + pos];
        Float t = pData[dOffset + pos];
        Float diff = a - fabsf(t);
        diff = (t > Zero) ? min(Zero, diff) : max(Zero, diff);
        pDelta[uOffset + pos] = w * diff;
    }
}

template<>
__global__ void LAUNCH_BOUNDS() kCalculateLinearL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, unsigned char* pData, Float* pDataWeight)
{
    constexpr Float Zero = static_cast<Float>(0.0);

    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<Float>(1.0);
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (static_cast<Float>(1.0) / static_cast<Float>(256.0));
        Float diff = a - fabsf(t);
        diff = (t > Zero) ? min(Zero, diff) : max(Zero, diff);
        pDelta[uOffset + pos] = w * diff;
    }
}

template<>
__global__ void LAUNCH_BOUNDS() kCalculateLinearL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, char* pData, Float* pDataWeight)
{
    constexpr Float Zero = static_cast<Float>(0.0);

    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<Float>(1.0);
        Float a = pUnit[uOffset + pos];
        Float t = static_cast<Float>(pData[dOffset + pos]) * (static_cast<Float>(1.0) / static_cast<Float>(128.0));
        Float diff = a - fabsf(t);
        diff = (t > Zero) ? min(Zero, diff) : max(Zero, diff);
        pDelta[uOffset + pos] = w * diff;
    }
}

