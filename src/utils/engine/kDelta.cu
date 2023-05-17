#include "GpuTypes.h"
#include "Types.h"
#include <limits>

static __constant__ GpuData cData;

/**
 * @brief Copies the data from host to device symbol.
 */
void SetKDeltaGpuData()
{
    cudaError_t status = cudaMemcpyToSymbol(cData, &(getGpu()._data), sizeof(GpuData));
    if (status != cudaSuccess)
    {
        printf("cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(status));
    }
}

/**
 * @brief Copies the data from device symbol to host.
 */
void GetKDeltaGpuData()
{
    cudaError_t status = cudaMemcpyFromSymbol(&(getGpu()._data), cData, sizeof(GpuData));
    if (status != cudaSuccess)
    {
        printf("cudaMemcpyFromSymbol failed: %s\n", cudaGetErrorString(status));
    }
}
/**
 * @brief Calculates the sigmoid output delta for a given position, batch, and stride.
 *
 * @tparam T The data type for pData.
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<typename T>
__global__ void kCalculateSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat* pDataWeight)
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
        pDelta[uOffset + pos] = w * (a - t) * a * (static_cast<NNFloat>(1.0) - a);
    }
}
/**
 * @brief Calculates the sigmoid output delta for a given position, batch, and stride, specialized for uint8_t data type.
 *
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<>
__global__ void kCalculateSigmoidOutputDelta_kernel<uint8_t>(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint8_t* pData, NNFloat* pDataWeight)
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
        pDelta[uOffset + pos] = w * (a - t) * a * (static_cast<NNFloat>(1.0) - a);
    }
}
/**
 * @brief Calculates the sigmoid output delta for a given position, batch, and stride, specialized for char data type.
 *
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<>
__global__ void kCalculateSigmoidOutputDelta_kernel<char>(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData, NNFloat* pDataWeight)
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
        pDelta[uOffset + pos] = w * (a - t) * a * (static_cast<NNFloat>(1.0) - a);
    }
}
/**
 * @brief Calculates the hyperbolic tangent output delta for a given position, batch, and stride.
 *
 * @tparam T The data type for pData.
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<typename T>
__global__ void kCalculateTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]);
        pDelta[uOffset + pos] = w * (a - t) * (static_cast<NNFloat>(1.0) - a * a);
    }
}
/**
 * @brief Calculates the hyperbolic tangent output delta for a given position, batch, and stride, specialized for uint8_t data type.
 *
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<>
__global__ void kCalculateTanhOutputDelta_kernel<uint8_t>(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint8_t* pData, NNFloat* pDataWeight)
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
        pDelta[uOffset + pos] = w * (a - t) * (static_cast<NNFloat>(1.0) - a * a);
    }
}
/**
 * @brief Calculates the hyperbolic tangent output delta for a given position, batch, and stride, specialized for char data type.
 *
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<>
__global__ void kCalculateTanhOutputDelta_kernel<char>(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData, NNFloat* pDataWeight)
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
        pDelta[uOffset + pos] = w * (a - t) * (static_cast<NNFloat>(1.0) - a * a);
    }
}
/**
 * @brief Calculates the linear output delta for a given position, batch, and stride.
 *
 * @tparam T The data type for pData.
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<typename T>
__global__ void kCalculateLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]);
        pDelta[uOffset + pos] = w * (a - t);
    }
}
/**
 * @brief Calculates the linear output delta for a given position, batch, and stride, specialized for uint8_t data type.
 *
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<>
__global__ void kCalculateLinearOutputDelta_kernel<uint8_t>(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint8_t* pData, NNFloat* pDataWeight)
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
/**
 * @brief Calculates the linear output delta for a given position, batch, and stride, specialized for char data type.
 *
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<>
__global__ void kCalculateLinearOutputDelta_kernel<char>(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData, NNFloat* pDataWeight)
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
        pDelta[uOffset + pos] = w * (a - t);
    }
}
/**
 * @brief Calculates the rectified linear unit (ReLU) output delta for a given position, batch, and stride.
 *
 * @tparam T The data type for pData.
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<typename T>
__global__ void kCalculateRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]);
        pDelta[uOffset + pos] = w * (a - t) * (a > static_cast<NNFloat>(0.0));
    }
}
/**
 * @brief Calculates the rectified linear unit (ReLU) output delta for a given position, batch, and stride, specialized for uint8_t data type.
 *
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<>
__global__ void kCalculateRELUOutputDelta_kernel<uint8_t>(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint8_t* pData, NNFloat* pDataWeight)
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
        pDelta[uOffset + pos] = w * (a - t) * (a > static_cast<NNFloat>(0.0));
    }
}
/**
 * @brief Calculates the rectified linear unit (ReLU) output delta for a given position, batch, and stride, specialized for char data type.
 *
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<>
__global__ void kCalculateRELUOutputDelta_kernel<char>(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData, NNFloat* pDataWeight)
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
/**
 * @brief Calculates the leaky rectified linear unit (LReLU) output delta for a given position, batch, and stride.
 *
 * @tparam T The data type for pData.
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 * @param slope The slope value for the LReLU activation function.
 */
template<typename T>
__global__ void kCalculateLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat* pDataWeight, NNFloat slope)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]);
        pDelta[uOffset + pos] = w * (a - t) * ((a > static_cast<NNFloat>(0.0)) + (a <= static_cast<NNFloat>(0.0)) * slope);
    }
}
/**
 * @brief Calculates the leaky rectified linear unit (LReLU) output delta for a given position, batch, and stride, specialized for uint8_t data type.
 *
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 * @param slope The slope value for the LReLU activation function.
 */
template<>
__global__ void kCalculateLRELUOutputDelta_kernel<uint8_t>(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint8_t* pData, NNFloat* pDataWeight, NNFloat slope)
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
        pDelta[uOffset + pos] = w * (a - t) * ((a > static_cast<NNFloat>(0.0)) + (a <= static_cast<NNFloat>(0.0)) * slope);
    }
}
/**
 * @brief Calculates the leaky rectified linear unit (LReLU) output delta for a given position, batch, and stride, specialized for char data type.
 *
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 * @param slope The slope value for the LReLU activation function.
 */
template<>
__global__ void kCalculateLRELUOutputDelta_kernel<char>(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData, NNFloat* pDataWeight, NNFloat slope)
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
        pDelta[uOffset + pos] = w * (a - t) * ((a > static_cast<NNFloat>(0.0)) + (a <= static_cast<NNFloat>(0.0)) * slope);
    }
}
/**
 * @brief Calculates the exponential linear unit (ELU) output delta for a given position, batch, and stride.
 *
 * @tparam T The data type for the pData array.
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 * @param alpha The alpha value for the ELU activation function.
 */
template<typename T>
__global__ void kCalculateELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat* pDataWeight, NNFloat alpha)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]);
        pDelta[uOffset + pos] = w * (a - t) * ((a >= static_cast<NNFloat>(0.0)) + (a < static_cast<NNFloat>(0.0)) * (a + alpha));
    }
}
/**
 * @brief Calculates the exponential linear unit (ELU) output delta for a given position, batch, and stride, specialized for unsigned char data type.
 *
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 * @param alpha The alpha value for the ELU activation function.
 */
template<>
__global__ void kCalculateELUOutputDelta_kernel<unsigned char>(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData, NNFloat* pDataWeight, NNFloat alpha)
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
        pDelta[uOffset + pos] = w * (a - t) * ((a >= static_cast<NNFloat>(0.0)) + (a < static_cast<NNFloat>(0.0)) * (a + alpha));
    }
}
/**
 * @brief Calculates the exponential linear unit (ELU) output delta for a given position, batch, and stride, specialized for char data type.
 *
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 * @param alpha The alpha value for the ELU activation function.
 */
template<>
__global__ void kCalculateELUOutputDelta_kernel<char>(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData, NNFloat* pDataWeight, NNFloat alpha)
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
/**
 * @brief Calculates the scaled exponential linear unit (SELU) output delta for a given position, batch, and stride.
 *
 * @tparam T The data type of the `pData` array.
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 * @param alpha The alpha value for the SELU activation function.
 * @param lambda The lambda value for the SELU activation function.
 */
template<typename T>
__global__ void kCalculateSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat* pDataWeight, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]);
        pDelta[uOffset + pos] = w * (a - t) * ((a >= static_cast<NNFloat>(0.0)) * lambda + (a < static_cast<NNFloat>(0.0)) * (lambda * alpha * exp(a)));
    }
}
/**
 * @brief Calculates the scaled exponential linear unit (SELU) output delta for a given position, batch, and stride.
 *
 * @tparam T The data type of the `pData` array.
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 * @param alpha The alpha value for the SELU activation function.
 * @param lambda The lambda value for the SELU activation function.
 */
template<>
__global__ void kCalculateSELUOutputDelta_kernel<uint8_t>(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint8_t* pData, NNFloat* pDataWeight, NNFloat alpha, NNFloat lambda)
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
        pDelta[uOffset + pos] = w * (a - t) * ((a >= static_cast<NNFloat>(0.0)) * lambda + (a < static_cast<NNFloat>(0.0)) * (lambda * alpha * exp(a)));
    }
}
/**
 * @brief Calculates the softmax output delta for a given position, batch, and stride.
 *
 * @tparam T The data type of the pData array.
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<typename T>
__global__ void kCalculateSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]);
        pDelta[uOffset + pos] = w * (a - t);
    }
}
/**
 * @brief Calculates the softmax output delta for a given position, batch, and stride.
 *
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<>
__global__ void kCalculateSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * static_cast<NNFloat>(1.0 / 256.0);
        pDelta[uOffset + pos] = w * (a - t);
    }
}
/**
 * @brief Calculates the softmax output delta for a given position, batch, and stride.
 *
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<>
__global__ void kCalculateSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * static_cast<NNFloat>(1.0 / 128.0);
        pDelta[uOffset + pos] = w * (a - t);
    }
}
/**
 * @brief Calculates the output delta based on the specified activation function.
 *
 * @tparam T The data type.
 * @param activation The activation function.
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 * @param slope The slope value (used for Leaky Rectified Linear activation).
 * @param alpha The alpha value (used for Exponential Linear activation).
 * @param lambda The lambda value (used for Scaled Exponential Linear activation).
 */
template<typename T>
void kCalculateOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat* pDataWeight, NNFloat slope, NNFloat alpha, NNFloat lambda)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);

    switch (activation)
    {
        /**
         * @brief Calculates the output delta with Sigmoid activation.
         *
         * @param position The position value.
         * @param batch The batch value.
         * @param stride The stride value.
         * @param pUnit Pointer to the unit array.
         * @param pDelta Pointer to the delta array.
         * @param pData Pointer to the data array.
         * @param pDataWeight Pointer to the data weight array.
         */
        case Sigmoid:
            kCalculateSigmoidOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            LAUNCHERROR("kCalculateSigmoidOutputDelta_kernel");
            break;

        /**
         * @brief Calculates the output delta with Tanh activation.
         *
         * @param position The position value.
         * @param batch The batch value.
         * @param stride The stride value.
         * @param pUnit Pointer to the unit array.
         * @param pDelta Pointer to the delta array.
         * @param pData Pointer to the data array.
         * @param pDataWeight Pointer to the data weight array.
         */
        case Tanh:
            kCalculateTanhOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            LAUNCHERROR("kCalculateTanhOutputDelta_kernel");
            break;

        /**
         * @brief Calculates the output delta with Linear activation.
         *
         * @param position The position value.
         * @param batch The batch value.
         * @param stride The stride value.
         * @param pUnit Pointer to the unit array.
         * @param pDelta Pointer to the delta array.
         * @param pData Pointer to the data array.
         * @param pDataWeight Pointer to the data weight array.
         */
        case Linear:
            kCalculateLinearOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            LAUNCHERROR("kCalculateLinearOutputDelta_kernel");
            break;

        /**
         * @brief Calculates the output delta with Rectified Linear Unit (ReLU) activation.
         *
         * @param position The position value.
         * @param batch The batch value.
         * @param stride The stride value.
         * @param pUnit Pointer to the unit array.
         * @param pDelta Pointer to the delta array.
         * @param pData Pointer to the data array.
         * @param pDataWeight Pointer to the data weight array.
         */
        case RectifiedLinear:
            kCalculateRELUOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            LAUNCHERROR("kCalculateRELUOutputDelta_kernel");
            break;
            
        /**
         * @brief Calculates the output delta with Leaky Rectified Linear Unit (LReLU) activation.
         *
         * @param position The position value.
         * @param batch The batch value.
         * @param stride The stride value.
         * @param pUnit Pointer to the unit array.
         * @param pDelta Pointer to the delta array.
         * @param pData Pointer to the data array.
         * @param pDataWeight Pointer to the data weight array.
         * @param slope The slope value.
         */
        case LeakyRectifiedLinear:
            kCalculateLRELUOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight, slope);
            LAUNCHERROR("kCalculateLRELUOutputDelta_kernel");
            break;

        /**
         * @brief Calculates the output delta with Exponential Linear Unit (ELU) activation.
         *
         * @param position The position value.
         * @param batch The batch value.
         * @param stride The stride value.
         * @param pUnit Pointer to the unit array.
         * @param pDelta Pointer to the delta array.
         * @param pData Pointer to the data array.
         * @param pDataWeight Pointer to the data weight array.
         * @param alpha The alpha value.
         */
        case ExponentialLinear:
            kCalculateELUOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight, alpha);
            LAUNCHERROR("kCalculateELUOutputDelta_kernel");
            break;

        /**
         * @brief Calculates the output delta with SELU activation for indexed data.
         *
         * @param position The position value.
         * @param batch The batch value.
         * @param stride The stride value.
         * @param pUnit Pointer to the unit array.
         * @param pDelta Pointer to the delta array.
         * @param pData Pointer to the data array.
         * @param pDataWeight Pointer to the data weight array.
         * @param alpha The alpha value.
         * @param lambda The lambda value.
         */
        case ScaledExponentialLinear:
            kCalculateSELUOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight, alpha, lambda);
            LAUNCHERROR("kCalculateSELUOutputDelta_kernel");
            break;

        /**
         * @brief Calculates the output delta with SoftMax activation for indexed data.
         *
         * @param position The position value.
         * @param batch The batch value.
         * @param stride The stride value.
         * @param pUnit Pointer to the unit array.
         * @param pDelta Pointer to the delta array.
         * @param pData Pointer to the data array.
         * @param pDataWeight Pointer to the data weight array.
         */
        case SoftMax:
            kCalculateSoftMaxOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            LAUNCHERROR("kCalculateSoftMaxOutputDelta_kernel");
            break;                
    }
}
/**
 * @brief Calculates the output delta with sigmoid activation for indexed data.
 *
 * @tparam T The data type.
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<typename T>
__global__ void kCalculateIndexedSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : (NNFloat)1.0;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = pData[dOffset + pos];
        pDelta[uOffset + pos] = w * (a - t) * a * ((NNFloat)1.0 - a);
    }
}

/**
 * @brief Calculates the output delta with sigmoid activation for indexed data.
 *
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<>
__global__ void kCalculateIndexedSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : (NNFloat)1.0;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 256.0);
        pDelta[uOffset + pos] = w * (a - t) * a * ((NNFloat)1.0 - a);
    }
}
/**
 * @brief CUDA kernel to calculate the output delta for indexed tanh activation function.
 *
 * @tparam T The data type (unused in this implementation).
 * @param position The position index.
 * @param batch The batch size.
 * @param stride The stride size.
 * @param pUnit Pointer to the unit data.
 * @param pDelta Pointer to the delta data.
 * @param pIndex Pointer to the index data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight (optional).
 */
template<typename T>
__global__ void kCalculateIndexedTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData, NNFloat* pDataWeight)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride + pos;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride + pos;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0;
        NNFloat a = pUnit[uOffset];
        NNFloat t = pData[dOffset];
        pDelta[uOffset] = w * (a - t) * (1.0 - a * a);
    }
}

/**
 * @brief CUDA kernel to calculate the output delta for indexed tanh activation function.
 *
 * @param position The position index.
 * @param batch The batch size.
 * @param stride The stride size.
 * @param pUnit Pointer to the unit data.
 * @param pDelta Pointer to the delta data.
 * @param pIndex Pointer to the index data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight (optional).
 */
__global__ void kCalculateIndexedTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        NNFloat w = pDataWeight ? pDataWeight[dpos] : 1.0;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = (NNFloat)pData[dOffset + pos];
        pDelta[uOffset + pos] = w * (a - t) * (1.0 - a * a);
    }
}

/**
 * @brief CUDA kernel to calculate the indexed tanh output delta.
 *
 * @param position The starting position of the data.
 * @param batch The batch size.
 * @param stride The stride of the data.
 * @param pUnit Pointer to the unit data.
 * @param pDelta Pointer to the delta data.
 * @param pIndex Pointer to the index data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight (optional).
 */
__global__ void LAUNCH_BOUNDS() kCalculateIndexedTanhOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride,
    NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex,
    const char* pData, const NNFloat* pDataWeight)
{
    int pos = threadIdx.x + blockIdx.x * blockDim.x;
    if (pos < stride)
    {
        int uOffset = blockIdx.x * stride;
        int dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        int dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : (NNFloat)1.0;

        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 128.0);

        pDelta[uOffset + pos] = w * (a - t) * ((NNFloat)1.0 - a * a);
    }
}
/**
 * @brief CUDA kernel to calculate indexed linear output delta.
 *
 * @param position The starting position in the data arrays.
 * @param batch The number of batches.
 * @param stride The stride size.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array (optional).
 */
__global__ void kCalculateIndexedLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, NNFloat* pData, NNFloat* pDataWeight)
{
    uint64_t pos = threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = pData[dOffset + pos];
        pDelta[uOffset + pos] = w * (a - t);
    }
}
/**
 * @brief CUDA kernel to calculate indexed linear output delta.
 *
 * @param position The starting position in the data arrays.
 * @param batch The number of batches.
 * @param stride The stride size.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array (optional).
 */
__global__ void kCalculateIndexedLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]);
        pDelta[uOffset + pos] = w * (a - t);
    }
}

/**
 * @brief CUDA kernel to calculate indexed linear output delta.
 *
 * @tparam UnusedTemplate Unused template parameter.
 * @param position The position parameter.
 * @param batch The batch parameter.
 * @param stride The stride parameter.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t dOffset = dpos * stride;
    NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : (NNFloat)1.0;

    // Load pDataWeight into shared memory (if needed)
    // __shared__ NNFloat sharedDataWeight[BLOCK_SIZE];
    // sharedDataWeight[threadIdx.x] = pDataWeight[dpos];
    // __syncthreads();

    if (pos < stride)
    {
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = pData[dOffset + pos] / 128.0;
        pDelta[uOffset + pos] = w * (a - t);
    }
}
/**
 * @brief CUDA kernel to calculate indexed RELU output delta.
 *
 * @param position The position parameter.
 * @param batch The batch parameter.
 * @param stride The stride parameter.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 */
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, NNFloat* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t dOffset = dpos * stride;
    NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : (NNFloat)1.0;

    if (pos < stride)
    {
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = pData[dOffset + pos];
        pDelta[uOffset + pos] = w * (a - t) * (a > (NNFloat)0.0);
    }
}
/**
 * @brief CUDA kernel to calculate the output delta for indexed RELU activation.
 *
 * @tparam Unused Template parameter (not used).
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit array.
 * @param pDelta Pointer to the delta array.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedRELUOutputDelta_kernel(size_t position, size_t batch, size_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData, NNFloat* pDataWeight)
{
    size_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        size_t uOffset = blockIdx.x * stride;
        size_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        size_t dOffset = dpos * stride;

        NNFloat w = pDataWeight[dpos] ?? 1.0;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (NNFloat)(1.0 / 256.0);

        pDelta[uOffset + pos] = w * (a - t) * (a > (NNFloat)0.0);
    }
}

/**
 * @brief CUDA kernel function for calculating indexed LRELU output delta.
 *
 * @tparam T The data type of pData.
 * @param position The position index used for indexing certain data arrays.
 * @param batch The batch size.
 * @param stride The stride size.
 * @param pUnit A pointer to an array of NNFloat values representing the input units.
 * @param pDelta A pointer to an array of NNFloat values representing the delta values (output of the previous layer).
 * @param pIndex A pointer to an array of uint32_t values representing indices.
 * @param pData A pointer to an array of T values representing the data.
 * @param pDataWeight A pointer to an array of NNFloat values representing the data weights.
 * @param slope The slope value for the LRELU activation function.
 */
template<typename T>
__global__ void kCalculateIndexedLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData, NNFloat* pDataWeight, NNFloat slope)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;

    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]);

        pDelta[uOffset + pos] = w * (a - t) * ((a > static_cast<NNFloat>(0.0)) + (a <= static_cast<NNFloat>(0.0)) * slope);
    }
}
/**
 * @brief CUDA kernel function for calculating indexed LRELU output delta.
 *
 * @param position The position index used for indexing certain data arrays.
 * @param batch The batch size.
 * @param stride The stride size.
 * @param pUnit A pointer to an array of NNFloat values representing the input units.
 * @param pDelta A pointer to an array of NNFloat values representing the delta values (output of the previous layer).
 * @param pIndex A pointer to an array of uint32_t values representing indices.
 * @param pData A pointer to an array of unsigned char values representing the data.
 * @param pDataWeight A pointer to an array of NNFloat values representing the data weights.
 * @param slope The slope value for the LRELU activation function.
 */
template<>
__global__ void kCalculateIndexedLRELUOutputDelta_kernel<uint8_t>(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint8_t* pData, NNFloat* pDataWeight, NNFloat slope)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;

    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (1.0 / 256.0);

        pDelta[uOffset + pos] = w * (a - t) * ((a > static_cast<NNFloat>(0.0)) + (a <= static_cast<NNFloat>(0.0)) * slope);
    }
}
/**
 * @brief CUDA kernel function for calculating indexed LRELU output delta.
 *
 * @param position The position index used for indexing certain data arrays.
 * @param batch The batch size.
 * @param stride The stride size.
 * @param pUnit A pointer to an array of NNFloat values representing the input units.
 * @param pDelta A pointer to an array of NNFloat values representing the delta values (output of the previous layer).
 * @param pIndex A pointer to an array of uint32_t values representing indices.
 * @param pData A pointer to an array of char values representing the data.
 * @param pDataWeight A pointer to an array of NNFloat values representing the data weights.
 * @param slope The slope value for the LRELU activation function.
 */
template<>
__global__ void kCalculateIndexedLRELUOutputDelta_kernel<char>(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData, NNFloat* pDataWeight, NNFloat slope)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;

    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (1.0 / 128.0);

        pDelta[uOffset + pos] = w * (a - t) * ((a > static_cast<NNFloat>(0.0)) + (a <= static_cast<NNFloat>(0.0)) * slope);
    }
}
/**
 * @brief CUDA kernel function for calculating indexed ELU output delta.
 *
 * @tparam T The data type of pData.
 * @param position The position index used for indexing certain data arrays.
 * @param batch The batch size.
 * @param stride The stride size.
 * @param pUnit A pointer to an array of NNFloat values representing the input units.
 * @param pDelta A pointer to an array of NNFloat values representing the delta values (output of the previous layer).
 * @param pIndex A pointer to an array of uint32_t values representing indices.
 * @param pData A pointer to an array of T values representing the data.
 * @param pDataWeight A pointer to an array of NNFloat values representing the data weights.
 * @param alpha The alpha value for the ELU activation function.
 */
template<typename T>
__global__ void kCalculateIndexedELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData, NNFloat* pDataWeight, NNFloat alpha)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;

    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]);

        pDelta[uOffset + pos] = w * (a - t) * ((a >= static_cast<NNFloat>(0.0)) + (a < static_cast<NNFloat>(0.0)) * (a + alpha));
    }
}
/**
 * @brief CUDA kernel function for calculating indexed ELU output delta.
 *
 * @param position The position index used for indexing certain data arrays.
 * @param batch The batch size.
 * @param stride The stride size.
 * @param pUnit A pointer to an array of NNFloat values representing the input units.
 * @param pDelta A pointer to an array of NNFloat values representing the delta values (output of the previous layer).
 * @param pIndex A pointer to an array of uint32_t values representing indices.
 * @param pData A pointer to an array of unsigned char values representing the data.
 * @param pDataWeight A pointer to an array of NNFloat values representing the data weights.
 * @param alpha The alpha value for the ELU activation function.
 */
template<>
__global__ void kCalculateIndexedELUOutputDelta_kernel<uint8_t>(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint8_t* pData, NNFloat* pDataWeight, NNFloat alpha)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;

    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (1.0 / 256.0);

        pDelta[uOffset + pos] = w * (a - t) * ((a >= static_cast<NNFloat>(0.0)) + (a < static_cast<NNFloat>(0.0)) * (a + alpha));
    }
}
/**
 * @brief CUDA kernel function for calculating indexed ELU output delta.
 *
 * @param position The position index used for indexing certain data arrays.
 * @param batch The batch size.
 * @param stride The stride size.
 * @param pUnit A pointer to an array of NNFloat values representing the input units.
 * @param pDelta A pointer to an array of NNFloat values representing the delta values (output of the previous layer).
 * @param pIndex A pointer to an array of uint32_t values representing indices.
 * @param pData A pointer to an array of char values representing the data.
 * @param pDataWeight A pointer to an array of NNFloat values representing the data weights.
 * @param alpha The alpha value for the ELU activation function.
 */
template<>
__global__ void kCalculateIndexedELUOutputDelta_kernel<char>(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData, NNFloat* pDataWeight, NNFloat alpha)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;

    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (1.0 / 128.0);

        pDelta[uOffset + pos] = w * (a - t) * ((a >= static_cast<NNFloat>(0.0)) + (a < static_cast<NNFloat>(0.0)) * (a + alpha));
    }
}
/**
 * @brief CUDA kernel function for calculating indexed SELU output delta.
 *
 * @tparam T The data type of pData.
 * @param position The position index used for indexing certain data arrays.
 * @param batch The batch size.
 * @param stride The stride size.
 * @param pUnit A pointer to an array of NNFloat values representing the input units.
 * @param pDelta A pointer to an array of NNFloat values representing the delta values (output of the previous layer).
 * @param pIndex A pointer to an array of uint32_t values representing indices.
 * @param pData A pointer to an array of T values representing the data.
 * @param pDataWeight A pointer to an array of NNFloat values representing the data weights.
 * @param alpha The alpha value for the SELU activation function.
 * @param lambda The lambda value for the SELU activation function.
 */
template<typename T>
__global__ void kCalculateIndexedSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData, NNFloat* pDataWeight, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;

    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]);

        pDelta[uOffset + pos] = w * (a - t) * ((a >= static_cast<NNFloat>(0.0)) * lambda + (a < static_cast<NNFloat>(0.0)) * (lambda * alpha * exp(a)));
    }
}
/**
 * @brief CUDA kernel function for calculating indexed SELU output delta.
 *
 * @param position The position index used for indexing certain data arrays.
 * @param batch The batch size.
 * @param stride The stride size.
 * @param pUnit A pointer to an array of NNFloat values representing the input units.
 * @param pDelta A pointer to an array of NNFloat values representing the delta values (output of the previous layer).
 * @param pIndex A pointer to an array of uint32_t values representing indices.
 * @param pData A pointer to an array of unsigned char values representing the data.
 * @param pDataWeight A pointer to an array of NNFloat values representing the data weights.
 * @param alpha The alpha value for the SELU activation function.
 * @param lambda The lambda value for the SELU activation function.
 */
template<>
__global__ void kCalculateIndexedSELUOutputDelta_kernel<uint8_t>(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint8_t* pData, NNFloat* pDataWeight, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;

    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (1.0 / 256.0);

        pDelta[uOffset + pos] = w * (a - t) * ((a >= static_cast<NNFloat>(0.0)) * lambda + (a < static_cast<NNFloat>(0.0)) * (lambda * alpha * exp(a)));
    }
}
/**
 * @brief CUDA kernel function for calculating indexed SELU output delta.
 *
 * @param position The position index used for indexing certain data arrays.
 * @param batch The batch size.
 * @param stride The stride size.
 * @param pUnit A pointer to an array of NNFloat values representing the input units.
 * @param pDelta A pointer to an array of NNFloat values representing the delta values (output of the previous layer).
 * @param pIndex A pointer to an array of uint32_t values representing indices.
 * @param pData A pointer to an array of char values representing the data.
 * @param pDataWeight A pointer to an array of NNFloat values representing the data weights.
 * @param alpha The alpha value for the SELU activation function.
 * @param lambda The lambda value for the SELU activation function.
 */
template<>
__global__ void kCalculateIndexedSELUOutputDelta_kernel<char>(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData, NNFloat* pDataWeight, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;

    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (1.0 / 128.0);

        pDelta[uOffset + pos] = w * (a - t) * ((a >= static_cast<NNFloat>(0.0)) * lambda + (a < static_cast<NNFloat>(0.0)) * (lambda * alpha * exp(a)));
    }
}
/**
 * @brief CUDA kernel function for calculating indexed SoftMax output delta.
 *
 * @tparam T The data type of pData.
 * @param position The position index used for indexing certain data arrays.
 * @param batch The batch size.
 * @param stride The stride size.
 * @param pUnit A pointer to an array of NNFloat values representing the input units.
 * @param pDelta A pointer to an array of NNFloat values representing the delta values (output of the previous layer).
 * @param pIndex A pointer to an array of uint32_t values representing indices.
 * @param pData A pointer to an array of T values representing the data.
 * @param pDataWeight A pointer to an array of NNFloat values representing the data weights.
 */
template<typename T>
__global__ void kCalculateIndexedSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;

    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]);

        pDelta[uOffset + pos] = w * (a - t);
    }
}
/**
 * @brief CUDA kernel function for calculating indexed SoftMax output delta.
 *
 * @param position The position index used for indexing certain data arrays.
 * @param batch The batch size.
 * @param stride The stride size.
 * @param pUnit A pointer to an array of NNFloat values representing the input units.
 * @param pDelta A pointer to an array of NNFloat values representing the delta values (output of the previous layer).
 * @param pIndex A pointer to an array of uint32_t values representing indices.
 * @param pData A pointer to an array of unsigned char values representing the data.
 * @param pDataWeight A pointer to an array of NNFloat values representing the data weights.
 */
template<>
__global__ void kCalculateIndexedSoftMaxOutputDelta_kernel<uint8_t>(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, uint8_t* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;

    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (1.0 / 256.0);

        pDelta[uOffset + pos] = w * (a - t);
    }
}
/**
 * @brief CUDA kernel function for calculating indexed SoftMax output delta.
 *
 * @param position The position index used for indexing certain data arrays.
 * @param batch The batch size.
 * @param stride The stride size.
 * @param pUnit A pointer to an array of NNFloat values representing the input units.
 * @param pDelta A pointer to an array of NNFloat values representing the delta values (output of the previous layer).
 * @param pIndex A pointer to an array of uint32_t values representing indices.
 * @param pData A pointer to an array of char values representing the data.
 * @param pDataWeight A pointer to an array of NNFloat values representing the data weights.
 */
template<>
__global__ void kCalculateIndexedSoftMaxOutputDelta_kernel<char>(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;

    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (1.0 / 128.0);

        pDelta[uOffset + pos] = w * (a - t);
    }
}
template<typename T> void kCalculateIndexedOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData, NNFloat* pDataWeight, NNFloat slope, NNFloat alpha, NNFloat lambda)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    switch (activation)
    {
        /**
         * Calculates the output delta for the Sigmoid activation function.
         *
         * @param position The current position in the input data.
         * @param batch The batch size.
         * @param stride The stride of the input data.
         * @param pUnit Pointer to the input units.
         * @param pDelta Pointer to the delta values.
         * @param pIndex Pointer to the index values.
         * @param pData Pointer to the data.
         * @param pDataWeight Pointer to the data weights.
         */
        case Sigmoid:
            kCalculateIndexedSigmoidOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            LAUNCHERROR("kCalculateIndexedSigmoidOutputDelta_kernel");
            break;

        /**
         * Calculates the output delta for the Hyperbolic Tangent (Tanh) activation function.
         *
         * @param position The current position in the input data.
         * @param batch The batch size.
         * @param stride The stride of the input data.
         * @param pUnit Pointer to the input units.
         * @param pDelta Pointer to the delta values.
         * @param pIndex Pointer to the index values.
         * @param pData Pointer to the data.
         * @param pDataWeight Pointer to the data weights.
         */
        case Tanh:
            kCalculateIndexedTanhOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            LAUNCHERROR("kCalculateIndexedTanhOutputDelta_kernel");
            break;

        /**
         * Calculates the output delta for the Linear activation function.
         *
         * @param position The current position in the input data.
         * @param batch The batch size.
         * @param stride The stride of the input data.
         * @param pUnit Pointer to the input units.
         * @param pDelta Pointer to the delta values.
         * @param pIndex Pointer to the index values.
         * @param pData Pointer to the data.
         * @param pDataWeight Pointer to the data weights.
         */
        case Linear:
            kCalculateIndexedLinearOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            LAUNCHERROR("kCalculateIndexedLinearOutputDelta_kernel");
            break;

        /**
         * Calculates the output delta for the Rectified Linear Unit (ReLU) activation function.
         *
         * @param position The current position in the input data.
         * @param batch The batch size.
         * @param stride The stride of the input data.
         * @param pUnit Pointer to the input units.
         * @param pDelta Pointer to the delta values.
         * @param pIndex Pointer to the index values.
         * @param pData Pointer to the data.
         * @param pDataWeight Pointer to the data weights.
         */
        case RectifiedLinear:
            kCalculateIndexedRELUOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            LAUNCHERROR("kCalculateIndexedRELUOutputDelta_kernel");
            break;
            
        /**
         * Calculates the output delta for the Leaky Rectified Linear Unit (LReLU) activation function.
         *
         * @param position The current position in the input data.
         * @param batch The batch size.
         * @param stride The stride of the input data.
         * @param pUnit Pointer to the input units.
         * @param pDelta Pointer to the delta values.
         * @param pIndex Pointer to the index values.
         * @param pData Pointer to the data.
         * @param pDataWeight Pointer to the data weights.
         * @param slope The slope of the LReLU function.
         */
        case LeakyRectifiedLinear:
            kCalculateIndexedLRELUOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight, slope);
            LAUNCHERROR("kCalculateIndexedLRELUOutputDelta_kernel");
            break;

        /**
         * Calculates the output delta for the Exponential Linear Unit (ELU) activation function.
         *
         * @param position The current position in the input data.
         * @param batch The batch size.
         * @param stride The stride of the input data.
         * @param pUnit Pointer to the input units.
         * @param pDelta Pointer to the delta values.
         * @param pIndex Pointer to the index values.
         * @param pData Pointer to the data.
         * @param pDataWeight Pointer to the data weights.
         * @param alpha The alpha value for ELU.
         */
        case ExponentialLinear:
            kCalculateIndexedELUOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight, alpha);
            LAUNCHERROR("kCalculateIndexedELUOutputDelta_kernel");
            break;

        /**
         * Calculates the output delta for the Scaled Exponential Linear (SELU) activation function.
         *
         * @param position The current position in the input data.
         * @param batch The batch size.
         * @param stride The stride of the input data.
         * @param pUnit Pointer to the input units.
         * @param pDelta Pointer to the delta values.
         * @param pIndex Pointer to the index values.
         * @param pData Pointer to the data.
         * @param pDataWeight Pointer to the data weights.
         * @param alpha The alpha value for SELU.
         * @param lambda The lambda value for SELU.
         */
        case ScaledExponentialLinear:
            kCalculateIndexedSELUOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight, alpha, lambda);
            LAUNCHERROR("kCalculateIndexedSELUOutputDelta_kernel");
            break;

        /**
         * Calculates the output delta for the SoftMax activation function.
         *
         * @param position The current position in the input data.
         * @param batch The batch size.
         * @param stride The stride of the input data.
         * @param pUnit Pointer to the input units.
         * @param pDelta Pointer to the delta values.
         * @param pIndex Pointer to the index values.
         * @param pData Pointer to the data.
         * @param pDataWeight Pointer to the data weights.
         */
        case SoftMax:
            kCalculateIndexedSoftMaxOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            LAUNCHERROR("kCalculateIndexedSoftMaxOutputDelta_kernel");
            break;                
    }
}

/**
 * Calculates the output delta for the Sigmoid L2 Hinge activation function.
 *
 * @tparam T The data type of the input data.
 * @param position The current position in the input data.
 * @param batch The batch size.
 * @param stride The stride of the input data.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the delta values.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights.
 */
template<typename T>
__global__ void kCalculateSigmoidL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = pData[dOffset + pos];
        NNFloat diff = a - fabsf(t);
        diff = (t > static_cast<T>(0.0)) ? min(0.0f, diff) : max(0.0f, diff);
        pDelta[uOffset + pos] = w * diff * a * (1.0f - a);
    }
}
/**
 * Calculates the output delta for the Sigmoid L2 Hinge activation function with unsigned char input data.
 *
 * @param position The current position in the input data.
 * @param batch The batch size.
 * @param stride The stride of the input data.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the delta values.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights.
 */
__global__ void kCalculateSigmoidL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (1.0f / 256.0f);
        NNFloat diff = a - fabsf(t);
        diff = (t > 0.0f) ? min(0.0f, diff) : max(0.0f, diff);
        pDelta[uOffset + pos] = w * diff * a * (1.0f - a);
    }
}
/**
 * Calculates the output delta for the Sigmoid L2 Hinge activation function with char input data.
 *
 * @param position The current position in the input data.
 * @param batch The batch size.
 * @param stride The stride of the input data.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the delta values.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights.
 */
__global__ void kCalculateSigmoidL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (1.0f / 128.0f);
        NNFloat diff = a - fabsf(t);
        diff = (t > 0.0f) ? min(0.0f, diff) : max(0.0f, diff);
        pDelta[uOffset + pos] = w * diff * a * (1.0f - a);
    }
}

/**
 * Calculates the output delta for the Tanh L2 Hinge activation function.
 *
 * @tparam T The data type of the input data.
 * @param position The current position in the input data.
 * @param batch The batch size.
 * @param stride The stride of the input data.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the delta values.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights.
 */
template<typename T>
__global__ void kCalculateTanhL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = pData[dOffset + pos];
        NNFloat diff = a - fabsf(t);
        diff = (t > static_cast<T>(0.0)) ? min(0.0f, diff) : max(0.0f, diff);
        pDelta[uOffset + pos] = w * diff * (1.0f - a * a);
    }
}
/**
 * Calculates the output delta for the Tanh L2 Hinge activation function with unsigned char input data.
 *
 * @param position The current position in the input data.
 * @param batch The batch size.
 * @param stride The stride of the input data.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the delta values.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights.
 */
__global__ void kCalculateTanhL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (1.0f / 256.0f);
        NNFloat diff = a - fabsf(t);
        diff = (t > 0.0f) ? min(0.0f, diff) : max(0.0f, diff);
        pDelta[uOffset + pos] = w * diff * (1.0f - a * a);
    }
}
/**
 * Calculates the output delta for the Tanh L2 Hinge activation function with char input data.
 *
 * @param position The current position in the input data.
 * @param batch The batch size.
 * @param stride The stride of the input data.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the delta values.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights.
 */
__global__ void kCalculateTanhL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (1.0f / 128.0f);
        NNFloat diff = a - fabsf(t);
        diff = (t > 0.0f) ? min(0.0f, diff) : max(0.0f, diff);
        pDelta[uOffset + pos] = w * diff * (1.0f - a * a);
    }
}
/**
 * Calculates the output delta for the Linear L2 Hinge activation function.
 *
 * @tparam T The data type of the input data.
 * @param position The current position in the input data.
 * @param batch The batch size.
 * @param stride The stride of the input data.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the delta values.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights.
 */
template<typename T>
__global__ void kCalculateLinearL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = pData[dOffset + pos];
        NNFloat diff = a - fabsf(t);
        diff = (t > static_cast<T>(0.0)) ? min(0.0f, diff) : max(0.0f, diff);
        pDelta[uOffset + pos] = w * diff;
    }
}
/**
 * Calculates the output delta for the Linear L2 Hinge activation function with unsigned char input data.
 *
 * @param position The current position in the input data.
 * @param batch The batch size.
 * @param stride The stride of the input data.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the delta values.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights.
 */
__global__ void kCalculateLinearL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (1.0f / 256.0f);
        NNFloat diff = a - fabsf(t);
        diff = (t > 0.0f) ? min(0.0f, diff) : max(0.0f, diff);
        pDelta[uOffset + pos] = w * diff;
    }
}
/**
 * Calculates the output delta for the Linear L2 Hinge activation function with char input data.
 *
 * @param position The current position in the input data.
 * @param batch The batch size.
 * @param stride The stride of the input data.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the delta values.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights.
 */
__global__ void kCalculateLinearL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (1.0f / 128.0f);
        NNFloat diff = a - fabsf(t);
        diff = (t > 0.0f) ? min(0.0f, diff) : max(0.0f, diff);
        pDelta[uOffset + pos] = w * diff;
    }
}
/**
 * Calculates the output delta for the ReLU L2 Hinge activation function.
 *
 * @tparam T The data type of the input data.
 * @param position The current position in the input data.
 * @param batch The batch size.
 * @param stride The stride of the input data.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the delta values.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights.
 */
template<typename T>
__global__ void kCalculateRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = pData[dOffset + pos];
        NNFloat diff = a - fabsf(t);
        diff = (t > static_cast<T>(0.0)) ? min(0.0f, diff) : max(0.0f, diff);
        pDelta[uOffset + pos] = w * diff * static_cast<NNFloat>(a > 0.0f);
    }
}
/**
 * Calculates the output delta for the ReLU L2 Hinge activation function with unsigned char input data.
 *
 * @param position The current position in the input data.
 * @param batch The batch size.
 * @param stride The stride of the input data.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the delta values.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights.
 */
__global__ void kCalculateRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (1.0f / 256.0f);
        NNFloat diff = a - fabsf(t);
        diff = (t > 0.0f) ? min(0.0f, diff) : max(0.0f, diff);
        pDelta[uOffset + pos] = w * diff * static_cast<NNFloat>(a > 0.0f);
    }
}
/**
 * Calculates the output delta for the ReLU L2 Hinge activation function with char input data.
 *
 * @param position The current position in the input data.
 * @param batch The batch size.
 * @param stride The stride of the input data.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the delta values.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights.
 */
__global__ void kCalculateRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (1.0f / 128.0f);
        NNFloat diff = a - fabsf(t);
        diff = (t > 0.0f) ? min(0.0f, diff) : max(0.0f, diff);
        pDelta[uOffset + pos] = w * diff * static_cast<NNFloat>(a > 0.0f);
    }
}
/**
 * Calculates the output delta for the Leaky ReLU L2 Hinge activation function.
 *
 * @tparam T The data type of the input data.
 * @param position The current position in the input data.
 * @param batch The batch size.
 * @param stride The stride of the input data.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the delta values.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights.
 * @param slope The slope parameter for the Leaky ReLU activation function.
 */
template<typename T>
__global__ void kCalculateLRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat* pDataWeight, NNFloat slope)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = pData[dOffset + pos];
        NNFloat diff = a - fabsf(t);
        diff = (t > static_cast<T>(0.0)) ? min(0.0f, diff) : max(0.0f, diff);
        pDelta[uOffset + pos] = w * diff * ((a > 0.0f) + (a <= 0.0f) * slope);
    }
}
/**
 * Calculates the output delta for the Leaky ReLU L2 Hinge activation function with unsigned char input data.
 *
 * @param position The current position in the input data.
 * @param batch The batch size.
 * @param stride The stride of the input data.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the delta values.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights.
 * @param slope The slope parameter for the Leaky ReLU activation function.
 */
__global__ void kCalculateLRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData, NNFloat* pDataWeight, NNFloat slope)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (1.0f / 256.0f);
        NNFloat diff = a - fabsf(t);
        diff = (t > 0.0f) ? min(0.0f, diff) : max(0.0f, diff);
        pDelta[uOffset + pos] = w * diff * ((a > 0.0f) + (a <= 0.0f) * slope);
    }
}
/**
 * Calculates the output delta for the Leaky ReLU L2 Hinge activation function with char input data.
 *
 * @param position The current position in the input data.
 * @param batch The batch size.
 * @param stride The stride of the input data.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the delta values.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights.
 * @param slope The slope parameter for the Leaky ReLU activation function.
 */
__global__ void kCalculateLRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData, NNFloat* pDataWeight, NNFloat slope)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (1.0f / 128.0f);
        NNFloat diff = a - fabsf(t);
        diff = (t > 0.0f) ? min(0.0f, diff) : max(0.0f, diff);
        pDelta[uOffset + pos] = w * diff * ((a > 0.0f) + (a <= 0.0f) * slope);
    }
}
/**
 * Calculates the output delta for the Exponential Linear Unit (ELU) L2 Hinge activation function.
 *
 * @tparam T The data type of the input data.
 * @param position The current position in the input data.
 * @param batch The batch size.
 * @param stride The stride of the input data.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the delta values.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights.
 * @param alpha The alpha parameter for the ELU activation function.
 */
template<typename T>
__global__ void kCalculateELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat* pDataWeight, NNFloat alpha)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = pData[dOffset + pos];
        NNFloat diff = a - fabsf(t);
        diff = (t > static_cast<T>(0.0)) ? min(0.0f, diff) : max(0.0f, diff);
        pDelta[uOffset + pos] = w * diff * ((a >= 0.0f) + (a < 0.0f) * (a + alpha));
    }
}
/**
 * Calculates the output delta for the Exponential Linear Unit (ELU) L2 Hinge activation function with unsigned char input data.
 *
 * @param position The current position in the input data.
 * @param batch The batch size.
 * @param stride The stride of the input data.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the delta values.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights.
 * @param alpha The alpha parameter for the ELU activation function.
 */
__global__ void kCalculateELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData, NNFloat* pDataWeight, NNFloat alpha)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (1.0f / 256.0f);
        NNFloat diff = a - fabsf(t);
        diff = (t > 0.0f) ? min(0.0f, diff) : max(0.0f, diff);
        pDelta[uOffset + pos] = w * diff * ((a >= 0.0f) + (a < 0.0f) * (a + alpha));
    }
}
/**
 * Calculates the output delta for the Exponential Linear Unit (ELU) L2 Hinge activation function with char input data.
 *
 * @param position The current position in the input data.
 * @param batch The batch size.
 * @param stride The stride of the input data.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the delta values.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights.
 * @param alpha The alpha parameter for the ELU activation function.
 */
__global__ void kCalculateELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData, NNFloat* pDataWeight, NNFloat alpha)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (1.0f / 128.0f);
        NNFloat diff = a - fabsf(t);
        diff = (t > 0.0f) ? min(0.0f, diff) : max(0.0f, diff);
        pDelta[uOffset + pos] = w * diff * ((a >= 0.0f) + (a < 0.0f) * (a + alpha));
    }
}
/**
 * Calculates the output delta for the Scaled Exponential Linear Unit (SELU) L2 Hinge activation function.
 *
 * @param position The current position in the input data.
 * @param batch The batch size.
 * @param stride The stride of the input data.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the delta values.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights.
 * @param alpha The alpha parameter for the SELU activation function.
 * @param lambda The lambda parameter for the SELU activation function.
 */
__global__ void kCalculateSELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, NNFloat* pData, NNFloat* pDataWeight, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = pData[dOffset + pos];
        NNFloat diff = a - fabsf(t);
        diff = (t > 0.0f) ? min(0.0f, diff) : max(0.0f, diff);
        pDelta[uOffset + pos] = w * diff * ((a >= 0.0f) * lambda + (a < 0.0f) * (lambda * alpha * expf(a)));
    }
}
/**
 * Calculates the output delta for the Scaled Exponential Linear Unit (SELU) L2 Hinge activation function.
 *
 * @param position The current position in the input data.
 * @param batch The batch size.
 * @param stride The stride of the input data.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the delta values.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights.
 * @param alpha The alpha parameter for the SELU activation function.
 * @param lambda The lambda parameter for the SELU activation function.
 */
__global__ void kCalculateSELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData, NNFloat* pDataWeight, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = (NNFloat)pData[dOffset + pos] * (1.0f / 256.0f);
        NNFloat diff = a - fabsf(t);
        diff = (t > 0.0f) ? min(0.0f, diff) : max(0.0f, diff);
        pDelta[uOffset + pos] = w * diff * ((a >= 0.0f) * lambda + (a < 0.0f) * (lambda * alpha * expf(a)));
    }
}
/**
 * Calculates the output delta for the Scaled Exponential Linear Unit (SELU) L2 Hinge activation function.
 *
 * @param position The current position in the input data.
 * @param batch The batch size.
 * @param stride The stride of the input data.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the delta values.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights.
 * @param alpha The alpha parameter for the SELU activation function.
 * @param lambda The lambda parameter for the SELU activation function.
 */
__global__ void kCalculateSELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData, NNFloat* pDataWeight, NNFloat alpha, NNFloat lambda)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (1.0f / 128.0f);
        NNFloat diff = a - fabsf(t);
        diff = (t > 0.0f) ? min(0.0f, diff) : max(0.0f, diff);
        pDelta[uOffset + pos] = w * diff * ((a >= 0.0f) * lambda + (a < 0.0f) * (lambda * alpha * expf(a)));
    }
}
/**
 * Calculates the output delta for the Softmax L2 Hinge activation function.
 *
 * @param position The current position in the input data.
 * @param batch The batch size.
 * @param stride The stride of the input data.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the delta values.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights.
 */
template<typename T>
__global__ void kCalculateSoftMaxL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]);
        NNFloat diff = a - fabsf(t);
        diff = (t > static_cast<T>(0.0)) ? min(0.0f, diff) : max(0.0f, diff);
        pDelta[uOffset + pos] = w * diff;
    }
}
/**
 * Calculates the output delta for the Softmax L2 Hinge activation function.
 *
 * @param position The current position in the input data.
 * @param batch The batch size.
 * @param stride The stride of the input data.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the delta values.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights.
 */
template<>
__global__ void kCalculateSoftMaxL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (1.0f / 256.0f);
        NNFloat diff = a - fabsf(t);
        diff = (t > 0.0f) ? min(0.0f, diff) : max(0.0f, diff);
        pDelta[uOffset + pos] = w * diff;
    }
}
/**
 * Calculates the output delta for the Softmax L2 Hinge activation function.
 *
 * @param position The current position in the input data.
 * @param batch The batch size.
 * @param stride The stride of the input data.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the delta values.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights.
 */
template<>
__global__ void kCalculateSoftMaxL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (1.0f / 128.0f);
        NNFloat diff = a - fabsf(t);
        diff = (t > 0.0f) ? min(0.0f, diff) : max(0.0f, diff);
        pDelta[uOffset + pos] = w * diff;
    }
}
template<typename T> void kCalculateL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat* pDataWeight, NNFloat slope, NNFloat alpha, NNFloat lambda)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    switch (activation)
    {
        /**
         * Calculates the output delta for the Sigmoid L2 Hinge activation function.
         *
         * @param position The current position in the input data.
         * @param batch The batch size.
         * @param stride The stride of the input data.
         * @param pUnit Pointer to the input units.
         * @param pDelta Pointer to the delta values.
         * @param pData Pointer to the data.
         * @param pDataWeight Pointer to the data weights.
         */
        case Sigmoid:
            kCalculateSigmoidL2HingeOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            LAUNCHERROR("kCalculateSigmoidL2HingeOutputDelta_kernel");
            break;

        /**
         * Calculates the output delta for the Tanh L2 Hinge activation function.
         *
         * @param position The current position in the input data.
         * @param batch The batch size.
         * @param stride The stride of the input data.
         * @param pUnit Pointer to the input units.
         * @param pDelta Pointer to the delta values.
         * @param pData Pointer to the data.
         * @param pDataWeight Pointer to the data weights.
         */
        case Tanh:
            kCalculateTanhL2HingeOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            LAUNCHERROR("kCalculateTanhL2HingeOutputDelta_kernel");
            break;

        /**
         * Calculates the output delta for the Linear L2 Hinge activation function.
         *
         * @param position The current position in the input data.
         * @param batch The batch size.
         * @param stride The stride of the input data.
         * @param pUnit Pointer to the input units.
         * @param pDelta Pointer to the delta values.
         * @param pData Pointer to the data.
         * @param pDataWeight Pointer to the data weights.
         */
        case Linear:
            kCalculateLinearL2HingeOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            LAUNCHERROR("kCalculateLinearL2HingeOutputDelta_kernel");
            break;

        /**
         * Calculates the output delta for the Rectified Linear (ReLU) L2 Hinge activation function.
         *
         * @param position The current position in the input data.
         * @param batch The batch size.
         * @param stride The stride of the input data.
         * @param pUnit Pointer to the input units.
         * @param pDelta Pointer to the delta values.
         * @param pData Pointer to the data.
         * @param pDataWeight Pointer to the data weights.
         */
        case RectifiedLinear:
            kCalculateRELUL2HingeOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            LAUNCHERROR("kCalculateRELUL2HingeOutputDelta_kernel");
            break;
            
        /**
         * Calculates the output delta for the Scaled Exponential Linear (SELu) L2 Hinge activation function.
         *
         * @param position The current position in the input data.
         * @param batch The batch size.
         * @param stride The stride of the input data.
         * @param pUnit Pointer to the input units.
         * @param pDelta Pointer to the delta values.
         * @param pData Pointer to the data.
         * @param pDataWeight Pointer to the data weights.
         * @param alpha The SELu alpha parameter.
         * @param lambda The SELu lambda parameter.
         */
        case ScaledExponentialLinear:
            kCalculateSELUL2HingeOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight, alpha, lambda);
            LAUNCHERROR("kCalculateSELUL2HingeOutputDelta_kernel");
            break;

        /**
         * Calculates the output delta for the Softmax L2 Hinge activation function.
         *
         * @param position The current position in the input data.
         * @param batch The batch size.
         * @param stride The stride of the input data.
         * @param pUnit Pointer to the input units.
         * @param pDelta Pointer to the delta values.
         * @param pData Pointer to the data.
         * @param pDataWeight Pointer to the data weights.
         */
        case SoftMax:
            kCalculateSoftMaxL2HingeOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            LAUNCHERROR("kCalculateSoftMaxL2HingeOutputDelta_kernel");
            break;

        /**
         * Calculates the output delta for the Scaled Exponential Linear (SELu) L2 Hinge activation function.
         *
         * @param position The current position in the input data.
         * @param batch The batch size.
         * @param stride The stride of the input data.
         * @param pUnit Pointer to the input units.
         * @param pDelta Pointer to the delta values.
         * @param pData Pointer to the data.
         * @param pDataWeight Pointer to the data weights.
         * @param alpha The SELu alpha parameter.
         * @param lambda The SELu lambda parameter.
         */
        case ScaledExponentialLinear:
            kCalculateSELUL2HingeOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight, alpha, lambda);
            LAUNCHERROR("kCalculateSELUL2HingeOutputDelta_kernel");
            break;

        /**
         * Calculates the output delta for the Softmax L2 Hinge activation function.
         *
         * @param position The current position in the input data.
         * @param batch The batch size.
         * @param stride The stride of the input data.
         * @param pUnit Pointer to the input units.
         * @param pDelta Pointer to the delta values.
         * @param pData Pointer to the data.
         * @param pDataWeight Pointer to the data weights.
         */
        case SoftMax:
            kCalculateSoftMaxL2HingeOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            LAUNCHERROR("kCalculateSoftMaxL2HingeOutputDelta_kernel");
            break;                
    }
}

/**
 * @brief CUDA kernel for calculating the output delta for the Indexed Sigmoid L2 Hinge activation function.
 *
 * @tparam T The data type of the input data.
 * @param position The position of the current data batch.
 * @param batch The size of the batch.
 * @param stride The stride of the data.
 * @param pUnit Pointer to the unit values.
 * @param pDelta Pointer to the delta values.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the input data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSigmoidL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != NULL) ? pDataWeight[dpos] : (NNFloat)1.0;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = pData[dOffset + pos];
        NNFloat diff = a - fabsf(t);
        diff = (t > static_cast<T>(0.0)) ? min(static_cast<NNFloat>(0.0), diff) : max(static_cast<NNFloat>(0.0), diff);
        pDelta[uOffset + pos] = w * diff * a * (static_cast<NNFloat>(1.0) - a);
    }
}
/**
 * @brief CUDA kernel for calculating the output delta for the Indexed Sigmoid L2 Hinge activation function.
 *
 * @param position The position of the current data batch.
 * @param batch The size of the batch.
 * @param stride The stride of the data.
 * @param pUnit Pointer to the unit values.
 * @param pDelta Pointer to the delta values.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the input data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSigmoidL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != NULL) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = (static_cast<NNFloat>(pData[dOffset + pos]) * static_cast<NNFloat>(1.0 / 256.0));
        NNFloat diff = a - fabsf(t);
        diff = (t > static_cast<NNFloat>(0.0)) ? min(static_cast<NNFloat>(0.0f), diff) : max(static_cast<NNFloat>(0.0), diff);
        pDelta[uOffset + pos] = w * diff * a * (static_cast<NNFloat>(1.0) - a);
    }
}
/**
 * @brief CUDA kernel for calculating the output delta for the Indexed Sigmoid L2 Hinge activation function.
 *
 * @param position The position of the current data batch.
 * @param batch The size of the batch.
 * @param stride The stride of the data.
 * @param pUnit Pointer to the unit values.
 * @param pDelta Pointer to the delta values.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the input data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSigmoidL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != NULL) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = (static_cast<NNFloat>(pData[dOffset + pos]) * static_cast<NNFloat>(1.0 / 128.0));
        NNFloat diff = a - fabsf(t);
        diff = (t > static_cast<NNFloat>(0.0)) ? min(static_cast<NNFloat>(0.0f), diff) : max(static_cast<NNFloat>(0.0), diff);
        pDelta[uOffset + pos] = w * diff * a * (static_cast<NNFloat>(1.0) - a);
    }
}
/**
 * @brief CUDA kernel for calculating the output delta for the Indexed Tanh L2 Hinge activation function.
 *
 * @param position The position of the current data batch.
 * @param batch The size of the batch.
 * @param stride The stride of the data.
 * @param pUnit Pointer to the unit values.
 * @param pDelta Pointer to the delta values.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the input data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedTanhL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != NULL) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = pData[dOffset + pos];
        NNFloat diff = a - fabsf(t);
        diff = (t > static_cast<T>(0.0)) ? min(static_cast<NNFloat>(0.0f), diff) : max(static_cast<NNFloat>(0.0), diff);
        pDelta[uOffset + pos] = w * diff * (static_cast<NNFloat>(1.0) - a * a);
    }
}
/**
 * @brief CUDA kernel for calculating the output delta for the Indexed Tanh L2 Hinge activation function.
 *
 * @param position The position of the current data batch.
 * @param batch The size of the batch.
 * @param stride The stride of the data.
 * @param pUnit Pointer to the unit values.
 * @param pDelta Pointer to the delta values.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the input data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedTanhL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != NULL) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = (NNFloat)pData[dOffset + pos] * (static_cast<NNFloat>(1.0) / static_cast<NNFloat>(256.0));
        NNFloat diff = a - fabsf(t);
        diff = (t > static_cast<NNFloat>(0.0)) ? min(static_cast<NNFloat>(0.0f), diff) : max(static_cast<NNFloat>(0.0), diff);
        pDelta[uOffset + pos] = w * diff * (static_cast<NNFloat>(1.0) - a * a);
    }
}
/**
 * @brief CUDA kernel for calculating the output delta for the Indexed Tanh L2 Hinge activation function.
 *
 * @param position The position of the current data batch.
 * @param batch The size of the batch.
 * @param stride The stride of the data.
 * @param pUnit Pointer to the unit values.
 * @param pDelta Pointer to the delta values.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the input data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedTanhL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != NULL) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = (NNFloat)pData[dOffset + pos] * (static_cast<NNFloat>(1.0) / static_cast<NNFloat>(128.0));
        NNFloat diff = a - fabsf(t);
        diff = (t > static_cast<NNFloat>(0.0)) ? min(static_cast<NNFloat>(0.0f), diff) : max(static_cast<NNFloat>(0.0), diff);
        pDelta[uOffset + pos] = w * diff * (static_cast<NNFloat>(1.0) - a * a);
    }
}
/**
 * @brief CUDA kernel for calculating the output delta for the Indexed Linear L2 Hinge activation function.
 *
 * @param position The position of the current data batch.
 * @param batch The size of the batch.
 * @param stride The stride of the data.
 * @param pUnit Pointer to the unit values.
 * @param pDelta Pointer to the delta values.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the input data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLinearL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != NULL) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]);
        NNFloat diff = a - fabsf(t);
        diff = (t > static_cast<T>(0.0)) ? min(static_cast<NNFloat>(0.0f), diff) : max(static_cast<NNFloat>(0.0), diff);
        pDelta[uOffset + pos] = w * diff;
    }
}
/**
 * @brief CUDA kernel for calculating the output delta for the Indexed Linear L2 Hinge activation function.
 *
 * @param position The position of the current data batch.
 * @param batch The size of the batch.
 * @param stride The stride of the data.
 * @param pUnit Pointer to the unit values.
 * @param pDelta Pointer to the delta values.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the input data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLinearL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (static_cast<NNFloat>(1.0) / 256.0);
        NNFloat diff = a - fabsf(t);
        diff = (t > static_cast<NNFloat>(0.0)) ? min(static_cast<NNFloat>(0.0f), diff) : max(static_cast<NNFloat>(0.0), diff);
        pDelta[uOffset + pos] = w * diff;
    }
}
/**
 * @brief CUDA kernel for calculating the output delta for the Indexed Linear L2 Hinge activation function.
 *
 * @param position The position of the current data batch.
 * @param batch The size of the batch.
 * @param stride The stride of the data.
 * @param pUnit Pointer to the unit values.
 * @param pDelta Pointer to the delta values.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the input data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLinearL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (static_cast<NNFloat>(1.0) / 128.0);
        NNFloat diff = a - fabsf(t);
        diff = (t > static_cast<NNFloat>(0.0)) ? min(static_cast<NNFloat>(0.0f), diff) : max(static_cast<NNFloat>(0.0), diff);
        pDelta[uOffset + pos] = w * diff;
    }
}
/**
 * @brief CUDA kernel for calculating the output delta for the Indexed ReLU L2 Hinge activation function.
 *
 * @param position The position of the current data batch.
 * @param batch The size of the batch.
 * @param stride The stride of the data.
 * @param pUnit Pointer to the unit values.
 * @param pDelta Pointer to the delta values.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the input data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = pData[dOffset + pos];
        NNFloat diff = a - fabsf(t);
        diff = (t > static_cast<T>(0.0)) ? min(static_cast<NNFloat>(0.0f), diff) : max(static_cast<NNFloat>(0.0), diff);
        pDelta[uOffset + pos] = w * diff * static_cast<NNFloat>(a > static_cast<NNFloat>(0.0));
    }
}
/**
 * @brief CUDA kernel for calculating the output delta for the Indexed ReLU L2 Hinge activation function with unsigned char data.
 *
 * @param position The position of the current data batch.
 * @param batch The size of the batch.
 * @param stride The stride of the data.
 * @param pUnit Pointer to the unit values.
 * @param pDelta Pointer to the delta values.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the input data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = (NNFloat)pData[dOffset + pos] * static_cast<NNFloat>(1.0 / 256.0);
        NNFloat diff = a - fabsf(t);
        diff = (t > static_cast<NNFloat>(0.0)) ? min(static_cast<NNFloat>(0.0f), diff) : max(static_cast<NNFloat>(0.0), diff);
        pDelta[uOffset + pos] = w * diff * static_cast<NNFloat>(a > static_cast<NNFloat>(0.0));
    }
}
/**
 * @brief CUDA kernel for calculating the output delta for the Indexed ReLU L2 Hinge activation function with char data.
 *
 * @param position The position of the current data batch.
 * @param batch The size of the batch.
 * @param stride The stride of the data.
 * @param pUnit Pointer to the unit values.
 * @param pDelta Pointer to the delta values.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the input data array.
 * @param pDataWeight Pointer to the data weight array.
 */
template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = (NNFloat)pData[dOffset + pos] * static_cast<NNFloat>(1.0 / 128.0);
        NNFloat diff = a - fabsf(t);
        diff = (t > static_cast<NNFloat>(0.0)) ? min(static_cast<NNFloat>(0.0f), diff) : max(static_cast<NNFloat>(0.0), diff);
        pDelta[uOffset + pos] = w * diff * static_cast<NNFloat>(a > static_cast<NNFloat>(0.0));
    }
}
/**
 * @brief CUDA kernel for calculating the output delta for the Indexed Leaky ReLU L2 Hinge activation function.
 *
 * @param position The position of the current data batch.
 * @param batch The size of the batch.
 * @param stride The stride of the data.
 * @param pUnit Pointer to the unit values.
 * @param pDelta Pointer to the delta values.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the input data array.
 * @param pDataWeight Pointer to the data weight array.
 * @param slope The slope of the Leaky ReLU activation function.
 */
template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData, NNFloat* pDataWeight, NNFloat slope)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = pData[dOffset + pos];
        NNFloat diff = a - fabsf(t);
        diff = (t > static_cast<T>(0.0)) ? min(static_cast<NNFloat>(0.0f), diff) : max(static_cast<NNFloat>(0.0), diff);
        pDelta[uOffset + pos] = w * diff * ((a > static_cast<NNFloat>(0.0)) + (a <= static_cast<NNFloat>(0.0)) * slope);
    }
}
/**
 * @brief CUDA kernel for calculating the output delta for the Indexed Leaky ReLU L2 Hinge activation function.
 *
 * @param position The position of the current data batch.
 * @param batch The size of the batch.
 * @param stride The stride of the data.
 * @param pUnit Pointer to the unit values.
 * @param pDelta Pointer to the delta values.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the input data array.
 * @param pDataWeight Pointer to the data weight array.
 * @param slope The slope of the Leaky ReLU activation function.
 */
template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData, NNFloat* pDataWeight, NNFloat slope)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = (NNFloat)pData[dOffset + pos] * (static_cast<NNFloat>(1.0) / static_cast<NNFloat>(256.0));
        NNFloat diff = a - fabsf(t);
        diff = (t > static_cast<NNFloat>(0.0)) ? min(static_cast<NNFloat>(0.0f), diff) : max(static_cast<NNFloat>(0.0), diff);
        pDelta[uOffset + pos] = w * diff * ((a > static_cast<NNFloat>(0.0)) + (a <= static_cast<NNFloat>(0.0)) * slope);
    }
}
/**
 * @brief CUDA kernel for calculating the output delta for the Indexed Leaky ReLU L2 Hinge activation function.
 *
 * @param position The position of the current data batch.
 * @param batch The size of the batch.
 * @param stride The stride of the data.
 * @param pUnit Pointer to the unit values.
 * @param pDelta Pointer to the delta values.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the input data array.
 * @param pDataWeight Pointer to the data weight array.
 * @param slope The slope of the Leaky ReLU activation function.
 */
template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData, NNFloat* pDataWeight, NNFloat slope)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = (NNFloat)pData[dOffset + pos] * (static_cast<NNFloat>(1.0) / static_cast<NNFloat>(128.0));
        NNFloat diff = a - fabsf(t);
        diff = (t > static_cast<NNFloat>(0.0)) ? min(static_cast<NNFloat>(0.0f), diff) : max(static_cast<NNFloat>(0.0), diff);
        pDelta[uOffset + pos] = w * diff * ((a > static_cast<NNFloat>(0.0)) + (a <= static_cast<NNFloat>(0.0)) * slope);
    }
}
/**
 * @brief CUDA kernel for calculating the output delta for the Indexed ELU L2 Hinge activation function.
 *
 * @param position The position of the current data batch.
 * @param batch The size of the batch.
 * @param stride The stride of the data.
 * @param pUnit Pointer to the unit values.
 * @param pDelta Pointer to the delta values.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the input data array.
 * @param pDataWeight Pointer to the data weight array.
 * @param alpha The alpha parameter of the ELU activation function.
 */
template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData, NNFloat* pDataWeight, NNFloat alpha)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = pData[dOffset + pos];
        NNFloat diff = a - fabsf(t);
        diff = (t > static_cast<T>(0.0)) ? min(static_cast<NNFloat>(0.0f), diff) : max(static_cast<NNFloat>(0.0), diff);
        pDelta[uOffset + pos] = w * diff * ((a >= static_cast<NNFloat>(0.0)) + (a < static_cast<NNFloat>(0.0)) * (a + alpha));
    }
}
