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
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : (NNFloat)1.0;
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
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData, NNFloat* pDataWeight, NNFloat alpha)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 128.0);
        NNFloat diff = a - fabsf(t);
        diff = (t > static_cast<NNFloat>(0.0)) ? min(static_cast<NNFloat>(0.0f), diff) : max(static_cast<NNFloat>(0.0), diff);
        pDelta[uOffset + pos] = w * diff * ((a >= static_cast<NNFloat>(0.0)) + (a < static_cast<NNFloat>(0.0)) * (a + alpha));
    }
}
/**
 * @brief CUDA kernel for calculating the output delta for the Indexed SELU L2 Hinge activation function.
 *
 * @param position The position of the current data batch.
 * @param batch The size of the batch.
 * @param stride The stride of the data.
 * @param pUnit Pointer to the unit values.
 * @param pDelta Pointer to the delta values.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the input data array.
 * @param pDataWeight Pointer to the data weight array.
 * @param alpha The alpha parameter of the SELU activation function.
 * @param lambda The lambda parameter of the SELU activation function.
 */
template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData, NNFloat* pDataWeight, NNFloat alpha, NNFloat lambda)
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
        pDelta[uOffset + pos] = w * diff * ((a >= static_cast<NNFloat>(0.0)) * lambda + (a < static_cast<NNFloat>(0.0)) * (lambda * alpha * exp(a)));
    }
}
/**
 * @brief CUDA kernel to calculate indexed SELU L2 hinge output delta.
 * 
 * @tparam DummyTemplate Dummy template parameter.
 * @param position Starting position.
 * @param batch Batch size.
 * @param stride Stride value.
 * @param pUnit Pointer to the unit data.
 * @param pDelta Pointer to the delta data.
 * @param pIndex Pointer to the index data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight.
 * @param alpha Alpha value.
 * @param lambda Lambda value.
 */
template <>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSELUL2HingeOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta,
    uint32_t* pIndex, unsigned char* pData, NNFloat* pDataWeight, NNFloat alpha,
    NNFloat lambda)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset =
            blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : (NNFloat)1.0;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = (NNFloat)pData[dOffset + pos] * (NNFloat)(1.0 / 256.0);

        NNFloat diff = a - fabsf(t);
        diff = (t > 0.0f) ? fminf(0.0f, diff) : fmaxf(0.0f, diff);

        pDelta[uOffset + pos] = w * diff * ((a >= 0.0f) * lambda + (a < 0.0f) * (lambda * alpha * expf(a)));
    }
}
/**
 * @brief CUDA kernel to calculate indexed SELU L2 hinge output delta.
 * 
 * @tparam DummyTemplate Dummy template parameter.
 * @param position Starting position.
 * @param batch Batch size.
 * @param stride Stride value.
 * @param pUnit Pointer to the unit data.
 * @param pDelta Pointer to the delta data.
 * @param pIndex Pointer to the index data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight.
 * @param alpha Alpha value.
 * @param lambda Lambda value.
 */
template <>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSELUL2HingeOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta,
    uint32_t* pIndex, char* pData, NNFloat* pDataWeight, NNFloat alpha, NNFloat lambda)
{
    uint32_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint32_t uOffset = blockIdx.x * stride;
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint32_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (1.0f / 128.0f);

        NNFloat diff = a - fabsf(t);
        diff = (t > 0.0f) ? fminf(0.0f, diff) : fmaxf(0.0f, diff);

        pDelta[uOffset + pos] = w * diff * ((a >= 0.0f) * lambda + (a < 0.0f) * (lambda * alpha * expf(a)));
    }
}
/**
 * @brief CUDA kernel to calculate indexed SoftMax L2 hinge output delta.
 * 
 * @tparam T Data type of the input pData.
 * @param position Starting position.
 * @param batch Batch size.
 * @param stride Stride value.
 * @param pUnit Pointer to the unit data.
 * @param pDelta Pointer to the delta data.
 * @param pIndex Pointer to the index data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight.
 */
template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSoftMaxL2HingeOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta,
    uint32_t* pIndex, T* pData, NNFloat* pDataWeight)
{
    uint32_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint32_t uOffset = blockIdx.x * stride;
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint32_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        NNFloat a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        NNFloat diff = a - fabsf(static_cast<NNFloat>(t));
        diff = (t > static_cast<T>(0.0)) ? fminf(0.0f, diff) : fmaxf(0.0f, diff);

        pDelta[uOffset + pos] = w * diff;
    }
}
/**
 * @brief CUDA kernel to calculate indexed SoftMax L2 hinge output delta.
 * 
 * @tparam DummyTemplate Dummy template parameter.
 * @param position Starting position.
 * @param batch Batch size.
 * @param stride Stride value.
 * @param pUnit Pointer to the unit data.
 * @param pDelta Pointer to the delta data.
 * @param pIndex Pointer to the index data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight.
 */
template <>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSoftMaxL2HingeOutputDelta_kernel(
    uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta,
    uint32_t* pIndex, unsigned char* pData, NNFloat* pDataWeight)
{
    uint32_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint32_t uOffset = blockIdx.x * stride;
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint32_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;
        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * (1.0f / 256.0f);
        NNFloat diff = a - fabsf(static_cast<float>(t));
        diff = (t > 0.0f) ? fminf(0.0f, diff) : fmaxf(0.0f, diff);

        pDelta[uOffset + pos] = w * diff;
    }
}
/**
 * @brief CUDA kernel to calculate Indexed SoftMax L2 Hinge Output Delta.
 *
 * This CUDA kernel calculates the output delta for the Indexed SoftMax L2 Hinge
 * activation function. It operates on a batch of units and updates the corresponding
 * delta values based on the specified indices and data. The delta values are scaled
 * by the data weights, if provided.
 *
 * @tparam UnusedTemplateParam Unused template parameter.
 * @param position The starting position of the batch.
 * @param batch The size of the batch.
 * @param stride The stride between units in memory.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the output deltas.
 * @param pIndex Pointer to the indices for data access.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights (optional).
 */
template<>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedSoftMaxL2HingeOutputDelta_kernel(
    uint32_t position,
    uint32_t batch,
    uint32_t stride,
    NNFloat* pUnit,
    NNFloat* pDelta,
    uint32_t* pIndex,
    char* pData,
    NNFloat* pDataWeight)
{
    constexpr NNFloat scaleFactor = 1.0f / 128.0f;

    uint64_t pos = blockIdx.y * blockDim.x + threadIdx.x;

    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;

        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0f;

        NNFloat a = pUnit[uOffset + pos];
        NNFloat t = static_cast<NNFloat>(pData[dOffset + pos]) * scaleFactor;

        NNFloat diff = a - fabsf(t);
        diff = (t > 0.0f) ? min(0.0f, diff) : max(0.0f, diff);

        pDelta[uOffset + pos] = w * diff;
    }
}
template<typename T> void kCalculateIndexedL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData, NNFloat* pDataWeight, NNFloat slope, NNFloat alpha, NNFloat lambda)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    switch (activation)
    {
        /**
         * @brief Launches the CUDA kernel for calculating the output delta of the Indexed Sigmoid activation.
         *
         * This function launches the CUDA kernel for calculating the output delta using the Indexed Sigmoid activation function.
         * It operates on a batch of units and updates the corresponding delta values based on the specified indices and data.
         *
         * @param position The starting position of the batch.
         * @param batch The size of the batch.
         * @param stride The stride between units in memory.
         * @param pUnit Pointer to the input units.
         * @param pDelta Pointer to the output deltas.
         * @param pIndex Pointer to the indices for data access.
         * @param pData Pointer to the data.
         * @param pDataWeight Pointer to the data weights (optional).
         */
        case Sigmoid:
            kCalculateIndexedSigmoidL2HingeOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            LAUNCHERROR("kCalculateIndexedSigmoidL2HingeOutputDelta_kernel");
            break;

        /**
         * @brief Launches the CUDA kernel for calculating the output delta of the Indexed Tanh activation.
         *
         * This function launches the CUDA kernel for calculating the output delta using the Indexed Tanh activation function.
         * It operates on a batch of units and updates the corresponding delta values based on the specified indices and data.
         *
         * @param position The starting position of the batch.
         * @param batch The size of the batch.
         * @param stride The stride between units in memory.
         * @param pUnit Pointer to the input units.
         * @param pDelta Pointer to the output deltas.
         * @param pIndex Pointer to the indices for data access.
         * @param pData Pointer to the data.
         * @param pDataWeight Pointer to the data weights (optional).
         */
        case Tanh:
            kCalculateIndexedTanhL2HingeOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            LAUNCHERROR("kCalculateIndexedTanhL2HingeOutputDelta_kernel");
            break;

        /**
         * @brief Launches the CUDA kernel for calculating the output delta of the Indexed Linear activation.
         *
         * This function launches the CUDA kernel for calculating the output delta using the Indexed Linear activation function.
         * It operates on a batch of units and updates the corresponding delta values based on the specified indices and data.
         *
         * @param position The starting position of the batch.
         * @param batch The size of the batch.
         * @param stride The stride between units in memory.
         * @param pUnit Pointer to the input units.
         * @param pDelta Pointer to the output deltas.
         * @param pIndex Pointer to the indices for data access.
         * @param pData Pointer to the data.
         * @param pDataWeight Pointer to the data weights (optional).
         */
        case Linear:
            kCalculateIndexedLinearL2HingeOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            LAUNCHERROR("kCalculateIndexedLinearL2HingeOutputDelta_kernel");
            break;

        /**
         * @brief Launches the CUDA kernel for calculating the output delta of the Indexed Rectified Linear activation.
         *
         * This function launches the CUDA kernel for calculating the output delta using the Indexed Rectified Linear activation function.
         * It operates on a batch of units and updates the corresponding delta values based on the specified indices and data.
         *
         * @param position The starting position of the batch.
         * @param batch The size of the batch.
         * @param stride The stride between units in memory.
         * @param pUnit Pointer to the input units.
         * @param pDelta Pointer to the output deltas.
         * @param pIndex Pointer to the indices for data access.
         * @param pData Pointer to the data.
         * @param pDataWeight Pointer to the data weights (optional).
         */
        case RectifiedLinear:
            kCalculateIndexedRELUL2HingeOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            LAUNCHERROR("kCalculateIndexedRELUL2HingeOutputDelta_kernel");
            break;
            
        /**
         * @brief Launches the CUDA kernel for calculating the output delta of the Indexed Leaky Rectified Linear activation.
         *
         * This function launches the CUDA kernel for calculating the output delta using the Indexed Leaky Rectified Linear activation function.
         * It operates on a batch of units and updates the corresponding delta values based on the specified indices and data.
         *
         * @param position The starting position of the batch.
         * @param batch The size of the batch.
         * @param stride The stride between units in memory.
         * @param pUnit Pointer to the input units.
         * @param pDelta Pointer to the output deltas.
         * @param pIndex Pointer to the indices for data access.
         * @param pData Pointer to the data.
         * @param pDataWeight Pointer to the data weights (optional).
         * @param slope The slope parameter for Leaky Rectified Linear activation.
         */
        case LeakyRectifiedLinear:
            kCalculateIndexedLRELUL2HingeOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight, slope);
            LAUNCHERROR("kCalculateIndexedLRELUL2HingeOutputDelta_kernel");
            break;

        /**
         * @brief Launches the CUDA kernel for calculating the output delta of the Indexed Exponential Linear activation.
         *
         * This function launches the CUDA kernel for calculating the output delta using the Indexed Exponential Linear activation function.
         * It operates on a batch of units and updates the corresponding delta values based on the specified indices and data.
         *
         * @param position The starting position of the batch.
         * @param batch The size of the batch.
         * @param stride The stride between units in memory.
         * @param pUnit Pointer to the input units.
         * @param pDelta Pointer to the output deltas.
         * @param pIndex Pointer to the indices for data access.
         * @param pData Pointer to the data.
         * @param pDataWeight Pointer to the data weights (optional).
         * @param alpha The alpha parameter for Exponential Linear activation.
         */
        case ExponentialLinear:
            kCalculateIndexedELUL2HingeOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight, alpha);
            LAUNCHERROR("kCalculateIndexedELUL2HingeOutputDelta_kernel");
            break;

        /**
         * @brief Launches the CUDA kernel for calculating the output delta of the Indexed Scaled Exponential Linear (SELU) L2 Hinge activation.
         *
         * This function launches the CUDA kernel for calculating the output delta using the Indexed Scaled Exponential Linear (SELU) L2 Hinge activation function.
         * It operates on a batch of units and updates the corresponding delta values based on the specified indices and data.
         *
         * @param position The starting position of the batch.
         * @param batch The size of the batch.
         * @param stride The stride between units in memory.
         * @param pUnit Pointer to the input units.
         * @param pDelta Pointer to the output deltas.
         * @param pIndex Pointer to the indices for data access.
         * @param pData Pointer to the data.
         * @param pDataWeight Pointer to the data weights (optional).
         * @param alpha The alpha parameter for SELU activation.
         * @param lambda The lambda parameter for SELU activation.
         */
        case ScaledExponentialLinear:
            kCalculateIndexedSELUL2HingeOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight, alpha, lambda);
            LAUNCHERROR("kCalculateIndexedSELUL2HingeOutputDelta_kernel");
            break;

        /**
         * @brief Launches the CUDA kernel for calculating the output delta of the Indexed SoftMax L2 Hinge activation.
         *
         * This function launches the CUDA kernel for calculating the output delta using the Indexed SoftMax L2 Hinge activation function.
         * It operates on a batch of units and updates the corresponding delta values based on the specified indices and data.
         *
         * @param position The starting position of the batch.
         * @param batch The size of the batch.
         * @param stride The stride between units in memory.
         * @param pUnit Pointer to the input units.
         * @param pDelta Pointer to the output deltas.
         * @param pIndex Pointer to the indices for data access.
         * @param pData Pointer to the data.
         * @param pDataWeight Pointer to the data weights (optional).
         */
        case SoftMax:
            kCalculateIndexedSoftMaxL2HingeOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            LAUNCHERROR("kCalculateIndexedSoftMaxL2HingeOutputDelta_kernel");
            break;
                
    }
}

/**
 * @brief CUDA kernel to calculate the output delta for the Hinge activation.
 *
 * This CUDA kernel calculates the output delta for the Hinge activation function.
 * It operates on a batch of units and updates the corresponding delta values based on the specified indices and data.
 *
 * @tparam T The data type of the pData array.
 * @param position The starting position of the batch.
 * @param batch The size of the batch.
 * @param stride The stride between units in memory.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the output deltas.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights (optional).
 */
template<typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateHingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat* pDataWeight)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;

    NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0;

    pUnit += uOffset;
    pDelta += uOffset;
    pData += dOffset;

    while (pos < stride)
    {
        NNFloat a = pUnit[pos];
        NNFloat t = static_cast<NNFloat>(pData[pos]);

        pDelta[pos] = w * ((a < 0.0) ? -t : 0.0);

        pos += blockDim.x * gridDim.x;

    }
}

/**
 * @brief CUDA kernel to calculate the output delta for the Hinge activation.
 *
 * This CUDA kernel calculates the output delta for the Hinge activation function.
 * It operates on a batch of units and updates the corresponding delta values based on the specified indices and data.
 *
 * @param position The starting position of the batch.
 * @param batch The size of the batch.
 * @param stride The stride between units in memory.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the output deltas.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights (optional).
 */
template<>
__global__ void LAUNCH_BOUNDS()
kCalculateHingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, unsigned char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;

    NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0;

    pUnit += uOffset;
    pDelta += uOffset;
    pData += dOffset;

    while (pos < stride)
    {
        NNFloat a = pUnit[pos];
        NNFloat t = static_cast<NNFloat>(pData[pos]) * (1.0 / 256.0);

        pDelta[pos] = w * ((a < 0.0) ? -t : 0.0);

        pos += blockDim.x * gridDim.x;
    }
}
/**
 * @brief CUDA kernel to calculate the output delta for the Hinge activation.
 *
 * This CUDA kernel calculates the output delta for the Hinge activation function.
 * It operates on a batch of units and updates the corresponding delta values based on the specified indices and data.
 *
 * @param position The starting position of the batch.
 * @param batch The size of the batch.
 * @param stride The stride between units in memory.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the output deltas.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights (optional).
 */
template<>
__global__ void LAUNCH_BOUNDS()
kCalculateHingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;

    NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0;

    pUnit += uOffset;
    pDelta += uOffset;
    pData += dOffset;

    while (pos < stride)
    {
        NNFloat a = pUnit[pos];
        NNFloat t = static_cast<NNFloat>(pData[pos]) * (1.0 / 128.0);

        pDelta[pos] = w * ((a < 0.0) ? -t : 0.0);

        pos += blockDim.x * gridDim.x;
    }
}
/**
 * @brief Calculate the output delta for the Hinge activation.
 *
 * This function calculates the output delta for the Hinge activation function.
 * It operates on a batch of units and updates the corresponding delta values based on the specified indices and data.
 *
 * @tparam T The data type of the pData array.
 * @param activation The activation function.
 * @param position The starting position of the batch.
 * @param batch The size of the batch.
 * @param stride The stride between units in memory.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the output deltas.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights (optional).
 */
template<typename T>
void kCalculateHingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, T* pData, NNFloat* pDataWeight)
{
    unsigned long threads = max(32UL, min(static_cast<unsigned long>(stride), getGpu()._threadsPerBlock));
    unsigned long blocks = static_cast<unsigned long>(batch);
    kCalculateHingeOutputDelta_kernel<<<blocks, threads>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
    LAUNCHERROR("kCalculateHingeOutputDelta_kernel");    
}
/**
 * @brief CUDA kernel to calculate the output delta for the indexed Hinge activation.
 *
 * This CUDA kernel calculates the output delta for the indexed Hinge activation function.
 * It operates on a batch of units and updates the corresponding delta values based on the specified indices and data.
 *
 * @tparam T The data type of the pData array.
 * @param position The starting position of the batch.
 * @param batch The size of the batch.
 * @param stride The stride between units in memory.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the output deltas.
 * @param pIndex Pointer to the indices for data access.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights (optional).
 */
template<typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedHingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData, NNFloat* pDataWeight)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t dOffset = dpos * stride;

    NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0;

    pUnit += uOffset;
    pDelta += uOffset;
    pData += dOffset;

    while (pos < stride)
    {
        NNFloat a = pUnit[pos];
        NNFloat t = static_cast<NNFloat>(pData[pos]);

        pDelta[pos] = w * ((a < 0.0) ? -t : 0.0);

        pos += blockDim.x * gridDim.x;
    }
}
/**
 * @brief CUDA kernel to calculate the output delta for the indexed Hinge activation.
 *
 * This CUDA kernel calculates the output delta for the indexed Hinge activation function.
 * It operates on a batch of units and updates the corresponding delta values based on the specified indices and data.
 *
 * @param position The starting position of the batch.
 * @param batch The size of the batch.
 * @param stride The stride between units in memory.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the output deltas.
 * @param pIndex Pointer to the indices for data access.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights (optional).
 */
template<>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedHingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, unsigned char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t dOffset = dpos * stride;

    NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0;

    pUnit += uOffset;
    pDelta += uOffset;
    pData += dOffset;

    while (pos < stride)
    {
        NNFloat a = pUnit[pos];
        NNFloat t = static_cast<NNFloat>(pData[pos]) * (1.0 / 256.0);

        pDelta[pos] = w * ((a < 0.0) ? -t : 0.0);

        pos += blockDim.x * gridDim.x;
    }
}
/**
 * @brief CUDA kernel to calculate the output delta for the indexed Hinge activation.
 *
 * This CUDA kernel calculates the output delta for the indexed Hinge activation function.
 * It operates on a batch of units and updates the corresponding delta values based on the specified indices and data.
 *
 * @param position The starting position of the batch.
 * @param batch The size of the batch.
 * @param stride The stride between units in memory.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the output deltas.
 * @param pIndex Pointer to the indices for data access.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights (optional).
 */
template<>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedHingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, char* pData, NNFloat* pDataWeight)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t dOffset = dpos * stride;

    NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0;

    pUnit += uOffset;
    pDelta += uOffset;
    pData += dOffset;

    while (pos < stride)
    {
        NNFloat a = pUnit[pos];
        NNFloat t = static_cast<NNFloat>(pData[pos]) * (1.0 / 128.0);

        pDelta[pos] = w * ((a < 0.0) ? -t : 0.0);

        pos += blockDim.x * gridDim.x;
    }
}
/**
 * @brief Calculate the output delta for the indexed Hinge activation.
 *
 * This function calculates the output delta for the indexed Hinge activation function.
 * It operates on a batch of units and updates the corresponding delta values based on the specified indices and data.
 *
 * @tparam T The data type of the pData array.
 * @param activation The activation function.
 * @param position The starting position of the batch.
 * @param batch The size of the batch.
 * @param stride The stride between units in memory.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the output deltas.
 * @param pIndex Pointer to the indices for data access.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weights (optional).
 */
template<typename T>
void kCalculateIndexedHingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint32_t* pIndex, T* pData, NNFloat* pDataWeight)
{
    unsigned long threads = max(32UL, min(static_cast<unsigned long>(stride), getGpu()._threadsPerBlock));
    unsigned long blocks = static_cast<unsigned long>(batch);
    kCalculateIndexedHingeOutputDelta_kernel<<<blocks, threads>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
    LAUNCHERROR("kCalculateIndexedHingeOutputDelta_kernel");    
}
/**
 * @brief CUDA kernel to calculate the output delta for the sparse raw Sigmoid activation.
 *
 * This CUDA kernel calculates the output delta for the sparse raw Sigmoid activation function.
 * It operates on a batch of units and updates the corresponding delta values based on the specified position, data weights, input units, and deltas.
 *
 * @param position The starting position of the batch.
 * @param pDataWeight Pointer to the data weights (optional).
 * @param stride The stride between units in memory.
 * @param size The size of the batch.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the output deltas.
 */
__global__ void LAUNCH_BOUNDS()
kCalculateSparseRawSigmoidOutputDelta_kernel(uint32_t position, NNFloat* pDataWeight, uint32_t stride, uint64_t size, NNFloat* pUnit,  NNFloat* pDelta)
{
    uint64_t pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    if (pos < size)
    {
        NNFloat w = cData._deltaBoost_zero;
        
        if (pDataWeight != nullptr)
        {
            uint64_t dpos = (pos / stride) + position;
            dpos = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w *= pDataWeight[dpos];
        }
        
        NNFloat a = pUnit[pos];
        pDelta[pos] = w * a * a * (1.0 - a);
    }
}

/**
 * @brief CUDA kernel to calculate the output delta for the sparse non-zero Sigmoid activation.
 *
 * This CUDA kernel calculates the output delta for the sparse non-zero Sigmoid activation function.
 * It operates on a batch of units and updates the corresponding delta values based on the specified position, batch size, stride, input units, output deltas, sparse start indices, sparse end indices, sparse indices, and data weights.
 *
 * @param position The starting position of the batch.
 * @param batch The size of the batch.
 * @param stride The stride between units in memory.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the output deltas.
 * @param pSparseStart Pointer to the start indices of the sparse data.
 * @param pSparseEnd Pointer to the end indices of the sparse data.
 * @param pSparseIndex Pointer to the sparse indices.
 * @param pDataWeight Pointer to the data weights (optional).
 */
__global__ void LAUNCH_BOUNDS()
kCalculateSparseNonZeroSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight)
{
    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        NNFloat w = cData._deltaBoost_one * ((pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0);
        uint64_t offset = pos * stride;

        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            NNFloat a = pUnit[pos2];
            pDelta[pos2] = w * (a - 1.0) * a * (1.0 - a);
            pos1 += cData._warpSize;
        }
    }
}
/**
 * @brief CUDA kernel to calculate the output delta for the sparse raw Tanh activation.
 *
 * This CUDA kernel calculates the output delta for the sparse raw Tanh activation function.
 * It operates on a batch of units and updates the corresponding delta values based on the specified position, data weights, input units, and deltas.
 *
 * @param position The starting position of the batch.
 * @param pDataWeight Pointer to the data weights (optional).
 * @param stride The stride between units in memory.
 * @param size The size of the batch.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the output deltas.
 */
__global__ void LAUNCH_BOUNDS()
kCalculateSparseRawTanhOutputDelta_kernel(uint32_t position, NNFloat* pDataWeight, uint32_t stride, uint64_t size, NNFloat* pUnit, NNFloat* pDelta)
{
    uint64_t pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    if (pos < size)
    {
        NNFloat w = 1.0;
        
        if (pDataWeight != nullptr)
        {
            uint64_t dpos = (pos / stride) + position;
            dpos = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w *= pDataWeight[dpos];
        }
        
        NNFloat a = pUnit[pos];
        pDelta[pos] = w * a * (1.0 - a * a);
    }
}
/**
 * @brief CUDA kernel to calculate the output delta for the sparse non-zero Tanh activation.
 *
 * This CUDA kernel calculates the output delta for the sparse non-zero Tanh activation function.
 * It operates on a batch of units and updates the corresponding delta values based on the specified position, batch size, stride, input units, output deltas, sparse start indices, sparse end indices, sparse indices, and data weights.
 *
 * @param position The starting position of the batch.
 * @param batch The size of the batch.
 * @param stride The stride between units in memory.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the output deltas.
 * @param pSparseStart Pointer to the start indices of the sparse data.
 * @param pSparseEnd Pointer to the end indices of the sparse data.
 * @param pSparseIndex Pointer to the sparse indices.
 * @param pDataWeight Pointer to the data weights (optional).
 */
__global__ void LAUNCH_BOUNDS()
kCalculateSparseNonZeroTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight)
{
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0;
        uint64_t offset = pos * stride;

        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            NNFloat a = pUnit[pos2];
            pDelta[pos2] = w * (a - 1.0) * (1.0 - a * a);
            pos1 += cData._warpSize;
        }
    }
}
/**
 * @brief CUDA kernel to calculate the output delta for the sparse raw Linear activation.
 *
 * This CUDA kernel calculates the output delta for the sparse raw Linear activation function.
 * It operates on a batch of units and updates the corresponding delta values based on the specified position, data weights, input units, and deltas.
 *
 * @param position The starting position of the batch.
 * @param pDataWeight Pointer to the data weights (optional).
 * @param stride The stride between units in memory.
 * @param size The size of the batch.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the output deltas.
 */
__global__ void LAUNCH_BOUNDS()
kCalculateSparseRawLinearOutputDelta_kernel(uint32_t position, NNFloat* pDataWeight, uint32_t stride, uint64_t size, NNFloat* pUnit, NNFloat* pDelta)
{
    uint64_t pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    if (pos < size)
    {
        NNFloat w = 1.0;
        
        if (pDataWeight != nullptr)
        {
            uint64_t dpos = (pos / stride) + position;
            dpos = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w *= pDataWeight[dpos];
        }
        
        NNFloat a = pUnit[pos];
        pDelta[pos] = w * a;
    }
}
/**
 * @brief CUDA kernel to calculate the output delta for the sparse non-zero Linear activation.
 *
 * This CUDA kernel calculates the output delta for the sparse non-zero Linear activation function.
 * It operates on a batch of units and updates the corresponding delta values based on the specified position, batch size, stride, input units, output deltas, sparse start indices, sparse end indices, sparse indices, and data weights.
 *
 * @param position The starting position of the batch.
 * @param batch The size of the batch.
 * @param stride The stride between units in memory.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the output deltas.
 * @param pSparseStart Pointer to the start indices of the sparse data.
 * @param pSparseEnd Pointer to the end indices of the sparse data.
 * @param pSparseIndex Pointer to the sparse indices.
 * @param pDataWeight Pointer to the data weights (optional).
 */
__global__ void LAUNCH_BOUNDS()
kCalculateSparseNonZeroLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight)
{
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[dpos] : 1.0;
        uint64_t offset = pos * stride;

        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            NNFloat a = pUnit[pos2];
            pDelta[pos2] = w * (a - 1.0);
            pos1 += cData._warpSize;
        }
    }
}
/**
 * @brief CUDA kernel to calculate the output delta for the sparse raw ReLU activation.
 *
 * This CUDA kernel calculates the output delta for the sparse raw ReLU activation function.
 * It operates on a batch of units and updates the corresponding delta values based on the specified position, data weights, input units, and deltas.
 *
 * @param position The starting position of the batch.
 * @param pDataWeight Pointer to the data weights (optional).
 * @param stride The stride between units in memory.
 * @param size The size of the batch.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the output deltas.
 */
__global__ void LAUNCH_BOUNDS()
kCalculateSparseRawRELUOutputDelta_kernel(uint32_t position, NNFloat* pDataWeight, uint32_t stride, uint64_t size, NNFloat* pUnit, NNFloat* pDelta)
{
    uint64_t pos = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (pos < size)
    {
        NNFloat w = 1.0;

        if (pDataWeight != nullptr)
        {
            uint64_t dpos = (pos / stride) + position;
            dpos = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w *= pDataWeight[dpos];
        }

        NNFloat a = pUnit[pos];
        pDelta[pos] = w * a * (a > 0.0);
    }
}
/**
 * @brief CUDA kernel to calculate the output delta for the sparse raw Leaky ReLU activation.
 *
 * This CUDA kernel calculates the output delta for the sparse raw Leaky ReLU activation function.
 * It operates on a batch of units and updates the corresponding delta values based on the specified position, data weights, input units, output deltas, and slope.
 *
 * @param position The starting position of the batch.
 * @param pDataWeight Pointer to the data weights (optional).
 * @param stride The stride between units in memory.
 * @param size The size of the batch.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the output deltas.
 * @param slope The slope parameter for the Leaky ReLU activation.
 */
__global__ void LAUNCH_BOUNDS()
kCalculateSparseRawLRELUOutputDelta_kernel(uint32_t position, NNFloat* pDataWeight, uint32_t stride, uint64_t size, NNFloat* pUnit, NNFloat* pDelta, NNFloat slope)
{
    uint64_t pos = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (pos < size)
    {
        NNFloat w = 1.0;

        if (pDataWeight != nullptr)
        {
            uint64_t dpos = (pos / stride) + position;
            dpos = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w *= pDataWeight[dpos];
        }

        NNFloat a = pUnit[pos];
        pDelta[pos] = w * a * ((a > 0.0) + (a <= 0.0) * slope);
    }
}
/**
 * @brief CUDA kernel to calculate the output delta for the sparse raw ELU activation.
 *
 * This CUDA kernel calculates the output delta for the sparse raw ELU activation function.
 * It operates on a batch of units and updates the corresponding delta values based on the specified position, data weights, input units, output deltas, and alpha parameter.
 *
 * @param position The starting position of the batch.
 * @param pDataWeight Pointer to the data weights (optional).
 * @param stride The stride between units in memory.
 * @param size The size of the batch.
 * @param pUnit Pointer to the input units.
 * @param pDelta Pointer to the output deltas.
 * @param alpha The alpha parameter for the ELU activation.
 */
__global__ void LAUNCH_BOUNDS()
kCalculateSparseRawELUOutputDelta_kernel(uint32_t position, NNFloat* pDataWeight, uint32_t stride, uint64_t size, NNFloat* pUnit, NNFloat* pDelta, NNFloat alpha)
{
    uint64_t pos = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (pos < size)
    {
        NNFloat w = 1.0;

        if (pDataWeight != nullptr)
        {
            uint64_t dpos = (pos / stride) + position;
            dpos = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w *= pDataWeight[dpos];
        }

        NNFloat a = pUnit[pos];
        pDelta[pos] = w * a * ((a > 0.0) + (a <= 0.0) * (a + alpha));
    }
}
