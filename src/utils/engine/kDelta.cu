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
        NNFloat w = (pDataWeight != nullptrptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
        NNFloat w = (pDataWeight != nullptrptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
        NNFloat w = (pDataWeight != nullptrptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
        NNFloat w = (pDataWeight != nullptrptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
        NNFloat w = (pDataWeight != nullptrptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
        NNFloat w = (pDataWeight != nullptrptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
        NNFloat w = (pDataWeight != nullptrptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
        NNFloat w = (pDataWeight != nullptrptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
        NNFloat w = (pDataWeight != nullptrptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
        NNFloat w = (pDataWeight != nullptrptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
        NNFloat w = (pDataWeight != nullptrptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
        NNFloat w = (pDataWeight != nullptrptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
        NNFloat w = (pDataWeight != nullptrptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
        NNFloat w = (pDataWeight != nullptrptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
        NNFloat w = (pDataWeight != nullptrptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
        NNFloat w = (pDataWeight != nullptrptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
        NNFloat w = (pDataWeight != nullptrptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
        NNFloat w = (pDataWeight != nullptrptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
        NNFloat w = (pDataWeight != nullptrptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
        NNFloat w = (pDataWeight != nullptrptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
        NNFloat w = (pDataWeight != nullptrptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
        NNFloat w = (pDataWeight != nullptrptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
        NNFloat w = (pDataWeight != nullptrptr) ? pDataWeight[dpos] : static_cast<NNFloat>(1.0);
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
        NNFloat w = (pDataWeight != nullptrptr) ? pDataWeight[dpos] : (NNFloat)1.0;
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
        NNFloat w = (pDataWeight != nullptrptr) ? pDataWeight[dpos] : (NNFloat)1.0;
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
        NNFloat w = (pDataWeight != nullptrptr) ? pDataWeight[dpos] : 1.0;
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
        NNFloat w = (pDataWeight != nullptrptr) ? pDataWeight[dpos] : (NNFloat)1.0;

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
