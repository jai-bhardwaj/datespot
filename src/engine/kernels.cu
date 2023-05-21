#include "GpuTypes.h"
#include "Types.h"
#include <limits>
#include "cub/util_allocator.cuh"
#include "cub/device/device_radix_sort.cuh"

/**
 * \brief GPU data structure stored in constant memory.
 */
static __constant__ GpuData cData;

/**
 * \brief Set the GPU data for the kernels.
 */
void SetKernelsGpuData()
{
    cudaError_t status = cudaMemcpyToSymbol(cData, &(getGpu()._data), sizeof(GpuData));
    RTERROR(status, "cudaMemcpyToSymbol: SetKernelsGpuData copy to cData failed");
}

/**
 * \brief Get the GPU data from the kernels.
 */
void GetKernelsGpuData()
{
    cudaError_t status = cudaMemcpyFromSymbol(&(getGpu()._data), cData, sizeof(GpuData));
    RTERROR(status, "cudaMemcpyFromSymbol: GetKernelsGpuData copy From cData failed");
}

/**
 * \brief Calculate the number of blocks needed based on the size and threads per block.
 * \param size The total size.
 * \return The number of blocks.
 */
uint32_t CalculateBlocks(uint64_t size)
{
    return (size + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;
}

/**
 * \brief GPU kernel to scale and bias the data.
 * \param pData Pointer to the data array.
 * \param size The size of the array.
 * \param scale The scale factor.
 * \param bias The bias factor.
 */
__global__ void kScaleAndBias_kernel(Float* pData, uint64_t size, Float scale, Float bias)
{
    uint64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset < size)
    {
        Float value = pData[offset];
        pData[offset] = scale * value - bias;
    }
}

/**
 * \brief Scale and bias the data on the GPU.
 * \param pData Pointer to the data array.
 * \param size The size of the array.
 * \param scale The scale factor.
 * \param bias The bias factor.
 */
void kScaleAndBias(Float* pData, uint64_t size, Float scale, Float bias)
{
    const uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    const uint32_t blocks = CalculateBlocks(size);
    kScaleAndBias_kernel<<<blocks, threadsPerBlock>>>(pData, size, scale, bias);
    LAUNCHERROR("kScaleAndBias_kernel");
}

/**
 * \brief GPU kernel to clear a unit with a bias vector.
 * \param pUnit Pointer to the unit array.
 * \param pBias Pointer to the bias vector.
 * \param stride The stride of the unit.
 * \param size The size of the unit.
 */
__global__ void kClearUnit_kernel(Float* pUnit, Float* pBias, uint32_t stride, uint64_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        uint32_t bpos = pos % stride;
        pUnit[pos] = pBias[bpos];
    }
}

/**
 * \brief Clear a unit with a bias vector on the GPU.
 * \param pUnit Pointer to the unit array.
 * \param pBias Pointer to the bias vector.
 * \param stride The stride of the unit.
 * \param batch The batch size.
 */
void kClearUnit(Float* pUnit, Float* pBias, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = CalculateBlocks(size);
    kClearUnit_kernel<<<blocks, threadsPerBlock>>>(pUnit, pBias, stride, size);
    LAUNCHERROR("kClearUnit_kernel");
}

/**
 * \brief GPU kernel to clear a unit with dual source biases.
 * \param pUnit Pointer to the unit array.
 * \param pBias1 Pointer to the first bias vector.
 * \param pBias2 Pointer to the second bias vector.
 * \param stride The stride of the unit.
 * \param size The size of the unit.
 */
__global__ void kClearDualSourceUnit_kernel(Float* pUnit, Float* pBias1, Float* pBias2, uint32_t stride, uint32_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        uint32_t bpos = pos % stride;
        pUnit[pos] = pBias1[bpos] + pBias2[bpos];
    }
}

/**
 * \brief Clear a unit with dual source biases on the GPU.
 * \param pUnit Pointer to the unit array.
 * \param pBias1 Pointer to the first bias vector.
 * \param pBias2 Pointer to the second bias vector.
 * \param stride The stride of the unit.
 * \param batch The batch size.
 */
void kClearDualSourceUnit(Float* pUnit, Float* pBias1, Float* pBias2, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = CalculateBlocks(size);
    kClearDualSourceUnit_kernel<<<blocks, threadsPerBlock>>>(pUnit, pBias1, pBias2, stride, size);
    LAUNCHERROR("kClearDualSourceUnit_kernel");
}

/**
 * \brief GPU kernel to clear a unit with triple source biases.
 * \param pUnit Pointer to the unit array.
 * \param pBias1 Pointer to the first bias vector.
 * \param pBias2 Pointer to the second bias vector.
 * \param pBias3 Pointer to the third bias vector.
 * \param stride The stride of the unit.
 * \param size The size of the unit.
 */
__global__ void kClearTripleSourceUnit_kernel(Float* pUnit, Float* pBias1, Float* pBias2, Float* pBias3, uint32_t stride, uint32_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        uint32_t bpos = pos % stride;
        pUnit[pos] = pBias1[bpos] + pBias2[bpos] + pBias3[pos];
    }
}

/**
 * \brief Clear a unit with triple source biases on the GPU.
 * \param pUnit Pointer to the unit array.
 * \param pBias1 Pointer to the first bias vector.
 * \param pBias2 Pointer to the second bias vector.
 * \param pBias3 Pointer to the third bias vector.
 * \param stride The stride of the unit.
 * \param batch The batch size.
 */
void kClearTripleSourceUnit(Float* pUnit, Float* pBias1, Float* pBias2, Float* pBias3, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = CalculateBlocks(size);
    kClearTripleSourceUnit_kernel<<<blocks, threadsPerBlock>>>(pUnit, pBias1, pBias2, pBias3, stride, size);
    LAUNCHERROR("kClearTripleSourceUnit_kernel");
}

/**
 * \brief GPU kernel to clear a unit with quadruple source biases.
 * \param pUnit Pointer to the unit array.
 * \param pBias1 Pointer to the first bias vector.
 * \param pBias2 Pointer to the second bias vector.
 * \param pBias3 Pointer to the third bias vector.
 * \param pBias4 Pointer to the fourth bias vector.
 * \param stride The stride of the unit.
 * \param size The size of the unit.
 */
__global__ void kClearQuadSourceUnit_kernel(Float* pUnit, Float* pBias1, Float* pBias2, Float* pBias3, Float* pBias4, uint32_t stride, uint32_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        uint32_t bpos = pos % stride;
        pUnit[pos] = pBias1[bpos] + pBias2[bpos] + pBias3[pos] + pBias4[pos];
    }
}

/**
 * \brief Clear a unit with quadruple source biases on the GPU.
 * \param pUnit Pointer to the unit array.
 * \param pBias1 Pointer to the first bias vector.
 * \param pBias2 Pointer to the second bias vector.
 * \param pBias3 Pointer to the third bias vector.
 * \param pBias4 Pointer to the fourth bias vector.
 * \param stride The stride of the unit.
 * \param batch The batch size.
 */
void kClearQuadSourceUnit(Float* pUnit, Float* pBias1, Float* pBias2, Float* pBias3, Float* pBias4, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = CalculateBlocks(size);
    kClearQuadSourceUnit_kernel<<<blocks, threadsPerBlock>>>(pUnit, pBias1, pBias2, pBias3, pBias4, stride, size);
    LAUNCHERROR("kClearQuadSourceUnit_kernel");
}

/**
 * \brief GPU kernel to load a sparse input unit.
 * \param position The starting position.
 * \param batch The batch size.
 * \param stride The stride of the unit.
 * \param pUnit Pointer to the unit array.
 * \param pSparseStart Pointer to the sparse start array.
 * \param pSparseEnd Pointer to the sparse end array.
 * \param pSparseIndex Pointer to the sparse index array.
 * \param pDataWeight Pointer to the data weight array.
 */
__global__ void kLoadSparseInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < batch)
    {
        uint32_t pos1 = pos + position;
        pos1 = cData._bShuffleIndices ? cData._pShuffleIndex[pos1] : pos1;
        uint64_t start = pSparseStart[pos1];
        uint64_t end = pSparseEnd[pos1];
        Float w = (pDataWeight != NULL) ? pDataWeight[pos1] : (Float)1.0;
        uint64_t offset = pos * stride;
        for (uint64_t i = start + threadIdx.x; i < end; i += blockDim.x)
        {
            uint64_t pos2 = offset + pSparseIndex[i];
            pUnit[pos2] = w;
        }
    }
}

/**
 * \brief Load a sparse input unit on the GPU.
 * \param position The starting position.
 * \param batch The batch size.
 * \param stride The stride of the unit.
 * \param pUnit Pointer to the unit array.
 * \param pSparseStart Pointer to the sparse start array.
 * \param pSparseEnd Pointer to the sparse end array.
 * \param pSparseIndex Pointer to the sparse index array.
 * \param pDataWeight Pointer to the data weight array.
 */
void kLoadSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{
    uint32_t last = position + batch;
    uint32_t count = last - position;
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (count + threadsPerBlock - 1) / threadsPerBlock;

    cudaError_t status = cudaMemset(pUnit, 0, static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride) * sizeof(Float));
    RTERROR(status, "kLoadSparseInputUnit failed");
    kLoadSparseInputUnit_kernel<<<blocks, threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight);
    LAUNCHERROR("kLoadSparseInputUnit_kernel");
}

/**
 * \brief GPU kernel to load an indexed sparse input unit.
 * \param position The starting position.
 * \param batch The batch size.
 * \param stride The stride of the unit.
 * \param pUnit Pointer to the unit array.
 * \param pIndex Pointer to the index array.
 * \param pSparseStart Pointer to the sparse start array.
 * \param pSparseEnd Pointer to the sparse end array.
 * \param pSparseIndex Pointer to the sparse index array.
 * \param pDataWeight Pointer to the data weight array.
 */
__global__ void kLoadIndexedSparseInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < batch)
    {
        uint32_t pos1 = pos + position;
        pos1 = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[pos1] : pos1];
        uint64_t start = pSparseStart[pos1];
        uint64_t end = pSparseEnd[pos1];
        Float w = (pDataWeight != NULL) ? pDataWeight[pos1] : (Float)1.0;
        uint64_t offset = pos * stride;
        for (uint64_t i = start + threadIdx.x; i < end; i += blockDim.x)
        {
            uint64_t pos2 = offset + pSparseIndex[i];
            pUnit[pos2] = w;
        }
    }
}

/**
 * \brief Load an indexed sparse input unit on the GPU.
 * \param position The starting position.
 * \param batch The batch size.
 * \param stride The stride of the unit.
 * \param pUnit Pointer to the unit array.
 * \param pIndex Pointer to the index array.
 * \param pSparseStart Pointer to the sparse start array.
 * \param pSparseEnd Pointer to the sparse end array.
 * \param pSparseIndex Pointer to the sparse index array.
 * \param pDataWeight Pointer to the data weight array.
 */
void kLoadIndexedSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{
    uint32_t last = position + batch;
    uint32_t count = last - position;
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (count + threadsPerBlock - 1) / threadsPerBlock;

    cudaError_t status = cudaMemset(pUnit, 0, static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride) * sizeof(Float));
    RTERROR(status, "kLoadIndexedSparseInputUnit failed");
    kLoadIndexedSparseInputUnit_kernel<<<blocks, threadsPerBlock>>>(position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight);
    LAUNCHERROR("kLoadIndexedSparseInputUnit_kernel");
}

/**
 * \brief GPU kernel to load a sparse analog input unit.
 * \tparam T The data type of the sparse data.
 * \param position The starting position.
 * \param batch The batch size.
 * \param stride The stride of the unit.
 * \param pUnit Pointer to the unit array.
 * \param pSparseStart Pointer to the sparse start array.
 * \param pSparseEnd Pointer to the sparse end array.
 * \param pSparseIndex Pointer to the sparse index array.
 * \param pDataWeight Pointer to the data weight array.
 * \param pSparseData Pointer to the sparse data array.
 */
template<typename T>
__global__ void kLoadSparseAnalogInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < batch)
    {
        uint32_t pos1 = pos + position;
        pos1 = cData._bShuffleIndices ? cData._pShuffleIndex[pos1] : pos1;
        uint64_t start = pSparseStart[pos1];
        uint64_t end = pSparseEnd[pos1];
        Float w = (pDataWeight != NULL) ? pDataWeight[pos1] : static_cast<Float>(1.0);
        uint64_t offset = static_cast<uint64_t>(pos) * static_cast<uint64_t>(stride);
        for (uint64_t i = start + threadIdx.x; i < end; i += blockDim.x)
        {
            uint64_t pos2 = offset + pSparseIndex[i];
            T data = pSparseData[i];
            pUnit[pos2] = w * static_cast<Float>(data);
        }
    }
}

/**
 * \brief Load a sparse analog input unit on the GPU.
 * \tparam T The data type of the sparse data.
 * \param position The starting position.
 * \param batch The batch size.
 * \param stride The stride of the unit.
 * \param pUnit Pointer to the unit array.
 * \param pSparseStart Pointer to the sparse start array.
 * \param pSparseEnd Pointer to the sparse end array.
 * \param pSparseIndex Pointer to the sparse index array.
 * \param pDataWeight Pointer to the data weight array.
 * \param pSparseData Pointer to the sparse data array.
 */
template<typename T>
void kLoadSparseAnalogInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData)
{
    uint32_t last = position + batch;
    uint32_t count = last - position;
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (count + threadsPerBlock - 1) / threadsPerBlock;

    cudaError_t status = cudaMemset(pUnit, 0, static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride) * sizeof(Float));
    RTERROR(status, "kLoadSparseAnalogInputUnit failed");
    kLoadSparseAnalogInputUnit_kernel<<<blocks, threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
    LAUNCHERROR("kLoadSparseAnalogInputUnit_kernel");
}

/**
 * \brief GPU kernel to load an indexed sparse analog input unit.
 * \tparam T The data type of the sparse data.
 * \param position The starting position.
 * \param batch The batch size.
 * \param stride The stride of the unit.
 * \param pUnit Pointer to the unit array.
 * \param pIndex Pointer to the index array.
 * \param pSparseStart Pointer to the sparse start array.
 * \param pSparseEnd Pointer to the sparse end array.
 * \param pSparseIndex Pointer to the sparse index array.
 * \param pDataWeight Pointer to the data weight array.
 * \param pSparseData Pointer to the sparse data array.
 */
template<typename T>
__global__ void kLoadIndexedSparseAnalogInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < batch)
    {
        uint32_t pos1 = pos + position;
        pos1 = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[pos1] : pos1];
        uint64_t start = pSparseStart[pos1];
        uint64_t end = pSparseEnd[pos1];
        Float w = (pDataWeight != NULL) ? pDataWeight[pos1] : static_cast<Float>(1.0);
        uint64_t offset = static_cast<uint64_t>(pos) * static_cast<uint64_t>(stride);
        for (uint64_t i = start + threadIdx.x; i < end; i += blockDim.x)
        {
            uint64_t pos2 = offset + pSparseIndex[i];
            T data = pSparseData[i];
            pUnit[pos2] = w * static_cast<Float>(data);
        }
    }
}

/**
 * \brief Load an indexed sparse analog input unit on the GPU.
 * \tparam T The data type of the sparse data.
 * \param position The starting position.
 * \param batch The batch size.
 * \param stride The stride of the unit.
 * \param pUnit Pointer to the unit array.
 * \param pIndex Pointer to the index array.
 * \param pSparseStart Pointer to the sparse start array.
 * \param pSparseEnd Pointer to the sparse end array.
 * \param pSparseIndex Pointer to the sparse index array.
 * \param pDataWeight Pointer to the data weight array.
 * \param pSparseData Pointer to the sparse data array.
 */
template<typename T>
void kLoadIndexedSparseAnalogInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData)
{
    uint32_t last = position + batch;
    uint32_t count = last - position;
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (count + threadsPerBlock - 1) / threadsPerBlock;

    cudaError_t status = cudaMemset(pUnit, 0, static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride) * sizeof(Float));
    RTERROR(status, "kLoadIndexedSparseAnalogInputUnit failed");
    kLoadIndexedSparseAnalogInputUnit_kernel<<<blocks, threadsPerBlock>>>(position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
    LAUNCHERROR("kLoadIndexedSparseAnalogInputUnit_kernel");
}

/**
 * \brief GPU kernel to load a sparse denoised input unit.
 * \param position The starting position.
 * \param batch The batch size.
 * \param stride The stride of the unit.
 * \param pUnit Pointer to the unit array.
 * \param pSparseStart Pointer to the sparse start array.
 * \param pSparseEnd Pointer to the sparse end array.
 * \param pSparseIndex Pointer to the sparse index array.
 * \param pDataWeight Pointer to the data weight array.
 * \param pRandom Pointer to the random values array.
 */
__global__ void kLoadSparseDenoisedInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, Float* pRandom)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < batch)
    {
        uint32_t pos1 = cData._bShuffleIndices ? cData._pShuffleIndex[pos + position] : pos + position;
        uint64_t start = pSparseStart[pos1];
        uint64_t end = pSparseEnd[pos1];
        Float w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[pos1] : static_cast<Float>(1.0));
        uint64_t offset = static_cast<uint64_t>(pos) * static_cast<uint64_t>(stride);
        for (uint64_t i = start + threadIdx.x; i < end; i += blockDim.x)
        {
            Float value = pRandom[i];
            uint64_t pos2 = offset + pSparseIndex[i];
            if (value >= cData._denoising_p)
                pUnit[pos2] = w;
        }
    }
}

/**
 * \brief Load a sparse denoised input unit on the GPU.
 * \param position The starting position.
 * \param batch The batch size.
 * \param stride The stride of the unit.
 * \param pUnit Pointer to the unit array.
 * \param pSparseStart Pointer to the sparse start array.
 * \param pSparseEnd Pointer to the sparse end array.
 * \param pSparseIndex Pointer to the sparse index array.
 * \param pDataWeight Pointer to the data weight array.
 * \param pRandom Pointer to the random values array.
 */
void kLoadSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, Float* pRandom)
{
    uint32_t last = position + batch;
    uint32_t count = last - position;
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (count + threadsPerBlock - 1) / threadsPerBlock;

    cudaError_t status = cudaMemset(pUnit, 0, static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride) * sizeof(Float));
    RTERROR(status, "kLoadSparseDenoisedInputUnit failed");
    kLoadSparseDenoisedInputUnit_kernel<<<blocks, threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pRandom);
    LAUNCHERROR("kLoadSparseDenoisedInputUnit_kernel");
}

/**
 * \brief GPU kernel to load an indexed sparse denoised input unit.
 * \param position The starting position.
 * \param batch The batch size.
 * \param stride The stride of the unit.
 * \param pUnit Pointer to the unit array.
 * \param pIndex Pointer to the index array.
 * \param pSparseStart Pointer to the sparse start array.
 * \param pSparseEnd Pointer to the sparse end array.
 * \param pSparseIndex Pointer to the sparse index array.
 * \param pDataWeight Pointer to the data weight array.
 * \param pRandom Pointer to the random values array.
 */
__global__ void kLoadIndexedSparseDenoisedInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, Float* pRandom)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < batch)
    {
        uint32_t pos1 = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[pos + position] : pos + position];
        uint64_t start = pSparseStart[pos1];
        uint64_t end = pSparseEnd[pos1];
        Float w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[pos1] : static_cast<Float>(1.0));
        uint64_t offset = static_cast<uint64_t>(pos) * static_cast<uint64_t>(stride);
        for (uint64_t i = start + threadIdx.x; i < end; i += blockDim.x)
        {
            Float value = pRandom[i];
            uint64_t pos2 = offset + pSparseIndex[i];
            if (value >= cData._denoising_p)
                pUnit[pos2] = w;
        }
    }
}

/**
 * \brief Load an indexed sparse denoised input unit on the GPU.
 * \param position The starting position.
 * \param batch The batch size.
 * \param stride The stride of the unit.
 * \param pUnit Pointer to the unit array.
 * \param pIndex Pointer to the index array.
 * \param pSparseStart Pointer to the sparse start array.
 * \param pSparseEnd Pointer to the sparse end array.
 * \param pSparseIndex Pointer to the sparse index array.
 * \param pDataWeight Pointer to the data weight array.
 * \param pRandom Pointer to the random values array.
 */
void kLoadIndexedSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, Float* pRandom)
{
    uint32_t last = position + batch;
    uint32_t count = last - position;
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (count + threadsPerBlock - 1) / threadsPerBlock;

    cudaError_t status = cudaMemset(pUnit, 0, static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride) * sizeof(Float));
    RTERROR(status, "kLoadIndexedSparseDenoisedInputUnit failed");
    kLoadIndexedSparseDenoisedInputUnit_kernel<<<blocks, threadsPerBlock>>>(position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pRandom);
    LAUNCHERROR("kLoadIndexedSparseDenoisedInputUnit_kernel");
}

/**
 * @brief CUDA kernel to load sparse analog denoised input units.
 *
 * @tparam T The data type of the sparse input data.
 * @param position The starting position of the input units.
 * @param batch The number of input units to load.
 * @param stride The stride of the input units.
 * @param pUnit Pointer to the memory for storing the input units.
 * @param pSparseStart Pointer to the start indices of the sparse data.
 * @param pSparseEnd Pointer to the end indices of the sparse data.
 * @param pSparseIndex Pointer to the indices of the sparse data.
 * @param pDataWeight Pointer to the weight values associated with the sparse data (optional).
 * @param pSparseData Pointer to the sparse input data.
 * @param pRandom Pointer to the random values used for denoising.
 */
template<typename T>
__global__ void kLoadSparseAnalogDenoisedInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, Float* pRandom)
{
    // Calculate the global position of the current thread
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the position is within the batch range
    if (pos < batch)
    {
        // Calculate the position in the shuffled or original data based on the flag
        uint32_t pos1 = cData._bShuffleIndices ? cData._pShuffleIndex[pos + position] : pos + position;

        // Get the start and end indices of the sparse data
        uint64_t start = pSparseStart[pos1];
        uint64_t end = pSparseEnd[pos1];

        // Calculate the weight for denoising
        Float w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[pos1] : static_cast<Float>(1.0));

        // Calculate the offset for accessing the input unit data
        uint64_t offset = static_cast<uint64_t>(pos) * static_cast<uint64_t>(stride);

        // Iterate over the sparse data elements assigned to the current thread
        for (uint64_t i = start + threadIdx.x; i < end; i += blockDim.x)
        {
            // Get the random value
            Float value = pRandom[i];

            // Calculate the position in the input unit data
            uint64_t pos2 = offset + pSparseIndex[i];

            // Get the sparse data value
            T data = pSparseData[i];

            // Check if the value satisfies the denoising threshold
            if (value >= cData._denoising_p)
                pUnit[pos2] = w * data;
        }
    }
}

/**
 * @brief Loads sparse analog denoised input units.
 *
 * @tparam T The data type of the sparse input data.
 * @param position The starting position of the input units.
 * @param batch The number of input units to load.
 * @param stride The stride of the input units.
 * @param pUnit Pointer to the memory for storing the input units.
 * @param pSparseStart Pointer to the start indices of the sparse data.
 * @param pSparseEnd Pointer to the end indices of the sparse data.
 * @param pSparseIndex Pointer to the indices of the sparse data.
 * @param pDataWeight Pointer to the weight values associated with the sparse data (optional).
 * @param pSparseData Pointer to the sparse input data.
 * @param pRandom Pointer to the random values used for denoising.
 */
template<typename T>
void kLoadSparseAnalogDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, Float* pRandom)
{
    // Calculate the last position in the batch
    uint32_t last = position + batch;

    // Calculate the count of input units
    uint32_t count = last - position;

    // Get the number of threads per block
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;

    // Calculate the number of blocks required
    uint32_t blocks = (count + threadsPerBlock - 1) / threadsPerBlock;

    // Clear the memory for the input units
    cudaError_t status = cudaMemset(pUnit, 0, static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride) * sizeof(Float));
    RTERROR(status, "kLoadSparseAnalogDenoisedInputUnit failed");

    // Launch the CUDA kernel to load the input units
    kLoadSparseAnalogDenoisedInputUnit_kernel<<<blocks, threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pRandom);
    LAUNCHERROR("kLoadSparseAnalogDenoisedInputUnit_kernel");
}

/**
 * @brief CUDA kernel to load indexed sparse analog denoised input units.
 *
 * @tparam T The data type of the sparse input data.
 * @param position The starting position of the input units.
 * @param batch The number of input units to load.
 * @param stride The stride of the input units.
 * @param pUnit Pointer to the memory for storing the input units.
 * @param pIndex Pointer to the indices of the input units.
 * @param pSparseStart Pointer to the start indices of the sparse data.
 * @param pSparseEnd Pointer to the end indices of the sparse data.
 * @param pSparseIndex Pointer to the indices of the sparse data.
 * @param pDataWeight Pointer to the weight values associated with the sparse data (optional).
 * @param pSparseData Pointer to the sparse input data.
 * @param pRandom Pointer to the random values used for denoising.
 */
template<typename T>
__global__ void kLoadIndexedSparseAnalogDenoisedInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, Float* pRandom)
{
    // Calculate the global position of the current thread
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the position is within the batch range
    if (pos < batch)
    {
        // Calculate the position in the shuffled or original index data based on the flag
        uint32_t pos1 = pIndex[pos + position];
        pos1 = cData._bShuffleIndices ? cData._pShuffleIndex[pos1] : pos1;

        // Get the start and end indices of the sparse data
        uint64_t start = pSparseStart[pos1] + threadIdx.x;
        uint64_t end = pSparseEnd[pos1];

        // Calculate the weight for denoising
        Float w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[pos1] : static_cast<Float>(1.0));

        // Calculate the offset for accessing the input unit data
        uint64_t offset = static_cast<uint64_t>(pos) * static_cast<uint64_t>(stride);

        // Iterate over the sparse data elements assigned to the current thread
        for (uint64_t i = start; i < end; i += blockDim.x)
        {
            // Get the random value
            Float value = pRandom[i];

            // Calculate the position in the input unit data
            uint64_t pos2 = offset + pSparseIndex[i];

            // Get the sparse data value
            T data = pSparseData[i];

            // Check if the value satisfies the denoising threshold
            if (value >= cData._denoising_p)
                pUnit[pos2] = w * data;
        }
    }
}

/**
 * @brief Loads indexed sparse analog denoised input units.
 *
 * @tparam T The data type of the sparse input data.
 * @param position The starting position of the input units.
 * @param batch The number of input units to load.
 * @param stride The stride of the input units.
 * @param pUnit Pointer to the memory for storing the input units.
 * @param pIndex Pointer to the indices of the input units.
 * @param pSparseStart Pointer to the start indices of the sparse data.
 * @param pSparseEnd Pointer to the end indices of the sparse data.
 * @param pSparseIndex Pointer to the indices of the sparse data.
 * @param pDataWeight Pointer to the weight values associated with the sparse data (optional).
 * @param pSparseData Pointer to the sparse input data.
 * @param pRandom Pointer to the random values used for denoising.
 */
template<typename T>
void kLoadIndexedSparseAnalogDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, Float* pRandom)
{
    // Calculate the last position in the batch
    uint32_t last = position + batch;

    // Calculate the count of input units
    uint32_t count = last - position;

    // Get the number of threads per block
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;

    // Calculate the number of blocks required
    uint32_t blocks = (count + threadsPerBlock - 1) / threadsPerBlock;

    // Clear the memory for the input units
    cudaError_t status = cudaMemset(pUnit, 0, static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride) * sizeof(Float));
    RTERROR(status, "kLoadIndexedSparseAnalogDenoisedInputUnit failed");

    // Launch the CUDA kernel to load the input units
    kLoadIndexedSparseAnalogDenoisedInputUnit_kernel<<<blocks, threadsPerBlock>>>(position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pRandom);
    LAUNCHERROR("kLoadIndexedSparseAnalogDenoisedInputUnit_kernel");
}

/**
 * @brief CUDA kernel to load input units.
 *
 * @tparam T The data type of the input data.
 * @param position The starting position of the input units.
 * @param stride The stride of the input units.
 * @param pUnit Pointer to the memory for storing the input units.
 * @param pData Pointer to the input data.
 */
template<typename T>
__global__ void kLoadInputUnit_kernel(uint32_t position, uint32_t stride, Float* pUnit, T* pData)
{
    // Calculate the blockIdx.x value
    uint32_t blockIdx_x = blockIdx.x;

    // Calculate the global position of the current thread
    uint64_t pos = (blockIdx_x * blockDim.x) + threadIdx.x;

    // Check if the position is within the stride range
    if (pos < stride)
    {
        // Calculate the position in the shuffled or original data based on the flag
        uint32_t pos1 = blockIdx_x + position;
        pos1 = cData._bShuffleIndices ? cData._pShuffleIndex[pos1] : pos1;

        // Calculate the source and destination offsets for accessing the input unit data
        uint64_t soffset = static_cast<uint64_t>(pos1) * static_cast<uint64_t>(stride) + pos;
        uint64_t doffset = static_cast<uint64_t>(blockIdx_x) * static_cast<uint64_t>(stride) + pos;

        // Store the input data value in the input unit data
        pUnit[doffset] = static_cast<Float>(pData[soffset]);
    }
}
__global__ void kLoadNormalizedInputUnit_kernel(int position, int stride, Float* pUnit, unsigned char* pData)
{
    // Get the thread id
    int tid = (blockIdx.y * blockDim.x) + threadIdx.x;

    // Check if the thread id is within the stride range
    if (tid < stride)
    {
        // Calculate the position in the shuffled or original data based on the flag
        int pos1 = cData._bShuffleIndices ? cData._pShuffleIndex[blockIdx.x + position] : (blockIdx.x + position);

        // Calculate the offset for accessing the input unit data
        int offset = blockIdx.x * stride + tid;

        // Store the normalized input data value in the input unit data
        pUnit[offset] = static_cast<Float>(pData[pos1 * stride + tid]) * cData._inv256 - cData._half;
    }
}

/**
 * @brief Loads normalized input units from device memory into the global memory.
 *
 * @param position The starting position of the input units.
 * @param stride The stride value.
 * @param pUnit Pointer to the output array in global memory.
 * @param pData Pointer to the input data in device memory.
 */
__global__ void kLoadNormalizedInputUnit_kernel(uint32_t position, uint32_t stride, Float* pUnit, unsigned char* pData)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint32_t pos1 = cData._bShuffleIndices ? cData._pShuffleIndex[blockIdx.x + position] : (blockIdx.x + position);
        uint64_t soffset = pos1 * stride + pos;
        uint64_t doffset = blockIdx.x * stride + pos;
        pUnit[doffset] = static_cast<Float>(pData[soffset]) * cData._inv128;
    }
}

/**
 * @brief Loads input units from device memory into the global memory.
 *
 * @tparam T The type of the input data.
 * @param position The starting position of the input units.
 * @param batch The number of batches.
 * @param stride The stride value.
 * @param pUnit Pointer to the output array in global memory.
 * @param pData Pointer to the input data in device memory.
 */
template<typename T>
void kLoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, T* pData)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kLoadInputUnit_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pData);
    LAUNCHERROR("kLoadInputUnit_kernel");
}

/**
 * @brief Loads input units from device memory into the global memory.
 *        Specialized version for unsigned char input data.
 *
 * @param position The starting position of the input units.
 * @param batch The number of batches.
 * @param stride The stride value.
 * @param pUnit Pointer to the output array in global memory.
 * @param pData Pointer to the input data in device memory.
 */
template<>
void kLoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, unsigned char* pData)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kLoadNormalizedInputUnit_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pData);
    LAUNCHERROR("kLoadNormalizedInputUnit_kernel");
}

/**
 * @brief Loads indexed input units from device memory into the global memory.
 *
 * @tparam T The type of the input data.
 * @param position The starting position of the input units.
 * @param batch The number of batches.
 * @param stride The stride value.
 * @param pUnit Pointer to the output array in global memory.
 * @param pIndex Pointer to the index array in device memory.
 * @param pData Pointer to the input data in device memory.
 */
template<typename T>
__global__ void kLoadIndexedInputUnit_kernel(uint32_t position, uint32_t stride, Float* pUnit, uint32_t* pIndex, T* pData, bool bShuffleIndices, uint32_t* pShuffleIndex)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < stride)
    {
        uint32_t pos1 = pIndex[bShuffleIndices ? pShuffleIndex[blockIdx.y + position] : blockIdx.y + position];
        uint64_t soffset = pos1 * stride + tid;
        uint64_t doffset = blockIdx.y * stride + tid;
        pUnit[doffset] = pData[soffset];
    }
}

/**
 * @brief Loads indexed input units from device memory into the global memory.
 *        Specialized version for unsigned char input data.
 *
 * @param position The starting position of the input units.
 * @param stride The stride value.
 * @param pUnit Pointer to the output array in global memory.
 * @param pIndex Pointer to the index array in device memory.
 * @param pData Pointer to the input data in device memory.
 * @param bShuffleIndices Flag indicating whether to shuffle indices.
 * @param pShuffleIndex Pointer to the shuffle index array in device memory.
 */
__global__ void kLoadIndexedNormalizedInputUnit_kernel(uint32_t position, uint32_t stride, Float* pUnit, uint32_t* pIndex, unsigned char* pData, bool bShuffleIndices, uint32_t* pShuffleIndex)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < stride)
    {
        uint32_t pos1 = pIndex[bShuffleIndices ? pShuffleIndex[blockIdx.y + position] : blockIdx.y + position];
        uint64_t soffset = pos1 * stride + tid;
        uint64_t doffset = blockIdx.y * stride + tid;
        unsigned char pixel = pData[soffset];
        pUnit[doffset] = static_cast<Float>(pixel) * static_cast<Float>(1.0 / 256.0) - static_cast<Float>(0.5);
    }
}

/**
 * @brief Loads indexed input units from device memory into the global memory.
 *        Specialized version for floating-point input data.
 *
 * @tparam T The type of the input data.
 * @param position The starting position of the input units.
 * @param stride The stride value.
 * @param pUnit Pointer to the output array in global memory.
 * @param pIndex Pointer to the index array in device memory.
 * @param pData Pointer to the input data in device memory.
 */
template<typename T>
__global__ void kLoadIndexedNormalizedInputUnit_kernel(uint32_t position, uint32_t stride, Float* pUnit, uint32_t* pIndex, T* pData)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < stride)
    {
        uint32_t pos1 = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[blockIdx.y + position] : blockIdx.y + position];
        uint64_t soffset = pos1 * stride + tid;
        uint64_t doffset = blockIdx.y * stride + tid;
        pUnit[doffset] = static_cast<Float>(pData[soffset]) * static_cast<Float>(1.0 / 128.0);
    }
}

/**
 * @brief Loads indexed input units from device memory into the global memory.
 *
 * @tparam T The type of the input data.
 * @param position The starting position of the input units.
 * @param batch The number of batches.
 * @param stride The stride value.
 * @param pUnit Pointer to the output array in global memory.
 * @param pIndex Pointer to the index array in device memory.
 * @param pData Pointer to the input data in device memory.
 */
template<typename T>
void kLoadIndexedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, T* pData)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kLoadIndexedNormalizedInputUnit_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pIndex, pData);
    LAUNCHERROR("kLoadIndexedNormalizedInputUnit_kernel");
}

/**
 * \brief Adds bias to the units of a tensor in parallel using CUDA.
 *
 * \param pUnit Pointer to the tensor units.
 * \param pBias Pointer to the bias values.
 * \param stride Stride of the tensor.
 * \param size Size of the tensor.
 */
__global__ void kAddBias_kernel(Float* pUnit, Float* pBias, uint32_t stride, uint32_t size)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
    {
        uint32_t bpos = tid % stride;
        pUnit[tid] += pBias[bpos];
    }
}

/**
 * \brief Adds bias to the units of a tensor using CUDA.
 *
 * \param pUnit Pointer to the tensor units.
 * \param pBias Pointer to the bias values.
 * \param stride Stride of the tensor.
 * \param batch Batch size of the tensor.
 */
void kAddBias(Float* pUnit, Float* pBias, uint32_t stride, uint32_t batch)
{
    uint32_t size = stride * batch;
    uint32_t blocks = (size + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;
    kAddBias_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pUnit, pBias, stride, size);
    LAUNCHERROR("kAddBias_kernel");
}

/**
 * \brief Adds dual bias to the units of a tensor in parallel using CUDA.
 *
 * \param pUnit Pointer to the tensor units.
 * \param pBias1 Pointer to the first bias values.
 * \param pBias2 Pointer to the second bias values.
 * \param stride Stride of the tensor.
 * \param size Size of the tensor.
 */
__global__ void kAddDualBias_kernel(Float* pUnit, Float* pBias1, Float* pBias2, uint32_t stride, uint32_t size)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
    {
        uint32_t bpos = tid % stride;
        pUnit[tid] += pBias1[bpos] + pBias2[bpos];
    }
}

/**
 * \brief Adds dual bias to the units of a tensor using CUDA.
 *
 * \param pUnit Pointer to the tensor units.
 * \param pBias1 Pointer to the first bias values.
 * \param pBias2 Pointer to the second bias values.
 * \param stride Stride of the tensor.
 * \param batch Batch size of the tensor.
 */
void kAddDualBias(Float* pUnit, Float* pBias1, Float* pBias2, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    uint32_t blocks = (size + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;
    kAddDualBias_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pUnit, pBias1, pBias2, stride, size);
    LAUNCHERROR("kAddDualBias_kernel");
}

/**
 * \brief Adds triple bias to the units of a tensor in parallel using CUDA.
 *
 * \param pUnit Pointer to the tensor units.
 * \param pBias1 Pointer to the first bias values.
 * \param pBias2 Pointer to the second bias values.
 * \param pBias3 Pointer to the third bias values.
 * \param stride Stride of the tensor.
 * \param size Size of the tensor.
 */
__global__ void kAddTripleBias_kernel(Float* pUnit, Float* pBias1, Float* pBias2, Float* pBias3, uint32_t stride, uint32_t size)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
    {
        uint32_t bpos = tid % stride;
        pUnit[tid] += pBias1[bpos] + pBias2[bpos] + pBias3[tid];
    }
}

/**
 * \brief Adds triple bias to the units of a tensor using CUDA.
 *
 * \param pUnit Pointer to the tensor units.
 * \param pBias1 Pointer to the first bias values.
 * \param pBias2 Pointer to the second bias values.
 * \param pBias3 Pointer to the third bias values.
 * \param stride Stride of the tensor.
 * \param batch Batch size of the tensor.
 */
void kAddTripleBias(Float* pUnit, Float* pBias1, Float* pBias2, Float* pBias3, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    uint32_t blocks = (size + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;
    kAddTripleBias_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pUnit, pBias1, pBias2, pBias3, stride, size);
    LAUNCHERROR("kAddTripleBias_kernel");
}

/**
 * \brief Adds quadruple bias to the units of a tensor in parallel using CUDA.
 *
 * \param pUnit Pointer to the tensor units.
 * \param pBias1 Pointer to the first bias values.
 * \param pBias2 Pointer to the second bias values.
 * \param pBias3 Pointer to the third bias values.
 * \param pBias4 Pointer to the fourth bias values.
 * \param stride Stride of the tensor.
 * \param size Size of the tensor.
 */
__global__ void kAddQuadBias_kernel(Float* pUnit, Float* pBias1, Float* pBias2, Float* pBias3, Float* pBias4, uint32_t stride, uint32_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        uint32_t bpos = pos % stride;
        pUnit[pos] += pBias1[bpos] + pBias2[bpos] + pBias3[pos] + pBias4[pos];
    }
}

/**
 * \brief Adds quadruple bias to the units of a tensor using CUDA.
 *
 * \param pUnit Pointer to the tensor units.
 * \param pBias1 Pointer to the first bias values.
 * \param pBias2 Pointer to the second bias values.
 * \param pBias3 Pointer to the third bias values.
 * \param pBias4 Pointer to the fourth bias values.
 * \param stride Stride of the tensor.
 * \param batch Batch size of the tensor.
 */
void kAddQuadBias(Float* pUnit, Float* pBias1, Float* pBias2, Float* pBias3, Float* pBias4, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    kAddQuadBias_kernel<<<blocks, threadsPerBlock>>>(pUnit, pBias1, pBias2, pBias3, pBias4, stride, size);
    LAUNCHERROR("kAddQuadBias_kernel");
}

#if (__CUDA_ARCH__ >= 600)
static const uint32_t MAXSPARSE = SM_6X_MAXSPARSE;
static const uint32_t MAXSPARSEANALOG = SM_6X_MAXSPARSEANALOG;
#elif (__CUDA_ARCH__ >= 500)
static const uint32_t MAXSPARSE = SM_5X_MAXSPARSE;
static const uint32_t MAXSPARSEANALOG = SM_5X_MAXSPARSEANALOG;
#else
static const uint32_t MAXSPARSE = SM_3X_MAXSPARSE;
static const uint32_t MAXSPARSEANALOG = SM_3X_MAXSPARSEANALOG;
#endif
/**
 * \brief CUDA kernel to calculate sparse Z values.
 *
 * \param position The position value.
 * \param stride The stride value.
 * \param pWeight Pointer to the weight array.
 * \param pSparseStart Pointer to the array of sparse start indices.
 * \param pSparseEnd Pointer to the array of sparse end indices.
 * \param pSparseIndex Pointer to the array of sparse indices.
 * \param pDataWeight Pointer to the array of data weights.
 * \param pUnit Pointer to the unit array.
 * \param beta The beta value.
 */
__global__ void kCalculateSparseZ_kernel(uint32_t position, uint32_t stride, Float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, Float* pUnit, Float beta)
{
    __shared__ uint32_t sOffset[MAXSPARSE];

    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    Float w = (pDataWeight != NULL) ? pDataWeight[position] : (Float)1.0;
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        uint32_t inputs = min(static_cast<uint32_t>(end - start), static_cast<uint32_t>(MAXSPARSE));
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos] = pSparseIndex[tstart] * stride;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __syncthreads();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;

        while (opos < stride)
        {
            opos += tgx;

            if (opos < stride)
            {
                Float unit = (beta == static_cast<Float>(0.0)) ? static_cast<Float>(0.0) : (beta * pUnit[opos]);

                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    unit += w * pWeight[offset + opos];
                }

                pUnit[opos] = unit;
            }

            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOffset[0], cData._warpSize);
            }

            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;

        __syncthreads();

        if (start < end)
        {
            beta = static_cast<Float>(1.0);
        }
    }
}

void kCalculateSparseZ(uint32_t position, uint32_t batch, uint32_t stride, Float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, Float* pUnit, Float beta)
{
    uint32_t threads = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateSparseZ_kernel<<<batch, threads>>>(position, stride, pWeight, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pUnit, beta);
    LAUNCHERROR("kCalculateSparseZ_kernel");
}
/**
 * @brief CUDA kernel for calculating indexed sparse Z.
 *
 * @param position The position value.
 * @param stride The stride value.
 * @param pWeight Pointer to the weight array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pUnit Pointer to the unit array.
 * @param beta The beta value.
 */
__global__ void kCalculateIndexedSparseZ_kernel(uint32_t position, uint32_t stride, Float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, Float* pUnit, Float beta)
{
    __shared__ uint32_t sOffset[MAXSPARSE];

    uint32_t sOpos = blockDim.x;
    position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    Float w = (pDataWeight != NULL) ? pDataWeight[position] : static_cast<Float>(1.0);
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        uint32_t inputs = static_cast<uint32_t>(min(static_cast<uint64_t>(end - start), static_cast<uint64_t>(MAXSPARSE)));
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos] = pSparseIndex[tstart] * stride;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __syncthreads();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;

        while (opos < stride)
        {
            opos += tgx;

            if (opos < stride)
            {
                Float unit = (beta == static_cast<Float>(0.0)) ? static_cast<Float>(0.0) : (beta * pUnit[opos]);

                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    unit += w * pWeight[offset + opos];
                }

                pUnit[opos] = unit;
            }

            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }

            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;

        __syncthreads();

        if (start < end)
        {
            beta = static_cast<Float>(1.0);
        }
    }
}

void kCalculateIndexedSparseZ(uint32_t position, uint32_t batch, uint32_t stride, Float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, Float* pUnit, Float beta)
{
    uint32_t threads = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateIndexedSparseZ_kernel<<<batch, threads>>>(position, stride, pWeight, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pUnit, beta);
    LAUNCHERROR("kCalculateIndexedSparseZ_kernel");
}
/**
 * \brief Calculates the sparse analog Z using CUDA kernels.
 *
 * \tparam T The data type for the sparse data.
 * \param position The position parameter.
 * \param stride The stride parameter.
 * \param pWeight Pointer to the weights.
 * \param pSparseStart Pointer to the start indices of sparse data.
 * \param pSparseEnd Pointer to the end indices of sparse data.
 * \param pSparseIndex Pointer to the indices of sparse data.
 * \param pDataWeight Pointer to the data weights.
 * \param pSparseData Pointer to the sparse data.
 * \param pUnit Pointer to the unit data.
 * \param beta The beta parameter.
 */
template<typename T>
__global__ void LAUNCH_BOUNDS256() kCalculateSparseAnalogZ_kernel(uint32_t position, uint32_t stride, Float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, Float* pUnit, Float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sIndex[MAXSPARSEANALOG];
    __shared__ T sData[MAXSPARSEANALOG];

    sOpos = blockDim.x;
    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;

    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    Float w = (pDataWeight != NULL) ? pDataWeight[position] : (Float)1.0;
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        sOpos = blockDim.x;
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        // Load sparse index and data into shared memory
        while (tstart < tend)
        {
            sIndex[pos] = pSparseIndex[tstart];
            sData[pos] = w * pSparseData[tstart];
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                Float unit = (beta == (Float)0.0) ? (Float)0.0 : (beta * pUnit[opos]);

                // Compute weighted sum using shared memory
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sIndex[i] * stride;
                    unit += pWeight[offset + opos] * sData[i];
                }

                pUnit[opos] = unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xffffffff, opos, 0);
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta = (Float)1.0;
    }
}
/**
 * \brief CUDA kernel for calculating sparse analog Z values.
 *
 * \tparam DummyTemplate (template parameter, no actual meaning)
 *
 * \param position The position value.
 * \param stride The stride value.
 * \param pWeight Pointer to the weight data.
 * \param pSparseStart Pointer to the sparse start data.
 * \param pSparseEnd Pointer to the sparse end data.
 * \param pSparseIndex Pointer to the sparse index data.
 * \param pDataWeight Pointer to the data weight.
 * \param pSparseData Pointer to the sparse data.
 * \param pUnit Pointer to the unit data.
 * \param beta The beta value.
 */
template<>
__global__ void LAUNCH_BOUNDS256() kCalculateSparseAnalogZ_kernel(uint32_t position, uint32_t stride, Float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, unsigned char* pSparseData, Float* pUnit, Float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sIndex[MAXSPARSEANALOG];
    __shared__ Float sValue[MAXSPARSEANALOG];

    sOpos = blockDim.x;
    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;

    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    Float w = (pDataWeight != NULL) ? pDataWeight[position] : (Float)1.0;
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        sOpos = blockDim.x;
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        // Load sparse index and data into shared memory
        while (tstart < tend)
        {
            sIndex[pos] = pSparseIndex[tstart];
            sValue[pos] = w * ((Float)pSparseData[tstart] * (Float)(1.0 / 256.0));
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                Float unit = (beta == (Float)0.0) ? (Float)0.0 : (beta * pUnit[opos]);

                // Compute weighted sum using shared memory
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sIndex[i] * stride;
                    unit += pWeight[offset + opos] * sValue[i];
                }

                pUnit[opos] = unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xffffffff, opos, 0);
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta = (Float)1.0;
    }
}
/**
 * @brief CUDA kernel for calculating sparse analog Z.
 *
 * @param position The position parameter.
 * @param stride The stride parameter.
 * @param pWeight Pointer to the weight data.
 * @param pSparseStart Pointer to the start of the sparse data.
 * @param pSparseEnd Pointer to the end of the sparse data.
 * @param pSparseIndex Pointer to the sparse index data.
 * @param pDataWeight Pointer to the weight data.
 * @param pSparseData Pointer to the sparse data.
 * @param pUnit Pointer to the unit data.
 * @param beta The beta parameter.
 */
template<>
__global__ void LAUNCH_BOUNDS256() kCalculateSparseAnalogZ_kernel(uint32_t position, uint32_t stride, Float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, char* pSparseData, Float* pUnit, Float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ Float sValue[MAXSPARSEANALOG];

    sOpos = blockDim.x;
    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    Float w = (pDataWeight != NULL) ? pDataWeight[position] : 1.0f;
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        sOpos = blockDim.x;
        uint32_t inputs = ullmin(end - start, static_cast<uint64_t>(MAXSPARSEANALOG));
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos] = pSparseIndex[tstart] * stride;
            sValue[pos] = w * static_cast<Float>(pSparseData[tstart]) * (1.0f / 256.0f);
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;

        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                Float unit = (beta == 0.0f) ? 0.0f : beta * pUnit[opos];
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    unit = fmaf(pWeight[opos + offset], sValue[i], unit);
                }

                pUnit[opos] = unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = SHFL(opos, 0);
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
    }
}

/**
 * @brief Template function for calculating sparse analog Z.
 *
 * @tparam T The data type.
 * @param position The position parameter.
 * @param batch The batch parameter.
 * @param stride The stride parameter.
 * @param pWeight Pointer to the weight data.
 * @param pSparseStart Pointer to the start of the sparse data.
 * @param pSparseEnd Pointer to the end of the sparse data.
 * @param pSparseIndex Pointer to the sparse index data.
 * @param pDataWeight Pointer to the weight data.
 * @param pSparseData Pointer to the sparse data.
 * @param pUnit Pointer to the unit data.
 * @param beta The beta parameter.
 */
template<typename T>
void kCalculateSparseAnalogZ(uint32_t position, uint32_t batch, uint32_t stride, Float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, Float* pUnit, Float beta)
{
    uint32_t threads = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateSparseAnalogZ_kernel<<<batch, threads>>>(position, stride, pWeight, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pUnit, beta);
    LAUNCHERROR("kCalculateSparseAnalogZ_kernel");
}


/**
 * @brief Kernel function to calculate the indexed sparse analog Z.
 *
 * @tparam T The data type.
 * @param position The position.
 * @param stride The stride.
 * @param pWeight Pointer to weights.
 * @param pIndex Pointer to indices.
 * @param pSparseStart Pointer to sparse start.
 * @param pSparseEnd Pointer to sparse end.
 * @param pSparseIndex Pointer to sparse indices.
 * @param pDataWeight Pointer to data weights.
 * @param pSparseData Pointer to sparse data.
 * @param pUnit Pointer to units.
 * @param beta The beta value.
 */
template<typename T>
__global__ void LAUNCH_BOUNDS256_kCalculateIndexedSparseAnalogZ_kernel(
    uint32_t position,
    uint32_t stride,
    Float* pWeight,
    uint32_t* pIndex,
    uint64_t* pSparseStart,
    uint64_t* pSparseEnd,
    uint32_t* pSparseIndex,
    Float* pDataWeight,
    T* pSparseData,
    Float* pUnit,
    Float beta)
{
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ T sValue[MAXSPARSEANALOG];

    position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;

        for (uint32_t i = threadIdx.x; i < inputs; i += blockDim.x)
        {
            sOffset[i] = pSparseIndex[tstart + i * blockDim.x] * stride;
            sValue[i] = (pDataWeight != NULL) ? pDataWeight[position] * pSparseData[tstart + i * blockDim.x] : pSparseData[tstart + i * blockDim.x];
        }

        __syncthreads();

        for (uint32_t opos = threadIdx.x; opos < stride; opos += blockDim.x)
        {
            Float unit = (beta == 0.0) ? 0.0 : (beta * pUnit[opos]);
            for (uint32_t i = 0; i < inputs; i++)
            {
                uint32_t offset = sOffset[i];
                unit += pWeight[offset + opos] * sValue[i];
            }

            pUnit[opos] = unit;
        }

        start = tend;
        if (start < end)
            __syncthreads();

        beta = 1.0;
    }
}
/**
 * @brief Kernel function for calculating indexed sparse analog Z.
 *
 * @tparam Float The data type of the input and output arrays.
 *
 * @param position The position index.
 * @param stride The stride value.
 * @param pWeight Pointer to the weight array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the start positions of the sparse data.
 * @param pSparseEnd Pointer to the end positions of the sparse data.
 * @param pSparseIndex Pointer to the index values of the sparse data.
 * @param pDataWeight Pointer to the weight values of the sparse data.
 * @param pSparseData Pointer to the sparse data.
 * @param pUnit Pointer to the unit array.
 * @param beta The beta value.
 */
template <>
__global__ void LAUNCH_BOUNDS256() kCalculateIndexedSparseAnalogZ_kernel(uint32_t position, uint32_t stride, const Float* pWeight, const uint32_t* pIndex, const uint64_t* pSparseStart, const uint64_t* pSparseEnd, const uint32_t* pSparseIndex, const Float* pDataWeight, const unsigned char* pSparseData, Float* pUnit, const Float beta)
{
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ Float sValue[MAXSPARSEANALOG];

    position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    const uint64_t start = pSparseStart[position];
    const uint64_t end = pSparseEnd[position];
    const Float w = (pDataWeight != NULL) ? pDataWeight[position] : static_cast<Float>(1.0);
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        const uint32_t inputs = ullmin(end - start, static_cast<uint64_t>(MAXSPARSEANALOG));
        const uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos] = pSparseIndex[tstart] * stride;
            sValue[pos] = w * (static_cast<Float>(pSparseData[tstart]) * static_cast<Float>(1.0 / 256.0));
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __syncwarp();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                Float unit = (beta == static_cast<Float>(0.0)) ? static_cast<Float>(0.0) : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    const uint32_t offset = sOffset[i];
                    unit += pWeight[offset + opos] * sValue[i];
                }
                pUnit[opos] = unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;

        if (start < end)
        {
            __syncwarp();
        }

        beta = static_cast<Float>(1.0);
    }
}

/**
 * \brief Kernel function to calculate indexed sparse analog Z.
 *
 * \tparam T The data type.
 *
 * \param position The position.
 * \param stride The stride.
 * \param pWeight Pointer to the weight data.
 * \param pIndex Pointer to the index data.
 * \param pSparseStart Pointer to the start indices of sparse data.
 * \param pSparseEnd Pointer to the end indices of sparse data.
 * \param pSparseIndex Pointer to the sparse indices.
 * \param pDataWeight Pointer to the weight data for sparse indices.
 * \param pSparseData Pointer to the sparse data.
 * \param pUnit Pointer to the unit data.
 * \param beta The beta value.
 */
template <>
__global__ void LAUNCH_BOUNDS256() kCalculateIndexedSparseAnalogZ_kernel(uint32_t position, uint32_t stride, const Float* pWeight, const uint32_t* pIndex, const uint64_t* pSparseStart, const uint64_t* pSparseEnd, const uint32_t* pSparseIndex, const Float* pDataWeight, const char* pSparseData, Float* pUnit, const Float beta)
{
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ Float sValue[MAXSPARSEANALOG];

    position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    const uint64_t start = pSparseStart[position];
    const uint64_t end = pSparseEnd[position];
    const Float w = (pDataWeight != NULL) ? pDataWeight[position] : static_cast<Float>(1.0);
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        uint32_t inputs = ullmin(end - start, static_cast<uint64_t>(MAXSPARSEANALOG));
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos] = pSparseIndex[tstart] * stride;
            sValue[pos] = w * (static_cast<Float>(pSparseData[tstart]) * static_cast<Float>(1.0 / 128.0));
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __syncwarp();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                Float unit = (beta == static_cast<Float>(0.0)) ? static_cast<Float>(0.0) : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    const uint32_t offset = sOffset[i];
                    unit += pWeight[offset + opos] * sValue[i];
                }
                pUnit[opos] = unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;

        if (start < end)
        {
            __syncwarp();
        }

        beta = static_cast<Float>(1.0);
    }
}

/**
 * \brief Function to calculate indexed sparse analog Z.
 *
 * \tparam T The data type.
 *
 * \param position The position.
 * \param batch The batch size.
 * \param stride The stride.
 * \param pWeight Pointer to the weight data.
 * \param pIndex Pointer to the index data.
 * \param pSparseStart Pointer to the start indices of sparse data.
 * \param pSparseEnd Pointer to the end indices of sparse data.
 * \param pSparseIndex Pointer to the sparse indices.
 * \param pDataWeight Pointer to the weight data for sparse indices.
 * \param pSparseData Pointer to the sparse data.
 * \param pUnit Pointer to the unit data.
 * \param beta The beta value.
 */
template <typename T>
void kCalculateIndexedSparseAnalogZ(uint32_t position, uint32_t batch, uint32_t stride, Float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, Float* pUnit, Float beta)
{
    uint32_t threads = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateIndexedSparseAnalogZ_kernel<<<batch, threads>>>(position, stride, pWeight, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pUnit, beta);
    LAUNCHERROR("kCalculateIndexedSparseAnalogZ_kernel");
}

/**
 * @brief GPU kernel to calculate sparse denoised Z values.
 * 
 * @param position The position index.
 * @param stride The stride value.
 * @param pWeight Pointer to weights array.
 * @param pSparseStart Pointer to the start index of sparse values.
 * @param pSparseEnd Pointer to the end index of sparse values.
 * @param pSparseIndex Pointer to the indices of sparse values.
 * @param pDataWeight Pointer to weight data array.
 * @param pRandom Pointer to random values array.
 * @param pUnit Pointer to the unit array.
 * @param beta The beta value.
 */
__global__ void __forceinline__
kCalculateSparseDenoisedZ_kernel(uint32_t position, uint32_t stride, Float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, Float* pRandom, Float* pUnit, Float beta)
{
    uint32_t sOffset[MAXSPARSE];
    uint32_t sOpos = blockDim.x;

    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    Float w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (Float)1.0);
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        uint32_t inputs = min(static_cast<uint32_t>(end - start), static_cast<uint32_t>(MAXSPARSEANALOG));
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            Float value = pRandom[tstart];
            sOffset[pos] = (value < cData._denoising_p) ? cData._maxUint32_t : static_cast<int32_t>(pSparseIndex[tstart]) * stride;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __syncthreads();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;

        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                Float unit = (beta == (Float)0.0) ? (Float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit += pWeight[offset + opos];
                }

                pUnit[opos] = w * unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;
        beta = (Float)1.0;
    }
}

/**
 * @brief Function to calculate sparse denoised Z values.
 * 
 * @param position The position index.
 * @param batch The batch size.
 * @param stride The stride value.
 * @param pWeight Pointer to weights array.
 * @param pSparseStart Pointer to the start index of sparse values.
 * @param pSparseEnd Pointer to the end index of sparse values.
 * @param pSparseIndex Pointer to the indices of sparse values.
 * @param pDataWeight Pointer to weight data array.
 * @param pRandom Pointer to random values array.
 * @param pUnit Pointer to the unit array.
 * @param beta The beta value.
 */
void kCalculateSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, Float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, Float* pRandom, Float* pUnit, Float beta)
{
    uint32_t threads = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateSparseDenoisedZ_kernel<<<batch, threads>>>(position, stride, pWeight, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pRandom, pUnit, beta);
    LAUNCHERROR("kCalculateSparseDenoisedZ_kernel");
}

/**
 * \brief CUDA kernel to calculate indexed sparse denoised Z.
 *
 * \param position The position value.
 * \param stride The stride value.
 * \param pWeight Pointer to the weight data.
 * \param pIndex Pointer to the index data.
 * \param pSparseStart Pointer to the sparse start data.
 * \param pSparseEnd Pointer to the sparse end data.
 * \param pSparseIndex Pointer to the sparse index data.
 * \param pDataWeight Pointer to the data weight.
 * \param pRandom Pointer to the random data.
 * \param pUnit Pointer to the unit data.
 * \param beta The beta value.
 */
__global__ void kCalculateIndexedSparseDenoisedZ_kernel(const uint32_t position, const uint32_t stride, const Float* pWeight, const uint32_t* pIndex, const uint64_t* pSparseStart, const uint64_t* pSparseEnd, const uint32_t* pSparseIndex, const Float* pDataWeight, const Float* pRandom, Float* pUnit, const Float beta)
{
    __shared__ uint32_t sOffset[MAXSPARSE];

    uint32_t sOpos;
    sOpos = blockDim.x;
    position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    const uint64_t start = pSparseStart[position];
    const uint64_t end = pSparseEnd[position];
    const Float w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (Float)1.0);
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        sOpos = blockDim.x;
        const uint32_t inputs = ullMin(end - start, (uint64_t)MAXSPARSEANALOG);
        const uint64_t tstart = start + threadIdx.x;

        for (uint32_t i = threadIdx.x; i < inputs; i += blockDim.x)
        {
            const Float value = pRandom[tstart + i * blockDim.x];
            sOffset[i] = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart + i * blockDim.x] * stride;
        }

        __syncthreads();

        const uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                const Float unit = (beta == (Float)0.0) ? (Float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    const uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit += pWeight[offset + opos];
                }

                pUnit[opos] = w * unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = SHFL(opos, 0);
        }

        start += inputs;
        beta = (Float)1.0;
    }
}

/**
 * \brief Wrapper function to launch the kernel for calculating indexed sparse denoised Z.
 *
 * \param position The position value.
 * \param batch The batch value.
 * \param stride The stride value.
 * \param pWeight Pointer to the weight data.
 * \param pIndex Pointer to the index data.
 * \param pSparseStart Pointer to the sparse start data.
 * \param pSparseEnd Pointer to the sparse end data.
 * \param pSparseIndex Pointer to the sparse index data.
 * \param pDataWeight Pointer to the data weight.
 * \param pRandom Pointer to the random data.
 * \param pUnit Pointer to the unit data.
 * \param beta The beta value.
 */
void kCalculateIndexedSparseDenoisedZ(const uint32_t position, const uint32_t batch, const uint32_t stride, const Float* pWeight, const uint32_t* pIndex, const uint64_t* pSparseStart, const uint64_t* pSparseEnd, const uint32_t* pSparseIndex, const Float* pDataWeight, const Float* pRandom, Float* pUnit, const Float beta)
{
    const uint32_t threads = min(256, stride >> getGpu()._warpBits << getGpu()._warpBits);
    kCalculateIndexedSparseDenoisedZ_kernel<<<batch, threads>>>(position, stride, pWeight, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pRandom, pUnit, beta);
    LAUNCHERROR("kCalculateIndexedSparseDenoisedZ_kernel");
}

/**
 * @brief Performs a computation using CUDA with Doxygen comments.
 *
 * This function is a CUDA kernel that performs a computation using the CUDA architecture. It operates on global memory and uses shared memory for intermediate calculations.
 *
 * @param position The position parameter.
 * @param stride The stride parameter.
 * @param pWeight Pointer to the weights.
 * @param pSparseStart Pointer to the start of the sparse data.
 * @param pSparseEnd Pointer to the end of the sparse data.
 * @param pSparseIndex Pointer to the index of the sparse data.
 * @param pDataWeight Pointer to the data weights.
 * @param pSparseData Pointer to the sparse data.
 * @param pRandom Pointer to the random values.
 * @param pUnit Pointer to the unit values.
 * @param beta The beta value.
 */
__global__ void LAUNCH_BOUNDS256(uint32_t position, uint32_t stride, Float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, Float* pSparseData, Float* pRandom, Float* pUnit, Float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ Float sValue[MAXSPARSEANALOG];

    sOpos = blockDim.x;
    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    Float w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (Float)1.0);
    pUnit += blockIdx.x * stride;

    uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);

    while (start < end)
    {
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            Float value = pRandom[tstart];
            sOffset[pos] = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart] * stride;
            sValue[pos] = pSparseData[tstart] * w;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;

        for (uint32_t o = 0; o < stride; o += cData._warpSize)
        {
            Float unit = (beta == (Float)0.0) ? (Float)0.0 : pUnit[opos];
            for (uint32_t i = 0; i < inputs; i++)
            {
                uint32_t offset = sOffset[i];
                if (offset != cData._maxUint32_t)
                    unit += pWeight[offset + opos] * sValue[i];
            }

            pUnit[opos] = unit;
            opos += cData._warpSize;
        }

        __syncthreads();
        if (threadIdx.x == 0)
        {
            opos = atomicAdd(&sOpos, cData._warpSize);
        }
        opos = __shfl(opos, 0);

        start = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta = (Float)1.0;
    }
}

/**
 * Perform a kernel operation with 256 launch bounds.
 *
 * @param position The position of the kernel operation.
 * @param stride The stride value.
 * @param pWeight Pointer to the weight data.
 * @param pSparseStart Pointer to the start indices of sparse data.
 * @param pSparseEnd Pointer to the end indices of sparse data.
 * @param pSparseIndex Pointer to the sparse index data.
 * @param pDataWeight Pointer to the weight data used in denoising.
 * @param pSparseData Pointer to the sparse data.
 * @param pRandom Pointer to the random data.
 * @param pUnit Pointer to the unit data.
 * @param beta The beta value used in the kernel operation.
 */
__global__ void LAUNCH_BOUNDS256(uint32_t position, uint32_t stride, Float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, unsigned char* pSparseData, Float* pRandom, Float* pUnit, Float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ int32_t sOffset[MAXSPARSEANALOG];
    __shared__ Float sValue[MAXSPARSEANALOG];

    sOpos = blockDim.x;
    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    Float w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (Float)1.0);
    pUnit += blockIdx.x * stride;

    uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);

    while (start < end)
    {
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            Float value = pRandom[tstart];
            sOffset[pos] = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart] * stride;
            sValue[pos] = (Float)pSparseData[tstart] * (Float)(1.0 / 256.0) * w;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;

        for (uint32_t o = 0; o < stride; o += cData._warpSize)
        {
            Float unit = (beta == (Float)0.0) ? (Float)0.0 : (beta * pUnit[opos]);
            for (uint32_t i = 0; i < inputs; i++)
            {
                uint32_t offset = sOffset[i];
                if (offset != cData._maxUint32_t)
                    unit += pWeight[offset + opos] * sValue[i];
            }

            pUnit[opos] = unit;
            opos += cData._warpSize;
        }

        __syncthreads();
        if (threadIdx.x == 0)
        {
            opos = atomicAdd(&sOpos, cData._warpSize);
        }
        opos = __shfl(opos, 0);

        start = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta = (Float)1.0;
    }
}

/**
 * @brief Calculates the sparse analog denoised Z using a CUDA kernel.
 *
 * @tparam T The data type of the sparse data.
 * @param position The position parameter.
 * @param stride The stride parameter.
 * @param pWeight Pointer to the weight data.
 * @param pSparseStart Pointer to the start indices of the sparse data.
 * @param pSparseEnd Pointer to the end indices of the sparse data.
 * @param pSparseIndex Pointer to the indices of the sparse data.
 * @param pDataWeight Pointer to the weight data.
 * @param pSparseData Pointer to the sparse data.
 * @param pRandom Pointer to the random data.
 * @param pUnit Pointer to the unit data.
 * @param beta The beta parameter.
 */
template<typename T>
__global__ void kCalculateSparseAnalogDenoisedZ_kernel(uint32_t position, uint32_t stride, Float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, Float* pRandom, Float* pUnit, Float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ Float sValue[MAXSPARSEANALOG];

    sOpos = blockDim.x;
    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    Float w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (Float)1.0);
    pUnit += blockIdx.x * stride;

    // Load pWeight and pSparseData into shared memory
    __shared__ Float sWeight[MAXSPARSEANALOG][MAX_WEIGHT_STRIDE];
    __shared__ T sSparseData[MAXSPARSEANALOG];

    for (uint32_t i = threadIdx.x; i < stride; i += blockDim.x)
    {
        for (uint32_t j = 0; j < MAXSPARSEANALOG; j++)
        {
            sWeight[j][i] = pWeight[j * stride + i];
        }
    }

    __syncthreads();

    while (start < end)
    {
        uint32_t inputs = ullmin(end - start, static_cast<uint64_t>(MAXSPARSEANALOG));
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            Float value = pRandom[tstart];
            sOffset[pos] = (value < cData._denoising_p) ? cData._maxUint32_t : static_cast<uint32_t>(pSparseIndex[tstart]) * stride;
            sValue[pos] = static_cast<Float>(pSparseData[tstart]) * (Float)(1.0 / 128.0) * w;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __syncthreads();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;

        while (opos < stride)
        {
            opos += tgx;

            if (opos < stride)
            {
                Float unit = (beta == (Float)0.0) ? (Float)0.0 : (beta * pUnit[opos]);

                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit += sWeight[i][opos + offset] * sValue[i];
                }

                pUnit[opos] = unit;
            }

            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }

            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;

        __syncthreads();
        beta = (Float)1.0;
    }
}

/**
 * @brief Calculates the sparse analog denoised Z.
 *
 * @tparam T The data type of the sparse data.
 * @param position The position parameter.
 * @param batch The batch parameter.
 * @param stride The stride parameter.
 * @param pWeight Pointer to the weight data.
 * @param pSparseStart Pointer to the start indices of the sparse data.
 * @param pSparseEnd Pointer to the end indices of the sparse data.
 * @param pSparseIndex Pointer to the indices of the sparse data.
 * @param pDataWeight Pointer to the weight data.
 * @param pSparseData Pointer to the sparse data.
 * @param pRandom Pointer to the random data.
 * @param pUnit Pointer to the unit data.
 * @param beta The beta parameter.
 */
template<typename T>
void kCalculateSparseAnalogDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, Float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, Float* pRandom, Float* pUnit, Float beta)
{
    uint32_t threads = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);

    // Copy cData to constant memory
    Constants cDataConst;
    cudaMemcpyToSymbol(cData, &cDataConst, sizeof(Constants));

    kCalculateSparseAnalogDenoisedZ_kernel<<<batch, threads>>>(position, stride, pWeight, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pRandom, pUnit, beta);
    LAUNCHERROR("kCalculateSparseAnalogDenoisedZ_kernel");
}

/**
 * Performs the LAUNCH_BOUNDS256 kernel operation.
 *
 * @tparam T The data type.
 * @param stride The stride value.
 * @param pWeight Pointer to the weight data.
 * @param pIndex Pointer to the index data.
 * @param pSparseStart Pointer to the sparse start data.
 * @param pSparseEnd Pointer to the sparse end data.
 * @param pSparseIndex Pointer to the sparse index data.
 * @param pDataWeight Pointer to the data weight.
 * @param pSparseData Pointer to the sparse data.
 * @param pRandom Pointer to the random data.
 * @param pUnit Pointer to the unit data.
 * @param beta The beta value.
 */
template <typename T>
__global__ void LAUNCH_BOUNDS256(uint32_t stride, Float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, Float* pRandom, Float* pUnit, Float beta)
{
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ T sValue[MAXSPARSEANALOG];

    uint32_t position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[blockIdx.x] : blockIdx.x];
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    Float w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : 1.0);
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        uint64_t inputs = min(end - start, static_cast<uint64_t>(MAXSPARSEANALOG));
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;

        for (uint32_t i = threadIdx.x; i < inputs; i += blockDim.x)
        {
            Float value = pRandom[tstart + i * blockDim.x];
            sOffset[i] = (value < cData._denoising_p) ? cData._maxUint32_t : static_cast<int32_t>(pSparseIndex[tstart + i * blockDim.x]) * stride;
            sValue[i] = pSparseData[tstart + i * blockDim.x] * w;
        }

        __syncthreads();

        uint32_t opos = threadIdx.x;

        while (opos < stride)
        {
            Float unit = (beta == 0.0) ? 0.0 : (beta * pUnit[opos]);
            for (uint32_t i = 0; i < inputs; i++)
            {
                uint32_t offset = sOffset[i];
                if (offset != cData._maxUint32_t)
                    unit += pWeight[offset + opos] * sValue[i];
            }

            pUnit[opos] = unit;

            opos += blockDim.x;
        }

        start = tend;

        if (start < end)
            __syncthreads();

        beta = 1.0;
    }
}

/**
 * @brief Computes the LAUNCH_BOUNDS256 kernel.
 *
 * This kernel performs computations using CUDA on a GPU.
 *
 * @tparam T The data type of the input.
 * @param position The position parameter.
 * @param stride The stride parameter.
 * @param pWeight Pointer to the weight data.
 * @param pIndex Pointer to the index data.
 * @param pSparseStart Pointer to the sparse start data.
 * @param pSparseEnd Pointer to the sparse end data.
 * @param pSparseIndex Pointer to the sparse index data.
 * @param pDataWeight Pointer to the data weight.
 * @param pSparseData Pointer to the sparse data.
 * @param pRandom Pointer to the random data.
 * @param pUnit Pointer to the unit data.
 * @param beta The beta parameter.
 */
template<typename T>
__global__ void LAUNCH_BOUNDS256(const T position, const T stride, const T* pWeight, const T* pIndex, const T* pSparseStart, const T* pSparseEnd, const T* pSparseIndex, const T* pDataWeight, const unsigned char* pSparseData, const T* pRandom, T* pUnit, const T beta)
{
    constexpr int MAXSPARSEANALOG = 128;
    __shared__ T sOpos;
    __shared__ int32_t sOffset[MAXSPARSEANALOG];
    __shared__ T sValue[MAXSPARSEANALOG];

    sOpos = blockDim.x;
    position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    const T start = pSparseStart[position];
    const T end = pSparseEnd[position];
    const T w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : T(1.0));
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        const T inputs = min(end - start, T(MAXSPARSEANALOG));
        const T tend = start + inputs;
        T tstart = start + threadIdx.x;
        T pos = threadIdx.x;

        while (tstart < tend)
        {
            const T value = pRandom[tstart];
            sOffset[pos] = (value < cData._denoising_p) ? cData._maxUint32_t : static_cast<int32_t>(pSparseIndex[tstart]) * stride;
            sValue[pos] = static_cast<T>(pSparseData[tstart]) * (T(1.0) / T(256.0)) * w;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence_block();
        __syncthreads();

        const T tgx = threadIdx.x & cData._warpMask;
        const T opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            const T laneId = threadIdx.x % warpSize;
            const T warpId = threadIdx.x / warpSize;
            const T isLaneActive = (opos < stride) ? T(1) : T(0);
            const T unit = (beta == T(0.0)) ? T(0.0) : (beta * pUnit[opos]);
            T result = T(0.0);

            for (T i = 0; i < inputs; i += warpSize)
            {
                const T offset = sOffset[i + laneId];
                if (offset != cData._maxUint32_t)
                {
                    const T weight = pWeight[offset + opos];
                    result += weight * sValue[i + laneId];
                }
            }

            result = warpReduceSum(result);

            if (laneId == 0 && isLaneActive)
            {
                pUnit[opos] = unit + result;
            }

            opos += warpSize;
        }

        start = tend;

        if (start < end)
        {
            __threadfence_block();
            __syncthreads();
        }

        beta = T(1.0);
    }
}

/**
 * @brief Computes the warpReduceSum function.
 *
 * This function performs a reduction sum within a warp.
 *
 * @tparam T The data type of the input.
 * @param val The input value to be reduced.
 * @return The reduced value.
 */
template <typename T>
__device__ T warpReduceSum(T val)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

/**
 * @brief CUDA kernel to calculate indexed sparse analog denoised Z.
 * 
 * This kernel performs calculations on indexed sparse data in parallel using CUDA.
 * 
 * @tparam T The data type of the sparse data.
 * @param position The position index.
 * @param stride The stride value.
 * @param pWeight Pointer to the weight data.
 * @param pIndex Pointer to the index data.
 * @param pSparseStart Pointer to the start indices of sparse data.
 * @param pSparseEnd Pointer to the end indices of sparse data.
 * @param pSparseIndex Pointer to the index values of sparse data.
 * @param pDataWeight Pointer to the weight data of sparse data.
 * @param pSparseData Pointer to the sparse data.
 * @param pRandom Pointer to the random data.
 * @param pUnit Pointer to the unit data.
 * @param beta The beta value.
 */
template<typename T>
__global__ void kCalculateIndexedSparseAnalogDenoisedZ_kernel(uint32_t position, uint32_t stride, Float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, Float* pRandom, Float* pUnit, Float beta)
{
    const uint32_t MAXSPARSEANALOG = 32;  // Adjust the value as needed

    uint32_t sOffset[MAXSPARSEANALOG];
    Float sValue[MAXSPARSEANALOG];

    position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    Float w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (Float)1.0);
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        uint32_t inputs = ullmin(end - start, static_cast<uint64_t>(MAXSPARSEANALOG));
        uint64_t tend = start + inputs;

        // Load sOffset and sValue
        for (uint32_t i = threadIdx.x; i < inputs; i += blockDim.x)
        {
            Float value = pRandom[start + i];
            sOffset[i] = (value < cData._denoising_p) ? cData._maxUint32_t : static_cast<uint32_t>(pSparseIndex[start + i]) * stride;
            sValue[i] = static_cast<Float>(pSparseData[start + i]) * static_cast<Float>(1.0 / 128.0) * w;
        }

        __syncthreads();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;

        while (opos < stride)
        {
            opos += tgx;

            if (opos < stride)
            {
                Float unit = (beta == (Float)0.0) ? (Float)0.0 : (beta * pUnit[opos]);

                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit += pWeight[offset + opos] * sValue[i];
                }

                pUnit[opos] = unit;
            }

            opos -= tgx;

            if (tgx == 0)
            {
                // Reduction to calculate sum of sOpos across the warp
                uint32_t sum = warpSum(opos);
                if (threadIdx.x == 0)
                {
                    atomicAdd(&sOpos, sum);
                }
            }

            __syncthreads();
        }

        start = tend;

        if (start < end)
        {
            __syncthreads();
        }

        beta = static_cast<Float>(1.0);
    }
}

/**
 * @brief Function to calculate indexed sparse analog denoised Z.
 * 
 * This function launches the CUDA kernel for calculating indexed sparse analog denoised Z.
 * 
 * @tparam T The data type of the sparse data.
 * @param position The position index.
 * @param batch The batch size.
 * @param stride The stride value.
 * @param pWeight Pointer to the weight data.
 * @param pIndex Pointer to the index data.
 * @param pSparseStart Pointer to the start indices of sparse data.
 * @param pSparseEnd Pointer to the end indices of sparse data.
 * @param pSparseIndex Pointer to the index values of sparse data.
 * @param pDataWeight Pointer to the weight data of sparse data.
 * @param pSparseData Pointer to the sparse data.
 * @param pRandom Pointer to the random data.
 * @param pUnit Pointer to the unit data.
 * @param beta The beta value.
 */
template<typename T>
void kCalculateIndexedSparseAnalogDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, Float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, Float* pRandom, Float* pUnit, Float beta)
{
    uint32_t threads = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateIndexedSparseAnalogDenoisedZ_kernel<<<batch, threads>>>(position, stride, pWeight, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pRandom, pUnit, beta);
    LAUNCHERROR("kCalculateIndexedSparseAnalogDenoisedZ_kernel");
}

/**
 * @brief Kernel function to calculate the transposed sparse matrix.
 *
 * This function is executed on the GPU and performs the calculation of the transposed
 * sparse matrix for a given position and batch.
 *
 * @param position The position of the matrix.
 * @param batch The batch size.
 * @param pSparseStart Pointer to the start positions of the sparse matrix.
 * @param pSparseEnd Pointer to the end positions of the sparse matrix.
 * @param pSparseIndex Pointer to the index values of the sparse matrix.
 * @param pSparseTransposedEnd Pointer to the end positions of the transposed sparse matrix.
 * @param pSparseTransposedIndex Pointer to the index values of the transposed sparse matrix.
 */
__global__ void kCalculateSparseTransposedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex)
{
    // Calculate thread and block indices
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        // Calculate position based on shuffle indices if enabled
        position = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;    
        
        // Calculate start and end positions of the sparse matrix
        uint64_t startPos = pSparseStart[position] + tgx;
        uint64_t endPos = pSparseEnd[position];

        // Iterate over the sparse matrix elements
        for (uint64_t start = startPos; start < endPos; start += cData._warpSize)
        {
            // Retrieve the index value
            uint32_t index = pSparseIndex[start];
            
            // Add the batch position to the transposed sparse matrix
            uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos] = bpos;
        }
    }
}

/**
 * \brief CUDA kernel to calculate the weighted sparse transposed matrix.
 *
 * \param position The position parameter.
 * \param batch The batch size.
 * \param pSparseStart Pointer to the start indices of the sparse matrix.
 * \param pSparseEnd Pointer to the end indices of the sparse matrix.
 * \param pSparseIndex Pointer to the indices of the sparse matrix.
 * \param pDataWeight Pointer to the weights of the data.
 * \param pSparseTransposedEnd Pointer to the end indices of the transposed sparse matrix.
 * \param pSparseTransposedIndex Pointer to the indices of the transposed sparse matrix.
 * \param pSparseTransposedData Pointer to the data of the transposed sparse matrix.
 */
__global__ void kCalculateWeightedSparseTransposedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, Float* pSparseTransposedData)
{
    uint32_t bpos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (bpos < batch)
    {
        position = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start = pSparseStart[position] + threadIdx.x;
        uint64_t end = pSparseEnd[position];
        Float w = pDataWeight[position];
        while (start < end)
        {
            uint32_t index = pSparseIndex[start];
            uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos] = bpos;
            pSparseTransposedData[opos] = w;
            start += blockDim.x;
        }
    }
}

/**
 * \brief Function to calculate the sparse transposed matrix.
 *
 * \param position The position parameter.
 * \param batch The batch size.
 * \param pSparseStart Pointer to the start indices of the sparse matrix.
 * \param pSparseEnd Pointer to the end indices of the sparse matrix.
 * \param pSparseIndex Pointer to the indices of the sparse matrix.
 * \param pDataWeight Pointer to the weights of the data. Set to NULL if not used.
 * \param pSparseTransposedEnd Pointer to the end indices of the transposed sparse matrix.
 * \param pSparseTransposedIndex Pointer to the indices of the transposed sparse matrix.
 * \param pSparseTransposedData Pointer to the data of the transposed sparse matrix.
 */
void kCalculateSparseTransposedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, Float* pSparseTransposedData)
{
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blockSize = threadsPerBlock;
    uint32_t numBlocks = (batch + blockSize - 1) / blockSize;
    
    if (pDataWeight == NULL)
    {
        kCalculateSparseTransposedMatrix_kernel<<<numBlocks, blockSize>>>(position, batch, pSparseStart, pSparseEnd, pSparseIndex, pSparseTransposedEnd, pSparseTransposedIndex);
        LAUNCHERROR("kCalculateSparseTransposedMatrix_kernel");
    }
    else
    {
        kCalculateWeightedSparseTransposedMatrix_kernel<<<numBlocks, blockSize>>>(position, batch, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);
        LAUNCHERROR("kCalculateWeightedSparseTransposedMatrix_kernel");
    }
}

/**
 * @brief CUDA kernel to calculate the transposed matrix for indexed sparse data.
 *
 * @param position The starting position in the index array.
 * @param batch The number of elements to process.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the start indices of the sparse data.
 * @param pSparseEnd Pointer to the end indices of the sparse data.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pSparseTransposedEnd Pointer to the end indices of the transposed sparse data.
 * @param pSparseTransposedIndex Pointer to the transposed sparse index array.
 */
__global__ void kCalculateIndexedSparseTransposedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex)
{
    uint32_t bpos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (bpos < batch)
    {
        position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        uint64_t start = pSparseStart[position] + threadIdx.x;
        uint64_t end = pSparseEnd[position];
        while (start < end)
        {
            uint32_t index = pSparseIndex[start];
            uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos] = bpos;
            start += blockDim.x;
        }
    }
}


/**
 * @brief CUDA kernel to calculate the indexed weighted sparse transposed matrix.
 *
 * @param position The starting position.
 * @param batch The number of batches.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the start array of the sparse matrix.
 * @param pSparseEnd Pointer to the end array of the sparse matrix.
 * @param pSparseIndex Pointer to the index array of the sparse matrix.
 * @param pDataWeight Pointer to the weight array of the data.
 * @param pSparseTransposedEnd Pointer to the end array of the transposed sparse matrix.
 * @param pSparseTransposedIndex Pointer to the index array of the transposed sparse matrix.
 * @param pSparseTransposedData Pointer to the data array of the transposed sparse matrix.
 */
__global__ void kCalculateIndexedWeightedSparseTransposedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, Float* pSparseTransposedData)
{
    uint32_t bpos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (bpos < batch)
    {
        position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        uint64_t start = pSparseStart[position] + threadIdx.x;
        uint64_t end = pSparseEnd[position];
        Float w = pDataWeight[position];
        while (start < end)
        {
            uint32_t index = pSparseIndex[start];
            uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos] = bpos;
            pSparseTransposedData[opos] = w;
            start += blockDim.x;
        }
    }
}

/**
 * @brief Function to calculate the indexed sparse transposed matrix.
 *
 * @param position The starting position.
 * @param batch The number of batches.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the start array of the sparse matrix.
 * @param pSparseEnd Pointer to the end array of the sparse matrix.
 * @param pSparseIndex Pointer to the index array of the sparse matrix.
 * @param pDataWeight Pointer to the weight array of the data.
 * @param pSparseTransposedEnd Pointer to the end array of the transposed sparse matrix.
 * @param pSparseTransposedIndex Pointer to the index array of the transposed sparse matrix.
 * @param pSparseTransposedData Pointer to the data array of the transposed sparse matrix.
 */
void kCalculateIndexedSparseTransposedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, Float* pSparseTransposedData)
{
    uint32_t threadsPerBlock = getGpu()._warpSize;
    uint32_t numBlocks = (batch + threadsPerBlock - 1) / threadsPerBlock;
    
    if (pDataWeight == NULL)
    {
        kCalculateIndexedSparseTransposedMatrix_kernel<<<numBlocks, threadsPerBlock>>>(position, batch, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseTransposedEnd, pSparseTransposedIndex);
        LAUNCHERROR("kCalculateIndexedSparseTransposedMatrix_kernel");
    }
    else
    {
        kCalculateIndexedWeightedSparseTransposedMatrix_kernel<<<numBlocks, threadsPerBlock>>>(position, batch, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);
        LAUNCHERROR("kCalculateIndexedWeightedSparseTransposedMatrix_kernel");
    }
}

/**
 * @brief CUDA kernel to calculate the sparse transposed denoised matrix.
 *
 * @param position The starting position in the sparse matrix.
 * @param batch The number of elements to process.
 * @param pSparseStart Pointer to the start indices of each sparse row.
 * @param pSparseEnd Pointer to the end indices of each sparse row.
 * @param pSparseIndex Pointer to the indices of non-zero elements in the sparse matrix.
 * @param pRandom Pointer to the random values associated with each sparse element.
 * @param pSparseTransposedEnd Pointer to the end indices of each transposed sparse row.
 * @param pSparseTransposedIndex Pointer to the indices of transposed non-zero elements.
 */
__global__ void kCalculateSparseTransposedDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex)
{
    uint32_t bpos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (bpos < batch)
    {
        position = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start = pSparseStart[position] + threadIdx.x;
        uint64_t end = pSparseEnd[position];
        while (start < end)
        {
            Float rnd = pRandom[start];
            uint32_t index = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                atomicAdd(&pSparseTransposedEnd[index], 1);
                uint32_t opos = atomicSub(&pSparseTransposedEnd[index], 1) - 1;
                pSparseTransposedIndex[opos] = bpos;
            }
            start += blockDim.x;
        }
    }
}

/**
 * @brief GPU kernel to calculate the weighted sparse transposed denoised matrix.
 *
 * @param position The position.
 * @param batch The batch size.
 * @param pSparseStart Pointer to the start indices of the sparse matrix.
 * @param pSparseEnd Pointer to the end indices of the sparse matrix.
 * @param pSparseIndex Pointer to the indices of the sparse matrix.
 * @param pDataWeight Pointer to the weights of the data.
 * @param pRandom Pointer to the random numbers.
 * @param pSparseTransposedEnd Pointer to the end indices of the transposed sparse matrix.
 * @param pSparseTransposedIndex Pointer to the indices of the transposed sparse matrix.
 * @param pSparseTransposedData Pointer to the data of the transposed sparse matrix.
 */
__global__ void kCalculateWeightedSparseTransposedDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, Float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, Float* pSparseTransposedData)
{
    uint32_t bpos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (bpos < batch)
    {
        position = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start = pSparseStart[position] + threadIdx.x;
        uint64_t end = pSparseEnd[position];
        Float w = cData._denoising_q * pDataWeight[position];
        while (start < end)
        {
            Float rnd = pRandom[start];
            uint32_t index = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos] = bpos;
                pSparseTransposedData[opos] = w;
            }
            start += blockDim.x;
        }
    }
}

/**
 * @brief Function to calculate the sparse transposed denoised matrix.
 *
 * @param position The position.
 * @param batch The batch size.
 * @param pSparseStart Pointer to the start indices of the sparse matrix.
 * @param pSparseEnd Pointer to the end indices of the sparse matrix.
 * @param pSparseIndex Pointer to the indices of the sparse matrix.
 * @param pDataWeight Pointer to the weights of the data.
 * @param pRandom Pointer to the random numbers.
 * @param pSparseTransposedEnd Pointer to the end indices of the transposed sparse matrix.
 * @param pSparseTransposedIndex Pointer to the indices of the transposed sparse matrix.
 * @param pSparseTransposedData Pointer to the data of the transposed sparse matrix.
 */
void kCalculateSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, Float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, Float* pSparseTransposedData)
{
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blockSize = threadsPerBlock;
    uint32_t numBlocks = (batch + blockSize - 1) / blockSize;

    uint32_t sharedMemSize = 0;  // Assuming no shared memory usage in the kernel
    
    if (pDataWeight == NULL)
    {
        kCalculateSparseTransposedDenoisedMatrix_kernel<<<numBlocks, blockSize, sharedMemSize>>>(position, batch, pSparseStart, pSparseEnd, pSparseIndex, pRandom, pSparseTransposedEnd, pSparseTransposedIndex);
        LAUNCHERROR("kCalculateSparseTransposedDenoisedMatrix_kernel");
    }
    else
    {
        kCalculateWeightedSparseTransposedDenoisedMatrix_kernel<<<numBlocks, blockSize, sharedMemSize>>>(position, batch, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pRandom, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);
        LAUNCHERROR("kCalculateWeightedSparseTransposedDenoisedMatrix_kernel");
    }
}

/**
 * @brief CUDA kernel to calculate an indexed sparse transposed denoised matrix.
 *
 * This kernel processes a batch of data and updates the sparse transposed denoised matrix
 * based on the indexed values. It uses CUDA atomic operations to ensure correct updates
 * to the shared memory.
 *
 * @param position The starting position of the batch in the index array.
 * @param batch The size of the batch to process.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the array containing the start positions of the sparse matrix.
 * @param pSparseEnd Pointer to the array containing the end positions of the sparse matrix.
 * @param pSparseIndex Pointer to the array containing the indices of the sparse matrix.
 * @param pRandom Pointer to the array containing random values.
 * @param pSparseTransposedEnd Pointer to the array containing the end positions of the sparse transposed matrix.
 * @param pSparseTransposedIndex Pointer to the array containing the indices of the sparse transposed matrix.
 */
__global__ void kCalculateIndexedSparseTransposedDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex)
{
    uint32_t bpos = blockIdx.x * blockDim.x + threadIdx.x;

    if (bpos < batch)
    {
        position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        uint64_t start = pSparseStart[position] + threadIdx.x;
        uint64_t end = pSparseEnd[position];
        while (start < end)
        {
            Float rnd = pRandom[start];
            uint32_t index = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                atomicAdd(&pSparseTransposedEnd[index], 1);
                uint32_t opos = atomicSub(&pSparseTransposedEnd[index], 1) - 1;
                pSparseTransposedIndex[opos] = bpos;
            }
            start += blockDim.x;
        }
    }
}

/**
 * @brief CUDA kernel to calculate the indexed weighted sparse transposed denoised matrix.
 *
 * @param position The position.
 * @param batch The batch size.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the array of sparse start positions.
 * @param pSparseEnd Pointer to the array of sparse end positions.
 * @param pSparseIndex Pointer to the array of sparse indices.
 * @param pDataWeight Pointer to the array of data weights.
 * @param pRandom Pointer to the random array.
 * @param pSparseTransposedEnd Pointer to the array of sparse transposed end positions.
 * @param pSparseTransposedIndex Pointer to the array of sparse transposed indices.
 * @param pSparseTransposedData Pointer to the array of sparse transposed data.
 */
__global__ void kCalculateIndexedWeightedSparseTransposedDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, Float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, Float* pSparseTransposedData)
{
    uint32_t bpos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (bpos < batch)
    {
        position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        uint64_t start = pSparseStart[position] + threadIdx.x;
        uint64_t end = pSparseEnd[position];
        Float w = cData._denoising_q * pDataWeight[position];
        while (start < end)
        {
            Float rnd = pRandom[start];
            uint32_t index = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                atomicAdd(&pSparseTransposedEnd[index], 1);
                uint32_t opos = atomicSub(&pSparseTransposedEnd[index], 1) - 1;
                pSparseTransposedIndex[opos] = bpos;
                pSparseTransposedData[opos] = w;
            }
            start += blockDim.x;
        }
    }
}

/**
 * @brief CUDA function to calculate the indexed sparse transposed denoised matrix.
 *
 * @param position The position.
 * @param batch The batch size.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the array of sparse start positions.
 * @param pSparseEnd Pointer to the array of sparse end positions.
 * @param pSparseIndex Pointer to the array of sparse indices.
 * @param pDataWeight Pointer to the array of data weights.
 * @param pRandom Pointer to the random array.
 * @param pSparseTransposedEnd Pointer to the array of sparse transposed end positions.
 * @param pSparseTransposedIndex Pointer to the array of sparse transposed indices.
 * @param pSparseTransposedData Pointer to the array of sparse transposed data.
 */
void kCalculateIndexedSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, Float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, Float* pSparseTransposedData)
{
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blockSize = threadsPerBlock;
    uint32_t numBlocks = (batch + blockSize - 1) / blockSize;

    uint32_t sharedMemSize = 0;  // Assuming no shared memory usage in the kernel
    
    if (pDataWeight == NULL)
    {
        kCalculateIndexedSparseTransposedDenoisedMatrix_kernel<<<numBlocks, blockSize, sharedMemSize>>>(position, batch, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pRandom, pSparseTransposedEnd, pSparseTransposedIndex);
        LAUNCHERROR("kCalculateIndexedSparseTransposedDenoisedMatrix_kernel");
    }
    else
    {
        kCalculateIndexedWeightedSparseTransposedDenoisedMatrix_kernel<<<numBlocks, blockSize, sharedMemSize>>>(position, batch, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pRandom, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);
        LAUNCHERROR("kCalculateIndexedWeightedSparseTransposedDenoisedMatrix_kernel");
    }
}

/**
 * @brief Kernel function to calculate the transposed analog matrix for sparse data.
 * 
 * This kernel function is executed on the GPU.
 * 
 * @tparam T The data type of the sparse data.
 * @param position The starting position of the sparse data.
 * @param batch The number of data points in the batch.
 * @param pSparseStart Pointer to the array of sparse start indices.
 * @param pSparseEnd Pointer to the array of sparse end indices.
 * @param pSparseIndex Pointer to the array of sparse indices.
 * @param pDataWeight Pointer to the array of data weights (optional, can be set to nullptr).
 * @param pSparseData Pointer to the array of sparse data.
 * @param pSparseTransposedEnd Pointer to the array of transposed sparse end indices.
 * @param pSparseTransposedIndex Pointer to the array of transposed sparse indices.
 * @param pSparseTransposedData Pointer to the array of transposed sparse data.
 */
template<typename T>
__global__ void kCalculateSparseTransposedAnalogMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, Float* pSparseTransposedData)
{
    uint32_t bpos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (bpos < batch)
    {
        position = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start = pSparseStart[position] + threadIdx.x;
        uint64_t end = pSparseEnd[position];
        Float w = (pDataWeight != NULL) ? pDataWeight[position] : (Float)1.0;
        while (start < end)
        {
            uint32_t index = pSparseIndex[start];
            T value = pSparseData[start];
            uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos] = bpos;
            pSparseTransposedData[opos] = w * static_cast<Float>(value);
            start += blockDim.x;
        }
    }
}

/**
 * @brief Function to calculate the transposed analog matrix for sparse data.
 * 
 * This function launches the kernel function on the GPU.
 * 
 * @tparam T The data type of the sparse data.
 * @param position The starting position of the sparse data.
 * @param batch The number of data points in the batch.
 * @param pSparseStart Pointer to the array of sparse start indices.
 * @param pSparseEnd Pointer to the array of sparse end indices.
 * @param pSparseIndex Pointer to the array of sparse indices.
 * @param pDataWeight Pointer to the array of data weights (optional, can be set to nullptr).
 * @param pSparseData Pointer to the array of sparse data.
 * @param pSparseTransposedEnd Pointer to the array of transposed sparse end indices.
 * @param pSparseTransposedIndex Pointer to the array of transposed sparse indices.
 * @param pSparseTransposedData Pointer to the array of transposed sparse data.
 */
template<typename T>
void kCalculateSparseTransposedAnalogMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, Float* pSparseTransposedData)
{
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blockSize = threadsPerBlock;
    uint32_t numBlocks = (batch + blockSize - 1) / blockSize;

    uint32_t sharedMemSize = 0;  // Assuming no shared memory usage in the kernel

    kCalculateSparseTransposedAnalogMatrix_kernel<<<numBlocks, blockSize, sharedMemSize>>>(position, batch, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);
    LAUNCHERROR("kCalculateSparseTransposedAnalogMatrix_kernel");
}

/**
 * @brief CUDA kernel function for calculating indexed sparse transposed analog matrix.
 *
 * @tparam T The data type of the sparse data.
 * @param position The starting position of the calculation.
 * @param batch The number of batches.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 * @param pSparseTransposedEnd Pointer to the sparse transposed end array.
 * @param pSparseTransposedIndex Pointer to the sparse transposed index array.
 * @param pSparseTransposedData Pointer to the sparse transposed data array.
 */
template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseTransposedAnalogMatrix_kernel(uint32_t position, uint32_t batch, const uint32_t* pIndex, const uint64_t* pSparseStart, const uint64_t* pSparseEnd, const uint32_t* pSparseIndex, const Float* pDataWeight, const T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, Float* pSparseTransposedData)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        uint32_t pos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        uint64_t start = pSparseStart[pos] + tgx;
        uint64_t end = pSparseEnd[pos];
        Float w = (pDataWeight != NULL) ? pDataWeight[pos] : static_cast<Float>(1.0);

        while (start < end)
        {
            uint32_t index = pSparseIndex[start];
            T value = pSparseData[start];
            uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos] = bpos;
            pSparseTransposedData[opos] = w * value;
            start += cData._warpSize * cData._warpSize;
            __syncwarp();
        }
    }
}

/**
 * @brief Function for calculating indexed sparse transposed analog matrix.
 *
 * @tparam T The data type of the sparse data.
 * @param position The starting position of the calculation.
 * @param batch The number of batches.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 * @param pSparseTransposedEnd Pointer to the sparse transposed end array.
 * @param pSparseTransposedIndex Pointer to the sparse transposed index array.
 * @param pSparseTransposedData Pointer to the sparse transposed data array.
 */
template<typename T>
void kCalculateIndexedSparseTransposedAnalogMatrix(uint32_t position, uint32_t batch, const uint32_t* pIndex, const uint64_t* pSparseStart, const uint64_t* pSparseEnd, const uint32_t* pSparseIndex, const Float* pDataWeight, const T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, Float* pSparseTransposedData)
{
    uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
    kCalculateIndexedSparseTransposedAnalogMatrix_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);
    LAUNCHERROR("kCalculateIndexedSparseTransposedAnalogMatrix_kernel");
}


/**
 * @brief Performs the sparse transposed analog denoised matrix calculation.
 *
 * This kernel function calculates the sparse transposed analog denoised matrix based on the given parameters.
 * It operates on a CUDA device using the GPU.
 *
 * @tparam T The data type of the sparse data array.
 * @param position The starting position of the current batch.
 * @param batch The number of batches.
 * @param pSparseStart Pointer to the array of sparse start positions.
 * @param pSparseEnd Pointer to the array of sparse end positions.
 * @param pSparseIndex Pointer to the array of sparse indices.
 * @param pDataWeight Pointer to the array of data weights.
 * @param pSparseData Pointer to the array of sparse data.
 * @param pRandom Pointer to the array of random values.
 * @param pSparseTransposedEnd Pointer to the array of sparse transposed end positions.
 * @param pSparseTransposedIndex Pointer to the array of sparse transposed indices.
 * @param pSparseTransposedData Pointer to the array of sparse transposed data.
 */
template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, const uint64_t* pSparseStart, const uint64_t* pSparseEnd, const uint32_t* pSparseIndex, const Float* pDataWeight, const T* pSparseData, const Float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, Float* pSparseTransposedData)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        position = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start = pSparseStart[position] + tgx;
        uint64_t end = pSparseEnd[position];
        Float w = (pDataWeight != NULL) ? pDataWeight[position] : static_cast<Float>(1.0);
        while (start < end)
        {
            Float rnd = pRandom[start];
            uint32_t index = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                T value = pSparseData[start];
                uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos] = bpos;
                pSparseTransposedData[opos] = w * value;
            }
            start += cData._warpSize * cData._warpSize;
            __syncwarp();
        }
    }
}

/**
 * @brief CUDA kernel for calculating the sparse transposed analog denoised matrix.
 *
 * This template specialization is for a specific data type.
 *
 * @param position The starting position for the current batch.
 * @param batch The size of the batch.
 * @param pSparseStart Pointer to the array of sparse matrix start positions.
 * @param pSparseEnd Pointer to the array of sparse matrix end positions.
 * @param pSparseIndex Pointer to the array of sparse matrix indices.
 * @param pDataWeight Pointer to the array of data weights (optional).
 * @param pSparseData Pointer to the array of sparse matrix data.
 * @param pRandom Pointer to the array of random numbers.
 * @param pSparseTransposedEnd Pointer to the array of transposed sparse matrix end positions.
 * @param pSparseTransposedIndex Pointer to the array of transposed sparse matrix indices.
 * @param pSparseTransposedData Pointer to the array of transposed sparse matrix data.
 */
template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, const uint64_t* pSparseStart, const uint64_t* pSparseEnd, const uint32_t* pSparseIndex, const Float* pDataWeight, const unsigned char* pSparseData, const Float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, Float* pSparseTransposedData)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        uint32_t pos = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start = pSparseStart[pos] + tgx;
        uint64_t end = pSparseEnd[pos];
        Float w = (pDataWeight != NULL) ? pDataWeight[pos] : static_cast<Float>(1.0);

        while (start < end)
        {
            Float rnd = pRandom[start];
            uint32_t index = pSparseIndex[start];

            if (rnd >= cData._denoising_p)
            {
                Float value = static_cast<Float>(pSparseData[start]) * (1.0 / 256.0);
                uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos] = bpos;
                pSparseTransposedData[opos] = w * value;
            }

            start += cData._warpSize * cData._warpSize;
            __syncwarp();
        }
    }
}

/**
 * @brief CUDA kernel for calculating the sparse transposed analog denoised matrix.
 *
 * @param position The starting position.
 * @param batch The number of batches.
 * @param pSparseStart Pointer to the start positions of the sparse matrix.
 * @param pSparseEnd Pointer to the end positions of the sparse matrix.
 * @param pSparseIndex Pointer to the indices of the sparse matrix.
 * @param pDataWeight Pointer to the weight data.
 * @param pSparseData Pointer to the sparse data.
 * @param pRandom Pointer to the random values.
 * @param pSparseTransposedEnd Pointer to the end positions of the transposed sparse matrix.
 * @param pSparseTransposedIndex Pointer to the indices of the transposed sparse matrix.
 * @param pSparseTransposedData Pointer to the data of the transposed sparse matrix.
 */
template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, const uint64_t* pSparseStart, const uint64_t* pSparseEnd, const uint32_t* pSparseIndex, const Float* pDataWeight, const char* pSparseData, const Float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, Float* pSparseTransposedData)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        uint32_t pos = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start = pSparseStart[pos] + tgx;
        uint64_t end = pSparseEnd[pos];
        Float w = (pDataWeight != NULL) ? pDataWeight[pos] : static_cast<Float>(1.0);

        while (start < end)
        {
            Float rnd = pRandom[start];
            uint32_t index = pSparseIndex[start];

            if (rnd >= cData._denoising_p)
            {
                Float value = static_cast<Float>(pSparseData[start]) * (1.0 / 128.0);
                uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos] = bpos;
                pSparseTransposedData[opos] = w * value;
            }

            start += cData._warpSize * cData._warpSize;
            __syncwarp();
        }
    }
}

/**
 * @brief Calculates the sparse transposed analog denoised matrix.
 *
 * @tparam T The data type of pSparseData.
 * @param position The starting position.
 * @param batch The number of batches.
 * @param pSparseStart Pointer to the start positions of the sparse matrix.
 * @param pSparseEnd Pointer to the end positions of the sparse matrix.
 * @param pSparseIndex Pointer to the indices of the sparse matrix.
 * @param pDataWeight Pointer to the weight data.
 * @param pSparseData Pointer to the sparse data.
 * @param pRandom Pointer to the random values.
 * @param pSparseTransposedEnd Pointer to the end positions of the transposed sparse matrix.
 * @param pSparseTransposedIndex Pointer to the indices of the transposed sparse matrix.
 * @param pSparseTransposedData Pointer to the data of the transposed sparse matrix.
 */
template<typename T>
void kCalculateSparseTransposedAnalogDenoisedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, Float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, Float* pSparseTransposedData)
{
    uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
    kCalculateSparseTransposedAnalogDenoisedMatrix_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pRandom, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);
    LAUNCHERROR("kCalculateSparseTransposedAnalogDenoisedMatrix_kernel");
}

/**
 * @brief CUDA kernel to calculate indexed sparse transposed analog denoised matrix.
 *
 * @tparam T The data type.
 * @param position The position parameter.
 * @param batch The batch parameter.
 * @param pIndex Pointer to the index data.
 * @param pSparseStart Pointer to the sparse start data.
 * @param pSparseEnd Pointer to the sparse end data.
 * @param pSparseIndex Pointer to the sparse index data.
 * @param pDataWeight Pointer to the data weight.
 * @param pSparseData Pointer to the sparse data.
 * @param pRandom Pointer to the random data.
 * @param pSparseTransposedEnd Pointer to the sparse transposed end data.
 * @param pSparseTransposedIndex Pointer to the sparse transposed index data.
 * @param pSparseTransposedData Pointer to the sparse transposed data.
 */
template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, const uint32_t* pIndex, const uint64_t* pSparseStart, const uint64_t* pSparseEnd, const uint32_t* pSparseIndex, const Float* pDataWeight, const T* pSparseData, const Float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, Float* pSparseTransposedData)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        uint32_t pos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        uint64_t start = pSparseStart[pos] + tgx;
        uint64_t end = pSparseEnd[pos];
        Float w = (pDataWeight != NULL) ? pDataWeight[pos] : static_cast<Float>(1.0);

        while (start < end)
        {
            Float rnd = pRandom[start];
            uint32_t index = pSparseIndex[start];

            if (rnd >= cData._denoising_p)
            {
                T value = pSparseData[start];
                uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos] = bpos;
                pSparseTransposedData[opos] = w * value;
            }

            start += cData._warpSize * cData._warpSize;
            __syncwarp();
        }
    }
}

/**
 * @brief Kernel function to calculate indexed sparse transposed analog denoised matrix.
 *
 * @tparam Float The data type for floating-point values.
 * @param position The position value.
 * @param batch The batch value.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 * @param pRandom Pointer to the random array.
 * @param pSparseTransposedEnd Pointer to the sparse transposed end array.
 * @param pSparseTransposedIndex Pointer to the sparse transposed index array.
 * @param pSparseTransposedData Pointer to the sparse transposed data array.
 */
template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, const uint32_t* pIndex, const uint64_t* pSparseStart, const uint64_t* pSparseEnd, const uint32_t* pSparseIndex, const Float* pDataWeight, const unsigned char* pSparseData, const Float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, Float* pSparseTransposedData)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        uint32_t pos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        uint64_t start = pSparseStart[pos] + tgx;
        uint64_t end = pSparseEnd[pos];
        Float w = (pDataWeight != NULL) ? pDataWeight[pos] : static_cast<Float>(1.0);
        while (start < end)
        {
            Float rnd = pRandom[start];
            uint32_t index = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                Float value = static_cast<Float>(pSparseData[start]) * (1.0 / 256.0);
                uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos] = bpos;
                pSparseTransposedData[opos] = w * value;
            }
            start += cData._warpSize * cData._warpSize;
            __syncwarp();
        }
    }
}

/**
 * \brief GPU kernel for calculating indexed sparse transposed analog denoised matrix.
 * 
 * \tparam T The data type.
 * 
 * \param position The position.
 * \param batch The batch size.
 * \param pIndex The pointer to the index array.
 * \param pSparseStart The pointer to the sparse start array.
 * \param pSparseEnd The pointer to the sparse end array.
 * \param pSparseIndex The pointer to the sparse index array.
 * \param pDataWeight The pointer to the data weight array.
 * \param pSparseData The pointer to the sparse data array.
 * \param pRandom The pointer to the random array.
 * \param pSparseTransposedEnd The pointer to the sparse transposed end array.
 * \param pSparseTransposedIndex The pointer to the sparse transposed index array.
 * \param pSparseTransposedData The pointer to the sparse transposed data array.
 */
template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, const uint32_t* pIndex, const uint64_t* pSparseStart, const uint64_t* pSparseEnd, const uint32_t* pSparseIndex, const Float* pDataWeight, const char* pSparseData, const Float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, Float* pSparseTransposedData)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        uint32_t pos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        uint64_t start = pSparseStart[pos] + tgx;
        uint64_t end = pSparseEnd[pos];
        Float w = (pDataWeight != NULL) ? pDataWeight[pos] : static_cast<Float>(1.0);
        while (start < end)
        {
            Float rnd = pRandom[start];
            uint32_t index = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                Float value = static_cast<Float>(pSparseData[start]) * (1.0 / 128.0);
                uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos] = bpos;
                pSparseTransposedData[opos] = w * value;
            }
            start += cData._warpSize * cData._warpSize;
            __syncwarp();
        }
    }
}

/**
 * \brief Calculates indexed sparse transposed analog denoised matrix.
 * 
 * \tparam T The data type.
 * 
 * \param position The position.
 * \param batch The batch size.
 * \param pIndex The pointer to the index array.
 * \param pSparseStart The pointer to the sparse start array.
 * \param pSparseEnd The pointer to the sparse end array.
 * \param pSparseIndex The pointer to the sparse index array.
 * \param pDataWeight The pointer to the data weight array.
 * \param pSparseData The pointer to the sparse data array.
 * \param pRandom The pointer to the random array.
 * \param pSparseTransposedEnd The pointer to the sparse transposed end array.
 * \param pSparseTransposedIndex The pointer to the sparse transposed index array.
 * \param pSparseTransposedData The pointer to the sparse transposed data array.
 */
template<typename T>
void kCalculateIndexedSparseTransposedAnalogDenoisedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, Float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, Float* pSparseTransposedData)
{
    uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
    kCalculateIndexedSparseTransposedAnalogDenoisedMatrix_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pRandom, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);
    LAUNCHERROR("kCalculateIndexedSparseTransposedAnalogDenoisedMatrix_kernel");
}
__global__ void
LAUNCH_BOUNDS256()
kCalculateSparseTransposedWeightGradient_kernel(Float alpha, Float beta, uint32_t n, const uint32_t* pSparseTransposedStart, const uint32_t* pSparseTransposedEnd, const uint32_t* pSparseTransposedIndex, const Float* pDelta, Float* pWeightGradient)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSE];

    uint64_t start = pSparseTransposedStart[blockIdx.x];
    uint64_t end = pSparseTransposedEnd[blockIdx.x];
    alpha *= cData._denoising_q;
    pWeightGradient += blockIdx.x * n;

    do
    {
        sOpos = blockDim.x;
        uint32_t inputs = ullmin(end - start, static_cast<uint64_t>(MAXSPARSE));
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos] = pSparseTransposedIndex[tstart] * n;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __syncwarp();

        uint32_t opos = threadIdx.x;
        uint32_t tgx = threadIdx.x & getGpu()._warpMask;

        while (opos < n)
        {
            Float oldgradient = (beta == 0.0) ? 0.0 : beta * pWeightGradient[opos];
            int64_t sum = 0;
            for (uint32_t i = 0; i < inputs; i++)
            {
                uint32_t offset = sOffset[i];
                sum += llrintf(ERRORSCALEF * pDelta[offset + opos]);
            }

            Float fsum = alpha * static_cast<Float>(__popc(__ballot_sync(getGpu()._warpMask, sum))) * ONEOVERERRORSCALE;
            pWeightGradient[opos] = oldgradient + fsum;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, getGpu()._warpSize);
            }
            opos = SHFL(opos, 0);
            opos += tgx;
        }

        start = tend;
        if (start < end)
        {
            __syncwarp();
            beta = 1.0;
        }
    } while (start < end);
}

/**
 * @brief Calculates the sparse transposed weight gradient.
 *
 * This function launches the CUDA kernel to calculate the weight gradient for
 * a sparse transposed matrix. It determines the number of threads to use and
 * launches the kernel accordingly. It also handles any launch errors.
 *
 * @param alpha The scaling factor for the weight gradient.
 * @param beta The scaling factor for the old gradient.
 * @param m The number of rows in the weight matrix.
 * @param n The number of columns in the weight matrix.
 * @param pSparseTransposedStart Pointer to the start indices of the sparse transposed matrix.
 * @param pSparseTransposedEnd Pointer to the end indices of the sparse transposed matrix.
 * @param pSparseTransposedIndex Pointer to the indices of the sparse transposed matrix.
 * @param pDelta Pointer to the delta values.
 * @param pWeightGradient Pointer to the weight gradient.
 */
void kCalculateSparseTransposedWeightGradient(Float alpha, Float beta, uint32_t m, uint32_t n, const uint32_t* pSparseTransposedStart, const uint32_t* pSparseTransposedEnd, const uint32_t* pSparseTransposedIndex, const Float* pDelta, Float* pWeightGradient)
{
    uint32_t threads = min(256, ((m + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateSparseTransposedWeightGradient_kernel<<<m, threads>>>(alpha, beta, n, pSparseTransposedStart, pSparseTransposedEnd, pSparseTransposedIndex, pDelta, pWeightGradient);
    LAUNCHERROR("kCalculateSparseTransposedWeightGradient_kernel");
}
__global__ void
LAUNCH_BOUNDS256()
kCalculateSparseTransposedAnalogWeightGradient_kernel(Float alpha, Float beta, uint32_t n, const uint32_t* pSparseTransposedStart, const uint32_t* pSparseTransposedEnd, const uint32_t* pSparseTransposedIndex, const Float* pSparseTransposedData, const Float* pDelta, Float* pWeightGradient)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ Float sValue[MAXSPARSEANALOG];

    uint64_t start = pSparseTransposedStart[blockIdx.x];
    uint64_t end = pSparseTransposedEnd[blockIdx.x];
    alpha *= cData._denoising_q;
    pWeightGradient += blockIdx.x * n;

    do
    {
        sOpos = blockDim.x;
        uint32_t inputs = ullmin(end - start, static_cast<uint64_t>(MAXSPARSEANALOG));
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos] = pSparseTransposedIndex[tstart] * n;
            sValue[pos] = pSparseTransposedData[start];
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __syncwarp();

        uint32_t opos = threadIdx.x;
        uint32_t tgx = threadIdx.x & getGpu()._warpMask;

        while (opos < n)
        {
            Float oldgradient = (beta == 0.0) ? 0.0 : beta * pWeightGradient[opos];
            int64_t sum = 0;
            for (uint32_t i = 0; i < inputs; i++)
            {
                uint32_t offset = sOffset[i];
                Float value = sValue[i];
                sum += llrintf(ERRORSCALEF * value * pDelta[offset + opos]);
            }

            Float fsum = alpha * static_cast<Float>(__popc(__ballot_sync(getGpu()._warpMask, sum))) * ONEOVERERRORSCALE;
            pWeightGradient[opos] = oldgradient + fsum;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, getGpu()._warpSize);
            }
            opos = SHFL(opos, 0);
            opos += tgx;
        }

        start = tend;
        if (start < end)
        {
            __syncwarp();
            beta = 1.0;
        }
    } while (start < end);
}

/**
 * @brief Calculates the sparse transposed analog weight gradient.
 *
 * @param alpha         The value to multiply with the denoising_q parameter.
 * @param beta          The beta value.
 * @param m             The number of blocks to launch.
 * @param n             The number of weights.
 * @param pSparseTransposedStart    Pointer to the start indices of the sparse transposed matrix.
 * @param pSparseTransposedEnd      Pointer to the end indices of the sparse transposed matrix.
 * @param pSparseTransposedIndex    Pointer to the indices of the sparse transposed matrix.
 * @param pSparseTransposedData     Pointer to the data of the sparse transposed matrix.
 * @param pDelta                    Pointer to the delta values.
 * @param pWeightGradient           Pointer to the weight gradients.
 */
void kCalculateSparseTransposedAnalogWeightGradient(Float alpha, Float beta, uint32_t m, uint32_t n, const uint32_t* pSparseTransposedStart, const uint32_t* pSparseTransposedEnd, const uint32_t* pSparseTransposedIndex, const Float* pSparseTransposedData, const Float* pDelta, Float* pWeightGradient)
{
    uint32_t threads = min(256, ((m + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateSparseTransposedAnalogWeightGradient_kernel<<<m, threads>>>(alpha, beta, n, pSparseTransposedStart, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData, pDelta, pWeightGradient);
    LAUNCHERROR("kCalculateSparseTransposedAnalogWeightGradient_kernel");
}
__global__ void
LAUNCH_BOUNDS()
kUpdateBiases_kernel(Float alpha, uint32_t batch, uint32_t width, const Float* pDelta, Float* pBias)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        Float sum = 0.0;
        pDelta += pos;
        for (uint32_t i = 0; i < batch; i++, pDelta += width)
        {
            sum += *pDelta;
        }
        pBias[pos] -= alpha * sum;
    }
    __syncthreads();
}

/**
 * @brief Update biases on the GPU.
 *
 * This function calculates the number of blocks required and launches the `kUpdateBiases_kernel` GPU kernel.
 *
 * @param alpha The scaling factor.
 * @param batch The batch size.
 * @param width The width of the bias array.
 * @param pDelta Pointer to the delta array.
 * @param pBias Pointer to the bias array.
 */
void kUpdateBiases(Float alpha, uint32_t batch, uint32_t width, const Float* pDelta, Float* pBias)
{
    uint32_t blocks = CalculateBlocks(width);
    kUpdateBiases_kernel<<<blocks, getGpu()._threadsPerBlock>>>(alpha, batch, width, pDelta, pBias);
    LAUNCHERROR("kUpdateBiases_kernel");
}
__global__ void
LAUNCH_BOUNDS()
kCalculateRegularizationError_kernel(Float* pWeight, uint64_t size, Float lambda, Float lambda1)
{
    uint64_t pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    Float error = 0.0;

    if (pos < size)
    {
        Float w = pWeight[pos];
        error = lambda * w * w + lambda1 * abs(w);
    }

    uint32_t mask = __ballot_sync(getGpu()._warpMask, pos < size);
    Float warpError = 0.0;

    #pragma unroll
    for (int i = 0; i < WARP_SIZE; i++)
    {
        if ((mask >> i) & 1)
        {
            warpError += error;
        }
    }

    REDUCEERROR(warpError)
    __syncthreads();
}

/**
 * @brief Calculates the regularization error for a given set of weights.
 *
 * @param lambda Regularization parameter.
 * @param lambda1 Regularization parameter for L1 regularization.
 * @param pWeight Pointer to the weight array.
 * @param size The size of the weight array.
 *
 * @return The calculated regularization error.
 */
Float kCalculateRegularizationError(Float lambda, Float lambda1, Float* pWeight, uint64_t size)
{
    uint32_t blocks = CalculateBlocks(size);
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    kCalculateRegularizationError_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pWeight, size, 0.5 * lambda, lambda1);
    LAUNCHERROR("kCalculateRegularizationError_kernel");
    return (Float)((double)(*getGpu()._pbAccumulator->_pSysData) * ONEOVERERRORSCALE);
}

/**
 * @brief Computes the sign of a number.
 *
 * @param x The input number.
 *
 * @return 1.0 if the number is positive, -1.0 if the number is negative.
 */
__device__ inline Float sgn(Float x) {
    return (x >= 0) ? 1.0f : -1.0f;
}


/**
 * @brief CUDA kernel for updating weights using Stochastic Gradient Descent (SGD).
 *
 * @param alpha The learning rate.
 * @param lambda The weight decay coefficient.
 * @param lambda1 The L1 regularization coefficient.
 * @param size The size of the weight gradient and weight arrays.
 * @param pWeightGradient Pointer to the weight gradient array.
 * @param pWeight Pointer to the weight array.
 */
__global__ void kSGDUpdateWeights_kernel(Float alpha, Float lambda, Float lambda1, uint64_t size, Float* pWeightGradient, Float* pWeight) {
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size) {
        Float g = pWeightGradient[pos];
        Float w = pWeight[pos];
        pWeight[pos] = w + alpha * (g - lambda * w - lambda1 * sgn(w));
    }
}

/**
 * @brief Updates weights using Stochastic Gradient Descent (SGD) on the GPU.
 *
 * @param alpha The learning rate.
 * @param lambda The weight decay coefficient.
 * @param lambda1 The L1 regularization coefficient.
 * @param size The size of the weight gradient and weight arrays.
 * @param pWeightGradient Pointer to the weight gradient array.
 * @param pWeight Pointer to the weight array.
 */
void kSGDUpdateWeights(Float alpha, Float lambda, Float lambda1, uint64_t size, Float* pWeightGradient, Float* pWeight) {
    uint32_t threadsPerBlock = 256; // Modify this value based on your specific GPU architecture
    uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    kSGDUpdateWeights_kernel<<<blocks, threadsPerBlock>>>(alpha, lambda, lambda1, size, pWeightGradient, pWeight);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}

/**
 * @brief CUDA kernel to update biases using Stochastic Gradient Descent (SGD).
 *
 * @param alpha   Learning rate.
 * @param batch   Number of samples in a mini-batch.
 * @param width   Width of the bias array.
 * @param pDelta  Pointer to the delta array.
 * @param pBias   Pointer to the bias array.
 */
__global__ void kSGDUpdateBiases_kernel(Float alpha, uint32_t batch, uint32_t width, Float* pDelta, Float* pBias)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        Float sum = 0.0f;
        pDelta += pos;

        for (uint32_t i = 0; i < batch; i++, pDelta += width)
            sum += *pDelta;

        Float bias = pBias[pos];
        pBias[pos] = bias - alpha * (sum / (Float)batch);
    }
}

/**
 * @brief Updates biases using Stochastic Gradient Descent (SGD).
 *
 * @param alpha   Learning rate.
 * @param batch   Number of samples in a mini-batch.
 * @param width   Width of the bias array.
 * @param pDelta  Pointer to the delta array.
 * @param pBias   Pointer to the bias array.
 */
void kSGDUpdateBiases(Float alpha, uint32_t batch, uint32_t width, Float* pDelta, Float* pBias)
{
    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (width + threadsPerBlock - 1) / threadsPerBlock;
    kSGDUpdateBiases_kernel<<<blocks, threadsPerBlock>>>(alpha, batch, width, pDelta, pBias);
    cudaDeviceSynchronize();
    LAUNCHERROR("kSGDUpdateBiases_kernel");
}

/**
 * @brief Updates the weights using momentum optimizer.
 *
 * This function updates the weights using the momentum optimizer algorithm.
 *
 * @param alpha Learning rate.
 * @param lambda Weight decay coefficient.
 * @param lambda1 L1 regularization coefficient.
 * @param mu Momentum coefficient.
 * @param size Number of elements in the weight arrays.
 * @param pWeightVelocity Pointer to the weight velocity array.
 * @param pWeightGradient Pointer to the weight gradient array.
 * @param pWeight Pointer to the weight array.
 */
__global__ void kMomentumUpdateWeights_kernel(Float alpha, Float lambda, Float lambda1, Float mu, uint64_t size, Float* pWeightVelocity, Float* pWeightGradient, Float* pWeight)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        Float g = pWeightGradient[pos];
        Float w = pWeight[pos];
        Float v = pWeightVelocity[pos];
        v = mu * v + alpha * (g - lambda * w - lambda1 * signbit(w));
        pWeightVelocity[pos] = v;
        pWeight[pos] = w + v;
    }
}

/**
 * @brief Calls the momentum weight update kernel.
 *
 * This function calculates the number of blocks and threads required,
 * and launches the kernel to update the weights using momentum optimizer.
 *
 * @param alpha Learning rate.
 * @param lambda Weight decay coefficient.
 * @param lambda1 L1 regularization coefficient.
 * @param mu Momentum coefficient.
 * @param size Number of elements in the weight arrays.
 * @param pWeightVelocity Pointer to the weight velocity array.
 * @param pWeightGradient Pointer to the weight gradient array.
 * @param pWeight Pointer to the weight array.
 */
void kMomentumUpdateWeights(Float alpha, Float lambda, Float lambda1, Float mu, uint64_t size, Float* pWeightVelocity, Float* pWeightGradient, Float* pWeight)
{
    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    kMomentumUpdateWeights_kernel<<<blocks, threadsPerBlock>>>(alpha, lambda, lambda1, mu, size, pWeightVelocity, pWeightGradient, pWeight);
    cudaDeviceSynchronize();
    LAUNCHERROR("kMomentumUpdateWeights_kernel");
}

/**
 * @brief Updates the biases using the momentum method.
 *
 * @param alpha The learning rate.
 * @param mu The momentum coefficient.
 * @param batch The batch size.
 * @param width The width of the bias array.
 * @param pDelta Pointer to the delta array.
 * @param pBiasVelocity Pointer to the bias velocity array.
 * @param pBias Pointer to the bias array.
 */
__global__ void kMomentumUpdateBiases_kernel(Float alpha, Float mu, uint32_t batch, uint32_t width, Float* pDelta, Float* pBiasVelocity, Float* pBias)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        Float sum = 0.0f;
        pDelta += pos;

        for (uint32_t i = 0; i < batch; i++, pDelta += width)
            sum += *pDelta;

        Float v = pBiasVelocity[pos];
        v = mu * v - alpha * (sum / (Float)batch);
        pBiasVelocity[pos] = v;
        pBias[pos] += v;
    }
}

/**
 * @brief Updates the biases using the momentum method.
 *
 * @param alpha The learning rate.
 * @param mu The momentum coefficient.
 * @param batch The batch size.
 * @param width The width of the bias array.
 * @param pDelta Pointer to the delta array.
 * @param pBiasVelocity Pointer to the bias velocity array.
 * @param pBias Pointer to the bias array.
 */
void kMomentumUpdateBiases(Float alpha, Float mu, uint32_t batch, uint32_t width, Float* pDelta, Float* pBiasVelocity, Float* pBias)
{
    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (width + threadsPerBlock - 1) / threadsPerBlock;
    kMomentumUpdateBiases_kernel<<<blocks, threadsPerBlock>>>(alpha, mu, batch, width, pDelta, pBiasVelocity, pBias);
    cudaDeviceSynchronize();
    LAUNCHERROR("kMomentumUpdateBiases_kernel");
}

/**
 * @brief Updates the weights using the AdaGrad algorithm.
 *
 * This CUDA kernel performs weight updates using the AdaGrad algorithm. It computes the gradient
 * update, updates the velocity, and updates the weights based on the velocity and learning rate.
 *
 * @param alpha The learning rate for weight updates.
 * @param lambda The regularization parameter for weight decay.
 * @param lambda1 The regularization parameter for L1 weight decay.
 * @param size The number of weights to update.
 * @param pWeightVelocity Pointer to the array storing weight velocities.
 * @param pWeightGradient Pointer to the array storing weight gradients.
 * @param pWeight Pointer to the array storing weights.
 */
__global__ void kAdaGradUpdateWeights_kernel(Float alpha, Float lambda, Float lambda1, uint64_t size, Float* pWeightVelocity, Float* pWeightGradient, Float* pWeight)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        Float g = pWeightGradient[pos];
        Float w = pWeight[pos];
        Float v = pWeightVelocity[pos];

        g -= lambda * w + lambda1 * signbit(w);
        v += g * g;

        pWeightVelocity[pos] = v;
        pWeight[pos] = w + alpha * g / sqrtf(max(v, 0.000000001f));
    }
}

/**
 * @brief Wrapper function for launching the AdaGrad weight update kernel.
 *
 * This function calculates the number of blocks and threads per block, and then launches the
 * AdaGrad weight update kernel with the specified parameters. It also synchronizes the device
 * after kernel execution and checks for any launch errors.
 *
 * @param alpha The learning rate for weight updates.
 * @param lambda The regularization parameter for weight decay.
 * @param lambda1 The regularization parameter for L1 weight decay.
 * @param size The number of weights to update.
 * @param pWeightVelocity Pointer to the array storing weight velocities.
 * @param pWeightGradient Pointer to the array storing weight gradients.
 * @param pWeight Pointer to the array storing weights.
 */
void kAdaGradUpdateWeights(Float alpha, Float lambda, Float lambda1, uint64_t size, Float* pWeightVelocity, Float* pWeightGradient, Float* pWeight)
{
    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    kAdaGradUpdateWeights_kernel<<<blocks, threadsPerBlock>>>(alpha, lambda, lambda1, size, pWeightVelocity, pWeightGradient, pWeight);
    cudaDeviceSynchronize();
    LAUNCHERROR("kAdaGradUpdateWeights_kernel");
}

/**
 * @brief Updates the biases using the AdaGrad algorithm.
 *
 * This function is a CUDA kernel that updates the biases of a neural network using the AdaGrad algorithm.
 *
 * @param alpha The learning rate.
 * @param batch The number of samples in a mini-batch.
 * @param width The number of biases to update.
 * @param pDelta Pointer to the delta values.
 * @param pBiasVelocity Pointer to the velocity of the biases.
 * @param pBias Pointer to the biases.
 */
__global__ void kAdaGradUpdateBiases_kernel(Float alpha, uint32_t batch, uint32_t width, Float* pDelta, Float* pBiasVelocity, Float* pBias)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        Float sum = 0.0f;
        pDelta += pos;

        for (uint32_t i = 0; i < batch; i++, pDelta += width)
            sum += *pDelta;

        sum /= (Float)batch;

        Float v = pBiasVelocity[pos];
        v += sum * sum;

        pBiasVelocity[pos] = v;
        pBias[pos] -= alpha * sum / sqrtf(max(v, 0.000000001f));
    }
}

/**
 * @brief Updates the biases using the AdaGrad algorithm.
 *
 * This function launches the CUDA kernel to update the biases using the AdaGrad algorithm.
 *
 * @param alpha The learning rate.
 * @param batch The number of samples in a mini-batch.
 * @param width The number of biases to update.
 * @param pDelta Pointer to the delta values.
 * @param pBiasVelocity Pointer to the velocity of the biases.
 * @param pBias Pointer to the biases.
 */
void kAdaGradUpdateBiases(Float alpha, uint32_t batch, uint32_t width, Float* pDelta, Float* pBiasVelocity, Float* pBias)
{
    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (width + threadsPerBlock - 1) / threadsPerBlock;
    kAdaGradUpdateBiases_kernel<<<blocks, threadsPerBlock>>>(alpha, batch, width, pDelta, pBiasVelocity, pBias);
    cudaDeviceSynchronize();
    LAUNCHERROR("kAdaGradUpdateBiases_kernel");
}

/**
 * @brief CUDA kernel to update weights using AdaDelta algorithm.
 *
 * @param lambda                  The regularization factor.
 * @param lambda1                 The regularization factor for L1 regularization.
 * @param mu                      The decay rate.
 * @param size                    The size of the weight array.
 * @param pWeightVelocity         Pointer to the velocity array.
 * @param pWeightGradient         Pointer to the gradient array.
 * @param pWeightGradientVelocity Pointer to the gradient velocity array.
 * @param pWeight                 Pointer to the weight array.
 */
__global__ void kAdaDeltaUpdateWeights_kernel(Float lambda, Float lambda1, Float mu, uint64_t size, Float* pWeightVelocity, Float* pWeightGradient, Float* pWeightGradientVelocity, Float* pWeight)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        Float g = pWeightGradient[pos];
        Float w = pWeight[pos];
        Float v = pWeightVelocity[pos];
        Float vg = pWeightGradientVelocity[pos];

        g -= lambda * w + lambda1 * signbit(w);
        vg = mu * vg + (1.0 - mu) * g * g;

        Float dw = sqrtf(max(v, 0.000000001f) / max(vg, 0.000000001f)) * g;
        v = mu * v + (1.0 - mu) * dw * dw;

        pWeightVelocity[pos] = v;
        pWeightGradientVelocity[pos] = vg;
        pWeight[pos] = w + dw;
    }
}

/**
 * @brief Update weights using AdaDelta algorithm.
 *
 * @param lambda                  The regularization factor.
 * @param lambda1                 The regularization factor for L1 regularization.
 * @param mu                      The decay rate.
 * @param size                    The size of the weight array.
 * @param pWeightVelocity         Pointer to the velocity array.
 * @param pWeightGradient         Pointer to the gradient array.
 * @param pWeightGradientVelocity Pointer to the gradient velocity array.
 * @param pWeight                 Pointer to the weight array.
 */
void kAdaDeltaUpdateWeights(Float lambda, Float lambda1, Float mu, uint64_t size, Float* pWeightVelocity, Float* pWeightGradient, Float* pWeightGradientVelocity, Float* pWeight)
{
    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    kAdaDeltaUpdateWeights_kernel<<<blocks, threadsPerBlock>>>(lambda, lambda1, mu, size, pWeightVelocity, pWeightGradient, pWeightGradientVelocity, pWeight);
    cudaDeviceSynchronize();
    LAUNCHERROR("kAdaDeltaUpdateWeights_kernel");
}

/**
 * @brief CUDA kernel to update biases using AdaDelta algorithm.
 *
 * @param mu                     The decay rate.
 * @param batch                  The batch size.
 * @param width                  The width of the bias array.
 * @param pDelta                 Pointer to the delta array.
 * @param pBiasVelocity          Pointer to the velocity array.
 * @param pBiasGradientVelocity  Pointer to the gradient velocity array.
 * @param pBias                  Pointer to the bias array.
 */
__global__ void kAdaDeltaUpdateBiases_kernel(Float mu, uint32_t batch, uint32_t width, Float* pDelta, Float* pBiasVelocity, Float* pBiasGradientVelocity, Float* pBias)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        Float sum = 0.0f;
        pDelta += pos;

        for (uint32_t i = 0; i < batch; i++)
        {
            sum += *pDelta;
            pDelta += width;
        }
        sum /= static_cast<Float>(batch);

        Float v = pBiasVelocity[pos];
        Float vg = pBiasGradientVelocity[pos];
        vg = mu * vg + (1.0f - mu) * sum * sum;
        Float dw = sqrtf(fmaxf(v, 0.000000001f) / fmaxf(vg, 0.000000001f)) * sum;
        v = mu * v + (1.0f - mu) * dw * dw;
        pBiasVelocity[pos] = v;
        pBiasGradientVelocity[pos] = vg;
        pBias[pos] -= dw;
    }
}

/**
 * @brief Update biases using AdaDelta algorithm.
 *
 * @param mu                     The decay rate.
 * @param batch                  The batch size.
 * @param width                  The width of the bias array.
 * @param pDelta                 Pointer to the delta array.
 * @param pBiasVelocity          Pointer to the velocity array.
 * @param pBiasGradientVelocity  Pointer to the gradient velocity array.
 * @param pBias                  Pointer to the bias array.
 */
void kAdaDeltaUpdateBiases(Float mu, uint32_t batch, uint32_t width, Float* pDelta, Float* pBiasVelocity, Float* pBiasGradientVelocity, Float* pBias)
{
    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (width + threadsPerBlock - 1) / threadsPerBlock;
    kAdaDeltaUpdateBiases_kernel<<<blocks, threadsPerBlock>>>(mu, batch, width, pDelta, pBiasVelocity, pBiasGradientVelocity, pBias);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in kAdaDeltaUpdateBiases: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

/**
 * @brief GPU kernel for updating weights using the Adam optimizer.
 *
 * @param alpha The learning rate.
 * @param lambda The weight decay rate.
 * @param lambda1 The L1 regularization rate.
 * @param beta1 The exponential decay rate for the first moment estimates.
 * @param beta2 The exponential decay rate for the second moment estimates.
 * @param t The current iteration count.
 * @param size The size of the weight vectors.
 * @param pWeightVelocity The velocity of the weights.
 * @param pWeightGradient The gradient of the weights.
 * @param pWeightGradientVelocity The velocity of the weight gradients.
 * @param pWeight The weights to be updated.
 */
__global__ void kAdamUpdateWeights_kernel(Float alpha, Float lambda, Float lambda1, Float beta1, Float beta2, Float t, uint64_t size, Float* pWeightVelocity, Float* pWeightGradient, Float* pWeightGradientVelocity, Float* pWeight)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        Float dw = pWeightGradient[pos];
        Float w = pWeight[pos];
        Float vdw = pWeightVelocity[pos];
        Float sdw = pWeightGradientVelocity[pos];

        dw -= lambda * w + lambda1 * sgn(w);
        vdw = beta1 * vdw + (1.0f - beta1) * dw;
        sdw = beta2 * sdw + (1.0f - beta2) * dw * dw;
        t += 1.0f;

        pWeightVelocity[pos] = vdw;
        pWeightGradientVelocity[pos] = sdw;

        vdw /= 1.0f - powf(beta1, t);
        sdw /= 1.0f - powf(beta2, t);
        dw = alpha * vdw / (sqrtf(sdw) + 1.0e-8f);

        pWeight[pos] = w + dw;
    }
}

/**
 * @brief Function for updating weights using the Adam optimizer on the GPU.
 *
 * @param alpha The learning rate.
 * @param lambda The weight decay rate.
 * @param lambda1 The L1 regularization rate.
 * @param beta1 The exponential decay rate for the first moment estimates.
 * @param beta2 The exponential decay rate for the second moment estimates.
 * @param t The current iteration count.
 * @param size The size of the weight vectors.
 * @param pWeightVelocity The velocity of the weights.
 * @param pWeightGradient The gradient of the weights.
 * @param pWeightGradientVelocity The velocity of the weight gradients.
 * @param pWeight The weights to be updated.
 */
void kAdamUpdateWeights(Float alpha, Float lambda, Float lambda1, Float beta1, Float beta2, Float t, uint64_t size, Float* pWeightVelocity, Float* pWeightGradient, Float* pWeightGradientVelocity, Float* pWeight)
{
    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    kAdamUpdateWeights_kernel<<<blocks, threadsPerBlock>>>(alpha, lambda, lambda1, beta1, beta2, t, size, pWeightVelocity, pWeightGradient, pWeightGradientVelocity, pWeight);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in kAdamUpdateWeights: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

/**
 * @brief CUDA kernel to update biases using the Adam optimizer.
 *
 * @param alpha                 Learning rate.
 * @param beta1                 Exponential decay rate for the first moment estimates.
 * @param beta2                 Exponential decay rate for the second moment estimates.
 * @param t                     Time step.
 * @param batch                 Batch size.
 * @param width                 Width of the biases.
 * @param pDelta                Pointer to the delta values.
 * @param pBiasVelocity         Pointer to the velocity values of the biases.
 * @param pBiasGradientVelocity Pointer to the gradient velocity values of the biases.
 * @param pBias                 Pointer to the biases.
 */
__global__ void kAdamUpdateBiases_kernel(Float alpha, Float beta1, Float beta2, Float t, uint32_t batch, uint32_t width, Float* pDelta, Float* pBiasVelocity, Float* pBiasGradientVelocity, Float* pBias)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        Float sum = 0.0f;
        pDelta += pos;

        for (uint32_t i = 0; i < batch; i++)
        {
            sum += *pDelta;
            pDelta += width;
        }
        sum /= static_cast<Float>(batch);

        Float vdw = pBiasVelocity[pos];
        Float sdw = pBiasGradientVelocity[pos];
        vdw = beta1 * vdw + (1.0f - beta1) * sum;
        sdw = beta2 * sdw + (1.0f - beta2) * sum * sum;
        t += 1.0f;
        pBiasVelocity[pos] = vdw;
        pBiasGradientVelocity[pos] = sdw;
        vdw /= 1.0f - powf(beta1, t);
        sdw /= 1.0f - powf(beta2, t);
        Float dw = alpha * vdw / (sqrtf(sdw) + 1.0e-8f);
        pBias[pos] -= dw;
    }
}

/**
 * @brief Function to update biases using the Adam optimizer.
 *
 * @param alpha                 Learning rate.
 * @param beta1                 Exponential decay rate for the first moment estimates.
 * @param beta2                 Exponential decay rate for the second moment estimates.
 * @param t                     Time step.
 * @param batch                 Batch size.
 * @param width                 Width of the biases.
 * @param pDelta                Pointer to the delta values.
 * @param pBiasVelocity         Pointer to the velocity values of the biases.
 * @param pBiasGradientVelocity Pointer to the gradient velocity values of the biases.
 * @param pBias                 Pointer to the biases.
 */
void kAdamUpdateBiases(Float alpha, Float beta1, Float beta2, Float t, uint32_t batch, uint32_t width, Float* pDelta, Float* pBiasVelocity, Float* pBiasGradientVelocity, Float* pBias)
{
    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (width + threadsPerBlock - 1) / threadsPerBlock;
    kAdamUpdateBiases_kernel<<<blocks, threadsPerBlock>>>(alpha, beta1, beta2, t, batch, width, pDelta, pBiasVelocity, pBiasGradientVelocity, pBias);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in kAdamUpdateBiases: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

/**
 * @brief CUDA kernel to perform Nesterov weight update.
 *
 * @param alpha The learning rate.
 * @param lambda The weight decay coefficient.
 * @param lambda1 The L1 regularization coefficient.
 * @param mu The momentum coefficient.
 * @param size The size of the weight arrays.
 * @param pWeightVelocity Pointer to the weight velocity array.
 * @param pWeightGradient Pointer to the weight gradient array.
 * @param pWeight Pointer to the weight array.
 */
__global__ void kNesterovUpdateWeights_kernel(Float alpha, Float lambda, Float lambda1, Float mu, uint64_t size, Float* pWeightVelocity, Float* pWeightGradient, Float* pWeight)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        Float g = pWeightGradient[pos];
        Float w = pWeight[pos];
        Float vOld = pWeightVelocity[pos];
        Float vNew = mu * vOld + alpha * (g - lambda * w - lambda1 * sgn(w));
        pWeightVelocity[pos] = vNew;
        w = w + vNew + mu * (vNew - vOld);
        pWeight[pos] = w;
    }
}

/**
 * @brief Function to perform Nesterov weight update on the GPU.
 *
 * @param alpha The learning rate.
 * @param lambda The weight decay coefficient.
 * @param lambda1 The L1 regularization coefficient.
 * @param mu The momentum coefficient.
 * @param size The size of the weight arrays.
 * @param pWeightVelocity Pointer to the weight velocity array.
 * @param pWeightGradient Pointer to the weight gradient array.
 * @param pWeight Pointer to the weight array.
 */
void kNesterovUpdateWeights(Float alpha, Float lambda, Float lambda1, Float mu, uint64_t size, Float* pWeightVelocity, Float* pWeightGradient, Float* pWeight)
{
    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    kNesterovUpdateWeights_kernel<<<blocks, threadsPerBlock>>>(alpha, lambda, lambda1, mu, size, pWeightVelocity, pWeightGradient, pWeight);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in kNesterovUpdateWeights: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

/**
 * @brief Updates the biases using Nesterov accelerated gradient descent.
 *
 * @param alpha The learning rate.
 * @param mu The momentum factor.
 * @param batch The batch size.
 * @param width The width of the biases.
 * @param pDelta Pointer to the delta values.
 * @param pBiasVelocity Pointer to the bias velocity values.
 * @param pBias Pointer to the bias values.
 */
__global__ void kNesterovUpdateBiases_kernel(Float alpha, Float mu, uint32_t batch, uint32_t width, Float* pDelta, Float* pBiasVelocity, Float* pBias)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        Float sum = 0.0f;
        pDelta += pos;

        for (uint32_t i = 0; i < batch; i++)
        {
            sum += *pDelta;
            pDelta += width;
        }
        sum /= static_cast<Float>(batch);

        Float vOld = pBiasVelocity[pos];
        Float vNew = mu * vOld - alpha * sum;
        pBiasVelocity[pos] = vNew;
        pBias[pos] += vNew + mu * (vNew - vOld);
    }
}

/**
 * @brief Updates the biases using Nesterov accelerated gradient descent.
 *
 * @param alpha The learning rate.
 * @param mu The momentum factor.
 * @param batch The batch size.
 * @param width The width of the biases.
 * @param pDelta Pointer to the delta values.
 * @param pBiasVelocity Pointer to the bias velocity values.
 * @param pBias Pointer to the bias values.
 */
void kNesterovUpdateBiases(Float alpha, Float mu, uint32_t batch, uint32_t width, Float* pDelta, Float* pBiasVelocity, Float* pBias)
{
    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (width + threadsPerBlock - 1) / threadsPerBlock;
    kNesterovUpdateBiases_kernel<<<blocks, threadsPerBlock>>>(alpha, mu, batch, width, pDelta, pBiasVelocity, pBias);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in kNesterovUpdateBiases: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

/**
 * @brief Performs Nesterov's momentum weight update.
 *
 * This CUDA kernel updates the weights of a neural network using Nesterov's momentum method.
 * It calculates the new weight values based on the weight velocities and a given momentum coefficient.
 *
 * @param mu The momentum coefficient.
 * @param size The number of weights.
 * @param pWeightVelocity Pointer to the weight velocities array.
 * @param pWeight Pointer to the weight array.
 */
__global__ void kNesterovShiftWeights_kernel(Float mu, uint64_t size, Float* pWeightVelocity, Float* pWeight)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        Float w = pWeight[pos];
        Float v = pWeightVelocity[pos];
        pWeight[pos] = w + mu * v;
    }
}

/**
 * @brief Calls the CUDA kernel for Nesterov's momentum weight update.
 *
 * This function sets up the CUDA kernel parameters and launches the kernel to update the weights
 * using Nesterov's momentum method. It also performs error checking and prints any CUDA errors
 * encountered during the kernel execution.
 *
 * @param mu The momentum coefficient.
 * @param size The number of weights.
 * @param pWeightVelocity Pointer to the weight velocities array.
 * @param pWeight Pointer to the weight array.
 */
void kNesterovShiftWeights(Float mu, uint64_t size, Float* pWeightVelocity, Float* pWeight)
{
    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    kNesterovShiftWeights_kernel<<<blocks, threadsPerBlock>>>(mu, size, pWeightVelocity, pWeight);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in kNesterovShiftWeights: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

/**
 * @brief Performs Nesterov momentum shift for biases.
 *
 * This CUDA kernel updates the biases using Nesterov momentum.
 * Each thread processes a single bias element.
 *
 * @param mu The momentum factor.
 * @param width The width of the bias array.
 * @param pBiasVelocity Pointer to the bias velocity array.
 * @param pBias Pointer to the bias array.
 */
__global__ void kNesterovShiftBiases_kernel(Float mu, uint32_t width, Float* pBiasVelocity, Float* pBias)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        Float b = pBias[pos];
        Float v = pBiasVelocity[pos];
        pBias[pos] = b + mu * v;
    }
}

/**
 * @brief Performs Nesterov momentum shift for biases.
 *
 * This function launches the Nesterov momentum shift kernel with the specified parameters.
 * It calculates the number of blocks and threads needed based on the given width.
 * It also performs error checking and synchronization after the kernel launch.
 *
 * @param mu The momentum factor.
 * @param width The width of the bias array.
 * @param pBiasVelocity Pointer to the bias velocity array.
 * @param pBias Pointer to the bias array.
 */
void kNesterovShiftBiases(Float mu, uint32_t width, Float* pBiasVelocity, Float* pBias)
{
    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (width + threadsPerBlock - 1) / threadsPerBlock;
    kNesterovShiftBiases_kernel<<<blocks, threadsPerBlock>>>(mu, width, pBiasVelocity, pBias);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in kNesterovShiftBiases: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

/**
 * @brief Performs RMSProp weight update for a given kernel.
 *
 * This CUDA kernel updates the weights using the RMSProp algorithm.
 * Each thread processes a single weight element.
 *
 * @param alpha The learning rate.
 * @param lambda Regularization parameter.
 * @param lambda1 Regularization parameter for the sign function.
 * @param mu Momentum factor.
 * @param size The total number of weight elements.
 * @param pWeightVelocity Pointer to the weight velocity array.
 * @param pWeightGradient Pointer to the weight gradient array.
 * @param pWeight Pointer to the weight array.
 */
__global__ void kRMSPropUpdateWeights_kernel(Float alpha, Float lambda, Float lambda1, Float mu, uint64_t size, Float* pWeightVelocity, Float* pWeightGradient, Float* pWeight)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        Float g = pWeightGradient[pos];
        Float w = pWeight[pos];
        Float v = pWeightVelocity[pos];
        g -= lambda * w + lambda1 * sgn(w);
        v = mu * v + (1.0f - mu) * g * g;
        pWeightVelocity[pos] = v;
        pWeight[pos] = w + alpha * g * rsqrtf(fmaxf(0.000000001f, v));
    }
}

/**
 * @brief Performs RMSProp weight update.
 *
 * This function launches the RMSProp weight update kernel with the specified parameters.
 * It calculates the number of blocks and threads needed based on the given size.
 * It also performs error checking and synchronization after the kernel launch.
 *
 * @param alpha The learning rate.
 * @param lambda Regularization parameter.
 * @param lambda1 Regularization parameter for the sign function.
 * @param mu Momentum factor.
 * @param size The total number of weight elements.
 * @param pWeightVelocity Pointer to the weight velocity array.
 * @param pWeightGradient Pointer to the weight gradient array.
 * @param pWeight Pointer to the weight array.
 */
void kRMSPropUpdateWeights(Float alpha, Float lambda, Float lambda1, Float mu, uint64_t size, Float* pWeightVelocity, Float* pWeightGradient, Float* pWeight)
{
    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    kRMSPropUpdateWeights_kernel<<<blocks, threadsPerBlock>>>(alpha, lambda, lambda1, mu, size, pWeightVelocity, pWeightGradient, pWeight);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in kRMSPropUpdateWeights: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

/**
 * @brief Updates the biases using the RMSProp algorithm.
 *
 * This kernel function updates the biases based on the RMSProp algorithm.
 * It calculates the velocity of the biases and updates the biases using the velocity.
 *
 * @param alpha     The learning rate.
 * @param mu        The decay rate for the velocity.
 * @param batch     The batch size.
 * @param width     The width of the biases array.
 * @param pDelta    Pointer to the delta array.
 * @param pBiasVelocity    Pointer to the biases velocity array.
 * @param pBias     Pointer to the biases array.
 */
__global__ void kRMSPropUpdateBiases_kernel(Float alpha, Float mu, uint32_t batch, uint32_t width, Float* pDelta, Float* pBiasVelocity, Float* pBias)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        Float sum = 0.0f;
        pDelta += pos;

        for (uint32_t i = 0; i < batch; i++)
        {
            sum += *pDelta;
            pDelta += width;
        }
        sum /= static_cast<Float>(batch);

        Float v = pBiasVelocity[pos];
        v = mu * v + (1.0f - mu) * sum * sum;
        pBiasVelocity[pos] = v;
        pBias[pos] -= alpha * sum * rsqrtf(fmaxf(0.000000001f, v));
    }
}

/**
 * @brief Updates the biases using the RMSProp algorithm.
 *
 * This function updates the biases using the RMSProp algorithm.
 *
 * @param alpha     The learning rate.
 * @param mu        The decay rate for the velocity.
 * @param batch     The batch size.
 * @param width     The width of the biases array.
 * @param pDelta    Pointer to the delta array.
 * @param pBiasVelocity    Pointer to the biases velocity array.
 * @param pBias     Pointer to the biases array.
 */
void kRMSPropUpdateBiases(Float alpha, Float mu, uint32_t batch, uint32_t width, Float* pDelta, Float* pBiasVelocity, Float* pBias)
{
    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (width + threadsPerBlock - 1) / threadsPerBlock;
    kRMSPropUpdateBiases_kernel<<<blocks, threadsPerBlock>>>(alpha, mu, batch, width, pDelta, pBiasVelocity, pBias);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in kRMSPropUpdateBiases: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

#include "bitonic.h"

/**
 * @brief CUDA kernel for calculating output.
 *
 * This CUDA kernel calculates the output based on the input buffers and parameters.
 *
 * @param pOutputBuffer The output buffer.
 * @param pKeyBuffer The key buffer.
 * @param pValueBuffer The value buffer.
 * @param batch The number of batches.
 * @param width The width of the data.
 * @param k The value of k.
 */
__global__ void kCalculateOutput_32_kernel(Float* pOutputBuffer, Float* pKeyBuffer, uint32_t* pValueBuffer, uint32_t batch, uint32_t width, uint32_t k)
{
    uint32_t pos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (pos < batch)
    {
        Float* pOutput = pOutputBuffer + pos * width;
        uint32_t offset = threadIdx.x >> cData._warpBits;
        volatile Float* psKey = &sKey[64 * offset];
        volatile uint32_t* psValue = &sValue[64 * offset];

        Float k0 = -MAX_VALUE;
        Float k1 = -MAX_VALUE;
        uint32_t v0 = 0;
        uint32_t v1 = 0;

        uint32_t wpos = tgx;
        if (wpos < width)
        {
            k0 = pOutput[wpos];
            v0 = wpos;
        }
        wpos += cData._warpSize;

        Float minValue = -MAX_VALUE;
        uint32_t rpos = 32;
        uint32_t bufferSize = 0;
        while (rpos < width)
        {
            unsigned wpos = rpos + tgx;
            Float key = -MAX_VALUE;
            uint32_t value = wpos;
            if (wpos < width)
            {
                key = pOutput[wpos];
            }

            uint32_t count = BALLOT(key > minValue);
            if (key > minValue)
            {
                uint32_t mask = 0xffffffff >> (32 - tgx);
                uint32_t offset = __popc(count & mask);
                offset += bufferSize;
                psKey[offset] = key;
                psValue[offset] = value;
            }
            bufferSize += __popc(count);

            if (bufferSize >= 32)
            {
                k1 = psKey[tgx];
                v1 = psValue[tgx];
                BITONICSORT64_64();

                minValue = SHFL(k0, cData._warpSize - 1);

                bufferSize -= 32;
                if (tgx < bufferSize)
                {
                    psKey[tgx] = psKey[tgx + 32];
                    psValue[tgx] = psValue[tgx + 32];
                }
            }

            rpos += cData._warpSize;
        }

        if ((bufferSize > 0) || (width <= 32))
        {
            k1 = -MAX_VALUE;
            v1 = 0;

            if (tgx < bufferSize)
            {
                k1 = psKey[tgx];
                v1 = psValue[tgx];
            }
            BITONICSORT64_64();
        }

        Float* pKey = pKeyBuffer + pos * k;
        uint32_t* pValue = pValueBuffer + pos * k;
        wpos = tgx;
        if (wpos < k)
        {
            pKey[wpos] = k0;
            pValue[wpos] = v0;
        }
        wpos += cData._warpSize;
    }
}

#include "bitonic.h"

/**
 * @brief CUDA kernel for calculating output values using bitonic sorting.
 *
 * @param pOutputBuffer Pointer to the output buffer.
 * @param pKeyBuffer Pointer to the key buffer.
 * @param pValueBuffer Pointer to the value buffer.
 * @param batch Number of batches.
 * @param width Width of the data.
 * @param k Number of elements to be sorted.
 */
__global__ void kCalculateOutput_64_kernel(Float* pOutputBuffer, Float* pKeyBuffer, uint32_t* pValueBuffer, uint32_t batch, uint32_t width, uint32_t k)
{
    uint32_t pos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (pos < batch)
    {
        Float* pOutput = pOutputBuffer + pos * width;
        uint32_t offset = threadIdx.x >> cData._warpBits;
        volatile Float* psKey = &sKey[96 * offset];
        volatile uint32_t* psValue = &sValue[96 * offset];

        Float k0 = -MAX_VALUE;
        Float k1 = -MAX_VALUE;
        Float k2 = -MAX_VALUE;
        Float k3 = -MAX_VALUE;
        uint32_t v0 = 0;
        uint32_t v1 = 0;
        uint32_t v2 = 0;
        uint32_t v3 = 0;

        uint32_t wpos = tgx;
        if (wpos < width)
        {
            k0 = pOutput[wpos];
            v0 = wpos;
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k1 = pOutput[wpos];
            v1 = wpos;
        }
        wpos += cData._warpSize;

        Float minValue = -MAX_VALUE;
        uint32_t rpos = 64;
        uint32_t bufferSize = 0;
        while (rpos < width)
        {
            unsigned wpos = rpos + tgx;
            Float key = -MAX_VALUE;
            uint32_t value = wpos;
            if (wpos < width)
            {
                key = pOutput[wpos];
            }

            uint32_t count = BALLOT(key > minValue);
            if (key > minValue)
            {
                uint32_t mask = 0xffffffff >> (32 - tgx);
                uint32_t offset = __popc(count & mask);
                offset += bufferSize;
                psKey[offset] = key;
                psValue[offset] = value;
            }
            bufferSize += __popc(count);

            if (bufferSize >= 64)
            {
                k2 = psKey[tgx];
                v2 = psValue[tgx];
                k3 = psKey[tgx + cData._warpSize];
                v3 = psValue[tgx + cData._warpSize];
                BITONICSORT128_128();

                minValue = SHFL(k1, cData._warpSize - 1);

                bufferSize -= 64;
                if (tgx < bufferSize)
                {
                    psKey[tgx] = psKey[tgx + 64];
                    psValue[tgx] = psValue[tgx + 64];
                }
            }

            rpos += cData._warpSize;
        }

        if ((bufferSize > 0) || (width <= 64))
        {
            k2 = -MAX_VALUE;
            k3 = -MAX_VALUE;
            v2 = 0;
            v3 = 0;

            if (tgx < bufferSize)
            {
                k2 = psKey[tgx];
                v2 = psValue[tgx];
            }
            if (tgx + cData._warpSize < bufferSize)
            {
                k3 = psKey[tgx + cData._warpSize];
                v3 = psValue[tgx + cData._warpSize];
            }

            BITONICSORT128_128();
        }

        Float* pKey = pKeyBuffer + pos * k;
        uint32_t* pValue = pValueBuffer + pos * k;
        wpos = tgx;
        if (wpos < k)
        {
            pKey[wpos] = k0;
            pValue[wpos] = v0;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k1;
            pValue[wpos] = v1;
        }
        wpos += cData._warpSize;
    }
}

/**
 * @brief CUDA kernel to calculate the output using 256-bit data width.
 *
 * @param pOutputBuffer Pointer to the output buffer.
 * @param pKeyBuffer Pointer to the key buffer.
 * @param pValueBuffer Pointer to the value buffer.
 * @param batch The number of batches.
 * @param width The width of the data.
 * @param k The value of k.
 */
__global__ void kCalculateOutput_256_kernel(Float* pOutputBuffer, Float* pKeyBuffer, uint32_t* pValueBuffer, uint32_t batch, uint32_t width, uint32_t k)
{
    __shared__ Float sKey[288 * 4];
    __shared__ uint32_t sValue[288 * 4];

    uint32_t pos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (pos < batch)
    {
        Float *pOutput = pOutputBuffer + pos * width;
        uint32_t offset = threadIdx.x >> cData._warpBits;
        Float* psKey = &sKey[288 * offset];
        uint32_t* psValue = &sValue[288 * offset];

        Float minValue = -MAX_VALUE;
        uint32_t bufferSize = 0;
        uint32_t rpos = 256;
        while (rpos < width - cData._warpSize)
        {
            Float key = -MAX_VALUE;
            uint32_t value = rpos + tgx;
            if (value < width)
            {
                key = pOutput[value];
            }

            uint32_t count = BALLOT(key > minValue);
            if (key > minValue)
            {
                uint32_t mask = 0xffffffff >> (32 - tgx);
                uint32_t offset = __popc(count & mask);
                offset += bufferSize;
                psKey[offset] = key;
                psValue[offset] = value;
            }
            bufferSize += __popc(count);

            if (bufferSize >= 256)
            {
                Float tempKey, tempValue;
                tempKey = psKey[tgx];
                tempValue = psValue[tgx];
                BITONICSORT512_512();
                minValue = SHFL(k7, cData._warpSize - 1);
                bufferSize -= 256;
                if (tgx < bufferSize)
                {
                    psKey[tgx] = psKey[tgx + 256];
                    psValue[tgx] = psValue[tgx + 256];
                }
            }

            rpos += cData._warpSize;
        }

        if (bufferSize > 0 || width <= 256)
        {
            Float tempKey, tempValue;
            tempKey = psKey[tgx];
            tempValue = psValue[tgx];
            BITONICSORT512_512();
        }

        Float* pKey = pKeyBuffer + pos * k;
        uint32_t* pValue = pValueBuffer + pos * k;
        uint32_t wpos = tgx;
        if (wpos < k)
        {
            pKey[wpos] = psKey[wpos];
            pValue[wpos] = psValue[wpos];
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = psKey[wpos];
            pValue[wpos] = psValue[wpos];
        }
    }
}

/**
 * @brief Calculates the output using the specified parameters.
 *
 * @param pOutput Pointer to the output buffer.
 * @param pKey Pointer to the key buffer.
 * @param pValue Pointer to the value buffer.
 * @param batch The number of batches.
 * @param width The width of the data.
 * @param k The value of k.
 */
void kCalculateOutput(Float* pOutput, Float *pKey, uint32_t* pValue, uint32_t batch, uint32_t width, uint32_t k)
{
    uint32_t blocks = (batch + 3) / 4;
    int threadsPerBlock = 128;

    if (k > 32 && k <= 64)
    {
        kCalculateOutput_kernel<<<blocks, threadsPerBlock>>>(pOutput, pKey, pValue, batch, width, k, 64);
        LAUNCHERROR("kCalculateOutput_kernel (64)");
    }
    else if (k > 64 && k <= 128)
    {
        kCalculateOutput_kernel<<<blocks, threadsPerBlock>>>(pOutput, pKey, pValue, batch, width, k, 128);
        LAUNCHERROR("kCalculateOutput_kernel (128)");
    }
    else if (k > 128)
    {
        kCalculateOutput_kernel<<<blocks, threadsPerBlock>>>(pOutput, pKey, pValue, batch, width, k, 256);
        LAUNCHERROR("kCalculateOutput_kernel (256)");
    }
    else
    {
        kCalculateOutput_kernel<<<blocks, threadsPerBlock>>>(pOutput, pKey, pValue, batch, width, k, 32);
        LAUNCHERROR("kCalculateOutput_kernel (32)");
    }
}

/**
 * \brief CUDA kernel to calculate output.
 *
 * \param pOutputKey Pointer to the output key array.
 * \param pOutputValue Pointer to the output value array.
 * \param pKeyBuffer Pointer to the key buffer.
 * \param pValueBuffer Pointer to the value buffer.
 * \param batch The number of batches.
 * \param width The width of the data.
 * \param k The value of k.
 */
__global__ void kCalculateOutput_kernel(Float* pOutputKey, Float* pOutputValue, Float* pKeyBuffer, Float* pValueBuffer, uint32_t batch, uint32_t width, uint32_t k)
{
    __shared__ volatile Float sKey[160 * 4];
    __shared__ volatile Float sValue[160 * 4];

    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (pos < batch)
    {
        pOutputKey += pos * width;
        pOutputValue += pos * width;
        uint32_t offset = threadIdx.x >> cData._warpBits;
        volatile Float* psKey = &sKey[160 * offset];
        volatile Float* psValue = &sValue[160 * offset];

        Float k0 = -MAX_VALUE;
        Float k1 = -MAX_VALUE;
        Float k2 = -MAX_VALUE;
        Float k3 = -MAX_VALUE;
        Float v0 = 0.0f;
        Float v1 = 0.0f;
        Float v2 = 0.0f;
        Float v3 = 0.0f;

        uint32_t wpos = tgx;
        if (wpos < width)
        {
            k0 = pOutputKey[wpos];
            v0 = pOutputValue[wpos];
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k1 = pOutputKey[wpos];
            v1 = pOutputValue[wpos];
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k2 = pOutputKey[wpos];
            v2 = pOutputValue[wpos];
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k3 = pOutputKey[wpos];
            v3 = pOutputValue[wpos];
        }

        Float minValue = -MAX_VALUE;
        uint32_t rpos = 128;
        uint32_t bufferSize = 0;
        Float key1, key2;
        Float value1, value2;
        bool flag;
        while (rpos < width)
        {
            uint32_t wpos = rpos + tgx;
            Float key = -MAX_VALUE;
            Float value = 0.0f;
            if (wpos < width)
            {
                key = pOutputKey[wpos];
                value = pOutputValue[wpos];
            }

            uint32_t count = BALLOT(key > minValue);
            if (key > minValue)
            {
                uint32_t mask = 0xffffffff >> (32 - tgx);
                uint32_t offset = __popc(count & mask);
                offset += bufferSize;
                psKey[offset] = key;
                psValue[offset] = value;
            }
            bufferSize += __popc(count);

            if (bufferSize >= 128)
            {
                if (tgx < bufferSize)
                {
                    k3 = psKey[tgx];
                    v3 = psValue[tgx];
                }
                BITONICSORT256_256();
                bufferSize -= 128;
                if (tgx < bufferSize)
                {
                    psKey[tgx] = psKey[tgx + 128];
                    psValue[tgx] = psValue[tgx + 128];
                }
            }

            rpos += cData._warpSize;
        }

        if ((bufferSize > 0) || (width <= 128))
        {
            if (tgx + 2 * cData._warpSize < bufferSize)
            {
                k3 = psKey[tgx + 2 * cData._warpSize];
                v3 = psValue[tgx + 2 * cData._warpSize];
            }
            if (tgx + 3 * cData._warpSize < bufferSize)
            {
                k3 = psKey[tgx + 3 * cData._warpSize];
                v3 = psValue[tgx + 3 * cData._warpSize];
            }
            BITONICSORT256_256();
        }

        Float* pKey = pKeyBuffer + pos * k;
        Float* pValue = pValueBuffer + pos * k;
        wpos = tgx;
        if (wpos < k)
        {
            pKey[wpos] = k0;
            pValue[wpos] = v0;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k1;
            pValue[wpos] = v1;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k2;
            pValue[wpos] = v2;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k3;
            pValue[wpos] = v3;
        }
    }
}

/**
 * \brief Function to calculate output.
 *
 * \param pOutputKey Pointer to the output key array.
 * \param pOutputValue Pointer to the output value array.
 * \param pKey Pointer to the key array.
 * \param pValue Pointer to the value array.
 * \param batch The number of batches.
 * \param width The width of the data.
 * \param k The value of k.
 */
void kCalculateOutput(Float* pOutputKey, Float* pOutputValue, Float* pKey, Float* pValue, uint32_t batch, uint32_t width, uint32_t k)
{
    uint32_t threads = 128;
    uint32_t blocks = (batch + 3) / 4;
    kCalculateOutput_kernel<<<blocks, threads>>>(pOutputKey, pOutputValue, pKey, pValue, batch, width, k);
    LAUNCHERROR("kCalculateOutput_kernel");
}

/**
 * @brief CUDA kernel to calculate the output key-value pairs.
 *
 * @param pOutputKey Pointer to the output key buffer.
 * @param pOutputValue Pointer to the output value buffer.
 * @param pKeyBuffer Pointer to the key buffer.
 * @param pValueBuffer Pointer to the value buffer.
 * @param batch The number of batches.
 * @param width The width of the data.
 * @param k The number of elements to be sorted.
 */
__global__ void kCalculateOutput_kernel(Float* pOutputKey, uint32_t* pOutputValue, Float* pKeyBuffer, uint32_t* pValueBuffer, uint32_t batch, uint32_t width, uint32_t k)
{
    __shared__ volatile Float sKey[160 * 4];
    __shared__ volatile uint32_t sValue[160 * 4];
    uint32_t pos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (pos < batch)
    {
        pOutputKey += pos * width;
        pOutputValue += pos * width;
        uint32_t offset = threadIdx.x >> cData._warpBits;
        volatile Float* psKey = &sKey[160 * offset];
        volatile uint32_t* psValue = &sValue[160 * offset];

        Float k0 = -MAX_VALUE;
        Float k1 = -MAX_VALUE;
        Float k2 = -MAX_VALUE;
        Float k3 = -MAX_VALUE;
        Float k4 = -MAX_VALUE;
        Float k5 = -MAX_VALUE;
        Float k6 = -MAX_VALUE;
        Float k7 = -MAX_VALUE;
        uint32_t v0 = 0;
        uint32_t v1 = 0;
        uint32_t v2 = 0;
        uint32_t v3 = 0;
        uint32_t v4 = 0;
        uint32_t v5 = 0;
        uint32_t v6 = 0;
        uint32_t v7 = 0;

        uint32_t wpos = tgx;
        if (wpos < width)
        {
            k0 = pOutputKey[wpos];
            v0 = pOutputValue[wpos];
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k1 = pOutputKey[wpos];
            v1 = pOutputValue[wpos];
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k2 = pOutputKey[wpos];
            v2 = pOutputValue[wpos];
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k3 = pOutputKey[wpos];
            v3 = pOutputValue[wpos];
        }

        Float minValue = -MAX_VALUE;
        uint32_t rpos = 128;
        uint32_t bufferSize = 0;
        Float key1, key2;
        uint32_t value1, value2;
        uint32_t otgx;
        bool flag;
        while (rpos < width)
        {
            uint32_t wpos = rpos + tgx;
            Float key = -MAX_VALUE;
            uint32_t value = 0;
            if (wpos < width)
            {
                key = pOutputKey[wpos];
                value = pOutputValue[wpos];
            }

            uint32_t count = BALLOT(key > minValue);
            if (key > minValue)
            {
                uint32_t mask = 0xffffffff >> (32 - tgx);
                uint32_t offset = __popc(count & mask);
                offset += bufferSize;
                psKey[offset] = key;
                psValue[offset] = value;
            }
            bufferSize += __popc(count);

            if (bufferSize >= 128)
            {
                if (tgx < bufferSize)
                {
                    k4 = psKey[tgx];
                    v4 = psValue[tgx];
                }
                BITONICSORT256_256();
                bufferSize -= 128;
                if (tgx < bufferSize)
                {
                    psKey[tgx] = psKey[tgx + 128];
                    psValue[tgx] = psValue[tgx + 128];
                }
            }

            rpos += cData._warpSize;
        }

        if ((bufferSize > 0) || (width <= 128))
        {
            k4 = -MAX_VALUE;
            k5 = -MAX_VALUE;
            k6 = -MAX_VALUE;
            k7 = -MAX_VALUE;
            v4 = 0;
            v5 = 0;
            v6 = 0;
            v7 = 0;

            if (tgx < bufferSize)
            {
                k4 = psKey[tgx];
                v4 = psValue[tgx];
            }
            if (tgx + cData._warpSize < bufferSize)
            {
                k5 = psKey[tgx + cData._warpSize];
                v5 = psValue[tgx + cData._warpSize];
            }
            if (tgx + 2 * cData._warpSize < bufferSize)
            {
                k6 = psKey[tgx + 2 * cData._warpSize];
                v6 = psValue[tgx + 2 * cData._warpSize];
            }
            if (tgx + 3 * cData._warpSize < bufferSize)
            {
                k7 = psKey[tgx + 3 * cData._warpSize];
                v7 = psValue[tgx + 3 * cData._warpSize];
            }

            BITONICSORT256_256();
        }

        Float* pKey = pKeyBuffer + pos * k;
        uint32_t* pValue = pValueBuffer + pos * k;
        wpos = tgx;
        if (wpos < k)
        {
            pKey[wpos] = k0;
            pValue[wpos] = v0;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k1;
            pValue[wpos] = v1;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k2;
            pValue[wpos] = v2;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k3;
            pValue[wpos] = v3;
        }
    }
}

/**
 * @brief Function to calculate the output key-value pairs using CUDA.
 *
 * @param pOutputKey Pointer to the output key buffer.
 * @param pOutputValue Pointer to the output value buffer.
 * @param pKey Pointer to the key buffer.
 * @param pValue Pointer to the value buffer.
 * @param batch The number of batches.
 * @param width The width of the data.
 * @param k The number of elements to be sorted.
 */
void kCalculateOutput(Float* pOutputKey, uint32_t* pOutputValue, Float* pKey, uint32_t* pValue, uint32_t batch, uint32_t width, uint32_t k)
{
    uint32_t threads = 128;
    uint32_t blocks = (batch + 3) / 4;
    kCalculateOutput_kernel<<<blocks, threads>>>(pOutputKey, pOutputValue, pKey, pValue, batch, width, k);
    LAUNCHERROR("kCalculateOutput_kernel");
}

/**
 * @brief Applies weight normalization to the input weights.
 *
 * This function normalizes the weights based on the provided norm value. It operates on a CUDA global memory array.
 *
 * @param norm The normalization factor to be applied to the weights.
 * @param outputStride The stride between consecutive output elements.
 * @param inputStride The stride between consecutive input elements.
 * @param pWeight Pointer to the weight array to be normalized.
 */
__global__ void kNormalizeWeights_kernel(Float norm, uint32_t outputStride, uint32_t inputStride, Float* pWeight)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < outputStride)
    {
        Float r2 = 0.0f;
        Float* pEnd = pWeight + pos + outputStride * inputStride;
        Float* p = pWeight + pos;

        while (p < pEnd)
        {
            Float x = *p;
            r2 += x * x;
            p += outputStride;
        }

        if (r2 > norm * norm)
        {
            norm *= rsqrt(r2);
            p = pWeight + pos;
            while (p < pEnd)
            {
                *p *= norm;
                p += outputStride;
            }
        }
    }
}

/**
 * @brief Invokes the weight normalization kernel function.
 *
 * This function determines the thread and block dimensions for invoking the weight normalization kernel,
 * and launches the kernel with the specified parameters.
 *
 * @param norm The normalization factor to be applied to the weights.
 * @param outputStride The stride between consecutive output elements.
 * @param inputStride The stride between consecutive input elements.
 * @param pWeight Pointer to the weight array to be normalized.
 */
void kNormalizeWeights(Float norm, uint32_t outputStride, uint32_t inputStride, Float* pWeight)
{
    uint32_t threads = 128;
    uint32_t blocks = (outputStride + threads - 1) / threads;
    kNormalizeWeights_kernel<<<blocks, threads>>>(norm, outputStride, inputStride, pWeight);
    LAUNCHERROR("kNormalizeWeights_kernel");
}

/**
 * @brief CUDA kernel to calculate weight magnitudes.
 *
 * This CUDA kernel calculates the magnitude of weights for each output stride.
 *
 * @param outputStride The output stride value.
 * @param inputStride The input stride value.
 * @param pWeight Pointer to the weight array.
 * @param pMagnitude Pointer to the magnitude array.
 */
__global__ void kCalculateWeightMagnitudes_kernel(uint32_t outputStride, uint32_t inputStride, Float* pWeight, Float* pMagnitude)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < outputStride)
    {
        Float r2 = 0.0f;
        Float* pEnd = pWeight + pos + outputStride * inputStride;
        Float* p = pWeight + pos;

        while (p < pEnd)
        {
            Float x = *p;
            r2 += x * x;
            p += outputStride;
        }

        pMagnitude[pos] = r2;
    }
}

/**
 * @brief Function to calculate weight magnitudes.
 *
 * This function launches the CUDA kernel to calculate the magnitude of weights for each output stride.
 *
 * @param outputStride The output stride value.
 * @param inputStride The input stride value.
 * @param pWeight Pointer to the weight array.
 * @param pMagnitude Pointer to the magnitude array.
 */
void kCalculateWeightMagnitudes(uint32_t outputStride, uint32_t inputStride, Float* pWeight, Float* pMagnitude)
{
    uint32_t threads = 128;
    uint32_t blocks = (outputStride + threads - 1) / threads;
    kCalculateWeightMagnitudes_kernel<<<blocks, threads>>>(outputStride, inputStride, pWeight, pMagnitude);
    LAUNCHERROR("kCalculateWeightMagnitudes_kernel");
}

/**
 * @brief Kernel function to normalize weight magnitudes.
 *
 * This kernel function normalizes the weight magnitudes based on a given norm.
 *
 * @param norm The normalization factor.
 * @param outputStride The stride of the output.
 * @param inputStride The stride of the input.
 * @param pWeight Pointer to the weight array.
 * @param pMagnitude Pointer to the magnitude array.
 */
__global__ void kNormalizeWeightMagnitudes_kernel(Float norm, uint32_t outputStride, uint32_t inputStride, Float* pWeight, Float* pMagnitude)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < outputStride)
    {
        Float r2 = pMagnitude[pos];
        Float* pEnd = pWeight + pos + outputStride * inputStride;
        Float* p = pWeight + pos;

        if (r2 > norm * norm)
        {
            norm *= rsqrt(r2);
            while (p < pEnd)
            {
                *p *= norm;
                p += outputStride;
            }
        }
    }
}

/**
 * @brief Normalize weight magnitudes.
 *
 * This function normalizes the weight magnitudes based on a given norm.
 *
 * @param norm The normalization factor.
 * @param outputStride The stride of the output.
 * @param inputStride The stride of the input.
 * @param pWeight Pointer to the weight array.
 * @param pMagnitude Pointer to the magnitude array.
 */
void kNormalizeWeightMagnitudes(Float norm, uint32_t outputStride, uint32_t inputStride, Float* pWeight, Float* pMagnitude)
{
    uint32_t threads = 128;
    uint32_t blocks = (outputStride + threads - 1) / threads;
    kNormalizeWeightMagnitudes_kernel<<<blocks, threads>>>(norm, outputStride, inputStride, pWeight, pMagnitude);
    LAUNCHERROR("kNormalizeWeightMagnitudes_kernel");
}

/**
 * @brief Kernel function to calculate scaled biased dropout.
 *
 * This CUDA global function applies scaled biased dropout to an input array based on randomly generated values.
 * It modifies the input array according to the dropout probability and scaling factors.
 *
 * @param pUnit    Pointer to the input array.
 * @param pRandom  Pointer to the array of randomly generated values.
 * @param p        Dropout probability.
 * @param target   Target value for dropout.
 * @param a        Scaling factor for the input values.
 * @param b        Bias factor for the input values.
 * @param size     Size of the input array.
 */
__global__ void kCalculateScaledBiasedDropout_kernel(Float* pUnit, Float* pRandom, Float p, Float target, Float a, Float b, size_t size)
{
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        Float r = pRandom[pos];
        pUnit[pos] = (r < p) ? target : a * pUnit[pos] + b;
    }
}

/**
 * @brief Function to calculate scaled biased dropout.
 *
 * This function generates random values, applies scaled biased dropout to the input array,
 * and handles the CUDA kernel launch.
 *
 * @param pUnit   Pointer to the input array.
 * @param pRandom Pointer to the array of randomly generated values.
 * @param batch   Number of batches.
 * @param stride  Stride of the input array.
 * @param p       Dropout probability.
 * @param target  Target value for dropout.
 * @param a       Scaling factor for the input values.
 * @param b       Bias factor for the input values.
 */
void kCalculateScaledBiasedDropout(Float* pUnit, Float* pRandom, uint32_t batch, uint32_t stride, Float p, Float target, Float a, Float b)
{
    curandGenerateUniform(getGpu()._RNG, pRandom, batch * stride);
    size_t threads = getGpu()._threadsPerBlock;
    size_t blocks = (batch * stride + threads - 1) / threads;
    kCalculateScaledBiasedDropout_kernel<<<blocks, threads>>>(pUnit, pRandom, p, target, a, b, batch * stride);
    LAUNCHERROR("kCalculateScaledBiasedDropout_kernel");
}

/**
 * \brief CUDA kernel to calculate dropout on a global memory array.
 *
 * \param pUnit Pointer to the array of units.
 * \param pRandom Pointer to the array of random numbers.
 * \param p Dropout probability.
 * \param scale Scaling factor.
 * \param target Target value for dropout.
 * \param size Size of the array.
 */
__global__ void kCalculateDropout_kernel(Float* pUnit, Float* pRandom, Float p, Float scale, Float target, size_t size)
{
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        Float r = pRandom[pos];
        pUnit[pos] = (r < p) ? target : scale * pUnit[pos];
    }
}

/**
 * \brief Function to calculate dropout using CUDA.
 *
 * \param pUnit Pointer to the array of units.
 * \param pRandom Pointer to the array of random numbers.
 * \param batch Number of batches.
 * \param stride Stride size.
 * \param p Dropout probability.
 * \param target Target value for dropout.
 */
void kCalculateDropout(Float* pUnit, Float* pRandom, uint32_t batch, uint32_t stride, Float p, Float target)
{
    curandGenerateUniform(getGpu()._RNG, pRandom, batch * stride);
    size_t threads = getGpu()._threadsPerBlock;
    size_t blocks = (batch * stride + threads - 1) / threads;
    Float scale = (target == 0.0f) ? 1.0f / (1.0f - p) : 1.0f;
    kCalculateDropout_kernel<<<blocks, threads>>>(pUnit, pRandom, p, scale, target, batch * stride);
    LAUNCHERROR("kCalculateDropout_kernel");
}

/**
 * @brief Kernel function to calculate the element-wise maximum between two arrays.
 *
 * This kernel function compares each element of the source array with the corresponding element
 * in the destination array and updates the destination element with the maximum value.
 *
 * @param pSrc Pointer to the source array.
 * @param size Number of elements in the arrays.
 * @param pDst Pointer to the destination array.
 */
__global__ void kCalculateMaxout_kernel(Float* pSrc, size_t size, Float* pDst)
{
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        Float s = pSrc[pos];
        Float d = pDst[pos];
        pDst[pos] = (s > d) ? s : d;
    }
}

/**
 * @brief Calculates the element-wise maximum between two arrays using CUDA.
 *
 * This function launches the CUDA kernel to perform the element-wise maximum operation
 * between the source array and the destination array. The result is stored in the destination array.
 *
 * @param pSrc Pointer to the source array.
 * @param size Number of elements in the arrays.
 * @param pDst Pointer to the destination array.
 */
void kCalculateMaxout(Float* pSrc, size_t size, Float* pDst)
{
    size_t threads = getGpu()._threadsPerBlock;
    size_t blocks = (size + threads - 1) / threads;
    kCalculateMaxout_kernel<<<blocks, threads>>>(pSrc, size, pDst);
    LAUNCHERROR("kCalculateMaxout_kernel");
}

/**
 * @brief CUDA kernel to calculate the cosine similarity between two vectors.
 *
 * @param pVector1 Pointer to the first input vector.
 * @param pVector2 Pointer to the second input vector.
 * @param stride Number of elements in each vector.
 * @param pDPOut Pointer to the output array for dot product values.
 * @param pAOut Pointer to the output array for the magnitude of the first vector.
 * @param pBOut Pointer to the output array for the magnitude of the second vector.
 * @param outStride Stride for accessing output arrays.
 */
__global__ void kCalculateCosine_kernel(Float* pVector1, Float* pVector2, uint32_t stride, Float* pDPOut, Float* pAOut, Float* pBOut, uint32_t outStride)
{
    extern __shared__ Float sData[];
    Float* sDP = sData;
    Float* sA = sData + blockDim.x;
    Float* sB = sData + 2 * blockDim.x;

    uint32_t pos = blockIdx.x * stride + threadIdx.x;
    Float dp = 0.0f;
    Float al = 0.0f;
    Float bl = 0.0f;

    while (pos < blockIdx.x * stride + stride)
    {
        Float a = pVector1[pos];
        Float b = pVector2[pos];
        dp += a * b;
        al += a * a;
        bl += b * b;
        pos += blockDim.x;
    }

    sDP[threadIdx.x] = dp;
    sA[threadIdx.x] = al;
    sB[threadIdx.x] = bl;

    __syncthreads();

    uint32_t limit = min(blockDim.x, stride);
    for (uint32_t s = limit / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s && threadIdx.x + s < limit)
        {
            sDP[threadIdx.x] += sDP[threadIdx.x + s];
            sA[threadIdx.x] += sA[threadIdx.x + s];
            sB[threadIdx.x] += sB[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        al = sqrtf(sA[0]) + 1.0e-08f;
        bl = sqrtf(sB[0]) + 1.0e-08f;
        dp = sDP[0] / (al * bl);
        pAOut[blockIdx.x * outStride] = al;
        pBOut[blockIdx.x * outStride] = bl;
        pDPOut[blockIdx.x * outStride] = dp;
    }
}

/**
 * @brief CUDA wrapper function to calculate the cosine similarity between two sets of vectors.
 *
 * @param pVector1 Pointer to the first set of input vectors.
 * @param pVector2 Pointer to the second set of input vectors.
 * @param batch Number of vectors in each set.
 * @param stride Number of elements in each vector.
 * @param pDPOut Pointer to the output array for dot product values.
 * @param pAOut Pointer to the output array for the magnitude of the vectors in the first set.
 * @param pBOut Pointer to the output array for the magnitude of the vectors in the second set.
 * @param outStride Stride for accessing output arrays.
 */
void kCalculateCosine(Float* pVector1, Float* pVector2, uint32_t batch, uint32_t stride, Float* pDPOut, Float* pAOut, Float* pBOut, uint32_t outStride)
{
    uint32_t threads = min(stride, getGpu()._threadsPerBlock);
    kCalculateCosine_kernel<<<batch, threads, 3 * threads * sizeof(Float)>>>(pVector1, pVector2, stride, pDPOut, pAOut, pBOut, outStride);
    LAUNCHERROR("kCalculateCosine_kernel");
}

/**
 * \brief Calculates the dot product of two vectors on the GPU.
 *
 * This CUDA kernel function calculates the dot product of two vectors represented
 * by the input arrays pVector1In and pVector2In. The dot product is computed
 * in parallel by multiple threads running on the GPU.
 *
 * \param pVector1In      Pointer to the first input vector.
 * \param pVector2In      Pointer to the second input vector.
 * \param strideIn        Stride value indicating the length of the input vectors.
 * \param pDPOut          Pointer to the output array to store the dot product.
 * \param strideOut       Stride value indicating the length of the output array.
 */
__global__ void kCalculateDotProduct_kernel(const float* pVector1In, const float* pVector2In, uint32_t strideIn, float* pDPOut, uint32_t strideOut)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t threadStride = gridDim.x * blockDim.x;

    float dp = 0.0f;

    for (uint32_t pos = idx; pos < strideIn; pos += threadStride)
    {
        float a = pVector1In[pos];
        float b = pVector2In[pos];
        dp += a * b;
    }

    dp = warpReduceSum(dp);

    if (threadIdx.x % warpSize == 0)
    {
        uint32_t warpIdx = threadIdx.x / warpSize;
        __shared__ float sDP[32];
        sDP[warpIdx] = dp;
        __syncthreads();

        if (warpIdx == 0)
        {
            float blockSum = warpReduceSum(sDP[threadIdx.x]);
            if (threadIdx.x == 0)
            {
                pDPOut[blockIdx.x * strideOut] = blockSum;
            }
        }
    }
}

/**
 * \brief Wrapper function to launch the dot product calculation on the GPU.
 *
 * This function sets up the necessary parameters and launches the CUDA kernel
 * function kCalculateDotProduct_kernel to calculate the dot product of two vectors
 * on the GPU.
 *
 * \param pVector1In      Pointer to the first input vector.
 * \param pVector2In      Pointer to the second input vector.
 * \param batch           Number of vectors in the batch.
 * \param strideIn        Stride value indicating the length of the input vectors.
 * \param pDPOut          Pointer to the output array to store the dot products.
 * \param strideOut       Stride value indicating the length of the output array.
 */
void kCalculateDotProduct(const float* pVector1In, const float* pVector2In, uint32_t batch, uint32_t strideIn, float* pDPOut, uint32_t strideOut)
{
    const unsigned int threads = max(32, min(strideIn, getGpu()._threadsPerBlock));
    const unsigned int blocks = batch;
    kCalculateDotProduct_kernel<<<blocks, threads>>>(pVector1In, pVector2In, strideIn, pDPOut, strideOut);
    LAUNCHERROR("kCalculateDotProduct_kernel");
}

/**
 * \brief Initializes the sorting operation on the GPU.
 *
 * This template function initializes the sorting operation for a given number
 * of items on the GPU. It sets up the necessary buffers and returns the size of
 * temporary storage required for the sorting operation.
 *
 * \param items           Number of items to be sorted.
 * \param pbKey           Pointer to the buffer containing the keys to be sorted.
 * \param pbValue         Pointer to the buffer containing the values to be sorted.
 * \return                Size of the temporary storage required for sorting.
 */
template<typename KeyType, typename ValueType>
size_t kInitSort(uint32_t items, GpuBuffer<KeyType>* pbKey, GpuBuffer<ValueType>* pbValue)
{
    uint32_t itemStride = ((items + 511) >> 9) << 9;
    size_t tempBytes;
    cub::DoubleBuffer<KeyType> d_keys(pbKey->_pDevData, pbKey->_pDevData + itemStride);
    cub::DoubleBuffer<ValueType> d_values(pbValue->_pDevData, pbValue->_pDevData + itemStride);
    cub::DeviceRadixSort::SortPairs(NULL, tempBytes, d_keys, d_values, items);
    return tempBytes;
}

/**
 * \brief Performs sorting operation on the GPU.
 *
 * This template function performs the sorting operation on the GPU for a given
 * number of items. It takes the input key-value pairs and their temporary storage
 * as input and sorts them using the DeviceRadixSort algorithm provided by the CUB
 * library.
 *
 * \param items           Number of items to be sorted.
 * \param pKey0           Pointer to the first array of keys.
 * \param pKey1           Pointer to the second array of keys.
 * \param pValue0         Pointer to the first array of values.
 * \param pValue1         Pointer to the second array of values.
 * \param pTemp           Pointer to the temporary storage.
 * \param tempBytes       Size of the temporary storage in bytes.
 * \return                Boolean indicating the success of the sorting operation.
 */
template<typename KeyType, typename ValueType>
bool kSort(uint32_t items, KeyType* pKey0, KeyType* pKey1, ValueType* pValue0, ValueType* pValue1, char* pTemp, size_t tempBytes)
{
    cub::DoubleBuffer<KeyType> d_keys(pKey0, pKey1);
    cub::DoubleBuffer<ValueType> d_values(pValue0, pValue1);
    cub::DeviceRadixSort::SortPairs(pTemp, tempBytes, d_keys, d_values, items);
    return true;
}

/**
 * @brief CUDA kernel for element-wise addition of scaled buffers.
 *
 * @param pDst Pointer to the destination buffer.
 * @param pSrc Pointer to the source buffer.
 * @param scale Scaling factor to be applied.
 * @param size Size of the buffers.
 */
__global__ void kAddScaleBuffers_kernel(Float* pDst, Float* pSrc, Float scale, unsigned int size)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        pDst[idx] += pSrc[idx] * scale;
    }
}

/**
 * @brief Perform element-wise addition of scaled buffers using CUDA.
 *
 * @param pDst Pointer to the destination buffer.
 * @param pSrc Pointer to the source buffer.
 * @param scale Scaling factor to be applied.
 * @param size Size of the buffers.
 */
void kAddScaleBuffers(Float* pDst, Float* pSrc, Float scale, unsigned int size)
{
    unsigned int threads = getGpu()._threadsPerBlock;
    unsigned int blocks = (size + threads - 1) / threads;
    kAddScaleBuffers_kernel<<<blocks, threads>>>(pDst, pSrc, scale, size);
    LAUNCHERROR("kAddScaleBuffers_kernel");
}
/**
 * @brief CUDA kernel for element-wise addition of buffers.
 *
 * @param pDst Pointer to the destination buffer.
 * @param pSrc Pointer to the source buffer.
 * @param size Size of the buffers.
 */
__global__ void kAddBuffers_kernel(Float* pDst, Float* pSrc, unsigned int size)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        pDst[idx] += pSrc[idx];
    }
}

/**
 * @brief Perform element-wise addition of buffers using CUDA.
 *
 * @param pDst Pointer to the destination buffer.
 * @param pSrc Pointer to the source buffer.
 * @param size Size of the buffers.
 * @param stream CUDA stream to be used for the kernel launch.
 */
void kAddBuffers(Float* pDst, Float* pSrc, unsigned int size, cudaStream_t stream)
{
    if (size == 0)
        return;

    unsigned int threads = getGpu()._threadsPerBlock;
    unsigned int blocks = (size + threads - 1) / threads;
    kAddBuffers_kernel<<<blocks, threads, 0, stream>>>(pDst, pSrc, size);
    LAUNCHERROR("kAddBuffers_kernel");
}

/**
 * @brief CUDA kernel for element-wise addition of 2D buffers.
 *
 * @param pDst Pointer to the destination buffer.
 * @param dpitch Pitch (in bytes) of the destination buffer.
 * @param pSrc Pointer to the source buffer.
 * @param spitch Pitch (in bytes) of the source buffer.
 * @param width Width of the buffers.
 */
__global__ void kAddBuffers2D_kernel(Float* pDst, uint32_t dpitch, Float* pSrc, uint32_t spitch, uint32_t width)
{
    unsigned int yOffset = blockIdx.y * blockDim.x + threadIdx.x;
    if (yOffset < width)
    {
        uint64_t dpos = blockIdx.x * dpitch + yOffset;
        uint64_t spos = blockIdx.x * spitch + yOffset;
        pDst[dpos] += pSrc[spos];
    }
}

/**
 * @brief Perform element-wise addition of 2D buffers using CUDA.
 *
 * @param pDst Pointer to the destination buffer.
 * @param dpitch Pitch (in bytes) of the destination buffer.
 * @param pSrc Pointer to the source buffer.
 * @param spitch Pitch (in bytes) of the source buffer.
 * @param width Width of the buffers.
 * @param height Height of the buffers.
 * @param stream CUDA stream to be used for the kernel launch.
 */
void kAddBuffers2D(Float* pDst, uint32_t dpitch, Float* pSrc, uint32_t spitch, uint32_t width, uint32_t height, cudaStream_t stream)
{
    if ((height == 0) || (width == 0))
        return;

    unsigned int threads = getGpu()._threadsPerBlock;
    dim3 grid(height, (width + threads - 1) / threads);
    kAddBuffers2D_kernel<<<grid, threads, 0, stream>>>(pDst, dpitch, pSrc, spitch, width);
    LAUNCHERROR("kAddBuffers2D_kernel");
}

/**
 * @brief CUDA kernel for copying 2D data from source to destination.
 *
 * @param pDst Pointer to the destination memory.
 * @param dpitch Pitch (in bytes) of the destination memory.
 * @param pSrc Pointer to the source memory.
 * @param spitch Pitch (in bytes) of the source memory.
 * @param width Width of the data to be copied.
 */
__global__ void kCopy2D_kernel(Float* pDst, uint32_t dpitch, Float* pSrc, uint32_t spitch, uint32_t width)
{
    unsigned int yOffset = blockIdx.y * blockDim.x + threadIdx.x;
    if (yOffset < width)
    {
        uint64_t dpos = blockIdx.x * dpitch + yOffset;
        uint64_t spos = blockIdx.x * spitch + yOffset;
        pDst[dpos] = pSrc[spos];
    }
}

/**
 * @brief Copy 2D data from source to destination using CUDA.
 *
 * @param pDst Pointer to the destination memory.
 * @param dpitch Pitch (in bytes) of the destination memory.
 * @param pSrc Pointer to the source memory.
 * @param spitch Pitch (in bytes) of the source memory.
 * @param width Width of the data to be copied.
 * @param height Height of the data to be copied.
 * @param stream CUDA stream to be used for the kernel launch.
 */
void kCopy2D(Float* pDst, uint32_t dpitch, Float* pSrc, uint32_t spitch, uint32_t width, uint32_t height, cudaStream_t stream)
{
    if ((height == 0) || (width == 0))
        return;

    unsigned int threads = getGpu()._threadsPerBlock;
    dim3 grid(height, (width + threads - 1) / threads);
    kCopy2D_kernel<<<grid, threads, 0, stream>>>(pDst, dpitch, pSrc, spitch, width);
    LAUNCHERROR("kCopy2D_kernel");
}
template size_t kInitSort<Float, Float>  (uint32_t, GpuBuffer<Float>*, GpuBuffer<Float>*);
template size_t kInitSort<uint32_t, Float> (uint32_t, GpuBuffer<uint32_t>*, GpuBuffer<Float>*);
template size_t kInitSort<Float, uint32_t> (uint32_t, GpuBuffer<Float>*, GpuBuffer<uint32_t>*);
template size_t kInitSort<uint32_t, uint32_t>(uint32_t, GpuBuffer<uint32_t>*, GpuBuffer<uint32_t>*);

/**
 * @brief Sorts data for Float and Float data types.
 * 
 * @tparam Float The data type for the buffers.
 * @param uint32_t The size parameter.
 * @param Float* The pointer to the first buffer.
 * @param Float* The pointer to the second buffer.
 * @param Float* The pointer to the third buffer.
 * @param Float* The pointer to the fourth buffer.
 * @param char* The pointer to the fifth buffer.
 * @param size_t The size parameter.
 * @return bool True if sorting was successful, false otherwise.
 */
template bool kSort<Float, Float>(uint32_t, Float*, Float*, Float*, Float*, char*, size_t);

/**
 * @brief Sorts data for Float and uint32_t data types.
 * 
 * @tparam Float The data type for the first buffers.
 * @tparam uint32_t The data type for the second buffers.
 * @param uint32_t The size parameter.
 * @param Float* The pointer to the first buffer.
 * @param Float* The pointer to the second buffer.
 * @param uint32_t* The pointer to the third buffer.
 * @param uint32_t* The pointer to the fourth buffer.
 * @param char* The pointer to the fifth buffer.
 * @param size_t The size parameter.
 * @return bool True if sorting was successful, false otherwise.
 */
template bool kSort<Float, uint32_t>(uint32_t, Float*, Float*, uint32_t*, uint32_t*, char*, size_t);

/**
 * @brief Sorts data for uint32_t and Float data types.
 * 
 * @tparam uint32_t The data type for the first buffers.
 * @tparam Float The data type for the second buffers.
 * @param uint32_t The size parameter.
 * @param uint32_t* The pointer to the first buffer.
 * @param uint32_t* The pointer to the second buffer.
 * @param Float* The pointer to the third buffer.
 * @param Float* The pointer to the fourth buffer.
 * @param char* The pointer to the fifth buffer.
 * @param size_t The size parameter.
 * @return bool True if sorting was successful, false otherwise.
 */
template bool kSort<uint32_t, Float>(uint32_t, uint32_t*, uint32_t*, Float*, Float*, char*, size_t);

/**
 * @brief Sorts data for uint32_t and uint32_t data types.
 * 
 * @tparam uint32_t The data type for the buffers.
 * @param uint32_t The size parameter.
 * @param uint32_t* The pointer to the first buffer.
 * @param uint32_t* The pointer to the second buffer.
 * @param uint32_t* The pointer to the third buffer.
 * @param uint32_t* The pointer to the fourth buffer.
 * @param char* The pointer to the fifth buffer.
 * @param size_t The size parameter.
 * @return bool True if sorting was successful, false otherwise.
 */
template bool kSort<uint32_t, uint32_t>(uint32_t, uint32_t*, uint32_t*, uint32_t*, uint32_t*, char*, size_t);

/**
 * @brief Explicit instantiation of kernels for a specific data type.
 * 
 * @param T The data type for instantiation.
 */
#define EXPLICITLY_INSTANTIATE_KERNELS(T)                                                                                                                                                       \
template void kLoadSparseAnalogDenoisedInputUnit<T>(uint32_t, uint32_t, uint32_t, Float*, uint64_t*, uint64_t*, uint32_t*, Float*, T*, Float*);                                           \
template void kLoadIndexedSparseAnalogDenoisedInputUnit<T>(uint32_t, uint32_t, uint32_t, Float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, Float*, T*, Float*);                         \
template void kLoadSparseAnalogInputUnit<T>(uint32_t, uint32_t, uint32_t, Float*, uint64_t*, uint64_t*, uint32_t*, Float*, T*);                                                             \
template void kLoadIndexedSparseAnalogInputUnit<T>(uint32_t, uint32_t, uint32_t, Float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, Float*, T*);                                           \
template void kCalculateSparseAnalogZ<T>(uint32_t, uint32_t, uint32_t, Float*, uint64_t*, uint64_t*, uint32_t*, Float*, T*, Float*, Float);                                             \
template void kCalculateIndexedSparseAnalogZ<T>(uint32_t, uint32_t, uint32_t, Float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, Float*, T*, Float*, Float);                           \
template void kCalculateSparseAnalogDenoisedZ<T>(uint32_t, uint32_t, uint32_t, Float*, uint64_t*, uint64_t*, uint32_t*, Float*, T*, Float*, Float*, Float);                           \
template void kCalculateIndexedSparseAnalogDenoisedZ<T>(uint32_t, uint32_t, uint32_t, Float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, Float*, T*, Float*, Float*, Float);         \
template void kCalculateSparseTransposedAnalogMatrix<T>(uint32_t, uint32_t, uint64_t*, uint64_t*, uint32_t*, Float*, T*, uint32_t*, uint32_t*, Float*);                                     \
template void kCalculateIndexedSparseTransposedAnalogMatrix<T>(uint32_t, uint32_t, uint32_t*, uint64_t*, uint64_t*, uint32_t*, Float*, T*, uint32_t*, uint32_t*, Float*);                   \
template void kCalculateSparseTransposedAnalogDenoisedMatrix<T>(uint32_t, uint32_t, uint64_t*, uint64_t*, uint32_t*, Float*, T*, Float*, uint32_t*, uint32_t*, Float*);                   \
template void kCalculateIndexedSparseTransposedAnalogDenoisedMatrix<T>(uint32_t, uint32_t, uint32_t*, uint64_t*, uint64_t*, uint32_t*, Float*, T*, Float*, uint32_t*, uint32_t*, Float*); \
template void kLoadInputUnit<T>(uint32_t, uint32_t, uint32_t, Float*, T*);                                                                                                                    \
template void kLoadIndexedInputUnit<T>(uint32_t, uint32_t, uint32_t, Float*, uint32_t*, T*);




                                                                                 \

EXPLICITLY_INSTANTIATE_KERNELS(Float)

/**
 * @brief Explicit instantiation of kernels for the double data type.
 * 
 * @param T The data type for instantiation (double).
 */
EXPLICITLY_INSTANTIATE_KERNELS(double)

/**
 * @brief Explicit instantiation of kernels for the unsigned char data type.
 * 
 * @param T The data type for instantiation (unsigned char).
 */
EXPLICITLY_INSTANTIATE_KERNELS(unsigned char)

/**
 * @brief Explicit instantiation of kernels for the char data type.
 * 
 * @param T The data type for instantiation (char).
 */
EXPLICITLY_INSTANTIATE_KERNELS(char)

/**
 * @brief Explicit instantiation of kernels for the uint32_t data type.
 * 
 * @param T The data type for instantiation (uint32_t).
 */
EXPLICITLY_INSTANTIATE_KERNELS(uint32_t)

/**
 * @brief Explicit instantiation of kernels for the uint64_t data type.
 * 
 * @param T The data type for instantiation (uint64_t).
 */
EXPLICITLY_INSTANTIATE_KERNELS(uint64_t)

/**
 * @brief Explicit instantiation of kernels for the int32_t data type.
 * 
 * @param T The data type for instantiation (int32_t).
 */
EXPLICITLY_INSTANTIATE_KERNELS(int32_t)

/**
 * @brief Explicit instantiation of kernels for the int64_t data type.
 * 
 * @param T The data type for instantiation (int64_t).
 */
EXPLICITLY_INSTANTIATE_KERNELS(int64_t)
