#include "GpuTypes.h"
#include "Types.h"
#include <limits>

/**
 * @brief Constant data stored on the GPU.
 */
static __constant__ GpuData cData;

/**
 * @brief Copies the GPU data to the constant symbol cData.
 */
void SetKernelsGpuData()
{
    cudaError_t status;
    status = cudaMemcpyToSymbol(cData, &(getGpu()._data), sizeof(GpuData));
    RTERROR(status, "cudaMemcpyToSymbol: SetKernelsGpuData copy to cData failed");
}

/**
 * @brief Copies the GPU data from the constant symbol cData.
 */
void GetKernelsGpuData()
{
    cudaError_t status;
    status = cudaMemcpyFromSymbol(&(getGpu()._data), cData, sizeof(GpuData));
    RTERROR(status, "cudaMemcpyFromSymbol: GetKernelsGpuData copy from cData failed");
}

/**
 * @brief Calculates the number of blocks required for the given size.
 *
 * @param size The total number of elements.
 * @return The number of blocks needed to process the elements.
 */
uint32_t CalculateBlocks(uint64_t size)
{
    return (size + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;
}

/**
 * @brief CUDA kernel to scale and bias an array in parallel.
 *
 * @param pData Pointer to the data array.
 * @param size Size of the data array.
 * @param scale Scale factor to apply to each element.
 * @param bias Bias value to subtract from each element.
 */
__global__ void kScaleAndBias_kernel(NNFloat* pData, uint64_t size, NNFloat scale, NNFloat bias)
{
    uint64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset < size)
    {
        NNFloat value = pData[offset];
        pData[offset] = scale * value - bias;
    }
}

/**
 * @brief Applies scale and bias to an array using CUDA.
 *
 * @param pData Pointer to the data array.
 * @param size Size of the data array.
 * @param scale Scale factor to apply to each element.
 * @param bias Bias value to subtract from each element.
 */
void kScaleAndBias(NNFloat* pData, uint64_t size, NNFloat scale, NNFloat bias)
{
    uint32_t threadsPerBlock = 256; // Choose an appropriate value based on your GPU architecture
    uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    kScaleAndBias_kernel<<<blocks, threadsPerBlock>>>(pData, size, scale, bias);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

/**
 * @brief CUDA kernel to clear a unit by assigning bias values in parallel.
 *
 * @param pUnit Pointer to the unit array.
 * @param pBias Pointer to the bias array.
 * @param stride Stride of the unit array.
 * @param size Size of the unit array.
 */
__global__ void kClearUnit_kernel(NNFloat* pUnit, NNFloat* pBias, uint32_t stride, uint64_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bpos = pos % stride;
    if (pos < size)
    {
        pUnit[pos] = pBias[bpos];
    }
}

/**
 * @brief Clears a unit using CUDA.
 *
 * @param pUnit Pointer to the unit array.
 * @param pBias Pointer to the bias array.
 * @param stride Stride of the unit array.
 * @param batch Batch size.
 */
void kClearUnit(NNFloat* pUnit, NNFloat* pBias, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    uint32_t threadsPerBlock = 256; // Choose an appropriate value based on your GPU architecture
    uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    kClearUnit_kernel<<<blocks, threadsPerBlock>>>(pUnit, pBias, stride, size);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        // Handle or report the CUDA kernel launch error
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

/**
 * @brief CUDA kernel to clear a dual source unit by summing bias values in parallel.
 *
 * This kernel function sets the values of a dual source unit by summing the corresponding bias values from two different bias arrays.
 *
 * @param pUnit Pointer to the unit array.
 * @param pBias1 Pointer to the first bias array.
 * @param pBias2 Pointer to the second bias array.
 * @param stride Stride of the unit array.
 * @param size Size of the unit array.
 */
__global__ void kClearDualSourceUnit_kernel(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, uint32_t stride, uint32_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bpos = pos % stride;

    if (pos < size)
    {
        pUnit[pos] = pBias1[bpos] + pBias2[bpos];
    }
}

/**
 * @brief Clears a dual source unit by summing bias values in parallel.
 *
 * This function clears a dual source unit by summing the bias values from two different bias arrays in parallel using CUDA.
 *
 * @param pUnit Pointer to the unit array.
 * @param pBias1 Pointer to the first bias array.
 * @param pBias2 Pointer to the second bias array.
 * @param stride Stride of the unit array.
 * @param batch Batch size.
 */
void kClearDualSourceUnit(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);

    uint32_t threadsPerBlock = 256;

    uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    kClearDualSourceUnit_kernel<<<blocks, threadsPerBlock>>>(pUnit, pBias1, pBias2, stride, size);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

/**
 * @brief CUDA kernel to compute the triple source unit values.
 *
 * @param pUnit    Pointer to the output unit array
 * @param pBias1   Pointer to the first bias array
 * @param pBias2   Pointer to the second bias array
 * @param pBias3   Pointer to the third bias array
 * @param stride   Stride of the bias arrays
 * @param size     Size of the output unit array
 */
__global__ void kClearTripleSourceUnit_kernel(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, NNFloat* pBias3, uint32_t stride, uint32_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        uint32_t bpos = pos % stride;
        pUnit[pos] = pBias1[bpos] + pBias2[bpos] + pBias3[pos];
    }
}

/**
 * @brief Compute the triple source unit values using CUDA.
 *
 * @param pUnit    Pointer to the output unit array
 * @param pBias1   Pointer to the first bias array
 * @param pBias2   Pointer to the second bias array
 * @param pBias3   Pointer to the third bias array
 * @param stride   Stride of the bias arrays
 * @param batch    Batch size
 */
void kClearTripleSourceUnit(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, NNFloat* pBias3, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    kClearTripleSourceUnit_kernel<<<blocks, threadsPerBlock>>>(pUnit, pBias1, pBias2, pBias3, stride, size);
    LAUNCHERROR("kClearTripleSource_kernel");
}

/**
 * @brief CUDA kernel to compute the quad source unit values.
 *
 * @param pUnit    Pointer to the output unit array
 * @param pBias1   Pointer to the first bias array
 * @param pBias2   Pointer to the second bias array
 * @param pBias3   Pointer to the third bias array
 * @param pBias4   Pointer to the fourth bias array
 * @param stride   Stride of the bias arrays
 * @param size     Size of the output unit array
 */
__global__ void kClearQuadSourceUnit_kernel(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, NNFloat* pBias3, NNFloat* pBias4, uint32_t stride, uint32_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        uint32_t bpos = pos % stride;
        pUnit[pos] = pBias1[bpos] + pBias2[bpos] + pBias3[pos] + pBias4[pos];
    }
}

/**
 * @brief Compute the quad source unit values using CUDA.
 *
 * @param pUnit    Pointer to the output unit array
 * @param pBias1   Pointer to the first bias array
 * @param pBias2   Pointer to the second bias array
 * @param pBias3   Pointer to the third bias array
 * @param pBias4   Pointer to the fourth bias array
 * @param stride   Stride of the bias arrays
 * @param batch    Batch size
 */
void kClearQuadSourceUnit(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, NNFloat* pBias3, NNFloat* pBias4, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    kClearQuadSourceUnit_kernel<<<blocks, threadsPerBlock>>>(pUnit, pBias1, pBias2, pBias3, pBias4, stride, size);
    LAUNCHERROR("kClearQuadSource_kernel");
}

/**
 * @brief CUDA kernel to load sparse input units into a dense unit matrix.
 *
 * @param position      The starting position of the batch.
 * @param batch         The number of batches to process.
 * @param stride        The stride of the unit matrix.
 * @param pUnit         Pointer to the unit matrix.
 * @param pSparseStart  Pointer to the array containing the start positions of sparse data.
 * @param pSparseEnd    Pointer to the array containing the end positions of sparse data.
 * @param pSparseIndex  Pointer to the array containing the sparse indices.
 * @param pDataWeight   Pointer to the array containing the data weights (optional).
 */
__global__ void kLoadSparseInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < batch)
    {
        uint32_t pos1 = pos + position;
        pos1 = cData._bShuffleIndices ? cData._pShuffleIndex[pos1] : pos1;
        uint64_t start = pSparseStart[pos1];
        uint64_t end = pSparseEnd[pos1];

        __shared__ NNFloat weight;
        if (threadIdx.x == 0)
        {
            weight = (pDataWeight != NULL) ? pDataWeight[pos1] : (NNFloat)1.0;
        }
        __syncthreads();

        uint64_t offset = pos * stride;

        for (uint64_t i = threadIdx.x; i < (end - start); i += blockDim.x)
        {
            uint64_t pos2 = offset + pSparseIndex[start + i];
            pUnit[pos2] = weight;
        }
    }
}

/**
 * @brief Load sparse input units into a dense unit matrix.
 *
 * @param position      The starting position of the batch.
 * @param batch         The number of batches to process.
 * @param stride        The stride of the unit matrix.
 * @param pUnit         Pointer to the unit matrix.
 * @param pSparseStart  Pointer to the array containing the start positions of sparse data.
 * @param pSparseEnd    Pointer to the array containing the end positions of sparse data.
 * @param pSparseIndex  Pointer to the array containing the sparse indices.
 * @param pDataWeight   Pointer to the array containing the data weights (optional).
 */
void kLoadSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight)
{
    uint32_t last = position + batch;
    uint32_t count = last - position;
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (count + threadsPerBlock - 1) / threadsPerBlock;

    cudaError_t status = cudaMemset(pUnit, 0, static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride) * sizeof(NNFloat));
    RTERROR(status, "kLoadSparseInputUnit failed");
    kLoadSparseInputUnit_kernel<<<blocks, threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight);
    LAUNCHERROR("kLoadSparseInputUnit_kernel");
}

/**
 * @brief CUDA kernel for loading indexed sparse input units.
 *
 * @param position The starting position.
 * @param batch The number of batches.
 * @param stride The stride of the input.
 * @param pUnit Pointer to the input units.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the start positions of sparse data.
 * @param pSparseEnd Pointer to the end positions of sparse data.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 */
__global__ void kLoadIndexedSparseInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t warpId = tid / cData._warpSize;
    
    if (warpId < batch)
    {
        uint32_t pos = position + warpId;
        pos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[pos] : pos];
        
        uint64_t start = pSparseStart[pos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[pos];
        NNFloat w = (pDataWeight != NULL) ? pDataWeight[pos] : (NNFloat)1.0;
        uint64_t offset = warpId * stride;
        
        while (start < end)
        {
            uint64_t pos2 = offset + pSparseIndex[start];
            pUnit[pos2] = w;
            start += cData._warpSize;
        }
    }
}

/**
 * @brief Function to load indexed sparse input units.
 *
 * @param position The starting position.
 * @param batch The number of batches.
 * @param stride The stride of the input.
 * @param pUnit Pointer to the input units.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the start positions of sparse data.
 * @param pSparseEnd Pointer to the end positions of sparse data.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 */
void kLoadIndexedSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight)
{
    uint32_t last = position + batch;
    uint32_t count = last - position;
    uint32_t blocks = (count + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;

    cudaError_t status = cudaMemset(pUnit, 0, (uint64_t)batch * (uint64_t)stride * sizeof(NNFloat));
    RTERROR(status, "kLoadIndexedSparseInputUnit failed");
    
    kLoadIndexedSparseInputUnit_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight);
    LAUNCHERROR("kLoadIndexedSparseInputUnit_kernel");
}

/**
 * \brief CUDA kernel for loading sparse analog input units.
 *
 * \tparam T Data type for sparse data.
 *
 * \param position The starting position.
 * \param batch The batch size.
 * \param stride The stride.
 * \param pUnit Pointer to the input unit.
 * \param pSparseStart Pointer to the sparse start array.
 * \param pSparseEnd Pointer to the sparse end array.
 * \param pSparseIndex Pointer to the sparse index array.
 * \param pDataWeight Pointer to the data weight array.
 * \param pSparseData Pointer to the sparse data array.
 */
template<typename T>
__global__ void LAUNCH_BOUNDS()
kLoadSparseAnalogInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, T* pSparseData)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < batch)
    {
        uint32_t pos1 = pos + position;
        pos1 = cData._bShuffleIndices ? cData._pShuffleIndex[pos1] : pos1;
        uint64_t start = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[pos1];
        NNFloat w = (pDataWeight != NULL) ? pDataWeight[pos1] : (NNFloat)1.0;
        uint64_t offset = pos * stride;

        for (uint64_t i = start; i < end; i += cData._warpSize)
        {
            uint64_t pos2 = offset + pSparseIndex[i];
            T data = pSparseData[i];
            pUnit[pos2] = w * data;
        }
    }
}

/**
 * @brief Load sparse analog input unit data into GPU memory.
 *
 * @tparam T Type of the sparse data.
 * @param position Starting position of the data.
 * @param batch Number of elements to load.
 * @param stride Stride between elements.
 * @param pUnit Pointer to the GPU memory for storing the loaded data.
 * @param pSparseStart Pointer to the start indices of the sparse data.
 * @param pSparseEnd Pointer to the end indices of the sparse data.
 * @param pSparseIndex Pointer to the indices of the sparse data.
 * @param pDataWeight Pointer to the data weights of the sparse data.
 * @param pSparseData Pointer to the sparse data.
 */
template<typename T>
void kLoadSparseAnalogInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, T* pSparseData)
{
    uint32_t last = position + batch;
    uint32_t count = last - position;
    uint32_t blocks = (count * getGpu()._warpSize + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;

    cudaError_t status = cudaMemcpy(pUnit, &NNFloat(0), (uint64_t)batch * (uint64_t)stride * sizeof(NNFloat), cudaMemcpyHostToDevice);
    RTERROR(status, "kLoadSparseAnalogInputUnit failed");

    kLoadSparseAnalogInputUnit_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
    LAUNCHERROR("kLoadSparseAnalogInputUnit_kernel");
}

/**
 * @brief CUDA kernel for loading indexed sparse analog input units.
 *
 * @tparam T The type of the sparse data.
 * @param position The starting position.
 * @param batch The number of batches.
 * @param stride The stride value.
 * @param pUnit Pointer to the output unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 */
template<typename T>
__global__ void LAUNCH_BOUNDS()
kLoadIndexedSparseAnalogInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* __restrict__ pUnit, uint32_t* __restrict__ pIndex, uint64_t* __restrict__ pSparseStart, uint64_t* __restrict__ pSparseEnd, uint32_t* __restrict__ pSparseIndex, NNFloat* __restrict__ pDataWeight, T* __restrict__ pSparseData)
{
    constexpr uint32_t warpSize = cData._warpSize;
    constexpr uint32_t warpMask = cData._warpMask;

    uint32_t pos = threadIdx.x / warpSize;
    if (pos < batch)
    {
        uint32_t pos1 = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[pos + position] : pos + position];
        uint64_t start = pSparseStart[pos1] + (threadIdx.x & warpMask);
        uint64_t end = pSparseEnd[pos1];
        NNFloat w = (pDataWeight != nullptr) ? pDataWeight[pos1] : (NNFloat)1.0;
        uint64_t offset = pos * stride;

        #pragma unroll
        for (uint64_t i = start; i < end; i += warpSize)
        {
            uint64_t pos2 = offset + pSparseIndex[i];
            T data = pSparseData[i];
            pUnit[pos2] = w * data;
        }
    }
}

/**
 * @brief Load sparse analog input units using index-based lookup.
 *
 * This function loads sparse analog input units from the given sparse data and index arrays
 * into the output array. It sets the memory of the output array to zero before loading the values.
 *
 * @tparam T The data type of the sparse data.
 * @param position The starting position of the sparse input units to load.
 * @param batch The number of sparse input units to load.
 * @param stride The stride between consecutive elements in the output array.
 * @param pUnit Pointer to the output array for the sparse input units.
 * @param pIndex Pointer to the index array for the sparse input units.
 * @param pSparseStart Pointer to the array of starting indices for the sparse data.
 * @param pSparseEnd Pointer to the array of ending indices for the sparse data.
 * @param pSparseIndex Pointer to the array of indices for the sparse data.
 * @param pDataWeight Pointer to the array of data weights for the sparse data.
 * @param pSparseData Pointer to the sparse data array.
 */
template<typename T>
void kLoadIndexedSparseAnalogInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, T* pSparseData)
{
    uint32_t last = position + batch;
    uint32_t count = last - position;
    uint32_t blocks = (count * getGpu()._warpSize + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;

    cudaError_t status = cudaMemset(pUnit, 0, static_cast<size_t>(batch) * static_cast<size_t>(stride) * sizeof(NNFloat));
    RTERROR(status, "cudaMemset failed");

    kLoadIndexedSparseAnalogInputUnit_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
    LAUNCHERROR("kLoadIndexedSparseAnalogInputUnit_kernel");

    status = cudaGetLastError();
    RTERROR(status, "Kernel execution failed");
}
