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

/**
 * @brief CUDA kernel for loading sparse denoised input units.
 *
 * @param position    The starting position in the batch.
 * @param batch       The number of elements to process in the batch.
 * @param stride      The stride value.
 * @param pUnit       Pointer to the output units.
 * @param pSparseStart Pointer to the start indices of sparse data for each position.
 * @param pSparseEnd   Pointer to the end indices of sparse data for each position.
 * @param pSparseIndex Pointer to the indices of sparse data.
 * @param pDataWeight  Pointer to the weight data.
 * @param pRandom      Pointer to the random data.
 */
__global__ void kLoadSparseDenoisedInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride,
                                                    NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd,
                                                    uint32_t* pSparseIndex, NNFloat* pDataWeight, NNFloat* pRandom)
{
    uint32_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {                           
        uint32_t pos1 = cData._bShuffleIndices ? cData._pShuffleIndex[pos + position] : pos + position;
        uint64_t start = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[pos1];
        NNFloat w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[pos1] : (NNFloat)1.0);
        uint64_t offset = pos * stride;

        __shared__ NNFloat sDataWeight[BLOCK_SIZE];
        if (pDataWeight != NULL && threadIdx.x < cData._warpSize)
            sDataWeight[threadIdx.x] = pDataWeight[pos1];
        
        uint64_t loopEnd = end - cData._warpSize;

        #pragma unroll
        for (; start < loopEnd; start += cData._warpSize)
        {
            NNFloat value = pRandom[start];
            uint64_t pos2 = offset + pSparseIndex[start];

            NNFloat mask = __ballot_sync(FULL_MASK, value >= cData._denoising_p);
            if (threadIdx.x % cData._warpSize == 0)
                pUnit[pos2] = w * (__popc(mask) == cData._warpSize ? 1.0f : 0.0f);
        }

        if (start < end)
        {
            NNFloat value = pRandom[start];
            uint64_t pos2 = offset + pSparseIndex[start];
            NNFloat mask = __ballot_sync(FULL_MASK, value >= cData._denoising_p);
            if (threadIdx.x % cData._warpSize == 0 && (start < loopEnd || threadIdx.x < (end - start)))
                pUnit[pos2] = w * (__popc(mask) == cData._warpSize ? 1.0f : 0.0f);
        }
    }
}

/**
 * @brief Loads sparse denoised input units into a CUDA device memory.
 *
 * This function sets the specified memory region to zero and then launches a CUDA kernel
 * to load sparse denoised input units into the memory.
 *
 * @param position       Starting position of the input units.
 * @param batch          Number of input units to load.
 * @param stride         Stride of the input units.
 * @param pUnit          Pointer to the device memory for storing the input units.
 * @param pSparseStart   Pointer to the start indices of the sparse input units.
 * @param pSparseEnd     Pointer to the end indices of the sparse input units.
 * @param pSparseIndex   Pointer to the indices of the sparse input units.
 * @param pDataWeight    Pointer to the data weights of the sparse input units.
 * @param pRandom        Pointer to random values for sparse input generation.
 */
void kLoadSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pDataWeight, NNFloat* pRandom)
{
    uint32_t last = position + batch;
    uint32_t count = last - position;
    uint32_t blocks = (count * getGpu()._warpSize + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;

    size_t unitSize = (uint64_t)batch * (uint64_t)stride * sizeof(NNFloat);

    cudaError_t status = cudaMemsetAsync(pUnit, 0, unitSize);
    RTERROR(status, "cudaMemsetAsync failed");

    kLoadSparseDenoisedInputUnit_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pRandom);
    CUDA_CHECK_ERROR("kLoadSparseDenoisedInputUnit_kernel launch failed");

    cudaDeviceSynchronize();
}

/**
 * @brief Kernel for loading indexed sparse denoised input units.
 *
 * @param position       Starting position in the input data.
 * @param batch          Number of input samples to process.
 * @param stride         Stride between consecutive input samples.
 * @param pUnit          Pointer to the output unit array.
 * @param pIndex         Pointer to the index array.
 * @param pSparseStart   Pointer to the start indices of sparse data.
 * @param pSparseEnd     Pointer to the end indices of sparse data.
 * @param pSparseIndex   Pointer to the indices of sparse data.
 * @param pDataWeight    Pointer to the weight array.
 * @param pRandom        Pointer to the random values array.
 */
__global__ void LAUNCH_BOUNDS()
kLoadIndexedSparseDenoisedInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride,
                                           NNFloat* pUnit, uint32_t* pIndex, uint64_t* pSparseStart,
                                           uint64_t* pSparseEnd, uint32_t* pSparseIndex,
                                           NNFloat* pDataWeight, NNFloat* pRandom)
{
    const uint32_t warpIndex = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    if (warpIndex < batch)
    {
        const uint32_t globalIndex = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[warpIndex + position] : warpIndex + position];
        const uint64_t warpStart = pSparseStart[globalIndex] + (threadIdx.x & cData._warpMask);
        const uint64_t warpEnd = pSparseEnd[globalIndex];
        const NNFloat weight = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[globalIndex] : (NNFloat)1.0);
        const uint64_t offset = warpIndex * stride;

        for (uint64_t i = warpStart; i < warpEnd; i += cData._warpSize)
        {
            const NNFloat value = pRandom[i];
            const uint64_t unitIndex = offset + pSparseIndex[i];

            if (value >= cData._denoising_p)
            {
                pUnit[unitIndex] = weight;
            }
        }
    }
}

/**
 * @brief Loads denoised input units from indexed sparse data into a CUDA memory buffer.
 *
 * @param position The starting position of the input units.
 * @param batch The number of input units to load.
 * @param stride The stride of the input units.
 * @param pUnit Pointer to the CUDA memory buffer to store the loaded units.
 * @param pIndex Pointer to the index array containing the indices of the input units.
 * @param pSparseStart Pointer to the array of starting positions for each index in the sparse data.
 * @param pSparseEnd Pointer to the array of ending positions for each index in the sparse data.
 * @param pSparseIndex Pointer to the array of indices for each data entry in the sparse data.
 * @param pDataWeight Pointer to the weight values associated with each data entry in the sparse data.
 * @param pRandom Pointer to the array of random values.
 */
void kLoadIndexedSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pDataWeight, NNFloat* pRandom)
{
    uint32_t last = position + batch;
    uint32_t count = last - position;
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (count * getGpu()._warpSize + threadsPerBlock - 1) / threadsPerBlock;

    cudaError_t status = cudaMemset(pUnit, 0, static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride) * sizeof(NNFloat));
    RTERROR(status, "kLoadIndexedSparseDenoisedInputUnit failed");
    kLoadIndexedSparseDenoisedInputUnit_kernel<<<blocks, threadsPerBlock>>>(position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pRandom);
    LAUNCHERROR("kLoadIndexedSparseDenoisedInputUnit_kernel");
}

/**
 * @brief Kernel function for loading sparse analog denoised input units.
 *
 * @tparam T The type of sparse data.
 * @param position The position parameter.
 * @param batch The batch parameter.
 * @param stride The stride parameter.
 * @param pUnit Pointer to the unit data.
 * @param pSparseStart Pointer to the sparse start data.
 * @param pSparseEnd Pointer to the sparse end data.
 * @param pSparseIndex Pointer to the sparse index data.
 * @param pDataWeight Pointer to the data weight.
 * @param pSparseData Pointer to the sparse data.
 * @param pRandom Pointer to the random data.
 */
template<typename T>
__global__ void LAUNCH_BOUNDS()
kLoadSparseAnalogDenoisedInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, T* pSparseData, NNFloat* pRandom)
{
    uint32_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {                           
        uint32_t pos1 = cData._bShuffleIndices ? cData._pShuffleIndex[pos + position] : pos + position;
        uint64_t start = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[pos1];
        NNFloat w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[pos1] : (NNFloat)1.0);
        uint64_t offset = pos * stride;

        uint32_t* pIndex = pSparseIndex + start;
        T* pData = pSparseData + start;

        __shared__ NNFloat randomShared[cData._warpSize];
        if (threadIdx.x < cData._warpSize)
            randomShared[threadIdx.x] = pRandom[start];

        for (; start < end; start += cData._warpSize)
        {
            NNFloat value = randomShared[threadIdx.x & cData._warpMask];
            uint64_t pos2 = offset + pIndex[threadIdx.x];
            T data = pData[threadIdx.x];
            if (value >= cData._denoising_p)
                pUnit[pos2] = w * data;
        }
    }
}

/**
 * @brief Loads sparse analog denoised input unit.
 *
 * This function loads the sparse analog denoised input unit using CUDA.
 * It sets the memory to zero using CUDA memsetAsync for asynchronous memory initialization,
 * and then launches the kernel to load the sparse data into the input unit.
 *
 * @tparam T         The type of the sparse data.
 * @param position   The starting position in the input unit.
 * @param batch      The batch size.
 * @param stride     The stride of the input unit.
 * @param pUnit      Pointer to the input unit.
 * @param pSparseStart Pointer to the start indices of the sparse data.
 * @param pSparseEnd Pointer to the end indices of the sparse data.
 * @param pSparseIndex Pointer to the indices of the sparse data.
 * @param pDataWeight Pointer to the weights of the sparse data.
 * @param pSparseData Pointer to the sparse data.
 * @param pRandom    Pointer to the random data.
 */
template<typename T>
void kLoadSparseAnalogDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, NNFloat* pDataWeight, T *pSparseData, NNFloat* pRandom)
{
    uint32_t last = position + batch;
    uint32_t count = last - position;
    uint32_t blocks = (count * getGpu()._warpSize + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaError_t status = cudaMemsetAsync(pUnit, 0, (uint64_t)batch * (uint64_t)stride * sizeof(NNFloat), stream);
    RTERROR(status, "kLoadSparseAnalogDenoisedInputUnit failed");

    kLoadSparseAnalogDenoisedInputUnit_kernel<<<blocks, getGpu()._threadsPerBlock, 0, stream>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pRandom);
    LAUNCHERROR("kLoadSparseAnalogDenoisedInputUnit_kernel");

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

/**
 * @brief Kernel function for loading indexed sparse analog denoised input units.
 *
 * @tparam T The data type for the sparse data.
 * @param position The starting position of the batch.
 * @param batch The number of elements in the batch.
 * @param stride The stride between elements.
 * @param pUnit Pointer to the output unit data.
 * @param pIndex Pointer to the index data.
 * @param pSparseStart Pointer to the start indices of sparse data.
 * @param pSparseEnd Pointer to the end indices of sparse data.
 * @param pSparseIndex Pointer to the indices of sparse data.
 * @param pDataWeight Pointer to the weight data.
 * @param pSparseData Pointer to the sparse data.
 * @param pRandom Pointer to the random data.
 */
template<typename T>
__global__ void LAUNCH_BOUNDS(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, const uint64_t* pSparseStart, const uint64_t* pSparseEnd, const uint32_t* pSparseIndex, const NNFloat* pDataWeight, const T* pSparseData, const NNFloat* pRandom)
{
    uint32_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {                           
        uint32_t pos1 = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[pos + position] : pos + position];
        uint64_t start = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[pos1];
        NNFloat w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[pos1] : (NNFloat)1.0);
        uint64_t offset = pos * stride;

        __shared__ NNFloat sDataWeight[BLOCK_SIZE];
        if (threadIdx.x < BLOCK_SIZE)
            sDataWeight[threadIdx.x] = (pDataWeight != NULL) ? pDataWeight[pos1] : (NNFloat)1.0;
        __syncthreads();

        for (uint64_t i = start; i < end; i += cData._warpSize)
        {
            NNFloat values[WARP_SIZE];
            T data[WARP_SIZE];
            uint64_t pos2[WARP_SIZE];

            #pragma unroll
            for (int j = 0; j < WARP_SIZE; ++j)
            {
                if (i + j < end)
                {
                    values[j] = pRandom[i + j];
                    pos2[j] = offset + pSparseIndex[i + j];
                    data[j] = pSparseData[i + j];
                }
            }

            #pragma unroll
            for (int j = 0; j < WARP_SIZE; ++j)
            {
                if (i + j < end)
                {
                    NNFloat value = values[j];
                    uint64_t pos2_1 = pos2[j];
                    T data1 = data[j];
                    NNFloat output1 = w * data1;

                    if (value < cData._denoising_p)
                        output1 = 0.0;

                    pUnit[pos2_1] = output1;
                }
            }
        }
    }
}

/**
 * @brief Loads indexed sparse analog denoised input units into a CUDA device memory array.
 *
 * This function initializes the device memory array `pUnit` with indexed sparse analog denoised input units
 * based on the provided parameters. It performs memory initialization asynchronously using `cudaMemsetAsync`.
 * Then, it copies the necessary data from the host to the device memory using `cudaMemcpyAsync`.
 * Finally, it launches the kernel `kLoadIndexedSparseAnalogDenoisedInputUnit_kernel` to compute the input units.
 *
 * @tparam T The type of the sparse data.
 * @param position The starting position of the input units.
 * @param batch The number of input units to process.
 * @param stride The stride of the input units.
 * @param pUnit Pointer to the device memory array for storing the input units.
 * @param pIndex Pointer to the device memory array containing the indices.
 * @param pSparseStart Pointer to the device memory array containing the sparse start indices.
 * @param pSparseEnd Pointer to the device memory array containing the sparse end indices.
 * @param pSparseIndex Pointer to the device memory array containing the sparse indices.
 * @param pDataWeight Pointer to the device memory array containing the data weights.
 * @param pSparseData Pointer to the device memory array containing the sparse data.
 * @param pRandom Pointer to the device memory array containing random values.
 */
template<typename T>
void kLoadIndexedSparseAnalogDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, T* pSparseData, NNFloat* pRandom)
{
    uint32_t last = position + batch;
    uint32_t count = last - position;
    uint32_t blocks = (count * getGpu()._warpSize + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;

    cudaMemsetAsync(pUnit, 0, batch * stride * sizeof(NNFloat));
    cudaError_t status = cudaGetLastError();
    RTERROR(status, "cudaMemsetAsync failed");

    cudaStream_t memoryStream;
    cudaStreamCreate(&memoryStream);

    cudaMemcpyAsync(pUnit, pUnit, batch * stride * sizeof(NNFloat), cudaMemcpyDeviceToDevice, memoryStream);
    cudaMemcpyAsync(pIndex, pIndex, batch * sizeof(uint32_t), cudaMemcpyDeviceToDevice, memoryStream);
    cudaMemcpyAsync(pSparseStart, pSparseStart, batch * sizeof(uint64_t), cudaMemcpyDeviceToDevice, memoryStream);
    cudaMemcpyAsync(pSparseEnd, pSparseEnd, batch * sizeof(uint64_t), cudaMemcpyDeviceToDevice, memoryStream);
    cudaMemcpyAsync(pSparseIndex, pSparseIndex, batch * stride * sizeof(uint32_t), cudaMemcpyDeviceToDevice, memoryStream);
    cudaMemcpyAsync(pDataWeight, pDataWeight, batch * stride * sizeof(NNFloat), cudaMemcpyDeviceToDevice, memoryStream);
    cudaMemcpyAsync(pSparseData, pSparseData, batch * stride * sizeof(T), cudaMemcpyDeviceToDevice, memoryStream);
    cudaMemcpyAsync(pRandom, pRandom, batch * stride * sizeof(NNFloat), cudaMemcpyDeviceToDevice, memoryStream);

    kLoadIndexedSparseAnalogDenoisedInputUnit_kernel<<<blocks, getGpu()._threadsPerBlock, 0, memoryStream>>>(position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pRandom);
    cudaError_t kernelLaunchStatus = cudaGetLastError();
    RTERROR(kernelLaunchStatus, "Kernel launch error: kLoadIndexedSparseAnalogDenoisedInputUnit_kernel");

    cudaStreamDestroy(memoryStream);
    cudaDeviceSynchronize();
}

/**
 * @brief Loads input units from input data into the specified output array.
 *
 * @tparam T The data type of the input data.
 * @param position The position within the input data.
 * @param stride The stride value.
 * @param pUnit The output array for storing the input units.
 * @param pInputData The input data array.
 */
template<typename T>
__global__ void kLoadInputUnit_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, T* pInputData)
{
    uint32_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint32_t pos1 = cData._bShuffleIndices ? cData._pShuffleIndex[blockIdx.x + position] : (blockIdx.x + position);
        uint32_t soffset = pos1 * stride + pos;
        uint32_t doffset = blockIdx.x * stride + pos;
        pUnit[doffset] = pInputData[soffset];
    }
}

/**
 * @brief CUDA kernel to load and normalize input units.
 *
 * This kernel loads input units from device memory and applies normalization
 * by scaling the data and subtracting a constant value.
 *
 * @param position The starting position of the input units.
 * @param stride The stride between input units.
 * @param pUnit Pointer to the output array of normalized units.
 * @param pData Pointer to the input data array.
 */
__global__ void kLoadNormalizedInputUnit_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, unsigned char* pData)
{
    uint32_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;

    if (pos < stride)
    {
        uint32_t pos1 = cData._bShuffleIndices ? cData._pShuffleIndex[blockIdx.x + position] : blockIdx.x + position;
        uint32_t soffset = pos1 * stride + pos;
        uint32_t doffset = blockIdx.x * stride + pos;
        pUnit[doffset] = static_cast<NNFloat>(pData[soffset]) * static_cast<NNFloat>(1.0 / 256.0) - static_cast<NNFloat>(0.5);
    }
}

/**
 * @brief CUDA kernel to load normalized input units.
 *
 * @param position The position of the input unit.
 * @param stride The stride between input units.
 * @param pUnit Pointer to the output unit array.
 * @param pData Pointer to the input data array.
 */
__global__ void kLoadNormalizedInputUnit_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, char* pData)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {

        uint32_t pos1 = cData._bShuffleIndices ?  cData._pShuffleIndex[blockIdx.x + position] : blockIdx.x + position;
        uint64_t soffset = pos1 * stride + pos;
        uint64_t doffset = blockIdx.x * stride + pos;

        char dataValue = pData[soffset];
        pUnit[doffset] = static_cast<NNFloat>(dataValue) * (1.0f / 128.0f);
    }
}

/**
 * @brief Loads input unit data from the given pData array into the pUnit array.
 *
 * @tparam T The data type of the pData array.
 * @param position The position of the input unit.
 * @param batch The batch size.
 * @param stride The stride of the data.
 * @param pUnit Pointer to the output unit array.
 * @param pData Pointer to the input data array.
 */
template <typename T>
void kLoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, T* pData)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kLoadInputUnit_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pData);
    cudaDeviceSynchronize(); // Wait for the kernel to finish execution
    checkCudaErrors(cudaGetLastError()); // Check for any errors during kernel execution
}

/**
 * @brief Specialization of kLoadInputUnit for unsigned char data type.
 *
 * @param position The position of the input unit.
 * @param batch The batch size.
 * @param stride The stride of the data.
 * @param pUnit Pointer to the output unit array.
 * @param pData Pointer to the input data array.
 */
template <>
void kLoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, unsigned char* pData)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kLoadNormalizedInputUnit_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pData);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

/**
 * @brief Specialization of kLoadInputUnit for char data type.
 *
 * @param position The position of the input unit.
 * @param batch The batch size.
 * @param stride The stride of the data.
 * @param pUnit Pointer to the output unit array.
 * @param pData Pointer to the input data array.
 */
template <>
void kLoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, char* pData)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kLoadNormalizedInputUnit_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pData);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

/**
 * @brief CUDA kernel to load indexed input units into the output unit array.
 *
 * @param position The starting position of the input units.
 * @param stride The stride value.
 * @param pUnit Pointer to the output unit array.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the input data array.
 */
template<typename T>
__global__ void kLoadIndexedInputUnit_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, T* pData)
{
    size_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        size_t blockIdxPos = blockIdx.x + position;
        uint32_t pos1 = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[blockIdxPos] : blockIdxPos];
        size_t soffset = pos1 * stride + pos;
        size_t doffset = blockIdx.x * stride + pos;
        pUnit[doffset] = pData[soffset];
    }
}

/**
 * @brief CUDA kernel to load indexed and normalized input units into the neural network.
 *
 * @param position Starting position in the input data.
 * @param stride Stride size for accessing the input units.
 * @param pUnit Pointer to the output unit array.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the input data array.
 */
__global__ void kLoadIndexedNormalizedInputUnit_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, unsigned char* pData)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint32_t pos1 = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[blockIdx.x + position] : blockIdx.x + position];
        uint64_t soffset = pos1 * stride + pos;
        uint64_t doffset = blockIdx.x * stride + pos;
        NNFloat dataValue = static_cast<NNFloat>(pData[soffset]) * static_cast<NNFloat>(1.0 / 256.0) - static_cast<NNFloat>(0.5);
        pUnit[doffset] = dataValue;
    }
}

/**
 * @brief CUDA kernel to load indexed and normalized input units into the neural network.
 *
 * @param position Starting position in the input data.
 * @param stride Stride size for accessing the input units.
 * @param pUnit Pointer to the output unit array.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the input data array.
 */
__global__ void kLoadIndexedNormalizedInputUnit_kernel(uint32_t position, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, char* pData)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint32_t pos1 = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[blockIdx.x + position] : blockIdx.x + position];
        uint64_t soffset = pos1 * stride + pos;
        uint64_t doffset = blockIdx.x * stride + pos;
        NNFloat dataValue = static_cast<NNFloat>(static_cast<unsigned char>(pData[soffset])) * static_cast<NNFloat>(1.0 / 128.0);
        pUnit[doffset] = dataValue;
    }
}

/**
 * @brief Loads indexed input units into the neural network.
 *
 * @tparam T Type of the input data.
 * @param position Starting position in the input data.
 * @param batch Number of batches.
 * @param stride Stride size for accessing the input units.
 * @param pUnit Pointer to the output unit array.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the input data array.
 */
template<typename T>
void kLoadIndexedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, T* pData)
{
    const uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    const dim3 grid(batch, (stride + threadsPerBlock - 1) / threadsPerBlock);
    kLoadIndexedInputUnit_kernel<<<grid, threadsPerBlock>>>(position, stride, pUnit, pIndex, pData);
    LAUNCHERROR("kLoadIndexedInputUnit_kernel");
}

/**
 * @brief Loads indexed input units from unsigned char data into NNFloat array.
 *
 * @param position The position of the input unit.
 * @param batch The number of input units to load.
 * @param stride The stride between consecutive input units.
 * @param pUnit Pointer to the NNFloat array to store the loaded input units.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the unsigned char data.
 */
template<> void kLoadIndexedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, unsigned char* pData)
{
    uint32_t numBlocks = (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;
    dim3 grid(batch, numBlocks);

    kLoadIndexedNormalizedInputUnit_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pIndex, pData);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    LAUNCHERROR("kLoadIndexedNormalizedInputUnit_kernel");
}

/**
 * @brief Loads indexed input units from char data into NNFloat array.
 *
 * @param position The position of the input unit.
 * @param batch The number of input units to load.
 * @param stride The stride between consecutive input units.
 * @param pUnit Pointer to the NNFloat array to store the loaded input units.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the char data.
 */
template<> void kLoadIndexedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, uint32_t* pIndex, char* pData)
{
    uint32_t numBlocks = (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;
    dim3 grid(batch, numBlocks);

    kLoadIndexedNormalizedInputUnit_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pIndex, pData);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    LAUNCHERROR("kLoadIndexedNormalizedInputUnit_kernel");
}

/**
 * @brief CUDA kernel to add bias values to a unit array.
 *
 * @param pUnit Pointer to the unit array.
 * @param pBias Pointer to the bias array.
 * @param stride Stride value.
 * @param size Total number of elements in the unit array.
 */
__global__ void kAddBias_kernel(NNFloat* pUnit, NNFloat* pBias, uint32_t stride, uint32_t size)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        uint32_t bpos = pos % stride;
        pUnit[pos] += pBias[bpos];
    }
}

/**
 * @brief Function to launch the kAddBias_kernel CUDA kernel.
 *
 * @param pUnit Pointer to the unit array.
 * @param pBias Pointer to the bias array.
 * @param stride Stride value.
 * @param batch Batch size.
 */
void kAddBias(NNFloat* pUnit, NNFloat* pBias, uint32_t stride, uint32_t batch)
{
    uint32_t size = stride * batch;
    uint32_t threadsPerBlock = 256;  // Choose an appropriate value based on the GPU architecture
    uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    kAddBias_kernel<<<blocks, threadsPerBlock>>>(pUnit, pBias, stride, size);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "CUDA error in kAddBias: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

/**
 * @brief CUDA kernel to add dual biases to the unit array.
 *
 * @param pUnit    Pointer to the unit array.
 * @param pBias1   Pointer to the first bias array.
 * @param pBias2   Pointer to the second bias array.
 * @param stride   Stride value for indexing the bias arrays.
 * @param size     Total size of the unit array.
 */
__global__ void kAddDualBias_kernel(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, uint32_t stride, uint32_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bpos = pos % stride;
    if (pos < size)
    {
        pUnit[pos] += pBias1[bpos] + pBias2[bpos];
    }
}

/**
 * @brief Adds dual biases to the unit array using CUDA.
 *
 * @param pUnit    Pointer to the unit array.
 * @param pBias1   Pointer to the first bias array.
 * @param pBias2   Pointer to the second bias array.
 * @param stride   Stride value for indexing the bias arrays.
 * @param batch    Number of batches in the unit array.
 */
void kAddDualBias(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    kAddDualBias_kernel<<<blocks, threadsPerBlock>>>(pUnit, pBias1, pBias2, stride, size);
    LAUNCHERROR("kAddDualBias_kernel");
}

/**
 * @brief CUDA kernel to add triple biases to the unit array.
 *
 * @param pUnit    Pointer to the unit array.
 * @param pBias1   Pointer to the first bias array.
 * @param pBias2   Pointer to the second bias array.
 * @param pBias3   Pointer to the third bias array.
 * @param stride   Stride value for indexing the bias arrays.
 * @param size     Total size of the unit array.
 */
__global__ void kAddTripleBias_kernel(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, NNFloat* pBias3, uint32_t stride, uint32_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bpos = pos % stride;
    if (pos < size)
    {
        pUnit[pos] += pBias1[bpos] + pBias2[bpos] + pBias3[pos];
    }
}

/**
 * @brief Adds triple biases to the unit array using CUDA.
 *
 * @param pUnit    Pointer to the unit array.
 * @param pBias1   Pointer to the first bias array.
 * @param pBias2   Pointer to the second bias array.
 * @param pBias3   Pointer to the third bias array.
 * @param stride   Stride value for indexing the bias arrays.
 * @param batch    Number of batches in the unit array.
 */
void kAddTripleBias(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, NNFloat* pBias3, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    kAddTripleBias_kernel<<<blocks, threadsPerBlock>>>(pUnit, pBias1, pBias2, pBias3, stride, size);
    LAUNCHERROR("kAddTripleBias_kernel");
}

/**
 * @brief CUDA kernel to add quad biases to the unit array.
 *
 * @param pUnit    Pointer to the unit array.
 * @param pBias1   Pointer to the first bias array.
 * @param pBias2   Pointer to the second bias array.
 * @param pBias3   Pointer to the third bias array.
 * @param pBias4   Pointer to the fourth bias array.
 * @param stride   Stride value for indexing the bias arrays.
 * @param size     Total size of the unit array.
 */
__global__ void kAddQuadBias_kernel(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, NNFloat* pBias3, NNFloat* pBias4, uint32_t stride, uint32_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bpos = pos % stride;
    if (pos < size)
    {
        pUnit[pos] += pBias1[bpos] + pBias2[bpos] + pBias3[pos] + pBias4[pos];
    }
}

/**
 * @brief Adds quad biases to the unit array using CUDA.
 *
 * @param pUnit    Pointer to the unit array.
 * @param pBias1   Pointer to the first bias array.
 * @param pBias2   Pointer to the second bias array.
 * @param pBias3   Pointer to the third bias array.
 * @param pBias4   Pointer to the fourth bias array.
 * @param stride   Stride value for indexing the bias arrays.
 * @param batch    Number of batches in the unit array.
 */
void kAddQuadBias(NNFloat* pUnit, NNFloat* pBias1, NNFloat* pBias2, NNFloat* pBias3, NNFloat* pBias4, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    kAddQuadBias_kernel<<<blocks, threadsPerBlock>>>(pUnit, pBias1, pBias2, pBias3, pBias4, stride, size);
    LAUNCHERROR("kAddQuadBias_kernel");
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
constexpr uint32_t MAXSPARSE = SM_6X_MAXSPARSE;
constexpr uint32_t MAXSPARSEANALOG = SM_6X_MAXSPARSEANALOG;
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 500)
constexpr uint32_t MAXSPARSE = SM_5X_MAXSPARSE;
constexpr uint32_t MAXSPARSEANALOG = SM_5X_MAXSPARSEANALOG;
#else
constexpr uint32_t MAXSPARSE = SM_3X_MAXSPARSE;
constexpr uint32_t MAXSPARSEANALOG = SM_3X_MAXSPARSEANALOG;
#endif

/**
 * @brief CUDA kernel for calculating sparse Z values with improved optimizations.
 *
 * @param position        Starting position for the sparse indices.
 * @param stride          Stride value for accessing data.
 * @param pWeight         Pointer to the weight data.
 * @param pSparseStart    Pointer to the start indices of the sparse data.
 * @param pSparseEnd      Pointer to the end indices of the sparse data.
 * @param pSparseIndex    Pointer to the sparse indices.
 * @param pDataWeight     Pointer to the weight data (optional).
 * @param pUnit           Pointer to the unit data.
 * @param beta            Beta value for calculating the unit.
 */
__global__ void LAUNCH_BOUNDS256_kCalculateSparseZ_kernel(uint32_t position, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, NNFloat* pUnit, NNFloat beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSE];

    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    NNFloat w = (pDataWeight != nullptr) ? pDataWeight[position] : 1.0f;
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        sOpos = blockDim.x;
        uint64_t inputs = min(end - start, static_cast<uint64_t>(MAXSPARSE));
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
                NNFloat unit = (beta == 0.0f) ? 0.0f : (beta * pUnit[opos]);

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
                opos = __shfl_sync(0xFFFFFFFF, opos, 0);
            }
        }

        start = tend;
        beta = 1.0f;
    }
}

/**
 * @brief Calculates the sparse Z values for a given position and batch using CUDA.
 *
 * @param position The position index.
 * @param batch The batch size.
 * @param stride The stride value.
 * @param pWeight Pointer to the weight array.
 * @param pSparseStart Pointer to the array storing the start indices of sparse data.
 * @param pSparseEnd Pointer to the array storing the end indices of sparse data.
 * @param pSparseIndex Pointer to the array storing the indices of sparse data.
 * @param pDataWeight Pointer to the array storing the weight values of sparse data.
 * @param pUnit Pointer to the array storing the unit values.
 * @param beta The beta value.
 */
void kCalculateSparseZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, NNFloat* pUnit, NNFloat beta)
{
    constexpr uint32_t warpSize = getGpu()._warpSize;
    constexpr uint32_t warpBits = getGpu()._warpBits;

    uint32_t threads = min(256, ((stride + warpSize - 1) >> warpBits) << warpBits);
    kCalculateSparseZ_kernel<<<batch, threads>>>(position, stride, pWeight, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pUnit, beta);
    cudaError_t launchError = cudaGetLastError();
    if (launchError != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(launchError));
    }
}
/**
 * @brief CUDA kernel to calculate indexed sparse Z values with optimizations.
 *
 * @param position       Starting position in the sparse index.
 * @param stride         Stride value for accessing weight and unit arrays.
 * @param pWeight        Pointer to the weight array.
 * @param pIndex         Pointer to the index array.
 * @param pSparseStart   Pointer to the sparse start array.
 * @param pSparseEnd     Pointer to the sparse end array.
 * @param pSparseIndex   Pointer to the sparse index array.
 * @param pDataWeight    Pointer to the data weight array.
 * @param pUnit          Pointer to the unit array.
 * @param beta           Beta value.
 */

__global__ void LAUNCH_BOUNDS256_kCalculateIndexedSparseZ_kernel(uint32_t position, uint32_t stride, NNFloat* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, NNFloat* pUnit, NNFloat beta)
{
    __shared__ uint32_t sOffset[MAXSPARSE];

    uint32_t blockPosition = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t start = pSparseStart[blockPosition];
    uint64_t end = pSparseEnd[blockPosition];
    NNFloat w = (pDataWeight != NULL) ? pDataWeight[blockPosition] : (NNFloat)1.0;

    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        uint32_t inputs = ullmin(end - start, static_cast<uint64_t>(MAXSPARSE));
        uint64_t tstart = start + threadIdx.x;
        uint64_t tend = start + inputs;

        for (uint32_t pos = threadIdx.x; pos < inputs; pos += blockDim.x)
        {
            sOffset[pos] = pSparseIndex[tstart + pos] * stride;
        }

        __syncthreads();

        uint32_t opos = threadIdx.x;

        while (opos < stride)
        {
            NNFloat unit = (beta == (NNFloat)0.0) ? (NNFloat)0.0 : (beta * pUnit[opos]);

            for (uint32_t i = 0; i < inputs; i++)
            {
                uint32_t offset = sOffset[i];
                unit += w * pWeight[offset + opos];
            }

            pUnit[opos] = unit;

            opos += blockDim.x;
        }

        start = tend;
        __syncthreads();
        beta = (NNFloat)1.0;
    }
}

/**
 * @brief Calculates indexed sparse Z values using CUDA.
 *
 * @param position     The position value.
 * @param batch        The batch size.
 * @param stride       The stride value.
 * @param pWeight      Pointer to the weight array.
 * @param pIndex       Pointer to the index array.
 * @param pSparseStart Pointer to the start array of the sparse indices.
 * @param pSparseEnd   Pointer to the end array of the sparse indices.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight  Pointer to the data weight array.
 * @param pUnit        Pointer to the unit array.
 * @param beta         The beta value.
 */
void kCalculateIndexedSparseZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, NNFloat* pUnit, NNFloat beta)
{
    // Calculate the number of threads per block
    constexpr uint32_t kBlockSize = 256;
    constexpr uint32_t kWarpSize = 32;
    uint32_t threads = min(kBlockSize, ((stride + kWarpSize - 1) / kWarpSize) * kWarpSize);

    // Launch the kernel with optimized parameters
    dim3 grid(batch);
    dim3 block(threads);
    kCalculateIndexedSparseZ_kernel<<<grid, block>>>(position, stride, pWeight, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pUnit, beta);
    LAUNCHERROR("kCalculateIndexedSparseZ_kernel");
}
/**
 * @brief CUDA kernel to calculate sparse analog activations.
 *
 * @tparam T Data type for sparse data.
 * @param position Starting position in the sparse data.
 * @param stride Stride for accessing the unit data.
 * @param pWeight Pointer to the weight data.
 * @param pSparseStart Pointer to the start indices of sparse data.
 * @param pSparseEnd Pointer to the end indices of sparse data.
 * @param pSparseIndex Pointer to the index data of sparse data.
 * @param pDataWeight Pointer to the weight data for sparse data (optional, can be nullptr).
 * @param pSparseData Pointer to the sparse data.
 * @param pUnit Pointer to the unit data.
 * @param beta Beta value for unit multiplication.
 */
template<typename T>
__global__ void kCalculateSparseAnalogZ_kernel(uint32_t position, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, T* pSparseData, NNFloat* pUnit, NNFloat beta)
{
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ T sValue[MAXSPARSEANALOG];

    uint32_t threadIdxX = threadIdx.x;
    uint32_t blockIdxX = blockIdx.x;
    uint32_t positionIndex = position + blockIdxX;

    if (cData._bShuffleIndices)
        positionIndex = cData._pShuffleIndex[positionIndex];

    uint64_t start = pSparseStart[positionIndex];
    uint64_t end = pSparseEnd[positionIndex];
    NNFloat w = (pDataWeight != nullptr) ? pDataWeight[positionIndex] : 1.0f;
    pUnit += blockIdxX * stride;

    while (start < end)
    {
        uint32_t inputs = min(static_cast<uint32_t>(end - start), MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdxX;
        uint32_t pos = threadIdxX;

        while (tstart < tend)
        {
            sOffset[pos] = pSparseIndex[tstart] * stride;
            sValue[pos] = w * pSparseData[tstart];
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __syncthreads();

        uint32_t tgx = threadIdxX & cData._warpMask;
        uint32_t opos = threadIdxX - tgx;

        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                NNFloat unit = (beta == 0.0f) ? 0.0f : (beta * pUnit[opos]);

                #pragma unroll
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
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
    }
}
/**
 * @brief CUDA kernel to calculate sparse analog activations.
 *
 * @tparam Unused Unused template parameter.
 * @param position The position of the current neuron.
 * @param stride The stride between consecutive neurons in the output layer.
 * @param pWeight Pointer to the weight matrix.
 * @param pSparseStart Pointer to the array of start indices for sparse connections.
 * @param pSparseEnd Pointer to the array of end indices for sparse connections.
 * @param pSparseIndex Pointer to the array of sparse indices.
 * @param pDataWeight Pointer to the array of weight values for sparse connections (optional).
 * @param pSparseData Pointer to the array of input data for sparse connections.
 * @param pUnit Pointer to the output unit activations.
 * @param beta The scaling factor for previous activations.
 */
template <>
__global__ void LAUNCH_BOUNDS256()
kCalculateSparseAnalogZ_kernel(uint32_t position, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, unsigned char* pSparseData, NNFloat* pUnit, NNFloat beta)
{
    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    NNFloat w = (pDataWeight != NULL) ? pDataWeight[position] : static_cast<NNFloat>(1.0);
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        uint32_t inputs = ullmin(end - start, static_cast<uint64_t>(MAXSPARSEANALOG));
        uint64_t tend = start + inputs;

        for (uint32_t tstart = start + threadIdx.x; tstart < tend; tstart += blockDim.x)
        {
            uint32_t sparseIndex = pSparseIndex[tstart];
            NNFloat sparseData = static_cast<NNFloat>(pSparseData[tstart]) * static_cast<NNFloat>(1.0 / 256.0);

            uint32_t offset = sparseIndex * stride;
            NNFloat value = w * sparseData;

            // Compute unit
            NNFloat unit = (beta == static_cast<NNFloat>(0.0)) ? static_cast<NNFloat>(0.0) : (beta * pUnit[threadIdx.x]);
            for (uint32_t i = 0; i < inputs; i++)
            {
                unit += pWeight[offset + i * blockDim.x + threadIdx.x] * value;
            }

            pUnit[threadIdx.x] = unit;
        }

        __syncthreads();

        start = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta = static_cast<NNFloat>(1.0);
    }
}
/**
 * @brief CUDA kernel to calculate sparse analog activations (Z) for a given position and stride.
 * 
 * @tparam The template parameter (typically unused).
 * @param position The position index.
 * @param stride The stride value.
 * @param pWeight Pointer to the weight data.
 * @param pSparseStart Pointer to the start indices of sparse data.
 * @param pSparseEnd Pointer to the end indices of sparse data.
 * @param pSparseIndex Pointer to the indices of sparse data.
 * @param pDataWeight Pointer to the weight data for sparse analog activations.
 * @param pSparseData Pointer to the sparse data.
 * @param pUnit Pointer to the output units.
 * @param beta The beta value for scaling previous unit values.
 */
template <>
__global__ void LAUNCH_BOUNDS256()
kCalculateSparseAnalogZ_kernel(uint32_t position, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, char* pSparseData, NNFloat* pUnit, NNFloat beta)
{
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ NNFloat sValue[MAXSPARSEANALOG];

    uint32_t inputs;
    uint64_t start;
    uint64_t end;
    NNFloat w;

    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    start = pSparseStart[position];
    end = pSparseEnd[position];
    w = (pDataWeight != NULL) ? pDataWeight[position] : (NNFloat)1.0;

    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos] = pSparseIndex[tstart] * stride;
            sValue[pos] = w * ((NNFloat)pSparseData[tstart] * (NNFloat)(1.0 / 256.0));
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __syncthreads();

        uint32_t opos = threadIdx.x;
        while (opos < stride)
        {
            NNFloat unit = (beta == (NNFloat)0.0) ? (NNFloat)0.0 : (beta * pUnit[opos]);

            for (uint32_t i = 0; i < inputs; i++)
            {
                uint32_t offset = sOffset[i];
                NNFloat weight = pWeight[offset + opos];
                NNFloat value = sValue[i];
                unit = fmaf(weight, value, unit);
            }

            pUnit[opos] = unit;

            opos += blockDim.x;
        }

        start = tend;
        __syncthreads();
        beta = (NNFloat)1.0;
    }
}

/**
 * @brief Calculates the sparse analog Z using CUDA.
 *
 * This function calculates the sparse analog Z using CUDA for the given parameters.
 *
 * @tparam T The type of the sparse data.
 * @param position The position parameter.
 * @param batch The batch parameter.
 * @param stride The stride parameter.
 * @param pWeight Pointer to the weight data.
 * @param pSparseStart Pointer to the sparse start data.
 * @param pSparseEnd Pointer to the sparse end data.
 * @param pSparseIndex Pointer to the sparse index data.
 * @param pDataWeight Pointer to the data weight.
 * @param pSparseData Pointer to the sparse data.
 * @param pUnit Pointer to the unit data.
 * @param beta The beta parameter.
 */
template<typename T> 
void kCalculateSparseAnalogZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, 
                             uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, 
                             NNFloat* pDataWeight, T* pSparseData, NNFloat* pUnit, NNFloat beta)
{
    uint32_t warpSize = getGpu()._warpSize;
    uint32_t warpBits = getGpu()._warpBits;
    uint32_t threads = min(256, ((stride + warpSize - 1) >> warpBits) << warpBits);
    dim3 grid(batch);
    dim3 block(threads);
    kCalculateSparseAnalogZ_kernel<<<grid, block>>>(position, stride, pWeight, pSparseStart, 
                                                    pSparseEnd, pSparseIndex, pDataWeight, 
                                                    pSparseData, pUnit, beta);
    LAUNCHERROR("kCalculateSparseAnalogZ_kernel");
}
/**
 * @brief CUDA kernel to calculate indexed sparse analog Z values.
 *
 * @tparam T The data type of the sparse data.
 * @param position The starting position in the index array.
 * @param stride The stride value.
 * @param pWeight Pointer to the weight array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the start array of sparse indices.
 * @param pSparseEnd Pointer to the end array of sparse indices.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 * @param pUnit Pointer to the unit array.
 * @param beta The beta value.
 */
template<typename T>
__global__ void LAUNCH_BOUNDS256() kCalculateIndexedSparseAnalogZ_kernel(uint32_t position, uint32_t stride, NNFloat* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, T* pSparseData, NNFloat* pUnit, NNFloat beta)
{
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ T sValue[MAXSPARSEANALOG];

    uint32_t inputs = 0;
    uint64_t start = 0;
    uint64_t end = 0;
    NNFloat w = (NNFloat)1.0;

    for (uint32_t blockIdx_x = blockIdx.x; blockIdx_x < gridDim.x; blockIdx_x += gridDim.x)
    {
        if (threadIdx.x == 0)
        {
            position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx_x] : position + blockIdx_x];
            start = pSparseStart[position];
            end = pSparseEnd[position];
            w = (pDataWeight != NULL) ? pDataWeight[position] : (NNFloat)1.0;
        }

        __syncthreads();

        while (start < end)
        {
            inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
            uint64_t tend = start + inputs;
            uint64_t tstart = start + threadIdx.x;
            uint32_t pos = threadIdx.x;

            while (tstart < tend)
            {
                sOffset[pos] = pSparseIndex[tstart] * stride;
                sValue[pos] = w * pSparseData[tstart];
                pos += blockDim.x;
                tstart += blockDim.x;
            }

            __syncthreads();

            uint32_t tgx = threadIdx.x & cData._warpMask;
            uint32_t opos = threadIdx.x - tgx;
            NNFloat unit = (beta == (NNFloat)0.0) ? (NNFloat)0.0 : (beta * pUnit[opos]);

            for (uint32_t i = 0; i < inputs; i++)
            {
                uint32_t offset = sOffset[i];
                unit += pWeight[offset + opos] * sValue[i];
            }

            pUnit[opos] = unit;

            for (uint32_t i = 1; i < cData._warpSize; i *= 2)
            {
                opos = __shfl_down_sync(0xFFFFFFFF, opos, i);
                if (opos < stride)
                {
                    unit += pWeight[sOffset[threadIdx.x]] * sValue[threadIdx.x];
                    pUnit[opos] = unit;
                }
            }

            start = tend;

            __syncthreads();

            beta = (NNFloat)1.0;
        }
    }
}
/**
 * @brief CUDA kernel for calculating indexed sparse analog Z.
 *
 * @tparam Unused template parameter.
 * @param position Index position.
 * @param stride Stride value.
 * @param pWeight Pointer to the weight array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 * @param pUnit Pointer to the unit array.
 * @param beta Beta value.
 */
template<>
__global__ void LAUNCH_BOUNDS256()
kCalculateIndexedSparseAnalogZ_kernel(uint32_t position, uint32_t stride, NNFloat* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, unsigned char* pSparseData, NNFloat* pUnit, NNFloat beta)
{
    uint32_t sOpos = blockDim.x;
    
    position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    NNFloat w = (pDataWeight != NULL) ? pDataWeight[position] : (NNFloat)1.0;
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        uint32_t inputs = min(static_cast<uint32_t>(end - start), static_cast<uint32_t>(MAXSPARSEANALOG));
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;

        while (tstart < tend)
        {
            uint32_t offset = pSparseIndex[tstart] * stride;
            NNFloat value = w * (static_cast<NNFloat>(pSparseData[tstart]) * (1.0 / 256.0));
            NNFloat unit = (beta == 0.0) ? 0.0 : (beta * pUnit[offset + threadIdx.x]);
            
            for (uint32_t i = 0; i < inputs; i++)
            {
                unit += pWeight[sparseIndex[start + i] * stride + threadIdx.x] * value;
            }
            
            pUnit[offset + threadIdx.x] = unit;

            tstart += blockDim.x;
        }

        start = tend;
        beta = 1.0;
    }
}
/**
 * @brief CUDA kernel for calculating indexed sparse analog Z values.
 *
 * @tparam Unused Unused template parameter.
 * @param position Starting position for the sparse data.
 * @param stride Stride value.
 * @param pWeight Pointer to the weight data.
 * @param pIndex Pointer to the index data.
 * @param pSparseStart Pointer to the start indices of the sparse data.
 * @param pSparseEnd Pointer to the end indices of the sparse data.
 * @param pSparseIndex Pointer to the indices of the sparse data.
 * @param pDataWeight Pointer to the weight data.
 * @param pSparseData Pointer to the sparse data.
 * @param pUnit Pointer to the unit data.
 * @param beta Beta value.
 */
template<>
__global__ void LAUNCH_BOUNDS256()
kCalculateIndexedSparseAnalogZ_kernel(uint32_t position, uint32_t stride, NNFloat* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, char* pSparseData, NNFloat* pUnit, NNFloat beta)
{
    __shared__ uint32_t sOpos;

    sOpos = blockDim.x;
    position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];

    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    NNFloat w = (pDataWeight != nullptr) ? pDataWeight[position] : 1.0f;
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        uint32_t inputs = ullmin(end - start, static_cast<uint64_t>(MAXSPARSEANALOG));
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            uint32_t offset = pSparseIndex[tstart] * stride;
            NNFloat value = w * (static_cast<NNFloat>(pSparseData[tstart]) * (1.0f / 128.0f));
            sOffset[pos] = offset;
            sValue[pos] = value;
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
                NNFloat unit = (beta == 0.0f) ? 0.0f : (beta * pUnit[opos]);
                NNFloat* pUnitPtr = &pUnit[opos];

                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    unit += pWeight[offset + opos] * sValue[i];
                }

                *pUnitPtr = unit;
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
        beta = 1.0f;
    }
}

/**
 * Calculates the analog activation value for indexed sparse data.
 *
 * @tparam T              The data type of the sparse data.
 * @param position        The position of the data in the batch.
 * @param batch           The number of data points in the batch.
 * @param stride          The stride of the sparse data.
 * @param pWeight         Pointer to the weight data.
 * @param pIndex          Pointer to the index data.
 * @param pSparseStart    Pointer to the start indices of sparse data.
 * @param pSparseEnd      Pointer to the end indices of sparse data.
 * @param pSparseIndex    Pointer to the indices of sparse data.
 * @param pDataWeight     Pointer to the weight data for sparse data.
 * @param pSparseData     Pointer to the sparse data.
 * @param pUnit           Pointer to the unit data.
 * @param beta            The beta value.
 */
template<typename T>
void kCalculateIndexedSparseAnalogZ(uint32_t position, uint32_t batch, uint32_t stride,
                                   NNFloat* pWeight, uint32_t* pIndex, uint64_t* pSparseStart,
                                   uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight,
                                   T* pSparseData, NNFloat* pUnit, NNFloat beta)
{
    uint32_t threads = min(256, stride);
    uint32_t blocks = (batch + threads - 1) / threads;
    kCalculateIndexedSparseAnalogZ_kernel<T><<<blocks, threads>>>(position, stride, pWeight, pIndex,
                                                                   pSparseStart, pSparseEnd, pSparseIndex,
                                                                   pDataWeight, pSparseData, pUnit, beta);
    cudaDeviceSynchronize();
    LAUNCHERROR("kCalculateIndexedSparseAnalogZ_kernel");
}
/**
 * @brief CUDA kernel for calculating sparse denoised Z values.
 *
 * @param position The position of the sparse data.
 * @param stride The stride between consecutive units.
 * @param pWeight Pointer to the weight data.
 * @param pSparseStart Pointer to the start indices of the sparse data.
 * @param pSparseEnd Pointer to the end indices of the sparse data.
 * @param pSparseIndex Pointer to the index data of the sparse data.
 * @param pDataWeight Pointer to the weight data of the sparse data.
 * @param pRandom Pointer to the random data.
 * @param pUnit Pointer to the output unit data.
 * @param beta The beta value used in the calculation.
 */
__global__ void kCalculateSparseDenoisedZ_kernel(uint32_t position, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta)
{
    uint32_t tid = threadIdx.x;
    uint32_t blockIdxX = blockIdx.x;
    
    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdxX] : position + blockIdxX;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    NNFloat w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (NNFloat)1.0);
    pUnit += blockIdxX * stride;
    
    while (start < end)
    {
        uint64_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t inputStart = start + tid;
        uint64_t inputEnd = start + inputs;

        NNFloat unit = (beta == (NNFloat)0.0) ? (NNFloat)0.0 : (beta * pUnit[tid]);
        
        for (uint32_t i = inputStart; i < inputEnd; i += blockDim.x)
        {
            NNFloat value = pRandom[i];
            uint32_t offset = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[i] * stride;
            unit += (offset != cData._maxUint32_t) ? pWeight[offset + tid] : (NNFloat)0.0;
        }

        __syncthreads();
        
        for (uint32_t offset = tid; offset < stride; offset += blockDim.x)
        {
            pUnit[offset] = w * unit;
            unit = (beta == (NNFloat)0.0) ? (NNFloat)0.0 : (beta * pUnit[offset + blockDim.x]);
        }

        start = inputEnd;
        __syncthreads();
        beta = (NNFloat)1.0;
    }
}
/**
 * @brief Calculates the sparse denoised Z values.
 *
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pWeight The weight.
 * @param pSparseStart The start of the sparse data.
 * @param pSparseEnd The end of the sparse data.
 * @param pSparseIndex The sparse index.
 * @param pDataWeight The data weight.
 * @param pRandom The random values.
 * @param pUnit The unit values.
 * @param beta The beta value.
 */
void kCalculateSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight,
                              uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex,
                              NNFloat* pDataWeight, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta)
{
    uint32_t threads = min(256, (stride + getGpu()._warpSize - 1) / getGpu()._warpSize) * getGpu()._warpSize;
    uint32_t blocks = (stride + threads - 1) / threads;

    kCalculateSparseDenoisedZ_kernel<<<batch, threads>>>(position, stride, pWeight, pSparseStart,
                                                         pSparseEnd, pSparseIndex, pDataWeight,
                                                         pRandom, pUnit, beta);

    cudaDeviceSynchronize();

    LAUNCHERROR("kCalculateSparseDenoisedZ_kernel");
}
/**
 * @brief CUDA kernel for calculating indexed sparse denoised Z values.
 *
 * @param position The position parameter.
 * @param stride The stride parameter.
 * @param pWeight Pointer to the weight array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pRandom Pointer to the random array.
 * @param pUnit Pointer to the unit array.
 * @param beta The beta parameter.
 */
__global__ void kCalculateIndexedSparseDenoisedZ_kernel(uint32_t position, uint32_t stride, NNFloat* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta)
{
    uint32_t positionIdx = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t start = pSparseStart[positionIdx];
    uint64_t end = pSparseEnd[positionIdx];
    NNFloat w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[positionIdx] : (NNFloat)1.0);
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;

        for (uint32_t i = threadIdx.x; i < inputs; i += blockDim.x)
        {
            NNFloat value = pRandom[start + i];
            uint32_t offset = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[start + i] * stride;
            pUnit[offset + threadIdx.x] += (offset != cData._maxUint32_t) ? pWeight[offset + threadIdx.x] : (NNFloat)0.0;
        }

        __syncwarp();

        for (uint32_t opos = threadIdx.x; opos < stride; opos += blockDim.x)
        {
            NNFloat unit = (beta == (NNFloat)0.0) ? (NNFloat)0.0 : (beta * pUnit[opos]);
            for (uint32_t i = 1; i < blockDim.x; i++)
            {
                unit += __shfl_sync(0xFFFFFFFF, unit, i);
            }
            pUnit[opos] = w * unit;
        }

        __syncwarp();

        start = tend;
        if (start < end)
        {
            beta = (NNFloat)1.0;
        }
    }
}
/**
 * @brief Calculates indexed sparse denoised Z values using CUDA.
 *
 * @param position The position parameter.
 * @param batch The number of batches.
 * @param stride The stride parameter.
 * @param pWeight Pointer to the weight array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pRandom Pointer to the random array.
 * @param pUnit Pointer to the unit array.
 * @param beta The beta parameter.
 */
void kCalculateIndexedSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta)
{
    uint32_t threads = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    dim3 blocks(batch);
    dim3 threadsPerBlock(threads);
    kCalculateIndexedSparseDenoisedZ_kernel<<<blocks, threadsPerBlock>>>(position, stride, pWeight, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pRandom, pUnit, beta);
    LAUNCHERROR("kCalculateIndexedSparseDenoisedZ_kernel");
}
/**
 * @brief CUDA kernel for calculating sparse analog denoised Z.
 *
 * @tparam T The data type for the sparse data.
 * @param position The position parameter.
 * @param stride The stride parameter.
 * @param pWeight Pointer to the weight data.
 * @param pSparseStart Pointer to the sparse start data.
 * @param pSparseEnd Pointer to the sparse end data.
 * @param pSparseIndex Pointer to the sparse index data.
 * @param pDataWeight Pointer to the data weight.
 * @param pSparseData Pointer to the sparse data.
 * @param pRandom Pointer to the random data.
 * @param pUnit Pointer to the unit data.
 * @param beta The beta parameter.
 */
constexpr uint32_t MAXSPARSEANALOG = 256;

template <typename T>
__global__ void LAUNCH_BOUNDS256(uint32_t position, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, NNFloat* pSparseData, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta)
{
    __shared__ uint32_t sOpos;
    __shared__ T sValue[MAXSPARSEANALOG];
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];

    sOpos = blockDim.x;
    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;

    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    NNFloat w = cData._denoising_q * ((pDataWeight != nullptr) ? pDataWeight[position] : 1.0);
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        uint32_t inputs = ullmin(end - start, static_cast<uint64_t>(MAXSPARSEANALOG));
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            for (uint32_t i = 0; i < inputs; i += blockDim.x)
            {
                uint64_t tindex = tstart + i;
                if (tindex < tend)
                {
                    NNFloat value = pRandom[tindex];
                    sOffset[pos] = (value < cData._denoising_p) ? cData._maxUint32_t : static_cast<uint32_t>(pSparseIndex[tindex]) * stride;
                    sValue[pos] = pSparseData[tindex] * w;
                    pos += blockDim.x;
                }
            }
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
                NNFloat unit = (beta == 0.0) ? 0.0 : pUnit[opos];

                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];

                    if (offset != cData._maxUint32_t)
                    {
                        unit += pWeight[offset + opos] * sValue[i];
                    }
                }

                pUnit[opos] = unit;
            }

            opos -= tgx;

            if (tgx == 0)
            {
                opos = __shfl(opos, 0);
            }

            opos = __syncwarp(opos);
        }

        start = tend;
        beta = 1.0;

        __syncthreads();
    }
}
/**
 * @brief CUDA kernel for calculating sparse analog denoised Z values.
 *
 * @tparam Unused Unused template parameter.
 * @param position Position parameter.
 * @param stride Stride parameter.
 * @param pWeight Pointer to weights.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to sparse indices.
 * @param pDataWeight Pointer to data weights.
 * @param pSparseData Pointer to sparse data.
 * @param pRandom Pointer to random values.
 * @param pUnit Pointer to unit values.
 * @param beta Beta parameter.
 */
template <>
__global__ void LAUNCH_BOUNDS256()
    kCalculateSparseAnalogDenoisedZ_kernel(uint32_t position, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, unsigned char* pSparseData, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta)
{
    __shared__ uint32_t sOpos;
    __shared__ int32_t sOffset[MAXSPARSEANALOG];
    __shared__ NNFloat sValue[MAXSPARSEANALOG];

    constexpr uint32_t numThreads = blockDim.x;
    constexpr uint32_t warpSize = cData._warpSize;
    constexpr uint32_t warpMask = warpSize - 1;
    constexpr uint32_t maxSparseAnalog = MAXSPARSEANALOG;

    sOpos = numThreads;
    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    NNFloat w = cData._denoising_q * (static_cast<NNFloat>(pDataWeight != nullptr) * pDataWeight[position] + static_cast<NNFloat>(pDataWeight == nullptr));
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        uint32_t inputs = ullmin(end - start, static_cast<uint64_t>(maxSparseAnalog));
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            NNFloat value = pRandom[tstart];
            sOffset[pos] = (value < cData._denoising_p) ? cData._maxUint32_t : static_cast<int32_t>(pSparseIndex[tstart]) * stride;
            sValue[pos] = static_cast<NNFloat>(pSparseData[tstart]) * static_cast<NNFloat>(1.0 / 256.0) * w;
            pos += numThreads;
            tstart += numThreads;
        }

        __syncthreads();

        uint32_t tgx = threadIdx.x & warpMask;
        uint32_t opos = threadIdx.x - tgx;

        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                NNFloat unit = (beta == static_cast<NNFloat>(0.0)) ? static_cast<NNFloat>(0.0) : (beta * pUnit[opos]);

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
                opos = atomicAdd(&sOpos, warpSize);
            }

            opos = __shfl(opos, 0, warpSize);
        }

        start = tend;

        if (start < end)
        {
            __syncthreads();
        }

        beta = static_cast<NNFloat>(1.0);
    }
}
/**
 * @brief CUDA kernel to calculate sparse analog denoised Z
 *
 * @param position       The position parameter
 * @param stride         The stride parameter
 * @param pWeight        Pointer to the weight array
 * @param pSparseStart   Pointer to the sparse start array
 * @param pSparseEnd     Pointer to the sparse end array
 * @param pSparseIndex   Pointer to the sparse index array
 * @param pDataWeight    Pointer to the data weight array
 * @param pSparseData    Pointer to the sparse data array
 * @param pRandom        Pointer to the random array
 * @param pUnit          Pointer to the unit array
 * @param beta           The beta parameter
 */
constexpr uint32_t MAXSPARSEANALOG = 128;

__global__ void kCalculateSparseAnalogDenoisedZ_kernel(
    uint32_t position,
    uint32_t stride,
    NNFloat* pWeight,
    uint64_t* pSparseStart,
    uint64_t* pSparseEnd,
    const uint32_t* pSparseIndex,
    NNFloat* pDataWeight,
    const char* pSparseData,
    NNFloat* pRandom,
    NNFloat* pUnit,
    NNFloat beta)
{
    uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ NNFloat sValue[MAXSPARSEANALOG];

    sOpos = blockDim.x;
    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    NNFloat w = cData._denoising_q * ((pDataWeight != nullptr) ? pDataWeight[position] : static_cast<NNFloat>(1.0));
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        uint32_t inputs = ullmin(end - start, static_cast<uint64_t>(MAXSPARSEANALOG));
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            NNFloat value = pRandom[tstart];
            sOffset[pos] = (value < cData._denoising_p) ? cData._maxUint32_t : static_cast<uint32_t>(pSparseIndex[tstart]) * stride;
            sValue[pos] = static_cast<NNFloat>(pSparseData[tstart]) * (1.0 / 128.0) * w;
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
                NNFloat unit = (beta == static_cast<NNFloat>(0.0)) ? static_cast<NNFloat>(0.0) : (beta * pUnit[opos]);

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
                uint32_t localOpos = warpReduceSum(opos);
                if (threadIdx.x % cData._warpSize == 0)
                    atomicAdd(&sOpos, localOpos);
            }

            __syncthreads();
        }

        start = tend;

        if (start < end)
            __syncthreads();

        beta = static_cast<NNFloat>(1.0);
    }
}

/**
 * @brief Calculates the sparse analog denoised Z values using CUDA.
 *
 * @tparam T The data type of the sparse data.
 * @param position The position parameter.
 * @param batch The batch size.
 * @param stride The stride value.
 * @param pWeight Pointer to the weight array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 * @param pRandom Pointer to the random array.
 * @param pUnit Pointer to the unit array.
 * @param beta The beta value.
 */
template<typename T>
void kCalculateSparseAnalogDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, T* pSparseData, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta)
{
    uint32_t blockSize = 256;
    uint32_t gridSize = (batch + blockSize - 1) / blockSize;

    /**
     * @brief Kernel function to calculate sparse analog denoised Z values.
     *
     * @param position The position parameter.
     * @param stride The stride value.
     * @param pWeight Pointer to the weight array.
     * @param pSparseStart Pointer to the sparse start array.
     * @param pSparseEnd Pointer to the sparse end array.
     * @param pSparseIndex Pointer to the sparse index array.
     * @param pDataWeight Pointer to the data weight array.
     * @param pSparseData Pointer to the sparse data array.
     * @param pRandom Pointer to the random array.
     * @param pUnit Pointer to the unit array.
     * @param beta The beta value.
     */
    kCalculateSparseAnalogDenoisedZ_kernel<<<gridSize, blockSize>>>(position, stride, pWeight, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pRandom, pUnit, beta);

    cudaDeviceSynchronize();
    CUDAERROR("kCalculateSparseAnalogDenoisedZ_kernel");
}
/**
 * @brief CUDA kernel to calculate indexed sparse analog denoised Z values.
 *
 * @tparam T The type of the sparse data.
 * @param position The starting position of the indexed sparse data.
 * @param stride The stride value.
 * @param pWeight Pointer to the weight data.
 * @param pIndex Pointer to the index data.
 * @param pSparseStart Pointer to the start indices of the sparse data.
 * @param pSparseEnd Pointer to the end indices of the sparse data.
 * @param pSparseIndex Pointer to the indices of the sparse data.
 * @param pDataWeight Pointer to the data weight.
 * @param pSparseData Pointer to the sparse data.
 * @param pRandom Pointer to the random data.
 * @param pUnit Pointer to the unit data.
 * @param beta The beta value.
 */
template<typename T>
__global__ void kCalculateIndexedSparseAnalogDenoisedZ_kernel(uint32_t position, uint32_t stride, NNFloat* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, T* pSparseData, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta)
{
    __shared__ uint32_t sOpos;
    extern __shared__ uint8_t sSharedMemory[];
    T* sValue = reinterpret_cast<T*>(sSharedMemory);
    uint32_t* sOffset = reinterpret_cast<uint32_t*>(sSharedMemory + sizeof(T) * MAXSPARSEANALOG);

    sOpos = blockDim.x;
    position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    NNFloat w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : static_cast<NNFloat>(1.0));
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        uint32_t inputs = ullmin(end - start, static_cast<uint64_t>(MAXSPARSEANALOG));
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            NNFloat value = pRandom[tstart];
            sOffset[pos] = (value < cData._denoising_p) ? cData._maxUint32_t : static_cast<uint32_t>(pSparseIndex[tstart]) * stride;
            sValue[pos] = pSparseData[tstart] * w;
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
                NNFloat unit = (beta == static_cast<NNFloat>(0.0)) ? static_cast<NNFloat>(0.0) : (beta * pUnit[opos]);
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
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;
        beta = static_cast<NNFloat>(1.0);
    }
}
/**
 * @brief CUDA kernel for calculating indexed sparse analog denoised Z values.
 * 
 * @tparam Unused Unused template parameter.
 * @param position Position parameter.
 * @param stride Stride parameter.
 * @param pWeight Pointer to weights.
 * @param pIndex Pointer to indices.
 * @param pSparseStart Pointer to sparse start.
 * @param pSparseEnd Pointer to sparse end.
 * @param pSparseIndex Pointer to sparse indices.
 * @param pDataWeight Pointer to data weights.
 * @param pSparseData Pointer to sparse data.
 * @param pRandom Pointer to random values.
 * @param pUnit Pointer to units.
 * @param beta Beta parameter.
 */
template <>
__global__ void LAUNCH_BOUNDS256()
kCalculateIndexedSparseAnalogDenoisedZ_kernel(uint32_t position, uint32_t stride, NNFloat* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, unsigned char* pSparseData, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta)
{
    __shared__ uint32_t sOpos;
    __shared__ int32_t sOffset[MAXSPARSEANALOG];
    __shared__ NNFloat sValue[MAXSPARSEANALOG];

    uint32_t tid = threadIdx.x;
    uint32_t blockIndex = blockIdx.x;

    sOpos = blockDim.x;
    position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIndex] : position + blockIndex];
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    NNFloat w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (NNFloat)1.0);
    pUnit += blockIndex * stride;

    for (uint32_t i = tid; i < MAXSPARSEANALOG; i += blockDim.x)
    {
        sOffset[i] = 0;
        sValue[i] = 0.0;
    }

    while (start < end)
    {
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + tid;
        uint32_t pos = tid;

        while (tstart < tend)
        {
            NNFloat value = pRandom[tstart];
            sOffset[pos] = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart] * stride;
            sValue[pos] = (NNFloat)pSparseData[tstart] * (NNFloat)(1.0 / 256.0) * w;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __syncthreads();

        uint32_t tgx = tid & cData._warpMask;
        uint32_t opos = tid - tgx;

        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                NNFloat unit = (beta == (NNFloat)0.0) ? (NNFloat)0.0 : (beta * pUnit[opos]);

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
        beta = (NNFloat)1.0;
    }
}
/**
 * @brief Kernel function to calculate indexed sparse analog denoised Z.
 * @tparam T Type of the input data.
 * @param position Index of the current position.
 * @param stride Stride value.
 * @param pWeight Pointer to the weight data.
 * @param pIndex Pointer to the index data.
 * @param pSparseStart Pointer to the sparse start data.
 * @param pSparseEnd Pointer to the sparse end data.
 * @param pSparseIndex Pointer to the sparse index data.
 * @param pDataWeight Pointer to the data weight data.
 * @param pSparseData Pointer to the sparse data.
 * @param pRandom Pointer to the random data.
 * @param pUnit Pointer to the unit data.
 * @param beta Beta value.
 */
template <>
__global__ void LAUNCH_BOUNDS256(uint32_t position, uint32_t stride, const NNFloat* pWeight, const uint32_t* pIndex, const uint64_t* pSparseStart, const uint64_t* pSparseEnd, const uint32_t* pSparseIndex, const NNFloat* pDataWeight, const char* pSparseData, const NNFloat* pRandom, NNFloat* pUnit, const NNFloat beta)
{
    __shared__ uint32_t sOpos;
    sOpos = blockDim.x;
    position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    const uint64_t start = pSparseStart[position];
    const uint64_t end = pSparseEnd[position];
    const NNFloat w = cData._denoising_q * ((pDataWeight != nullptr) ? pDataWeight[position] : (NNFloat)1.0);
    pUnit += blockIdx.x * stride;
    while (start < end)
    {
        const uint32_t inputs = ullmin(end - start, static_cast<uint64_t>(MAXSPARSEANALOG));
        const uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            const NNFloat value = pRandom[tstart];
            const uint32_t offset = (value < cData._denoising_p) ? cData._maxUint32_t : static_cast<int32_t>(pSparseIndex[tstart]) * stride;
            const NNFloat valueScaled = static_cast<NNFloat>(pSparseData[tstart]) * (NNFloat)(0.0078125) * w;
            sOffset[pos] = offset;
            sValue[pos] = valueScaled;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        const uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                NNFloat unit = (beta == (NNFloat)0.0) ? (NNFloat)0.0 : (beta * pUnit[opos]);
                #pragma unroll
                for (uint32_t i = 0; i < inputs; i++)
                {
                    const uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit += pWeight[offset + opos] * sValue[i];
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
        beta = (NNFloat)1.0;
    }
}

/**
 * \brief Calculates the indexed sparse analog denoised Z values using CUDA.
 *
 * \tparam T The data type of the sparse data.
 * \param position The position parameter.
 * \param batch The batch size.
 * \param stride The stride parameter.
 * \param pWeight Pointer to the weight data.
 * \param pIndex Pointer to the index data.
 * \param pSparseStart Pointer to the sparse start data.
 * \param pSparseEnd Pointer to the sparse end data.
 * \param pSparseIndex Pointer to the sparse index data.
 * \param pDataWeight Pointer to the data weight data.
 * \param pSparseData Pointer to the sparse data.
 * \param pRandom Pointer to the random data.
 * \param pUnit Pointer to the unit data.
 * \param beta The beta parameter.
 */
template<typename T>
void kCalculateIndexedSparseAnalogDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, T* pSparseData, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta)
{
    uint32_t threads = min(256, stride);
    uint32_t blocks = batch;

    size_t sharedMemSize = sizeof(NNFloat) * threads * 2;

    kCalculateIndexedSparseAnalogDenoisedZ_kernel<<<blocks, threads, sharedMemSize>>>(position, stride, pWeight, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pRandom, pUnit, beta);

    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief CUDA kernel to calculate the sparse transposed matrix.
 *
 * @param position The starting position in the sparse matrix.
 * @param batch The number of batches to process.
 * @param pSparseStart Pointer to the start indices of the sparse matrix.
 * @param pSparseEnd Pointer to the end indices of the sparse matrix.
 * @param pSparseIndex Pointer to the indices of the sparse matrix.
 * @param pSparseTransposedEnd Pointer to the end indices of the transposed sparse matrix.
 * @param pSparseTransposedIndex Pointer to the indices of the transposed sparse matrix.
 */
__global__ void kCalculateSparseTransposedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < batch)
    {
        position = cData._bShuffleIndices ? cData._pShuffleIndex[position + tid] : position + tid;
        
        uint64_t start = pSparseStart[position] + threadIdx.x;
        uint64_t end = pSparseEnd[position];
        
        while (start < end)
        {
            uint32_t index = pSparseIndex[start];
            uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos] = tid;
            start += blockDim.x;
        }
    }
}
/**
 * @brief CUDA kernel for calculating weighted sparse transposed matrix.
 *
 * @param pSparseStart Array of starting positions for each position in the sparse matrix.
 * @param pSparseEnd Array of ending positions (exclusive) for each position in the sparse matrix.
 * @param pSparseIndex Array of indices for each non-zero element in the sparse matrix.
 * @param pDataWeight Array of weights for each position in the sparse matrix.
 * @param pSparseTransposedEnd Array of ending positions (exclusive) for each transposed index.
 * @param pSparseTransposedIndex Array of indices in the transposed matrix.
 * @param pSparseTransposedData Array of data values in the transposed matrix.
 * @param bShuffleIndices Flag indicating whether to shuffle indices.
 * @param pShuffleIndex Array of shuffled indices.
 * @param warpSize Size of a warp (typically 32).
 * @param warpMask Mask to extract lane ID within a warp.
 */
__global__ void kCalculateWeightedSparseTransposedMatrix_kernel(const uint64_t* __restrict__ pSparseStart,
                                                                const uint64_t* __restrict__ pSparseEnd,
                                                                const uint32_t* __restrict__ pSparseIndex,
                                                                const NNFloat* __restrict__ pDataWeight,
                                                                uint32_t* __restrict__ pSparseTransposedEnd,
                                                                uint32_t* __restrict__ pSparseTransposedIndex,
                                                                NNFloat* __restrict__ pSparseTransposedData,
                                                                const bool bShuffleIndices,
                                                                const uint32_t* __restrict__ pShuffleIndex,
                                                                const uint32_t warpSize,
                                                                const uint32_t warpMask)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t laneId = tid & warpMask;
    const uint32_t warpId = tid / warpSize;
    const uint32_t numWarps = blockDim.x / warpSize;
    const uint32_t bpos = (blockIdx.x * numWarps) + warpId;

    if (bpos < batch)
    {
        uint32_t position = bpos;
        if (bShuffleIndices)
            position = pShuffleIndex[position];

        const uint64_t start = pSparseStart[position] + laneId;
        const uint64_t end = pSparseEnd[position];
        const NNFloat w = pDataWeight[position];
        uint32_t blockSum = 0;

        for (uint64_t i = start; i < end; i += warpSize)
        {
            uint32_t index = pSparseIndex[i];
            blockSum += (index < MAX_SPARSE_TRANSPOSED_INDICES) ? 1 : 0;
        }
        __shared__ uint32_t warpSums[MAX_NUM_WARPS];
        warpSums[warpId] = blockSum;
        __syncthreads();

        if (laneId < warpSize)
        {
            uint32_t mask = (1u << numWarps) - 1;
            for (uint32_t offset = numWarps >> 1; offset > 0; offset >>= 1)
            {
                uint32_t val = (warpId < offset) ? warpSums[laneId + offset] : 0;
                warpSums[laneId] += __shfl_xor_sync(mask, val, offset, warpSize);
            }
        }

        const uint32_t warpOffset = (warpId > 0) ? warpSums[warpId - 1] : 0;
        const uint32_t transposedIndex = warpOffset + laneId;

        if (transposedIndex < MAX_SPARSE_TRANSPOSED_INDICES)
        {
            const uint32_t opos = atomicAdd(&pSparseTransposedEnd[transposedIndex], 1);
            pSparseTransposedIndex[opos] = bpos;
            pSparseTransposedData[opos] = w;
        }
    }
}

/**
 * @brief Calculates the transposed matrix for sparse data.
 *
 * @param position              The position of the matrix.
 * @param batch                 The batch size.
 * @param pSparseStart          Pointer to the start of sparse data.
 * @param pSparseEnd            Pointer to the end of sparse data.
 * @param pSparseIndex          Pointer to the indices of sparse data.
 * @param pDataWeight           Pointer to the weights of sparse data (optional).
 * @param pSparseTransposedEnd  Pointer to the end of transposed sparse data.
 * @param pSparseTransposedIndex Pointer to the indices of transposed sparse data.
 * @param pSparseTransposedData Pointer to the transposed sparse data.
 */
void kCalculateSparseTransposedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData)
{
    const uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
    const uint32_t elementsPerBatch = batch * getGpu()._warpSize;

    if (pDataWeight == nullptr)
    {
        kCalculateSparseTransposedMatrix_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, pSparseStart, pSparseEnd, pSparseIndex, pSparseTransposedEnd, pSparseTransposedIndex);
    }
    else
    {
        kCalculateWeightedSparseTransposedMatrix_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);
    }
}

/**
 * @brief CUDA kernel to calculate the indexed sparse transposed matrix.
 *
 * @param position The starting position in the index array.
 * @param batch The number of batches to process.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the array storing the start indices of the sparse matrix.
 * @param pSparseEnd Pointer to the array storing the end indices of the sparse matrix.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pSparseTransposedEnd Pointer to the array storing the end indices of the transposed sparse matrix.
 * @param pSparseTransposedIndex Pointer to the transposed sparse index array.
 */
__global__ void LAUNCH_BOUNDS_kCalculateIndexedSparseTransposedMatrix_kernel(
    uint32_t position,
    uint32_t batch,
    uint32_t* pIndex,
    uint64_t* pSparseStart,
    uint64_t* pSparseEnd,
    uint32_t* pSparseIndex,
    uint32_t* pSparseTransposedEnd,
    uint32_t* pSparseTransposedIndex)
{
    __shared__ uint32_t sharedIndex;
    __shared__ uint64_t sharedStart;
    __shared__ uint64_t sharedEnd;

    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        sharedIndex = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        sharedStart = pSparseStart[sharedIndex];
        sharedEnd = pSparseEnd[sharedIndex];

        uint64_t start = sharedStart + tgx;
        while (start < sharedEnd)
        {
            uint32_t index = pSparseIndex[start];

            uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);

            pSparseTransposedIndex[opos] = bpos;
            start += cData._warpSize;
        }
    }
}

/**
 * @brief CUDA kernel to calculate indexed weighted sparse transposed matrix.
 *
 * @param position Input position.
 * @param batch Number of batches.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseTransposedEnd Pointer to the sparse transposed end array.
 * @param pSparseTransposedIndex Pointer to the sparse transposed index array.
 * @param pSparseTransposedData Pointer to the sparse transposed data array.
 */
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedWeightedSparseTransposedMatrix_kernel(const uint32_t position, const uint32_t batch, const uint32_t* pIndex, const uint64_t* pSparseStart, const uint64_t* pSparseEnd, const uint32_t* pSparseIndex, const NNFloat* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        uint32_t pos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        uint64_t start = pSparseStart[pos] + tgx;
        uint64_t end = pSparseEnd[pos];
        NNFloat w = pDataWeight[pos];

        __shared__ uint64_t sharedSparseStart;
        __shared__ uint64_t sharedSparseEnd;

        if (tgx == 0)
        {
            sharedSparseStart = start;
            sharedSparseEnd = end;
        }

        __syncthreads();

        start = sharedSparseStart;
        end = sharedSparseEnd;

        while (start < end)
        {
            uint32_t index = pSparseIndex[start];
            uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos] = bpos;
            pSparseTransposedData[opos] = w;
            start += cData._warpSize;
        }
    }
}

/**
 * @brief Calculates the indexed sparse transposed matrix.
 *
 * This function calculates the transposed matrix for the given indexed sparse matrix. It supports both weighted and unweighted versions.
 *
 * @param position The position parameter.
 * @param batch The number of batches.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array (NULL for unweighted version).
 * @param pSparseTransposedEnd Pointer to the sparse transposed end array.
 * @param pSparseTransposedIndex Pointer to the sparse transposed index array.
 * @param pSparseTransposedData Pointer to the sparse transposed data array.
 */
void kCalculateIndexedSparseTransposedMatrix(uint32_t position, uint32_t batch, const uint32_t* __restrict__ pIndex, const uint64_t* __restrict__ pSparseStart, const uint64_t* __restrict__ pSparseEnd, const uint32_t* __restrict__ pSparseIndex, const NNFloat* __restrict__ pDataWeight, uint32_t* __restrict__ pSparseTransposedEnd, uint32_t* __restrict__ pSparseTransposedIndex, NNFloat* __restrict__ pSparseTransposedData)
{
    uint32_t blocks = CalculateBlocks(batch);
    if (pDataWeight == NULL)
    {
        /**
         * @brief Kernel for calculating indexed sparse transposed matrix (unweighted version).
         *
         * @param position The position parameter.
         * @param batch The number of batches.
         * @param pIndex Pointer to the index array.
         * @param pSparseStart Pointer to the sparse start array.
         * @param pSparseEnd Pointer to the sparse end array.
         * @param pSparseIndex Pointer to the sparse index array.
         * @param pSparseTransposedEnd Pointer to the sparse transposed end array.
         * @param pSparseTransposedIndex Pointer to the sparse transposed index array.
         */
        kCalculateIndexedSparseTransposedMatrix_kernel<<<blocks, getGpu()._warpSize>>>(position, batch, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseTransposedEnd, pSparseTransposedIndex);
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            fprintf(stderr, "CUDA error in kCalculateIndexedSparseTransposedMatrix_kernel: %s\n", cudaGetErrorString(error));
            exit(-1);
        }
    }
    else
    {
        /**
         * @brief Kernel for calculating indexed weighted sparse transposed matrix.
         *
         * @param position The position parameter.
         * @param batch The number of batches.
         * @param pIndex Pointer to the index array.
         * @param pSparseStart Pointer to the sparse start array.
         * @param pSparseEnd Pointer to the sparse end array.
         * @param pSparseIndex Pointer to the sparse index array.
         * @param pDataWeight Pointer to the data weight array.
         * @param pSparseTransposedEnd Pointer to the sparse transposed end array.
         * @param pSparseTransposedIndex Pointer to the sparse transposed index array.
         * @param pSparseTransposedData Pointer to the sparse transposed data array.
         */
        kCalculateIndexedWeightedSparseTransposedMatrix_kernel<<<blocks, getGpu()._warpSize>>>(position, batch, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            fprintf(stderr, "CUDA error in kCalculateIndexedWeightedSparseTransposedMatrix_kernel: %s\n", cudaGetErrorString(error));
            exit(-1);
        }
    }
}
/**
 * @brief CUDA kernel to calculate sparse transposed denoised matrix
 *
 * @param pSparseStart         Pointer to the start indices of sparse matrix rows
 * @param pSparseEnd           Pointer to the end indices of sparse matrix rows
 * @param pSparseIndex         Pointer to the column indices of sparse matrix
 * @param pRandom              Pointer to the random values for denoising
 * @param pSparseTransposedEnd Pointer to the end indices of transposed sparse matrix rows
 * @param pSparseTransposedIndex Pointer to the column indices of transposed sparse matrix
 */
__global__ void kCalculateSparseTransposedDenoisedMatrix_kernel(uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex)
{
    __shared__ NNFloat sharedRandom[THREADS_PER_BLOCK];
    __shared__ uint32_t sharedIndex[THREADS_PER_BLOCK];

    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t laneId = threadIdx.x & cData._warpMask;
    const uint32_t bpos = tid / cData._warpSize;

    if (bpos < cData._batch)
    {
        const uint32_t position = cData._bShuffleIndices ? cData._pShuffleIndex[tid] : tid;
        uint64_t start = pSparseStart[position] + laneId;
        uint64_t end = pSparseEnd[position];

        sharedRandom[threadIdx.x] = pRandom[tid];
        sharedIndex[threadIdx.x] = pSparseIndex[tid];

        __syncthreads();

        for (; start < end; start += THREADS_PER_BLOCK)
        {
            NNFloat4 rnd4 = reinterpret_cast<NNFloat4*>(sharedRandom)[threadIdx.x];
            uint32_t4 index4 = reinterpret_cast<uint32_t4*>(sharedIndex)[threadIdx.x];

            if (rnd4.x >= cData._denoising_p)
            {
                atomicAdd(&pSparseTransposedEnd[index4.x], 1);
                uint32_t opos = atomicAdd(&pSparseTransposedEnd[index4.y], 1);
                pSparseTransposedIndex[opos] = bpos;
            }

            if (rnd4.y >= cData._denoising_p)
            {
                atomicAdd(&pSparseTransposedEnd[index4.z], 1);
                uint32_t opos = atomicAdd(&pSparseTransposedEnd[index4.w], 1);
                pSparseTransposedIndex[opos] = bpos;
            }
        }

        __syncthreads();
    }
}
/**
 * @brief CUDA kernel to calculate the weighted sparse transposed denoised matrix.
 *
 * @param position          Starting position in the sparse matrix.
 * @param batch             Number of batches to process.
 * @param pSparseStart      Start indices of the sparse matrix.
 * @param pSparseEnd        End indices of the sparse matrix.
 * @param pSparseIndex      Indices of the sparse matrix.
 * @param pDataWeight       Weight data for each position in the matrix.
 * @param pRandom           Random values for each position in the matrix.
 * @param pSparseTransposedEnd  End indices of the transposed sparse matrix.
 * @param pSparseTransposedIndex  Indices of the transposed sparse matrix.
 * @param pSparseTransposedData   Data of the transposed sparse matrix.
 */
__global__ void LAUNCH_BOUNDS()
kCalculateWeightedSparseTransposedDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, NNFloat *pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        position = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;

        uint64_t start = pSparseStart[position] + tgx;
        uint64_t end = pSparseEnd[position];
        NNFloat w = cData._denoising_q * pDataWeight[position];

        __shared__ uint32_t sCounter[cData._warpSize];
        sCounter[threadIdx.x] = 0;
        __syncthreads();

        __shared__ NNFloat sRandom[cData._warpSize];
        __shared__ uint32_t sSparseIndex[cData._warpSize];
        for (uint32_t i = threadIdx.x; i < cData._warpSize; i += blockDim.x)
        {
            sRandom[i] = pRandom[start + i];
            sSparseIndex[i] = pSparseIndex[start + i];
        }
        __syncthreads();

        while (start < end)
        {
            #pragma unroll 4
            for (int i = 0; i < 4; ++i)
            {
                uint32_t currentIndex = start + i * cData._warpSize;
                if (currentIndex < end)
                {
                    NNFloat rnd = sRandom[i];
                    uint32_t index = sSparseIndex[i];

                    if (rnd >= cData._denoising_p)
                    {
                        uint32_t opos = atomicAdd(&sCounter[index], 1);
                        pSparseTransposedIndex[opos] = bpos;
                        pSparseTransposedData[opos] = w;
                    }
                }
            }
            start += cData._warpSize * 4;
        }

        __syncthreads();

        if (tgx == 0)
        {
            for (uint32_t i = 0; i < cData._warpSize; ++i)
            {
                uint32_t index = sSparseIndex[i];
                atomicAdd(&pSparseTransposedEnd[index], sCounter[i]);
            }
        }
    }
}

/**
 * @brief Calculate the sparse transposed denoised matrix.
 *
 * This function calculates the transposed denoised matrix based on the given sparse matrix data.
 *
 * @param position The position parameter.
 * @param batch The batch size.
 * @param pSparseStart Pointer to the start of the sparse matrix.
 * @param pSparseEnd Pointer to the end of the sparse matrix.
 * @param pSparseIndex Pointer to the index data of the sparse matrix.
 * @param pDataWeight Pointer to the weight data.
 * @param pRandom Pointer to the random data.
 * @param pSparseTransposedEnd Pointer to the end of the transposed sparse matrix.
 * @param pSparseTransposedIndex Pointer to the index data of the transposed sparse matrix.
 * @param pSparseTransposedData Pointer to the data of the transposed sparse matrix.
 */
void kCalculateSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, NNFloat* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData)
{
    uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
    cudaStream_t stream = getGpu()._cudaStream;

    if (pDataWeight == nullptr)
    {
        /**
         * @brief Launch the kernel to calculate the sparse transposed denoised matrix.
         *
         * @param blocks The number of blocks to launch.
         * @param getGpu()._threadsPerBlock The number of threads per block.
         * @param stream The CUDA stream to use for execution.
         */
        kCalculateSparseTransposedDenoisedMatrix_kernel<<<blocks, getGpu()._threadsPerBlock, 0, stream>>>(position, batch, pSparseStart, pSparseEnd, pSparseIndex, pRandom, pSparseTransposedEnd, pSparseTransposedIndex);
        checkCudaErrors(cudaGetLastError());
    }
    else
    {
        /**
         * @brief Launch the kernel to calculate the weighted sparse transposed denoised matrix.
         *
         * @param blocks The number of blocks to launch.
         * @param getGpu()._threadsPerBlock The number of threads per block.
         * @param stream The CUDA stream to use for execution.
         */
        kCalculateWeightedSparseTransposedDenoisedMatrix_kernel<<<blocks, getGpu()._threadsPerBlock, 0, stream>>>(position, batch, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pRandom, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);
        checkCudaErrors(cudaGetLastError());
    }
}

/**
 * @brief CUDA kernel to calculate the indexed sparse transposed denoised matrix.
 *
 * @param position Starting position in the index array.
 * @param batch Number of batches to process.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the start positions of the sparse matrix.
 * @param pSparseEnd Pointer to the end positions of the sparse matrix.
 * @param pSparseIndex Pointer to the sparse matrix indices.
 * @param pRandom Pointer to the random values array.
 * @param pSparseTransposedEnd Pointer to the end positions of the transposed sparse matrix.
 * @param pSparseTransposedIndex Pointer to the transposed sparse matrix indices.
 */
__global__ void kCalculateIndexedSparseTransposedDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat *pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];

        uint64_t start = pSparseStart[position] + tgx;
        uint64_t end = pSparseEnd[position];

        __shared__ uint32_t sharedSparseIndex[THREADS_PER_BLOCK];
        if (tgx < THREADS_PER_BLOCK)
            sharedSparseIndex[tgx] = pSparseIndex[start + tgx];

        __syncthreads();

        uint32_t denoisedCount = 0;

        while (start < end)
        {
            NNFloat rnd = pRandom[start];
            uint32_t index = sharedSparseIndex[tgx];

            if (rnd >= cData._denoising_p)
            {
                uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos] = bpos;
                denoisedCount++;
            }

            start += cData._warpSize;

            if (tgx < THREADS_PER_BLOCK)
                sharedSparseIndex[tgx] = pSparseIndex[start + tgx];

            __syncthreads();
        }

        atomicAdd(&pSparseTransposedEnd[position], denoisedCount);
    }
}
/**
 * @brief GPU kernel for calculating the indexed weighted sparse transposed denoised matrix.
 *
 * @param position              The position parameter.
 * @param batch                 The batch parameter.
 * @param pIndex                Pointer to the index array.
 * @param pSparseStart          Pointer to the sparse start array.
 * @param pSparseEnd            Pointer to the sparse end array.
 * @param pSparseIndex          Pointer to the sparse index array.
 * @param pDataWeight           Pointer to the data weight array.
 * @param pRandom               Pointer to the random array.
 * @param pSparseTransposedEnd  Pointer to the sparse transposed end array.
 * @param pSparseTransposedIndex  Pointer to the sparse transposed index array.
 * @param pSparseTransposedData  Pointer to the sparse transposed data array.
 */
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedWeightedSparseTransposedDenoisedMatrix_kernel(
  uint32_t position, uint32_t batch, 
  uint32_t* __restrict__ pIndex, 
  uint64_t* __restrict__ pSparseStart, 
  uint64_t* __restrict__ pSparseEnd, 
  uint32_t* __restrict__ pSparseIndex, 
  NNFloat* __restrict__ pDataWeight, 
  NNFloat* __restrict__ pRandom, 
  uint32_t* __restrict__ pSparseTransposedEnd, 
  uint32_t* __restrict__ pSparseTransposedIndex, 
  NNFloat* __restrict__ pSparseTransposedData)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;
    
    __shared__ NNFloat shared_w[BLOCK_SIZE];
    __shared__ NNFloat shared_rnd[BLOCK_SIZE];
    __shared__ uint32_t shared_index[BLOCK_SIZE];

    if (bpos < batch)
    {
        position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];    
        uint64_t start = pSparseStart[position] + tgx;
        uint64_t end = pSparseEnd[position];
        shared_w[threadIdx.x] = cData._denoising_q * pDataWeight[position];
        while (start < end)
        {
            shared_rnd[threadIdx.x] = pRandom[start];
            shared_index[threadIdx.x] = pSparseIndex[start];
            __syncthreads();
            if (shared_rnd[threadIdx.x] >= cData._denoising_p)
            {
                uint32_t opos = atomicAdd(&pSparseTransposedEnd[shared_index[threadIdx.x]], 1);
                pSparseTransposedIndex[opos] = bpos;
                pSparseTransposedData[opos] = shared_w[threadIdx.x];
            }
            start += cData._warpSize;                   
        }
    }
}
/**
 * @brief Calculates the indexed sparse transposed denoised matrix.
 *
 * @param position               The position parameter.
 * @param batch                  The batch parameter.
 * @param pIndex                 Pointer to the index data.
 * @param pSparseStart           Pointer to the sparse start data.
 * @param pSparseEnd             Pointer to the sparse end data.
 * @param pSparseIndex           Pointer to the sparse index data.
 * @param pDataWeight            Pointer to the data weight.
 * @param pRandom                Pointer to the random data.
 * @param pSparseTransposedEnd   Pointer to the sparse transposed end data.
 * @param pSparseTransposedIndex Pointer to the sparse transposed index data.
 * @param pSparseTransposedData  Pointer to the sparse transposed data.
 */
void kCalculateIndexedSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, NNFloat* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData)
{
    const uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    const uint32_t warpSize = getGpu()._warpSize;
    uint32_t blocks = CalculateBlocks(batch * warpSize);
    
    if (pDataWeight == nullptr)
    {
        kCalculateIndexedSparseTransposedDenoisedMatrix_kernel<<<blocks, threadsPerBlock>>>(position, batch, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pRandom, pSparseTransposedEnd, pSparseTransposedIndex);
        LAUNCHERROR("kCalculateIndexedSparseTransposedDenoisedMatrix_kernel");
    }
    else
    {
        kCalculateIndexedWeightedSparseTransposedDenoisedMatrix_kernel<<<blocks, threadsPerBlock>>>(position, batch, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pRandom, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);
        LAUNCHERROR("kCalculateIndexedWeightedSparseTransposedDenoisedMatrix_kernel");
    }
}
/**
 * @brief CUDA kernel for calculating the transposed analog matrix for sparse data.
 *
 * @param position          The starting position in the sparse data arrays.
 * @param batch             The number of batches to process.
 * @param pSparseStart      Pointer to the start indices of sparse data.
 * @param pSparseEnd        Pointer to the end indices of sparse data.
 * @param pSparseIndex      Pointer to the indices of sparse data.
 * @param pDataWeight       Pointer to the weight data (optional).
 * @param pSparseData       Pointer to the sparse data values.
 * @param pSparseTransposedEnd   Pointer to the end indices of transposed sparse data.
 * @param pSparseTransposedIndex Pointer to the indices of transposed sparse data.
 * @param pSparseTransposedData  Pointer to the transposed sparse data values.
 */
__global__ void kCalculateSparseTransposedAnalogMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData)
{
    const uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    const uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        position = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        const uint64_t start = pSparseStart[position] + tgx;
        const uint64_t end = pSparseEnd[position];
        const NNFloat w = (pDataWeight != NULL) ? pDataWeight[position] : (NNFloat)1.0;

        __shared__ uint32_t sCount[cData._warpSize];
        __shared__ uint32_t sIndex[cData._warpSize];
        __shared__ NNFloat sData[cData._warpSize];

        uint64_t i = start;
        while (i < end)
        {
            sIndex[threadIdx.x] = pSparseIndex[i];
            sData[threadIdx.x] = w * pSparseData[i];

            uint32_t count = 1;
            for (uint32_t stride = cData._warpSize / 2; stride > 0; stride >>= 1)
            {
                __syncthreads();
                if (tgx < stride)
                {
                    const uint32_t otherIdx = threadIdx.x + stride;
                    if (i + stride < end && sIndex[threadIdx.x] == sIndex[otherIdx])
                    {
                        sData[threadIdx.x] += sData[otherIdx];
                        count += 1;
                    }
                }
            }

            __syncthreads();
            if (tgx == 0)
            {
                const uint32_t opos = atomicAdd(&pSparseTransposedEnd[sIndex[0]], count);
                pSparseTransposedIndex[opos] = bpos;
                pSparseTransposedData[opos] = sData[0];
            }

            i += cData._warpSize;
        }
    }
}

/**
 * @brief Calculate sparse transposed analog matrix.
 *
 * This function calculates the sparse transposed analog matrix using CUDA.
 *
 * @tparam T Type of the sparse data.
 * @param position The position.
 * @param batch The batch size.
 * @param pSparseStart Pointer to the start of the sparse matrix.
 * @param pSparseEnd Pointer to the end of the sparse matrix.
 * @param pSparseIndex Pointer to the sparse indices.
 * @param pDataWeight Pointer to the weight data.
 * @param pSparseData Pointer to the sparse data.
 * @param pSparseTransposedEnd Pointer to the end of the transposed sparse matrix.
 * @param pSparseTransposedIndex Pointer to the transposed sparse indices.
 * @param pSparseTransposedData Pointer to the transposed sparse data.
 */
template<typename T>
void kCalculateSparseTransposedAnalogMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData)
{
    constexpr uint32_t warpSize = getGpu()._warpSize;
    constexpr uint32_t threadsPerBlock = getGpu()._threadsPerBlock;

    dim3 blocks(batch, 1, 1);
    dim3 threads(threadsPerBlock, 1, 1);

    kCalculateSparseTransposedAnalogMatrix_kernel<<<blocks, threads>>>(position, batch, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);
}

/**
 * @brief CUDA kernel to calculate the transposed analog matrix for indexed sparse data.
 *
 * @param pIndex Array of indices for data positions.
 * @param pSparseStart Array of start indices for each position in the sparse data.
 * @param pSparseEnd Array of end indices for each position in the sparse data.
 * @param pSparseIndex Array of indices for the sparse data.
 * @param pDataWeight Array of weights for each position in the data.
 * @param pSparseData Array of values for each position in the sparse data.
 * @param pSparseTransposedEnd Array of end indices for each index in the transposed matrix.
 * @param pSparseTransposedIndex Array of indices for the transposed matrix.
 * @param pSparseTransposedData Array of values for the transposed matrix.
 */
__global__ void kCalculateIndexedSparseTransposedAnalogMatrix_kernel(uint32_t* pIndex,
                                                                     uint64_t* pSparseStart,
                                                                     uint64_t* pSparseEnd,
                                                                     uint32_t* pSparseIndex,
                                                                     NNFloat* pDataWeight,
                                                                     NNFloat* pSparseData,
                                                                     uint32_t* pSparseTransposedEnd,
                                                                     uint32_t* pSparseTransposedIndex,
                                                                     NNFloat* pSparseTransposedData)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t position = tid; position < batch; position += stride)
    {
        const uint32_t bpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position] : position];
        const uint64_t start = pSparseStart[bpos] + threadIdx.x;
        const uint64_t end = pSparseEnd[bpos];
        const NNFloat w = (pDataWeight != nullptr) ? pDataWeight[bpos] : 1.0;

        for (uint64_t i = start; i < end; i += blockDim.x)
        {
            const uint32_t index = pSparseIndex[i];
            const NNFloat value = pSparseData[i];
            const uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos] = position;
            pSparseTransposedData[opos] = w * value;
        }
    }
}

/**
 * @brief Calculates the transposed analog matrix for indexed sparse data.
 *
 * This function calculates the transposed analog matrix for indexed sparse data
 * using a CUDA kernel. It takes the position, batch size, index, sparse start,
 * sparse end, sparse index, data weight, sparse data, sparse transposed end,
 * sparse transposed index, and sparse transposed data as input parameters.
 *
 * @tparam T The data type of the sparse data.
 * @param position The position.
 * @param batch The batch size.
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
void kCalculateIndexedSparseTransposedAnalogMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData)
{
    uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
    kCalculateIndexedSparseTransposedAnalogMatrix_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);
    LAUNCHERROR("kCalculateIndexedSparseTransposedAnalogMatrix_kernel");
}
/**
 * @brief CUDA kernel for calculating the sparse transposed analog denoised matrix.
 *
 * @tparam T The data type of the sparse data.
 *
 * @param position The starting position in the sparse data for the current thread.
 * @param batch The number of batches to process.
 * @param pSparseStart Array containing the start positions of each batch in the sparse data.
 * @param pSparseEnd Array containing the end positions of each batch in the sparse data.
 * @param pSparseIndex Array containing the indices of the sparse data.
 * @param pDataWeight Array containing the weight values for each batch (optional, can be NULL).
 * @param pSparseData Array containing the sparse data values.
 * @param pRandom Array containing random values.
 * @param pSparseTransposedEnd Array containing the end positions of the transposed sparse data.
 * @param pSparseTransposedIndex Array containing the indices of the transposed sparse data.
 * @param pSparseTransposedData Array containing the transposed sparse data values.
 */
template<typename T>
__global__ void kCalculateSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, T* pSparseData, NNFloat *pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start = pSparseStart[position] + tgx;
        uint64_t end = pSparseEnd[position];
        NNFloat w = (pDataWeight != NULL) ? pDataWeight[position] : (NNFloat)1.0;
        __shared__ NNFloat sDataWeight[BLOCK_SIZE];
        if (threadIdx.x < BLOCK_SIZE)
            sDataWeight[threadIdx.x] = w;
        __syncthreads();

        for (; start < end; start += cData._warpSize)
        {
            NNFloat rnd = pRandom[start];
            uint32_t index = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                T value = pSparseData[start];
                uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos] = bpos;
                pSparseTransposedData[opos] = sDataWeight[threadIdx.x & cData._warpMask] * value;
            }
        }
    }
}
/**
 * @brief CUDA kernel for calculating the sparse transposed analog denoised matrix.
 * 
 * @tparam position   The position parameter type.
 * @tparam batch      The batch parameter type.
 * @param position    The starting position.
 * @param batch       The batch size.
 * @param pSparseStart    Pointer to the start indices of sparse data.
 * @param pSparseEnd      Pointer to the end indices of sparse data.
 * @param pSparseIndex    Pointer to the sparse indices.
 * @param pDataWeight     Pointer to the data weight.
 * @param pSparseData     Pointer to the sparse data.
 * @param pRandom         Pointer to the random values.
 * @param pSparseTransposedEnd       Pointer to the end indices of the transposed sparse data.
 * @param pSparseTransposedIndex     Pointer to the transposed sparse indices.
 * @param pSparseTransposedData      Pointer to the transposed sparse data.
 */
__global__ void kCalculateSparseTransposedAnalogDenoisedMatrix_kernel(
    uint32_t position,
    uint32_t batch,
    uint64_t* pSparseStart,
    uint64_t* pSparseEnd,
    uint32_t* pSparseIndex,
    NNFloat* pDataWeight,
    unsigned char* pSparseData,
    NNFloat* pRandom,
    uint32_t* pSparseTransposedEnd,
    uint32_t* pSparseTransposedIndex,
    NNFloat* pSparseTransposedData)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        position = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start = pSparseStart[position] + tgx;
        uint64_t end = pSparseEnd[position];
        NNFloat w = (pDataWeight != NULL) ? pDataWeight[position] : (NNFloat)1.0;

        __shared__ NNFloat sDataWeight[cData._warpSize];
        if (threadIdx.x < cData._warpSize && pDataWeight != NULL)
            sDataWeight[threadIdx.x] = pDataWeight[position];

        while (start < end)
        {
            NNFloat rnd = pRandom[start];
            uint32_t index = pSparseIndex[start];

            if (rnd >= cData._denoising_p)
            {
                NNFloat value = static_cast<NNFloat>(pSparseData[start]) * (NNFloat)(1.0 / 256.0);
                uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);

                NNFloat dataWeight = (pDataWeight != NULL) ? sDataWeight[threadIdx.x] : w;

                pSparseTransposedIndex[opos] = bpos;
                pSparseTransposedData[opos] = dataWeight * value;
            }

            start += cData._warpSize;
        }
    }
}

/**
 * @brief CUDA kernel for calculating the sparse transposed analog denoised matrix.
 * 
 * @tparam T The data type of the input arrays.
 * @param position The starting position in the sparse matrix.
 * @param batch The number of batches to process.
 * @param pSparseStart Pointer to the start indices of the sparse matrix.
 * @param pSparseEnd Pointer to the end indices of the sparse matrix.
 * @param pSparseIndex Pointer to the indices of the sparse matrix.
 * @param pDataWeight Pointer to the weight data for each position.
 * @param pSparseData Pointer to the data of the sparse matrix.
 * @param pRandom Pointer to the random values used for denoising.
 * @param pSparseTransposedEnd Pointer to the end indices of the transposed sparse matrix.
 * @param pSparseTransposedIndex Pointer to the indices of the transposed sparse matrix.
 * @param pSparseTransposedData Pointer to the data of the transposed sparse matrix.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, char* pSparseData, NNFloat *pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        position = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start = pSparseStart[position] + tgx;
        uint64_t end = pSparseEnd[position];
        NNFloat w = (pDataWeight != NULL) ? pDataWeight[position] : (NNFloat)1.0;
        uint32_t warpSize = cData._warpSize;
        NNFloat valueScale = 1.0 / 128.0;

        while (start < end)
        {
            NNFloat rnd = pRandom[start];
            uint32_t index = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                NNFloat value = static_cast<NNFloat>(pSparseData[start]) * valueScale;
                uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos] = bpos;
                pSparseTransposedData[opos] = w * value;
            }
            start += warpSize;
        }
    }
}

/**
 * @brief Calculates the sparse transposed analog denoised matrix.
 *
 * @tparam T The data type of the sparse data.
 * @param position The position parameter.
 * @param batch The batch parameter.
 * @param pSparseStart Pointer to the start of the sparse data.
 * @param pSparseEnd Pointer to the end of the sparse data.
 * @param pSparseIndex Pointer to the index of the sparse data.
 * @param pDataWeight Pointer to the data weight.
 * @param pSparseData Pointer to the sparse data.
 * @param pRandom Pointer to the random data.
 * @param pSparseTransposedEnd Pointer to the end of the transposed sparse data.
 * @param pSparseTransposedIndex Pointer to the index of the transposed sparse data.
 * @param pSparseTransposedData Pointer to the transposed sparse data.
 */
template<typename T>
void kCalculateSparseTransposedAnalogDenoisedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, T* pSparseData, NNFloat* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData)
{
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (batch * getGpu()._warpSize + threadsPerBlock - 1) / threadsPerBlock;

    kCalculateSparseTransposedAnalogDenoisedMatrix_kernel<<<blocks, threadsPerBlock>>>(position, batch, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pRandom, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);

    LAUNCHERROR("kCalculateSparseTransposedAnalogDenoisedMatrix_kernel");
}
/**
 * @brief Kernel function to calculate indexed sparse transposed analog denoised matrix.
 *
 * @param position          The starting position for processing.
 * @param batch             The number of elements to process.
 * @param pIndex            Pointer to the index array.
 * @param pSparseStart      Pointer to the start positions of sparse data.
 * @param pSparseEnd        Pointer to the end positions of sparse data.
 * @param pSparseIndex      Pointer to the index array of sparse data.
 * @param pDataWeight       Pointer to the weight data.
 * @param pSparseData       Pointer to the sparse data.
 * @param pRandom           Pointer to the random data.
 * @param pSparseTransposedEnd    Pointer to the end positions of the transposed sparse data.
 * @param pSparseTransposedIndex  Pointer to the index array of the transposed sparse data.
 * @param pSparseTransposedData   Pointer to the transposed sparse data.
 */
__global__ void LAUNCH_BOUNDS_kCalculateIndexedSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, NNFloat* pSparseData, NNFloat* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData)
{
    const uint32_t bpos = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        const uint64_t start = pSparseStart[position] + tgx;
        const uint64_t end = pSparseEnd[position];
        const NNFloat w = (pDataWeight != NULL) ? pDataWeight[position] : (NNFloat)1.0;

        __shared__ NNFloat sDataWeight[blockDim.x];
        __shared__ NNFloat sRandom[blockDim.x];
        sDataWeight[threadIdx.x] = w;
        sRandom[threadIdx.x] = pRandom[start];

        __syncthreads();

        uint32_t opos = 0;
        NNFloat value = 0.0;

        while (start < end)
        {
            const NNFloat rnd = sRandom[threadIdx.x];
            const uint32_t index = pSparseIndex[start];

            if (rnd >= cData._denoising_p)
            {
                value = pSparseData[start];
                opos = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos] = bpos;
                pSparseTransposedData[opos] = sDataWeight[threadIdx.x] * value;
            }

            start += cData._warpSize;

            // Load next pRandom value into shared memory
            sRandom[threadIdx.x] = pRandom[start];

            __syncthreads();
        }
    }
}
/**
 * @brief Kernel function to calculate the indexed sparse transposed analog denoised matrix.
 *
 * @param position        Starting position in the index array.
 * @param batch           Number of elements to process.
 * @param pIndex          Index array.
 * @param pSparseStart    Array containing the starting positions of sparse data.
 * @param pSparseEnd      Array containing the ending positions of sparse data.
 * @param pSparseIndex    Array containing the sparse indices.
 * @param pDataWeight     Array containing the data weights (optional).
 * @param pSparseData     Array containing the sparse data.
 * @param pRandom         Array containing random values.
 * @param pSparseTransposedEnd     Array containing the ending positions of the transposed sparse data.
 * @param pSparseTransposedIndex   Array containing the transposed sparse indices.
 * @param pSparseTransposedData    Array containing the transposed sparse data.
 */
__global__ void kCalculateIndexedSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, unsigned char* pSparseData, NNFloat *pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < batch)
    {
        uint32_t bpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + tid] : position + tid];
        uint64_t start = pSparseStart[bpos] + threadIdx.x;
        uint64_t end = pSparseEnd[bpos];
        NNFloat w = (pDataWeight != NULL) ? pDataWeight[bpos] : (NNFloat)1.0;

        __shared__ uint32_t sCounter[cData._warpSize];
        __shared__ uint32_t sIndices[cData._warpSize];
        __shared__ NNFloat sData[cData._warpSize];

        if (threadIdx.x < cData._warpSize)
        {
            sCounter[threadIdx.x] = 0;
        }
        __syncthreads();

        while (start < end)
        {
            NNFloat rnd = pRandom[start];
            uint32_t index = pSparseIndex[start];

            if (rnd >= cData._denoising_p)
            {
                NNFloat value = static_cast<NNFloat>(pSparseData[start]) * static_cast<NNFloat>(1.0 / 256.0);
                uint32_t opos = atomicAdd(&sCounter[index & cData._warpMask], 1);

                if (opos < cData._warpSize)
                {
                    sIndices[opos] = tid;
                    sData[opos] = w * value;
                }
            }

            start += blockDim.x;
        }

        __syncthreads();

        for (uint32_t i = threadIdx.x; i < cData._warpSize; i += blockDim.x)
        {
            uint32_t index = (bpos * cData._warpSize) + i;
            uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], sCounter[i]);

            if (opos < pSparseTransposedEnd[index])
            {
                for (uint32_t j = 0; j < sCounter[i]; ++j)
                {
                    pSparseTransposedIndex[opos + j] = sIndices[(i * cData._warpSize) + j];
                    pSparseTransposedData[opos + j] = sData[(i * cData._warpSize) + j];
                }
            }
        }
    }
}
/**
 * @brief CUDA kernel for calculating indexed sparse transposed analog denoised matrix.
 *
 * @tparam Unused Unused template parameter.
 * @param position Starting position.
 * @param batch Batch size.
 * @param pIndex Pointer to index array.
 * @param pSparseStart Pointer to sparse start array.
 * @param pSparseEnd Pointer to sparse end array.
 * @param pSparseIndex Pointer to sparse index array.
 * @param pDataWeight Pointer to data weight array.
 * @param pSparseData Pointer to sparse data array.
 * @param pRandom Pointer to random array.
 * @param pSparseTransposedEnd Pointer to sparse transposed end array.
 * @param pSparseTransposedIndex Pointer to sparse transposed index array.
 * @param pSparseTransposedData Pointer to sparse transposed data array.
 */
template<>
__global__ void kCalculateIndexedSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, char* pSparseData, NNFloat* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        uint64_t startPos = pSparseStart[position];
        uint64_t endPos = pSparseEnd[position];

        NNFloat w = (pDataWeight != NULL) ? pDataWeight[position] : (NNFloat)1.0;

        for (uint64_t start = startPos + tgx; start < endPos; start += cData._warpSize)
        {
            NNFloat rnd = pRandom[start];
            uint32_t index = pSparseIndex[start];

            if (rnd >= cData._denoising_p)
            {
                NNFloat value = static_cast<NNFloat>(pSparseData[start]) * (1.0 / 128.0);
                uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos] = bpos;
                pSparseTransposedData[opos] = w * value;
            }
        }
    }
}
/**
 * @brief Calculates the indexed sparse transposed analog denoised matrix.
 *
 * @tparam T The type of sparse data.
 * @param position The position value.
 * @param batch The batch size.
 * @param pIndex Pointer to the index.
 * @param pSparseStart Pointer to the start of the sparse data.
 * @param pSparseEnd Pointer to the end of the sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @param pSparseData Pointer to the sparse data.
 * @param pRandom Pointer to the random values.
 * @param pSparseTransposedEnd Pointer to the end of the sparse transposed data.
 * @param pSparseTransposedIndex Pointer to the sparse transposed index.
 * @param pSparseTransposedData Pointer to the sparse transposed data.
 */
template<typename T>
void kCalculateIndexedSparseTransposedAnalogDenoisedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, T* pSparseData, NNFloat* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData)
{
    uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
    kCalculateIndexedSparseTransposedAnalogDenoisedMatrix_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pRandom, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);
    LAUNCHERROR("kCalculateIndexedSparseTransposedAnalogDenoisedMatrix_kernel");
}
/**
 * @brief CUDA kernel for calculating sparse transposed weight gradients.
 *
 * @param alpha               Scaling factor for the weight gradient update.
 * @param beta                Scaling factor for the previous weight gradient.
 * @param n                   Number of weight gradient elements.
 * @param pSparseTransposedStart  Pointer to the start indices of the sparse transposed data.
 * @param pSparseTransposedEnd    Pointer to the end indices of the sparse transposed data.
 * @param pSparseTransposedIndex  Pointer to the indices of the sparse transposed data.
 * @param pDelta              Pointer to the delta values.
 * @param pWeightGradient     Pointer to the weight gradients.
 */
__global__ void LAUNCH_BOUNDS256()
kCalculateSparseTransposedWeightGradient_kernel(NNFloat alpha, NNFloat beta, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pDelta, NNFloat* pWeightGradient)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSE];
    __shared__ uint32_t sStart;
    __shared__ uint32_t sEnd;

    // Load start and end indices into shared memory
    if (threadIdx.x == 0)
    {
        sStart = pSparseTransposedStart[blockIdx.x];
        sEnd = pSparseTransposedEnd[blockIdx.x];
    }
    __syncthreads();

    uint64_t start = sStart;
    uint64_t end = sEnd;
    alpha *= cData._denoising_q;
    pWeightGradient += blockIdx.x * n;

    do
    {
        sOpos = blockDim.x;

        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSE);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            uint32_t index = pSparseTransposedIndex[tstart];
            sOffset[pos] = index * n;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __syncthreads();

        uint32_t opos = threadIdx.x;
        uint32_t tgx = threadIdx.x & cData._warpMask;

        NNFloat* sWeightGradient = &pWeightGradient[opos];
        NNFloat oldgradient = (beta == (NNFloat)0.0) ? (NNFloat)0.0 : beta * sWeightGradient[0];

        while (opos < n)
        {
            int64_t sum = 0;

            for (uint32_t i = 0; i < inputs; i++)
            {
                uint32_t offset = sOffset[i];
                sum += llrintf(ERRORSCALEF * pDelta[offset + opos]);
            }

            NNFloat fsum = alpha * (NNFloat)((double)sum * ONEOVERERRORSCALE);
            sWeightGradient[0] = oldgradient + fsum;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
            opos += tgx;
            sWeightGradient += opos;
        }

        start = tend;

        if (start < end)
        {
            __syncthreads();
            beta = (NNFloat)1.0;
        }
    } while (start < end);
}
/**
 * @brief CUDA kernel function to calculate sparse transposed analog weight gradient.
 *
 * @param alpha Scaling factor for the weight gradient update.
 * @param beta Coefficient to scale the existing weight gradient.
 * @param n Number of weights in a single weight matrix.
 * @param pSparseTransposedStart Array containing the starting positions of each sparse transposed block.
 * @param pSparseTransposedEnd Array containing the ending positions of each sparse transposed block.
 * @param pSparseTransposedIndex Array containing the indices of the sparse transposed elements.
 * @param pSparseTransposedData Array containing the values of the sparse transposed elements.
 * @param pDelta Array containing the delta values.
 * @param pWeightGradient Array containing the weight gradients.
 */
__global__ void LAUNCH_BOUNDS256()
kCalculateSparseTransposedAnalogWeightGradient_kernel(NNFloat alpha, NNFloat beta, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData, NNFloat* pDelta, NNFloat* pWeightGradient)
{
    const uint32_t warpSize = 32;
    const uint32_t maxSparseAnalog = ...;

    uint32_t warpIdx = threadIdx.x / warpSize;
    uint32_t laneIdx = threadIdx.x % warpSize;

    uint64_t start = pSparseTransposedStart[blockIdx.x];
    uint64_t end = pSparseTransposedEnd[blockIdx.x];
    alpha *= cData._denoising_q;
    pWeightGradient += blockIdx.x * n;

    while (start < end)
    {
        uint32_t inputs = min(static_cast<uint32_t>(end - start), maxSparseAnalog);
        uint64_t tend = start + inputs;

        for (uint32_t i = warpIdx; i < inputs; i += warpSize)
        {
            uint32_t offset = pSparseTransposedIndex[start + i] * n;
            NNFloat value = pSparseTransposedData[start + i];
            sOffset[i] = offset;
            sValue[i] = value;
        }

        __syncthreads();

        for (uint32_t opos = threadIdx.x; opos < n; opos += blockDim.x)
        {
            NNFloat oldgradient = (beta == 0.0) ? 0.0 : beta * pWeightGradient[opos];
            int64_t sum = 0;

            for (uint32_t i = 0; i < inputs; i++)
            {
                uint32_t offset = sOffset[i];
                NNFloat value = sValue[i];
                sum += llrintf(ERRORSCALEF * value * pDelta[offset + opos]);
            }

            NNFloat fsum = alpha * static_cast<NNFloat>(sum) * ONEOVERERRORSCALE;
            pWeightGradient[opos] = oldgradient + fsum;
        }

        start = tend;

        if (start < end)
            __syncthreads();
    }
}

/**
 * @brief Calculate sparse transposed analog weight gradient.
 *
 * @param alpha Scaling factor for the weight gradient update.
 * @param beta Coefficient to scale the existing weight gradient.
 * @param m Number of sparse transposed blocks.
 * @param n Number of weights in a single weight matrix.
 * @param pSparseTransposedStart Array containing the starting positions of each sparse transposed block.
 * @param pSparseTransposedEnd Array containing the ending positions of each sparse transposed block.
 * @param pSparseTransposedIndex Array containing the indices of the sparse transposed elements.
 * @param pSparseTransposedData Array containing the values of the sparse transposed elements.
 * @param pDelta Array containing the delta values.
 * @param pWeightGradient Array containing the weight gradients.
 */
void kCalculateSparseTransposedAnalogWeightGradient(NNFloat alpha, NNFloat beta, uint32_t m, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData, NNFloat* pDelta, NNFloat* pWeightGradient)
{
    uint32_t threads = min(256, ((m + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);    
    kCalculateSparseTransposedAnalogWeightGradient_kernel<<<m, threads>>>(alpha, beta, n, pSparseTransposedStart, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData, pDelta, pWeightGradient);
    LAUNCHERROR("kCalculateSparseTransposedAnalogWeightGradient_kernel");
}

/**
 * @brief CUDA kernel function to update biases.
 *
 * @param alpha Scaling factor for the bias update.
 * @param batch Number of samples in a batch.
 * @param width Number of elements in the bias array.
 * @param pDelta Array containing the delta values.
 * @param pBias Array containing the bias values.
 */
__global__ void LAUNCH_BOUNDS()
kUpdateBiases_kernel(NNFloat alpha, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBias)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos < width)
    {
        NNFloat sum = 0.0;

        for (uint32_t i = 0; i < batch; i++)
        {
            sum += pDelta[i * width + pos];
        }

        pBias[pos] -= alpha * sum;
    }
}

/**
 * @brief Update biases.
 *
 * @param alpha Scaling factor for the bias update.
 * @param batch Number of samples in a batch.
 * @param width Number of elements in the bias array.
 * @param pDelta Array containing the delta values.
 * @param pBias Array containing the bias values.
 */
void kUpdateBiases(NNFloat alpha, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBias)
{
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = CalculateBlocks(width, threadsPerBlock);
    kUpdateBiases_kernel<<<blocks, threadsPerBlock>>>(alpha, batch, width, pDelta, pBias);
    LAUNCHERROR("kUpdateBiases_kernel");
}

/**
 * @brief CUDA kernel function to calculate regularization error.
 *
 * @param pWeight Array containing the weight values.
 * @param size Number of elements in the weight array.
 * @param lambda Regularization coefficient.
 * @param lambda1 Regularization coefficient for L1 regularization.
 */
__global__ void LAUNCH_BOUNDS()
kCalculateRegularizationError_kernel(NNFloat* pWeight, uint64_t size, NNFloat lambda, NNFloat lambda1)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    NNFloat error = 0.0;

    if (pos < size)
    {
        NNFloat w = pWeight[pos];
        error = lambda * w * w + lambda1 * fabs(w);
    }

    REDUCEERROR(error)
}

/**
 * @brief Calculate regularization error.
 *
 * @param lambda Regularization coefficient.
 * @param lambda1 Regularization coefficient for L1 regularization.
 * @param pWeight Array containing the weight values.
 * @param size Number of elements in the weight array.
 * @return Regularization error.
 */
NNFloat kCalculateRegularizationError(NNFloat lambda, NNFloat lambda1, NNFloat* pWeight, uint64_t size)
{
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = CalculateBlocks(size, threadsPerBlock);

    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    kCalculateRegularizationError_kernel<<<blocks, threadsPerBlock>>>(pWeight, size, 0.5 * lambda, lambda1);
    LAUNCHERROR("kCalculateRegularizationError_kernel");

    getGpu()._pbAccumulator->Download();

    return static_cast<NNFloat>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE;
}

/**
 * @brief Compute the signum function.
 *
 * @param x Input value.
 * @return Sign of the input value (-1.0 or 1.0).
 */
__device__ inline NNFloat sgn(NNFloat x) {
    return (x >= 0) ? 1.0f : -1.0f;
}

/**
 * @brief CUDA kernel to update weights using SGD (Stochastic Gradient Descent).
 *
 * @param alpha Learning rate.
 * @param lambda L2 regularization parameter.
 * @param lambda1 L1 regularization parameter.
 * @param size Size of the weight vectors.
 * @param pWeightGradient Array of weight gradients.
 * @param pWeight Array of weights.
 */
__global__ void LAUNCH_BOUNDS_kSGDUpdateWeights_kernel(NNFloat alpha, NNFloat lambda, NNFloat lambda1, uint64_t size, NNFloat* pWeightGradient, NNFloat* pWeight) {
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pos < size) {
        NNFloat g = pWeightGradient[pos];
        NNFloat w = pWeight[pos];
        
        // Compute the weight update using SGD with L1 and L2 regularization
        NNFloat update = alpha * (g - lambda * w - lambda1 * sgn(w));
        
        // Update the weight
        pWeight[pos] = w + update;
    }
}

/**
 * @brief Update weights using Stochastic Gradient Descent (SGD).
 *
 * @param alpha Learning rate.
 * @param lambda L2 regularization parameter.
 * @param lambda1 L1 regularization parameter.
 * @param size Size of the weight vectors.
 * @param pWeightGradient Array of weight gradients.
 * @param pWeight Array of weights.
 */
void kSGDUpdateWeights(NNFloat alpha, NNFloat lambda, NNFloat lambda1, uint64_t size, NNFloat* pWeightGradient, NNFloat* pWeight) {
    uint32_t blocks = CalculateBlocks(size);
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    
    LAUNCH_BOUNDS_kSGDUpdateWeights_kernel<<<blocks, threadsPerBlock>>>(alpha, lambda, lambda1, size, pWeightGradient, pWeight);
    LAUNCHERROR("kSGDUpdateWeights_kernel");
}
/**
 * @brief CUDA kernel to update biases using Stochastic Gradient Descent (SGD).
 *
 * @param alpha Learning rate.
 * @param batch Number of samples in a batch.
 * @param width Width of the bias vector.
 * @param pDelta Array of delta values.
 * @param pBias Array of biases.
 */
__global__ void LAUNCH_BOUNDS_kSGDUpdateBiases_kernel(NNFloat alpha, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBias) {
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos < width) {
        NNFloat sum = 0.0f;
        pDelta += pos;

        for (uint32_t i = 0; i < batch; i++) {
            sum += *pDelta;
            pDelta += width;
        }
        sum /= static_cast<NNFloat>(batch);

        NNFloat bias = pBias[pos];
        pBias[pos] = bias - alpha * sum;
    }
}
/**
 * @brief Update biases using Stochastic Gradient Descent (SGD).
 *
 * @param alpha Learning rate.
 * @param batch Number of samples in a batch.
 * @param width Width of the bias vector.
 * @param pDelta Array of delta values.
 * @param pBias Array of biases.
 */
void kSGDUpdateBiases(NNFloat alpha, uint32_t batch, uint32_t width, NNFloat* pDelta, NNFloat* pBias) {
    uint32_t blocks = CalculateBlocks(width);
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;

    LAUNCH_BOUNDS_kSGDUpdateBiases_kernel<<<blocks, threadsPerBlock>>>(alpha, batch, width, pDelta, pBias);
    LAUNCHERROR("kSGDUpdateBiases_kernel");
}
