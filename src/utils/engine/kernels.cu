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
