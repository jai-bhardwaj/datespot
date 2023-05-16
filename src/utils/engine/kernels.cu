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