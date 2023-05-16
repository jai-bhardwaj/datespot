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
