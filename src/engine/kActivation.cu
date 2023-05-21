#include "GpuTypes.h"
#include "Types.h"
#include <limits>

static __constant__ GpuData cData;

/**
 * @brief Computes the atomic maximum of a floating-point value.
 *
 * @param address Pointer to the value to update.
 * @param val The value to compare and update.
 * @return The previous value before the update.
 */
__device__ inline float atomicMax(float* address, float val)
{
    int* address_as_i = reinterpret_cast<int*>(address);
    int old = *address_as_i;
    int assumed;
    do 
    {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } 
    while (assumed != old);
    return __int_as_float(old);
}

/**
 * @brief Copies GpuData to the constant memory symbol cData.
 */
void SetKActivationGpuData()
{
    cudaError_t status;
    status = cudaMemcpyToSymbol(cData, &(getGpu()._data), sizeof(GpuData));     
    RTERROR(status, "cudaMemcpyToSymbol: SetKActivationGpuData copy to cData failed");
}

/**
 * @brief Copies GpuData from the constant memory symbol cData.
 */
void GetKActivationGpuData()
{
    cudaError_t status;
    status = cudaMemcpyFromSymbol(&(getGpu()._data), cData, sizeof(GpuData));     
    RTERROR(status, "cudaMemcpyFromSymbol: GetKActivationGpuData copy From cData failed");
}

/**
 * @brief Kernel function to compute sigmoid activation for each element in pData.
 *
 * @param pData Pointer to the data array.
 * @param size Number of elements in the array.
 */
__global__ void LAUNCH_BOUNDS()
kCalculateSigmoidActivation_kernel(Float* pData, uint64_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        Float a = 1.0f / (1.0f + expf(-pData[pos]));
        pData[pos] = a;
    }
}

/**
 * @brief Computes sigmoid activation for the given data array on the GPU.
 *
 * @param pData Pointer to the data array.
 * @param size Number of elements in the array.
 */
void kCalculateSigmoidActivation(Float* pData, uint64_t size)
{
    uint32_t blocks = CalculateBlocks(size);
    kCalculateSigmoidActivation_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pData, size);
    LAUNCHERROR("kCalculateSigmoidActivation_kernel");
}

/**
 * @brief Kernel function to compute hyperbolic tangent activation for each element in pData.
 *
 * @param pData Pointer to the data array.
 * @param size Number of elements in the array.
 */
__global__ void LAUNCH_BOUNDS()
kCalculateTanhActivation_kernel(Float* pData, uint64_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
        pData[pos] = tanhf(pData[pos]);
}

/**
 * @brief Computes hyperbolic tangent activation for the given data array on the GPU.
 *
 * @param pData Pointer to the data array.
 * @param size Number of elements in the array.
 */
void kCalculateTanhActivation(Float* pData, uint64_t size)
{
    uint32_t blocks = CalculateBlocks(size);
    kCalculateTanhActivation_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pData, size);
    LAUNCHERROR("kCalculateTanhActivation_kernel");
}

/**
 * @brief Kernel function to compute ReLU activation for each element in pData.
 *
 * @param pData Pointer to the data array.
 * @param size Number of elements in the array.
 */
__global__ void LAUNCH_BOUNDS()
kCalculateRELUActivation_kernel(Float* pData, uint64_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
        pData[pos] = fmaxf(0.0f, pData[pos]);
}

/**
 * @brief Computes ReLU activation for the given data array on the GPU.
 *
 * @param pData Pointer to the data array.
 * @param size Number of elements in the array.
 */
void kCalculateRELUActivation(Float* pData, uint64_t size)
{
    uint32_t blocks = CalculateBlocks(size);
    kCalculateRELUActivation_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pData, size);
    LAUNCHERROR("kCalculateRELUActivation_kernel");
}

/**
 * @brief Kernel function to compute Leaky ReLU activation for each element in pData.
 *
 * @param pData Pointer to the data array.
 * @param size Number of elements in the array.
 * @param slope Slope of the activation function for negative input values.
 */
__global__ void LAUNCH_BOUNDS()
kCalculateLRELUActivation_kernel(Float* pData, uint64_t size, Float slope)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        Float val = pData[pos];
        pData[pos] = fmaxf(val, val * slope);
    }
}

/**
 * @brief Computes Leaky ReLU activation for the given data array on the GPU.
 *
 * @param pData Pointer to the data array.
 * @param size Number of elements in the array.
 * @param slope Slope of the activation function for negative input values.
 */
void kCalculateLRELUActivation(Float* pData, uint64_t size, Float slope)
{
    uint32_t blocks = CalculateBlocks(size);
    kCalculateLRELUActivation_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pData, size, slope);
    LAUNCHERROR("kCalculateLRELUActivation_kernel");
}

/**
 * @brief Kernel function to compute ELU activation for each element in pData.
 *
 * @param pData Pointer to the data array.
 * @param size Number of elements in the array.
 * @param alpha Alpha value for the activation function.
 */
__global__ void LAUNCH_BOUNDS()
kCalculateELUActivation_kernel(Float* pData, uint64_t size, Float alpha)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {   
        Float x = pData[pos];
        pData[pos] = (x > 0.0f) ? x : alpha * (expf(x) - 1.0f);
    }
}

/**
 * @brief Computes ELU activation for the given data array on the GPU.
 *
 * @param pData Pointer to the data array.
 * @param size Number of elements in the array.
 * @param alpha Alpha value for the activation function.
 */
void kCalculateELUActivation(Float* pData, uint64_t size, Float alpha)
{
    uint32_t blocks = CalculateBlocks(size);
    kCalculateELUActivation_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pData, size, alpha);
    LAUNCHERROR("kCalculateELUActivation_kernel");
}

/**
 * @brief Kernel function to compute SELU activation for each element in pData.
 *
 * @param pData Pointer to the data array.
 * @param size Number of elements in the array.
 * @param alpha Alpha value for the activation function.
 * @param lambda Lambda value for the activation function.
 */
__global__ void LAUNCH_BOUNDS()
kCalculateSELUActivation_kernel(Float* pData, uint64_t size, Float alpha, Float lambda)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {   
        Float x = pData[pos];
        pData[pos] = (x > 0.0f) ? lambda * x : lambda * alpha * (expf(x) - 1.0f);
    }
}

/**
 * @brief Computes SELU activation for the given data array on the GPU.
 *
 * @param pData Pointer to the data array.
 * @param size Number of elements in the array.
 * @param alpha Alpha value for the activation function.
 * @param lambda Lambda value for the activation function.
 */
void kCalculateSELUActivation(Float* pData, uint64_t size, Float alpha, Float lambda)
{
    uint32_t blocks = CalculateBlocks(size);
    kCalculateSELUActivation_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pData, size, alpha, lambda);
    LAUNCHERROR("kCalculateSELUActivation_kernel");
}

/**
 * @brief Kernel function to compute softmax activation for each element in pData with a given stride.
 *
 * @param pData Pointer to the data array.
 * @param stride The stride between elements in pData.
 */
__global__ void LAUNCH_BOUNDS()
kCalculateSoftMaxActivation_kernel(Float* pData, uint32_t stride)
{
    __shared__ unsigned long long int sAccumulator;
    __shared__ Float sMaxValue;

    if (threadIdx.x == 0)
    {
        sAccumulator = 0;
        sMaxValue = -std::numeric_limits<Float>::infinity();
    }
    __syncthreads();
    

    pData += blockIdx.x * stride;
    uint32_t pos = threadIdx.x;
    Float maxValue = -std::numeric_limits<Float>::infinity();
    
    while (pos < stride)
    {
        Float z = pData[pos];
        maxValue = fmaxf(z, maxValue);
        pos += blockDim.x;
    }
    
    uint32_t tgx = threadIdx.x & cData._warpMask;    
    maxValue = fmaxf(maxValue, __shfl_xor_sync(0xffffffff, maxValue, tgx ^ 1));
    maxValue = fmaxf(maxValue, __shfl_xor_sync(0xffffffff, maxValue, tgx ^ 2));
    maxValue = fmaxf(maxValue, __shfl_xor_sync(0xffffffff, maxValue, tgx ^ 4));
    maxValue = fmaxf(maxValue, __shfl_xor_sync(0xffffffff, maxValue, tgx ^ 8));
    maxValue = fmaxf(maxValue, __shfl_xor_sync(0xffffffff, maxValue, tgx ^ 16));

    if (tgx == 0) 
        atomicMax(&sMaxValue, maxValue);
    __syncthreads();        
    maxValue = sMaxValue;       

    pos = threadIdx.x;
    Float sum = 0.0f;
    while (pos < stride)
    {
        Float z = pData[pos];
        sum += expf(z - maxValue);
        pos += blockDim.x;
    }    
         
    sum += __shfl_xor_sync(0xffffffff, sum, tgx ^ 1);
    sum += __shfl_xor_sync(0xffffffff, sum, tgx ^ 2);
    sum += __shfl_xor_sync(0xffffffff, sum, tgx ^ 4);
    sum += __shfl_xor_sync(0xffffffff, sum, tgx ^ 8);
    sum += __shfl_xor_sync(0xffffffff, sum, tgx ^ 16);
    unsigned long long int lsum = llitoulli(llrintf(ERRORSCALEF * sum));
    if (tgx == 0) 
        atomicAdd(&sAccumulator, lsum);
    __syncthreads();               
    Float norm = 1.0f / static_cast<Float>((double)sAccumulator * ONEOVERERRORSCALE);
    

    pos = threadIdx.x;
    while (pos < stride)
    {
        Float z = pData[pos];
        Float a = expf(z - maxValue);
        pData[pos] = fminf(1.0f, a * norm);
        pos += blockDim.x;
    }    
}

/**
 * @brief Computes softmax activation for the given data array on the GPU with a specified batch size and stride.
 *
 * @param pData Pointer to the data array.
 * @param batch Number of batches in the data array.
 * @param stride The stride between elements in pData.
 */
void kCalculateSoftMaxActivation(Float* pData, uint32_t batch, uint32_t stride)
{
    uint32_t warps = getGpu()._threadsPerBlock / getGpu()._warpSize;
    kCalculateSoftMaxActivation_kernel<<<batch, getGpu()._threadsPerBlock>>>(pData, stride);
    LAUNCHERROR("kCalculateSoftMaxActivation_kernel");
}
