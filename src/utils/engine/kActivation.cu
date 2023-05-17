#include "GpuTypes.h"
#include "Types.h"
#include <limits>
#include <algorithm>
#include <math.h>

static __constant__ GpuData cData;

/**
 * @brief CUDA kernel for calculating the maximum value using atomicMax operation.
 *
 * @param address Pointer to the address to perform atomicMax operation on.
 * @param val The value to compare and update.
 * @return float The old value before the update.
 */
__device__ inline float atomicMax(float* address, float val)
{
    int* address_as_i = reinterpret_cast<int*>(address);
    int old = __float_as_int(*address);
    int assumed;
    do {
        assumed = old;
        old = atomicMax(reinterpret_cast<int*>(address), __float_as_int(val));
    } while (assumed != old);
    return __int_as_float(old);
}

/**
 * @brief Copies the GPU data to the constant memory symbol cData.
 */
void SetKActivationGpuData()
{
    cudaError_t status = cudaMemcpyToSymbol(cData, getGpu()._data, sizeof(GpuData));
    cudaCheckError();
}

/**
 * @brief Copies the GPU data from the constant memory symbol cData.
 */
void GetKActivationGpuData()
{
    cudaError_t status = cudaMemcpyFromSymbol(getGpu()._data, cData, sizeof(GpuData));
    cudaCheckError();
}

/**
 * @brief CUDA kernel for calculating the sigmoid activation function.
 *
 * @param pData Pointer to the data array.
 * @param size The size of the data array.
 */
__global__ void kCalculateSigmoidActivation_kernel(NNFloat* pData, size_t size)
{
    constexpr NNFloat ONE = 1.0;
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        NNFloat x = pData[pos];
        pData[pos] = ONE / (ONE + exp(-x));
    }
}

/**
 * @brief Calculates the sigmoid activation for the given data array using CUDA.
 *
 * @param pData Pointer to the data array.
 * @param size The size of the data array.
 */
void kCalculateSigmoidActivation(NNFloat* pData, size_t size)
{
    constexpr uint32_t DEFAULT_THREADS_PER_BLOCK = 128;
    uint32_t threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
    uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    kCalculateSigmoidActivation_kernel<<<blocks, threadsPerBlock>>>(pData, size);
    cudaCheckError();
}

/**
 * @brief CUDA kernel for calculating the hyperbolic tangent (tanh) activation function.
 *
 * @param pData Pointer to the data array.
 * @param size The size of the data array.
 */
__global__ void kCalculateTanhActivation_kernel(NNFloat* pData, size_t size)
{
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        NNFloat x = pData[pos];
        pData[pos] = tanh(x);
    }
}

/**
 * @brief Calculates the hyperbolic tangent (tanh) activation for the given data array using CUDA.
 *
 * @param pData Pointer to the data array.
 * @param size The size of the data array.
 */
void kCalculateTanhActivation(NNFloat* pData, size_t size)
{
    constexpr uint32_t DEFAULT_THREADS_PER_BLOCK = 128;
    uint32_t threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
    uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    kCalculateTanhActivation_kernel<<<blocks, threadsPerBlock>>>(pData, size);
    cudaCheckError();
}

/**
 * @brief CUDA kernel for calculating the rectified linear unit (ReLU) activation function.
 *
 * @param pData Pointer to the data array.
 * @param size The size of the data array.
 */
__global__ void kCalculateRELUActivation_kernel(NNFloat* pData, size_t size)
{
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        NNFloat val = pData[pos];
        pData[pos] = max(0.0f, val);
    }
}

/**
 * @brief Calculates the rectified linear unit (ReLU) activation for the given data array using CUDA.
 *
 * @param pData Pointer to the data array.
 * @param size The size of the data array.
 */
void kCalculateRELUActivation(NNFloat* pData, size_t size)
{
    constexpr uint32_t DEFAULT_THREADS_PER_BLOCK = 128;
    uint32_t threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
    uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    kCalculateRELUActivation_kernel<<<blocks, threadsPerBlock>>>(pData, size);
    cudaCheckError();
}

/**
 * @brief CUDA kernel for calculating the leaky rectified linear unit (Leaky ReLU) activation function.
 *
 * @param pData Pointer to the data array.
 * @param size The size of the data array.
 * @param slope The slope parameter for the Leaky ReLU function.
 */
__global__ void kCalculateLRELUActivation_kernel(NNFloat* pData, size_t size, NNFloat slope)
{
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        NNFloat val = pData[pos];
        pData[pos] = max(val, val * slope);
    }
}

/**
 * @brief Calculates the leaky rectified linear unit (Leaky ReLU) activation for the given data array using CUDA.
 *
 * @param pData Pointer to the data array.
 * @param size The size of the data array.
 * @param slope The slope parameter for the Leaky ReLU function.
 */
void kCalculateLRELUActivation(NNFloat* pData, size_t size, NNFloat slope)
{
    constexpr uint32_t DEFAULT_THREADS_PER_BLOCK = 128;
    uint32_t threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
    uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    kCalculateLRELUActivation_kernel<<<blocks, threadsPerBlock>>>(pData, size, slope);
    cudaCheckError();
}

/**
 * @brief CUDA kernel for calculating the exponential linear unit (ELU) activation function.
 *
 * @param pData Pointer to the data array.
 * @param size The size of the data array.
 * @param alpha The alpha parameter for the ELU function.
 */
__global__ void kCalculateELUActivation_kernel(NNFloat* pData, size_t size, NNFloat alpha)
{
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        NNFloat x = pData[pos];
        pData[pos] = (x > 0.0f) ? x : alpha * (expf(x) - 1.0f);
    }
}

/**
 * @brief Calculates the exponential linear unit (ELU) activation for the given data array using CUDA.
 *
 * @param pData Pointer to the data array.
 * @param size The size of the data array.
 * @param alpha The alpha parameter for the ELU function.
 */
void kCalculateELUActivation(NNFloat* pData, size_t size, NNFloat alpha)
{
    constexpr uint32_t DEFAULT_THREADS_PER_BLOCK = 128;
    uint32_t threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
    uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    kCalculateELUActivation_kernel<<<blocks, threadsPerBlock>>>(pData, size, alpha);
    cudaCheckError();
}

/**
 * @brief Calculates the exponential linear unit (ELU) activation for the given data array using CUDA.
 *
 * @param pData Pointer to the data array.
 * @param size The size of the data array.
 * @param alpha The alpha parameter for the ELU function.
 */
void calculateELUActivation(NNFloat* pData, size_t size, NNFloat alpha)
{
    constexpr uint32_t DEFAULT_THREADS_PER_BLOCK = 128;
    uint32_t threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
    uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    kCalculateELUActivation_kernel<<<blocks, threadsPerBlock>>>(pData, size, alpha);
    cudaCheckError();
}

/**
 * @brief CUDA kernel for calculating the scaled exponential linear unit (SELU) activation function.
 *
 * @param pData Pointer to the data array.
 * @param size The size of the data array.
 * @param alpha The alpha parameter for the SELU function.
 * @param lambda The lambda parameter for the SELU function.
 */
__global__ void kCalculateSELUActivation_kernel(NNFloat* pData, size_t size, NNFloat alpha, NNFloat lambda)
{
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        NNFloat x = pData[pos];
        pData[pos] = (x > 0.0f) ? lambda * x : lambda * alpha * (expf(x) - 1.0f);
    }
}

/**
 * @brief Calculates the scaled exponential linear unit (SELU) activation for the given data array using CUDA.
 *
 * @param pData Pointer to the data array.
 * @param size The size of the data array.
 * @param alpha The alpha parameter for the SELU function.
 * @param lambda The lambda parameter for the SELU function.
 */
void kCalculateSELUActivation(NNFloat* pData, size_t size, NNFloat alpha, NNFloat lambda)
{
    constexpr uint32_t DEFAULT_THREADS_PER_BLOCK = 128;
    uint32_t threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
    uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    kCalculateSELUActivation_kernel<<<blocks, threadsPerBlock>>>(pData, size, alpha, lambda);
    cudaCheckError();
}

/**
 * @brief CUDA kernel for calculating the softmax activation function with warp-based reduction.
 *
 * @param pData Pointer to the data array.
 * @param stride The stride value.
 */
__global__ void LAUNCH_BOUNDS_kCalculateSoftMaxActivation_kernel(NNFloat* pData, uint32_t stride)
{
    extern __shared__ NNFloat sData[];
    unsigned int tid = threadIdx.x;
    unsigned int warpSize = 32;
    unsigned int lane = tid % warpSize;
    unsigned int warpIdx = tid / warpSize;
    unsigned int numWarps = blockDim.x / warpSize;

    // Find the maximum value within each warp
    NNFloat maxValue = -INFINITY;
    while (warpIdx < stride)
    {
        NNFloat z = pData[blockIdx.x * stride + warpIdx * warpSize + lane];
        maxValue = max(maxValue, z);
        warpIdx += numWarps;
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        maxValue = max(maxValue, __shfl_down_sync(0xffffffff, maxValue, offset));

    // Compute the sum of exponentials within each warp
    NNFloat sum = 0.0;
    warpIdx = tid / warpSize;
    while (warpIdx < stride)
    {
        NNFloat z = pData[blockIdx.x * stride + warpIdx * warpSize + lane];
        sum += exp(z - maxValue);
        warpIdx += numWarps;
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    // Reduce across warps using atomic operations
    __shared__ NNFloat sMaxValue;
    __shared__ unsigned long long int sAccumulator;
    if (tid % warpSize == 0)
    {
        atomicMax(&sMaxValue, maxValue);
        atomicAdd(&sAccumulator, __double_as_longlong(sum));
    }
    __syncthreads();

    // Normalize the values and store the softmax activations
    NNFloat maxVal = sMaxValue;
    unsigned long long int accumulator = sAccumulator;
    NNFloat norm = 1.0 / (accumulator * ONEOVERERRORSCALE);
    warpIdx = tid / warpSize;
    while (warpIdx < stride)
    {
        NNFloat z = pData[blockIdx.x * stride + warpIdx * warpSize + lane];
        NNFloat a = exp(z - maxVal);
        pData[blockIdx.x * stride + warpIdx * warpSize + lane] = min((NNFloat)1.0, a * norm);
        warpIdx += numWarps;
    }
}
/**
 * @brief Calculates the softmax activation for the given data array using CUDA.
 *
 * @param pData Pointer to the data array.
 * @param batch The batch size.
 * @param stride The stride value.
 */
void kCalculateSoftMaxActivation(NNFloat* pData, uint32_t batch, uint32_t stride)
{
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (batch + threadsPerBlock - 1) / threadsPerBlock;
    kCalculateSoftMaxActivation_kernel<<<blocks, threadsPerBlock>>>(pData, stride);
    cudaCheckError();
}
