#include "GpuTypes.h"
#include "Types.h"
#include <limits>

static __constant__ GpuData cData;

/**
 * @brief Sets the GPU data for loss computation.
 *
 * This function copies the GPU data to the `cData` symbol using `cudaMemcpyToSymbol`.
 *
 * @remarks This function assumes that the `getGpu()` function returns a valid GPU object.
 */
void SetLossGpuData()
{
    cudaError_t status = cudaMemcpyToSymbol(cData, &(getGpu()._data), sizeof(GpuData));
    if (status != cudaSuccess)
    {
        printf("cudaMemcpyToSymbol: SetKernelsGpuData copy to cData failed: %s\n", cudaGetErrorString(status));
        // Handle the error or return an error code
    }
}

/**
 * @brief Retrieves the GPU data for loss computation.
 *
 * This function copies the `cData` symbol to the GPU data using `cudaMemcpyFromSymbol`.
 *
 * @remarks This function assumes that the `getGpu()` function returns a valid GPU object.
 */
void GetLossGpuData()
{
    cudaError_t status = cudaMemcpyFromSymbol(&(getGpu()._data), cData, sizeof(GpuData));
    if (status != cudaSuccess)
    {
        printf("cudaMemcpyFromSymbol: SetKernelsGpuData copy from cData failed: %s\n", cudaGetErrorString(status));
        // Handle the error or return an error code
    }
}

/**
 * @brief GPU kernel to calculate the sparse raw L1 error.
 *
 * This kernel calculates the sparse raw L1 error for a given position using the provided data weight and unit values.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param pDataWeight Pointer to the data weight array.
 * @param pUnit Pointer to the unit array.
 * @param stride The stride between consecutive data elements.
 * @param size The size of the data elements.
 *
 * @remarks This kernel assumes that it will be launched with a grid of blocks and each block will have a number of threads equal to `blockDim.x`.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 */
__global__ void kCalculateSparseRawL1Error_kernel(uint32_t position, Float* pDataWeight, Float* pUnit, uint64_t stride, uint64_t size)
{
    uint64_t pos = blockDim.x * blockIdx.x + threadIdx.x;
    if (pos < size)
    {
        Float w = 1.0;
        if (pDataWeight != nullptr)
        {
            uint64_t dpos = (pos / stride) + position;
            dpos = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w *= pDataWeight[dpos];
        }

        Float a = pUnit[pos];
        Float error = w * fabsf(a);
        REDUCEERROR(error);
    }
}

/**
 * @brief GPU kernel to calculate the sparse non-zero L1 error.
 *
 * This kernel calculates the sparse non-zero L1 error for a given position and batch using the provided data.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data elements in the batch.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 *
 * @remarks This kernel assumes that it will be launched with a grid of blocks and each block will have a number of threads equal to `blockDim.x`.
 * The `cData._warpSize` and `cData._warpMask` variables are expected to be defined elsewhere.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The `s_pUnit`, `s_pSparseStart`, and `s_pSparseEnd` shared memory arrays are used to cache data for efficient access within each warp.
 */
__global__ void LAUNCH_BOUNDS() kCalculateSparseNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        uint64_t offset = pos * stride;
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        __syncthreads();

        // Calculate the error.
        for (uint64_t i = threadIdx.x; i < end - pos1; i += cData._warpSize)
        {
            Float a = s_pUnit[i];
            error += w * (fabsf(a - (Float)1.0) - fabsf(a));   
        }
    }  

    REDUCEERROR(error)
}

/**
 * @brief GPU kernel to calculate the sparse only non-zero L1 error.
 *
 * This kernel calculates the sparse only non-zero L1 error for a given position and batch using the provided data.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data elements in the batch.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 *
 * @remarks This kernel assumes that it will be launched with a grid of blocks and each block will have a number of threads equal to `blockDim.x`.
 * The `cData._warpSize` and `cData._warpMask` variables are expected to be defined elsewhere.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The `s_pUnit`, `s_pSparseStart`, `s_pSparseEnd`, and `s_bNonZero` shared memory arrays are used to cache data for efficient access within each warp.
 */
__global__ void LAUNCH_BOUNDS() kCalculateSparseOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        // Load the mask into shared memory.
        s_bNonZero[threadIdx.x] = (pSparseIndex[pos1] < end);

        __syncthreads();

        // Calculate the error.
        for (uint64_t i = threadIdx.x; i < end - pos1; i += cData._warpSize)
        {
            if (s_bNonZero[i])
            {
                Float a = s_pUnit[i];
                error += w * fabsf(a - (Float)1.0);   
            }
        }
    }  

    REDUCEERROR(error)
}

/**
 * @brief Calculates the sparse L1 error for a given position and batch.
 *
 * This function calculates the sparse L1 error using the provided data. It uses shared memory to reduce global memory accesses.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data elements in the batch.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param bSparseIgnoreZero Flag indicating whether to ignore zero values in the sparse data.
 * @return The calculated sparse L1 error.
 *
 * @remarks This function assumes that the `getGpu()` function returns a valid GPU object.
 * The `cData._warpSize` and `cData._warpMask` variables are expected to be defined elsewhere.
 * The `getGpu()._data._pAccumulator` and `getGpu()._pbAccumulator` variables are used for accumulating the error.
 * The `ONEOVERERRORSCALE` constant is used to scale the error value.
 */
Float kCalculateSparseL1Error(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        // Load the mask into shared memory.
        s_bNonZero[threadIdx.x] = (pSparseIndex[pos1] < end);

        __syncthreads();

        // Calculate the error.
        for (uint64_t i = threadIdx.x; i < end - pos1; i += cData._warpSize)
        {
            if (!bSparseIgnoreZero || s_bNonZero[i])
            {
                Float a = s_pUnit[i];
                error += w * fabsf(a - (Float)1.0);   
            }
        }
    }  

    getGpu()._pbAccumulator->Download(); 
    return (Float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/**
 * @brief GPU kernel to calculate the indexed sparse non-zero L1 error.
 *
 * This kernel calculates the indexed sparse non-zero L1 error for a given position and batch using the provided data.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data elements in the batch.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 *
 * @remarks This kernel assumes that it will be launched with a grid of blocks and each block will have a number of threads equal to `blockDim.x`.
 * The `cData._warpSize` and `cData._warpMask` variables are expected to be defined elsewhere.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The `s_pUnit`, `s_pSparseStart`, `s_pSparseEnd`, and `s_bNonZero` shared memory arrays are used to cache data for efficient access within each warp.
 */
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        // Load the mask into shared memory.
        s_bNonZero[threadIdx.x] = (pSparseIndex[pos1] < end);

        __syncthreads();

        // Calculate the error.
        for (uint64_t i = threadIdx.x; i < end - pos1; i += cData._warpSize)
        {
            if (s_bNonZero[i])
            {
                Float a = s_pUnit[i];
                error += w * (fabsf(a - (Float)1.0) - fabsf(a));   
            }
        }
    }  

    REDUCEERROR(error)
}

/**
 * @brief GPU kernel to calculate the indexed sparse only non-zero L1 error.
 *
 * This kernel calculates the indexed sparse only non-zero L1 error for a given position and batch using the provided data.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data elements in the batch.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 *
 * @remarks This kernel assumes that it will be launched with a grid of blocks and each block will have a number of threads equal to `blockDim.x`.
 * The `cData._warpSize` and `cData._warpMask` variables are expected to be defined elsewhere.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The `s_pUnit`, `s_pSparseStart`, `s_pSparseEnd`, and `s_bNonZero` shared memory arrays are used to cache data for efficient access within each warp.
 */
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        // Load the mask into shared memory.
        s_bNonZero[threadIdx.x] = (pSparseIndex[pos1] < end);

        __syncthreads();

        // Calculate the error.
        for (uint64_t i = threadIdx.x; i < end - pos1; i += cData._warpSize)
        {
            if (s_bNonZero[i])
            {
                Float a = s_pUnit[i];
                error += w * fabsf(a - (Float)1.0);   
            }
        }
    }  

    REDUCEERROR(error)
}

/**
 * @brief Calculates the indexed sparse L1 error for a given position and batch.
 *
 * This function calculates the indexed sparse L1 error using the provided data. It uses shared memory to reduce global memory accesses.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data elements in the batch.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param bSparseIgnoreZero Flag indicating whether to ignore zero values in the sparse data.
 * @return The calculated indexed sparse L1 error.
 *
 * @remarks This function assumes that the `getGpu()` function returns a valid GPU object.
 * The `cData._warpSize` variable is expected to be defined elsewhere.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The `LAUNCHERROR` macro is used to check for kernel launch errors.
 * The `CalculateBlocks` function is used to determine the number of blocks needed for the kernel launches.
 * The `getGpu()._threadsPerBlock` variable is used to specify the number of threads per block for kernel launches.
 * The `getGpu()._pbAccumulator` variable is used for accumulating the error.
 * The `ONEOVERERRORSCALE` constant is used to scale the error value.
 */
Float kCalculateIndexedSparseL1Error(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (bSparseIgnoreZero)
    {
        uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateIndexedSparseOnlyNonZeroL1Error_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight);
        LAUNCHERROR("kCalculateIndexedSparseOnlyNonZeroL1Error_kernel");
    }
    else
    {
        uint64_t size = (uint64_t)batch * (uint64_t)stride;
        uint32_t blocks = CalculateBlocks(size);
        kCalculateSparseRawL1Error_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, pDataWeight, pUnit, stride, size);
        LAUNCHERROR("kCalculateSparseRawL1Error_kernel");
        blocks = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateIndexedSparseNonZeroL1Error_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight);
        LAUNCHERROR("kCalculateIndexedSparseNonZeroL1Error_kernel");
    }

    REDUCEERROR(error)

    getGpu()._pbAccumulator->Download();
    return (Float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/**
 * @brief GPU kernel to calculate the sparse analog only non-zero L1 error with template specialization.
 *
 * This kernel calculates the sparse analog only non-zero L1 error using the provided data and template specialization for the sparse data type.
 *
 * @tparam T The data type of the sparse data.
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data elements in the batch.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 *
 * @remarks This kernel assumes that it will be launched with a grid of blocks and each block will have a number of threads equal to `blockDim.x`.
 * The `cData._warpSize` and `cData._warpMask` variables are expected to be defined elsewhere.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The `s_pUnit`, `s_pSparseStart`, `s_pSparseEnd`, and `s_bNonZero` shared memory arrays are used to cache data for efficient access within each warp.
 */
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        // Load the mask into shared memory.
        s_bNonZero[threadIdx.x] = (pSparseIndex[pos1] < end);

        __syncthreads();

        // Calculate the error.
        for (uint64_t i = threadIdx.x; i < end - pos1; i += cData._warpSize)
        {
            if (s_bNonZero[i])
            {
                Float a = s_pUnit[i];
                T t = pSparseData[i];
                error += w * fabsf(a - t);   
            }
        }
    }  

    REDUCEERROR(error)
}

/**
 * @brief GPU kernel to calculate the sparse analog non-zero L1 error with template specialization.
 *
 * This kernel calculates the sparse analog non-zero L1 error using the provided data and template specialization for the sparse data type.
 *
 * @tparam T The data type of the sparse data.
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data elements in the batch.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 *
 * @remarks This kernel assumes that it will be launched with a grid of blocks and each block will have a number of threads equal to `blockDim.x`.
 * The `cData._warpSize` and `cData._warpMask` variables are expected to be defined elsewhere.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The `s_pUnit`, `s_pSparseStart`, `s_pSparseEnd`, and `s_bNonZero` shared memory arrays are used to cache data for efficient access within each warp.
 */
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        // Load the mask into shared memory.
        s_bNonZero[threadIdx.x] = (pSparseIndex[pos1] < end);

        __syncthreads();

        // Calculate the error.
        for (uint64_t i = threadIdx.x; i < end - pos1; i += cData._warpSize)
        {
            if (s_bNonZero[i])
            {
                Float a = s_pUnit[i];
                T t = pSparseData[i];
                error += w * (fabsf(a - t) - fabsf(a));   
            }
        }
    }  

    REDUCEERROR(error)
}

/**
 * @brief GPU kernel to calculate the sparse analog only non-zero L1 error for unsigned char sparse data specialization.
 *
 * This kernel calculates the sparse analog only non-zero L1 error using the provided data and specialization for the unsigned char sparse data type.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data elements in the batch.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the unsigned char sparse data array.
 *
 * @remarks This kernel assumes that it will be launched with a grid of blocks and each block will have a number of threads equal to `blockDim.x`.
 * The `cData._warpSize` and `cData._warpMask` variables are expected to be defined elsewhere.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The `s_pUnit`, `s_pSparseStart`, `s_pSparseEnd`, and `s_bNonZero` shared memory arrays are used to cache data for efficient access within each warp.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, unsigned char* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        // Load the mask into shared memory.
        s_bNonZero[threadIdx.x] = (pSparseIndex[pos1] < end);

        __syncthreads();

        // Calculate the error.
        for (uint64_t i = threadIdx.x; i < end - pos1; i += cData._warpSize)
        {
            if (s_bNonZero[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[i] * (Float)(1.0 / 256.0);
                error += w * fabsf(a - t);   
            }
        }
    }  

    REDUCEERROR(error)
}


/**
 * @brief GPU kernel to calculate the sparse analog non-zero L1 error for unsigned char sparse data specialization.
 *
 * This kernel calculates the sparse analog non-zero L1 error using the provided data and specialization for the unsigned char sparse data type.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data elements in the batch.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the unsigned char sparse data array.
 *
 * @remarks This kernel assumes that it will be launched with a grid of blocks and each block will have a number of threads equal to `blockDim.x`.
 * The `cData._warpSize` and `cData._warpMask` variables are expected to be defined elsewhere.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The `s_pUnit`, `s_pSparseStart`, `s_pSparseEnd`, and `s_bNonZero` shared memory arrays are used to cache data for efficient access within each warp.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, unsigned char* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        // Load the mask into shared memory.
        s_bNonZero[threadIdx.x] = (pSparseIndex[pos1] < end);

        __syncthreads();

        // Calculate the error.
        for (uint64_t i = threadIdx.x; i < end - pos1; i += cData._warpSize)
        {
            if (s_bNonZero[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[i] * (Float)(1.0 / 256.0);
                error += w * (fabsf(a - t) - fabsf(a));   
            }
        }
    }  

    REDUCEERROR(error)
}

/**
 * @brief GPU kernel to calculate the sparse analog only non-zero L1 error for char sparse data specialization.
 *
 * This kernel calculates the sparse analog only non-zero L1 error using the provided data and specialization for the char sparse data type.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data elements in the batch.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the char sparse data array.
 *
 * @remarks This kernel assumes that it will be launched with a grid of blocks and each block will have a number of threads equal to `blockDim.x`.
 * The `cData._warpSize` and `cData._warpMask` variables are expected to be defined elsewhere.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The `s_pUnit`, `s_pSparseStart`, `s_pSparseEnd`, and `s_bNonZero` shared memory arrays are used to cache data for efficient access within each warp.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, char* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        // Load the mask into shared memory.
        s_bNonZero[threadIdx.x] = (pSparseIndex[pos1] < end);

        __syncthreads();

        // Calculate the error.
        for (uint64_t i = threadIdx.x; i < end - pos1; i += cData._warpSize)
        {
            if (s_bNonZero[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[i] * (Float)(1.0 / 128.0);
                error += w * fabsf(a - t);   
            }
        }
    }  

    REDUCEERROR(error)
}

/**
 * @brief GPU kernel to calculate the sparse analog non-zero L1 error for char sparse data specialization.
 *
 * This kernel calculates the sparse analog non-zero L1 error using the provided data and specialization for the char sparse data type.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data elements in the batch.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the char sparse data array.
 *
 * @remarks This kernel assumes that it will be launched with a grid of blocks and each block will have a number of threads equal to `blockDim.x`.
 * The `cData._warpSize` and `cData._warpMask` variables are expected to be defined elsewhere.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The `s_pUnit`, `s_pSparseStart`, `s_pSparseEnd`, and `s_bNonZero` shared memory arrays are used to cache data for efficient access within each warp.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, char* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        // Load the mask into shared memory.
        s_bNonZero[threadIdx.x] = (pSparseIndex[pos1] < end);

        __syncthreads();

        // Calculate the error.
        for (uint64_t i = threadIdx.x; i < end - pos1; i += cData._warpSize)
        {
            if (s_bNonZero[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[i] * (Float)(1.0 / 128.0);
                error += w * (fabsf(a - t) - fabsf(a));   
            }
        }
    }  

    REDUCEERROR(error)
}

/**
 * @brief Calculate the sparse analog L1 error.
 *
 * This function calculates the sparse analog L1 error using the provided data and specialization for the given template type.
 *
 * @tparam T The data type of the sparse data.
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data elements in the batch.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 * @param bSparseIgnoreZero Flag indicating whether to ignore zero values in the sparse data.
 * @return The calculated sparse analog L1 error.
 *
 * @remarks This function assumes that the GPU device is accessible through the `getGpu()` function and the device memory is properly allocated.
 * The `cData._warpSize`, `cData._threadsPerBlock`, and `cData._pbAccumulator` variables are expected to be defined elsewhere.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The `CalculateBlocks` function is expected to be defined elsewhere.
 * The `kCalculateSparseAnalogOnlyNonZeroL1Error_kernel` and `kCalculateSparseAnalogNonZeroL1Error_kernel` kernels are called conditionally based on the `bSparseIgnoreZero` flag.
 * The `kCalculateSparseRawL1Error_kernel` kernel is called to calculate the raw L1 error when `bSparseIgnoreZero` is false.
 */
template<typename T>
Float kCalculateSparseAnalogL1Error(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, Float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (bSparseIgnoreZero)
    {
        uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);    
        kCalculateSparseAnalogOnlyNonZeroL1Error_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
        LAUNCHERROR("kCalculateSparseAnalogOnlyNonZeroL1Error_kernel");   
    }
    else
    {
        uint64_t size = (uint64_t)batch * (uint64_t)stride;
        uint32_t blocks = CalculateBlocks(size);    
        kCalculateSparseRawL1Error_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, pDataWeight, pUnit, stride, size);
        LAUNCHERROR("kCalculateSparseRawL1Error_kernel");
        blocks = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateSparseAnalogNonZeroL1Error_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
        LAUNCHERROR("kCalculateSparseAnalogNonZeroL1Error_kernel");
    }

    getGpu()._pbAccumulator->Download(); 
    return (Float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/**
 * @brief Calculate the indexed sparse analog only non-zero L1 error.
 *
 * This kernel calculates the L1 error for indexed sparse data, considering only non-zero elements, using the provided data and specialization for the given template type.
 *
 * @tparam T The data type of the sparse data.
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data elements in the batch.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `getGpu()` function and the device memory is properly allocated.
 * The `cData._warpSize`, `cData._bShuffleIndices`, `cData._pShuffleIndex`, and `cData._pbAccumulator` variables are expected to be defined elsewhere.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The `CalculateBlocks` function is expected to be defined elsewhere.
 */
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        // Load the mask into shared memory.
        s_bNonZero[threadIdx.x] = (pSparseIndex[pos1] < end);

        __syncthreads();

        // Calculate the error.
        for (uint64_t i = threadIdx.x; i < end - pos1; i += cData._warpSize)
        {
            if (s_bNonZero[i])
            {
                Float a = s_pUnit[i];
                T t = pSparseData[i];
                error += w * fabsf(a - t);
            }
        }
    }

    REDUCEERROR(error)
}

/**
 * @brief Calculate the indexed sparse analog non-zero L1 error.
 *
 * This kernel calculates the L1 error for indexed sparse data, considering only non-zero elements, using the provided data and specialization for the given template type.
 *
 * @tparam T The data type of the sparse data.
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data elements in the batch.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `getGpu()` function and the device memory is properly allocated.
 * The `cData._warpSize`, `cData._bShuffleIndices`, `cData._pShuffleIndex`, `cData._pbAccumulator` variables are expected to be defined elsewhere.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The `CalculateBlocks` function is expected to be defined elsewhere.
 */
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        // Load the mask into shared memory.
        s_bNonZero[threadIdx.x] = (pSparseIndex[pos1] < end);

        __syncthreads();

        // Calculate the error.
        for (uint64_t i = threadIdx.x; i < end - pos1; i += cData._warpSize)
        {
            if (s_bNonZero[i])
            {
                Float a = s_pUnit[i];
                T t = pSparseData[i];
                error += w * (fabsf(a - t) - fabsf(a));
            }
        }
    }

    REDUCEERROR(error)
}

/**
 * @brief Calculate the indexed sparse analog only non-zero L1 error.
 *
 * This kernel calculates the L1 error for indexed sparse data, considering only non-zero elements, using the provided data and specialization for the unsigned char template type.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data elements in the batch.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `getGpu()` function and the device memory is properly allocated.
 * The `cData._warpSize`, `cData._bShuffleIndices`, `cData._pShuffleIndex`, `cData._pbAccumulator` variables are expected to be defined elsewhere.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, unsigned char* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        // Load the mask into shared memory.
        s_bNonZero[threadIdx.x] = (pSparseIndex[pos1] < end);

        __syncthreads();

        // Calculate the error.
        for (uint64_t i = threadIdx.x; i < end - pos1; i += cData._warpSize)
        {
            if (s_bNonZero[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[i] * (Float)(1.0 / 256.0);
                error += w * fabsf(a - t);
            }
        }
    }

    REDUCEERROR(error)
}

/**
 * @brief Calculate the indexed sparse analog non-zero L1 error.
 *
 * This kernel calculates the L1 error for indexed sparse data, considering only non-zero elements, using the provided data and specialization for the unsigned char template type.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data elements in the batch.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `getGpu()` function and the device memory is properly allocated.
 * The `cData._warpSize`, `cData._bShuffleIndices`, `cData._pShuffleIndex`, `cData._pbAccumulator` variables are expected to be defined elsewhere.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, unsigned char* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        // Load the mask into shared memory.
        s_bNonZero[threadIdx.x] = (pSparseIndex[pos1] < end);

        __syncthreads();

        // Calculate the error.
        for (uint64_t i = threadIdx.x; i < end - pos1; i += cData._warpSize)
        {
            if (s_bNonZero[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[i] * (Float)(1.0 / 256.0);
                error += w * (fabsf(a - t) - fabsf(a));
            }
        }
    }

    REDUCEERROR(error)
}

/**
 * @brief Calculate the indexed sparse analog only non-zero L1 error.
 *
 * This kernel calculates the L1 error for indexed sparse data, considering only non-zero elements, using the provided data and specialization for the char template type.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data elements in the batch.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `getGpu()` function and the device memory is properly allocated.
 * The `cData._warpSize`, `cData._bShuffleIndices`, `cData._pShuffleIndex`, `cData._pbAccumulator` variables are expected to be defined elsewhere.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, char* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        // Load the mask into shared memory.
        s_bNonZero[threadIdx.x] = (pSparseIndex[pos1] < end);

        __syncthreads();

        // Calculate the error.
        for (uint64_t i = threadIdx.x; i < end - pos1; i += cData._warpSize)
        {
            if (s_bNonZero[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[i] * (Float)(1.0 / 128.0);
                error += w * fabsf(a - t);
            }
        }
    }

    REDUCEERROR(error)
}

/**
 * @brief Calculate the indexed sparse analog non-zero L1 error.
 *
 * This kernel calculates the L1 error for indexed sparse data, considering only non-zero elements, using the provided data and specialization for the char template type.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data elements in the batch.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `getGpu()` function and the device memory is properly allocated.
 * The `cData._warpSize`, `cData._bShuffleIndices`, `cData._pShuffleIndex`, `cData._pbAccumulator` variables are expected to be defined elsewhere.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, char* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        // Load the mask into shared memory.
        s_bNonZero[threadIdx.x] = (pSparseIndex[pos1] < end);

        __syncthreads();

        // Calculate the error.
        for (uint64_t i = threadIdx.x; i < end - pos1; i += cData._warpSize)
        {
            if (s_bNonZero[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[i] * (Float)(1.0 / 128.0);
                error += w * (fabsf(a - t) - fabsf(a));
            }
        }
    }

    REDUCEERROR(error)
}

/**
 * @brief Calculate the indexed sparse analog L1 error.
 *
 * This function calculates the L1 error for indexed sparse data, considering both zero and non-zero elements, using the provided data and specialization for the template type T.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data elements in the batch.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 * @param bSparseIgnoreZero Flag indicating whether to ignore zero elements or not.
 *
 * @return The calculated L1 error.
 *
 * @remarks This function assumes that the GPU device is accessible through the `getGpu()` function and the device memory is properly allocated.
 * The `CalculateBlocks` and `LAUNCHERROR` functions are expected to be defined elsewhere.
 * The `ONEOVERERRORSCALE` macro is expected to be defined elsewhere.
 */
template<typename T>
Float kCalculateIndexedSparseAnalogL1Error(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, Float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    if (bSparseIgnoreZero)
    {
        uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateIndexedSparseAnalogOnlyNonZeroL1Error_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
        LAUNCHERROR("kCalculateSparseAnalogOnlyNonZeroL1Error_kernel");
    }
    else
    {
        uint64_t size = (uint64_t)batch * (uint64_t)stride;
        uint32_t blocks = CalculateBlocks(size);
        kCalculateSparseRawL1Error_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, pDataWeight, pUnit, stride, size);
        LAUNCHERROR("kCalculateSparseRawL1Error_kernel");
        blocks = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateIndexedSparseAnalogNonZeroL1Error_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
        LAUNCHERROR("kCalculateIndexedSparseAnalogNonZeroL1Error_kernel");
    }

    getGpu()._pbAccumulator->Download();
    return (Float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/**
 * @brief Calculate the raw L2 error for sparse data.
 *
 * This kernel function calculates the L2 error for sparse data using the provided data and weight information.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param pDataWeight Pointer to the data weight array.
 * @param pUnit Pointer to the unit array.
 * @param stride The stride between consecutive data elements.
 * @param size The size of the data to process.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 */
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawL2Error_kernel(uint32_t position, Float* pDataWeight, Float* pUnit, uint32_t stride, uint64_t size)
{
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ Float s_pDataWeight[cData._warpSize];

    uint64_t pos = blockDim.x * blockIdx.x + threadIdx.x;
    Float error = (Float)0.0;

    if (pos < size)
    {
        Float w = (Float)0.5;

        if (pDataWeight != NULL)
        {
            uint64_t dpos = (pos / stride) + position;
            dpos = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            s_pDataWeight[threadIdx.x] = pDataWeight[dpos];
        }

        s_pUnit[threadIdx.x] = pUnit[pos];

        __syncthreads();

        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            error += w * s_pUnit[i] * s_pUnit[i];
        }
    }

    REDUCEERROR(error)
}

/**
 * @brief Calculate the L2 error for sparse data, considering only non-zero values.
 *
 * This kernel function calculates the L2 error for sparse data, considering only non-zero values, using the provided data and weight information.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the array storing the starting positions of sparse data.
 * @param pSparseEnd Pointer to the array storing the ending positions of sparse data.
 * @param pSparseIndex Pointer to the array storing the indices of sparse data.
 * @param pDataWeight Pointer to the data weight array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 */
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                error += w * ((a - (Float)1.0) * (a - (Float)1.0));
            }
        }
    }

    REDUCEERROR(error)
}

/**
 * @brief Calculate the L2 error for sparse data, considering non-zero values.
 *
 * This kernel function calculates the L2 error for sparse data, considering only non-zero values, using the provided data and weight information.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the array storing the starting positions of sparse data.
 * @param pSparseEnd Pointer to the array storing the ending positions of sparse data.
 * @param pSparseIndex Pointer to the array storing the indices of sparse data.
 * @param pDataWeight Pointer to the data weight array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 */
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                error += w * ((a - (Float)1.0) * (a - (Float)1.0) - a * a);
            }
        }
    }

    REDUCEERROR(error)
}


/**
 * @brief Calculate the L2 error for sparse data.
 *
 * This function calculates the L2 error for sparse data using the provided data and weight information.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the array storing the starting positions of sparse data.
 * @param pSparseEnd Pointer to the array storing the ending positions of sparse data.
 * @param pSparseIndex Pointer to the array storing the indices of sparse data.
 * @param pDataWeight Pointer to the data weight array.
 * @param bSparseIgnoreZero Flag indicating whether to ignore zero values in the sparse data.
 *
 * @return The calculated L2 error.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` and `ONEOVERERRORSCALE` macros are expected to be defined elsewhere.
 */
Float kCalculateSparseL2Error(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                error += w * ((a - (Float)1.0) * (a - (Float)1.0) - a * a);
            }
        }
    }

    REDUCEERROR(error)

    return (Float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/**
 * @brief Calculate the L2 error for sparse analog data (non-zero values only).
 *
 * This function calculates the L2 error for sparse analog data (non-zero values only) using the provided data and weight information.
 *
 * @tparam T The type of the sparse data.
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the array storing the starting positions of sparse data.
 * @param pSparseEnd Pointer to the array storing the ending positions of sparse data.
 * @param pSparseIndex Pointer to the array storing the indices of sparse data.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 */
template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                T t = pSparseData[s_pSparseStart[i] + i];
                error += w * ((a - t) * (a - t));
            }
        }
    }

    REDUCEERROR(error)
}

/**
 * @brief Calculate the L2 error for sparse analog data (including zero values).
 *
 * This function calculates the L2 error for sparse analog data (including zero values) using the provided data and weight information.
 *
 * @tparam T The type of the sparse data.
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the array storing the starting positions of sparse data.
 * @param pSparseEnd Pointer to the array storing the ending positions of sparse data.
 * @param pSparseIndex Pointer to the array storing the indices of sparse data.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 */
template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                T t = pSparseData[s_pSparseStart[i] + i];
                error += w * ((a - t) * (a - t) - a * a);
            }
        }
    }

    REDUCEERROR(error)
}

/**
 * @brief Calculate the L2 error for sparse analog data (non-zero values only) with unsigned char sparse data.
 *
 * This function calculates the L2 error for sparse analog data (non-zero values only) using the provided data and weight information.
 * The sparse data is assumed to be of type unsigned char, which is converted to Float for the calculation.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the array storing the starting positions of sparse data.
 * @param pSparseEnd Pointer to the array storing the ending positions of sparse data.
 * @param pSparseIndex Pointer to the array storing the indices of sparse data.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the unsigned char sparse data array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 */
template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, unsigned char* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[s_pSparseStart[i] + i] * (Float)(1.0 / 256.0);
                error += w * ((a - t) * (a - t));
            }
        }
    }

    REDUCEERROR(error)
}

/**
 * @brief Calculate the L2 error for sparse analog data (non-zero values only) with unsigned char sparse data.
 *
 * This function calculates the L2 error for sparse analog data (non-zero values only) using the provided data and weight information.
 * The sparse data is assumed to be of type unsigned char, which is converted to Float for the calculation.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the array storing the starting positions of sparse data.
 * @param pSparseEnd Pointer to the array storing the ending positions of sparse data.
 * @param pSparseIndex Pointer to the array storing the indices of sparse data.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the unsigned char sparse data array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 */
template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, unsigned char* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[s_pSparseStart[i] + i] * (Float)(1.0 / 256.0);
                error += w * ((a - t) * (a - t));
            }
        }
    }

    REDUCEERROR(error)
}

/**
 * @brief Calculate the L2 error for sparse analog data (non-zero values only) with char sparse data.
 *
 * This function calculates the L2 error for sparse analog data (non-zero values only) using the provided data and weight information.
 * The sparse data is assumed to be of type char, which is converted to Float for the calculation.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the array storing the starting positions of sparse data.
 * @param pSparseEnd Pointer to the array storing the ending positions of sparse data.
 * @param pSparseIndex Pointer to the array storing the indices of sparse data.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the char sparse data array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 */
template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, char* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[s_pSparseStart[i] + i] * (Float)(1.0 / 128.0);
                error += w * ((a - t) * (a - t));
            }
        }
    }

    REDUCEERROR(error)
}

/**
 * @brief Calculate the L2 error for sparse analog data (non-zero values only) with char sparse data.
 *
 * This function calculates the L2 error for sparse analog data (non-zero values only) using the provided data and weight information.
 * The sparse data is assumed to be of type char, which is converted to Float for the calculation.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the array storing the starting positions of sparse data.
 * @param pSparseEnd Pointer to the array storing the ending positions of sparse data.
 * @param pSparseIndex Pointer to the array storing the indices of sparse data.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the char sparse data array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 */
template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, char* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[s_pSparseStart[i] + i] * (Float)(1.0 / 128.0);
                error += w * ((a - t) * (a - t));
            }
        }
    }

    REDUCEERROR(error)
}


/**
 * @brief Calculate the L2 error for sparse analog data.
 *
 * This function calculates the L2 error for sparse analog data using the provided data and weight information.
 * The sparse data is assumed to be of type T, which is converted to Float for the calculation.
 *
 * @tparam T The type of the sparse data array.
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the array storing the starting positions of sparse data.
 * @param pSparseEnd Pointer to the array storing the ending positions of sparse data.
 * @param pSparseIndex Pointer to the array storing the indices of sparse data.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 * @param bSparseIgnoreZero Flag indicating whether to ignore zero values in sparse data.
 * @return The calculated L2 error.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The result is scaled by `ONEOVERERRORSCALE` and returned as Float.
 */
template<typename T>
Float kCalculateSparseAnalogL2Error(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, Float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[s_pSparseStart[i] + i] * (Float)(1.0 / 128.0);
                error += w * ((a - t) * (a - t));
            }
        }
    }

    REDUCEERROR(error)

    getGpu()._pbAccumulator->Download();
    return (Float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/**
 * @brief Calculate the L2 error for indexed sparse data.
 *
 * This function calculates the L2 error for indexed sparse data using the provided data and weight information.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the array storing the indices of sparse data.
 * @param pSparseStart Pointer to the array storing the starting positions of sparse data.
 * @param pSparseEnd Pointer to the array storing the ending positions of sparse data.
 * @param pSparseIndex Pointer to the array storing the indices of sparse data.
 * @param pDataWeight Pointer to the data weight array.
 * @return The calculated L2 error.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The result is scaled by `ONEOVERERRORSCALE` and returned as Float.
 */
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                error += w * ((a - (Float)1.0) * (a - (Float)1.0));
            }
        }
    }

    REDUCEERROR(error)
}

/**
 * @brief Calculate the non-zero L2 error for indexed sparse data.
 *
 * This function calculates the non-zero L2 error for indexed sparse data using the provided data and weight information.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the array storing the indices of sparse data.
 * @param pSparseStart Pointer to the array storing the starting positions of sparse data.
 * @param pSparseEnd Pointer to the array storing the ending positions of sparse data.
 * @param pSparseIndex Pointer to the array storing the indices of sparse data.
 * @param pDataWeight Pointer to the data weight array.
 * @return The calculated non-zero L2 error.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The result is scaled by `ONEOVERERRORSCALE` and returned as Float.
 */
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                error += w * ((a - (Float)1.0) * (a - (Float)1.0));
            }
        }
    }

    REDUCEERROR(error)
}

/**
 * @brief Calculate the L2 error for indexed sparse data.
 *
 * This function calculates the L2 error for indexed sparse data using the provided data and weight information.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the array storing the indices of sparse data.
 * @param pSparseStart Pointer to the array storing the starting positions of sparse data.
 * @param pSparseEnd Pointer to the array storing the ending positions of sparse data.
 * @param pSparseIndex Pointer to the array storing the indices of sparse data.
 * @param pDataWeight Pointer to the data weight array.
 * @param bSparseIgnoreZero Flag indicating whether to ignore zero values in sparse data.
 * @return The calculated L2 error.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The result is scaled by `ONEOVERERRORSCALE` and returned as Float.
 */
Float kCalculateIndexedSparseL2Error(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                error += w * ((a - (Float)1.0) * (a - (Float)1.0));
            }
        }
    }

    REDUCEERROR(error)

    getGpu()._pbAccumulator->Download();
    return (Float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/**
 * @brief Calculate the L2 error for indexed sparse analog data (non-zero values only).
 *
 * This function calculates the L2 error for indexed sparse analog data (non-zero values only) using the provided data and weight information.
 *
 * @tparam T The data type of the sparse data.
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the array storing the indices of sparse data.
 * @param pSparseStart Pointer to the array storing the starting positions of sparse data.
 * @param pSparseEnd Pointer to the array storing the ending positions of sparse data.
 * @param pSparseIndex Pointer to the array storing the indices of sparse data.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 */
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                T t = pSparseData[s_pSparseStart[i] + i];
                error += w * ((a - t) * (a - t));
            }
        }
    }

    REDUCEERROR(error)
}

/**
 * @brief Calculate the non-zero L2 error for indexed sparse analog data.
 *
 * This function calculates the non-zero L2 error for indexed sparse analog data using the provided data and weight information.
 *
 * @tparam T The data type of the sparse data.
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the array storing the indices of sparse data.
 * @param pSparseStart Pointer to the array storing the starting positions of sparse data.
 * @param pSparseEnd Pointer to the array storing the ending positions of sparse data.
 * @param pSparseIndex Pointer to the array storing the indices of sparse data.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 */
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                T t = pSparseData[s_pSparseStart[i] + i];
                error += w * ((a - t) * (a - t) - a * a);
            }
        }
    }

    REDUCEERROR(error)
}

/**
 * @brief Calculate the non-zero L2 error for indexed sparse analog data with unsigned char sparse data.
 *
 * This function calculates the non-zero L2 error for indexed sparse analog data using unsigned char sparse data, using the provided data and weight information.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the array storing the indices of sparse data.
 * @param pSparseStart Pointer to the array storing the starting positions of sparse data.
 * @param pSparseEnd Pointer to the array storing the ending positions of sparse data.
 * @param pSparseIndex Pointer to the array storing the indices of sparse data.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, unsigned char* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[s_pSparseStart[i] + i] * (Float)(1.0 / 256.0);
                error += w * ((a - t) * (a - t));
            }
        }
    }

    REDUCEERROR(error)
}

/**
 * @brief Calculate the non-zero L2 error for indexed sparse analog data with unsigned char sparse data.
 *
 * This function calculates the non-zero L2 error for indexed sparse analog data using unsigned char sparse data, using the provided data and weight information.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the array storing the indices of sparse data.
 * @param pSparseStart Pointer to the array storing the starting positions of sparse data.
 * @param pSparseEnd Pointer to the array storing the ending positions of sparse data.
 * @param pSparseIndex Pointer to the array storing the indices of sparse data.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, unsigned char* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[s_pSparseStart[i] + i] * (Float)(1.0 / 256.0);
                error += w * ((a - t) * (a - t));
            }
        }
    }

    REDUCEERROR(error)
}

/**
 * @brief Calculate the non-zero L2 error for indexed sparse analog data with char sparse data.
 *
 * This function calculates the non-zero L2 error for indexed sparse analog data using char sparse data, using the provided data and weight information.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the array storing the indices of sparse data.
 * @param pSparseStart Pointer to the array storing the starting positions of sparse data.
 * @param pSparseEnd Pointer to the array storing the ending positions of sparse data.
 * @param pSparseIndex Pointer to the array storing the indices of sparse data.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, char* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[s_pSparseStart[i] + i] * (Float)(1.0 / 128.0);
                error += w * ((a - t) * (a - t));
            }
        }
    }

    REDUCEERROR(error)
}

/**
 * @brief Calculate the non-zero L2 error for indexed sparse analog data with char sparse data.
 *
 * This function calculates the non-zero L2 error for indexed sparse analog data using char sparse data, using the provided data and weight information.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the array storing the indices of sparse data.
 * @param pSparseStart Pointer to the array storing the starting positions of sparse data.
 * @param pSparseEnd Pointer to the array storing the ending positions of sparse data.
 * @param pSparseIndex Pointer to the array storing the indices of sparse data.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, char* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[s_pSparseStart[i] + i] * (Float)(1.0 / 128.0);
                error += w * ((a - t) * (a - t));
            }
        }
    }

    REDUCEERROR(error)
}

/**
 * @brief Calculate the L2 error for indexed sparse analog data.
 *
 * This function calculates the L2 error for indexed sparse analog data, using the provided data and weight information.
 *
 * @tparam T The type of the sparse data.
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the array storing the indices of sparse data.
 * @param pSparseStart Pointer to the array storing the starting positions of sparse data.
 * @param pSparseEnd Pointer to the array storing the ending positions of sparse data.
 * @param pSparseIndex Pointer to the array storing the indices of sparse data.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 * @param bSparseIgnoreZero Flag indicating whether to ignore zero values in sparse data.
 *
 * @return The calculated L2 error.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory and returns the scaled error value.
 */
template<typename T>
Float kCalculateIndexedSparseAnalogL2Error(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, Float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];

        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[s_pSparseStart[i] + i] * (Float)(1.0 / 128.0);
                error += w * ((a - t) * (a - t));
            }
        }
    }

    REDUCEERROR(error)

    getGpu()._pbAccumulator->Download();
    return (Float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/**
 * @brief Calculate the L2 hinge error for sparse raw data.
 *
 * This function calculates the L2 hinge error for sparse raw data using the provided data and weight information.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param pDataWeight Pointer to the data weight array.
 * @param pUnit Pointer to the unit array.
 * @param stride The stride between consecutive data elements.
 * @param size The size of the data.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory.
 */
__global__ void LAUNCH_BOUNDS() kCalculateSparseRawL2HingeError_kernel(uint32_t position, Float* pDataWeight, Float* pUnit, uint32_t stride, uint64_t size)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ Float s_pWeight[cData._warpSize];
    __shared__ uint64_t s_pStart[cData._warpSize];
    __shared__ uint64_t s_pEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockDim.x * blockIdx.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < size)
    {
        Float w = (Float)0.5;

        if (pDataWeight != NULL)
        {
            uint64_t dpos = (pos / stride) + position;
            dpos = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w *= pDataWeight[dpos];
        }

        Float a = max((Float)0.0, pUnit[pos]);
        error = w * a * a;
    }

    // Load the data into shared memory.
    s_pUnit[threadIdx.x] = pUnit[pos * stride + threadIdx.x];
    s_pWeight[threadIdx.x] = pDataWeight[pos];
    s_pStart[threadIdx.x] = pos * stride;
    s_pEnd[threadIdx.x] = pos * stride + cData._warpSize;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++)
    {
        if (s_pStart[i] < s_pEnd[i])
        {
            Float a = max((Float)0.0, s_pUnit[i]);
            error = w * a * a;
        }
    }

    REDUCEERROR(error)
}

/**
 * @brief Calculate the L2 hinge error for sparse data excluding zero entries.
 *
 * This function calculates the L2 hinge error for sparse data excluding zero entries using the provided data and weight information.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory.
 */
__global__ void LAUNCH_BOUNDS() kCalculateSparseOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                Float diff = min((Float)0.0, a - (Float)1.0);
                error += w * diff * diff;
            }
        }
    }  

    REDUCEERROR(error)
}

/**
 * @brief Calculate the non-zero L2 hinge error for sparse data.
 *
 * This function calculates the non-zero L2 hinge error for sparse data using the provided data and weight information.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory.
 */
__global__ void LAUNCH_BOUNDS() kCalculateSparseNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        s_pUnit[threadIdx.x] = pUnit[offset + pSparseIndex[pos1]];
        s_pSparseStart[threadIdx.x] = pSparseStart[dpos];
        s_pSparseEnd[threadIdx.x] = pSparseEnd[dpos];
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                Float diff = min((Float)0.0, a - (Float)1.0);
                a = max((Float)0.0, a);
                error += w * (diff * diff - a * a);
            }
        }
    }  

    REDUCEERROR(error)
}

/**
 * @brief Calculate the L2 hinge error for sparse data.
 *
 * This function calculates the L2 hinge error for sparse data using the provided data and weight information.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param bSparseIgnoreZero Flag indicating whether to ignore zero values in the sparse data.
 * 
 * @return The calculated L2 hinge error.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory and returns the calculated error scaled by `ONEOVERERRORSCALE`.
 */
Float kCalculateSparseL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (bSparseIgnoreZero)
    {
        uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateSparseOnlyNonZeroL2HingeError_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight);
        LAUNCHERROR("kCalculateSparseOnlyNonZeroL2HingeError_kernel");
    }
    else
    {
        uint64_t size = batch * stride;
        uint32_t blocks = CalculateBlocks(size);
        kCalculateSparseRawL2HingeError_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, pDataWeight, pUnit, stride, size);
        LAUNCHERROR("kCalculateSparseRawL2HingeError_kernel");

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[pos];
            s_pSparseEnd[i] = pSparseEnd[pos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                Float diff = min((Float)0.0, a - (Float)1.0);
                a = max((Float)0.0, a);
                error += (Float)w * (diff * diff - a * a);
            }
        }
    }

    REDUCEERROR(error)

    getGpu()._pbAccumulator->Download();
    return (Float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/**
 * @brief Calculate the L2 hinge error for sparse analog data (non-zero values only).
 *
 * This kernel function calculates the L2 hinge error for sparse analog data (non-zero values only).
 *
 * @tparam T The data type of the sparse data.
 * 
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 * 
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory.
 */
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                T t = pSparseData[i];
                Float diff = a - fabsf(t);
                diff = (t > (T)0.0) ? min((Float)0.0f , diff) : max((Float)0.0, diff);
                error += w * diff * diff;
            }
        }
    }

    REDUCEERROR(error)
}

/**
 * @brief Calculate the non-zero L2 hinge error for sparse analog data.
 *
 * This kernel function calculates the non-zero L2 hinge error for sparse analog data.
 *
 * @tparam T The data type of the sparse data.
 * 
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 * 
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory.
 */
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                T t = pSparseData[i];
                Float diff = a - fabsf(t);
                a = max((Float)0.0, a);
                diff = (t > (T)0.0) ? min((Float)0.0f , diff) : max((Float)0.0, diff);          
                error += w * (diff * diff - a * a);   
            }
        }
    }  

    REDUCEERROR(error)
}

/**
 * @brief Calculate the non-zero L2 hinge error for sparse analog data.
 *
 * This kernel function calculates the non-zero L2 hinge error for sparse analog data.
 * Only non-zero elements of the sparse data are considered.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, unsigned char* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[i] * (Float)(1.0 / 256.0);
                Float diff = a - t;
                diff = (t > (Float)0.0) ? min((Float)0.0f, diff) : max((Float)0.0, diff);
                error += w * diff * diff;
            }
        }
    }  

    REDUCEERROR(error)
}

/**
 * @brief Calculate the non-zero L2 hinge error for sparse analog data.
 *
 * This kernel function calculates the non-zero L2 hinge error for sparse analog data.
 * Only non-zero elements of the sparse data are considered.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, unsigned char* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[i] * (Float)(1.0 / 256.0);
                Float diff = a - t;
                diff = (t > (Float)0.0) ? min((Float)0.0f, diff) : max((Float)0.0, diff);
                a = max((Float)0.0, a);
                error += w * (diff * diff - a * a);
            }
        }
    }

    REDUCEERROR(error)
}

/**
 * @brief Calculate the non-zero L2 hinge error for sparse analog data.
 *
 * This kernel function calculates the non-zero L2 hinge error for sparse analog data.
 * Only non-zero elements of the sparse data are considered.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, char* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[i] * (Float)(1.0 / 128.0);
                Float diff = a - fabsf((Float)t);
                diff = (t > (Float)0.0) ? min((Float)0.0f, diff) : max((Float)0.0, diff);
                error += w * diff * diff;
            }
        }
    }

    REDUCEERROR(error)
}

/**
 * @brief Calculate the non-zero L2 hinge error for sparse analog data.
 *
 * This kernel function calculates the non-zero L2 hinge error for sparse analog data.
 * Only non-zero elements of the sparse data are considered.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, char* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[i] * (Float)(1.0 / 128.0);
                Float diff = a - fabsf(t);
                a = max((Float)0.0, a);
                diff = (t > (Float)0.0) ? min((Float)0.0f , diff) : max((Float)0.0, diff);       
                error += w * (diff * diff - a * a);   
            }
        }
    }

    REDUCEERROR(error)
}


/**
 * @brief Calculate the L2 hinge error for sparse analog data.
 *
 * This function calculates the L2 hinge error for sparse analog data.
 * It supports both zero and non-zero elements of the sparse data based on the specified flag.
 *
 * @tparam T The data type of the sparse data.
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 * @param bSparseIgnoreZero Flag indicating whether to ignore zero elements in the sparse data.
 * @return The calculated L2 hinge error.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory.
 */
template<typename T>
Float kCalculateSparseAnalogL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, Float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];
    
    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;
    
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;
    
        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();
    
        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[i] * (Float)(1.0 / 128.0);
                Float diff = a - fabsf(t);
                a = max((Float)0.0, a);
                diff = (t > (Float)0.0) ? min((Float)0.0f , diff) : max((Float)0.0, diff);       
                error += w * (diff * diff - a * a);   
            }
        }
    }  
    
    REDUCEERROR(error)
    
    return error;
}

/**
 * @brief Calculate the L2 hinge error for indexed sparse data with non-zero elements only.
 *
 * This kernel function calculates the L2 hinge error for indexed sparse data with non-zero elements only.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory.
 */
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{

    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];
    
    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;
    
    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;
    
        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();
    
        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)1.0 - a;
                Float diff = min((Float)0.0, t);
                error += w * diff * diff;   
            }
        }
    }  
    
    REDUCEERROR(error)
    
    return;
}

/**
 * @brief Calculate the non-zero L2 hinge error for indexed sparse data.
 *
 * This kernel function calculates the non-zero L2 hinge error for indexed sparse data.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory.
 */
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{

    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];
    
    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;
    
    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;
    
        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();
    
        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)1.0 - a;
                Float diff = min((Float)0.0, t);
                a = max((Float)0.0, a);
                diff = (t > (Float)0.0) ? min((Float)0.0f , diff) : max((Float)0.0, diff);
                error += w * (diff * diff - a * a); 
            }
        }
    }  
    
    REDUCEERROR(error)
    
    return;
}


/**
 * @brief Calculate the L2 hinge error for indexed sparse data.
 *
 * This function calculates the L2 hinge error for indexed sparse data.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param bSparseIgnoreZero Flag indicating whether to ignore zero values in the sparse data.
 *
 * @return The calculated L2 hinge error.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory and returns the calculated error value.
 */
Float kCalculateIndexedSparseL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];
    
    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;
    
    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;
    
        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();
    
        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)1.0 - a;
                Float diff = min((Float)0.0, t);
                error += w * (diff * diff - a * a); 
            }
        }
    }  
    
    REDUCEERROR(error)
    
    return error;
}

/**
 * @brief Calculate the L2 hinge error for indexed sparse analog data.
 *
 * This function calculates the L2 hinge error for indexed sparse analog data.
 *
 * @tparam T The data type of the sparse data.
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory and returns the calculated error value.
 */
template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData)
{

    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];
    
    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;
    
    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;
    
        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();
    
        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                T t = pSparseData[i];
                Float diff = a - fabsf(t);
                diff = (t > (T)0.0) ? min((Float)0.0f , diff) : max((Float)0.0, diff);         
                error += w * diff * diff;   
            }
        }
    }  
    
    REDUCEERROR(error)
}

/**
 * @brief Calculate the non-zero L2 hinge error for indexed sparse analog data.
 *
 * This function calculates the non-zero L2 hinge error for indexed sparse analog data.
 *
 * @tparam T The data type of the sparse data.
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory and returns the calculated error value.
 */
template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData)
{

    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];
    
    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;
    
    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;
    
        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();
    
        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                T t = pSparseData[i];
                Float diff = a - fabsf(t);
                a = max((Float)0.0, a);
                diff = (t > (T)0.0) ? min((Float)0.0f , diff) : max((Float)0.0, diff);          
                error += w * (diff * diff - a * a);               
            }
        }
    }  
    
    REDUCEERROR(error)
}

/**
 * @brief Calculate the non-zero L2 hinge error for indexed sparse analog data with unsigned char sparse data type.
 *
 * This function calculates the non-zero L2 hinge error for indexed sparse analog data with unsigned char sparse data type.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array of unsigned char type.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory and returns the calculated error value.
 */
template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, unsigned char* pSparseData)
{

    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];
    
    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;
    
    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;
    
        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();
    
        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[i] * (Float)(1.0 / 256.0);
                Float diff = a - t;
                diff = (t > (Float)0.0) ? min((Float)0.0f, diff) : max((Float)0.0, diff); 
                error += w * diff * diff;   
            }
        }
    }  
    
    REDUCEERROR(error)
}

/**
 * @brief Calculate the non-zero L2 hinge error for indexed sparse analog data with unsigned char sparse data type.
 *
 * This function calculates the non-zero L2 hinge error for indexed sparse analog data with unsigned char sparse data type.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array of unsigned char type.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory and returns the calculated error value.
 */
template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, unsigned char* pSparseData)
{

    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];
    
    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;
    
    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;
    
        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();
    
        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[i] * (Float)(1.0 / 256.0);
                Float diff = a - t;
                diff = (t > (Float)0.0) ? min((Float)0.0f, diff) : max((Float)0.0, diff);    
                a = max((Float)0.0, a);  
                error += w * (diff * diff - a * a);   
            }
        }
    }  
    
    REDUCEERROR(error)
}

/**
 * @brief Calculate the non-zero L2 hinge error for indexed sparse analog data with char sparse data type.
 *
 * This function calculates the non-zero L2 hinge error for indexed sparse analog data with char sparse data type.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array of char type.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory and returns the calculated error value.
 */
template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, char* pSparseData)
{

    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];
    
    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;
    
    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;
    
        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();
    
        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[i] * (Float)(1.0 / 128.0);
                Float diff = a - fabsf(t);
                diff = (t > (Float)0.0) ? min((Float)0.0f, diff) : max((Float)0.0, diff);      
                error += w * diff * diff;  
            }
        }
    }  
    
    REDUCEERROR(error)
}

/**
 * @brief Calculate the non-zero L2 hinge error for indexed sparse analog data with char sparse data type.
 *
 * This function calculates the non-zero L2 hinge error for indexed sparse analog data with char sparse data type.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array of char type.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` object and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory and returns the calculated error value.
 */
template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, char* pSparseData)
{

    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];
    
    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;
    
    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (Float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;
    
        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();
    
        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[i] * (Float)(1.0 / 128.0);
                Float diff = a - fabsf(t);
                a = max((Float)0.0, a);
                diff = (t > (Float)0.0) ? min((Float)0.0f , diff) : max((Float)0.0, diff);          
                error += w * (diff * diff - a * a);  
            }
        }
    }  
    
    REDUCEERROR(error)
}


/**
 * @brief Calculate the L2 hinge error for indexed sparse analog data with a generic sparse data type.
 *
 * This function calculates the L2 hinge error for indexed sparse analog data with a generic sparse data type.
 *
 * @tparam T The type of the sparse data.
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The number of data batches.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array of generic type T.
 * @param bSparseIgnoreZero Flag indicating whether to ignore zero values in sparse data.
 *
 * @remarks This function assumes that the GPU device is accessible through the `getGpu()` function and the device memory is properly allocated.
 * The `LAUNCHERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory and returns the calculated error value.
 */
template<typename T>
Float kCalculateIndexedSparseAnalogL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, Float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    if (bSparseIgnoreZero)
    {
        uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateIndexedSparseAnalogOnlyNonZeroL2HingeError_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
        LAUNCHERROR("kCalculateIndexedSparseAnalogOnlyNonZeroL2HingeError_kernel");    
    }
    else
    {
        uint64_t size = batch * stride;
        uint32_t blocks = CalculateBlocks(size);    
        kCalculateSparseRawL2HingeError_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, pDataWeight, pUnit, stride, size);
        LAUNCHERROR("kCalculateSparseRawL2HingeError_kernel");
        blocks = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateIndexedSparseAnalogNonZeroL2HingeError_kernel<<<blocks, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
        LAUNCHERROR("kCalculateIndexedSparseAnalogNonZeroL2HingeError_kernel");
    }
    getGpu()._pbAccumulator->Download(); 
    return (Float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/**
 * @brief Calculate the cross-entropy error for raw sparse data.
 *
 * This function calculates the cross-entropy error for raw sparse data.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param pDataWeight Pointer to the data weight array.
 * @param pUnit Pointer to the unit array.
 * @param stride The stride between consecutive data elements.
 * @param size The total size of the data.
 *
 * @remarks This function assumes that the GPU device is accessible through the `getGpu()` function and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory.
 */
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawCrossEntropyError_kernel(uint32_t position, Float* pDataWeight, Float* pUnit, uint32_t stride, uint64_t size)
{
    uint64_t pos = blockDim.x * blockIdx.x + threadIdx.x;
    Float error = (Float)0.0;
    if (pos < size)
    {
        Float w = (pDataWeight != NULL) ? pDataWeight[pos / stride] : (Float)1.0;
        Float a = pUnit[pos];
        error = -w * log(max(MIN_ERROR, (Float)1.0 - a));     
    }

    REDUCEERROR(error)
}

/**
 * @brief Calculate the cross-entropy error for sparse data with non-zero values only.
 *
 * This function calculates the cross-entropy error for sparse data with non-zero values only.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The batch size of the data.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` variable and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory.
 */
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseOnlyNonZeroCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{

    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];
    
    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;
    
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;
    
        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();
    
        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                Float a = s_pUnit[i];
                error += -w * log(max(MIN_ERROR, a));   
            }
        }
    }  

    REDUCEERROR(error)
}

/**
 * @brief Calculate the cross-entropy error for sparse data with non-zero values.
 *
 * This function calculates the cross-entropy error for sparse data with non-zero values.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The batch size of the data.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` variable and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory.
 */
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{

    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];
    
    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;
    
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;
    
        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();
    
        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                Float a = s_pUnit[i];
                error += w * (-log(max(MIN_ERROR, a)) + log(max(MIN_ERROR, (Float)1.0 - a)));   
            }
        }
    }  

    REDUCEERROR(error)
}

/**
 * @brief Calculate the cross-entropy error for sparse data.
 *
 * This function calculates the cross-entropy error for sparse data.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The batch size of the data.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param bSparseIgnoreZero Whether to ignore zero values in the sparse data.
 *
 * @return The cross-entropy error for the specified data.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` variable and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory and returns the calculated error.
 */
Float kCalculateSparseCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                Float a = s_pUnit[i];
                error += w * (-log(max(MIN_ERROR, a)) + log(max(MIN_ERROR, (Float)1.0 - a)));   
            }
        }
    }  

    REDUCEERROR(error)

    return (Float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/**
 * @brief Calculate the cross-entropy error for indexed sparse data ignoring zero values.
 *
 * This function calculates the cross-entropy error for indexed sparse data while ignoring zero values.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The batch size of the data.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` variable and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory.
 */
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseOnlyNonZeroCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                Float a = s_pUnit[i];
                error += -w * log(max(MIN_ERROR, a));   
            }
        }
    }  

    REDUCEERROR(error)
}

/**
 * @brief Calculate the non-zero cross-entropy error for indexed sparse data.
 *
 * This function calculates the non-zero cross-entropy error for indexed sparse data.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The batch size of the data.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` variable and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory.
 */
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                Float a = s_pUnit[i];
                error += w * (-log(max(MIN_ERROR, a)) + log(max(MIN_ERROR, (Float)1.0 - a)));   
            }
        }
    }  

    REDUCEERROR(error)
}

/**
 * @brief Calculate the cross-entropy error for indexed sparse data.
 *
 * This function calculates the cross-entropy error for indexed sparse data.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The batch size of the data.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param bSparseIgnoreZero Flag indicating whether to ignore zero values in the sparse data.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` variable and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory.
 *
 * @return The cross-entropy error value.
 */
Float kCalculateIndexedSparseCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                Float a = s_pUnit[i];
                error += w * (-log(max(MIN_ERROR, a)) + log(max(MIN_ERROR, (Float)1.0 - a)));   
            }
        }
    }  

    REDUCEERROR(error)

    return (Float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/**
 * @brief Calculate the multinomial cross-entropy error for sparse data.
 *
 * This function calculates the multinomial cross-entropy error for sparse data.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The batch size of the data.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` variable and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory.
 *
 * @return None.
 */
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                Float a = s_pUnit[i];
                error += -w * log(max(MIN_ERROR, a));   
            }
        }
    }  

    REDUCEERROR(error)

    return;
}

/**
 * @brief Calculate the multinomial cross-entropy error for sparse data.
 *
 * This function calculates the multinomial cross-entropy error for sparse data.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The batch size of the data.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` variable and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory and returns the calculated error value.
 *
 * @return The calculated multinomial cross-entropy error.
 */
Float kCalculateSparseMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, Float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(Float));

    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                Float a = s_pUnit[i];
                error += -w * log(max(MIN_ERROR, a));   
            }
        }
    }  

    REDUCEERROR(error)

    return error;
}

/**
 * @brief Calculate the indexed sparse multinomial cross-entropy error.
 *
 * This function calculates the indexed sparse multinomial cross-entropy error for a batch of data.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The batch size of the data.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` variable and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory and does not return any value.
 */
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                Float a = s_pUnit[i];
                error += -w * log(max(MIN_ERROR, a));   
            }
        }
    }  

    REDUCEERROR(error)

    return;
}

/**
 * @brief Calculate the indexed sparse multinomial cross-entropy error.
 *
 * This function calculates the indexed sparse multinomial cross-entropy error for a batch of data.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The batch size of the data.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` variable and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory and returns the calculated error value.
 *
 * @return The calculated indexed sparse multinomial cross-entropy error.
 */
Float kCalculateIndexedSparseMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, Float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(Float));

    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                Float a = s_pUnit[i];
                error += -w * log(max(MIN_ERROR, a));   
            }
        }
    }  

    REDUCEERROR(error)

    return error;
}

/**
 * @brief Calculate the sparse analog multinomial cross-entropy error.
 *
 * This function calculates the sparse analog multinomial cross-entropy error for a batch of data.
 *
 * @tparam T The data type of the sparse data.
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The batch size of the data.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` variable and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory and returns the calculated error value.
 *
 * @return The calculated sparse analog multinomial cross-entropy error.
 */
template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData)
{

    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                Float a = s_pUnit[i];
                T t = pSparseData[i];
                error += w * (-t * log(max(MIN_ERROR, a)));
            }
        }
    }  

    REDUCEERROR(error)
}

/**
 * @brief Calculate the sparse analog multinomial cross-entropy error for unsigned char sparse data.
 *
 * This specialization of the function calculates the sparse analog multinomial cross-entropy error for a batch of data with unsigned char sparse data.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The batch size of the data.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the unsigned char sparse data array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` variable and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory.
 */
template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, unsigned char* pSparseData)
{

    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[i] * (Float)(1.0 / 256.0);
                error += w * (-t * log(max(MIN_ERROR, a)));
            }
        }
    }  

    REDUCEERROR(error)
}

/**
 * @brief Calculate the sparse analog multinomial cross-entropy error for char sparse data.
 *
 * This specialization of the function calculates the sparse analog multinomial cross-entropy error for a batch of data with char sparse data.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The batch size of the data.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the char sparse data array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` variable and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory.
 */
template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, char* pSparseData)
{

    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[i] * (Float)(1.0 / 128.0);
                error += w * (-t * log(max(MIN_ERROR, a)));
            }
        }
    }  

    REDUCEERROR(error)
}


/**
 * @brief Calculate the sparse analog multinomial cross-entropy error for generic sparse data.
 *
 * This function calculates the sparse analog multinomial cross-entropy error for a batch of data with generic sparse data.
 *
 * @tparam T The data type of the sparse data.
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The batch size of the data.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the generic sparse data array.
 * @return The calculated error value.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` variable and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory and returns the calculated error value.
 */
template<typename T>
Float kCalculateSparseAnalogMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, Float* pDataWeight, T* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[i] * (Float)(1.0 / 128.0);
                error += w * (-t * log(max(MIN_ERROR, a)));
            }
        }
    }  

    REDUCEERROR(error)

    return error;
}

/**
 * @brief Calculate the indexed sparse analog multinomial cross-entropy error for generic sparse data.
 *
 * This function calculates the indexed sparse analog multinomial cross-entropy error for a batch of data with generic sparse data.
 *
 * @tparam T The data type of the sparse data.
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The batch size of the data.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the generic sparse data array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` variable and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory and does not return any value.
 */
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                Float a = s_pUnit[i];
                T t = pSparseData[i];
                error += w * (-t * log(max(MIN_ERROR, a)));
            }
        }
    }

    REDUCEERROR(error)

    return;
}

/**
 * @brief Calculate the indexed sparse analog multinomial cross-entropy error for sparse data of type 'unsigned char'.
 *
 * This function calculates the indexed sparse analog multinomial cross-entropy error for a batch of data with sparse data of type 'unsigned char'.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The batch size of the data.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array of type 'unsigned char'.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` variable and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory and does not return any value.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, unsigned char* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[i] * (Float)(1.0 / 256.0);
                error += w * (-t * log(max(MIN_ERROR, a)));
            }
        }
    }  

    REDUCEERROR(error)

    return;
}

/**
 * @brief Calculate the indexed sparse analog multinomial cross-entropy error for sparse data of type 'char'.
 *
 * This function calculates the indexed sparse analog multinomial cross-entropy error for a batch of data with sparse data of type 'char'.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The batch size of the data.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array of type 'char'.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` variable and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 * The function updates the error accumulator in GPU memory and does not return any value.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, char* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[i] * (Float)(1.0 / 128.0);
                error += w * (-t * log(max(MIN_ERROR, a)));
            }
        }
    }  

    REDUCEERROR(error)

    return;
}


/**
 * @brief Calculate the indexed sparse analog multinomial cross-entropy error for sparse data of type 'T'.
 *
 * This function calculates the indexed sparse analog multinomial cross-entropy error for a batch of data with sparse data of type 'T'.
 *
 * @tparam T The data type of the sparse data.
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The batch size of the data.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array of type 'T'.
 * @return The calculated error as an 'Float' value.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` variable and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 */
template<typename T>
Float kCalculateIndexedSparseAnalogMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, Float* pDataWeight, T* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                Float a = s_pUnit[i];
                Float t = (Float)pSparseData[i] * (Float)(1.0 / 128.0);
                error += w * (-t * log(max(MIN_ERROR, a)));
            }
        }
    }  

    REDUCEERROR(error)

    return error;
}

/**
 * @brief Calculate the sparse raw scaled marginal cross-entropy error.
 *
 * This function calculates the sparse raw scaled marginal cross-entropy error for a given position and size.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param pDataWeight Pointer to the data weight array.
 * @param pUnit Pointer to the unit array.
 * @param stride The stride between consecutive data elements.
 * @param size The size of the data.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` variable and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 */
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawScaledMarginalCrossEntropyError_kernel(uint32_t position, Float* pDataWeight, Float* pUnit, uint32_t stride, uint64_t size)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < size)
    {
        uint32_t dpos = pos / stride;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                Float a = s_pUnit[i];
                if (a > cData._SMCE_zeroTarget)
                    error = -w * log(max(MIN_ERROR, (Float)1.0 - a));     
            }
        }
    }  

    REDUCEERROR(error)

    return;
}

/**
 * @brief Calculate the sparse only non-zero scaled marginal cross-entropy error.
 *
 * This function calculates the sparse only non-zero scaled marginal cross-entropy error for a given position, batch, and stride.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The batch size.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` variable and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 */
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseOnlyNonZeroScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pos / stride;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = cData._SMCE_oneScale * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                Float a = s_pUnit[i];
                if (a < cData._SMCE_oneTarget)
                    error += -w * log(max(MIN_ERROR, a));
            }
        }
    }  

    REDUCEERROR(error)

    return;
}

/**
 * @brief Calculate the sparse non-zero scaled marginal cross-entropy error.
 *
 * This function calculates the sparse non-zero scaled marginal cross-entropy error for a given position, batch, and stride.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The batch size.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` variable and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 */
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pos / stride;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = cData._SMCE_oneScale * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                Float a = s_pUnit[i];
                if (a > cData._SMCE_zeroTarget)
                {
                    error += -w * cData._SMCE_zeroScale * log(max(MIN_ERROR, (Float)1.0 - a));
                }
                if (a < cData._SMCE_oneTarget)
                {
                    error += w * cData._SMCE_oneScale * log(max(MIN_ERROR, a));
                }
            }
        }
    }  

    REDUCEERROR(error)

    return;
}

/**
 * @brief Calculate the sparse scaled marginal cross-entropy error.
 *
 * This function calculates the sparse scaled marginal cross-entropy error for a given position, batch, and stride.
 *
 * @param position The starting position of the data for which the error is calculated.
 * @param batch The batch size.
 * @param stride The stride between consecutive data elements.
 * @param pUnit Pointer to the unit array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param bSparseIgnoreZero Flag indicating whether to ignore zero values in sparse data.
 *
 * @return The calculated error.
 *
 * @remarks This function assumes that the GPU device is accessible through the `cData` variable and the device memory is properly allocated.
 * The `REDUCEERROR` macro is expected to be defined elsewhere.
 */
Float kCalculateSparseScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pos / stride;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = cData._SMCE_oneScale * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                Float a = s_pUnit[i];
                if (bSparseIgnoreZero && a == 0.0)
                {
                    continue;
                }
                else if (a > cData._SMCE_zeroTarget)
                {
                    error += -w * cData._SMCE_zeroScale * log(max(MIN_ERROR, (Float)1.0 - a));
                }
                if (a < cData._SMCE_oneTarget)
                {
                    error += w * cData._SMCE_oneScale * log(max(MIN_ERROR, a));
                }
            }
        }
    }  

    REDUCEERROR(error)

    return error;
}

/**
 * \brief Kernel function for calculating indexed sparse-only non-zero scaled marginal cross-entropy error.
 *
 * This kernel function calculates the error using indexed sparse data and applies non-zero scaling and marginal cross-entropy error.
 *
 * \param position The position index.
 * \param batch The batch size.
 * \param stride The stride size.
 * \param pUnit Pointer to the unit data.
 * \param pIndex Pointer to the index data.
 * \param pSparseStart Pointer to the sparse start data.
 * \param pSparseEnd Pointer to the sparse end data.
 * \param pSparseIndex Pointer to the sparse index data.
 * \param pDataWeight Pointer to the data weight.
 */
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseOnlyNonZeroScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error               = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        Float w               = cData._SMCE_oneScale * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset         = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2       = offset + pSparseIndex[i];
                Float a           = s_pUnit[i];
                if (a < cData._SMCE_oneTarget)
                {
                    error          += -w * log(max(MIN_ERROR, a));
                }
            }
        }
    }  

    REDUCEERROR(error)

    return;
}

/**
 * \brief Kernel function for calculating indexed sparse non-zero scaled marginal cross-entropy error.
 *
 * This kernel function calculates the error using indexed sparse data and applies non-zero scaling and marginal cross-entropy error.
 *
 * \param position The position index.
 * \param batch The batch size.
 * \param stride The stride size.
 * \param pUnit Pointer to the unit data.
 * \param pIndex Pointer to the index data.
 * \param pSparseStart Pointer to the sparse start data.
 * \param pSparseEnd Pointer to the sparse end data.
 * \param pSparseIndex Pointer to the sparse index data.
 * \param pDataWeight Pointer to the data weight.
 */
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseNonZeroScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error               = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        Float w               = cData._SMCE_oneScale * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset         = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2       = offset + pSparseIndex[i];
                Float a           = s_pUnit[i];
                if (a > cData._SMCE_zeroTarget)
                {
                    error          += w * cData._SMCE_zeroScale * log(max(MIN_ERROR, (Float)1.0 - a));
                }
                if (a < cData._SMCE_oneTarget)
                {
                    error          += -w * cData._SMCE_oneScale * log(max(MIN_ERROR, a));
                }
            }
        }
    }  

    REDUCEERROR(error)

    return;
}

/**
 * \brief Calculate scaled marginal cross-entropy error using indexed sparse data.
 *
 * This function calculates the error using indexed sparse data and applies scaling and marginal cross-entropy error.
 *
 * \param position The position index.
 * \param batch The batch size.
 * \param stride The stride size.
 * \param pUnit Pointer to the unit data.
 * \param pIndex Pointer to the index data.
 * \param pSparseStart Pointer to the sparse start data.
 * \param pSparseEnd Pointer to the sparse end data.
 * \param pSparseIndex Pointer to the sparse index data.
 * \param pDataWeight Pointer to the data weight.
 * \param bSparseIgnoreZero Flag indicating whether to ignore zero values in sparse data.
 * \return The calculated error value.
 */
Float kCalculateIndexedSparseScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error               = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        Float w               = cData._SMCE_oneScale * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset         = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * stride + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2       = offset + pSparseIndex[i];
                Float a           = s_pUnit[i];
                if (bSparseIgnoreZero && a == 0.0)
                {
                    continue;
                }
                else if (a > cData._SMCE_zeroTarget)
                {
                    error          += w * cData._SMCE_zeroScale * log(max(MIN_ERROR, (Float)1.0 - a));
                }
                if (a < cData._SMCE_oneTarget)
                {
                    error          += -w * cData._SMCE_oneScale * log(max(MIN_ERROR, a));
                }
            }
        }
    }  

    REDUCEERROR(error)

    return error;
}

/**
 * \brief Kernel function for calculating scaled marginal cross-entropy error using sparse raw data.
 *
 * This kernel function calculates the error using sparse raw data and applies scaling and marginal cross-entropy error.
 *
 * \param pUnit Pointer to the unit data.
 * \param size The size of the data.
 */
__global__ void LAUNCH_BOUNDS() kCalculateSparseRawDataScaledMarginalCrossEntropyError_kernel(Float* pUnit, uint64_t size)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];

    uint64_t pos                = (blockDim.x * blockIdx.x + threadIdx.x) / cData._warpSize;
    Float error               = (Float)0.0;

    if (pos < size)
    {
        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * cData._warpSize + i];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pUnit[i] > cData._SMCE_zeroTarget)
            {
                error += -cData._SMCE_zeroScale * log(max(MIN_ERROR, (Float)1.0 - s_pUnit[i]));
            }
        }
    }

    REDUCEERROR(error)
}

/**
 * \brief Kernel function for calculating scaled marginal cross-entropy error using sparse non-zero data.
 *
 * This kernel function calculates the error using sparse non-zero data and applies scaling and marginal cross-entropy error.
 *
 * \tparam T Data type of the sparse data.
 * \param position The position index.
 * \param batch The batch size.
 * \param stride The stride size.
 * \param pUnit Pointer to the unit data.
 * \param pSparseStart Pointer to the sparse start data.
 * \param pSparseEnd Pointer to the sparse end data.
 * \param pSparseIndex Pointer to the sparse index data.
 * \param pSparseData Pointer to the sparse data.
 */
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSparseNonZeroDataScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error               = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * cData._warpSize + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2       = offset + pSparseIndex[i];
                Float a           = s_pUnit[i];
                T t                 = pSparseData[i];

                if (a > cData._SMCE_zeroTarget)
                {
                    error          += cData._SMCE_zeroScale * log(max(MIN_ERROR, (Float)1.0 - a));
                }

                if (a < cData._SMCE_oneTarget)
                {
                    error          += -cData._SMCE_oneScale * t * log(max(MIN_ERROR, a));
                }
            }
        }
    }

    REDUCEERROR(error)
}

/**
 * \brief Calculate scaled marginal cross-entropy error using sparse data.
 *
 * This function calculates the error using sparse data and applies scaling and marginal cross-entropy error.
 *
 * \tparam T Data type of the sparse data.
 * \param position The position index.
 * \param batch The batch size.
 * \param stride The stride size.
 * \param pUnit Pointer to the unit data.
 * \param pSparseStart Pointer to the sparse start data.
 * \param pSparseEnd Pointer to the sparse end data.
 * \param pSparseIndex Pointer to the sparse index data.
 * \param pSparseData Pointer to the sparse data.
 * \param bSparseIgnoreZero Flag indicating whether to ignore zero values in sparse data.
 * \return The calculated error value.
 */
template<typename T>
Float kCalculateSparseDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, T* pSparseData, bool bSparseIgnoreZero)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error               = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * cData._warpSize + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2       = offset + pSparseIndex[i];
                Float a           = s_pUnit[i];
                T t                 = pSparseData[i];

                if (bSparseIgnoreZero && a == 0.0)
                {
                    continue;
                }
                else if (a > cData._SMCE_zeroTarget)
                {
                    error          += cData._SMCE_zeroScale * log(max(MIN_ERROR, (Float)1.0 - a));
                }

                if (a < cData._SMCE_oneTarget)
                {
                    error          += -cData._SMCE_oneScale * t * log(max(MIN_ERROR, a));
                }
            }
        }
    }

    REDUCEERROR(error)

    return error;
}


/**
 * \brief Kernel function for calculating scaled marginal cross-entropy error using indexed sparse non-zero data.
 *
 * This kernel function calculates the error using indexed sparse non-zero data and applies scaling and marginal cross-entropy error.
 *
 * \tparam T Data type of the sparse data.
 * \param position The position index.
 * \param batch The batch size.
 * \param stride The stride size.
 * \param pUnit Pointer to the unit data.
 * \param pIndex Pointer to the index data.
 * \param pSparseStart Pointer to the sparse start data.
 * \param pSparseEnd Pointer to the sparse end data.
 * \param pSparseIndex Pointer to the sparse index data.
 * \param pSparseData Pointer to the sparse data.
 */
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseNonZeroDataScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error               = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * cData._warpSize + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2       = offset + pSparseIndex[i];
                Float a           = s_pUnit[i];
                T t                 = pSparseData[i];

                if (a > cData._SMCE_zeroTarget)
                {
                    error          += cData._SMCE_zeroScale * log(max(MIN_ERROR, (Float)1.0 - a));
                }

                if (a < cData._SMCE_oneTarget)
                {
                    error          += -cData._SMCE_oneScale * t * log(max(MIN_ERROR, a));
                }
            }
        }
    }

    REDUCEERROR(error)

    return;
}

/**
 * \brief Calculate scaled marginal cross-entropy error using indexed sparse data.
 *
 * This function calculates the error using indexed sparse data and applies scaling and marginal cross-entropy error.
 *
 * \tparam T Data type of the sparse data.
 * \param position The position index.
 * \param batch The batch size.
 * \param stride The stride size.
 * \param pUnit Pointer to the unit data.
 * \param pIndex Pointer to the index data.
 * \param pSparseStart Pointer to the sparse start data.
 * \param pSparseEnd Pointer to the sparse end data.
 * \param pSparseIndex Pointer to the sparse index data.
 * \param pSparseData Pointer to the sparse data.
 * \param bSparseIgnoreZero Flag indicating whether to ignore zero values in sparse data.
 * \return The calculated error value.
 */
template<typename T>
Float kCalculateIndexedSparseDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, T* pSparseData, bool bSparseIgnoreZero)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ bool s_bNonZero[cData._warpSize];

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error               = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * cData._warpSize + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2       = offset + pSparseIndex[i];
                Float a           = s_pUnit[i];
                T t                 = pSparseData[i];

                if (bSparseIgnoreZero && a == 0.0)
                {
                    continue;
                }
                else if (a > cData._SMCE_zeroTarget)
                {
                    error          += cData._SMCE_zeroScale * log(max(MIN_ERROR, (Float)1.0 - a));
                }

                if (a < cData._SMCE_oneTarget)
                {
                    error          += -cData._SMCE_oneScale * t * log(max(MIN_ERROR, a));
                }
            }
        }
    }

    REDUCEERROR(error)

    return error;
}

/**
 * \brief Kernel function for calculating scaled marginal cross-entropy error using sparse multinomial data.
 *
 * This kernel function calculates the error using sparse multinomial data and applies scaling and marginal cross-entropy error.
 *
 * \param position The position index.
 * \param batch The batch size.
 * \param stride The stride size.
 * \param pUnit Pointer to the unit data.
 * \param pSparseStart Pointer to the sparse start data.
 * \param pSparseEnd Pointer to the sparse end data.
 * \param pSparseIndex Pointer to the sparse index data.
 * \param pDataWeight Pointer to the data weight.
 */
__global__ void LAUNCH_BOUNDS() kCalculateSparseMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ Float s_w[cData._warpSize];

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error               = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        Float w               = cData._SMCE_oneScale * ((pDataWeight == NULL) ? (Float)1.0 / (Float)(end - pos1) : pDataWeight[dpos]);
        pos1                   += threadIdx.x & cData._warpMask;
        uint64_t offset         = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * cData._warpSize + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
            s_w[i] = w;
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2       = offset + pSparseIndex[i];
                Float a           = s_pUnit[i]; 
                if (a < cData._SMCE_oneTarget)
                    error          += -s_w[i] * log(max(MIN_ERROR, a));
            }
        }
    }  

    REDUCEERROR(error)
}

/**
 * \brief Calculate scaled marginal cross-entropy error using sparse multinomial data.
 *
 * This function calculates the error using sparse multinomial data and applies scaling and marginal cross-entropy error.
 *
 * \param position The position index.
 * \param batch The batch size.
 * \param stride The stride size.
 * \param pUnit Pointer to the unit data.
 * \param pSparseStart Pointer to the sparse start data.
 * \param pSparseEnd Pointer to the sparse end data.
 * \param pSparseIndex Pointer to the sparse index data.
 * \param pDataWeight Pointer to the data weight.
 * \return The calculated error value.
 */
Float kCalculateSparseMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, Float* pDataWeight)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ Float s_w[cData._warpSize];

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error               = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        Float w               = cData._SMCE_oneScale * ((pDataWeight == NULL) ? (Float)1.0 / (Float)(end - pos1) : pDataWeight[dpos]);
        pos1                   += threadIdx.x & cData._warpMask;
        uint64_t offset         = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * cData._warpSize + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
            s_w[i] = w;
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2       = offset + pSparseIndex[i];
                Float a           = s_pUnit[i]; 
                if (a < cData._SMCE_oneTarget)
                    error          += -s_w[i] * log(max(MIN_ERROR, a));
            }
        }
    }  

    REDUCEERROR(error)

    return error;
}

/**
 * \brief Calculate indexed sparse multinomial scaled marginal cross-entropy error.
 *
 * This function calculates the error using indexed sparse multinomial data and applies scaling and marginal cross-entropy error.
 *
 * \param position The position index.
 * \param batch The batch size.
 * \param stride The stride size.
 * \param pUnit Pointer to the unit data.
 * \param pIndex Pointer to the index data.
 * \param pSparseStart Pointer to the sparse start data.
 * \param pSparseEnd Pointer to the sparse end data.
 * \param pSparseIndex Pointer to the sparse index data.
 * \param pDataWeight Pointer to the data weight.
 */
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ Float s_w[cData._warpSize];

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error               = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        Float w               = cData._SMCE_oneScale * ((pDataWeight == NULL) ? (Float)1.0 / (Float)(end - pos1) : pDataWeight[dpos]);
        pos1                   += threadIdx.x & cData._warpMask;
        uint64_t offset         = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * cData._warpSize + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
            s_w[i] = w;
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2       = offset + pSparseIndex[i];
                Float a           = s_pUnit[i]; 
                if (a < cData._SMCE_oneTarget)
                    error          += -s_w[i] * log(max(MIN_ERROR, a));
            }
        }
    }  

    REDUCEERROR(error)

    return;
}

/**
 * \brief Calculate indexed sparse multinomial scaled marginal cross-entropy error.
 *
 * This function calculates the error using indexed sparse multinomial data and applies scaling and marginal cross-entropy error.
 *
 * \param position The position index.
 * \param batch The batch size.
 * \param stride The stride size.
 * \param pUnit Pointer to the unit data.
 * \param pIndex Pointer to the index data.
 * \param pSparseStart Pointer to the sparse start data.
 * \param pSparseEnd Pointer to the sparse end data.
 * \param pSparseIndex Pointer to the sparse index data.
 * \param pDataWeight Pointer to the data weight.
 * \return The calculated error.
 */
Float kCalculateIndexedSparseMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ Float s_w[cData._warpSize];

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error               = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        Float w               = cData._SMCE_oneScale * ((pDataWeight == NULL) ? (Float)1.0 / (Float)(end - pos1) : pDataWeight[dpos]);
        pos1                   += threadIdx.x & cData._warpMask;
        uint64_t offset         = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * cData._warpSize + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
            s_w[i] = w;
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2       = offset + pSparseIndex[i];
                Float a           = s_pUnit[i]; 
                if (a < cData._SMCE_oneTarget)
                    error          += -s_w[i] * log(max(MIN_ERROR, a));
            }
        }
    }  

    REDUCEERROR(error)

    return error;
}

/**
 * \brief Calculate sparse analog multinomial scaled marginal cross-entropy error.
 *
 * This function calculates the error using sparse analog multinomial data and applies scaling and marginal cross-entropy error.
 *
 * \tparam T The type of the sparse data.
 * \param position The position index.
 * \param batch The batch size.
 * \param stride The stride size.
 * \param pUnit Pointer to the unit data.
 * \param pSparseStart Pointer to the sparse start data.
 * \param pSparseEnd Pointer to the sparse end data.
 * \param pSparseIndex Pointer to the sparse index data.
 * \param pDataWeight Pointer to the data weight.
 * \param pSparseData Pointer to the sparse data.
 */
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ Float s_w[cData._warpSize];
    __shared__ T s_pSparseData[cData._warpSize];

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error               = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        Float w               = cData._SMCE_oneScale * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset         = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * cData._warpSize + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
            s_w[i] = w;
            s_pSparseData[i] = pSparseData[i];
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2       = offset + pSparseIndex[i];
                Float a           = s_pUnit[i];
                T t                 = s_pSparseData[i];  
                if (a < cData._SMCE_oneTarget)
                    error          += -w * t * log(max(MIN_ERROR, a));
            }
        }
    }  

    REDUCEERROR(error)
}

/**
 * \brief Calculate sparse analog multinomial scaled marginal cross-entropy error for unsigned char sparse data.
 *
 * This specialization of the function calculates the error using sparse analog multinomial data with unsigned char sparse data type.
 * It applies scaling and marginal cross-entropy error.
 *
 * \param position The position index.
 * \param batch The batch size.
 * \param stride The stride size.
 * \param pUnit Pointer to the unit data.
 * \param pSparseStart Pointer to the sparse start data.
 * \param pSparseEnd Pointer to the sparse end data.
 * \param pSparseIndex Pointer to the sparse index data.
 * \param pDataWeight Pointer to the data weight.
 * \param pSparseData Pointer to the unsigned char sparse data.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, unsigned char* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ Float s_w[cData._warpSize];
    __shared__ Float s_pSparseData[cData._warpSize];

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error               = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        Float w               = cData._SMCE_oneScale * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset         = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * cData._warpSize + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
            s_w[i] = w;
            s_pSparseData[i] = pSparseData[i] * (Float)(1.0 / 256.0);
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2       = offset + pSparseIndex[i];
                Float a           = s_pUnit[i];
                if (a < cData._SMCE_oneTarget)
                    error          += -w * s_pSparseData[i] * log(max(MIN_ERROR, a));
            }
        }
    }  

    REDUCEERROR(error)
}

/**
 * \brief Calculate sparse analog multinomial scaled marginal cross-entropy error for char sparse data.
 *
 * This specialization of the function calculates the error using sparse analog multinomial data with char sparse data type.
 * It applies scaling and marginal cross-entropy error.
 *
 * \param position The position index.
 * \param batch The batch size.
 * \param stride The stride size.
 * \param pUnit Pointer to the unit data.
 * \param pSparseStart Pointer to the sparse start data.
 * \param pSparseEnd Pointer to the sparse end data.
 * \param pSparseIndex Pointer to the sparse index data.
 * \param pDataWeight Pointer to the data weight.
 * \param pSparseData Pointer to the char sparse data.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, char* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ Float s_w[cData._warpSize];
    __shared__ Float s_pSparseData[cData._warpSize];

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error               = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        Float w               = cData._SMCE_oneScale * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset         = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * cData._warpSize + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
            s_w[i] = w;
            s_pSparseData[i] = pSparseData[i] * (Float)(1.0 / 128.0);
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2       = offset + pSparseIndex[i];
                Float a           = s_pUnit[i];
                if (a < cData._SMCE_oneTarget)
                    error          += -w * s_pSparseData[i] * log(max(MIN_ERROR, a));
            }
        }
    }  

    REDUCEERROR(error)
}

/**
 * \brief Calculate sparse analog multinomial scaled marginal cross-entropy error for general sparse data type.
 *
 * This function calculates the error using sparse analog multinomial data with a generic sparse data type.
 * It applies scaling and marginal cross-entropy error.
 *
 * \tparam T The type of the sparse data.
 * \param position The position index.
 * \param batch The batch size.
 * \param stride The stride size.
 * \param pUnit Pointer to the unit data.
 * \param pSparseStart Pointer to the sparse start data.
 * \param pSparseEnd Pointer to the sparse end data.
 * \param pSparseIndex Pointer to the sparse index data.
 * \param pDataWeight Pointer to the data weight.
 * \param pSparseData Pointer to the sparse data.
 * \return The calculated error.
 */
template<typename T>
Float kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, Float* pDataWeight, T* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ Float s_w[cData._warpSize];
    __shared__ T s_pSparseData[cData._warpSize];

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error               = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        Float w               = cData._SMCE_oneScale * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset         = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * cData._warpSize + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
            s_w[i] = w;
            s_pSparseData[i] = pSparseData[i] * (Float)(1.0 / 128.0);
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2       = offset + pSparseIndex[i];
                Float a           = s_pUnit[i];
                if (a < cData._SMCE_oneTarget)
                    error          += -w * s_pSparseData[i] * log(max(MIN_ERROR, a));
            }
        }
    }  

    REDUCEERROR(error)

    return error;
}


/**
 * \brief Calculate indexed sparse analog multinomial scaled marginal cross-entropy error for general sparse data type.
 *
 * This function calculates the error using indexed sparse analog multinomial data with a generic sparse data type.
 * It applies scaling and marginal cross-entropy error.
 *
 * \tparam T The type of the sparse data.
 * \param position The position index.
 * \param batch The batch size.
 * \param stride The stride size.
 * \param pUnit Pointer to the unit data.
 * \param pIndex Pointer to the index data.
 * \param pSparseStart Pointer to the sparse start data.
 * \param pSparseEnd Pointer to the sparse end data.
 * \param pSparseIndex Pointer to the sparse index data.
 * \param pDataWeight Pointer to the data weight.
 * \param pSparseData Pointer to the sparse data.
 */
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ Float s_w[cData._warpSize];
    __shared__ T s_pSparseData[cData._warpSize];

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error               = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        Float w               = cData._SMCE_oneScale * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset         = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * cData._warpSize + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
            s_w[i] = w;
            s_pSparseData[i] = pSparseData[i] * (Float)(1.0 / 128.0);
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2       = offset + pSparseIndex[i];
                Float a           = s_pUnit[i];
                if (a < cData._SMCE_oneTarget)
                    error          += -w * s_pSparseData[i] * log(max(MIN_ERROR, a));
            }
        }
    }  

    REDUCEERROR(error)

    return;
}

/**
 * \brief CUDA kernel to calculate indexed sparse analog multinomial scaled marginal cross-entropy error.
 *
 * \tparam Unused template parameter.
 *
 * \param position The position.
 * \param batch The batch size.
 * \param stride The stride.
 * \param pUnit Pointer to the unit data.
 * \param pIndex Pointer to the index data.
 * \param pSparseStart Pointer to the sparse start data.
 * \param pSparseEnd Pointer to the sparse end data.
 * \param pSparseIndex Pointer to the sparse index data.
 * \param pDataWeight Pointer to the data weight.
 * \param pSparseData Pointer to the sparse data.
 */
template<>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel(
    uint32_t position,
    uint32_t batch,
    uint32_t stride,
    Float *pUnit,
    uint32_t* pIndex,
    uint64_t* pSparseStart,
    uint64_t* pSparseEnd,
    uint32_t* pSparseIndex,
    Float* pDataWeight,
    unsigned char* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ Float s_w[cData._warpSize];
    __shared__ unsigned char s_pSparseData[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = cData._SMCE_oneScale * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * cData._warpSize + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
            s_w[i] = w;
            s_pSparseData[i] = pSparseData[i] * (Float)(1.0 / 256.0);
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                Float a = s_pUnit[i];
                if (a < cData._SMCE_oneTarget)
                    error += -w * s_pSparseData[i] * log(max(MIN_ERROR, a));
            }
        }
    }

    REDUCEERROR(error)

    return;
}

/**
 * \brief CUDA kernel to calculate indexed sparse analog multinomial scaled marginal cross-entropy error.
 *
 * \tparam Unused template parameter.
 *
 * \param position The position.
 * \param batch The batch size.
 * \param stride The stride.
 * \param pUnit Pointer to the unit data.
 * \param pIndex Pointer to the index data.
 * \param pSparseStart Pointer to the sparse start data.
 * \param pSparseEnd Pointer to the sparse end data.
 * \param pSparseIndex Pointer to the sparse index data.
 * \param pDataWeight Pointer to the data weight.
 * \param pSparseData Pointer to the sparse data.
 */
template<>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel(
    uint32_t position,
    uint32_t batch,
    uint32_t stride,
    Float *pUnit,
    uint32_t* pIndex,
    uint64_t* pSparseStart,
    uint64_t* pSparseEnd,
    uint32_t* pSparseIndex,
    Float* pDataWeight,
    char* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ Float s_w[cData._warpSize];
    __shared__ char s_pSparseData[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = cData._SMCE_oneScale * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * cData._warpSize + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
            s_w[i] = w;
            s_pSparseData[i] = pSparseData[i] * (char)(1.0 / 256.0);
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                Float a = s_pUnit[i];
                if (a < cData._SMCE_oneTarget)
                    error += -w * s_pSparseData[i] * log(max(MIN_ERROR, a));
            }
        }
    }

    REDUCEERROR(error)

    return;
}

/**
 * \brief Calculate indexed sparse analog multinomial scaled marginal cross-entropy error.
 *
 * \tparam T The type of the sparse data.
 *
 * \param position The position.
 * \param batch The batch size.
 * \param stride The stride.
 * \param pUnit Pointer to the unit data.
 * \param pIndex Pointer to the index data.
 * \param pSparseStart Pointer to the sparse start data.
 * \param pSparseEnd Pointer to the sparse end data.
 * \param pSparseIndex Pointer to the sparse index data.
 * \param pDataWeight Pointer to the data weight.
 * \param pSparseData Pointer to the sparse data.
 *
 * \return The calculated error.
 */
template<typename T>
Float kCalculateIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError(
    uint32_t position,
    uint32_t batch,
    uint32_t stride,
    Float* pUnit,
    uint32_t* pIndex,
    uint64_t* pSparseStart,
    uint64_t* pSparseEnd,
    uint32_t* pSparseIndex,
    Float* pDataWeight,
    T* pSparseData)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ uint64_t s_pSparseStart[cData._warpSize];
    __shared__ uint64_t s_pSparseEnd[cData._warpSize];
    __shared__ Float s_w[cData._warpSize];
    __shared__ T s_pSparseData[cData._warpSize];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    Float error = (Float)0.0;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        Float w = cData._SMCE_oneScale * ((pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0);
        uint64_t offset = pos * stride;

        // Load the data into shared memory.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            s_pUnit[i] = pUnit[pos * cData._warpSize + i];
            s_pSparseStart[i] = pSparseStart[dpos];
            s_pSparseEnd[i] = pSparseEnd[dpos];
            s_w[i] = w;
            s_pSparseData[i] = pSparseData[i] * (T)(1.0 / 256.0);
        }
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            if (s_pSparseStart[i] < s_pSparseEnd[i])
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                Float a = s_pUnit[i];
                if (a < cData._SMCE_oneTarget)
                    error += -w * s_pSparseData[i] * log(max(MIN_ERROR, a));
            }
        }
    }

    REDUCEERROR(error)

    return error;
}

/**
 * \brief CUDA kernel to calculate L1 error.
 *
 * \tparam T The type of the data.
 *
 * \param position The position.
 * \param stride The stride.
 * \param pUnit Pointer to the unit data.
 * \param pData Pointer to the data.
 * \param pDataWeight Pointer to the data weight.
 */
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateL1Error_kernel(
    uint32_t position,
    uint32_t stride,
    Float* pUnit,
    T* pData,
    Float* pDataWeight)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ T s_pData[cData._warpSize];
    __shared__ Float s_w[cData._warpSize];

    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    Float error = (Float)0.0;

    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        Float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        s_pUnit[threadIdx.x] = a;
        s_pData[threadIdx.x] = t;
        s_w[threadIdx.x] = w;
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            error += s_w[i] * fabsf(s_pUnit[i] - s_pData[i]);
        }
    }

    REDUCEERROR(error)
}

/**
 * \brief CUDA kernel to calculate L1 error.
 *
 * \tparam Unused template parameter.
 *
 * \param position The position.
 * \param stride The stride.
 * \param pUnit Pointer to the unit data.
 * \param pData Pointer to the data.
 * \param pDataWeight Pointer to the data weight.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateL1Error_kernel(
    uint32_t position,
    uint32_t stride,
    Float* pUnit,
    unsigned char* pData,
    Float* pDataWeight)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ unsigned char s_pData[cData._warpSize];
    __shared__ Float s_w[cData._warpSize];

    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    Float error = (Float)0.0;

    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        Float a = pUnit[uOffset + pos];
        unsigned char t = pData[dOffset + pos];
        s_pUnit[threadIdx.x] = a;
        s_pData[threadIdx.x] = t;
        s_w[threadIdx.x] = w;
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            error += s_w[i] * fabsf(s_pUnit[i] - s_pData[i]);
        }
    }

    REDUCEERROR(error)
}

/**
 * \brief CUDA kernel to calculate L1 error.
 *
 * \tparam Unused template parameter.
 *
 * \param position The position.
 * \param stride The stride.
 * \param pUnit Pointer to the unit data.
 * \param pData Pointer to the data.
 * \param pDataWeight Pointer to the data weight.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateL1Error_kernel(
    uint32_t position,
    uint32_t stride,
    Float* pUnit,
    char* pData,
    Float* pDataWeight)
{
    // Use shared memory to reduce global memory accesses.
    __shared__ Float s_pUnit[cData._warpSize];
    __shared__ char s_pData[cData._warpSize];
    __shared__ Float s_w[cData._warpSize];

    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    Float error = (Float)0.0;

    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        Float a = pUnit[uOffset + pos];
        char t = pData[dOffset + pos];
        s_pUnit[threadIdx.x] = a;
        s_pData[threadIdx.x] = t;
        s_w[threadIdx.x] = w;
        __syncthreads();

        // Calculate the error.
        for (uint32_t i = 0; i < cData._warpSize; i++)
        {
            error += s_w[i] * fabsf(s_pUnit[i] - s_pData[i]);
        }
    }

    REDUCEERROR(error)
}

/**
 * \brief Calculate L1 error.
 *
 * \tparam T The type of the data.
 *
 * \param position The position.
 * \param batch The batch size.
 * \param stride The stride.
 * \param pUnit Pointer to the unit data.
 * \param pData Pointer to the data.
 * \param pDataWeight Pointer to the data weight.
 *
 * \return The calculated error.
 */
template<typename T>
Float kCalculateL1Error(
    uint32_t position,
    uint32_t batch,
    uint32_t stride,
    Float* pUnit,
    T* pData,
    Float* pDataWeight) {

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ T s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    T t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      error += s_w[i] * fabsf(s_pUnit[i] - s_pData[i]);
    }
  }

  REDUCEERROR(error)

  return error;
}

/**
 * \brief CUDA kernel to calculate indexed L1 error.
 *
 * \tparam T The type of the data.
 *
 * \param position The position.
 * \param batch The batch size.
 * \param stride The stride.
 * \param pUnit Pointer to the unit data.
 * \param pIndex Pointer to the index data.
 * \param pData Pointer to the data.
 * \param pDataWeight Pointer to the data weight.
 */
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedL1Error_kernel(
    uint32_t position,
    uint32_t batch,
    uint32_t stride,
    Float* pUnit,
    uint32_t* pIndex,
    T* pData,
    Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ uint32_t s_pIndex[cData._warpSize];
  __shared__ T s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    T t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pIndex[threadIdx.x] = pIndex[dpos];
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      error += s_w[i] * fabsf(s_pUnit[i] - s_pData[i]);
    }
  }

  REDUCEERROR(error)

}

/**
 * \brief CUDA kernel to calculate indexed L1 error.
 *
 * \tparam Unused template parameter.
 *
 * \param position The position.
 * \param batch The batch size.
 * \param stride The stride.
 * \param pUnit Pointer to the unit data.
 * \param pIndex Pointer to the index data.
 * \param pData Pointer to the data.
 * \param pDataWeight Pointer to the data weight.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedL1Error_kernel(
    uint32_t position,
    uint32_t batch,
    uint32_t stride,
    Float* pUnit,
    uint32_t* pIndex,
    unsigned char* pData,
    Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ uint32_t s_pIndex[cData._warpSize];
  __shared__ unsigned char s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    unsigned char t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pIndex[threadIdx.x] = pIndex[dpos];
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      error += s_w[i] * fabsf(s_pUnit[i] - s_pData[i]);
    }
  }

  REDUCEERROR(error)

}

/**
 * \brief CUDA kernel to calculate indexed L1 error.
 *
 * \tparam Unused template parameter.
 *
 * \param position The position.
 * \param batch The batch size.
 * \param stride The stride.
 * \param pUnit Pointer to the unit data.
 * \param pIndex Pointer to the index data.
 * \param pData Pointer to the data.
 * \param pDataWeight Pointer to the data weight.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedL1Error_kernel(
    uint32_t position,
    uint32_t batch,
    uint32_t stride,
    Float* pUnit,
    uint32_t* pIndex,
    char* pData,
    Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ uint32_t s_pIndex[cData._warpSize];
  __shared__ char s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    char t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pIndex[threadIdx.x] = pIndex[dpos];
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      error += s_w[i] * fabsf(s_pUnit[i] - s_pData[i]);
    }
  }

  REDUCEERROR(error)

}

/**
 * \brief Calculate indexed L1 error.
 *
 * \tparam T The type of the data.
 *
 * \param position The position.
 * \param batch The batch size.
 * \param stride The stride.
 * \param pUnit Pointer to the unit data.
 * \param pIndex Pointer to the index data.
 * \param pData Pointer to the data.
 * \param pDataWeight Pointer to the data weight.
 *
 * \return The calculated error.
 */
template<typename T>
Float kCalculateIndexedL1Error(
    uint32_t position,
    uint32_t batch,
    uint32_t stride,
    Float* pUnit,
    uint32_t* pIndex,
    T* pData,
    Float* pDataWeight) {

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ uint32_t s_pIndex[cData._warpSize];
  __shared__ T s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    T t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pIndex[threadIdx.x] = pIndex[dpos];
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      error += s_w[i] * fabsf(s_pUnit[i] - s_pData[i]);
    }
  }

  return error;
}

/**
 * \brief CUDA kernel to calculate L2 error.
 *
 * \tparam T The type of the data.
 *
 * \param position The position.
 * \param batch The batch size.
 * \param stride The stride.
 * \param pUnit Pointer to the unit data.
 * \param pData Pointer to the data.
 * \param pDataWeight Pointer to the data weight.
 *
 * \return The calculated error.
 */
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateL2Error_kernel(
    uint32_t position,
    uint32_t batch,
    uint32_t stride,
    Float* pUnit,
    T* pData,
    Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ T s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    T t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      error += s_w[i] * (s_pUnit[i] - s_pData[i]) * (s_pUnit[i] - s_pData[i]);
    }
  }

  return error;
}

/**
 * \brief CUDA kernel to calculate L2 error.
 *
 * \param position The position.
 * \param batch The batch size.
 * \param stride The stride.
 * \param pUnit Pointer to the unit data.
 * \param pData Pointer to the data.
 * \param pDataWeight Pointer to the data weight.
 *
 * \return The calculated error.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateL2Error_kernel(
    uint32_t position,
    uint32_t batch,
    uint32_t stride,
    Float* pUnit,
    unsigned char* pData,
    Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ unsigned char s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    unsigned char t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      error += s_w[i] * (s_pUnit[i] - s_pData[i]) * (s_pUnit[i] - s_pData[i]);
    }
  }

  return error;
}

/**
 * \brief Calculates the L2 error for a given position, batch, and stride.
 *
 * \tparam Unused template parameter.
 * \param position The position value.
 * \param batch The batch value.
 * \param stride The stride value.
 * \param pUnit Pointer to the unit array.
 * \param pData Pointer to the data array.
 * \param pDataWeight Pointer to the data weight array.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, char* pData, Float* pDataWeight)
{
  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize]; /**< Shared memory for pUnit array. */
  __shared__ char s_pData[cData._warpSize]; /**< Shared memory for pData array. */
  __shared__ Float s_w[cData._warpSize]; /**< Shared memory for weights. */

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x; /**< Position in the grid. */
  Float error = (Float)0.0; /**< Accumulated error. */

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride; /**< Unit array offset. */
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x; /**< Data position. */
    uint64_t dOffset = dpos * stride; /**< Data array offset. */
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0; /**< Data weight value. */
    Float a = pUnit[uOffset + pos]; /**< Unit value. */
    char t = pData[dOffset + pos]; /**< Data value. */
    s_pUnit[threadIdx.x] = a;
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      error += s_w[i] * (s_pUnit[i] - s_pData[i]) * (s_pUnit[i] - s_pData[i]);
    }
  }

  return error;
}

/**
 * Calculates the L2 error for a given position in a batch.
 *
 * @tparam T The data type of `pData`.
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight.
 * @return The L2 error.
 */
template<typename T>
Float kCalculateL2Error(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, T* pData, Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ T s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    T t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      error += s_w[i] * (s_pUnit[i] - s_pData[i]) * (s_pUnit[i] - s_pData[i]);
    }
  }

  return error;
}

/**
 * Calculates the L2 error for a given position in a batch using indexed data.
 *
 * @tparam T The data type of `pData`.
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit data.
 * @param pIndex Pointer to the index data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight.
 */
template<typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, T* pData, Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ uint32_t s_pIndex[cData._warpSize];
  __shared__ T s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    T t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pIndex[threadIdx.x] = pIndex[dpos];
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      error += s_w[i] * (s_pUnit[i] - s_pData[i]) * (s_pUnit[i] - s_pData[i]);
    }
  }

  return error;
}

/**
 * Calculates the indexed L2 error for a given position in a batch using unsigned char data.
 *
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit data.
 * @param pIndex Pointer to the index data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight.
 */
template<>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, unsigned char* pData, Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ uint32_t s_pIndex[cData._warpSize];
  __shared__ unsigned char s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    unsigned char t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pIndex[threadIdx.x] = pIndex[dpos];
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      error += s_w[i] * (s_pUnit[i] - s_pData[i]) * (s_pUnit[i] - s_pData[i]);
    }
  }

  return error;
}

/**
 * Calculates the indexed L2 error for a given position in a batch using char data.
 *
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit data.
 * @param pIndex Pointer to the index data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight.
 */
template<>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, char* pData, Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ uint32_t s_pIndex[cData._warpSize];
  __shared__ char s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    char t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pIndex[threadIdx.x] = pIndex[dpos];
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      error += s_w[i] * (s_pUnit[i] - s_pData[i]) * (s_pUnit[i] - s_pData[i]);
    }
  }

  return error;
}

/**
 * Calculates the indexed L2 error for a given position in a batch.
 *
 * @tparam T The data type of `pData`.
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit data.
 * @param pIndex Pointer to the index data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight.
 * @return The L2 error.
 */
template<typename T>
Float kCalculateIndexedL2Error(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, T* pData, Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ uint32_t s_pIndex[cData._warpSize];
  __shared__ T s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    T t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pIndex[threadIdx.x] = pIndex[dpos];
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      error += s_w[i] * (s_pUnit[i] - s_pData[i]) * (s_pUnit[i] - s_pData[i]);
    }
  }

  return error;
}

/**
 * Calculates the L2 hinge error for a given position in a batch.
 *
 * @tparam T The data type of `pData`.
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight.
 */
template<typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, T* pData, Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ T s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    T t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      Float diff = s_pUnit[i] - fabsf(s_pData[i]);
      diff = (s_pData[i] > (T)0.0) ? min((Float)0.0f, diff) : max((Float)0.0, diff);
      error += s_w[i] * diff * diff;
    }
  }

  return error;
}

/**
 * Calculates the L2 hinge error for a given position in a batch using unsigned char data.
 *
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight.
 */
template<>
__global__ void LAUNCH_BOUNDS()
kCalculateL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, unsigned char* pData, Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ unsigned char s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    unsigned char t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      Float diff = s_pUnit[i] - t;
      diff = (t > (unsigned char)0) ? min((Float)0.0f, diff) : max((Float)0.0, diff);
      error += s_w[i] * diff * diff;
    }
  }

  return error;
}

/**
 * Calculates the L2 hinge error for a given position in a batch using char data.
 *
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight.
 */
template<>
__global__ void LAUNCH_BOUNDS()
kCalculateL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, char* pData, Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ char s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    char t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      Float diff = s_pUnit[i] - t;
      diff = (t > (char)0) ? min((Float)0.0f, diff) : max((Float)0.0, diff);
      error += s_w[i] * diff * diff;
    }
  }

  return error;
}

/**
 * Calculates the L2 hinge error for a given position in a batch.
 *
 * @tparam T The data type of `pData`.
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight.
 * @return The L2 hinge error.
 */
template<typename T>
Float kCalculateL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, T* pData, Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ T s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    T t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      Float diff = s_pUnit[i] - t;
      diff = (t > (T)0) ? min((Float)0.0f, diff) : max((Float)0.0, diff);
      error += s_w[i] * diff * diff;
    }
  }

  return error;
}

/**
 * Calculates the indexed L2 hinge error for a given position in a batch.
 *
 * @tparam T The data type of `pData`.
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit data.
 * @param pIndex Pointer to the index data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight.
 */
template<typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, T* pData, Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ uint32_t s_pIndex[cData._warpSize];
  __shared__ T s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    T t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pIndex[threadIdx.x] = pIndex[dpos];
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      Float diff = s_pUnit[i] - fabsf(s_pData[i]);
      diff = (s_pData[i] > (T)0) ? min((Float)0.0f, diff) : max((Float)0.0, diff);
      error += s_w[i] * diff * diff;
    }
  }

  return error;
}

/**
 * Calculates the indexed L2 hinge error for a given position in a batch using unsigned char data.
 *
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit data.
 * @param pIndex Pointer to the index data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight.
 */
template<>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, unsigned char* pData, Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ uint32_t s_pIndex[cData._warpSize];
  __shared__ unsigned char s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    unsigned char t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pIndex[threadIdx.x] = pIndex[dpos];
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      Float diff = s_pUnit[i] - s_pData[i];
      diff = (s_pData[i] > (unsigned char)0) ? min((Float)0.0f, diff) : max((Float)0.0, diff);
      error += s_w[i] * diff * diff;
    }
  }

  return error;
}

/**
 * Calculates the indexed L2 hinge error for a given position in a batch using char data.
 *
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit data.
 * @param pIndex Pointer to the index data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight.
 */
template<>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, char* pData, Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ uint32_t s_pIndex[cData._warpSize];
  __shared__ char s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    char t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pIndex[threadIdx.x] = pIndex[dpos];
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      Float diff = s_pUnit[i] - s_pData[i];
      diff = (s_pData[i] > (char)0) ? min((Float)0.0f, diff) : max((Float)0.0, diff);
      error += s_w[i] * diff * diff;
    }
  }

  return error;
}

/**
 * @brief Calculates the indexed L2 hinge error for a given position in a batch.
 * 
 * @tparam T The type of the data array.
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride of the arrays.
 * @param pUnit The array containing unit values.
 * @param pIndex The array containing indices.
 * @param pData The array containing data.
 * @param pDataWeight The array containing data weights (optional).
 * @return Float The calculated error.
 */
template<typename T>
Float kCalculateIndexedL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, T* pData, Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ uint32_t s_pIndex[cData._warpSize];
  __shared__ T s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    T t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pIndex[threadIdx.x] = pIndex[dpos];
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      Float diff = s_pUnit[i] - s_pData[i];
      diff = (s_pData[i] > (T)0) ? min((Float)0.0f, diff) : max((Float)0.0, diff);
      error += s_w[i] * diff * diff;
    }
  }

  return error;
}

/**
 * @brief Kernel function for calculating hinge error.
 * 
 * @tparam T The type of the data array.
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride of the arrays.
 * @param pUnit The array containing unit values.
 * @param pData The array containing data.
 * @param pDataWeight The array containing data weights (optional).
 */
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateHingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, T* pData, Float* pDataWeight)
{
  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ T s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    T t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      Float diff = s_pUnit[i] - s_pData[i];
      diff = (s_pData[i] > (T)0) ? min((Float)0.0f, diff) : max((Float)0.0, diff);
      error += s_w[i] * diff * diff;
    }
  }
}

/**
 * @brief Kernel function for calculating hinge error with unsigned char data type.
 * 
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride of the arrays.
 * @param pUnit The array containing unit values.
 * @param pData The array containing data.
 * @param pDataWeight The array containing data weights (optional).
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateHingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, unsigned char* pData, Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ unsigned char s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    unsigned char t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      Float diff = s_pUnit[i] - s_pData[i];
      diff = (s_pData[i] > (unsigned char)0) ? min((Float)0.0f, diff) : max((Float)0.0, diff);
      error += s_w[i] * max((Float)0.0, (Float)1.0 - diff * diff);
    }
  }
}

/**
 * @brief Kernel function for calculating hinge error with char data type.
 * 
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride of the arrays.
 * @param pUnit The array containing unit values.
 * @param pData The array containing data.
 * @param pDataWeight The array containing data weights (optional).
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateHingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, char* pData, Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ char s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    char t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      Float diff = s_pUnit[i] - s_pData[i];
      diff = (s_pData[i] > (char)0) ? min((Float)0.0f, diff) : max((Float)0.0, diff);
      error += s_w[i] * max((Float)0.0, (Float)1.0 - diff * diff);
    }
  }
}

/**
 * @brief Calculates the hinge error for a given position in a batch.
 * 
 * @tparam T The type of the data array.
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride of the arrays.
 * @param pUnit The array containing unit values.
 * @param pData The array containing data.
 * @param pDataWeight The array containing data weights (optional).
 * @return Float The calculated error.
 */
template<typename T>
Float kCalculateHingeError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, T* pData, Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ T s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    T t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      Float diff = s_pUnit[i] - s_pData[i];
      diff = (s_pData[i] > (T)0) ? min((Float)0.0f, diff) : max((Float)0.0, diff);
      error += s_w[i] * max((Float)0.0, (Float)1.0 - diff * diff);
    }
  }

  return error;
}

/**
 * @brief Kernel function for calculating indexed hinge error.
 * 
 * @tparam T The type of the data array.
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride of the arrays.
 * @param pUnit The array containing unit values.
 * @param pIndex The array containing indices.
 * @param pData The array containing data.
 * @param pDataWeight The array containing data weights (optional).
 */
template<typename T>
__global__ void kCalculateIndexedHingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, T* pData, Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ uint32_t s_pIndex[cData._warpSize];
  __shared__ T s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    T t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pIndex[threadIdx.x] = pIndex[dpos];
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      Float diff = s_pUnit[i] - s_pData[i];
      diff = (s_pData[i] > (T)0) ? min((Float)0.0f, diff) : max((Float)0.0, diff);
      error += s_w[i] * max((Float)0.0, (Float)1.0 - diff * diff);
    }
  }
}

/**
 * @brief Kernel function for calculating indexed hinge error with unsigned char data type.
 * 
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride of the arrays.
 * @param pUnit The array containing unit values.
 * @param pIndex The array containing indices.
 * @param pData The array containing data.
 * @param pDataWeight The array containing data weights (optional).
 */
template<>
__global__ void kCalculateIndexedHingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, unsigned char* pData, Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ uint32_t s_pIndex[cData._warpSize];
  __shared__ unsigned char s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    unsigned char t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pIndex[threadIdx.x] = pIndex[dpos];
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      Float diff = s_pUnit[i] - s_pData[i];
      diff = (s_pData[i] > (unsigned char)0) ? min((Float)0.0f, diff) : max((Float)0.0, diff);
      error += s_w[i] * max((Float)0.0, (Float)1.0 - diff * diff);
    }
  }
}

/**
 * @brief Kernel function for calculating indexed hinge error with char data type.
 * 
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride of the arrays.
 * @param pUnit The array containing unit values.
 * @param pIndex The array containing indices.
 * @param pData The array containing data.
 * @param pDataWeight The array containing data weights (optional).
 */
template<>
__global__ void kCalculateIndexedHingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, char* pData, Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ uint32_t s_pIndex[cData._warpSize];
  __shared__ char s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    char t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pIndex[threadIdx.x] = pIndex[dpos];
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      Float diff = s_pUnit[i] - s_pData[i];
      diff = (s_pData[i] > (char)0) ? min((Float)0.0f, diff) : max((Float)0.0, diff);
      error += s_w[i] * max((Float)0.0, (Float)1.0 - diff * diff);
    }
  }
}

/**
 * @brief Calculates the indexed hinge error for a given position in a batch.
 * 
 * @tparam T The type of the data array.
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride of the arrays.
 * @param pUnit The array containing unit values.
 * @param pIndex The array containing indices.
 * @param pData The array containing data.
 * @param pDataWeight The array containing data weights (optional).
 * @return Float The calculated error.
 */
template<typename T>
Float kCalculateIndexedHingeError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, T* pData, Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ uint32_t s_pIndex[cData._warpSize];
  __shared__ T s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    T t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pIndex[threadIdx.x] = pIndex[dpos];
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      Float diff = s_pUnit[i] - s_pData[i];
      diff = (s_pData[i] > (T)0) ? min((Float)0.0f, diff) : max((Float)0.0, diff);
      error += s_w[i] * max((Float)0.0, (Float)1.0 - diff * diff);
    }
  }

  return error;
}

/**
 * @brief Kernel function for calculating cross-entropy error.
 * 
 * @tparam T The type of the data array.
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride of the arrays.
 * @param pUnit The array containing unit values.
 * @param pData The array containing data.
 * @param pDataWeight The array containing data weights (optional).
 */
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, T* pData, Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ T s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    T t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      Float diff = s_pUnit[i] - s_pData[i];
      Float logA = log(max((Float)MIN_ERROR, a));
      Float logB = log(max((Float)MIN_ERROR, (Float)1.0 - a));
      error += s_w[i] * (-t * logA - ((Float)1.0 - t) * logB);
    }
  }
}

/**
 * @brief Kernel function for calculating cross-entropy error with char data type.
 * 
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride of the arrays.
 * @param pUnit The array containing unit values.
 * @param pData The array containing data.
 * @param pDataWeight The array containing data weights (optional).
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, char* pData, Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ char s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    char t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      Float diff = s_pUnit[i] - s_pData[i];
      Float logA = log(max((Float)MIN_ERROR, a));
      Float logB = log(max((Float)MIN_ERROR, (Float)1.0 - a));
      error += s_w[i] * (-t * logA - ((Float)1.0 - t) * logB);
    }
  }
}

/**
 * @brief Kernel function for calculating cross-entropy error with unsigned char data type.
 * 
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride of the arrays.
 * @param pUnit The array containing unit values.
 * @param pData The array containing data.
 * @param pDataWeight The array containing data weights (optional).
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, unsigned char* pData, Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ unsigned char s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    unsigned char t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      Float diff = s_pUnit[i] - s_pData[i];
      Float logA = log(max((Float)MIN_ERROR, a));
      Float logB = log(max((Float)MIN_ERROR, (Float)1.0 - a));
      error += s_w[i] * (-t * logA - ((Float)1.0 - t) * logB);
    }
  }
}

/**
 * @brief Calculate the cross-entropy error.
 * 
 * @tparam T The type of the data array.
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride of the arrays.
 * @param pUnit The array containing unit values.
 * @param pData The array containing data.
 * @param pDataWeight The array containing data weights (optional).
 * @return The calculated error value.
 */
template<typename T>
Float kCalculateCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, T* pData, Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ T s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    T t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      Float diff = s_pUnit[i] - s_pData[i];
      Float logA = log(max((Float)MIN_ERROR, a));
      Float logB = log(max((Float)MIN_ERROR, (Float)1.0 - a));
      error += s_w[i] * (-t * logA - ((Float)1.0 - t) * logB);
    }
  }

  return error;
}

/**
 * @brief Kernel function for calculating indexed cross-entropy error.
 * 
 * @tparam T The type of the data array.
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride of the arrays.
 * @param pUnit The array containing unit values.
 * @param pIndex The array containing indices.
 * @param pData The array containing data.
 * @param pDataWeight The array containing data weights (optional).
 */
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, T* pData, Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ uint32_t s_pIndex[cData._warpSize];
  __shared__ T s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    T t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pIndex[threadIdx.x] = pIndex[dpos];
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      Float diff = s_pUnit[i] - s_pData[i];
      Float logA = log(max((Float)MIN_ERROR, a));
      Float logB = log(max((Float)MIN_ERROR, (Float)1.0 - a));
      error += s_w[i] * (-t * logA - ((Float)1.0 - t) * logB);
    }
  }
}

/**
 * @brief Kernel function for calculating indexed cross-entropy error with char data type.
 * 
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride of the arrays.
 * @param pUnit The array containing unit values.
 * @param pIndex The array containing indices.
 * @param pData The array containing data.
 * @param pDataWeight The array containing data weights (optional).
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, char* pData, Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ uint32_t s_pIndex[cData._warpSize];
  __shared__ char s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    char t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pIndex[threadIdx.x] = pIndex[dpos];
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      Float diff = s_pUnit[i] - s_pData[i];
      Float logA = log(max((Float)MIN_ERROR, a));
      Float logB = log(max((Float)MIN_ERROR, (Float)1.0 - a));
      error += s_w[i] * (-t * logA - ((Float)1.0 - t) * logB);
    }
  }
}

/**
 * @brief Kernel function for calculating indexed cross-entropy error with unsigned char data type.
 * 
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride of the arrays.
 * @param pUnit The array containing unit values.
 * @param pIndex The array containing indices.
 * @param pData The array containing data.
 * @param pDataWeight The array containing data weights (optional).
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, unsigned char* pData, Float* pDataWeight)
{

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ uint32_t s_pIndex[cData._warpSize];
  __shared__ unsigned char s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    unsigned char t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pIndex[threadIdx.x] = pIndex[dpos];
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      Float diff = s_pUnit[i] - s_pData[i];
      Float logA = log(max((Float)MIN_ERROR, a));
      Float logB = log(max((Float)MIN_ERROR, (Float)1.0 - a));
      error += s_w[i] * (-t * logA - ((Float)1.0 - t) * logB);
    }
  }
}

/**
 * @brief Calculates the indexed cross-entropy error.
 * 
 * @tparam T The data type of the indexed data.
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride of the arrays.
 * @param pUnit The array containing unit values.
 * @param pIndex The array containing indices.
 * @param pData The array containing indexed data.
 * @param pDataWeight The array containing data weights (optional).
 * @return The calculated error value.
 */
template<typename T>
Float kCalculateIndexedCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, T* pData, Float* pDataWeight) {

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ uint32_t s_pIndex[cData._warpSize];
  __shared__ T s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    T t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pIndex[threadIdx.x] = pIndex[dpos];
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      Float diff = s_pUnit[i] - s_pData[i];
      Float logA = log(max((Float)MIN_ERROR, a));
      Float logB = log(max((Float)MIN_ERROR, (Float)1.0 - a));
      error += s_w[i] * (-t * logA - ((Float)1.0 - t) * logB);
    }
  }

  return error;
}

/**
 * @brief Calculates the multinomial cross-entropy error.
 * 
 * @tparam T The data type of the input data.
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride of the arrays.
 * @param pUnit The array containing unit values.
 * @param pData The array containing input data.
 * @param pDataWeight The array containing data weights (optional).
 * @return The calculated error value.
 */
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, T* pData, Float* pDataWeight) {

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ T s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    T t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      Float diff = s_pUnit[i] - s_pData[i];
      Float logA = log(max((Float)MIN_ERROR, a));
      error += s_w[i] * (-t * logA);
    }
  }

  return error;
}

/**
 * @brief Calculates the multinomial cross-entropy error.
 * 
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride of the arrays.
 * @param pUnit The array containing unit values.
 * @param pData The array containing input data.
 * @param pDataWeight The array containing data weights (optional).
 * @return The calculated error value.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, char* pData, Float* pDataWeight) {

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ char s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    char t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      Float diff = s_pUnit[i] - s_pData[i];
      Float logA = log(max((Float)MIN_ERROR, a));
      error += s_w[i] * (-t * logA);
    }
  }

  return error;
}

/**
 * @brief Calculates the multinomial cross-entropy error.
 * 
 * @param position The position within the batch.
 * @param batch The batch size.
 * @param stride The stride of the arrays.
 * @param pUnit The array containing unit values.
 * @param pData The array containing input data.
 * @param pDataWeight The array containing data weights (optional).
 * @return The calculated error value.
 */
template<>
__global__ void LAUNCH_BOUNDS() kCalculateMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, unsigned char* pData, Float* pDataWeight) {

  // Use shared memory to reduce global memory accesses.
  __shared__ Float s_pUnit[cData._warpSize];
  __shared__ unsigned char s_pData[cData._warpSize];
  __shared__ Float s_w[cData._warpSize];

  uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
  Float error = (Float)0.0;

  if (pos < stride) {
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    Float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
    Float a = pUnit[uOffset + pos];
    unsigned char t = pData[dOffset + pos];
    s_pUnit[threadIdx.x] = a;
    s_pData[threadIdx.x] = t;
    s_w[threadIdx.x] = w;
    __syncthreads();

    // Calculate the error.
    for (uint32_t i = 0; i < cData._warpSize; i++) {
      Float diff = s_pUnit[i] - s_pData[i];
      Float logA = log(max((Float)MIN_ERROR, a));
      error += s_w[i] * (-t * logA);
    }
  }

  return error;
} //TODO: Improve kernels

template<typename T> Float kCalculateMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, T* pData, Float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kCalculateMultinomialCrossEntropyError_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pData, pDataWeight);
    LAUNCHERROR("kCalculateMultinomialCrossEntropyError_kernel");
    getGpu()._pbAccumulator->Download();
    return (Float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}


template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t stride, Float* pUnit, uint32_t* pIndex, T* pData, Float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    Float error               = (Float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        Float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        Float a               = pUnit[uOffset + pos];
        T t                     = pData[dOffset + pos];
        error                   = w * (-t * log(max(MIN_ERROR, a)));  
    }

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t stride, Float* pUnit, uint32_t* pIndex, char* pData, Float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    Float error               = (Float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        Float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        Float a               = pUnit[uOffset + pos];
        Float t               = (Float)pData[dOffset + pos] * (Float)(1.0 / 128.0);
        error                   = w * (-t * log(max(MIN_ERROR, a)));     
    }

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t stride, Float* pUnit, uint32_t* pIndex, unsigned char* pData, Float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    Float error               = (Float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        Float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        Float a               = pUnit[uOffset + pos];
        Float t               = (Float)pData[dOffset + pos] * (Float)(1.0 / 256.0);
        error                   = w * (-t * log(max(MIN_ERROR, a)));     
    }

    REDUCEERROR(error)
}

template<typename T> Float kCalculateIndexedMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, T* pData, Float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kCalculateIndexedMultinomialCrossEntropyError_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pIndex, pData, pDataWeight);
    LAUNCHERROR("kCalculateIndexedMultinomialCrossEntropyError_kernel");
    getGpu()._pbAccumulator->Download();
    return (Float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, Float* pUnit, T* pData, Float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    Float error               = (Float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        Float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        Float a               = pUnit[uOffset + pos];
        T t                     = pData[dOffset + pos];
        if (((t == (T)1.0) && (a < cData._SMCE_oneTarget)) || 
            ((t == (T)0.0) && (a > cData._SMCE_zeroTarget)))
            error               = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)) - ( (Float)1.0 - t) * cData._SMCE_zeroScale * log(max(MIN_ERROR, (Float)1.0 - a)));     
    }

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, Float* pUnit, char* pData, Float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    Float error               = (Float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        Float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        Float a               = pUnit[uOffset + pos];
        Float t               = (Float)pData[dOffset + pos] * (Float)(1.0 / 128.0);
        if (((t == (Float)1.0) && (a < cData._SMCE_oneTarget)) || ((t == (Float)0.0) && (a > cData._SMCE_zeroTarget)))
            error               = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)) - ((Float)1.0 - t) * cData._SMCE_zeroScale * log(max(MIN_ERROR, (Float)1.0 - a)));  
    }

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, Float* pUnit, unsigned char* pData, Float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    Float error               = (Float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        Float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        Float a               = pUnit[uOffset + pos];
        Float t               = (Float)pData[dOffset + pos] * (Float)(1.0 / 256.0);
        if (((t == (Float)1.0) && (a < cData._SMCE_oneTarget)) || ((t == (Float)0.0) && (a > cData._SMCE_zeroTarget)))
            error               = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)) - ((Float)1.0 - t) * cData._SMCE_zeroScale * log(max(MIN_ERROR, (Float)1.0 - a)));  
    }

    REDUCEERROR(error)
}

template<typename T> Float kCalculateScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, T* pData, Float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kCalculateScaledMarginalCrossEntropyError_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pData, pDataWeight);
    LAUNCHERROR("kCalculateScaledMarginalCrossEntropyError_kernel");
    getGpu()._pbAccumulator->Download();
    return (Float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, Float* pUnit, uint32_t* pIndex, T* pData, Float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    Float error               = (Float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        Float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        Float a               = pUnit[uOffset + pos];
        T t                     = pData[dOffset + pos];
        if (((t == (T)1.0) && (a < cData._SMCE_oneTarget)) || 
            ((t == (T)0.0) && (a > cData._SMCE_zeroTarget)))
            error               = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)) - ( (Float)1.0 - t) * cData._SMCE_zeroScale * log(max(MIN_ERROR, (Float)1.0 - a)));     
    }

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, Float* pUnit, uint32_t* pIndex, char* pData, Float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    Float error               = (Float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        Float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        Float a               = pUnit[uOffset + pos];
        Float t               = (Float)pData[dOffset + pos] * (Float)(1.0 / 128.0);
        if (((t == (Float)1.0) && (a < cData._SMCE_oneTarget)) || ((t == (Float)0.0) && (a > cData._SMCE_zeroTarget)))
            error               = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)) - ((Float)1.0 - t) * cData._SMCE_zeroScale * log(max(MIN_ERROR, (Float)1.0 - a)));  
    }

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, Float* pUnit, uint32_t* pIndex, unsigned char* pData, Float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    Float error               = (Float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        Float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        Float a               = pUnit[uOffset + pos];
        Float t               = (Float)pData[dOffset + pos] * (Float)(1.0 / 256.0);
        if (((t == (Float)1.0) && (a < cData._SMCE_oneTarget)) || ((t == (Float)0.0) && (a > cData._SMCE_zeroTarget)))
            error               = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)) - ((Float)1.0 - t) * cData._SMCE_zeroScale * log(max(MIN_ERROR, (Float)1.0 - a)));  
    }

    REDUCEERROR(error)
}

template<typename T> Float kCalculateIndexedScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, T* pData, Float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kCalculateIndexedScaledMarginalCrossEntropyError_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pIndex, pData, pDataWeight);
    LAUNCHERROR("kCalculateIndexedScaledMarginalCrossEntropyError_kernel");
    getGpu()._pbAccumulator->Download();
    return (Float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, Float* pUnit, T* pData, Float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    Float error               = (Float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        Float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        Float a               = pUnit[uOffset + pos];
        T t                     = pData[dOffset + pos];
        if ((t != (T)0.0) && (a < cData._SMCE_oneTarget)) 
            error               = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)));
    }

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, Float* pUnit, char* pData, Float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    Float error               = (Float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        Float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        Float a               = pUnit[uOffset + pos];
        Float t               = (Float)pData[dOffset + pos] * (Float)(1.0 / 128.0);
        if ((t != (Float)0.0) && (a < cData._SMCE_oneTarget)) 
            error               = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)));  
    }

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, Float* pUnit, unsigned char* pData, Float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    Float error               = (Float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        Float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        Float a               = pUnit[uOffset + pos];
        Float t               = (Float)pData[dOffset + pos] * (Float)(1.0 / 256.0);
        if ((t != (Float)0.0) && (a < cData._SMCE_oneTarget))
            error               = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)));  
    }

    REDUCEERROR(error)
}

template<typename T> Float kCalculateMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, T* pData, Float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kCalculateMultinomialScaledMarginalCrossEntropyError_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pData, pDataWeight);
    LAUNCHERROR("kCalculateMultinomialScaledMarginalCrossEntropyError_kernel");
    getGpu()._pbAccumulator->Download();
    return (Float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, Float* pUnit, uint32_t* pIndex, T* pData, Float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    Float error               = (Float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        Float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        Float a               = pUnit[uOffset + pos];
        T t                     = pData[dOffset + pos];
        if ((t != (T)0.0) && (a < cData._SMCE_oneTarget)) 
            error               = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)));
    }

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, Float* pUnit, uint32_t* pIndex, char* pData, Float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    Float error               = (Float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        Float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        Float a               = pUnit[uOffset + pos];
        Float t               = (Float)pData[dOffset + pos] * (Float)(1.0 / 128.0);
        if ((t != (Float)0.0) && (a < cData._SMCE_oneTarget)) 
            error               = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)));  
    }

    REDUCEERROR(error)
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, Float* pUnit, uint32_t* pIndex, unsigned char* pData, Float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    Float error               = (Float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        Float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (Float)1.0;
        Float a               = pUnit[uOffset + pos];
        Float t               = (Float)pData[dOffset + pos] * (Float)(1.0 / 256.0);
        if ((t != (Float)0.0) && (a < cData._SMCE_oneTarget))
            error               = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)));  
    }

    REDUCEERROR(error)
}

template<typename T> Float kCalculateIndexedMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, T* pData, Float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kCalculateIndexedMultinomialScaledMarginalCrossEntropyError_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pIndex, pData, pDataWeight);
    LAUNCHERROR("kCalculateIndexedMultinomialScaledMarginalCrossEntropyError_kernel");
    getGpu()._pbAccumulator->Download();
    return (Float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

#define EXPLICITLY_INSTANTIATE_KERNELS(T)                                                                                                                                                                  \
template Float kCalculateL1Error<T>(uint32_t, uint32_t, uint32_t, Float*, T*, Float*);                                                                                                               \
template Float kCalculateIndexedL1Error<T>(uint32_t, uint32_t, uint32_t, Float*, uint32_t*, T*, Float*);                                                                                             \
template Float kCalculateL2Error<T>(uint32_t, uint32_t, uint32_t, Float*, T*, Float*);                                                                                                               \
template Float kCalculateIndexedL2Error<T>(uint32_t, uint32_t, uint32_t, Float*, uint32_t*, T*, Float*);                                                                                             \
template Float kCalculateL2HingeError<T>(uint32_t, uint32_t, uint32_t, Float*, T*, Float*);                                                                                                          \
template Float kCalculateIndexedL2HingeError<T>(uint32_t, uint32_t, uint32_t, Float*, uint32_t*, T*, Float*);                                                                                        \
template Float kCalculateCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, Float*, T*, Float*);                                                                                                     \
template Float kCalculateIndexedCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, Float*, uint32_t*, T*, Float*);                                                                                   \
template Float kCalculateScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, Float*, T*, Float*);                                                                                       \
template Float kCalculateIndexedScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, Float*, uint32_t*, T*, Float*);                                                                     \
template Float kCalculateMultinomialCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, Float*, T*, Float*);                                                                                          \
template Float kCalculateIndexedMultinomialCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, Float*, uint32_t*, T*, Float*);                                                                        \
template Float kCalculateMultinomialScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, Float*, T*, Float*);                                                                            \
template Float kCalculateIndexedMultinomialScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, Float*, uint32_t*, T*, Float*);                                                          \
template Float kCalculateHingeError<T>(uint32_t, uint32_t, uint32_t, Float*, T*, Float*);                                                                                                            \
template Float kCalculateIndexedHingeError<T>(uint32_t, uint32_t, uint32_t, Float*, uint32_t*, T*, Float*);                                                                                          \
template Float kCalculateSparseAnalogL1Error<T>(uint32_t, uint32_t, uint32_t, Float*, uint64_t*, uint64_t*, uint32_t*, Float* pDataWeight, T*, bool);                                                \
template Float kCalculateIndexedSparseAnalogL1Error<T>(uint32_t, uint32_t, uint32_t, Float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, Float* pDataWeight, T*, bool);                              \
template Float kCalculateSparseAnalogL2Error<T>(uint32_t, uint32_t, uint32_t, Float*, uint64_t*, uint64_t*, uint32_t*, Float* pDataWeight, T*, bool);                                                \
template Float kCalculateIndexedSparseAnalogL2Error<T>(uint32_t, uint32_t, uint32_t, Float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, Float* pDataWeight, T*, bool);                              \
template Float kCalculateSparseAnalogL2HingeError<T>(uint32_t, uint32_t, uint32_t, Float*, uint64_t*, uint64_t*, uint32_t*, Float* pDataWeight, T*, bool);                                           \
template Float kCalculateIndexedSparseAnalogL2HingeError<T>(uint32_t, uint32_t, uint32_t, Float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, Float* pDataWeight, T*, bool);                         \
template Float kCalculateSparseAnalogMultinomialCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, Float*, uint64_t*, uint64_t*, uint32_t*, Float* pDataWeight, T*);                                 \
template Float kCalculateIndexedSparseAnalogMultinomialCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, Float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, Float* pDataWeight, T*);               \
template Float kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, Float*, uint64_t*, uint64_t*, uint32_t*, Float* pDataWeight, T*);                   \
template Float kCalculateIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, Float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, Float* pDataWeight, T*); \
template Float kCalculateSparseDataScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, Float*, uint64_t*, uint64_t*, uint32_t*, T*, bool);                                                \
template Float kCalculateIndexedSparseDataScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, Float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, T*, bool);                              \

EXPLICITLY_INSTANTIATE_KERNELS(Float)
EXPLICITLY_INSTANTIATE_KERNELS(double)
EXPLICITLY_INSTANTIATE_KERNELS(unsigned char)
EXPLICITLY_INSTANTIATE_KERNELS(char)
EXPLICITLY_INSTANTIATE_KERNELS(uint32_t)
EXPLICITLY_INSTANTIATE_KERNELS(uint64_t)
EXPLICITLY_INSTANTIATE_KERNELS(int32_t)
EXPLICITLY_INSTANTIATE_KERNELS(int64_t)