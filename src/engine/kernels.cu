#include "GpuTypes.h"
#include "Types.h"
#include <limits>

static __constant__ GpuData cData;

void SetKernelsGpuData()
{
    cudaError_t status = cudaMemcpyToSymbol(cData, &(getGpu()._data), sizeof(GpuData));
    RTERROR(status, "cudaMemcpyToSymbol: SetKernelsGpuData copy to cData failed");
}

void GetKernelsGpuData()
{
    cudaError_t status = cudaMemcpyFromSymbol(&(getGpu()._data), cData, sizeof(GpuData));
    RTERROR(status, "cudaMemcpyFromSymbol: GetKernelsGpuData copy From cData failed");
}

uint32_t CalculateBlocks(uint64_t size)
{
    return (size + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;
}

__global__ void kScaleAndBias_kernel(Float* pData, uint64_t size, Float scale, Float bias)
{
    uint64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset < size)
    {
        Float value = pData[offset];
        pData[offset] = scale * value - bias;
    }
}

void kScaleAndBias(Float* pData, uint64_t size, Float scale, Float bias)
{
    const uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    const uint32_t blocks = CalculateBlocks(size);
    kScaleAndBias_kernel<<<blocks, threadsPerBlock>>>(pData, size, scale, bias);
    LAUNCHERROR("kScaleAndBias_kernel");
}

__global__ void kClearUnit_kernel(Float* pUnit, Float* pBias, uint32_t stride, uint64_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        uint32_t bpos = pos % stride;
        pUnit[pos] = pBias[bpos];
    }
}

void kClearUnit(Float* pUnit, Float* pBias, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = CalculateBlocks(size);
    kClearUnit_kernel<<<blocks, threadsPerBlock>>>(pUnit, pBias, stride, size);
    LAUNCHERROR("kClearUnit_kernel");
}

__global__ void kClearDualSourceUnit_kernel(Float* pUnit, Float* pBias1, Float* pBias2, uint32_t stride, uint32_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        uint32_t bpos = pos % stride;
        pUnit[pos] = pBias1[bpos] + pBias2[bpos];
    }
}

void kClearDualSourceUnit(Float* pUnit, Float* pBias1, Float* pBias2, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = CalculateBlocks(size);
    kClearDualSourceUnit_kernel<<<blocks, threadsPerBlock>>>(pUnit, pBias1, pBias2, stride, size);
    LAUNCHERROR("kClearDualSourceUnit_kernel");
}
__global__ void kClearTripleSourceUnit_kernel(Float* pUnit, Float* pBias1, Float* pBias2, Float* pBias3, uint32_t stride, uint32_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        uint32_t bpos = pos % stride;
        pUnit[pos] = pBias1[bpos] + pBias2[bpos] + pBias3[pos];
    }
}

void kClearTripleSourceUnit(Float* pUnit, Float* pBias1, Float* pBias2, Float* pBias3, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = CalculateBlocks(size);
    kClearTripleSourceUnit_kernel<<<blocks, threadsPerBlock>>>(pUnit, pBias1, pBias2, pBias3, stride, size);
    LAUNCHERROR("kClearTripleSourceUnit_kernel");
}
__global__ void kClearQuadSourceUnit_kernel(Float* pUnit, Float* pBias1, Float* pBias2, Float* pBias3, Float* pBias4, uint32_t stride, uint32_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        uint32_t bpos = pos % stride;
        pUnit[pos] = pBias1[bpos] + pBias2[bpos] + pBias3[pos] + pBias4[pos];
    }
}

void kClearQuadSourceUnit(Float* pUnit, Float* pBias1, Float* pBias2, Float* pBias3, Float* pBias4, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = CalculateBlocks(size);
    kClearQuadSourceUnit_kernel<<<blocks, threadsPerBlock>>>(pUnit, pBias1, pBias2, pBias3, pBias4, stride, size);
    LAUNCHERROR("kClearQuadSourceUnit_kernel");
}
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
        Float w = (pDataWeight != nullptr) ? pDataWeight[pos1] : static_cast<Float>(1.0);
        uint64_t offset = static_cast<uint64_t>(pos) * static_cast<uint64_t>(stride);
        for (uint64_t i = start + threadIdx.x; i < end; i += blockDim.x)
        {
            uint64_t pos2 = offset + pSparseIndex[i];
            T data = pSparseData[i];
            pUnit[pos2] = w * static_cast<Float>(data);
        }
    }
}

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
        Float w = (pDataWeight != nullptr) ? pDataWeight[pos1] : static_cast<Float>(1.0);
        uint64_t offset = static_cast<uint64_t>(pos) * static_cast<uint64_t>(stride);
        for (uint64_t i = start + threadIdx.x; i < end; i += blockDim.x)
        {
            uint64_t pos2 = offset + pSparseIndex[i];
            T data = pSparseData[i];
            pUnit[pos2] = w * static_cast<Float>(data);
        }
    }
}

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
__global__ void kLoadSparseDenoisedInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, Float* pRandom)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < batch)
    {
        uint32_t pos1 = cData._bShuffleIndices ? cData._pShuffleIndex[pos + position] : pos + position;
        uint64_t start = pSparseStart[pos1];
        uint64_t end = pSparseEnd[pos1];
        Float w = cData._denoising_q * ((pDataWeight != nullptr) ? pDataWeight[pos1] : static_cast<Float>(1.0));
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
__global__ void kLoadIndexedSparseDenoisedInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, Float* pRandom)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < batch)
    {
        uint32_t pos1 = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[pos + position] : pos + position];
        uint64_t start = pSparseStart[pos1];
        uint64_t end = pSparseEnd[pos1];
        Float w = cData._denoising_q * ((pDataWeight != nullptr) ? pDataWeight[pos1] : static_cast<Float>(1.0));
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
template<typename T>
__global__ void kLoadSparseAnalogDenoisedInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, Float* pRandom)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < batch)
    {
        uint32_t pos1 = cData._bShuffleIndices ? cData._pShuffleIndex[pos + position] : pos + position;
        uint64_t start = pSparseStart[pos1];
        uint64_t end = pSparseEnd[pos1];
        Float w = cData._denoising_q * ((pDataWeight != nullptr) ? pDataWeight[pos1] : static_cast<Float>(1.0));
        uint64_t offset = static_cast<uint64_t>(pos) * static_cast<uint64_t>(stride);
        for (uint64_t i = start + threadIdx.x; i < end; i += blockDim.x)
        {
            Float value = pRandom[i];
            uint64_t pos2 = offset + pSparseIndex[i];
            T data = pSparseData[i];
            if (value >= cData._denoising_p)
                pUnit[pos2] = w * data;
        }
    }
}

template<typename T>
void kLoadSparseAnalogDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, Float* pRandom)
{
    uint32_t last = position + batch;
    uint32_t count = last - position;
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (count + threadsPerBlock - 1) / threadsPerBlock;

    cudaError_t status = cudaMemset(pUnit, 0, static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride) * sizeof(Float));
    RTERROR(status, "kLoadSparseAnalogDenoisedInputUnit failed");
    kLoadSparseAnalogDenoisedInputUnit_kernel<<<blocks, threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pRandom);
    LAUNCHERROR("kLoadSparseAnalogDenoisedInputUnit_kernel");
}
template<typename T>
__global__ void kLoadIndexedSparseAnalogDenoisedInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, Float* pRandom)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < batch)
    {
        uint32_t pos1 = pIndex[pos + position];
        pos1 = cData._bShuffleIndices ? cData._pShuffleIndex[pos1] : pos1;
        uint64_t start = pSparseStart[pos1] + threadIdx.x;
        uint64_t end = pSparseEnd[pos1];
        Float w = cData._denoising_q * ((pDataWeight != nullptr) ? pDataWeight[pos1] : static_cast<Float>(1.0));
        uint64_t offset = static_cast<uint64_t>(pos) * static_cast<uint64_t>(stride);
        for (uint64_t i = start; i < end; i += blockDim.x)
        {
            Float value = pRandom[i];
            uint64_t pos2 = offset + pSparseIndex[i];
            T data = pSparseData[i];
            if (value >= cData._denoising_p)
                pUnit[pos2] = w * data;
        }
    }
}

template<typename T>
void kLoadIndexedSparseAnalogDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, Float* pRandom)
{
    uint32_t last = position + batch;
    uint32_t count = last - position;
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (count + threadsPerBlock - 1) / threadsPerBlock;

    cudaError_t status = cudaMemset(pUnit, 0, static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride) * sizeof(Float));
    RTERROR(status, "kLoadIndexedSparseAnalogDenoisedInputUnit failed");
    kLoadIndexedSparseAnalogDenoisedInputUnit_kernel<<<blocks, threadsPerBlock>>>(position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pRandom);
    LAUNCHERROR("kLoadIndexedSparseAnalogDenoisedInputUnit_kernel");
}
template<typename T>
__global__ void kLoadInputUnit_kernel(uint32_t position, uint32_t stride, Float* pUnit, T* pData)
{
    uint32_t blockIdx_x = blockIdx.x;
    uint64_t pos = (blockIdx_x * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint32_t pos1 = blockIdx_x + position;
        pos1 = cData._bShuffleIndices ? cData._pShuffleIndex[pos1] : pos1;
        uint64_t soffset = static_cast<uint64_t>(pos1) * static_cast<uint64_t>(stride) + pos;
        uint64_t doffset = static_cast<uint64_t>(blockIdx_x) * static_cast<uint64_t>(stride) + pos;
        pUnit[doffset] = static_cast<Float>(pData[soffset]);
    }
}
__global__ void kLoadNormalizedInputUnit_kernel(int position, int stride, Float* pUnit, unsigned char* pData)
{
    int tid = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (tid < stride)
    {
        int pos1 = cData._bShuffleIndices ? cData._pShuffleIndex[blockIdx.x + position] : (blockIdx.x + position);
        int offset = blockIdx.x * stride + tid;
        pUnit[offset] = static_cast<Float>(pData[pos1 * stride + tid]) * cData._inv256 - cData._half;
    }
}
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

template<typename T>
void kLoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, T* pData)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kLoadInputUnit_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pData);
    LAUNCHERROR("kLoadInputUnit_kernel");
}

template<>
void kLoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, unsigned char* pData)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kLoadNormalizedInputUnit_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pData);
    LAUNCHERROR("kLoadNormalizedInputUnit_kernel");
}
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

template<typename T>
void kLoadIndexedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, T* pData)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kLoadIndexedNormalizedInputUnit_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pIndex, pData);
    LAUNCHERROR("kLoadIndexedNormalizedInputUnit_kernel");
}
__global__ void kAddBias_kernel(Float* pUnit, Float* pBias, uint32_t stride, uint32_t size)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
    {
        uint32_t bpos = tid % stride;
        pUnit[tid] += pBias[bpos];
    }
}

void kAddBias(Float* pUnit, Float* pBias, uint32_t stride, uint32_t batch)
{
    uint32_t size = stride * batch;
    uint32_t blocks = (size + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;
    kAddBias_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pUnit, pBias, stride, size);
    LAUNCHERROR("kAddBias_kernel");
}
__global__ void kAddDualBias_kernel(Float* pUnit, Float* pBias1, Float* pBias2, uint32_t stride, uint32_t size)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
    {
        uint32_t bpos = tid % stride;
        pUnit[tid] += pBias1[bpos] + pBias2[bpos];
    }
}

void kAddDualBias(Float* pUnit, Float* pBias1, Float* pBias2, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    uint32_t blocks = (size + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;
    kAddDualBias_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pUnit, pBias1, pBias2, stride, size);
    LAUNCHERROR("kAddDualBias_kernel");
}
__global__ void kAddTripleBias_kernel(Float* pUnit, Float* pBias1, Float* pBias2, Float* pBias3, uint32_t stride, uint32_t size)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
    {
        uint32_t bpos = tid % stride;
        pUnit[tid] += pBias1[bpos] + pBias2[bpos] + pBias3[tid];
    }
}

void kAddTripleBias(Float* pUnit, Float* pBias1, Float* pBias2, Float* pBias3, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    uint32_t blocks = (size + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;
    kAddTripleBias_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pUnit, pBias1, pBias2, pBias3, stride, size);
    LAUNCHERROR("kAddTripleBias_kernel");
}
__global__ void kAddQuadBias_kernel(Float* pUnit, Float* pBias1, Float* pBias2, Float* pBias3, Float* pBias4, uint32_t stride, uint32_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        uint32_t bpos = pos % stride;
        pUnit[pos] += pBias1[bpos] + pBias2[bpos] + pBias3[pos] + pBias4[pos];
    }
}

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

template<typename T>
void kCalculateSparseAnalogZ(uint32_t position, uint32_t batch, uint32_t stride, Float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, Float* pUnit, Float beta)
{
    uint32_t threads = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateSparseAnalogZ_kernel<<<batch, threads>>>(position, stride, pWeight, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pUnit, beta);
    LAUNCHERROR("kCalculateSparseAnalogZ_kernel");
}
template<typename T>
__global__ void LAUNCH_BOUNDS256_kCalculateIndexedSparseAnalogZ_kernel(
    uint32_t position,
    uint32_t stride,
    NNFloat* pWeight,
    uint32_t* pIndex,
    uint64_t* pSparseStart,
    uint64_t* pSparseEnd,
    uint32_t* pSparseIndex,
    NNFloat* pDataWeight,
    T* pSparseData,
    NNFloat* pUnit,
    NNFloat beta)
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
            sValue[i] = (pDataWeight != nullptr) ? pDataWeight[position] * pSparseData[tstart + i * blockDim.x] : pSparseData[tstart + i * blockDim.x];
        }

        __syncthreads();

        for (uint32_t opos = threadIdx.x; opos < stride; opos += blockDim.x)
        {
            NNFloat unit = (beta == 0.0) ? 0.0 : (beta * pUnit[opos]);
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
template <>
__global__ void LAUNCH_BOUNDS256() kCalculateIndexedSparseAnalogZ_kernel(uint32_t position, uint32_t stride, const NNFloat* pWeight, const uint32_t* pIndex, const uint64_t* pSparseStart, const uint64_t* pSparseEnd, const uint32_t* pSparseIndex, const NNFloat* pDataWeight, const unsigned char* pSparseData, NNFloat* pUnit, const NNFloat beta)
{
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ NNFloat sValue[MAXSPARSEANALOG];

    position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    const uint64_t start = pSparseStart[position];
    const uint64_t end = pSparseEnd[position];
    const NNFloat w = (pDataWeight != NULL) ? pDataWeight[position] : static_cast<NNFloat>(1.0);
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
            sValue[pos] = w * (static_cast<NNFloat>(pSparseData[tstart]) * static_cast<NNFloat>(1.0 / 256.0));
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
                NNFloat unit = (beta == static_cast<NNFloat>(0.0)) ? static_cast<NNFloat>(0.0) : (beta * pUnit[opos]);
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

        beta = static_cast<NNFloat>(1.0);
    }
}
template <>
__global__ void LAUNCH_BOUNDS256() kCalculateIndexedSparseAnalogZ_kernel(uint32_t position, uint32_t stride, const NNFloat* pWeight, const uint32_t* pIndex, const uint64_t* pSparseStart, const uint64_t* pSparseEnd, const uint32_t* pSparseIndex, const NNFloat* pDataWeight, const char* pSparseData, NNFloat* pUnit, const NNFloat beta)
{
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ NNFloat sValue[MAXSPARSEANALOG];

    position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    const uint64_t start = pSparseStart[position];
    const uint64_t end = pSparseEnd[position];
    const NNFloat w = (pDataWeight != NULL) ? pDataWeight[position] : static_cast<NNFloat>(1.0);
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
            sValue[pos] = w * (static_cast<NNFloat>(pSparseData[tstart]) * static_cast<NNFloat>(1.0 / 128.0));
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
                NNFloat unit = (beta == static_cast<NNFloat>(0.0)) ? static_cast<NNFloat>(0.0) : (beta * pUnit[opos]);
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

        beta = static_cast<NNFloat>(1.0);
    }
}

template <typename T>
void kCalculateIndexedSparseAnalogZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, T* pSparseData, NNFloat* pUnit, NNFloat beta)
{
    uint32_t threads = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateIndexedSparseAnalogZ_kernel<<<batch, threads>>>(position, stride, pWeight, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pUnit, beta);
    LAUNCHERROR("kCalculateIndexedSparseAnalogZ_kernel");
}
__global__ void __forceinline__
kCalculateSparseDenoisedZ_kernel(uint32_t position, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta)
{
    uint32_t sOffset[MAXSPARSE];
    uint32_t sOpos = blockDim.x;

    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    NNFloat w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (NNFloat)1.0);
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        uint32_t inputs = min(static_cast<uint32_t>(end - start), static_cast<uint32_t>(MAXSPARSEANALOG));
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            NNFloat value = pRandom[tstart];
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
                NNFloat unit = (beta == (NNFloat)0.0) ? (NNFloat)0.0 : (beta * pUnit[opos]);
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
        beta = (NNFloat)1.0;
    }
}

void kCalculateSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta)
{
    uint32_t threads = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateSparseDenoisedZ_kernel<<<batch, threads>>>(position, stride, pWeight, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pRandom, pUnit, beta);
    LAUNCHERROR("kCalculateSparseDenoisedZ_kernel");
}
__global__ void kCalculateIndexedSparseDenoisedZ_kernel(const uint32_t position, const uint32_t stride, const NNFloat* pWeight, const uint32_t* pIndex, const uint64_t* pSparseStart, const uint64_t* pSparseEnd, const uint32_t* pSparseIndex, const NNFloat* pDataWeight, const NNFloat* pRandom, NNFloat* pUnit, const NNFloat beta)
{
    __shared__ uint32_t sOffset[MAXSPARSE];

    uint32_t sOpos;
    sOpos = blockDim.x;
    position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    const uint64_t start = pSparseStart[position];
    const uint64_t end = pSparseEnd[position];
    const NNFloat w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (NNFloat)1.0);
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        sOpos = blockDim.x;
        const uint32_t inputs = ullMin(end - start, (uint64_t)MAXSPARSEANALOG);
        const uint64_t tstart = start + threadIdx.x;

        for (uint32_t i = threadIdx.x; i < inputs; i += blockDim.x)
        {
            const NNFloat value = pRandom[tstart + i * blockDim.x];
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
                const NNFloat unit = (beta == (NNFloat)0.0) ? (NNFloat)0.0 : (beta * pUnit[opos]);
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
        beta = (NNFloat)1.0;
    }
}

void kCalculateIndexedSparseDenoisedZ(const uint32_t position, const uint32_t batch, const uint32_t stride, const NNFloat* pWeight, const uint32_t* pIndex, const uint64_t* pSparseStart, const uint64_t* pSparseEnd, const uint32_t* pSparseIndex, const NNFloat* pDataWeight, const NNFloat* pRandom, NNFloat* pUnit, const NNFloat beta)
{
    const uint32_t threads = min(256, stride >> getGpu()._warpBits << getGpu()._warpBits);
    kCalculateIndexedSparseDenoisedZ_kernel<<<batch, threads>>>(position, stride, pWeight, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pRandom, pUnit, beta);
    LAUNCHERROR("kCalculateIndexedSparseDenoisedZ_kernel");
}
__global__ void LAUNCH_BOUNDS256(uint32_t position, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, NNFloat* pSparseData, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ NNFloat sValue[MAXSPARSEANALOG];

    sOpos = blockDim.x;
    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    NNFloat w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (NNFloat)1.0);
    pUnit += blockIdx.x * stride;

    uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);

    while (start < end)
    {
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            NNFloat value = pRandom[tstart];
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
            NNFloat unit = (beta == (NNFloat)0.0) ? (NNFloat)0.0 : pUnit[opos];
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
        beta = (NNFloat)1.0;
    }
}
__global__ void LAUNCH_BOUNDS256(uint32_t position, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, unsigned char* pSparseData, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta)
{
    __shared__ uint32_t sOpos;
    __shared__ int32_t sOffset[MAXSPARSEANALOG];
    __shared__ NNFloat sValue[MAXSPARSEANALOG];

    sOpos = blockDim.x;
    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    NNFloat w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (NNFloat)1.0);
    pUnit += blockIdx.x * stride;

    uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);

    while (start < end)
    {
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            NNFloat value = pRandom[tstart];
            sOffset[pos] = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart] * stride;
            sValue[pos] = (NNFloat)pSparseData[tstart] * (NNFloat)(1.0 / 256.0) * w;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;

        for (uint32_t o = 0; o < stride; o += cData._warpSize)
        {
            NNFloat unit = (beta == (NNFloat)0.0) ? (NNFloat)0.0 : (beta * pUnit[opos]);
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
        beta = (NNFloat)1.0;
    }
}
template<typename T>
__global__ void kCalculateSparseAnalogDenoisedZ_kernel(uint32_t position, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, T* pSparseData, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ NNFloat sValue[MAXSPARSEANALOG];

    sOpos = blockDim.x;
    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    NNFloat w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (NNFloat)1.0);
    pUnit += blockIdx.x * stride;

    // Load pWeight and pSparseData into shared memory
    __shared__ NNFloat sWeight[MAXSPARSEANALOG][MAX_WEIGHT_STRIDE];
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
            NNFloat value = pRandom[tstart];
            sOffset[pos] = (value < cData._denoising_p) ? cData._maxUint32_t : static_cast<uint32_t>(pSparseIndex[tstart]) * stride;
            sValue[pos] = static_cast<NNFloat>(pSparseData[tstart]) * (NNFloat)(1.0 / 128.0) * w;
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
                NNFloat unit = (beta == (NNFloat)0.0) ? (NNFloat)0.0 : (beta * pUnit[opos]);

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
        beta = (NNFloat)1.0;
    }
}

template<typename T>
void kCalculateSparseAnalogDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, T* pSparseData, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta)
{
    uint32_t threads = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);

    // Copy cData to constant memory
    Constants cDataConst;
    cudaMemcpyToSymbol(cData, &cDataConst, sizeof(Constants));

    kCalculateSparseAnalogDenoisedZ_kernel<<<batch, threads>>>(position, stride, pWeight, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pRandom, pUnit, beta);
    LAUNCHERROR("kCalculateSparseAnalogDenoisedZ_kernel");
}
template <typename T>
__global__ void LAUNCH_BOUNDS256(uint32_t stride, NNFloat* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, T* pSparseData, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta)
{
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ T sValue[MAXSPARSEANALOG];

    uint32_t position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[blockIdx.x] : blockIdx.x];
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    NNFloat w = cData._denoising_q * ((pDataWeight != nullptr) ? pDataWeight[position] : 1.0);
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        uint64_t inputs = min(end - start, static_cast<uint64_t>(MAXSPARSEANALOG));
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;

        for (uint32_t i = threadIdx.x; i < inputs; i += blockDim.x)
        {
            NNFloat value = pRandom[tstart + i * blockDim.x];
            sOffset[i] = (value < cData._denoising_p) ? cData._maxUint32_t : static_cast<int32_t>(pSparseIndex[tstart + i * blockDim.x]) * stride;
            sValue[i] = pSparseData[tstart + i * blockDim.x] * w;
        }

        __syncthreads();

        uint32_t opos = threadIdx.x;

        while (opos < stride)
        {
            NNFloat unit = (beta == 0.0) ? 0.0 : (beta * pUnit[opos]);
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
    const T w = cData._denoising_q * ((pDataWeight != nullptr) ? pDataWeight[position] : T(1.0));
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

template <typename T>
__device__ T warpReduceSum(T val)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}
template<typename T>
__global__ void kCalculateIndexedSparseAnalogDenoisedZ_kernel(uint32_t position, uint32_t stride, NNFloat* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, T* pSparseData, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta)
{
    const uint32_t MAXSPARSEANALOG = 32;  // Adjust the value as needed

    uint32_t sOffset[MAXSPARSEANALOG];
    NNFloat sValue[MAXSPARSEANALOG];

    position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    NNFloat w = cData._denoising_q * ((pDataWeight != nullptr) ? pDataWeight[position] : (NNFloat)1.0);
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        uint32_t inputs = ullmin(end - start, static_cast<uint64_t>(MAXSPARSEANALOG));
        uint64_t tend = start + inputs;

        // Load sOffset and sValue
        for (uint32_t i = threadIdx.x; i < inputs; i += blockDim.x)
        {
            NNFloat value = pRandom[start + i];
            sOffset[i] = (value < cData._denoising_p) ? cData._maxUint32_t : static_cast<uint32_t>(pSparseIndex[start + i]) * stride;
            sValue[i] = static_cast<NNFloat>(pSparseData[start + i]) * static_cast<NNFloat>(1.0 / 128.0) * w;
        }

        __syncthreads();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;

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

        beta = static_cast<NNFloat>(1.0);
    }
}

template<typename T>
void kCalculateIndexedSparseAnalogDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, T* pSparseData, NNFloat* pRandom, NNFloat* pUnit, NNFloat beta)
{
    uint32_t threads = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateIndexedSparseAnalogDenoisedZ_kernel<<<batch, threads>>>(position, stride, pWeight, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pRandom, pUnit, beta);
    LAUNCHERROR("kCalculateIndexedSparseAnalogDenoisedZ_kernel");
}
__global__ void kCalculateSparseTransposedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;    
        uint64_t startPos = pSparseStart[position] + tgx;
        uint64_t endPos = pSparseEnd[position];

        for (uint64_t start = startPos; start < endPos; start += cData._warpSize)
        {
            uint32_t index = pSparseIndex[start];
            uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos] = bpos;
        }
    }
}
__global__ void kCalculateWeightedSparseTransposedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData)
{
    uint32_t bpos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (bpos < batch)
    {
        position = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start = pSparseStart[position] + threadIdx.x;
        uint64_t end = pSparseEnd[position];
        NNFloat w = pDataWeight[position];
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

void kCalculateSparseTransposedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData)
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

__global__ void kCalculateIndexedWeightedSparseTransposedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData)
{
    uint32_t bpos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (bpos < batch)
    {
        position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        uint64_t start = pSparseStart[position] + threadIdx.x;
        uint64_t end = pSparseEnd[position];
        NNFloat w = pDataWeight[position];
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

void kCalculateIndexedSparseTransposedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData)
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
__global__ void kCalculateSparseTransposedDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex)
{
    uint32_t bpos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (bpos < batch)
    {
        position = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start = pSparseStart[position] + threadIdx.x;
        uint64_t end = pSparseEnd[position];
        while (start < end)
        {
            NNFloat rnd = pRandom[start];
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
__global__ void kCalculateWeightedSparseTransposedDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, NNFloat* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData)
{
    uint32_t bpos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (bpos < batch)
    {
        position = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start = pSparseStart[position] + threadIdx.x;
        uint64_t end = pSparseEnd[position];
        NNFloat w = cData._denoising_q * pDataWeight[position];
        while (start < end)
        {
            NNFloat rnd = pRandom[start];
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

void kCalculateSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, NNFloat* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData)
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
__global__ void kCalculateIndexedSparseTransposedDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex)
{
    uint32_t bpos = blockIdx.x * blockDim.x + threadIdx.x;

    if (bpos < batch)
    {
        position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        uint64_t start = pSparseStart[position] + threadIdx.x;
        uint64_t end = pSparseEnd[position];
        while (start < end)
        {
            NNFloat rnd = pRandom[start];
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
__global__ void kCalculateIndexedWeightedSparseTransposedDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, NNFloat* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData)
{
    uint32_t bpos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (bpos < batch)
    {
        position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        uint64_t start = pSparseStart[position] + threadIdx.x;
        uint64_t end = pSparseEnd[position];
        NNFloat w = cData._denoising_q * pDataWeight[position];
        while (start < end)
        {
            NNFloat rnd = pRandom[start];
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

void kCalculateIndexedSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, NNFloat* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData)
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
template<typename T>
__global__ void kCalculateSparseTransposedAnalogMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData)
{
    uint32_t bpos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (bpos < batch)
    {
        position = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start = pSparseStart[position] + threadIdx.x;
        uint64_t end = pSparseEnd[position];
        NNFloat w = (pDataWeight != NULL) ? pDataWeight[position] : (NNFloat)1.0;
        while (start < end)
        {
            uint32_t index = pSparseIndex[start];
            T value = pSparseData[start];
            uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos] = bpos;
            pSparseTransposedData[opos] = w * static_cast<NNFloat>(value);
            start += blockDim.x;
        }
    }
}

template<typename T>
void kCalculateSparseTransposedAnalogMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, NNFloat* pDataWeight, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, NNFloat* pSparseTransposedData)
{
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blockSize = threadsPerBlock;
    uint32_t numBlocks = (batch + blockSize - 1) / blockSize;

    uint32_t sharedMemSize = 0;  // Assuming no shared memory usage in the kernel

    kCalculateSparseTransposedAnalogMatrix_kernel<<<numBlocks, blockSize, sharedMemSize>>>(position, batch, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);
    LAUNCHERROR("kCalculateSparseTransposedAnalogMatrix_kernel");
}
