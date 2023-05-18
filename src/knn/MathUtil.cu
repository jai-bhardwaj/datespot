#include <stdexcept>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cudautil.h"

namespace astdl {
namespace math {

/**
 * @brief CUDA kernel to convert an array of floats to half-precision floats.
 *
 * @param src Input array of floats.
 * @param length Number of elements in the input array.
 * @param dst Output array of half-precision floats.
 */
__global__ void kFloatToHalf_kernel(const float* src, size_t length, half* dst) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < length) {
    dst[idx] = __float2half(src[idx]);
  }
}

/**
 * @brief Converts an array of floats to half-precision floats on the GPU.
 *
 * @param hSource Host array of floats.
 * @param sourceSizeInBytes Size of the host array in bytes.
 * @param dDest Device array of half-precision floats.
 * @param dBuffer Temporary buffer on the GPU for data transfer.
 * @param bufferSizeInBytes Size of the temporary buffer in bytes.
 */
void kFloatToHalf(const float* hSource, size_t sourceSizeInBytes, half* dDest, float* dBuffer, size_t bufferSizeInBytes) {
  if (sourceSizeInBytes % sizeof(float) != 0) {
    throw std::invalid_argument("sourceSizeInBytes must be divisible by sizeof(float)");
  }

  if (bufferSizeInBytes % sizeof(float) != 0) {
    throw std::invalid_argument("bufferSizeInBytes must be divisible by sizeof(float)");
  }

  size_t bufferLen = bufferSizeInBytes / sizeof(float);
  dim3 threads(kBlockSize);
  dim3 blocks((bufferLen + threads.x - 1) / threads.x);

  size_t srcLeftBytes = sourceSizeInBytes;
  size_t offset = 0;

  while (srcLeftBytes > 0) {
    size_t cpyBytes = srcLeftBytes < bufferSizeInBytes ? srcLeftBytes : bufferSizeInBytes;
    size_t cpyLength = cpyBytes / sizeof(float);

    CHECK_ERR(cudaMemcpyAsync(dBuffer, hSource + offset, cpyBytes, cudaMemcpyHostToDevice));
    LAUNCH_ERR((kFloatToHalf_kernel<<<blocks, threads>>>(dBuffer, cpyLength, dDest + offset)));
    CHECK_ERR(cudaGetLastError());
    CHECK_ERR(cudaDeviceSynchronize());

    offset += cpyLength;
    srcLeftBytes -= cpyBytes;
  }
}

/**
 * @brief Converts an array of floats to half-precision floats on the GPU.
 *        Allocates a temporary buffer on the GPU.
 *
 * @param hSource Host array of floats.
 * @param sourceSizeInBytes Size of the host array in bytes.
 * @param dDest Device array of half-precision floats.
 * @param bufferSizeInBytes Size of the temporary buffer in bytes.
 */
void kFloatToHalf(const float* hSource, size_t sourceSizeInBytes, half* dDest, size_t bufferSizeInBytes) {
  float* dBuffer = nullptr;
  CHECK_ERR(cudaMalloc(&dBuffer, bufferSizeInBytes));

  kFloatToHalf(hSource, sourceSizeInBytes, dDest, dBuffer, bufferSizeInBytes);

  CHECK_ERR(cudaFree(dBuffer));
}

/**
 * @brief CUDA kernel to convert an array of half-precision floats to floats.
 *
 * @param src Input array of half-precision floats.
 * @param length Number of elements in the input array.
 * @param dst Output array of floats.
 */
__global__ void kHalfToFloat_kernel(const half* src, size_t length, float* dst) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    dst[idx] = __half2float(src[idx]);
  }
}

/**
 * @brief Converts an array of half-precision floats to floats on the GPU.
 *
 * @param dSource Device array of half-precision floats.
 * @param sourceSizeInBytes Size of the device array in bytes.
 * @param hDest Host array of floats.
 * @param bufferSizeInBytes Size of the temporary buffer in bytes.
 */
void kHalfToFloat(const half* dSource, size_t sourceSizeInBytes, float* hDest, size_t bufferSizeInBytes) {
  if (sourceSizeInBytes % sizeof(half) != 0) {
    throw std::invalid_argument("sourceSizeInBytes must be divisible by sizeof(half)");
  }

  if (bufferSizeInBytes % sizeof(float) != 0) {
    throw std::invalid_argument("bufferSizeInBytes must be divisible by sizeof(float)");
  }

  size_t bufferLen = bufferSizeInBytes / sizeof(float);
  dim3 threads(kBlockSize);
  dim3 blocks((bufferLen + threads.x - 1) / threads.x);

  float* dBuffer = nullptr;
  CHECK_ERR(cudaMalloc(&dBuffer, bufferLen * sizeof(float)));

  size_t sourceLength = sourceSizeInBytes / sizeof(half);
  size_t srcLeftBytes = sourceLength * sizeof(float);
  size_t offset = 0;

  while (srcLeftBytes > 0) {
    size_t cpyBytes = srcLeftBytes < bufferSizeInBytes ? srcLeftBytes : bufferSizeInBytes;
    size_t cpyLength = cpyBytes / sizeof(float);

    LAUNCH_ERR((kHalfToFloat_kernel<<<blocks, threads>>>(dSource + offset, cpyLength, dBuffer)));
    CHECK_ERR(cudaMemcpyAsync(hDest + offset, dBuffer, cpyBytes, cudaMemcpyDeviceToHost));
    CHECK_ERR(cudaGetLastError());
    CHECK_ERR(cudaDeviceSynchronize());

    offset += cpyLength;
    srcLeftBytes -= cpyBytes;
  }

  CHECK_ERR(cudaFree(dBuffer));
}

}  // namespace math
}  // namespace astdl
