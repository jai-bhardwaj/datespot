#include <stdexcept>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cudautil.h"

namespace astdl {
namespace math {

constexpr int kBlockSize = 128;

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
 * @param hostInput Host array of floats.
 * @param sourceSizeInBytes Size of the host array in bytes.
 * @param deviceOutput Device array of half-precision floats.
 * @param bufferSizeInBytes Size of the temporary buffer in bytes.
 * @param freeBuffer Flag indicating whether to free the temporary buffer.
 *
 * @throws std::invalid_argument if sourceSizeInBytes is not divisible by sizeof(float) or bufferSizeInBytes is not divisible by sizeof(float).
 */
void kFloatToHalf(const float* hostInput, size_t sourceSizeInBytes, half* deviceOutput, size_t bufferSizeInBytes, bool freeBuffer = true) {
  if (sourceSizeInBytes % sizeof(float) != 0) {
    throw std::invalid_argument("sourceSizeInBytes must be divisible by sizeof(float)");
  }

  if (bufferSizeInBytes % sizeof(float) != 0) {
    throw std::invalid_argument("bufferSizeInBytes must be divisible by sizeof(float)");
  }

  size_t bufferLen = bufferSizeInBytes / sizeof(float);
  dim3 threads(kBlockSize);
  dim3 blocks((bufferLen + threads.x - 1) / threads.x);

  float* deviceBuffer = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(&deviceBuffer, bufferSizeInBytes));

  size_t sourceLength = sourceSizeInBytes / sizeof(float);
  size_t srcLeftBytes = sourceSizeInBytes;
  size_t offset = 0;

  while (srcLeftBytes > 0) {
    size_t copyLength = srcLeftBytes < bufferSizeInBytes ? srcLeftBytes / sizeof(float) : bufferLen;

    CHECK_CUDA_ERROR(cudaMemcpy(deviceBuffer, hostInput + offset, copyLength * sizeof(float), cudaMemcpyHostToDevice));
    LAUNCH_KERNEL(kFloatToHalf_kernel, blocks, threads)(deviceBuffer, copyLength, deviceOutput + offset);

    offset += copyLength;
    srcLeftBytes -= copyLength * sizeof(float);
  }

  if (freeBuffer) {
    CHECK_CUDA_ERROR(cudaFree(deviceBuffer));
  }
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
 * @param deviceInput Device array of half-precision floats.
 * @param sourceSizeInBytes Size of the device array in bytes.
 * @param hostOutput Host array of floats.
 * @param bufferSizeInBytes Size of the temporary buffer in bytes.
 *
 * @throws std::invalid_argument if sourceSizeInBytes is not divisible by sizeof(half) or bufferSizeInBytes is not divisible by sizeof(float).
 */
void kHalfToFloat(const half* deviceInput, size_t sourceSizeInBytes, float* hostOutput, size_t bufferSizeInBytes) {
  if (sourceSizeInBytes % sizeof(half) != 0) {
    throw std::invalid_argument("sourceSizeInBytes must be divisible by sizeof(half)");
  }

  if (bufferSizeInBytes % sizeof(float) != 0) {
    throw std::invalid_argument("bufferSizeInBytes must be divisible by sizeof(float)");
  }

  size_t bufferLen = bufferSizeInBytes / sizeof(float);
  dim3 threads(kBlockSize);
  dim3 blocks((bufferLen + threads.x - 1) / threads.x);

  float* deviceBuffer = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(&deviceBuffer, bufferLen * sizeof(float)));

  size_t sourceLength = sourceSizeInBytes / sizeof(half);
  size_t srcLeftBytes = sourceSizeInBytes;
  size_t offset = 0;

  while (srcLeftBytes > 0) {
    size_t copyLength = srcLeftBytes < bufferSizeInBytes ? srcLeftBytes / sizeof(float) : bufferLen;

    LAUNCH_KERNEL(kHalfToFloat_kernel, blocks, threads)(deviceInput + offset, copyLength, deviceBuffer);
    CHECK_CUDA_ERROR(cudaMemcpy(hostOutput + offset, deviceBuffer, copyLength * sizeof(float), cudaMemcpyDeviceToHost));

    offset += copyLength;
    srcLeftBytes -= copyLength * sizeof(float);
  }

  CHECK_CUDA_ERROR(cudaFree(deviceBuffer));
}

}  // namespace math
}
