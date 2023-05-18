#ifndef LIBKNN_MATHUTIL_H_
#define LIBKNN_MATHUTIL_H_

#include <cuda_fp16.h>
#include <span>
#include <numbers>

namespace astdl
{
namespace math
{
    /**
     * @brief Convert an array of float values to half-precision floating-point values.
     * 
     * @param hSource The input array of float values.
     * @param dDest The output array of half-precision floating-point values.
     * @param bufferSizeInBytes The size of the buffer in bytes. Default is 4 * 1024 * 1024.
     */
    void kFloatToHalf(const std::span<const float> hSource, std::span<half> dDest, size_t bufferSizeInBytes = 4 * 1024 * 1024);

    /**
     * @brief Convert an array of float values to half-precision floating-point values using a buffer.
     * 
     * @param hSource The input array of float values.
     * @param dDest The output array of half-precision floating-point values.
     * @param dBuffer The intermediate buffer for conversion.
     * @param bufferSizeInBytes The size of the buffer in bytes.
     */
    void kFloatToHalf(const std::span<const float> hSource, std::span<half> dDest, std::span<float> dBuffer, size_t bufferSizeInBytes);

    /**
     * @brief Convert an array of half-precision floating-point values to float values.
     * 
     * @param dSource The input array of half-precision floating-point values.
     * @param hDest The output array of float values.
     * @param bufferSizeInBytes The size of the buffer in bytes. Default is 4 * 1024 * 1024.
     */
    void kHalfToFloat(const std::span<const half> dSource, std::span<float> hDest, size_t bufferSizeInBytes = 4 * 1024 * 1024);
}
}

#endif
