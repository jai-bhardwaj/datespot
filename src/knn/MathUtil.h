#ifndef LIBKNN_MATHUTIL_H_
#define LIBKNN_MATHUTIL_H_

#include <cuda_fp16.h>
#include <vector>
#include <algorithm>
#include <span>

namespace astdl
{
    namespace math
    {
        /**
         * @brief Convert an array of floats to half-precision floating-point values.
         *
         * @param hSource Pointer to the source array of floats.
         * @param sourceLength Number of elements in the source array.
         * @param dDest Pointer to the destination array of half-precision floats.
         * @param bufferSizeInBytes Optional buffer size in bytes. Defaults to 4 * 1024 * 1024.
         */
        void kFloatToHalf(const float* hSource, size_t sourceLength, half* dDest, size_t bufferSizeInBytes = 4 * 1024 * 1024);

        /**
         * @brief Convert an array of floats to half-precision floating-point values, using a buffer.
         *
         * @param hSource Pointer to the source array of floats.
         * @param sourceSizeInBytes Size of the source array in bytes.
         * @param dDest Pointer to the destination array of half-precision floats.
         * @param dBuffer Pointer to the buffer used during conversion.
         * @param bufferSizeInBytes Size of the buffer in bytes.
         */
        void kFloatToHalf(const float* hSource, size_t sourceSizeInBytes, half* dDest, float* dBuffer, size_t bufferSizeInBytes);

        /**
         * @brief Convert an array of half-precision floats to floats.
         *
         * @param dSource Pointer to the source array of half-precision floats.
         * @param sourceLength Number of elements in the source array.
         * @param hDest Pointer to the destination array of floats.
         * @param bufferSizeInBytes Optional buffer size in bytes. Defaults to 4 * 1024 * 1024.
         */
        void kHalfToFloat(const half* dSource, size_t sourceLength, float* hDest, size_t bufferSizeInBytes = 4 * 1024 * 1024);
    }
}
#endif
