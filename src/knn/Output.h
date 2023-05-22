#ifndef LIBKNN_OUTPUT_H_
#define LIBKNN_OUTPUT_H_

#include <cstdint>

constexpr float max_value = 999999999999999.0f;

constexpr int launch_bounds_threads_per_block = 128;
constexpr int launch_bounds256 = 256;
constexpr int launch_bounds512 = 512;
constexpr int launch_bounds1024 = 1024;

#ifdef SYNCHRONOUS
/**
 * @brief Macro for checking CUDA kernel launch errors and device synchronization in synchronous mode.
 *
 * @param s The name of the kernel.
 */
#define LAUNCHERROR(s) \
    { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            exit(-1); \
        } \
        cudaDeviceSynchronize(); \
    }
#else
/**
 * @brief Macro for checking CUDA kernel launch errors in asynchronous mode.
 *
 * @param s The name of the kernel.
 */
#define LAUNCHERROR(s) \
    { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            exit(-1); \
        } \
    }
#endif

/**
 * @brief Calculates the output using the k-nearest neighbors algorithm.
 *
 * @param output      Pointer to the output array.
 * @param key         Pointer to the key array.
 * @param value       Pointer to the value array.
 * @param batch       The batch size.
 * @param width       The width of the input.
 * @param widthPadding The padding width.
 * @param k           The number of nearest neighbors.
 */
void calculateOutput(float* output, float* key, uint32_t* value, uint32_t batch, uint32_t width, uint32_t widthPadding, uint32_t k);

#endif
