#ifndef LIBKNN_CUDAUTIL_H_
#define LIBKNN_CUDAUTIL_H_

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/**
 * @brief Check for CUDA error and throw exception if an error occurred.
 *
 * @param e The CUDA error code.
 * @param fname The name of the source file where the error occurred.
 * @param line The line number where the error occurred.
 */
[[nodiscard]] static void CHECK_ERR2(cudaError_t e, const char* fname, int line)
{
    if (e != cudaSuccess)
    {
        std::fprintf(stderr, "FATAL ERROR: cuda failure(%d): %s in %s#%d\n", e, cudaGetErrorString(e), fname, line);
        throw std::runtime_error("CUDA error occurred");
    }
}

/**
 * @brief Check for CUBLAS error and throw exception if an error occurred.
 *
 * @param e The CUBLAS status code.
 * @param fname The name of the source file where the error occurred.
 * @param line The line number where the error occurred.
 */
[[nodiscard]] static void STATUS_ERR2(cublasStatus_t e, const char* fname, int line)
{
    if (e != CUBLAS_STATUS_SUCCESS)
    {
        std::fprintf(stderr, "FATAL ERROR: cublas failure %d in %s#%d\n", e, fname, line);
        throw std::runtime_error("CUBLAS error occurred");
    }
}

/**
 * @brief Check for CUDA kernel launch error and throw exception if an error occurred.
 *
 * @param kernelName The name of the kernel function.
 * @param fname The name of the source file where the error occurred.
 * @param line The line number where the error occurred.
 */
[[nodiscard]] static void LAUNCH_ERR2(const char* kernelName, const char* fname, int line)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        std::fprintf(stderr, "FATAL ERROR: %s launching kernel: %s\n in %s#%d\n", cudaGetErrorString(e), kernelName, fname, line);
        throw std::runtime_error("CUDA kernel launch error occurred");
    }
}

#define CHECK_ERR(e) { CHECK_ERR2(e, __FILE__, __LINE__); }

#define STATUS_ERR(e) { STATUS_ERR2(e, __FILE__, __LINE__); }

#define LAUNCH_ERR(expression) \
    do { \
        expression; \
        LAUNCH_ERR2(#expression, __FILE__, __LINE__); \
    } while (false)

namespace astdl
{
namespace cuda_util
{
    /**
     * @brief Print information about CUDA device memory.
     *
     * @param header Optional header to be printed before the memory information.
     */
    void printMemInfo(const char* header = "");

    /**
     * @brief Get the total and free memory in megabytes for a specific CUDA device.
     *
     * @param device The CUDA device number.
     * @param total Pointer to store the total memory.
     * @param free Pointer to store the free memory.
     */
    void getDeviceMemoryInfoInMb(int device, size_t* total, size_t* free);

    /**
     * @brief Get the number of available CUDA devices.
     *
     * @return The number of CUDA devices.
     */
    int getDeviceCount();

    /**
     * @brief Check if the system has GPUs available.
     *
     * @return `true` if GPUs are available, `false` otherwise.
     */
    bool hasGpus();
}
}

/**
 * @brief Check if a GPU is required. If no GPU is available, return from the calling function.
 */
[[nodiscard]] consteval bool requireGpu()
{
    return astdl::cuda_util::hasGpus();
}

/**
 * @brief Check if a specific number of GPUs are required. If the required number of GPUs is not available, return from the calling function.
 *
 * @param numGpus The number of GPUs required.
 */
[[nodiscard]] consteval bool requireGpus(int numGpus)
{
    return astdl::cuda_util::getDeviceCount() >= numGpus;
}

#define REQUIRE_GPU if (!requireGpu()) return;

#define REQUIRE_GPUS(numGpus) if (!requireGpus(numGpus)) return;

#endif
