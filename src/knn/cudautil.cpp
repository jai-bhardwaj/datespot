#include <cuda/std/atomic>
#include <cuda_runtime.h>
#include <iostream>
#include <format>
#include <system_error>
#include <optional>
#include <string_view>
#include <cstddef>

namespace astdl
{
namespace cuda_util
{
    /**
     * @brief Prints GPU memory information.
     *
     * @param header The header to be displayed in the output.
     */
    void printMemInfo(std::string_view header)
    {
        std::size_t bytesInMb = 1024 * 1024;
        int device;
        std::size_t free;
        std::size_t total;
        cudaError_t err = cudaGetDevice(&device);
        if (err != cudaSuccess)
        {
            std::cerr << "** ERROR (" << err << " - " << cudaGetErrorString(err) << ") calling cudaGetDevice(). "
                      << "The host probably does not have any GPUs or the driver is not installed." << std::endl;
            return;
        }

        err = cudaMemGetInfo(&free, &total);
        if (err != cudaSuccess)
        {
            std::cerr << "** ERROR (" << err << " - " << cudaGetErrorString(err) << ") calling cudaMemGetInfo()." << std::endl;
            return;
        }

        long long freeMb = free / bytesInMb;
        long long usedMb = (total - free) / bytesInMb;
        long long totalMb = total / bytesInMb;

        std::cout << std::format("--{:<50} GPU [{}] Mem Used: {:<6lld} MB. Free: {:<6lld} MB. Total: {:<6lld} MB\n", header,
                                device, usedMb, freeMb, totalMb);
    }

    /**
     * @brief Retrieves GPU device memory information in megabytes.
     *
     * @param device The device index.
     * @param total Pointer to store the total memory in megabytes.
     * @param free Pointer to store the free memory in megabytes.
     */
    void getDeviceMemoryInfoInMb(int device, std::size_t* total, std::size_t* free)
    {
        static const std::size_t bytesInMb = 1024 * 1024;
        std::size_t freeInBytes;
        std::size_t totalInBytes;
        cudaError_t err = cudaGetDevice(&device);
        if (err != cudaSuccess)
        {
            std::cerr << "** ERROR (" << err << " - " << cudaGetErrorString(err) << ") calling cudaGetDevice(). "
                      << "The host probably does not have any GPUs or the driver is not installed." << std::endl;
            return;
        }

        err = cudaMemGetInfo(&freeInBytes, &totalInBytes);
        if (err != cudaSuccess)
        {
            std::cerr << "** ERROR (" << err << " - " << cudaGetErrorString(err) << ") calling cudaMemGetInfo()." << std::endl;
            return;
        }

        *total = totalInBytes / bytesInMb;
        *free = freeInBytes / bytesInMb;
    }

    /**
     * @brief Retrieves the number of available GPU devices.
     *
     * @return The number of available GPU devices. Returns std::nullopt if an error occurs.
     */
    std::optional<int> getDeviceCount()
    {
        int deviceCount;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess)
        {
            std::cerr << "** ERROR (" << err << " - " << cudaGetErrorString(err) << ") calling cudaGetDeviceCount()."
                      << " The host probably does not have any GPUs or the driver is not installed." << std::endl;
            return std::nullopt;
        }

        return deviceCount;
    }

    /**
     * @brief Checks if the system has any available GPUs.
     *
     * @return True if the system has available GPUs, False otherwise.
     */
    bool hasGpus()
    {
        std::optional<int> deviceCountOpt = getDeviceCount();
        return deviceCountOpt.has_value() && deviceCountOpt.value() > 0;
    }

} // namespace cuda_util
} // namespace astdl
