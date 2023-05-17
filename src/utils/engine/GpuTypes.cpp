#include "GpuTypes.h"
#include "Types.h"
#include "kernels.h"
#include <cstdlib>
#include <iostream>
#include <cstdlib>
#include <memory>

/**
 * @brief Acceptable error used for floating-point comparisons.
 */
constexpr float cAcceptableError = 0.00001f;

/**
 * @brief Returns the position of the highest set bit in an integer.
 *
 * @param x The input integer.
 * @return The position of the highest set bit.
 */
inline int fls(int x)
{
    return x ? sizeof(x) * 8 - __builtin_clz(x) : 0;
}

/**
 * @brief Returns a reference to the GPU context.
 *
 * @return A reference to the GPU context.
 */
GpuContext& getGpu()
{
    static GpuContext gpu;
    return gpu;
}

/**
 * @brief Constructs a new GPU context.
 */
GpuContext::GpuContext()
    : _bECCSupport(false),
      _bCanMapHostMemory(false),
      _bCPUValidate(false),
      _bUnifiedMemory(false),
      _acceptableError(cAcceptableError),
      _totalCPUMemory(0),
      _totalGPUMemory(0),
      _numprocs(1),
      _id(0),
      _sm_version(SM_3X),
      _sm_major(0),
      _warpSize(32),
      _maxSparse(SM_3X_MAXSPARSE),
      _maxSparseAnalog(SM_3X_MAXSPARSEANALOG),
      _cuBLASHandle(0),
      _cuDNNHandle(0),
      _pbAccumulator()
{
}

/**
 * @brief Destroys the GPU context.
 */
GpuContext::~GpuContext()
{
}

/**
 * @brief Sets the flag indicating whether CPU validation is enabled.
 *
 * @param bValidate Flag indicating whether CPU validation is enabled.
 */
void GpuContext::SetCPUValidate(bool bValidate)
{
    _bCPUValidate = bValidate;
}
/**
 * @brief Initializes the GPU context.
 *
 * @param argc The number of command-line arguments.
 * @param argv The command-line argument values.
 */
void GpuContext::Startup(int argc, char** argv)
{
    /**
     * @brief Check if MPI is initialized and initialize it if necessary.
     * 
     * @param argc The number of command-line arguments.
     * @param argv The command-line arguments.
     */
    int flag = 0;
    MPI_Initialized(&flag);
    if (!flag) {
        MPI_Init(&argc, &argv);
    }

    /**
     * @brief Get the size of the MPI communicator.
     * 
     * @param MPI_COMM_WORLD The MPI communicator representing all processes.
     * @param _numprocs A reference to store the number of processes.
     */
    MPI_Comm_size(MPI_COMM_WORLD, &_numprocs);

    /**
     * @brief Get the rank of the current process in the MPI communicator.
     * 
     * @param MPI_COMM_WORLD The MPI communicator representing all processes.
     * @param _id A reference to store the rank of the current process.
     */
    MPI_Comm_rank(MPI_COMM_WORLD, &_id);

    /**
     * @brief Print the process initialization information.
     */
    std::cout << "GpuContext::Startup: Process " << _id << " out of " << _numprocs << " initialized.\n";

    /**
     * @brief Check if the CUDA_PROFILE environment variable is set and configure CUDA profiling if necessary.
     */
    if (std::getenv("CUDA_PROFILE") != nullptr)
    {
        std::string profileLog;
        if (std::getenv("CUDA_PROFILE_LOG") != nullptr)
        {
            profileLog = std::string(std::getenv("CUDA_PROFILE_LOG")) + std::to_string(_id);
        }
        else
        {
            profileLog = "cu" + std::to_string(_id) + ".csv";
        }
        std::setenv("CUDA_PROFILE_LOG", profileLog.c_str(), 1);
        std::setenv("CUDA_PROFILE_CSV", "1", 1);
    }

    /**
     * @brief Holds the ID of the selected GPU device. Initialized to -1.
     */
    int device = -1;

    /**
     * @brief Holds the count of CUDA-capable GPU devices.
     */
    int gpuCount = 0;

    /**
     * @brief Holds the CUDA API call status.
     */
    cudaError_t status;

    /**
     * @brief Retrieve the number of available CUDA-capable devices and store the count in `gpuCount`.
     */
    status = cudaGetDeviceCount(&gpuCount);
    RTERROR(status, "cudaGetDeviceCount failed");

    /**
     * @brief Check if there are no CUDA-capable devices available and exit the program if true.
     */
    if (gpuCount == 0)
    {
        std::cout << "GpuContext::Startup: No CUDA-capable devices found, exiting.\n";
        cudaDeviceReset();
        Shutdown();
        std::exit(-1);
    }

    /**
     * @brief Holds the size of the MPI communicator.
     */
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /**
     * @brief Holds the rank of the current process in the MPI communicator.
     */
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    /**
     * @brief Holds the length of the processor name.
     */
    int length;

    /**
     * @brief Holds the name of the current processor.
     */
    char myName[MPI_MAX_PROCESSOR_NAME + 1];

    /**
     * @brief Holds the names of all processors in the MPI_COMM_WORLD communicator.
     */
    std::vector<char> pName(world_size * (MPI_MAX_PROCESSOR_NAME + 1));

    /**
     * @brief Holds the count of characters in each processor name.
     */
    std::vector<int> pNameCount(world_size, MPI_MAX_PROCESSOR_NAME + 1);

    /**
     * @brief Holds the displacements of each processor name in the `pName` array.
     */
    std::vector<int> pNameDisp(world_size);

    /**
     * @brief Get the name of the current processor and store it in `myName`.
     */
    MPI_Get_processor_name(myName, &length);

    /**
     * @brief Copy the current processor name into the appropriate position in `pName` array.
     */
    std::memcpy(pName.data() + world_rank * (MPI_MAX_PROCESSOR_NAME + 1), myName, length + 1);

    /**
     * @brief Calculate the displacements for the `MPI_Allgatherv` call.
     */
    for (int i = 0; i < world_size; i++)
    {
        pNameDisp[i] = i * (MPI_MAX_PROCESSOR_NAME + 1);
    }

    /**
     * @brief Gather all processor names from all processes into the `pName` array.
     */
    MPI_Allgatherv(myName, MPI_MAX_PROCESSOR_NAME + 1, MPI_CHAR, pName.data(), pNameCount.data(), pNameDisp.data(),
        MPI_CHAR, MPI_COMM_WORLD);

    /**
     * @brief Check if all processes are running on a single node by comparing their processor names.
     */
    bool bSingleNode = std::all_of(pName.data(), pName.data() + _numprocs * (MPI_MAX_PROCESSOR_NAME + 1),
        [myName](const char* name) { return std::strcmp(name, myName) == 0; });

    /**
     * @brief Set the device flags to enable CUDA memory mapping on the current device.
     */
    cudaSetDeviceFlags(cudaDeviceMapHost);

    /**
     * @brief Count the number of processes running on the current node.
     */
    int localCount = std::count_if(pName.data(), pName.data() + world_size * (MPI_MAX_PROCESSOR_NAME + 1),
        [myName](const char* name) { return std::strcmp(name, myName) == 0; });

    /**
     * @brief Calculate the offset of the current process in the local node.
     */
    int offset = std::count_if(pName.data(), pName.data() + world_rank * (MPI_MAX_PROCESSOR_NAME + 1),
        [myName](const char* name) { return std::strcmp(name, myName) == 0; }) + 1;

    /**
     * @brief Find the device ID that can map host memory and has a major version of at least 3.
     */
    if (localCount > 1)
    {
        int pos = 0;
        while (offset > 0)
        {
            cudaGetDeviceProperties(&deviceProp, pos);
            if (deviceProp.canMapHostMemory && (deviceProp.major >= 3))
            {
                device = pos;
                offset--;
            }
            pos = (pos + 1) % gpuCount;
        }
        char hostname[128];
        gethostname(hostname, 127);
        std::cout << "GpuContext::Startup: Process " << _id << " running on device " << device
                << " out of " << gpuCount << " GPUs on " << hostname << '\n';
    }
    else
    {
        /**
         * @brief Searches for compatible GPUs that can map host memory and have a compute capability of at least 3.
         *
         * @param gpuCount The total number of available GPUs.
         *
         * @return The number of compatible GPUs found.
         */
        std::vector<int> gpuList;
        std::vector<unsigned int> gpuScore;
        int gpus = 0;
        for (int i = 0; i < gpuCount; i++)
        {
            cudaGetDeviceProperties(&deviceProp, i);
            if (deviceProp.canMapHostMemory && (deviceProp.major >= 3))
            {
                gpuList.push_back(i);
                gpuScore.push_back((deviceProp.major << 24) + (deviceProp.totalGlobalMem >> 20));
                gpus++;
            }
        }

        /**
         * @brief Sorts the list of compatible GPUs based on their scores.
         */
        if (gpus > 0)
        {
            bool done = true;
            do
            {
                done = true;
                for (int i = 0; i < gpus - 1; i++)
                {
                    if (gpuScore[i] < gpuScore[i + 1])
                    {
                        done = false;
                        std::swap(gpuList[i], gpuList[i + 1]);
                        std::swap(gpuScore[i], gpuScore[i + 1]);
                    }
                }
            } while (!done);
        }

        /**
         * @brief Sets the valid devices for CUDA based on the compatible GPUs found.
         *
         * @param gpuList Pointer to the array of compatible GPU IDs.
         * @param gpus The number of compatible GPUs.
         *
         * @return The status of setting the valid devices.
         */
        status = cudaSetValidDevices(gpuList.data(), gpus);
        RTERROR(status, "GpuContext::Startup: Error searching for compatible GPU");

        /**
         * @brief Frees any device memory previously allocated on the default GPU.
         *
         * @return The status of freeing device memory.
         */
        status = cudaFree(0);
        RTERROR(status, "GpuContext::Startup: Error selecting compatible GPU");

        /**
         * @brief Fetches the ID of the current GPU.
         *
         * @param device Reference to the variable to store the current GPU ID.
         *
         * @return The status of fetching the current GPU.
         */
        status = cudaGetDevice(&device);
        RTERROR(status, "GpuContext::Startup: Error fetching current GPU");

        if (device == -1)
        {
            std::cout << "GpuContext::Startup: No Kepler or later GPU located, exiting.\n";
            cudaDeviceReset();
            Shutdown();
            std::exit(-1);
        }

    /**
     * @brief Sets the current CUDA device for the GPU context.
     *
     * @param device The device ID of the CUDA device to set.
     *
     * @return The status of setting the CUDA device.
     */
    status = cudaSetDevice(device);
    RTERROR(status, "GpuContext::Startup: Error setting CUDA device");

    /**
     * @brief Updates the device ID of the GPU context.
     */
    _device = device;

    /**
     * @brief Synchronizes the GPU device.
     */
    cudaDeviceSynchronize();

    /**
     * @brief Creates a unique pointer to a GPU buffer for the accumulator data.
     */
    _pbAccumulator = std::make_unique<GpuBuffer<unsigned long long int>>((unsigned int)1, true);

    /**
     * @brief Sets the pointer to the accumulator data in the GPU context.
     */
    _data._pAccumulator = _pbAccumulator->_pDevData;

    cudaGetDeviceProperties(&deviceProp, _device);

    /**
     * @brief Sets the GPU context parameters based on the major compute capability of the device.
     */
    switch (deviceProp.major)
    {
        case 3:
            _sm_version = SM_3X;
            _threadsPerBlock = SM_3X_THREADS_PER_BLOCK;
            _maxSparse = SM_3X_MAXSPARSE;
            _maxSparseAnalog = SM_3X_MAXSPARSEANALOG;
            break;
        case 5:
            _sm_version = SM_5X;
            _threadsPerBlock = SM_5X_THREADS_PER_BLOCK;
            _maxSparse = SM_5X_MAXSPARSE;
            _maxSparseAnalog = SM_5X_MAXSPARSEANALOG;
            break;
        default:
            _sm_version = SM_6X;
            _threadsPerBlock = SM_6X_THREADS_PER_BLOCK;
            _maxSparse = SM_6X_MAXSPARSE;
            _maxSparseAnalog = SM_6X_MAXSPARSEANALOG;
            break;
    }

    /**
     * @brief Sets the GPU context parameters based on the device properties.
     */
    _sm_major = deviceProp.major;
    _warpSize = deviceProp.warpSize;
    _warpBits = std::bit_width(_warpSize) - 1;
    _warpMask = _warpSize - 1;
    _data._warpSize = _warpSize;
    _data._warpBits = _warpBits;
    _data._warpMask = _warpMask;
    _bUnifiedMemory = (deviceProp.managedMemory != 0);

    /**
     * @brief Sets the maximum values for different integer types in the GPU context.
     */
    _data._maxUint32_t = std::numeric_limits<uint32_t>::max();
    _data._maxInt32_t = std::numeric_limits<int32_t>::max();
    _data._maxUint64_t = std::numeric_limits<uint64_t>::max();
    _data._maxInt64_t = std::numeric_limits<int64_t>::max();

    if (getGpu()._id == 0)
    {
        std::cout << "GpuContext::Startup: Enumerating GPUs in use.\n";
    }

    /**
     * @brief Prints information about the GPUs in use by each process.
     */
    for (size_t i = 0; i < getGpu()._numprocs; i++)
    {
        if (getGpu()._id == i)
        {
            std::cout << "Process: " << i << ", GPU: " << deviceProp.name << ", running SM " << deviceProp.major << '.' << deviceProp.minor << '\n';
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    /**
     * @brief Prints the single-node flag for the GPU.
     *
     * @param bSingleNode Flag indicating whether the GPU context is running in a single-node configuration.
     */
    std::cout << "GpuContext::Startup: Single node flag on GPU for process " << _device << " is " << bSingleNode << '\n';

    if (bSingleNode)
    {
        bP2P = true;
        std::vector<int> pDevice(_numprocs, device);

        /**
         * @brief Gathers the GPU device IDs of all processes in the MPI communicator.
         *
         * @param sendbuf Input buffer (ignored as MPI_IN_PLACE is specified).
         * @param sendcount Number of elements in the send buffer (ignored as MPI_IN_PLACE is specified).
         * @param sendtype MPI datatype of send buffer elements (ignored as MPI_IN_PLACE is specified).
         * @param recvbuf Output buffer to store gathered data (array of device IDs).
         * @param recvcount Number of elements received from each process.
         * @param recvtype MPI datatype of receive buffer elements.
         * @param comm MPI communicator.
         */
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, pDevice.data(), sizeof(int), MPI_BYTE, MPI_COMM_WORLD);

        std::vector<int> pUnifiedAddressing(_numprocs);
        cudaGetDeviceProperties(&deviceProp, device);
        pUnifiedAddressing[_id] = deviceProp.unifiedAddressing;

        /**
         * @brief Gathers the unified addressing flags of all processes in the MPI communicator.
         *
         * @param sendbuf Input buffer (ignored as MPI_IN_PLACE is specified).
         * @param sendcount Number of elements in the send buffer (ignored as MPI_IN_PLACE is specified).
         * @param sendtype MPI datatype of send buffer elements (ignored as MPI_IN_PLACE is specified).
         * @param recvbuf Output buffer to store gathered data (array of unified addressing flags).
         * @param recvcount Number of elements received from each process.
         * @param recvtype MPI datatype of receive buffer elements.
         * @param comm MPI communicator.
         */
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, pUnifiedAddressing.data(), sizeof(int), MPI_BYTE, MPI_COMM_WORLD);

        for (int i = 0; i < _numprocs; i++)
        {
            if (pDevice[i] != device)
            {
                int canAccessPeer;

                /**
                 * @brief Checks if P2P communication is possible between two devices.
                 *
                 * @param canAccessPeer Pointer to an integer indicating whether the devices can access each other.
                 * @param device Device ID of the current GPU.
                 * @param peerDevice Device ID of the peer GPU.
                 *
                 * @return The status of the P2P access check.
                 */
                std::cout << "GpuContext::Startup: Testing P2P for processes " << device << " and " << pDevice[i] << '\n';
                cudaError_t status = cudaDeviceCanAccessPeer(&canAccessPeer, device, pDevice[i]);
                RTERROR(status, "cudaDeviceCanAccessPeer");

                if (canAccessPeer == 0)
                {
                    bP2P = false;
                }
                else
                {
                    status = cudaDeviceEnablePeerAccess(pDevice[i], 0);

                    if (status != cudaErrorPeerAccessAlreadyEnabled)
                    {
                        RTERROR(status, "cudaDeviceEnablePeerAccess");
                    }
                    else
                    {
                        cudaGetLastError();
                    }
                }
            }

            if (!pUnifiedAddressing[i])
            {
                bSingleNode = false;
            }
        }
    }

    /**
     * @brief Sets the flag indicating whether the GPU context is running in a single-node configuration.
     *
     * @param bSingleNode Flag indicating whether the GPU context is running in a single-node configuration.
     */
    _bSingleNode = bSingleNode;

    /**
     * @brief Sets the flag indicating whether the GPU context has peer-to-peer (P2P) support.
     *
     * @param bP2P Flag indicating whether the GPU context has P2P support.
     */
    _bP2P = bP2P;

    /**
     * @brief Prints the P2P support flags for the GPU.
     */
    std::cout << "GpuContext::Startup: P2P support flags on GPU for process " << _device << " are " << _bP2P << ' ' << _bSingleNode << '\n';

    /**
     * @brief Performs an all-reduce operation to check if all GPUs have P2P support, and falls back to MPI if not.
     */
    MPI_Allreduce(MPI_IN_PLACE, &_bP2P, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

    if (!_bP2P)
    {
        if (_id == 0)
        {
            std::cout << "GpuContext::Startup: Not all GPUs can P2P between each other, falling back to MPI.\n";
        }
    }

    /**
     * @brief Performs an all-reduce operation to check if P2P support is only available within a single node, and falls back to MPI if not.
     */
    MPI_Allreduce(MPI_IN_PLACE, &_bSingleNode, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

    if (!_bSingleNode)
    {
        if (_id == 0)
        {
            std::cout << "GpuContext::Startup: P2P support only works within a single node, falling back to MPI.\n";
        }
    }

    cudaGetDeviceProperties(&deviceProp, device);

    /**
     * @brief Sets the flag indicating whether the GPU supports ECC memory or is a Tesla-class device.
     */
    _bECCSupport = deviceProp.ECCEnabled || deviceProp.tccDriver || (strcasestr(deviceProp.name, "tesla") != nullptr);

    /**
     * @brief Sets the flag indicating whether the GPU can map host memory.
     */
    _bCanMapHostMemory = deviceProp.canMapHostMemory;

    /**
     * @brief Sets the total global memory available on the GPU.
     */
    _totalMemory = deviceProp.totalGlobalMem;


#ifdef GVERBOSE
    /**
     * @brief Calculates the memory size of the GPU in megabytes.
     *
     * @param deviceProp The GPU device properties.
     *
     * @return The memory size of the GPU in megabytes.
     */
    double memsize = static_cast<double>(deviceProp.totalGlobalMem) / (1024.0 * 1024.0);

    /**
     * @brief Prints information about the initialized GPU.
     *
     * @param device The device number of the GPU.
     * @param deviceProp The GPU device properties.
     */
    std::cout << "GpuContext::Startup: Using GPU " << device << ", " << deviceProp.name << ", SM " << deviceProp.major << '.' << deviceProp.minor << ", " << memsize << " MBytes of memory\n";
#endif

    /**
     * @brief Initializes the cuBLAS library on the GPU.
     *
     * @return The status of the cuBLAS initialization.
     */
    cublasStatus_t cstatus = cublasCreate(&_cuBLASHandle);

    if (cstatus != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "GpuContext::Startup: Failed to initialize cuBLAS on GPU for process " << _device << ", exiting.\n";
        Shutdown();
        std::exit(-1);
    }

    /**
     * @brief Initializes the cuDNN library on the GPU.
     *
     * @return The status of the cuDNN initialization.
     */
    cudnnStatus_t cdstatus = cudnnCreate(&_cuDNNHandle);

    if (cdstatus != CUDNN_STATUS_SUCCESS)
    {
        std::cout << "GpuContext::Startup: Failed to initialize cuDNN on GPU for process " << _device << ", exiting.\n";
        Shutdown();
        std::exit(-1);
    }

    /**
     * @brief Initializes the cuRand library on the GPU.
     *
     * @return The status of the cuRand initialization.
     */
    curandStatus_t crstatus = curandCreateGenerator(&_RNG, CURAND_RNG_PSEUDO_DEFAULT);

    if (crstatus != CURAND_STATUS_SUCCESS)
    {
        std::cout << "GpuContext::Startup: Failed to initialize cuRand on GPU for process " << _device << ", exiting.\n";
        Shutdown();
        std::exit(-1);
    }

    /**
     * @brief Prints the initialization status message for the GPU.
     */
    std::cout << "GpuContext::Startup: GPU for process " << device << " initialized.\n";


}

/**
 * @brief Copies the GPU constants.
 *
 * This function copies the GPU constants by invoking specific kernel functions to set the GPU data.
 * It calls different kernel functions to copy the GPU data for kernels related to kernels,
 * loss functions, activation functions, and delta calculations.
 */
void GpuContext::CopyConstants()
{
    auto copyGpuData = [this](auto& kernelFunction) {
        kernelFunction.SetGpuData();
    };

    copyGpuData(SetKernelsGpuData);
    copyGpuData(SetKLossGpuData);
    copyGpuData(SetKActivationGpuData);
    copyGpuData(SetKDeltaGpuData);
}

/**
 * @brief Sets the fast math mode for the GPU context.
 *
 * This function sets the fast math mode for the GPU context. Fast math mode can improve
 * computational performance by allowing certain optimizations. The availability of fast math mode
 * depends on the GPU architecture.
 *
 * @param flag Boolean flag indicating whether to enable or disable fast math mode.
 */
void GpuContext::SetFastMath(bool flag)
{
    if (_sm_major >= 7)
    {
        cublasMath_t mathMode = flag ? CUBLAS_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
        cublasStatus_t cstatus = cublasSetMathMode(_cuBLASHandle, mathMode);
        if (cstatus != CUBLAS_STATUS_SUCCESS)
        {
            std::cout << "GpuContext::SetFastMath: failed to set math mode\n";
        }
    }
    else
    {
        std::cout << "GpuContext::SetFastMath: failed to set math mode because GPU SM revision is <7.0\n";
    }
}

/**
 * @brief Shuts down the GPU context.
 *
 * This function performs the necessary cleanup and shutdown operations for the GPU context.
 * It releases resources associated with cuBLAS, cuDNN, cuRand, resets the CUDA device,
 * finalizes MPI (Message Passing Interface), and prints status messages indicating the shutdown process.
 */
void GpuContext::Shutdown()
{
    _pbAccumulator.reset();

    std::cout << "GpuContext::Shutdown: Shutting down cuBLAS on GPU for process " << _device << '\n';
    if (auto cstatus = cublasDestroy(_cuBLASHandle); cstatus != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "GpuContext::Shutdown: Failed to shut down cuBLAS on GPU for process " << _device << ".\n";
    }
    std::cout << "GpuContext::Shutdown: CuBLAS shut down on GPU for process " << _device << '\n';

    std::cout << "GpuContext::Shutdown: Shutting down cuDNN on GPU for process " << _device << '\n';
    if (auto cdstatus = cudnnDestroy(_cuDNNHandle); cdstatus != CUDNN_STATUS_SUCCESS)
    {
        std::cout << "GpuContext::Shutdown: Failed to shut down cuDNN on GPU for process " << _device << ".\n";
    }
    std::cout << "GpuContext::Shutdown: CuDNN shut down on GPU for process " << _device << '\n';

    std::cout << "GpuContext::Shutdown: Shutting down cuRand on GPU for process " << _device << '\n';
    if (auto crstatus = curandDestroyGenerator(_RNG); crstatus != CURAND_STATUS_SUCCESS)
    {
        std::cout << "GpuContext::Shutdown: Failed to shut down cuRand on GPU for process " << _device << ".\n";
    }
    std::cout << "GpuContext::Shutdown: CuRand shut down on GPU for process " << _device << '\n';

    cudaDeviceReset();

    MPI_Finalize();
    std::cout << "GpuContext::Shutdown: Process " << _id << " out of " << _numprocs << " finalized.\n";
}


/**
 * @brief Sets the neural network for the GPU context.
 *
 * This function sets the neural network associated with the GPU context.
 * It initializes various constants and parameters used during neural network training
 * based on the network configuration.
 *
 * @param pNetwork Pointer to the neural network to set.
 */
void GpuContext::SetNeuralNetwork(Network* pNetwork)
{
    _pNetwork = pNetwork;

    const auto& {
        _LRN_k,
        _LRN_n,
        _LRN_alpha,
        _LRN_beta,
        _maxout_k,
        _bSparsenessPenalty,
        _sparsenessPenalty_p,
        _sparsenessPenalty_beta,
        _bDenoising,
        _denoising_p,
        _deltaBoost_one,
        _deltaBoost_zero,
        _SMCE_oneTarget,
        _SMCE_zeroTarget,
        _SMCE_oneScale,
        _SMCE_zeroScale,
        _bShuffleIndices,
        _pShuffleIndex,
        _mode
    } = *pNetwork;

    const auto [_denoising_q] = [denoising_p = _denoising_p] { 
        return 1.0f / (1.0f - denoising_p);
    }();

    _data = {
        _LRN_k,
        _LRN_n,
        _LRN_alpha,
        _LRN_beta,
        _maxout_k,
        _bSparsenessPenalty,
        _sparsenessPenalty_p,
        _sparsenessPenalty_beta,
        _bDenoising,
        _denoising_p,
        _denoising_q,
        _deltaBoost_one,
        _deltaBoost_zero,
        _SMCE_oneTarget,
        _SMCE_zeroTarget,
        _SMCE_oneScale,
        _SMCE_zeroScale,
        _bShuffleIndices && (_mode == Mode::Training),
        _pShuffleIndex
    };

    CopyConstants();
}

/**
 * @brief Sets the random seed for the GPU context.
 *
 * This function sets the random seed for the GPU context using the provided seed value.
 * It uses the cuRand library to set the pseudo-random generator seed on the GPU.
 * If the seed setting fails, the function prints an error message, shuts down the GPU context,
 * and exits the program.
 *
 * @param seed The random seed value to set.
 */
void GpuContext::SetRandomSeed(unsigned long seed)
{
    if (auto crstatus = curandSetPseudoRandomGeneratorSeed(_RNG, seed + static_cast<unsigned long>(_device) * 76801ull); crstatus != CURAND_STATUS_SUCCESS)
    {
        if (getGpu()._id == 0)
            std::cout << "GpuContext::SetRandomSeed: Failed to set cuRand seed on GPU for process " << _device << ", exiting.\n";
        Shutdown();
        std::exit(-1);
    }
    
    std::srand(seed);

    if (getGpu()._id == 0)
        std::cout << "GpuContext::SetRandomSeed: Random seed set to " << seed << ".\n";
}

/**
 * @brief Retrieves the memory usage of the GPU context.
 *
 * This function retrieves the current memory usage of the GPU context and stores the values in the provided variables.
 * The memory values are measured in kilobytes (KB).
 *
 * @param gpuMemory Pointer to the variable that will store the GPU memory usage.
 * @param cpuMemory Pointer to the variable that will store the CPU memory usage.
 */
void GpuContext::GetMemoryUsage(int* gpuMemory, int* cpuMemory)
{
    *gpuMemory = static_cast<int>(_totalGPUMemory / 1024ll);
    *cpuMemory = static_cast<int>(_totalCPUMemory / 1024ll);
}

/**
 * @brief Verifies the result of matrix multiplication on the GPU.
 *
 * This function compares the result of matrix multiplication with the expected result
 * and prints any discrepancies found. If a discrepancy is detected, the function exits
 * with a status of -1.
 *
 * @param pbA Pointer to the GPU buffer containing matrix A.
 * @param pbB Pointer to the GPU buffer containing matrix B.
 * @param pbC Pointer to the GPU buffer containing the result matrix C.
 * @param m The number of rows in matrix A and matrix C.
 * @param k The number of columns in matrix A and matrix B.
 * @param n The number of columns in matrix B and matrix C.
 */
void verifySGEMM(GpuBuffer<NNFloat>* pbA, GpuBuffer<NNFloat>* pbB, GpuBuffer<NNFloat>* pbC, uint32_t m, uint32_t k, uint32_t n)
{
    std::vector<NNFloat> vA(m * k);
    std::vector<NNFloat> vB(k * n);
    std::vector<NNFloat> vC(m * n);
    
    pbA->Download(vA.data());
    pbB->Download(vB.data());
    pbC->Download(vC.data());
    
    for (uint32_t i = 0; i < m; i++)
    {
        for (uint32_t j = 0; j < n; j++)
        {
            NNFloat sum = 0.0;
            NNFloat* pA = vA.data() + i * k;
            NNFloat* pB = vB.data() + j;
            
            for (uint32_t kk = 0; kk < k; kk++)
            {
                sum += *pA * (*pB);
                pA++;
                pB += n;
            }
            
            if (std::fabs(sum - vC[i * n + j]) > 0.000001f)
            {
                std::cout << std::setw(3) << i << ' ' << std::setw(3) << j << ' '
                          << std::setw(16) << sum << ' ' << std::setw(16) << vC[i * n + j] << '\n';
            }
        }
    }
    
    std::exit(-1);
}

/**
 * @brief Verifies the result of matrix multiplication on the GPU.
 *
 * This function compares the result of matrix multiplication with the expected result
 * and prints any discrepancies found. If a discrepancy is detected, the function exits
 * with a status of -1.
 *
 * @param pbA Pointer to the GPU buffer containing matrix A.
 * @param pbB Pointer to the GPU buffer containing matrix B.
 * @param pbC Pointer to the GPU buffer containing the result matrix C.
 * @param m The number of rows in matrix A and matrix C.
 * @param k The number of columns in matrix A and matrix B.
 * @param n The number of columns in matrix B and matrix C.
 */
void verifySGEMMNT(GpuBuffer<NNFloat>* pbA, GpuBuffer<NNFloat>* pbB, GpuBuffer<NNFloat>* pbC, uint32_t m, uint32_t k, uint32_t n)
{
    std::vector<NNFloat> vA(m * k);
    std::vector<NNFloat> vB(k * n);
    std::vector<NNFloat> vC(m * n);
    
    pbA->Download(vA.data());
    pbB->Download(vB.data());
    pbC->Download(vC.data());

    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            NNFloat sum = 0.0;
            NNFloat* pA = vA.data() + i * k;
            NNFloat* pB = vB.data() + j * k;
            
            for (size_t kk = 0; kk < k; kk++)
            {
                sum += *pA * (*pB);
                pA++;
                pB++;
            }
            
            if (std::fabs(sum - vC[i * n + j]) / (std::fabs(sum) + 0.00000000000001f) > 0.000002f)
            {
                std::cout << std::setw(3) << i << ' ' << std::setw(3) << j << ' '
                          << std::setw(16) << sum << ' ' << std::setw(16) << vC[i * n + j] << '\n';
            }
        }
    }
    
    std::cout << m << ' ' << k << ' ' << n << '\n';
    std::exit(-1);
}

/**
 * @brief Verifies the result of matrix multiplication on the GPU.
 *
 * This function compares the result of matrix multiplication with the expected result
 * and prints any discrepancies found. If a discrepancy is detected, the function exits
 * with a status of -1.
 *
 * @param pbA Pointer to the GPU buffer containing matrix A.
 * @param pbB Pointer to the GPU buffer containing matrix B.
 * @param pbC Pointer to the GPU buffer containing the result matrix C.
 * @param m The number of rows in matrix A and matrix C.
 * @param k The number of columns in matrix A and the number of rows in matrix B.
 * @param n The number of columns in matrix B and matrix C.
 */
void verifySGEMMTN(GpuBuffer<NNFloat>* pbA, GpuBuffer<NNFloat>* pbB, GpuBuffer<NNFloat>* pbC, uint32_t m, uint32_t k, uint32_t n)
{
    std::cout << m << ' ' << k << ' ' << n << '\n';
    std::vector<NNFloat> vA(m * k);
    std::vector<NNFloat> vB(k * n);
    std::vector<NNFloat> vC(m * n);
    
    pbA->Download(vA.data());
    pbB->Download(vB.data());
    pbC->Download(vC.data());
    
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            NNFloat sum = 0.0;
            NNFloat* pA = vA.data() + i;
            NNFloat* pB = vB.data() + j;
            
            for (size_t kk = 0; kk < k; kk++)
            {
                sum += *pA * (*pB);
                pA += m;
                pB += n;
            }
            
            if (std::fabs(sum - vC[i * n + j]) / (std::fabs(sum) + 0.00000000000001f) > 0.000005f)
            {
                std::cout << std::setw(3) << i << ' ' << std::setw(3) << j << ' '
                          << std::setw(16) << sum << ' ' << std::setw(16) << vC[i * n + j] << '\n';
            }
        }
    }
    
    std::cout << m << ' ' << k << ' ' << n << '\n';
    std::exit(-1);
}
