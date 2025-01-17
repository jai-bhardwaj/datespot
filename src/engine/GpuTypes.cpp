#include "GpuTypes.h"
#include "Types.h"
#include "kernels.h"
#include <array>
#include <string_view>
#include <memory>
#include <vector>

static const float cAcceptableError = 0.00001f;

class GpuContext {
public:
    /**
     * @brief GpuContext constructor.
     */
    GpuContext();

    /**
     * @brief GpuContext destructor.
     */
    ~GpuContext();

    /**
     * @brief Set the CPU validation flag.
     *
     * @param bValidate Boolean value indicating whether CPU validation should be enabled or disabled.
     */
    void SetCPUValidate(bool bValidate);

    /**
     * @brief Initialize the GPU context.
     *
     * @param argc The number of command-line arguments.
     * @param argv The command-line arguments.
     */
    void Startup(int argc, char** argv);

    /**
     * @brief Copy constants to the GPU.
     */
    void CopyConstants();

    /**
     * @brief Set the fast math flag.
     *
     * @param flag Boolean value indicating whether fast math should be enabled or disabled.
     */
    void SetFastMath(bool flag);

    /**
     * @brief Shutdown the GPU context.
     */
    void Shutdown();

    /**
     * @brief Set the neural network for the GPU context.
     *
     * @param pNetwork Pointer to the neural network.
     */
    void SetNeuralNetwork(Network* pNetwork);

    /**
     * @brief Set the random seed for the GPU context.
     *
     * @param seed The random seed value.
     */
    void SetRandomSeed(unsigned long seed);

    /**
     * @brief Get the memory usage of the GPU and CPU.
     *
     * @param gpuMemory Pointer to store the GPU memory usage.
     * @param cpuMemory Pointer to store the CPU memory usage.
     */
    void GetMemoryUsage(int* gpuMemory, int* cpuMemory);


private:
    /**
     * @brief Maximum length of a processor name.
     */
    static constexpr int MaxProcessorName = MPI_MAX_PROCESSOR_NAME + 1;

    /**
     * @brief Maximum number of GPUs.
     */
    static constexpr int MaxGPUs = 32;

    /**
     * @brief Flag indicating whether ECC (Error Correction Code) support is available.
     */
    bool _bECCSupport;

    /**
     * @brief Flag indicating whether host memory can be mapped to the GPU.
     */
    bool _bCanMapHostMemory;

    /**
     * @brief Flag indicating whether CPU validation is enabled.
     */
    bool _bCPUValidate;

    /**
     * @brief Flag indicating whether unified memory is enabled.
     */
    bool _bUnifiedMemory;

    /**
     * @brief Flag indicating whether the current process is a single node.
     */
    bool _bSingleNode;

    /**
     * @brief Flag indicating whether peer-to-peer access between GPUs is enabled.
     */
    bool _bP2P;

    /**
     * @brief The acceptable error value for comparisons.
     */
    float _acceptableError;

    /**
     * @brief The total CPU memory available.
     */
    unsigned long long int _totalCPUMemory;

    /**
     * @brief The total GPU memory available.
     */
    unsigned long long int _totalGPUMemory;

    /**
     * @brief The number of processes.
     */
    int _numprocs;

    /**
     * @brief The ID of the current process.
     */
    int _id;

    /**
     * @brief The SM (Streaming Multiprocessor) version.
     */
    int _sm_version;

    /**
     * @brief The major version of SM.
     */
    int _sm_major;

    /**
     * @brief The warp size.
     */
    int _warpSize;

    /**
     * @brief The maximum number of sparse connections supported.
     */
    int _maxSparse;

    /**
     * @brief The maximum number of analog sparse connections supported.
     */
    int _maxSparseAnalog;

    /**
     * @brief The cuBLAS handle for GPU operations.
     */
    cudaStream_t _cuBLASHandle;

    /**
     * @brief The cuDNN handle for GPU deep neural network operations.
     */
    cudaStream_t _cuDNNHandle;

    /**
     * @brief The accumulator for parallel operations.
     */
    std::unique_ptr<GpuBuffer<unsigned long long int>> _pbAccumulator;

    /**
     * @brief Pointer to the accumulator for parallel operations.
     */
    GpuBuffer<unsigned long long int>* _pAccumulator;

    /**
     * @brief Pointer to the network object.
     */
    Network* _pNetwork;

    /**
     * @brief The device properties of the current GPU.
     */
    cudaDeviceProp _deviceProp;

    /**
     * @brief The GPU device ID.
     */
    int _device;

    /**
     * @brief The number of threads per block.
     */
    int _threadsPerBlock;

    /**
     * @brief The number of bits in a warp.
     */
    int _warpBits;

    /**
     * @brief The bitmask for a warp.
     */
    int _warpMask;

    /**
     * @brief GPU data used in computations.
     */
    GpuData _data;

    /**
     * @brief The cuRand generator for GPU random number generation.
     */
    curandGenerator_t _RNG;

    /**
     * @brief The total memory available.
     */
    int _totalMemory;

    /**
     * @brief Initialize CUDA.
     */
    void InitializeCUDA();

    /**
     * @brief Initialize MPI.
     *
     * @param argc The number of command-line arguments.
     * @param argv The command-line arguments.
     */
    void InitializeMPI(int argc, char** argv);

    /**
     * @brief Initialize GPU devices.
     */
    void InitializeGPUDevices();

    /**
     * @brief Select compatible GPU.
     */
    void SelectCompatibleGPU();

    /**
     * @brief Enable peer access between GPUs.
     */
    void EnablePeerAccess();

    /**
     * @brief Initialize cuBLAS.
     */
    void InitializeCuBLAS();

    /**
     * @brief Initialize cuDNN.
     */
    void InitializeCuDNN();

    /**
     * @brief Initialize cuRand.
     */
    void InitializeCuRand();

    /**
     * @brief Verify the correctness of SGEMM operation (A * B = C) on the GPU.
     *
     * @param pbA Pointer to the GPU buffer A.
     * @param pbB Pointer to the GPU buffer B.
     * @param pbC Pointer to the GPU buffer C.
     * @param m The number of rows in A and C.
     * @param k The number of columns in A and rows in B.
     * @param n The number of columns in B and C.
     */
    void VerifySGEMM(GpuBuffer<Float>* pbA, GpuBuffer<Float>* pbB, GpuBuffer<Float>* pbC, uint32_t m, uint32_t k, uint32_t n);

    /**
     * @brief Verify the correctness of SGEMM operation (A * B' = C) on the GPU.
     *
     * @param pbA Pointer to the GPU buffer A.
     * @param pbB Pointer to the GPU buffer B.
     * @param pbC Pointer to the GPU buffer C.
     * @param m The number of rows in A and C.
     * @param k The number of columns in A and B.
     * @param n The number of columns in B' and C.
     */
    void VerifySGEMMNT(GpuBuffer<Float>* pbA, GpuBuffer<Float>* pbB, GpuBuffer<Float>* pbC, uint32_t m, uint32_t k, uint32_t n);

    /**
     * @brief Verify the correctness of SGEMM operation (A' * B = C) on the GPU.
     *
     * @param pbA Pointer to the GPU buffer A.
     * @param pbB Pointer to the GPU buffer B.
     * @param pbC Pointer to the GPU buffer C.
     * @param m The number of rows in A' and C.
     * @param k The number of columns in A' and B.
     * @param n The number of columns in B and C.
     */
    void VerifySGEMMTN(GpuBuffer<Float>* pbA, GpuBuffer<Float>* pbB, GpuBuffer<Float>* pbC, uint32_t m, uint32_t k, uint32_t n);


public:
    /**
     * @brief GpuContext constructor.
     */
    GpuContext::GpuContext()
        : _bECCSupport(false),
        _bCanMapHostMemory(false),

        /**
         * @brief Flag indicating whether CPU validation is enabled.
         */
        _bCPUValidate(false),

        /**
         * @brief Flag indicating whether unified memory is enabled.
         */
        _bUnifiedMemory(false),

        /**
         * @brief Flag indicating whether the current process is a single node.
         */
        _bSingleNode(false),

        /**
         * @brief Flag indicating whether peer-to-peer access between GPUs is enabled.
         */
        _bP2P(false),

        /**
         * @brief The acceptable error value for comparisons.
         */
        _acceptableError(cAcceptableError),

        /**
         * @brief The total CPU memory available.
         */
        _totalCPUMemory(0),

        /**
         * @brief The total GPU memory available.
         */
        _totalGPUMemory(0),

        /**
         * @brief The number of processes.
         */
        _numprocs(1),

        /**
         * @brief The ID of the current process.
         */
        _id(0),

        /**
         * @brief The SM (Streaming Multiprocessor) version.
         */
        _sm_version(SM_3X),

        /**
         * @brief The major version of SM.
         */
        _sm_major(0),

        /**
         * @brief The warp size.
         */
        _warpSize(32),

        /**
         * @brief The maximum number of sparse connections supported.
         */
        _maxSparse(SM_3X_MAXSPARSE),

        /**
         * @brief The maximum number of analog sparse connections supported.
         */
        _maxSparseAnalog(SM_3X_MAXSPARSEANALOG),

        /**
         * @brief The cuBLAS handle for GPU operations.
         */
        _cuBLASHandle(0),

        /**
         * @brief The cuDNN handle for GPU deep neural network operations.
         */
        _cuDNNHandle(0),

        /**
         * @brief The accumulator for parallel operations.
         */
        _pbAccumulator()
    {
    }

    /**
     * @brief GpuContext destructor.
     */
    GpuContext::~GpuContext()
    {
    }

    /**
     * @brief Set the CPU validation flag.
     * 
     * @param bValidate Boolean value indicating whether CPU validation should be enabled or disabled.
     */
    void GpuContext::SetCPUValidate(bool bValidate)
    {
        _bCPUValidate = bValidate;
    }

    /**
     * @brief Initialize the GPU context.
     * 
     * @param argc The number of command-line arguments.
     * @param argv The command-line arguments.
     */
    void GpuContext::Startup(int argc, char** argv)
    {
        /**
         * @brief Initializes CUDA.
         */
        InitializeCUDA();

        /**
         * @brief Initializes MPI.
         *
         * @param argc Number of command line arguments.
         * @param argv Array of command line arguments.
         */
        InitializeMPI(argc, argv);

        /**
         * @brief Initializes GPU devices.
         */
        InitializeGPUDevices();

        /**
         * @brief Selects compatible GPU.
         */
        SelectCompatibleGPU();

        /**
         * @brief Enables peer access between GPUs.
         */
        EnablePeerAccess();

        /**
         * @brief Initializes cuBLAS on the GPU.
         */
        InitializeCuBLAS();

        /**
         * @brief Initializes cuDNN on the GPU.
         */
        InitializeCuDNN();

        /**
         * @brief Initializes cuRand on the GPU.
         */
        InitializeCuRand();

        /**
         * @brief Copies constants.
         */
        CopyConstants();
    }

    /**
     * @brief Copy constants to the GPU.
     */
    void GpuContext::CopyConstants()
    {
        /**
         * @brief Sets the GPU data for kernels.
         */
        SetKernelsGpuData();

        /**
         * @brief Sets the GPU data for KLoss.
         */
        SetKLossGpuData();

        /**
         * @brief Sets the GPU data for KActivation.
         */
        SetKActivationGpuData();

        /**
         * @brief Sets the GPU data for KDelta.
         */
        SetKDeltaGpuData();
    }

    /**
     * @brief Set the fast math flag.
     * 
     * @param flag Boolean value indicating whether fast math should be enabled or disabled.
     */
    void GpuContext::SetFastMath(bool flag)
    {
        cublasMath_t mathMode = flag ? CUBLAS_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
        cublasStatus_t cstatus = CUBLAS_STATUS_SUCCESS;
        if (_sm_major >= 7)
        {
            cstatus = cublasSetMathMode(_cuBLASHandle, mathMode);
            if (cstatus != CUBLAS_STATUS_SUCCESS)
            {
                std::cout("GpuContext::SetFastMath: failed to set math mode\n");
            }
        }
        else
        {
            std::cout("GpuContext::SetFastMath: failed to set math mode because GPU SM revision is <7.0\n");
        }
    }

    /**
     * @brief Shutdown the GPU context.
     */
    void GpuContext::Shutdown()
    {
        /**
         * @brief Resets the accumulator for the current instance.
         */
        _pbAccumulator.reset();

        /**
         * @brief Shuts down cuBLAS on the GPU for the current process.
         *
         * @param _device The GPU device ID.
         */
        std::cout("GpuContext::Shutdown: Shutting down cuBLAS on GPU for process %d\n", _device);
        cublasStatus_t cstatus = cublasDestroy(_cuBLASHandle);
        if (cstatus != CUBLAS_STATUS_SUCCESS)
        {
            std::cout("GpuContext::Shutdown: Failed to shut down cuBLAS on GPU for process %d.\n", _device);
        }
        std::cout("GpuContext::Shutdown: CuBLAS shut down on GPU for process %d\n", _device);

        /**
         * @brief Shuts down cuDNN on the GPU for the current process.
         *
         * @param _device The GPU device ID.
         */
        std::cout("GpuContext::Shutdown: Shutting down cuDNN on GPU for process %d\n", _device);
        cudnnStatus_t cdstatus = cudnnDestroy(_cuDNNHandle);
        if (cdstatus != CUDNN_STATUS_SUCCESS)
        {
            std::cout("GpuContext::Shutdown: Failed to shut down cuDNN on GPU for process %d.\n", _device);
        }
        std::cout("GpuContext::Shutdown: CuDNN shut down on GPU for process %d\n", _device);

        /**
         * @brief Shuts down cuRand on the GPU for the current process.
         *
         * @param _device The GPU device ID.
         */
        std::cout("GpuContext::Shutdown: Shutting down cuRand on GPU for process %d\n", _device);
        curandStatus_t crstatus = curandDestroyGenerator(_RNG);
        if (crstatus != CURAND_STATUS_SUCCESS)
        {
            std::cout("GpuContext::Shutdown: Failed to shut down cuRand on GPU for process %d.\n", _device);
        }
        std::cout("GpuContext::Shutdown: CuRand shut down on GPU for process %d\n", _device);

        /**
         * @brief Exits the CUDA thread.
         */
        cudaThreadExit();

        /**
         * @brief Finalizes the MPI process.
         */
        MPI_Finalize();
        std::cout("GpuContext::Shutdown: Process %d out of %d finalized.\n", _id, _numprocs);

    }

    /**
     * @brief Set the neural network for the GPU context.
     * 
     * @param pNetwork Pointer to the neural network.
     */
    void GpuContext::SetNeuralNetwork(Network* pNetwork)
    {
        /**
         * @brief Sets the network object for the current instance.
         *
         * @param pNetwork Pointer to the network object.
         */
        _pNetwork = pNetwork;

        /**
         * @brief Copies LRN (Local Response Normalization) parameters from the network object.
         */
        _data._LRN_k = pNetwork->_LRN_k;
        _data._LRN_n = pNetwork->_LRN_n;
        _data._LRN_alpha = pNetwork->_LRN_alpha;
        _data._LRN_beta = pNetwork->_LRN_beta;

        /**
         * @brief Copies Maxout parameters from the network object.
         */
        _data._maxout_k = pNetwork->_maxout_k;

        /**
         * @brief Copies sparseness penalty parameters from the network object.
         */
        _data._bSparsenessPenalty = pNetwork->_bSparsenessPenalty;
        _data._sparsenessPenalty_p = pNetwork->_sparsenessPenalty_p;
        _data._sparsenessPenalty_beta = pNetwork->_sparsenessPenalty_beta;

        /**
         * @brief Copies denoising parameters from the network object.
         */
        _data._bDenoising = pNetwork->_bDenoising;
        _data._denoising_p = pNetwork->_denoising_p;
        _data._denoising_q = 1.0f / (1.0f - pNetwork->_denoising_p);

        /**
         * @brief Copies delta boost parameters from the network object.
         */
        _data._deltaBoost_one = pNetwork->_deltaBoost_one;
        _data._deltaBoost_zero = pNetwork->_deltaBoost_zero;

        /**
         * @brief Copies SMCE (Softmax Cross Entropy) parameters from the network object.
         */
        _data._SMCE_oneTarget = pNetwork->_SMCE_oneTarget;
        _data._SMCE_zeroTarget = pNetwork->_SMCE_zeroTarget;
        _data._SMCE_oneScale = pNetwork->_SMCE_oneScale;
        _data._SMCE_zeroScale = pNetwork->_SMCE_zeroScale;

        /**
         * @brief Copies shuffle indices and checks if shuffling is enabled.
         */
        _data._bShuffleIndices = pNetwork->_bShuffleIndices && (pNetwork->_mode == Mode::Training);
        _data._pShuffleIndex = pNetwork->_pShuffleIndex;

        CopyConstants();
    }

    /**
     * @brief Set the random seed for the GPU context.
     * 
     * @param seed The random seed value.
     */
    void GpuContext::SetRandomSeed(unsigned long seed)
    {
        curandStatus_t crstatus = curandSetPseudoRandomGeneratorSeed(_RNG, seed + (unsigned long)_device * 76801ull);
        if (crstatus != CURAND_STATUS_SUCCESS)
        {
            if (getGpu()._id == 0)
                std::cout("GpuContext::SetRandomSeed: Failed to set cuRand seed on GPU for process %d, exiting.\n", _device);
            Shutdown();
            exit(-1);
        }
        srand(seed);

        if (getGpu()._id == 0)
            std::cout("GpuContext::SetRandomSeed: Random seed set to %lu.\n", seed);
    }

    /**
     * @brief Get the memory usage of the GPU and CPU.
     * 
     * @param gpuMemory Pointer to store the GPU memory usage.
     * @param cpuMemory Pointer to store the CPU memory usage.
     */
    void GpuContext::GetMemoryUsage(int* gpuMemory, int* cpuMemory)
    {
        *gpuMemory = (int)(_totalGPUMemory / 1024ll);
        *cpuMemory = (int)(_totalCPUMemory / 1024ll);
        return;
    }

private:
    /**
     * Initializes the CUDA environment and selects the GPU device.
     * Must be called before any other GPU operations.
     */
    void GpuContext::InitializeCUDA() {
        int deviceCount;
        cudaError_t status = cudaGetDeviceCount(&deviceCount);
        RTERROR(status, "cudaGetDeviceCount failed");

        if (deviceCount == 0) {
            std::cout("GpuContext::InitializeCUDA: No CUDA devices found, exiting.\n");
            exit(-1);
        }

        status = cudaSetDevice(_id);
        RTERROR(status, "cudaSetDevice failed");

        status = cudaGetDeviceProperties(&_deviceProp, _id);
        RTERROR(status, "cudaGetDeviceProperties failed");

        if (_deviceProp.major < 3) {
            std::cout("GpuContext::InitializeCUDA: CUDA devices with compute capability < 3.0 are not supported, exiting.\n");
            exit(-1);
        }

        _sm_major = _deviceProp.major;
        _sm_version = 10 * _sm_major;
        _warpSize = _deviceProp.warpSize;
        _threadsPerBlock = _deviceProp.maxThreadsPerBlock;

        if (_deviceProp.unifiedAddressing) {
            status = cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceScheduleBlockingSync);
            RTERROR(status, "cudaSetDeviceFlags failed");
            _bCanMapHostMemory = true;
            _bUnifiedMemory = true;
            std::cout("GpuContext::InitializeCUDA: Unified addressing and can map host memory available.\n");
        }

        std::cout("GpuContext::InitializeCUDA: Using GPU %d: %s, SM %d.%d, Warp Size %d\n",
                _id,
                _deviceProp.name,
                _deviceProp.major,
                _deviceProp.minor,
                _warpSize);
    }

    /**
     * Initializes the MPI environment.
     * Must be called before using MPI functions.
     * 
     * @param argc The number of command line arguments
     * @param argv The command line arguments
     */
    void GpuContext::InitializeMPI(int argc, char** argv) {
        int status = MPI_Init(&argc, &argv);
        if (status != MPI_SUCCESS) {
            std::cout("GpuContext::InitializeMPI: Failed to initialize MPI, exiting.\n");
            exit(-1);
        }

        MPI_Comm_size(MPI_COMM_WORLD, &_numprocs);
        MPI_Comm_rank(MPI_COMM_WORLD, &_id);

        if (_numprocs < 1) {
            std::cout("GpuContext::InitializeMPI: Invalid number of processes, exiting.\n");
            MPI_Finalize();
            exit(-1);
        }

        std::cout("GpuContext::InitializeMPI: Process %d out of %d started.\n", _id, _numprocs);
    }

    /**
     * Initializes the GPU devices and selects the specified device ID.
     * Must be called after initializing the CUDA environment.
     */
    void GpuContext::InitializeGPUDevices() {
        int deviceCount;
        cudaError_t status = cudaGetDeviceCount(&deviceCount);
        RTERROR(status, "cudaGetDeviceCount failed");

        if (_id < 0 || _id >= deviceCount) {
            std::cout("GpuContext::InitializeGPUDevices: Invalid device ID %d, exiting.\n", _id);
            MPI_Finalize();
            exit(-1);
        }

        std::cout("GpuContext::InitializeGPUDevices: %d devices found.\n", deviceCount);
        std::cout("GpuContext::InitializeGPUDevices: Using GPU %d for process %d.\n", _id, _id);

        _device = _id;
        status = cudaSetDevice(_device);
        RTERROR(status, "cudaSetDevice failed");

        status = cudaGetDeviceProperties(&_deviceProp, _device);
        RTERROR(status, "cudaGetDeviceProperties failed");

        std::cout("GpuContext::InitializeGPUDevices: Using GPU %d: %s, SM %d.%d, Warp Size %d\n",
                _device,
                _deviceProp.name,
                _deviceProp.major,
                _deviceProp.minor,
                _warpSize);
    }

    /**
     * Selects a compatible GPU device with compute capability >= 3.0.
     * Must be called after initializing the CUDA environment.
     */
    void GpuContext::SelectCompatibleGPU() {
        int deviceCount;
        cudaError_t status = cudaGetDeviceCount(&deviceCount);
        RTERROR(status, "cudaGetDeviceCount failed");

        std::vector<int> compatibleDevices;
        for (int i = 0; i < deviceCount; i++) {
            cudaDeviceProp prop;
            status = cudaGetDeviceProperties(&prop, i);
            RTERROR(status, "cudaGetDeviceProperties failed");

            if (prop.major >= 3) {
                compatibleDevices.push_back(i);
            }
        }

        int numCompatibleDevices = compatibleDevices.size();
        if (numCompatibleDevices == 0) {
            std::cout("GpuContext::SelectCompatibleGPU: No compatible CUDA devices found, exiting.\n");
            exit(-1);
        }

        int selectedDevice = compatibleDevices[_id % numCompatibleDevices];

        status = cudaSetDevice(selectedDevice);
        RTERROR(status, "cudaSetDevice failed");

        status = cudaGetDeviceProperties(&_deviceProp, selectedDevice);
        RTERROR(status, "cudaGetDeviceProperties failed");

        _device = selectedDevice;

        std::cout("GpuContext::SelectCompatibleGPU: Using compatible GPU %d: %s, SM %d.%d, Warp Size %d\n",
                _device,
                _deviceProp.name,
                _deviceProp.major,
                _deviceProp.minor,
                _warpSize);
    }

    /**
     * Enables peer access between GPUs with compute capability >= 3.0.
     * Must be called after initializing the CUDA environment and selecting the GPU device.
     */
    void GpuContext::EnablePeerAccess() {
        int deviceCount;
        cudaError_t status = cudaGetDeviceCount(&deviceCount);
        RTERROR(status, "cudaGetDeviceCount failed");

        for (int i = 0; i < deviceCount; i++) {
            if (i != _device) {
                cudaDeviceProp prop;
                status = cudaGetDeviceProperties(&prop, i);
                RTERROR(status, "cudaGetDeviceProperties failed");

                if (prop.major >= 3) {
                    cudaSetDevice(i);
                    status = cudaDeviceEnablePeerAccess(_device, 0);
                    if (status != cudaErrorPeerAccessAlreadyEnabled && status != cudaSuccess) {
                        std::cout("GpuContext::EnablePeerAccess: Failed to enable peer access between GPU %d and GPU %d, exiting.\n",
                                _device,
                                i);
                        MPI_Finalize();
                        exit(-1);
                    }
                }
            }
        }

        cudaSetDevice(_device);
        std::cout("GpuContext::EnablePeerAccess: Peer access enabled for GPU %d.\n", _device);
    }

    /**
     * Initializes the cuBLAS library on the GPU device.
     * Must be called after selecting the GPU device.
     */
    void GpuContext::InitializeCuBLAS() {
        cublasStatus_t status = cublasCreate(&_cuBLASHandle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cout("GpuContext::InitializeCuBLAS: Failed to initialize cuBLAS on GPU for process %d, exiting.\n", _device);
            Shutdown();
            exit(-1);
        }

        std::cout("GpuContext::InitializeCuBLAS: cuBLAS initialized on GPU for process %d.\n", _device);
    }

    /**
     * Initializes the cuDNN library on the GPU device.
     * Must be called after selecting the GPU device.
     */
    void GpuContext::InitializeCuDNN() {
        cudnnStatus_t status = cudnnCreate(&_cuDNNHandle);
        if (status != CUDNN_STATUS_SUCCESS) {
            std::cout("GpuContext::InitializeCuDNN: Failed to initialize cuDNN on GPU for process %d, exiting.\n", _device);
            Shutdown();
            exit(-1);
        }

        std::cout("GpuContext::InitializeCuDNN: cuDNN initialized on GPU for process %d.\n", _device);
    }

    /**
     * Initializes the cuRand library on the GPU device.
     * Must be called after selecting the GPU device.
     */
    void GpuContext::InitializeCuRand() {
        curandStatus_t status = curandCreateGenerator(&_RNG, CURAND_RNG_PSEUDO_DEFAULT);
        if (status != CURAND_STATUS_SUCCESS) {
            std::cout("GpuContext::InitializeCuRand: Failed to initialize cuRand on GPU for process %d, exiting.\n", _device);
            Shutdown();
            exit(-1);
        }

        status = curandSetStream(_RNG, 0);
        if (status != CURAND_STATUS_SUCCESS) {
            std::cout("GpuContext::InitializeCuRand: Failed to set cuRand stream on GPU for process %d, exiting.\n", _device);
            Shutdown();
            exit(-1);
        }

        std::cout("GpuContext::InitializeCuRand: cuRand initialized on GPU for process %d.\n", _device);
    }

    /**
     * Copies constant data to the GPU device.
     * Must be called after initializing the CUDA environment and selecting the GPU device.
     */
    void GpuContext::CopyConstants() {
        cudaError_t status = cudaMemcpyToSymbol(_cudaConstData, &_data, sizeof(GpuData));
        RTERROR(status, "cudaMemcpyToSymbol failed");
    }
};

#endif /* GPU_CONTEXT_H_ */
