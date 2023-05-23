#ifndef __GPUTYPES_H__
#define __GPUTYPES_H__

#include <cstdio>
#include <unistd.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cudnn.h>
#include <vector_functions.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <cassert>
#include <mpi.h>
#include <memory>

/**
 * @brief Defines the precision mode for the GPU calculations.
 */
#define use_SPFP

#if !(defined(use_DPFP) && !defined(use_HPFP) && !defined(use_SPFP)) && \
    !(defined(use_HPFP) && !defined(use_DPFP) && !defined(use_SPFP)) && \
    !(defined(use_SPFP) && !defined(use_DPFP) && !defined(use_HPFP))
#error "You must define one and only one precision mode (use_SPFP, use_HPFP, or use_DPFP). Aborting compilation."
#endif

#define ESCALE  (1ll << 30)
/**
 * @brief Scaling factor for error calculations.
 *
 * This constant is a static double value representing the scaling factor for error calculations.
 */
static const double ERRORSCALE = ESCALE;

/**
 * @brief Scaling factor for error calculations as a float.
 *
 * This constant is a static float value representing the scaling factor for error calculations.
 */
static const float ERRORSCALEF = ESCALE;

/**
 * @brief Inverse of the scaling factor for error calculations as a double.
 *
 * This constant is a static double value representing the inverse of the scaling factor for error calculations.
 */
static const double ONEOVERERRORSCALE = 1.0 / static_cast<double>(ERRORSCALE);

/**
 * @brief Inverse of the scaling factor for error calculations as a float.
 *
 * This constant is a static float value representing the inverse of the scaling factor for error calculations.
 */
static const float ONEOVERERRORSCALEF = static_cast<float>(1.0 / static_cast<double>(ERRORSCALE));

/**
 * @brief Custom type for aligned double values.
 */
typedef double                  __align__(8)    aligned_double;

/**
 * @brief Custom type for aligned unsigned long int values.
 */
typedef unsigned long int       __align__(8)    aligned_uli;

/**
 * @brief Custom type for aligned long long int values.
 */
typedef long long int           __align__(8)    aligned_lli;

/**
 * @brief Custom type for aligned unsigned long long int values.
 */
typedef unsigned long long int  __align__(8)    UllInt;

/**
 * @brief Definition of the double precision type based on the precision mode.
 */
#if defined(use_DPFP)
/**
 * @typedef NNAccumulator
 * @brief Alias for the double data type with an alignment of 8, representing an accumulator in a neural network.
 */
typedef double __align__(8) NNAccumulator;

/**
 * @typedef NNDouble
 * @brief Alias for the double data type with an alignment of 8.
 */
typedef double __align__(8) NNDouble;

/**
 * @typedef Float
 * @brief Alias for the double data type with an alignment of 8.
 */
typedef double __align__(8) Float;

/**
 * @typedef NNDouble2
 * @brief Alias for the double2 data type with an alignment of 16.
 */
typedef double2 __align__(16) NNDouble2;

/**
 * @typedef NNDouble4
 * @brief Alias for the double4 data type with an alignment of 32.
 */
typedef double4 __align__(32) NNDouble4;

/**
 * @typedef Float2
 * @brief Alias for the double2 data type with an alignment of 16.
 */
typedef double2 __align__(16) Float2;

/**
 * @typedef Float4
 * @brief Alias for the double4 data type with an alignment of 16.
 */
typedef double4 __align__(16) Float4;

/**
 * @brief MPI data type for NNDouble, equivalent to MPI_DOUBLE_PRECISION.
 */
static const MPI_Datatype MPI_NNDOUBLE = MPI_DOUBLE_PRECISION;

/**
 * @brief MPI data type for Float, equivalent to MPI_DOUBLE_PRECISION.
 */
static const MPI_Datatype MPI_Float = MPI_DOUBLE_PRECISION;

/**
 * @brief MPI data type for NNAccumulator, equivalent to MPI_FLOAT.
 */
static const MPI_Datatype MPI_NNACCUMULATOR = MPI_FLOAT;
#elif defined(use_SPFP)
/**
 * @typedef NNAccumulator
 * @brief Alias for the float data type representing an accumulator in a neural network.
 */
typedef float NNAccumulator;

/**
 * @typedef NNDouble
 * @brief Alias for the double data type with an alignment of 8.
 */
typedef double __align__(8) NNDouble;

/**
 * @typedef Float
 * @brief Alias for the float data type.
 */
typedef float Float;

/**
 * @typedef NNDouble2
 * @brief Alias for the double2 data type with an alignment of 16.
 */
typedef double2 __align__(16) NNDouble2;

/**
 * @typedef NNDouble4
 * @brief Alias for the double4 data type with an alignment of 32.
 */
typedef double4 __align__(32) NNDouble4;

/**
 * @typedef Float2
 * @brief Alias for the float2 data type with an alignment of 8.
 */
typedef float2 __align__(8) Float2;

/**
 * @typedef Float4
 * @brief Alias for the float4 data type with an alignment of 16.
 */
typedef float4 __align__(16) Float4;

/**
 * @brief MPI data type for NNDouble.
 */
static const MPI_Datatype MPI_NNDOUBLE = MPI_DOUBLE_PRECISION;

/**
 * @brief MPI data type for Float.
 */
static const MPI_Datatype MPI_Float = MPI_FLOAT;

/**
 * @brief MPI data type for NNAccumulator.
 */
static const MPI_Datatype MPI_NNACCUMULATOR = MPI_LONG_LONG_INT;
#else
/**
 * @typedef NNAccumulator
 * @brief Alias for the float data type representing an accumulator in a neural network.
 */
typedef float NNAccumulator;

/**
 * @typedef NNDouble
 * @brief Alias for the double data type with an alignment of 8.
 */
typedef double __align(8)__ NNDouble;

/**
 * @typedef Float
 * @brief Alias for the float data type.
 */
typedef float Float;

/**
 * @typedef NNDouble2
 * @brief Alias for the double2 data type with an alignment of 16.
 */
typedef double2 __align(16)__ NNDouble2;

/**
 * @typedef NNDouble4
 * @brief Alias for the double4 data type with an alignment of 32.
 */
typedef double4 __align(32)__ NNDouble4;

/**
 * @typedef Float2
 * @brief Alias for the float2 data type with an alignment of 8.
 */
typedef float2 __align(8)__ Float2;

/**
 * @typedef Float4
 * @brief Alias for the float4 data type with an alignment of 16.
 */
typedef float4 __align(16)__ Float4;

/**
 * @brief MPI data type for NNDouble.
 */
static const MPI_Datatype MPI_NNDOUBLE = MPI_DOUBLE_PRECISION;

/**
 * @brief MPI data type for Float.
 */
static const MPI_Datatype MPI_Float = MPI_FLOAT;

/**
 * @brief MPI data type for NNAccumulator.
 */
static const MPI_Datatype MPI_NNACCUMULATOR = MPI_LONG_LONG_INT;
#endif

/**
 * @brief Maximum number of threads per block for SM 3.x architecture.
 */
static const int SM_3X_THREADS_PER_BLOCK                        = 128;

/**
 * @brief Maximum number of threads per block for SM 5.x architecture.
 */
static const int SM_5X_THREADS_PER_BLOCK                        = 128;

/**
 * @brief Maximum number of threads per block for SM 6.x architecture.
 */
static const int SM_6X_THREADS_PER_BLOCK                        = 128;

/**
 * @brief Macro to define launch bounds based on the current SM architecture.
 */
#if (__CUDA_ARCH__ >= 600)
/**
 * @def LAUNCH_BOUNDS
 * @brief Macro that sets the launch bounds to a specified number of threads per block.
 *
 * This macro sets the launch bounds for CUDA kernels to a specified number of threads per block,
 * as defined by the constant SM_6X_THREADS_PER_BLOCK, with a maximum of 8 threads per multiprocessor.
 */
#define LAUNCH_BOUNDS() __launch_bounds__(SM_6X_THREADS_PER_BLOCK, 8)

/**
 * @def LAUNCH_BOUNDS256
 * @brief Macro that sets the launch bounds to 256 threads with 5 threads per warp.
 *
 * This macro sets the launch bounds for CUDA kernels to 256 threads with 5 threads per warp.
 */
#define LAUNCH_BOUNDS256() __launch_bounds__(256, 5)

#elif (__CUDA_ARCH__ >= 500)
/**
 * @def LAUNCH_BOUNDS
 * @brief Macro that sets the launch bounds to a specified number of threads per block.
 *
 * This macro sets the launch bounds for CUDA kernels to a specified number of threads per block,
 * as defined by the constant SM_5X_THREADS_PER_BLOCK, with a maximum of 8 threads per multiprocessor.
 */
#define LAUNCH_BOUNDS() __launch_bounds__(SM_5X_THREADS_PER_BLOCK, 8)

/**
 * @def LAUNCH_BOUNDS256
 * @brief Macro that sets the launch bounds to 256 threads with 5 threads per warp.
 *
 * This macro sets the launch bounds for CUDA kernels to 256 threads with 5 threads per warp.
 */
#define LAUNCH_BOUNDS256() __launch_bounds__(256, 5)

#else
/**
 * @def LAUNCH_BOUNDS
 * @brief Macro that sets the launch bounds to a specified number of threads per block.
 *
 * This macro sets the launch bounds for CUDA kernels to a specified number of threads per block,
 * as defined by the constant SM_3X_THREADS_PER_BLOCK, with a maximum of 10 threads per multiprocessor.
 */
#define LAUNCH_BOUNDS() __launch_bounds__(SM_3X_THREADS_PER_BLOCK, 10)

/**
 * @def LAUNCH_BOUNDS256
 * @brief Macro that sets the launch bounds to 256 threads with 4 threads per warp.
 *
 * This macro sets the launch bounds for CUDA kernels to 256 threads with 4 threads per warp.
 */
#define LAUNCH_BOUNDS256() __launch_bounds__(256, 4)
#endif
/**
 * @def LAUNCH_BOUNDS512
 * @brief Macro that sets the launch bounds to 512 threads with 2 threads per warp.
 *
 * This macro sets the launch bounds for CUDA kernels to 512 threads with 2 threads per warp.
 */
#define LAUNCH_BOUNDS512() __launch_bounds__(512, 2)

/**
 * @def LAUNCH_BOUNDS1024
 * @brief Macro that sets the launch bounds to 1024 threads with 1 thread per warp.
 *
 * This macro sets the launch bounds for CUDA kernels to 1024 threads with 1 thread per warp.
 */
#define LAUNCH_BOUNDS1024() __launch_bounds__(1024, 1)

/**
 * @brief Maximum number of sparse elements for SM 6.x architecture.
 */
static const uint32_t SM_6X_MAXSPARSE = 4608;

/**
 * @brief Maximum number of sparse analog elements for SM 6.x architecture.
 */
static const uint32_t SM_6X_MAXSPARSEANALOG = 2304;

/**
 * @brief Maximum number of sparse elements for SM 5.x architecture.
 */
static const uint32_t SM_5X_MAXSPARSE = 4608;

/**
 * @brief Maximum number of sparse analog elements for SM 5.x architecture.
 */
static const uint32_t SM_5X_MAXSPARSEANALOG = 2304;

/**
 * @brief Maximum number of sparse elements for SM 3.x architecture.
 */
static const uint32_t SM_3X_MAXSPARSE = 2304;

/**
 * @brief Maximum number of sparse analog elements for SM 3.x architecture.
 */
static const uint32_t SM_3X_MAXSPARSEANALOG = 1152;

/**
 * @brief Flag indicating whether shadowed output buffers are enabled.
 */
static const bool bShadowedOutputBuffers = false;

/**
 * @def FPSCALE
 * @brief Macro that represents a scaling factor for floating-point values.
 *
 * This macro represents a scaling factor for floating-point values, defined as (1ll << 40).
 * The scaling factor is 2^40, which is commonly used for various numerical computations.
 */
#define FPSCALE  (1ll << 40)

/**
 * @def DFSCALE
 * @brief Macro that represents a scaling factor for double-precision floating-point values.
 *
 * This macro represents a scaling factor for double-precision floating-point values, defined as (1ll << 44).
 * The scaling factor is 2^44, which is commonly used for various numerical computations requiring higher precision.
 */
#define DFSCALE (1ll << 44)

#ifdef GVERBOSE
#ifndef MEMTRACKING
#define MEMTRACKING
#endif

#ifdef SYNCHRONOUS
/**
 * @def LAUNCHERROR(s)
 * @brief Macro that launches a CUDA kernel with error handling and synchronous device synchronization.
 *
 * This macro launches a CUDA kernel specified by the parameter `s`. If an error occurs during kernel launch,
 * it prints an error message and shuts down the GPU. It also performs synchronous device synchronization after the kernel.
 */
#define LAUNCHERROR(s) \
    { \
        std::cerr << "Launched " << s << " on node " << getGpu()._id << std::endl; \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            std::cerr << "Error: " << cudaGetErrorString(status) << " launching kernel " << s << std::endl; \
            getGpu().Shutdown(); \
            exit(-1); \
        } \
        cudaDeviceSynchronize(); \
    }
#else
/**
 * @def LAUNCHERROR(s)
 * @brief Macro that launches a CUDA kernel with error handling.
 *
 * This macro launches a CUDA kernel specified by the parameter `s`. If an error occurs during kernel launch,
 * it prints an error message and shuts down the GPU.
 */
#define LAUNCHERROR(s) \
    { \
        std::cerr << "Launched " << s << " on node " << getGpu()._id << std::endl; \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            std::cerr << "Error: " << cudaGetErrorString(status) << " launching kernel " << s << std::endl; \
            getGpu().Shutdown(); \
            exit(-1); \
        } \
    }
#endif

/**
 * @def LAUNCHERROR_BLOCKING(s)
 * @brief Macro that launches a CUDA kernel with error handling and synchronous device synchronization.
 *
 * This macro launches a CUDA kernel specified by the parameter `s`. If an error occurs during kernel launch,
 * it prints an error message and shuts down the GPU. It also performs synchronous device synchronization after the kernel.
 */
#define LAUNCHERROR_BLOCKING(s) \
    { \
        std::cerr << "Launched " << s << " on node " << getGpu()._id << std::endl; \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            std::cerr << "Error: " << cudaGetErrorString(status) << " launching kernel " << s << std::endl; \
            getGpu().Shutdown(); \
            exit(-1); \
        } \
        cudaDeviceSynchronize(); \
    }

/**
 * @def LAUNCHERROR_NONBLOCKING(s)
 * @brief Macro that launches a CUDA kernel with error handling.
 *
 * This macro launches a CUDA kernel specified by the parameter `s`. If an error occurs during kernel launch,
 * it prints an error message and shuts down the GPU.
 */
#define LAUNCHERROR_NONBLOCKING(s) \
    { \
        std::cerr << "Launched " << s << " on node " << getGpu()._id << std::endl; \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            std::cerr << "Error: " << cudaGetErrorString(status) << " launching kernel " << s << std::endl; \
            getGpu().Shutdown(); \
            exit(-1); \
        } \
    }

#else

#ifdef SYNCHRONOUS
/**
 * @def LAUNCHERROR(s)
 * @brief Macro that launches a CUDA kernel with error handling and synchronous device synchronization.
 *
 * This macro launches a CUDA kernel specified by the parameter `s`. If an error occurs during kernel launch,
 * it prints an error message and shuts down the GPU. It also performs synchronous device synchronization after the kernel.
 */
#define LAUNCHERROR(s) \
    { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            std::cerr << "Error: " << cudaGetErrorString(status) << " launching kernel " << s << std::endl; \
            getGpu().Shutdown(); \
            exit(-1); \
        } \
        cudaDeviceSynchronize(); \
    }
#else
/**
 * @def LAUNCHERROR(s)
 * @brief Macro that launches a CUDA kernel with error handling.
 *
 * This macro launches a CUDA kernel specified by the parameter `s`. If an error occurs during kernel launch,
 * it prints an error message and shuts down the GPU.
 */
#define LAUNCHERROR(s) \
    { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            std::cerr << "Error: " << cudaGetErrorString(status) << " launching kernel " << s << std::endl; \
            getGpu().Shutdown(); \
            exit(-1); \
        } \
    }
#endif

/**
 * @def LAUNCHERROR_BLOCKING(s)
 * @brief Macro that launches a CUDA kernel with error handling and synchronous device synchronization.
 *
 * This macro launches a CUDA kernel specified by the parameter `s`. If an error occurs during kernel launch,
 * it prints an error message and shuts down the GPU. It also performs synchronous device synchronization after the kernel.
 */
#define LAUNCHERROR_BLOCKING(s) \
    { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            std::cerr << "Error: " << cudaGetErrorString(status) << " launching kernel " << s << std::endl; \
            getGpu().Shutdown(); \
            exit(-1); \
        } \
        cudaDeviceSynchronize(); \
    }

/**
 * @def LAUNCHERROR_NONBLOCKING(s)
 * @brief Macro that launches a CUDA kernel with error handling.
 *
 * This macro launches a CUDA kernel specified by the parameter `s`. If an error occurs during kernel launch,
 * it prints an error message and shuts down the GPU.
 */
#define LAUNCHERROR_NONBLOCKING(s) \
    { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            std::cerr << "Error: " << cudaGetErrorString(status) << " launching kernel " << s << std::endl; \
            getGpu().Shutdown(); \
            exit(-1); \
        } \
    }

#endif

/**
 * @def RTERROR(status, s)
 * @brief Macro that checks and handles a CUDA runtime error.
 *
 * This macro checks the CUDA runtime `status` for an error. If an error is detected,
 * it prints an error message, asserts, exits the program, and performs necessary cleanup.
 * The error message includes the provided string `s` and the error description.
 */
#define RTERROR(status, s) \
    if (status != cudaSuccess) { \
        std::cerr << s << " " << cudaGetErrorString(status) << std::endl; \
        assert(0); \
        cudaThreadExit(); \
        exit(-1); \
    }

/**
 * @def CUDNNERROR(status, s)
 * @brief Macro that checks and handles a cuDNN error.
 *
 * This macro checks the cuDNN `status` for an error. If an error is detected,
 * it prints an error message, asserts, exits the program, and performs necessary cleanup.
 * The error message includes the provided string `s` and the error description.
 */
#define CUDNNERROR(status, s) \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << s << " " << cudnnGetErrorString(status) << std::endl; \
        assert(0); \
        cudaThreadExit(); \
        exit(-1); \
    }

/**
 * @struct GpuData
 * @brief Structure containing GPU data and parameters.
 */
struct GpuData {
    /**
     * @brief Size of a warp in threads.
     */
    unsigned int _warpSize;

    /**
     * @brief Number of bits used to represent a warp.
     */
    unsigned int _warpBits;

    /**
     * @brief Bit mask used to identify threads within a warp.
     */
    unsigned int _warpMask;

    /**
     * @brief Pointer to an array of unsigned long long integers representing accumulators.
     */
    unsigned long long int* _pAccumulator;

    /**
     * @brief Constant value for the k parameter in LRN (Local Response Normalization).
     */
    float _LRN_k;

    /**
     * @brief Value for the n parameter in LRN (Local Response Normalization).
     */
    int _LRN_n;

    /**
     * @brief Constant value for the alpha parameter in LRN (Local Response Normalization).
     */
    float _LRN_alpha;

    /**
     * @brief Constant value for the beta parameter in LRN (Local Response Normalization).
     */
    float _LRN_beta;

    /**
     * @brief Value for the k parameter in Maxout activation.
     */
    int _maxout_k;

    /**
     * @brief Constant value used for boosting in deltaBoost regularization.
     */
    float _deltaBoost_one;

    /**
     * @brief Constant value used for boosting in deltaBoost regularization.
     */
    float _deltaBoost_zero;

    /**
     * @brief Constant target value for calculating Smoothed Multiclass Cross-Entropy (SMCE) loss.
     */
    float _SMCE_oneTarget;

    /**
     * @brief Constant target value for calculating Smoothed Multiclass Cross-Entropy (SMCE) loss.
     */
    float _SMCE_zeroTarget;

    /**
     * @brief Scaling factor applied to the positive class in SMCE loss.
     */
    float _SMCE_oneScale;

    /**
     * @brief Scaling factor applied to the negative class in SMCE loss.
     */
    float _SMCE_zeroScale;

    /**
     * @brief Flag indicating whether sparseness penalty is enabled.
     */
    bool _bSparsenessPenalty;

    /**
     * @brief Probability used in sparseness penalty calculation.
     */
    float _sparsenessPenalty_p;

    /**
     * @brief Beta value used in sparseness penalty calculation.
     */
    float _sparsenessPenalty_beta;

    /**
     * @brief Flag indicating whether denoising is enabled.
     */
    bool _bDenoising;

    /**
     * @brief Probability used in denoising calculation.
     */
    float _denoising_p;

    /**
     * @brief Parameter used in denoising calculation.
     */
    float _denoising_q;

    /**
     * @brief Flag indicating whether shuffle indices are enabled.
     */
    bool _bShuffleIndices;

    /**
     * @brief Pointer to an array of unsigned integers representing shuffled indices.
     */
    unsigned int* _pShuffleIndex;

    /**
     * @brief Maximum value for the uint32_t data type.
     */
    uint32_t _maxUint32_t;

    /**
     * @brief Maximum value for the int32_t data type.
     */
    int32_t _maxInt32_t;

    /**
     * @brief Maximum value for the uint64_t data type.
     */
    uint64_t _maxUint64_t;

    /**
     * @brief Maximum value for the int64_t data type.
     */
    int64_t _maxInt64_t;

    /**
     * @brief Maximum representable value for the float data type.
     */
    float _maxFloat;

    /**
     * @brief Minimum representable positive value for the float data type.
     */
    float _minFloat;

};

/**
 * @class GpuBuffer
 * @brief Template class representing a GPU buffer.
 * @tparam T The type of data stored in the buffer.
 */
template <typename T> struct GpuBuffer;

/**
 * @class MultiGpuBuffer
 * @brief Template class representing a buffer that spans multiple GPUs.
 * @tparam T The type of data stored in the buffer.
 */
template <typename T> struct MultiGpuBuffer;

/**
 * @class Network
 * @brief Forward declaration of the Network class.
 */
class Network;

/**
 * @struct GpuContext
 * @brief Structure representing the GPU context.
 */
struct GpuContext {
    /**
     * @enum SM_VERSION
     * @brief Enumeration for different SM versions.
     */
    enum SM_VERSION
    {
        SM_3X,
        SM_5X,
        SM_6X,
    };

    enum {
        PADDING                     = 32,
        PADDINGBITS                 = 5,
        PADDINGMASK                 = 0xffffffff - (PADDING - 1),
    };

    /**
     * @brief GPU data container.
     */
    GpuData _data;

    /**
     * @brief Flag indicating whether ECC (Error-Correcting Code) support is available.
     */
    bool _bECCSupport;

    /**
     * @brief Flag indicating whether host memory can be mapped to the GPU.
     */
    bool _bCanMapHostMemory;

    /**
     * @brief Aligned total memory size.
     */
    aligned_lli _totalMemory;

    /**
     * @brief Aligned total CPU memory size.
     */
    aligned_lli _totalCPUMemory;

    /**
     * @brief Aligned total GPU memory size.
     */
    aligned_lli _totalGPUMemory;

    /**
     * @brief Flag indicating whether unified memory is supported.
     */
    bool _bUnifiedMemory;

    /**
     * @brief SM (Streaming Multiprocessor) version of the GPU.
     */
    SM_VERSION _sm_version;

    /**
     * @brief Major version number of the SM (Streaming Multiprocessor) architecture.
     */
    unsigned int _sm_major;

    /**
     * @brief Number of threads per block.
     */
    unsigned int _threadsPerBlock;

    /**
     * @brief Size of a warp in threads.
     */
    unsigned int _warpSize;

    /**
     * @brief Number of bits used to represent a warp.
     */
    unsigned int _warpBits;

    /**
     * @brief Bit mask used to identify threads within a warp.
     */
    unsigned int _warpMask;

    /**
     * @brief Number of processes.
     */
    int _numprocs;

    /**
     * @brief Process ID.
     */
    int _id;

    /**
     * @brief Device ID.
     */
    int _device;

    /**
     * @brief Maximum value for sparse storage.
     */
    uint32_t _maxSparse;

    /**
     * @brief Maximum value for sparse analog storage.
     */
    uint32_t _maxSparseAnalog;

    /**
     * @brief cuBLAS library handle.
     */
    cublasHandle_t _cuBLASHandle;

    /**
     * @brief cuRAND generator handle.
     */
    curandGenerator_t _RNG;

    /**
     * @brief cuDNN library handle.
     */
    cudnnHandle_t _cuDNNHandle;

    /**
     * @brief Pointer to the neural network object.
     */
    Network* _pNetwork;

    /**
     * @brief Unique pointer to a GPU buffer of unsigned long long integers representing accumulators.
     */
    std::unique_ptr<GpuBuffer<unsigned long long int>> _pbAccumulator;

    /**
     * @brief Flag indicating whether CPU validation is enabled.
     */
    bool _bCPUValidate;


    /**
     * @brief Initialize the GPU context.
     */
    void Init();

    /**
     * @brief Shutdown the GPU context.
     */
    void Shutdown();
};

#endif  // __GPUTYPES_H__
