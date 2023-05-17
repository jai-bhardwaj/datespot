#ifndef __GPUTYPES_H__
#define __GPUTYPES_H__
#include <stdio.h>
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
#include <builtin_types.h>
#include <cstring>
#include <cstdint>
#include <assert.h>
#include <mpi.h>
#include <memory>
#include <algorithm>
#include <string>

/**
 * @def VALIDATION
 * @brief Preprocessor macro for enabling validation.
 * 
 * This macro is used to enable validation in the code. By defining this macro,
 * the code will include validation functionality.
 */

#if defined(CUDA_VERSION) && (CUDA_VERSION < 5000)
/**
 * @def CUDA_VERSION
 * @brief Compilation error message for unsupported CUDA toolkit version.
 * 
 * This compilation error message is triggered when the CUDA toolkit version used is
 * older than version 5.0. The code requires the use of a CUDA toolkit version 5.0 or later.
 */
#error "CUDA support requires the use of a 5.0 or later CUDA toolkit. Aborting compilation."
#endif

/**
 * @def use_SPFP
 * @brief Preprocessor macro for single-precision floating-point precision mode.
 * 
 * This macro is used to specify the single-precision floating-point (SPFP) precision mode.
 * By defining this macro, the code will be compiled with single-precision floating-point precision.
 */

#if !(defined(use_DPFP) && !defined(use_HPFP) && !defined(use_SPFP)) && \
    !(defined(use_HPFP) && !defined(use_DPFP) && !defined(use_SPFP)) && \
    !(defined(use_SPFP) && !defined(use_DPFP) && !defined(use_HPFP))
/**
 * @def use_DPFP
 * @def use_HPFP
 * @def use_SPFP
 * @brief Compilation error message for undefined or conflicting precision modes.
 * 
 * This compilation error message is triggered when none or multiple precision mode macros are defined.
 * The user must define exactly one precision mode macro (use_DPFP, use_HPFP, or use_SPFP).
 */
#error "You must define one and only one precision mode (use_SPFP, use_HPFP, or use_DPFP). Aborting compilation."
#endif

/**
 * @brief Error scale value.
 * 
 * This constant defines the error scale value used in the code. It is set to (1ll << 30).
 */
#define ESCALE  (1ll << 30)

/**
 * @brief Double precision error scale value.
 * 
 * This constant defines the double-precision error scale value. It is set to the value of ESCALE.
 */
static const double ERRORSCALE = ESCALE;

/**
 * @brief Single precision error scale value.
 * 
 * This constant defines the single-precision error scale value. It is set to the value of ESCALE.
 */
static const float ERRORSCALEF = ESCALE;

/**
 * @brief One over error scale value (double precision).
 * 
 * This constant defines the reciprocal of the error scale value in double precision. It is computed as
 * 1.0 divided by the value of ERRORSCALE.
 */
static const double ONEOVERERRORSCALE = 1.0 / static_cast<double>(ERRORSCALE);

/**
 * @brief One over error scale value (single precision).
 * 
 * This constant defines the reciprocal of the error scale value in single precision. It is computed as
 * 1.0 divided by the value of ERRORSCALE casted to a double and then converted to a float.
 */
static const float ONEOVERERRORSCALEF = static_cast<float>(1.0 / static_cast<double>(ERRORSCALE));


/**
 * @brief 8-byte aligned double type with a typedef name.
 * 
 * This typedef defines the aligned_double type, which represents a double-precision value that is
 * aligned to 8 bytes using the `__align__(8)` attribute.
 */
typedef double __align__(8) aligned_double;

/**
 * @brief 8-byte aligned unsigned long int type with a typedef name.
 * 
 * This typedef defines the aligned_uli type, which represents an unsigned long integer value that is
 * aligned to 8 bytes using the `__align__(8)` attribute.
 */
typedef unsigned long int __align__(8) aligned_uli;

/**
 * @brief 8-byte aligned long long int type with a typedef name.
 * 
 * This typedef defines the aligned_lli type, which represents a long long integer value that is
 * aligned to 8 bytes using the `__align__(8)` attribute.
 */
typedef long long int __align__(8) aligned_lli;

/**
 * @brief 8-byte aligned unsigned long long int type with a typedef name.
 * 
 * This typedef defines the UllInt type, which represents an unsigned long long integer value that is
 * aligned to 8 bytes using the `__align__(8)` attribute.
 */
typedef unsigned long long int __align__(8) UllInt;


#if defined(use_DPFP)
/**
 * @brief 8-byte aligned double type for NN (nearest neighbor) accumulators.
 * 
 * This typedef defines the NNAccumulator type, which represents the double-precision type used for NN (nearest
 * neighbor) accumulators. It is aligned to 8 bytes using the `__align__(8)` attribute.
 */
typedef double __align__(8) NNAccumulator;

/**
 * @brief 8-byte aligned double type for NN (nearest neighbor) computations.
 * 
 * This typedef defines the NNDouble type, which represents the double-precision type used for NN (nearest
 * neighbor) computations. It is aligned to 8 bytes using the `__align__(8)` attribute.
 */
typedef double __align__(8) NNDouble;

/**
 * @brief 8-byte aligned double type for NN (nearest neighbor) single-precision values.
 * 
 * This typedef defines the NNFloat type, which represents the double-precision type used for NN (nearest
 * neighbor) single-precision values. It is aligned to 8 bytes using the `__align__(8)` attribute.
 */
typedef double __align__(8) NNFloat;

/**
 * @brief 16-byte aligned double2 type for NN (nearest neighbor) computations.
 * 
 * This typedef defines the NNDouble2 type, which represents a double-precision complex number
 * with 2 elements. The type is aligned to 16 bytes using the `__align__(16)` attribute.
 */
typedef double2 __align__(16) NNDouble2;

/**
 * @brief 32-byte aligned double4 type for NN (nearest neighbor) computations.
 * 
 * This typedef defines the NNDouble4 type, which represents a double-precision complex number
 * with 4 elements. The type is aligned to 32 bytes using the `__align__(32)` attribute.
 */
typedef double4 __align__(32) NNDouble4;

/**
 * @brief 16-byte aligned double2 type for NN (nearest neighbor) computations.
 * 
 * This typedef defines the NNFloat2 type, which represents a double-precision complex number
 * with 2 elements. The type is aligned to 16 bytes using the `__align__(16)` attribute.
 */
typedef double2 __align__(16) NNFloat2;

/**
 * @brief 16-byte aligned double4 type for NN (nearest neighbor) computations.
 * 
 * This typedef defines the NNFloat4 type, which represents a double-precision complex number
 * with 4 elements. The type is aligned to 16 bytes using the `__align__(16)` attribute.
 */
typedef double4 __align__(16) NNFloat4;

/**
 * @brief MPI datatype for NN (nearest neighbor) double-precision values.
 * 
 * This constant defines the MPI datatype for NN (nearest neighbor) double-precision values.
 * It is set to MPI_DOUBLE_PRECISION.
 */
static const MPI_Datatype MPI_NNDOUBLE = MPI_DOUBLE_PRECISION;

/**
 * @brief MPI datatype for NN (nearest neighbor) single-precision values.
 * 
 * This constant defines the MPI datatype for NN (nearest neighbor) single-precision values.
 * It is set to MPI_DOUBLE_PRECISION.
 */
static const MPI_Datatype MPI_NNFLOAT = MPI_DOUBLE_PRECISION;

/**
 * @brief MPI datatype for NN (nearest neighbor) accumulators.
 * 
 * This constant defines the MPI datatype for NN (nearest neighbor) accumulators.
 * It is set to MPI_FLOAT.
 */
static const MPI_Datatype MPI_NNACCUMULATOR = MPI_FLOAT;

#elif defined(use_SPFP)
/**
 * @brief Type for NN (nearest neighbor) accumulators.
 * 
 * This typedef defines the NNAccumulator type, which represents the type used for NN (nearest neighbor)
 * accumulators. It is set to float.
 */
typedef float NNAccumulator;

/**
 * @brief 8-byte aligned double type for NN (nearest neighbor) computations.
 * 
 * This typedef defines the NNDouble type, which represents the double-precision type used for NN (nearest
 * neighbor) computations. It is aligned to 8 bytes using the `__align__(8)` attribute.
 */
typedef double __align__(8) NNDouble;

/**
 * @brief Type for NN (nearest neighbor) single-precision values.
 * 
 * This typedef defines the NNFloat type, which represents the type used for NN (nearest neighbor) single-
 * precision values. It is set to float.
 */
typedef float NNFloat;

/**
 * @brief 16-byte aligned double2 type for NN (nearest neighbor) computations.
 * 
 * This typedef defines the NNDouble2 type, which represents a double-precision complex number
 * with 2 elements. The type is aligned to 16 bytes using the `__align__(16)` attribute.
 */
typedef double2 __align__(16) NNDouble2;

/**
 * @brief 32-byte aligned double4 type for NN (nearest neighbor) computations.
 * 
 * This typedef defines the NNDouble4 type, which represents a double-precision complex number
 * with 4 elements. The type is aligned to 32 bytes using the `__align__(32)` attribute.
 */
typedef double4 __align__(32) NNDouble4;

/**
 * @brief 8-byte aligned float2 type for NN (nearest neighbor) computations.
 * 
 * This typedef defines the NNFloat2 type, which represents a single-precision complex number
 * with 2 elements. The type is aligned to 8 bytes using the `__align__(8)` attribute.
 */
typedef float2 __align__(8) NNFloat2;

/**
 * @brief 16-byte aligned float4 type for NN (nearest neighbor) computations.
 * 
 * This typedef defines the NNFloat4 type, which represents a single-precision complex number
 * with 4 elements. The type is aligned to 16 bytes using the `__align__(16)` attribute.
 */
typedef float4 __align__(16) NNFloat4;
/**
 * @brief MPI datatype for NN (nearest neighbor) double-precision values.
 * 
 * This constant defines the MPI datatype for NN (nearest neighbor) double-precision values.
 * It is set to MPI_DOUBLE_PRECISION.
 */
static const MPI_Datatype MPI_NNDOUBLE = MPI_DOUBLE_PRECISION;

/**
 * @brief MPI datatype for NN (nearest neighbor) single-precision values.
 * 
 * This constant defines the MPI datatype for NN (nearest neighbor) single-precision values.
 * It is set to MPI_FLOAT.
 */
static const MPI_Datatype MPI_NNFLOAT = MPI_FLOAT;

/**
 * @brief MPI datatype for NN (nearest neighbor) accumulators.
 * 
 * This constant defines the MPI datatype for NN (nearest neighbor) accumulators.
 * It is set to MPI_LONG_LONG_INT.
 */
static const MPI_Datatype MPI_NNACCUMULATOR = MPI_LONG_LONG_INT;
#else
/**
 * @brief Type for NN (nearest neighbor) accumulators.
 * 
 * This typedef defines the NNAccumulator type, which represents the type used for NN (nearest neighbor)
 * accumulators. It is set to float.
 */
typedef float NNAccumulator;

/**
 * @brief 8-byte aligned double type for NN (nearest neighbor) computations.
 * 
 * This typedef defines the NNDouble type, which represents the double-precision type used for NN (nearest
 * neighbor) computations. It is aligned to 8 bytes using the `__align(8)__` attribute.
 */
typedef double __align(8)__ NNDouble;

/**
 * @brief Type for NN (nearest neighbor) single-precision values.
 * 
 * This typedef defines the NNFloat type, which represents the type used for NN (nearest neighbor) single-
 * precision values. It is set to float.
 */
typedef float NNFloat;

/**
 * @brief 16-byte aligned double2 type for NN (nearest neighbor) computations.
 * 
 * This typedef defines the NNDouble2 type, which represents a double-precision complex number
 * with 2 elements. The type is aligned to 16 bytes using the `__align(16)__` attribute.
 */
typedef double2 __align(16)__ NNDouble2;

/**
 * @brief 32-byte aligned double4 type for NN (nearest neighbor) computations.
 * 
 * This typedef defines the NNDouble4 type, which represents a double-precision complex number
 * with 4 elements. The type is aligned to 32 bytes using the `__align(32)__` attribute.
 */
typedef double4 __align(32)__ NNDouble4;

/**
 * @brief 8-byte aligned float2 type for NN (nearest neighbor) computations.
 * 
 * This typedef defines the NNFloat2 type, which represents a single-precision complex number
 * with 2 elements. The type is aligned to 8 bytes using the `__align(8)__` attribute.
 */
typedef float2 __align(8)__ NNFloat2;

/**
 * @brief 16-byte aligned float4 type for NN (nearest neighbor) computations.
 * 
 * This typedef defines the NNFloat4 type, which represents a single-precision complex number
 * with 4 elements. The type is aligned to 16 bytes using the `__align(16)__` attribute.
 */
typedef float4 __align(16)__ NNFloat4;

/**
 * @brief MPI datatype for NN double-precision values.
 * 
 * This constant defines the MPI datatype for NN (nearest neighbor) double-precision values.
 * It is set to MPI_DOUBLE_PRECISION.
 */
static const MPI_Datatype MPI_NNDOUBLE = MPI_DOUBLE_PRECISION;

/**
 * @brief MPI datatype for NN single-precision values.
 * 
 * This constant defines the MPI datatype for NN (nearest neighbor) single-precision values.
 * It is set to MPI_FLOAT.
 */
static const MPI_Datatype MPI_NNFLOAT = MPI_FLOAT;

/**
 * @brief MPI datatype for NN accumulators.
 * 
 * This constant defines the MPI datatype for NN (nearest neighbor) accumulators.
 * It is set to MPI_LONG_LONG_INT.
 */
static const MPI_Datatype MPI_NNACCUMULATOR = MPI_LONG_LONG_INT;
#endif

/**
 * @brief Number of threads per block supported on SM 3.x architecture.
 * 
 * This constant defines the number of threads per block supported on SM 3.x architecture.
 * It is set to 128.
 */
static const int SM_3X_THREADS_PER_BLOCK = 128;

/**
 * @brief Number of threads per block supported on SM 5.x architecture.
 * 
 * This constant defines the number of threads per block supported on SM 5.x architecture.
 * It is set to 128.
 */
static const int SM_5X_THREADS_PER_BLOCK = 128;

/**
 * @brief Number of threads per block supported on SM 6.x architecture.
 * 
 * This constant defines the number of threads per block supported on SM 6.x architecture.
 * It is set to 128.
 */
static const int SM_6X_THREADS_PER_BLOCK = 128;

#if (__CUDA_ARCH__ >= 600)
/**
 * @def LAUNCH_BOUNDS()
 * @brief Macro to specify launch bounds for a CUDA kernel with SM 6.x architecture.
 * 
 * This macro uses the `__launch_bounds__` attribute to specify the launch bounds for a CUDA kernel
 * targeting SM 6.x architecture. The launch bounds are set to `SM_6X_THREADS_PER_BLOCK` with a
 * recommended maximum dynamic shared memory per block of 8.
 */
#define LAUNCH_BOUNDS() __launch_bounds__(SM_6X_THREADS_PER_BLOCK, 8)

/**
 * @def LAUNCH_BOUNDS256()
 * @brief Macro to specify launch bounds for a CUDA kernel with a block size of 256.
 * 
 * This macro uses the `__launch_bounds__` attribute to specify the launch bounds for a CUDA kernel
 * with a block size of 256. The recommended maximum dynamic shared memory per block is set to 5.
 */
#define LAUNCH_BOUNDS256() __launch_bounds__(256, 5)
#elif (__CUDA_ARCH__ >= 500)
/**
 * @def LAUNCH_BOUNDS()
 * @brief Macro to specify launch bounds for a CUDA kernel with SM 5.x architecture.
 * 
 * This macro uses the `__launch_bounds__` attribute to specify the launch bounds for a CUDA kernel
 * targeting SM 5.x architecture. The launch bounds are set to `SM_5X_THREADS_PER_BLOCK` with a
 * recommended maximum dynamic shared memory per block of 8.
 */
#define LAUNCH_BOUNDS() __launch_bounds__(SM_5X_THREADS_PER_BLOCK, 8)

/**
 * @def LAUNCH_BOUNDS256()
 * @brief Macro to specify launch bounds for a CUDA kernel with a block size of 256.
 * 
 * This macro uses the `__launch_bounds__` attribute to specify the launch bounds for a CUDA kernel
 * with a block size of 256. The recommended maximum dynamic shared memory per block is set to 5.
 */
#define LAUNCH_BOUNDS256() __launch_bounds__(256, 5)
#else
/**
 * @def LAUNCH_BOUNDS()
 * @brief Macro to specify launch bounds for a CUDA kernel with SM 3.x architecture.
 * 
 * This macro uses the `__launch_bounds__` attribute to specify the launch bounds for a CUDA kernel
 * targeting SM 3.x architecture. The launch bounds are set to `SM_3X_THREADS_PER_BLOCK` with a
 * recommended maximum dynamic shared memory per block of 10.
 */
#define LAUNCH_BOUNDS() __launch_bounds__(SM_3X_THREADS_PER_BLOCK, 10)

/**
 * @def LAUNCH_BOUNDS256()
 * @brief Macro to specify launch bounds for a CUDA kernel with a block size of 256.
 * 
 * This macro uses the `__launch_bounds__` attribute to specify the launch bounds for a CUDA kernel
 * with a block size of 256. The recommended maximum dynamic shared memory per block is set to 4.
 */
#define LAUNCH_BOUNDS256() __launch_bounds__(256, 4)

#ifdef SYNCHRONOUS
/**
 * @def LAUNCH_BOUNDS512()
 * @brief Macro to specify launch bounds for a CUDA kernel with a block size of 512 (synchronous).
 * 
 * This macro uses the `__launch_bounds__` attribute to specify the launch bounds for a CUDA kernel
 * with a block size of 512 targeting SM architecture. The recommended maximum dynamic shared memory
 * per block is set to 2. This macro is available only when `SYNCHRONOUS` is defined.
 */
#define LAUNCH_BOUNDS512() __launch_bounds__(512, 2)
#else
/**
 * @def LAUNCH_BOUNDS512()
 * @brief Macro to specify launch bounds for a CUDA kernel with a block size of 512 (asynchronous).
 * 
 * This macro uses the `__launch_bounds__` attribute to specify the launch bounds for a CUDA kernel
 * with a block size of 512 targeting SM architecture. The recommended maximum dynamic shared memory
 * per block is set to 2. This macro is available only when `SYNCHRONOUS` is not defined.
 */
#define LAUNCH_BOUNDS512() __launch_bounds__(512, 2)
#endif

/**
 * @def LAUNCH_BOUNDS1024()
 * @brief Macro to specify launch bounds for a CUDA kernel with a block size of 1024.
 * 
 * This macro uses the `__launch_bounds__` attribute to specify the launch bounds for a CUDA kernel
 * with a block size of 1024. The recommended maximum dynamic shared memory per block is set to 1.
 */
#define LAUNCH_BOUNDS1024() __launch_bounds__(1024, 1)

/**
 * @brief Maximum number of sparse cores supported on SM 6.x architecture.
 * 
 * This constant defines the maximum number of sparse cores supported on SM 6.x architecture.
 * It is set to 4608.
 */
static const uint32_t SM_6X_MAXSPARSE = 4608;

/**
 * @brief Maximum number of sparse analog cores supported on SM 6.x architecture.
 * 
 * This constant defines the maximum number of sparse analog cores supported on SM 6.x architecture.
 * It is set to 2304.
 */
static const uint32_t SM_6X_MAXSPARSEANALOG = 2304;

/**
 * @brief Maximum number of sparse cores supported on SM 5.x architecture.
 * 
 * This constant defines the maximum number of sparse cores supported on SM 5.x architecture.
 * It is set to 4608.
 */
static const uint32_t SM_5X_MAXSPARSE = 4608;

/**
 * @brief Maximum number of sparse analog cores supported on SM 5.x architecture.
 * 
 * This constant defines the maximum number of sparse analog cores supported on SM 5.x architecture.
 * It is set to 2304.
 */
static const uint32_t SM_5X_MAXSPARSEANALOG = 2304;

/**
 * @brief Maximum number of sparse cores supported on SM 3.x architecture.
 * 
 * This constant defines the maximum number of sparse cores supported on SM 3.x architecture.
 * It is set to 2304.
 */
static const uint32_t SM_3X_MAXSPARSE = 2304;

/**
 * @brief Maximum number of sparse analog cores supported on SM 3.x architecture.
 * 
 * This constant defines the maximum number of sparse analog cores supported on SM 3.x architecture.
 * It is set to 1152.
 */
static const uint32_t SM_3X_MAXSPARSEANALOG = 1152;


/**
 * @brief Flag indicating whether output buffers are shadowed.
 * 
 * This flag specifies whether the output buffers are shadowed, indicating if there is an additional
 * copy of the output buffers. It is set to false, indicating that the output buffers are not shadowed.
 * 
 * @note Shadowing output buffers can provide additional safety or debugging capabilities but may
 *       incur additional memory overhead or performance impact.
 */
static const bool bShadowedOutputBuffers = false;

/**
 * @brief Floating-point scale value.
 * 
 * This constant defines the floating-point scale value. It is set to (1ll << 40).
 */
#define FPSCALE  (1ll << 40)

/**
 * @brief Double-precision floating-point scale value.
 * 
 * This constant defines the double-precision floating-point scale value. It is set to (1ll << 44).
 */
#define DFSCALE (1ll << 44)



#ifdef GVERBOSE
#ifndef MEMTRACKING
#define MEMTRACKING
#endif

#ifdef SYNCHRONOUS
/**
 * @def LAUNCHERROR(s)
 * @brief Macro to handle CUDA kernel launch errors synchronously.
 * 
 * This macro checks for errors that occurred during the launch of a CUDA kernel.
 * If an error is detected, it prints an error message along with the CUDA error string
 * and the name of the kernel. It also displays the name of the kernel and the ID of the
 * GPU node on which it was launched. It then shuts down the GPU, exits the program with
 * a non-zero status code, and synchronizes the device to ensure all pending operations
 * are completed before continuing.
 * 
 * @param s The name of the kernel that was launched.
 */
#define LAUNCHERROR(s) \
    { \
        printf("Launched %s on node %d\n", s, getGpu()._id); \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            getGpu().Shutdown(); \
            exit(-1); \
        } \
        cudaDeviceSynchronize(); \
    }
#else
/**
 * @def LAUNCHERROR(s)
 * @brief Macro to handle CUDA kernel launch errors non-blocking.
 * 
 * This macro checks for errors that occurred during the launch of a CUDA kernel.
 * If an error is detected, it prints an error message along with the CUDA error string
 * and the name of the kernel. It also displays the name of the kernel and the ID of the
 * GPU node on which it was launched. It then shuts down the GPU and exits the program
 * with a non-zero status code. Unlike the LAUNCHERROR macro with SYNCHRONOUS defined,
 * it does not synchronize the device or wait for pending operations to complete before
 * continuing.
 * 
 * @param s The name of the kernel that was launched.
 */
#define LAUNCHERROR(s) \
    { \
        printf("Launched %s on node %d\n", s, getGpu()._id); \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            getGpu().Shutdown(); \
            exit(-1); \
        } \
    }
#endif

/**
 * @def LAUNCHERROR_BLOCKING(s)
 * @brief Macro to handle CUDA kernel launch errors synchronously.
 * 
 * This macro checks for errors that occurred during the launch of a CUDA kernel.
 * If an error is detected, it prints an error message along with the CUDA error string
 * and the name of the kernel. It also displays the name of the kernel and the ID of the
 * GPU node on which it was launched. It then shuts down the GPU, exits the program with
 * a non-zero status code, and synchronizes the device to ensure all pending operations
 * are completed before continuing.
 * 
 * @param s The name of the kernel that was launched.
 */
#define LAUNCHERROR_BLOCKING(s) \
    { \
        printf("Launched %s on node %d\n", s, getGpu()._id); \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            getGpu().Shutdown(); \
            exit(-1); \
        } \
        cudaDeviceSynchronize(); \
    }

/**
 * @def LAUNCHERROR_NONBLOCKING(s)
 * @brief Macro to handle CUDA kernel launch errors non-blocking.
 * 
 * This macro checks for errors that occurred during the launch of a CUDA kernel.
 * If an error is detected, it prints an error message along with the CUDA error string
 * and the name of the kernel. It also displays the name of the kernel and the ID of the
 * GPU node on which it was launched. It then shuts down the GPU and exits the program
 * with a non-zero status code. Unlike the LAUNCHERROR_BLOCKING macro, it does not
 * synchronize the device or wait for pending operations to complete before continuing.
 * 
 * @param s The name of the kernel that was launched.
 */
#define LAUNCHERROR_NONBLOCKING(s) \
    { \
        printf("Launched %s on node %d\n", s, getGpu()._id); \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            getGpu().Shutdown(); \
            exit(-1); \
        } \
    }

#else

#ifdef SYNCHRONOUS
/**
 * @def LAUNCHERROR(s)
 * @brief Macro to handle CUDA kernel launch errors synchronously.
 * 
 * This macro checks for errors that occurred during the launch of a CUDA kernel.
 * If an error is detected, it prints an error message along with the CUDA error string
 * and the name of the kernel. It then shuts down the GPU, exits the program with a
 * non-zero status code, and synchronizes the device to ensure all pending operations
 * are completed before continuing.
 * 
 * @param s The name of the kernel that was launched.
 */
#define LAUNCHERROR(s) \
    { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            getGpu().Shutdown(); \
            exit(-1); \
        } \
        cudaDeviceSynchronize(); \
    }
#else
/**
 * @def LAUNCHERROR(s)
 * @brief Macro to handle CUDA kernel launch errors non-blocking.
 * 
 * This macro checks for errors that occurred during the launch of a CUDA kernel.
 * If an error is detected, it prints an error message along with the CUDA error string
 * and the name of the kernel. It then shuts down the GPU and exits the program with a
 * non-zero status code. Unlike the LAUNCHERROR macro with SYNCHRONOUS defined, it does not synchronize
 * the device or wait for pending operations to complete before continuing.
 * 
 * @param s The name of the kernel that was launched.
 */
#define LAUNCHERROR(s) \
    { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            getGpu().Shutdown(); \
            exit(-1); \
        } \
    }
#endif

/**
 * @def LAUNCHERROR_BLOCKING(s)
 * @brief Macro to handle CUDA kernel launch errors synchronously.
 * 
 * This macro checks for errors that occurred during the launch of a CUDA kernel.
 * If an error is detected, it prints an error message along with the CUDA error string
 * and the name of the kernel. It then shuts down the GPU, exits the program with a
 * non-zero status code, and synchronizes the device to ensure all pending operations
 * are completed before continuing.
 * 
 * @param s The name of the kernel that was launched.
 */
#define LAUNCHERROR_BLOCKING(s) \
    { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            getGpu().Shutdown(); \
            exit(-1); \
        } \
        cudaDeviceSynchronize(); \
    }

/**
 * @def LAUNCHERROR_NONBLOCKING(s)
 * @brief Macro to handle CUDA kernel launch errors non-blocking.
 * 
 * This macro checks for errors that occurred during the launch of a CUDA kernel.
 * If an error is detected, it prints an error message along with the CUDA error string
 * and the name of the kernel. It then shuts down the GPU and exits the program with a
 * non-zero status code. Unlike the LAUNCHERROR_BLOCKING macro, it does not synchronize
 * the device, allowing for asynchronous operations to continue.
 * 
 * @param s The name of the kernel that was launched.
 */
#define LAUNCHERROR_NONBLOCKING(s) \
    { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            getGpu().Shutdown(); \
            exit(-1); \
        } \
    }

#endif

/**
 * @def RTERROR(status, s)
 * @brief Macro to handle CUDA runtime errors.
 * 
 * This macro checks the status of a CUDA function call and prints an error message
 * along with the CUDA error string if the status is not cudaSuccess. It also asserts
 * and terminates the program if an error occurs, performs device reset, and exits with
 * a non-zero status code.
 * 
 * @param status The status returned by the CUDA function call.
 * @param s The additional error message to be printed.
 */
#define RTERROR(status, s) \
    if (status != cudaSuccess) { \
        printf("%s %s\n", s, cudaGetErrorString(status)); \
        assert(0); \
        cudaDeviceReset(); \
        exit(-1); \
    }

/**
 * @def CUDNNERROR(status, s)
 * @brief Macro to handle cuDNN errors.
 * 
 * This macro checks the status of a cuDNN function call and prints an error message
 * along with the cuDNN error string if the status is not CUDNN_STATUS_SUCCESS. It also
 * asserts and terminates the program if an error occurs, performs device reset, and
 * exits with a non-zero status code.
 * 
 * @param status The status returned by the cuDNN function call.
 * @param s The additional error message to be printed.
 */
#define CUDNNERROR(status, s) \
    if (status != CUDNN_STATUS_SUCCESS) { \
        printf("%s %s\n", s, cudnnGetErrorString(status)); \
        assert(0); \
        cudaDeviceReset(); \
        exit(-1); \
    }

/**
 * @brief GpuData struct holds GPU data information.
 */
struct GpuData {
    /**
     * @brief The size of a warp in the GPU.
     */
    unsigned int _warpSize;

    /**
     * @brief The number of bits in a warp.
     */
    unsigned int _warpBits;

    /**
     * @brief The mask to extract the warp ID from a thread ID.
     */
    unsigned int _warpMask;

    /**
     * @brief Pointer to the accumulator for GPU operations.
     */
    unsigned long long int* _pAccumulator;

    /**
     * @brief The k parameter for Local Response Normalization (LRN).
     */
    float _LRN_k;

    /**
     * @brief The n parameter for Local Response Normalization (LRN).
     */
    int _LRN_n;

    /**
     * @brief The alpha parameter for Local Response Normalization (LRN).
     */
    float _LRN_alpha;

    /**
     * @brief The beta parameter for Local Response Normalization (LRN).
     */
    float _LRN_beta;

    /**
     * @brief The k parameter for the Maxout activation function.
     */
    int _maxout_k;

    /**
     * @brief The delta boost value for target value of 1 in certain loss functions.
     */
    float _deltaBoost_one;

    /**
     * @brief The delta boost value for target value of 0 in certain loss functions.
     */
    float _deltaBoost_zero;

    /**
     * @brief The value representing the target value of 1 in the Softmax Cross-Entropy (SMCE) loss function.
     */
    float _SMCE_oneTarget;

    /**
     * @brief The value representing the target value of 0 in the Softmax Cross-Entropy (SMCE) loss function.
     */
    float _SMCE_zeroTarget;

    /**
     * @brief The scale factor for the target value of 1 in the Softmax Cross-Entropy (SMCE) loss function.
     */
    float _SMCE_oneScale;

    /**
     * @brief The scale factor for the target value of 0 in the Softmax Cross-Entropy (SMCE) loss function.
     */
    float _SMCE_zeroScale;

    /**
     * @brief Flag indicating whether sparseness penalty is enabled.
     */
    bool _bSparsenessPenalty;

    /**
     * @brief The p parameter used for sparseness penalty.
     */
    float _sparsenessPenalty_p;

    /**
     * @brief The beta parameter used for sparseness penalty.
     */
    float _sparsenessPenalty_beta;

    /**
     * @brief Flag indicating whether denoising is enabled.
     */
    bool _bDenoising;

    /**
     * @brief The p parameter used for denoising.
     */
    float _denoising_p;

    /**
     * @brief The q parameter used for denoising.
     */
    float _denoising_q;

    /**
     * @brief Flag indicating whether shuffle indices are enabled.
     */
    bool _bShuffleIndices;

    /**
     * @brief Pointer to the shuffle indices.
     */
    unsigned int* _pShuffleIndex;

    /**
     * @brief Maximum value for uint32_t.
     */
    uint32_t _maxUint32_t;

    /**
     * @brief Maximum value for int32_t.
     */
    int32_t _maxInt32_t;

    /**
     * @brief Maximum value for uint64_t.
     */
    uint64_t _maxUint64_t;

    /**
     * @brief Maximum value for int64_t.
     */
    int64_t _maxInt64_t;

    /**
     * @brief Maximum value for float.
     */
    float _maxFloat;

    /**
     * @brief Minimum value for float.
     */
    float _minFloat;
};

/**
 * @brief Forward declaration of the GpuBuffer structure template.
 * @tparam T The data type of the buffer.
 */
template <typename T> struct GpuBuffer;

/**
 * @brief Forward declaration of the MultiGpuBuffer structure template.
 * @tparam T The data type of the buffer.
 */
template <typename T> struct MultiGpuBuffer;

/**
 * @brief Forward declaration of the Network class.
 */
class Network;

/**
 * @brief GpuContext struct represents a GPU context for managing GPU operations.
 */
struct GpuContext {
    /**
     * @brief Enumeration of SM (Streaming Multiprocessor) versions.
     */
    enum SM_VERSION
    {
        SM_3X,  // SM 3.x version.
        SM_5X,  // SM 5.x version.
        SM_6X,  // SM 6.x version.
    };

    /**
     * @brief Padding information for alignment.
     */
    enum {
        PADDING             = 32,                                // Padding value.
        PADDINGBITS         = 5,                                 // Number of padding bits.
        PADDINGMASK         = 0xffffffff - (PADDING - 1),        // Mask for padding.
    };

    /**
     * @brief Structure holding GPU data information.
     */
    GpuData _data;

    /**
     * @brief Flag indicating whether the GPU supports ECC (Error-Correcting Code) memory.
     */
    bool _bECCSupport;

    /**
     * @brief Flag indicating whether the GPU can map host memory.
     */
    bool _bCanMapHostMemory;

    /**
     * @brief Aligned total memory available in the GPU.
     */
    aligned_lli _totalMemory;

    /**
     * @brief Aligned total CPU memory used.
     */
    aligned_lli _totalCPUMemory;

    /**
     * @brief Aligned total GPU memory used.
     */
    aligned_lli _totalGPUMemory;

    /**
     * @brief Flag indicating whether the GPU uses unified memory.
     */
    bool _bUnifiedMemory;

    /**
     * @brief Structure holding the SM (Streaming Multiprocessor) version information.
     */
    SM_VERSION _sm_version;

    /**
     * @brief The major version number of the SM architecture.
     */
    unsigned int _sm_major;

    /**
     * @brief The number of threads per block in the GPU.
     */
    unsigned int _threadsPerBlock;

    /**
     * @brief The size of a warp in the GPU.
     */
    unsigned int _warpSize;

    /**
     * @brief The number of bits in a warp.
     */
    unsigned int _warpBits;

    /**
     * @brief The mask to extract the warp ID from a thread ID.
     */
    unsigned int _warpMask;

    /**
     * @brief The total number of processes.
     */
    int _numprocs;

    /**
     * @brief The ID of the current process.
     */
    int _id;

    /**
     * @brief The device ID of the GPU context.
     */
    int _device;

    /**
     * @brief Maximum value for sparse operations.
     */
    uint32_t _maxSparse;

    /**
     * @brief Maximum value for analog sparse operations.
     */
    uint32_t _maxSparseAnalog;

    /**
     * @brief cuBLAS handle for GPU operations.
     */
    cublasHandle_t _cuBLASHandle;

    /**
     * @brief cuRAND generator for random number generation.
     */
    curandGenerator_t _RNG;

    /**
     * @brief cuDNN handle for deep neural network operations.
     */
    cudnnHandle_t _cuDNNHandle;

    /**
     * @brief Pointer to the neural network associated with the GPU context.
     */
    Network* _pNetwork;

    /**
     * @brief Unique pointer to the accumulator GPU buffer.
     */
    std::unique_ptr<GpuBuffer<unsigned long long int>> _pbAccumulator;

    /**
     * @brief Flag indicating whether CPU validation is enabled.
     */
    bool _bCPUValidate;

    /**
     * @brief Acceptable error threshold for CPU validation.
     */
    float _acceptableError;

    /**
     * @brief Flag indicating whether the GPU context is configured for a single node.
     */
    bool _bSingleNode;

    /**
     * @brief Flag indicating whether Peer-to-Peer communication is enabled.
     */
    bool _bP2P;

    /**
     * @brief Constructs a GpuContext object.
     */
    GpuContext();

    /**
     * @brief Destroys the GpuContext object.
     */
    ~GpuContext();

    /**
     * @brief Retrieves the GPU and CPU memory usage.
     * @param[out] gpuMemory Pointer to store the GPU memory usage.
     * @param[out] cpuMemory Pointer to store the CPU memory usage.
     */
    void GetMemoryUsage(int* gpuMemory, int* cpuMemory);

    /**
     * @brief Sets the random seed for GPU operations.
     * @param seed The random seed value to set.
     */
    void SetRandomSeed(unsigned long seed);

    /**
     * @brief Sets the neural network associated with the GPU context.
     * @param pNetwork Pointer to the neural network object.
     */
    void SetNeuralNetwork(Network* pNetwork);

    /**
     * @brief Sets the fast math flag for GPU operations.
     * @param flag The fast math flag value to set.
     */
    void SetFastMath(bool flag);

    /**
     * @brief Initializes the GPU context.
     * @param argc The number of command-line arguments.
     * @param argv The array of command-line arguments.
     */
    void Startup(int argc, char** argv);

    /**
     * @brief Shuts down the GPU context and releases associated resources.
     */
    void Shutdown();

    /**
     * @brief Copies constants to the GPU.
     */
    void CopyConstants();

    /**
     * @brief Sets the CPU validation flag for GPU operations.
     * @param bCPUValidate The CPU validation flag value to set.
     */
    void SetCPUValidate(bool bCPUValidate);

    /**
     * @brief Pads the input value to the nearest multiple of PADDING.
     * @param x The input value to pad.
     * @return The padded value.
     */
    static unsigned int Pad(unsigned int x) { return (x + PADDING - 1) & PADDINGMASK; }
};

extern struct GpuContext& getGpu();

/**
 * @brief A generic GPU buffer structure.
 * @tparam T The data type of the buffer.
 */
template <typename T>
struct GpuBuffer
{
    size_t      _length;        // The length of the buffer.
    bool        _bSysMem;       // A flag indicating whether system memory is allocated.
    bool        _bManaged;      // A flag indicating whether managed memory is allocated.
    T*          _pSysData;      // Pointer to system memory data.
    T*          _pDevData;      // Pointer to device memory data.

    /**
     * @brief Constructs a GpuBuffer object with the specified length, system memory flag, and managed flag.
     * @param length The length of the buffer.
     * @param bSysMem A boolean flag indicating whether to allocate system memory. Default is false.
     * @param bManaged A boolean flag indicating whether to allocate managed memory. Default is false.
     */
    GpuBuffer(int length, bool bSysMem = false, bool bManaged = false);

    /**
     * @brief Constructs a GpuBuffer object with the specified length, system memory flag, and managed flag.
     * @param length The length of the buffer.
     * @param bSysMem A boolean flag indicating whether to allocate system memory. Default is false.
     * @param bManaged A boolean flag indicating whether to allocate managed memory. Default is false.
     */
    GpuBuffer(unsigned int length, bool bSysMem = false, bool bManaged = false);

    /**
     * @brief Constructs a GpuBuffer object with the specified length, system memory flag, and managed flag.
     * @param length The length of the buffer.
     * @param bSysMem A boolean flag indicating whether to allocate system memory. Default is false.
     * @param bManaged A boolean flag indicating whether to allocate managed memory. Default is false.
     */
    GpuBuffer(unsigned long long int length, bool bSysMem = false, bool bManaged = false);

    /**
     * @brief Constructs a GpuBuffer object with the specified length, system memory flag, and managed flag.
     * @param length The length of the buffer.
     * @param bSysMem A boolean flag indicating whether to allocate system memory. Default is false.
     * @param bManaged A boolean flag indicating whether to allocate managed memory. Default is false.
     */
    GpuBuffer(size_t length, bool bSysMem = false, bool bManaged = false);

    /**
     * @brief Destroys the GpuBuffer object and frees the allocated memory.
     */
    virtual ~GpuBuffer();

    /**
     * @brief Allocates GPU memory for the GpuBuffer.
     */
    void Allocate();

    /**
     * @brief Resizes the GpuBuffer to the specified length.
     * @param length The new length of the buffer.
     */
    void Resize(size_t length);

    /**
     * @brief Deallocates the GPU and system memory associated with the GpuBuffer.
     */
    void Deallocate();

    /**
     * @brief Uploads data from the specified buffer to the device memory.
     * @param pBuff Pointer to the source buffer. If NULL, system memory data is used.
     */
    void Upload(const T* pBuff = NULL) const;

    /**
     * @brief Downloads data from the device memory to the specified buffer.
     * @param pBuff Pointer to the destination buffer. If NULL, system memory data is used.
     */
    void Download(T* pBuff = NULL);

    /**
     * @brief Copies data from the specified buffer to the GpuBuffer.
     * @param pBuff Pointer to the source buffer.
     */
    void Copy(T* pBuff);

    /**
     * @brief Returns the length of the GpuBuffer.
     * @return The length of the buffer.
     */
    size_t GetLength();

    /**
     * @brief Returns the size of the GpuBuffer in bytes.
     * @return The size of the buffer.
     */
    size_t GetSize();
};

/**
 * @brief Constructs a GpuBuffer object with the specified length, system memory flag, and managed flag.
 * @tparam T The data type of the buffer.
 * @param length The length of the buffer.
 * @param bSysMem A boolean flag indicating whether to allocate system memory.
 * @param bManaged A boolean flag indicating whether to allocate managed memory.
 */
template <typename T>
GpuBuffer<T>::GpuBuffer(int length, bool bSysMem, bool bManaged) : _length(length), _bSysMem(bSysMem), _bManaged(bManaged), _pSysData(NULL), _pDevData(NULL)
{
    Allocate();
}

/**
 * @brief Constructs a GpuBuffer object with the specified length, system memory flag, and managed flag.
 * @tparam T The data type of the buffer.
 * @param length The length of the buffer.
 * @param bSysMem A boolean flag indicating whether to allocate system memory.
 * @param bManaged A boolean flag indicating whether to allocate managed memory.
 */
template <typename T>
GpuBuffer<T>::GpuBuffer(unsigned int length, bool bSysMem, bool bManaged) : _length(length), _bSysMem(bSysMem), _bManaged(bManaged), _pSysData(NULL), _pDevData(NULL)
{
    Allocate();
}

/**
 * @brief Constructs a GpuBuffer object with the specified length, system memory flag, and managed flag.
 * @tparam T The data type of the buffer.
 * @param length The length of the buffer.
 * @param bSysMem A boolean flag indicating whether to allocate system memory.
 * @param bManaged A boolean flag indicating whether to allocate managed memory.
 */
template <typename T>
GpuBuffer<T>::GpuBuffer(unsigned long long int length, bool bSysMem, bool bManaged) : _length(length), _bSysMem(bSysMem), _bManaged(bManaged), _pSysData(NULL), _pDevData(NULL)
{
    Allocate();
}

/**
 * @brief Constructs a GpuBuffer object with the specified length, system memory flag, and managed flag.
 * @tparam T The data type of the buffer.
 * @param length The length of the buffer.
 * @param bSysMem A boolean flag indicating whether to allocate system memory.
 * @param bManaged A boolean flag indicating whether to allocate managed memory.
 */
template <typename T>
GpuBuffer<T>::GpuBuffer(size_t length, bool bSysMem, bool bManaged) : _length(length), _bSysMem(bSysMem), _bManaged(bManaged), _pSysData(NULL), _pDevData(NULL)
{
    Allocate();
}

/**
 * @brief Destroys the GpuBuffer object and frees the allocated memory.
 * @tparam T The data type of the buffer.
 */
template <typename T>
GpuBuffer<T>::~GpuBuffer()
{
    Deallocate();
}

/**
 * @brief Allocates GPU memory for the GpuBuffer.
 * @tparam T The data type of the buffer.
 */
template <typename T>
void GpuBuffer<T>::Allocate()
{
    cudaError_t status;

    if (_bManaged)
        _bSysMem = true;

#ifdef MEMTRACKING
    printf("Allocating %llu bytes of GPU memory", _length * sizeof(T));
    if (!_bSysMem)
    {
        printf(", unshadowed");
    }
    else if (_bManaged)
    {
        printf(", managed");
    }
    printf("\n");
#endif

    if (_bManaged)
    {
        status = cudaMallocManaged((void**)&_pDevData, _length * sizeof(T), cudaMemAttachGlobal);
        getGpu()._totalGPUMemory += _length * sizeof(T);
        _pSysData = _pDevData;
        RTERROR(status, "GpuBuffer::Allocate failed (cudaMallocManaged)");
        memset(_pSysData, 0, _length * sizeof(T));
    }
    else
    {
        status = cudaMalloc((void**)&_pDevData, _length * sizeof(T));
        getGpu()._totalGPUMemory += _length * sizeof(T);
        RTERROR(status, "GpuBuffer::Allocate failed (cudaMalloc)");
        status = cudaMemset((void*)_pDevData, 0, _length * sizeof(T));
        RTERROR(status, "GpuBuffer::Allocate failed (cudaMemset)");

        if (_bSysMem)
        {
            _pSysData = new T[_length];
            getGpu()._totalCPUMemory += _length * sizeof(T);
            memset(_pSysData, 0, _length * sizeof(T));
        }
    }

#ifdef MEMTRACKING
    printf("Mem++: %llu %llu\n", getGpu()._totalGPUMemory, getGpu()._totalCPUMemory);
#endif
}


/**
 * @brief Resizes the GPU buffer to the specified length.
 *
 * @tparam T Type of elements stored in the buffer.
 * @param length The new length of the buffer.
 */
template<typename T>
void GpuBuffer<T>::Resize(size_t length)
{
    if(length > _length)
    {
        Deallocate();
        _length = length;
        Allocate();
    }
}

/**
 * @brief Deallocates the GPU buffer and frees associated memory.
 *
 * @tparam T Type of elements stored in the buffer.
 */
template <typename T>
void GpuBuffer<T>::Deallocate()
{
    cudaError_t status;

    status = cudaFree(_pDevData);
    RTERROR(status, "GpuBuffer::Deallocate failed (cudaFree)");
    getGpu()._totalGPUMemory           -=  _length * sizeof(T);

    if (_bSysMem && !_bManaged)
    {
        delete[] _pSysData;
        getGpu()._totalCPUMemory           -=  _length * sizeof(T);
    }

    _pSysData = NULL;
    _pDevData = NULL;
    _length = 0;
#ifdef MEMTRACKING
    printf("Mem--: %lld %lld\n", getGpu()._totalGPUMemory, getGpu()._totalCPUMemory);
#endif
}

/**
 * @brief Copies data from the given buffer to the GPU buffer on the same device.
 *
 * @tparam T Type of elements stored in the buffer.
 * @param pBuff Pointer to the source buffer containing the data to be copied.
 */
template <typename T>
void GpuBuffer<T>::Copy(T* pBuff)
{
    cudaError_t status;
    status = cudaMemcpy(_pDevData, pBuff, _length * sizeof(T), cudaMemcpyDeviceToDevice);
    RTERROR(status, "cudaMemcpy GpuBuffer::Copy failed");
}

/**
 * @brief Uploads data from the host memory to the GPU buffer.
 *
 * @tparam T Type of elements stored in the buffer.
 * @param pBuff Pointer to the source host buffer containing the data to be uploaded.
 */
template <typename T>
void GpuBuffer<T>::Upload(const T* pBuff) const
{
    if (pBuff)
    {
        cudaError_t status;
        status = cudaMemcpy(_pDevData, pBuff, _length * sizeof(T), cudaMemcpyHostToDevice);
        RTERROR(status, "cudaMemcpy GpuBuffer::Upload failed");
    }
    else if (_bSysMem && !_bManaged)
    {
        cudaError_t status;
        status = cudaMemcpy(_pDevData, _pSysData, _length * sizeof(T), cudaMemcpyHostToDevice);
        RTERROR(status, "cudaMemcpy GpuBuffer::Upload failed");
    }
}

/**
 * @brief Downloads the contents of the GPU buffer to the host memory.
 *
 * @tparam T Type of elements stored in the buffer.
 * @param pBuff Pointer to the host memory where the data will be downloaded.
 *              If null, the data will be downloaded to the system memory of the buffer.
 */
template <typename T>
void GpuBuffer<T>::Download(T* pBuff)
{
    if (pBuff)
    {
        cudaError_t status;
        status = cudaMemcpy(pBuff, _pDevData, _length * sizeof(T), cudaMemcpyDeviceToHost);
        RTERROR(status, "cudaMemcpy GpuBuffer::Download failed");
    }
    else if (_bSysMem && !_bManaged)
    {
        cudaError_t status;
        status = cudaMemcpy(_pSysData, _pDevData, _length * sizeof(T), cudaMemcpyDeviceToHost);
        RTERROR(status, "cudaMemcpy GpuBuffer::Download failed");
    }
}

/**
 * @brief Gets the length of the GPU buffer.
 *
 * @tparam T Type of elements stored in the buffer.
 * @return The length of the GPU buffer.
 */
template<typename T>
size_t GpuBuffer<T>::GetLength()
{
    return _length;
}

/**
 * @brief Gets the size of the GPU buffer in bytes.
 *
 * @tparam T Type of elements stored in the buffer.
 * @return The size of the GPU buffer in bytes.
 */
template<typename T>
size_t GpuBuffer<T>::GetSize()
{
    return _length * sizeof(T);
}

/**
 * @brief Verifies the result of the SGEMM operation on the GPU.
 *
 * @param pbA Pointer to the GPU buffer containing matrix A.
 * @param pbB Pointer to the GPU buffer containing matrix B.
 * @param pbC Pointer to the GPU buffer containing matrix C (result).
 * @param m Number of rows in matrices A and C.
 * @param k Number of columns in matrix A and rows in matrix B.
 * @param n Number of columns in matrices B and C.
 */
void verifySGEMM(GpuBuffer<NNFloat>* pbA, GpuBuffer<NNFloat>* pbB, GpuBuffer<NNFloat>* pbC, uint32_t m, uint32_t k, uint32_t n);

/**
 * @brief Verifies the result of the SGEMMNT operation on the GPU.
 *
 * @param pbA Pointer to the GPU buffer containing matrix A.
 * @param pbB Pointer to the GPU buffer containing matrix B.
 * @param pbC Pointer to the GPU buffer containing matrix C (result).
 * @param m Number of rows in matrices A and C.
 * @param k Number of columns in matrix A and rows in matrix B.
 * @param n Number of columns in matrices B and C.
 */
void verifySGEMMNT(GpuBuffer<NNFloat>* pbA, GpuBuffer<NNFloat>* pbB, GpuBuffer<NNFloat>* pbC, uint32_t m, uint32_t k, uint32_t n);

/**
 * @brief Verifies the result of the SGEMMTN operation on the GPU.
 *
 * @param pbA Pointer to the GPU buffer containing matrix A.
 * @param pbB Pointer to the GPU buffer containing matrix B.
 * @param pbC Pointer to the GPU buffer containing matrix C (result).
 * @param m Number of rows in matrices A and C.
 * @param k Number of columns in matrix A and rows in matrix B.
 * @param n Number of columns in matrices B and C.
 */
void verifySGEMMTN(GpuBuffer<NNFloat>* pbA, GpuBuffer<NNFloat>* pbB, GpuBuffer<NNFloat>* pbC, uint32_t m, uint32_t k, uint32_t n);

/**
 * @def SGEMM
 * @brief Macro for performing the SGEMM operation using cuBLAS.
 *
 * @param A Pointer to matrix A on the GPU.
 * @param B Pointer to matrix B on the GPU.
 * @param C Pointer to matrix C (result) on the GPU.
 * @param m Number of rows in matrices A and C.
 * @param n Number of columns in matrices B and C.
 * @param k Number of columns in matrix A and rows in matrix B.
 * @param alpha Scalar used for multiplication.
 * @param beta Scalar used for multiplication of matrix C.
 * @param transf_A Whether matrix A should be transposed.
 * @param transf_B Whether matrix B should be transposed.
 */
#define SGEMM(A,B,C,m,n,k,alpha,beta,transf_A,transf_B) \
        cublasSgemm(getGpu()._cuBLASHandle, transf_B, transf_A, n, m, k, alpha, B, n, A, k, beta, C, n)

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
/**
 * @def printf
 * @brief Macro for suppressing printf calls on architectures below CUDA 2.0.
 *
 * @param f Format string for printf.
 * @param ... Optional arguments for printf.
 */
#define printf
#endif
#endif