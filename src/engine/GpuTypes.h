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

#define VALIDATION

#if defined(CUDA_VERSION) && (CUDA_VERSION < 5000)
#error "CUDA support requires the use of a 5.0 or later CUDA toolkit. Aborting compilation."
#endif

#define use_SPFP

#if !(defined(use_DPFP) && !defined(use_HPFP) && !defined(use_SPFP)) && \
    !(defined(use_HPFP) && !defined(use_DPFP) && !defined(use_SPFP)) && \
    !(defined(use_SPFP) && !defined(use_DPFP) && !defined(use_HPFP))
#error "You must define one and only one precision mode (use_SPFP, use_HPFP, or use_DPFP). Aborting compilation."
#endif

#define ESCALE  (1ll << 30)
static const double ERRORSCALE              = ESCALE;
static const float ERRORSCALEF              = ESCALE;
static const double ONEOVERERRORSCALE       = 1.0 / static_cast<double>(ERRORSCALE);
static const float ONEOVERERRORSCALEF       = static_cast<float>(1.0 / static_cast<double>(ERRORSCALE));

typedef double                  __align__(8)    aligned_double;
typedef unsigned long int       __align__(8)    aligned_uli;
typedef long long int           __align__(8)    aligned_lli;
typedef unsigned long long int  __align__(8)    UllInt;

#if defined(use_DPFP)
typedef double                  __align__(8)    NNAccumulator;
typedef double                  __align__(8)    NNDouble;
typedef double                  __align__(8)    Float;
typedef double2                 __align__(16)   NNDouble2;
typedef double4                 __align__(32)   NNDouble4;
typedef double2                 __align__(16)   Float2;
typedef double4                 __align__(16)   Float4;
static const MPI_Datatype MPI_NNDOUBLE          = MPI_DOUBLE_PRECISION;
static const MPI_Datatype MPI_Float           = MPI_DOUBLE_PRECISION;
static const MPI_Datatype MPI_NNACCUMULATOR     = MPI_FLOAT;
#elif defined(use_SPFP)
typedef float                                   NNAccumulator;
typedef double                  __align__(8)    NNDouble;
typedef float                                   Float;
typedef double2                 __align__(16)   NNDouble2;
typedef double4                 __align__(32)   NNDouble4;
typedef float2                  __align__(8)    Float2;
typedef float4                  __align__(16)   Float4;
static const MPI_Datatype MPI_NNDOUBLE          = MPI_DOUBLE_PRECISION;
static const MPI_Datatype MPI_Float           = MPI_FLOAT;
static const MPI_Datatype MPI_NNACCUMULATOR     = MPI_LONG_LONG_INT;
#else
typedef float                                   NNAccumulator;
typedef double                  __align(8)__    NNDouble;
typedef float                                   Float;
typedef double2                 __align(16)__   NNDouble2;
typedef double4                 __align(32)__   NNDouble4;
typedef float2                  __align(8)__    Float2;
typedef float4                  __align(16)__   Float4;
static const MPI_Datatype MPI_NNDOUBLE          = MPI_DOUBLE_PRECISION;
static const MPI_Datatype MPI_Float           = MPI_FLOAT;
static const MPI_Datatype MPI_NNACCUMULATOR     = MPI_LONG_LONG_INT;
#endif

static const int SM_3X_THREADS_PER_BLOCK                        = 128;
static const int SM_5X_THREADS_PER_BLOCK                        = 128;
static const int SM_6X_THREADS_PER_BLOCK                        = 128;

#if (__CUDA_ARCH__ >= 600)
#define LAUNCH_BOUNDS() __launch_bounds__(SM_6X_THREADS_PER_BLOCK, 8)
#define LAUNCH_BOUNDS256() __launch_bounds__(256, 5)
#elif (__CUDA_ARCH__ >= 500)
#define LAUNCH_BOUNDS() __launch_bounds__(SM_5X_THREADS_PER_BLOCK, 8)
#define LAUNCH_BOUNDS256() __launch_bounds__(256, 5)
#else
#define LAUNCH_BOUNDS() __launch_bounds__(SM_3X_THREADS_PER_BLOCK, 10)
#define LAUNCH_BOUNDS256() __launch_bounds__(256, 4)
#endif
#define LAUNCH_BOUNDS512() __launch_bounds__(512, 2)
#define LAUNCH_BOUNDS1024() __launch_bounds__(1024, 1)

static const uint32_t SM_6X_MAXSPARSE = 4608;
static const uint32_t SM_6X_MAXSPARSEANALOG = 2304;
static const uint32_t SM_5X_MAXSPARSE = 4608;
static const uint32_t SM_5X_MAXSPARSEANALOG = 2304;
static const uint32_t SM_3X_MAXSPARSE = 2304;
static const uint32_t SM_3X_MAXSPARSEANALOG = 1152;

static const bool bShadowedOutputBuffers                        = false;

#define FPSCALE  (1ll << 40)
#define DFSCALE (1ll << 44)

#ifdef GVERBOSE
#ifndef MEMTRACKING
#define MEMTRACKING
#endif

#ifdef SYNCHRONOUS
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

#define RTERROR(status, s) \
    if (status != cudaSuccess) { \
        std::cerr << s << " " << cudaGetErrorString(status) << std::endl; \
        assert(0); \
        cudaThreadExit(); \
        exit(-1); \
    }

#define CUDNNERROR(status, s) \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << s << " " << cudnnGetErrorString(status) << std::endl; \
        assert(0); \
        cudaThreadExit(); \
        exit(-1); \
    }

struct GpuData {
    unsigned int            _warpSize;
    unsigned int            _warpBits;
    unsigned int            _warpMask;
    unsigned long long int* _pAccumulator;

    float                   _LRN_k;
    int                     _LRN_n;
    float                   _LRN_alpha;
    float                   _LRN_beta;

    int                     _maxout_k;

    float                   _deltaBoost_one;
    float                   _deltaBoost_zero;

    float                   _SMCE_oneTarget;
    float                   _SMCE_zeroTarget;
    float                   _SMCE_oneScale;
    float                   _SMCE_zeroScale;

    bool                    _bSparsenessPenalty;
    float                   _sparsenessPenalty_p;
    float                   _sparsenessPenalty_beta;

    bool                    _bDenoising;
    float                   _denoising_p;
    float                   _denoising_q;

    bool                    _bShuffleIndices;
    unsigned int*           _pShuffleIndex;

    uint32_t                _maxUint32_t;
    int32_t                 _maxInt32_t;
    uint64_t                _maxUint64_t;
    int64_t                 _maxInt64_t;
    float                   _maxFloat;
    float                   _minFloat;
};

template <typename T> struct GpuBuffer;
template <typename T> struct MultiGpuBuffer;
class Network;

struct GpuContext {
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

    GpuData                             _data;
    bool                                _bECCSupport;
    bool                                _bCanMapHostMemory;
    aligned_lli                         _totalMemory;
    aligned_lli                         _totalCPUMemory;
    aligned_lli                         _totalGPUMemory;
    bool                                _bUnifiedMemory;

    SM_VERSION                          _sm_version;
    unsigned int                        _sm_major;
    unsigned int                        _threadsPerBlock;
    unsigned int                        _warpSize;
    unsigned int                        _warpBits;
    unsigned int                        _warpMask;
    int                                 _numprocs;
    int                                 _id;
    int                                 _device;

    uint32_t                            _maxSparse;
    uint32_t                            _maxSparseAnalog;

    cublasHandle_t                      _cuBLASHandle;

    curandGenerator_t                   _RNG;

    cudnnHandle_t                       _cuDNNHandle;

    Network*                          _pNetwork;
    std::unique_ptr<GpuBuffer<unsigned long long int>> _pbAccumulator;
    bool                                _bCPUValidate;
    float                               _acceptableError;

    bool                                _bSingleNode;
    bool                                _bP2P;

    GpuContext();
    ~GpuContext();
    void GetMemoryUsage(int* gpuMemory, int* cpuMemory);
    void SetRandomSeed(unsigned long seed);
    void SetNeuralNetwork(Network* pNetwork);
    void SetFastMath(bool flag);
    void Startup(int argc, char** argv);
    void Shutdown();
    void CopyConstants();
    void SetCPUValidate(bool bCPUValidate);

    static unsigned int Pad(unsigned int x) { return (x + PADDING - 1) & PADDINGMASK; }
};

extern struct GpuContext& getGpu();

template <typename T>
struct GpuBuffer
{
    size_t                  _length;
    bool                    _bSysMem;
    bool                    _bManaged;
    T*                      _pSysData;
    T*                      _pDevData;
    GpuBuffer(int length, bool bSysMem = false, bool bManaged = false);
    GpuBuffer(unsigned int length, bool bSysMem = false, bool bManaged = false);
    GpuBuffer(unsigned long long int length, bool bSysMem = false, bool bManaged = false);
    GpuBuffer(size_t length, bool bSysMem = false, bool bManaged = false);
    virtual ~GpuBuffer();

    void Allocate();

    void Resize(size_t length);

    void Deallocate();

    void Upload(const T* pBuff = nullptr) const;

    void Download(T * pBuff = nullptr);

    void Copy(T* pBuff);

    size_t GetLength();

    size_t GetSize();
};

template <typename T>
GpuBuffer<T>::GpuBuffer(int length, bool bSysMem, bool bManaged) : _length(length), _bSysMem(bSysMem), _bManaged(bManaged), _pSysData(nullptr), _pDevData(nullptr)
{
    Allocate();
}

template <typename T>
GpuBuffer<T>::GpuBuffer(unsigned int length, bool bSysMem, bool bManaged) : _length(length), _bSysMem(bSysMem), _bManaged(bManaged), _pSysData(nullptr), _pDevData(nullptr)
{
    Allocate();
}

template <typename T>
GpuBuffer<T>::GpuBuffer(unsigned long long int length, bool bSysMem, bool bManaged) : _length(length), _bSysMem(bSysMem), _bManaged(bManaged), _pSysData(nullptr), _pDevData(nullptr)
{
    Allocate();
}

template <typename T>
GpuBuffer<T>::GpuBuffer(size_t length, bool bSysMem, bool bManaged) : _length(length), _bSysMem(bSysMem), _bManaged(bManaged), _pSysData(nullptr), _pDevData(nullptr)
{
    Allocate();
}

template <typename T>
GpuBuffer<T>::~GpuBuffer()
{
    Deallocate();
}

template <typename T>
void GpuBuffer<T>::Allocate()
{
    cudaError_t status;

    if (_bManaged)
        _bSysMem    = true;

#ifdef MEMTRACKING
    std::cerr << "Allocating " << _length * sizeof(T) << " bytes of GPU memory";
    if (!_bSysMem)
    {
        std::cerr << ", unshadowed";
    }
    else if (_bManaged)
    {
        std::cerr << ", managed";
    }
    std::cerr << std::endl;
#endif

    if (_bManaged)
    {
        status = cudaMallocManaged((void **) &_pDevData, _length * sizeof(T), cudaMemAttachGlobal);
        getGpu()._totalGPUMemory           +=  _length * sizeof(T);
        _pSysData = _pDevData;
        RTERROR(status, "GpuBuffer::Allocate failed (cudaMallocManaged)");
        memset(_pSysData, 0, _length * sizeof(T));
    }
    else
    {
        status = cudaMalloc((void **) &_pDevData, _length * sizeof(T));
        getGpu()._totalGPUMemory           +=  _length * sizeof(T);
        RTERROR(status, "GpuBuffer::Allocate failed (cudaMalloc)");
        status = cudaMemset((void *) _pDevData, 0, _length * sizeof(T));
        RTERROR(status, "GpuBuffer::Allocate failed (cudaMemset)");

        if (_bSysMem)
        {
            _pSysData                           =  new T[_length];
            getGpu()._totalCPUMemory           +=  _length * sizeof(T);
            memset(_pSysData, 0, _length * sizeof(T));
        }
    }

#ifdef MEMTRACKING
    std::cerr << "Mem++: " << getGpu()._totalGPUMemory << " " << getGpu()._totalCPUMemory << std::endl;
#endif
}

template<typename T> void GpuBuffer<T>::Resize(size_t length)
{
    if(length > _length)
    {
        Deallocate();
        _length = length;
        Allocate();
    }
}

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

    _pSysData = nullptr;
    _pDevData = nullptr;
    _length = 0;
#ifdef MEMTRACKING
    std::cerr << "Mem--: " << getGpu()._totalGPUMemory << " " << getGpu()._totalCPUMemory << std::endl;
#endif
}

template <typename T>
void GpuBuffer<T>::Copy(T* pBuff)
{
    cudaError_t status;
    status = cudaMemcpy(_pDevData, pBuff, _length * sizeof(T), cudaMemcpyDeviceToDevice);
    RTERROR(status, "cudaMemcpy GpuBuffer::Upload failed");
}

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

template<typename T> size_t GpuBuffer<T>::GetLength()
{
    return _length;
}

template<typename T> size_t GpuBuffer<T>::GetSize()
{
    return _length * sizeof(T);
}

void verifySGEMM(GpuBuffer<Float>* pbA, GpuBuffer<Float>* pbB, GpuBuffer<Float>* pbC, uint32_t m, uint32_t k, uint32_t n);
void verifySGEMMNT(GpuBuffer<Float>* pbA, GpuBuffer<Float>* pbB, GpuBuffer<Float>* pbC, uint32_t m, uint32_t k, uint32_t n);
void verifySGEMMTN(GpuBuffer<Float>* pbA, GpuBuffer<Float>* pbB, GpuBuffer<Float>* pbC, uint32_t m, uint32_t k, uint32_t n);

#define SGEMM(A,B,C,m,n,k,alpha,beta,transf_A,transf_B) \
        cublasSgemm(getGpu()._cuBLASHandle, transf_B, transf_A, n, m, k, alpha, B, n, A, k, beta, C, n)

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#define printf(f,...)
#endif
#endif
