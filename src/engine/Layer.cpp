
#include "GpuTypes.h"
#include "NcExcptionWrap.h"
#include "Types.h"
#include "kernels.h"
#define __STDC_FORMAT_MACROS
#include <iostream>
#include <vector>
#include <cstring>
#include <cstddef>
#include <cinttypes>

using namespace netCDF;
using namespace netCDF::exceptions;

/**
 * @brief Print the contents of a Float array with the specified name.
 *
 * @param name The name of the array.
 * @param p The pointer to the Float array.
 * @param stride The stride of the array.
 */
void DumpP(const char* name, Float* p, int stride) {
    std::cout << name << ": ";
    std::vector<Float> data(stride);
    std::memcpy(data.data(), p, stride * sizeof(Float));
    for (const auto& i : data) {
        std::cout << i << ", ";
    }
    std::cout << std::endl;
}

Layer::Layer(LayerDescriptor& d, uint32_t batch)
    /**
     * @brief The name of the layer.
     */
    _name(d._name),
    /**
     * @brief The kind of layer.
     */
    _kind(d._kind),

    /**
     * @brief The type of layer.
     */
    _type(d._type),

    /**
     * @brief The attributes of the layer.
     */
    _attributes(d._attributes),

    /**
     * @brief The pooling function used by the layer.
     */
    _poolingFunction(d._poolingFunction),

    /**
     * @brief The data set used by the layer.
     */
    _dataSet(d._dataSet),

    /**
     * @brief Pointer to the data set.
     */
    _pDataSet(nullptr),

    /**
     * @brief The vSource of the layer.
     */
    _vSource(d._vSource),

    /**
     * @brief The vSkip of the layer.
     */
    _vSkip(d._vSkip),

    /**
     * @brief The pbUnit of the layer.
     */
    _pbUnit(),

    /**
     * @brief The pbDelta of the layer.
     */
    _pbDelta(),

    /**
     * @brief The pbDropout of the layer.
     */
    _pbDropout(),

    /**
     * @brief The pbDeltaBN of the layer.
     */
    _pbDeltaBN(),

    /**
     * @brief The pbScaleGradientBN of the layer.
     */
    _pbScaleGradientBN(),

    /**
     * @brief The pbScaleGradientVelocityBN of the layer.
     */
    _pbScaleGradientVelocityBN(),

    /**
     * @brief The pbBiasGradientBN of the layer.
     */
    _pbBiasGradientBN(),

    /**
     * @brief The pbBiasGradientVelocityBN of the layer.
     */
    _pbBiasGradientVelocityBN(),

    /**
     * @brief The pbUnitBN of the layer.
     */
    _pbUnitBN(),

    /**
     * @brief The pbScaleBN of the layer.
     */
    _pbScaleBN(),

    /**
     * @brief The pbBiasBN of the layer.
     */
    _pbBiasBN(),

    /**
     * @brief The pbRunningMeanBN of the layer.
     */
    _pbRunningMeanBN(),

    /**
     * @brief The pbRunningVarianceBN of the layer.
     */
    _pbRunningVarianceBN(),

    /**
     * @brief The pbSaveMeanBN of the layer.
     */
    _pbSaveMeanBN(),

    /**
     * @brief The pbSaveInvVarianceBN of the layer.
     */
    _pbSaveInvVarianceBN(),

    /**
     * @brief The Nx dimension of the layer.
     */
    _Nx(d._Nx),

    /**
     * @brief The Ny dimension of the layer.
     */
    _Ny(d._Ny),

    /**
     * @brief The Nz dimension of the layer.
     */
    _Nz(d._Nz),

    /**
     * @brief The Nw dimension of the layer.
     */
    _Nw(d._Nw),

    /**
     * @brief The strideBN of the layer.
     */
    _strideBN(0),

    /**
     * @brief The dimensions of the layer.
     */
    _dimensions(d._dimensions),

    /**
     * @brief The weight initialization of the layer.
     */
    _weightInit(d._weightInit),

    /**
     * @brief The weight initialization scale of the layer.
     */
    _weightInitScale(d._weightInitScale),

    /**
     * @brief The bias initialization of the layer.
     */
    _biasInit(d._biasInit),

    /**
     * @brief The X dimension of the kernel.
     */
    _kernelX(d._kernelX),

    /**
     * @brief The Y dimension of the kernel.
     */
    _kernelY(d._kernelY),

    /**
     * @brief The Z dimension of the kernel.
     */
    _kernelZ(d._kernelZ),

    /**
     * @brief The X dimension stride of the kernel.
     */
    _kernelStrideX(d._kernelStrideX),

    /**
     * @brief The Y dimension stride of the kernel.
     */
    _kernelStrideY(d._kernelStrideY),

    /**
     * @brief The Z dimension stride of the kernel.
     */
    _kernelStrideZ(d._kernelStrideZ),

    /**
     * @brief The X dimension padding of the kernel.
     */
    _kernelPaddingX(d._kernelPaddingX),

    /**
     * @brief The Y dimension padding of the kernel.
     */
    _kernelPaddingY(d._kernelPaddingY),

    /**
     * @brief The Z dimension padding of the kernel.
     */
    _kernelPaddingZ(d._kernelPaddingZ),

    /**
     * @brief The kernel dimensions of the layer.
     */
    _kernelDimensions(d._kernelDimensions),

    /**
     * @brief The weight normalization of the layer.
     */
    _weightNorm(d._weightNorm),

    /**
     * @brief The delta normalization of the layer.
     */
    _deltaNorm(d._deltaNorm),

    /**
     * @brief The dropout probability of the layer.
     */
    _pDropout(d._pDropout),

    /**
     * @brief The activation function of the layer.
     */
    _activation(d._activation),

    /**
     * @brief Flag indicating whether the batch size is odd.
     */
    _oddBatch(0),

    /**
     * @brief Flag indicating whether the layer is sparse.
     */
    _bSparse(d._attributes & Layer::Attributes::Sparse),

    /**
     * @brief The sparseness penalty p value of the layer.
     */
    _sparsenessPenalty_p(d._sparsenessPenalty_p),

    /**
     * @brief The sparseness penalty beta value of the layer.
     */
    _sparsenessPenalty_beta(d._sparsenessPenalty_beta),

    /**
     * @brief Flag indicating whether the layer is denoising.
     */
    _bDenoising(d._attributes & Layer::Attributes::Denoising),

    /**
     * @brief Flag indicating whether fast sparse computation is used.
     */
    _bFastSparse(false),

    /**
     * @brief Flag indicating whether the layer is dirty.
     */
    _bDirty(true),

    /**
     * @brief The number of batch normalization calls.
     */
    _bnCalls(0),

    /**
     * @brief The priority of the layer.
     */
    _priority(-1),

    /**
     * @brief The delta update count of the layer.
     */
    _deltaUpdateCount(0),

    /**
     * @brief The unit update count of the layer.
     */
    _unitUpdateCount(0),

    /**
     * @brief The batch size for the layer.
     */
    _batch(batch),

    /**
     * @brief The local batch size for the layer.
     */
    _localBatch(batch),

    /**
     * @brief The slope value for the ReLU activation function.
     */
    _RELUSlope(d._RELUSlope),

    /**
     * @brief The alpha value for the ELU activation function.
     */
    _ELUAlpha(d._ELUAlpha),

    /**
     * @brief The lambda value for the SELU activation function.
     */
    _SELULambda(d._SELULambda),

    /**
     * @brief Flag indicating whether batch normalization is used.
     */
    _bBatchNormalization(d._attributes & Layer::Attributes::BatchNormalization)

{
    /**
     * @brief Calculate the stride of the layer based on the dimensions (_Nx, _Ny, _Nz, _Nw).
     */
    _stride = _Nx * _Ny * _Nz * _Nw;

    /**
     * @brief Determine the parallelization type based on the layer type.
     *
     * For FullyConnected type, the parallelization is Model.
     * For other types, the parallelization is Data.
     */
    if (_type == FullyConnected)
        _parallelization = Model;
    else
        _parallelization = Data;

    /**
     * @brief Calculate the minimum and maximum X values for the current GPU.
     */
    _minX = static_cast<size_t>(_Nx * getGpu()._id) / static_cast<size_t>(getGpu()._numprocs);
    _maxX = static_cast<size_t>(_Nx * (getGpu()._id + 1)) / static_cast<size_t>(getGpu()._numprocs);

    /**
     * @brief Calculate the local stride for the current GPU based on the X range (_minX, _maxX) and dimensions (_Ny, _Nz, _Nw).
     */
    _localStride = (_maxX - _minX) * _Ny * _Nz * _Nw;

    /**
     * @brief Calculate the maximum local stride across all GPUs based on the total X range and dimensions (_Ny, _Nz, _Nw).
     */
    _maxLocalStride = (((static_cast<size_t>(_Nx) + getGpu()._numprocs - 1) / static_cast<size_t>(getGpu()._numprocs)) * _Ny * _Nz * _Nw);

    /**
     * @brief Check if the layer type is Pooling or Convolutional.
     */
    if ((_type == Layer::Type::Pooling) || (_type == Layer::Type::Convolutional))

    {
        /**
         * @brief Create a tensor descriptor for the layer.
         *
         * @param _tensorDescriptor The pointer to the tensor descriptor.
         * @return The cudnnStatus_t indicating the success or failure of the operation.
         */
        cudnnStatus_t cudnnStatus = cudnnCreateTensorDescriptor(&_tensorDescriptor);
        CUDNNERROR(cudnnStatus, "Layer::Layer: unable to create _tensorDescriptor");

        /**
         * @brief Create a tensor descriptor for the odd batch size.
         *
         * @param _oddBatchTensorDescriptor The pointer to the odd batch tensor descriptor.
         * @return The cudnnStatus_t indicating the success or failure of the operation.
         */
        cudnnStatus = cudnnCreateTensorDescriptor(&_oddBatchTensorDescriptor);
        CUDNNERROR(cudnnStatus, "Layer::Layer: unable to create _oddBatchTensorDescriptor");
    }

    if (_bBatchNormalization)
    {
        /**
         * @brief The CUDA status.
         */
        cudaError_t status;

        /**
         * @brief Create a tensor descriptor for scale, bias, mean, and variance of batch normalization.
         *
         * @param _scaleBiasMeanVarDescBN The pointer to the tensor descriptor.
         * @return The cudnnStatus_t indicating the success or failure of the operation.
         */
        cudnnStatus_t cudnnStatus = cudnnCreateTensorDescriptor(&_scaleBiasMeanVarDescBN);
        CUDNNERROR(cudnnStatus, "Layer::Layer: unable to create _scaleBiasMeanVarDescBN");

        /**
         * @brief Create a tensor descriptor for batch normalization.
         *
         * @param _tensorDescriptorBN The pointer to the tensor descriptor.
         * @return The cudnnStatus_t indicating the success or failure of the operation.
         */
        cudnnStatus = cudnnCreateTensorDescriptor(&_tensorDescriptorBN);
        CUDNNERROR(cudnnStatus, "Layer::Layer: unable to create _tensorDescriptorBN");

        /**
         * @brief Set the strideBN based on the layer type.
         */
        if (_type == Layer::Type::Convolutional)
            _strideBN = _Nz;
        else
            _strideBN = _localStride;

        /**
         * @brief Reset the unique pointer for _pbScaleGradientBN.
         */
        _pbScaleGradientBN.reset(new GpuBuffer<Float>(_strideBN));

        /**
         * @brief Reset the unique pointer for _pbBiasGradientBN.
         */
        _pbBiasGradientBN.reset(new GpuBuffer<Float>(_strideBN));

        /**
         * @brief Reset the unique pointer for _pbScaleBN.
         */
        _pbScaleBN.reset(new GpuBuffer<Float>(_strideBN));

        /**
         * @brief Reset the unique pointer for _pbBiasBN.
         */
        _pbBiasBN.reset(new GpuBuffer<Float>(_strideBN));

        /**
         * @brief Reset the unique pointer for _pbRunningMeanBN.
         */
        _pbRunningMeanBN.reset(new GpuBuffer<Float>(_strideBN));

        /**
         * @brief Reset the unique pointer for _pbRunningVarianceBN.
         */
        _pbRunningVarianceBN.reset(new GpuBuffer<Float>(_strideBN));

        /**
         * @brief Reset the unique pointer for _pbSaveMeanBN.
         */
        _pbSaveMeanBN.reset(new GpuBuffer<Float>(_strideBN));

        /**
         * @brief Reset the unique pointer for _pbSaveInvVarianceBN.
         */
        _pbSaveInvVarianceBN.reset(new GpuBuffer<Float>(_strideBN));

        if (getGpu()._id == 0)
        {
            std::cout << "Layer::Layer: Allocating " << _strideBN * sizeof(Float)
                      << " bytes of BN scale diff for layer " << _name << std::endl;
            std::cout << "Layer::Layer: Allocating " << _strideBN * sizeof(Float)
                      << " bytes of BN bias diff for layer " << _name << std::endl;
            std::cout << "Layer::Layer: Allocating " << _strideBN * sizeof(Float)
                      << " bytes of BN scale for layer " << _name << std::endl;
            std::cout << "Layer::Layer: Allocating " << _strideBN * sizeof(Float)
                      << " bytes of BN bias for layer " << _name << std::endl;
            std::cout << "Layer::Layer: Allocating " << _strideBN * sizeof(Float)
                      << " bytes of BN running mean for layer " << _name << std::endl;
            std::cout << "Layer::Layer: Allocating " << _strideBN * sizeof(Float)
                      << " bytes of BN running variance for layer " << _name << std::endl;
            std::cout << "Layer::Layer: Allocating " << _strideBN * sizeof(Float)
                      << " bytes of BN saving mean for layer " << _name << std::endl;
            std::cout << "Layer::Layer: Allocating " << _strideBN * sizeof(Float)
                      << " bytes of BN saving InvVariance for layer " << _name << std::endl;
        }

        /**
         * @brief Copy the values from d._vScaleBN to _pbScaleBN if d._vScaleBN is not empty; otherwise, initialize _pbScaleBN with ones.
         */
        if (!d._vScaleBN.empty())
        {
            status = cudaMemcpy(_pbScaleBN->_pDevData, d._vScaleBN.data(), _strideBN * sizeof(Float), cudaMemcpyHostToDevice);
        }
        else
        {
            std::vector<Float> ones(_strideBN, 1);
            status = cudaMemcpy(_pbScaleBN->_pDevData, ones.data(), _strideBN * sizeof(Float), cudaMemcpyHostToDevice);
        }
        RTERROR(status, "Layer::Layer: cudaMemcpy failed on  _pbScaleBN");

        /**
         * @brief Copy the values from d._vBiasBN to _pbBiasBN if d._vBiasBN is not empty; otherwise, set _pbBiasBN to all zeros.
         */
        if (!d._vBiasBN.empty())
        {
            status = cudaMemcpy(_pbBiasBN->_pDevData, d._vBiasBN.data(), _strideBN * sizeof(Float), cudaMemcpyHostToDevice);
        }
        else
        {
            status = cudaMemset(_pbBiasBN->_pDevData, 0, _strideBN * sizeof(Float));
        }
        RTERROR(status, "Layer::Layer: cudaMemcpy failed on  _pbBiasBN");

        /**
         * @brief Copy the values from d._vRunningMeanBN to _pbRunningMeanBN if d._vRunningMeanBN is not empty; otherwise, set _pbRunningMeanBN to all zeros.
         */
        if (!d._vRunningMeanBN.empty())
        {
            status = cudaMemcpy(_pbRunningMeanBN->_pDevData, d._vRunningMeanBN.data(), _strideBN * sizeof(Float), cudaMemcpyHostToDevice);
        }
        else
        {
            status = cudaMemset(_pbRunningMeanBN->_pDevData, 0, _strideBN * sizeof(Float));
        }
        RTERROR(status, "Layer::Layer: cudaMemcpy failed on  _pbRunningMeanBN");

        /**
         * @brief Copy the values from d._vRunningVarianceBN to _pbRunningVarianceBN if d._vRunningVarianceBN is not empty; otherwise, set _pbRunningVarianceBN to all zeros.
         */
        if (!d._vRunningVarianceBN.empty())
        {
            status = cudaMemcpy(_pbRunningVarianceBN->_pDevData, d._vRunningVarianceBN.data(), _strideBN * sizeof(Float), cudaMemcpyHostToDevice);
        }
        else
        {
            status = cudaMemset(_pbRunningVarianceBN->_pDevData, 0, _strideBN * sizeof(Float));
        }
        RTERROR(status, "Layer::Layer: cudaMemcpy failed on  _pbRunningVarianceBN");

        /**
         * @brief Set _pbScaleGradientBN to all zeros.
         */
        status = cudaMemset(_pbScaleGradientBN->_pDevData, 0, _strideBN * sizeof(Float));
        RTERROR(status, "Layer::Layer: cudaMemset failed on  _pbScaleGradientBN");

        /**
         * @brief Set _pbBiasGradientBN to all zeros.
         */
        status = cudaMemset(_pbBiasGradientBN->_pDevData, 0, _strideBN * sizeof(Float));
        RTERROR(status, "Layer::Layer: cudaMemset failed on  _pbBiasGradientBN");

        /**
         * @brief Set _pbSaveMeanBN to all zeros.
         */
        status = cudaMemset(_pbSaveMeanBN->_pDevData, 0, _strideBN * sizeof(Float));
        RTERROR(status, "Layer::Layer: cudaMemset failed on  _pbSaveMeanBN");

        /**
         * @brief Set _pbSaveInvVarianceBN to all zeros.
         */
        status = cudaMemset(_pbSaveInvVarianceBN->_pDevData, 0, _strideBN * sizeof(Float));
        RTERROR(status, "Layer::Layer: cudaMemset failed on  _pbSaveInvVarianceBN");
    }

    if (_type == Layer::Type::Pooling)
    {
        /**
         * @brief Create a pooling descriptor for the layer.
         *
         * @param _poolingDescriptor The pointer to the pooling descriptor.
         * @return The cudnnStatus_t indicating the success or failure of the operation.
         */
        cudnnStatus_t cudnnStatus = cudnnCreatePoolingDescriptor(&_poolingDescriptor);
        CUDNNERROR(cudnnStatus, "Layer::Layer: unable to create pooling descriptor");

        /**
         * @brief Create vectors to store kernel dimensions, kernel padding, and kernel stride.
         */
        std::vector<int> vKernel(3);
        std::vector<int> vKernelPadding(3);
        std::vector<int> vKernelStride(3);

        /**
         * @brief Set the values for kernel dimensions, kernel padding, and kernel stride.
         */
        vKernel[0] = _kernelX;
        vKernel[1] = _kernelY;
        vKernel[2] = _kernelZ;
        vKernelPadding[0] = _kernelPaddingX;
        vKernelPadding[1] = _kernelPaddingY;
        vKernelPadding[2] = _kernelPaddingZ;
        vKernelStride[0] = _kernelStrideX;
        vKernelStride[1] = _kernelStrideY;
        vKernelStride[2] = _kernelStrideZ;

        switch (_poolingFunction)
        {
            case PoolingFunction::Max:
                cudnnSetPoolingNdDescriptor(_poolingDescriptor,
                                            CUDNN_POOLING_MAX,
                                            CUDNN_PROPAGATE_NAN,
                                            _kernelDimensions,
                                            vKernel.data(),
                                            vKernelPadding.data(),
                                            vKernelStride.data());
                CUDNNERROR(cudnnStatus, "Layer::Layer: unable to set max pooling descriptor");
                break;

            case PoolingFunction::Average:
                /**
                 * @brief Set the descriptor for pooling operation.
                 *
                 * @param _poolingDescriptor The pooling descriptor to be set.
                 * @param CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING The pooling mode (average pooling with count excluding padding).
                 * @param CUDNN_PROPAGATE_NAN Flag indicating whether to propagate NaN values.
                 * @param _kernelDimensions The dimensions of the pooling kernel.
                 * @param vKernel The data vector for the pooling kernel dimensions.
                 * @param vKernelPadding The data vector for the pooling kernel padding dimensions.
                 * @param vKernelStride The data vector for the pooling kernel stride dimensions.
                 * @return The cudnnStatus_t indicating the success or failure of the operation.
                 */
                cudnnSetPoolingNdDescriptor(_poolingDescriptor,
                                            CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
                                            CUDNN_PROPAGATE_NAN,
                                            _kernelDimensions,
                                            vKernel.data(),
                                            vKernelPadding.data(),
                                            vKernelStride.data());
                CUDNNERROR(cudnnStatus, "Layer::Layer: unable to set average pooling descriptor");
                break;

            case PoolingFunction::LRN:
                /**
                 * @brief Create a local response normalization (LRN) descriptor.
                 *
                 * @param _LRNDescriptor The pointer to the LRN descriptor.
                 * @return The cudnnStatus_t indicating the success or failure of the operation.
                 */
                cudnnStatus = cudnnCreateLRNDescriptor(&_LRNDescriptor);
                CUDNNERROR(cudnnStatus, "Layer::Layer: unable to create LRN descriptor");
                break;
        }
    }
}

Layer::~Layer()
{
    Deallocate();
    if ((_type == Layer::Type::Pooling) || (_type == Layer::Type::Convolutional))
    {
        /**
         * @brief Destroy the tensor descriptor for the layer.
         *
         * @param _tensorDescriptor The tensor descriptor to be destroyed.
         * @return The cudnnStatus_t indicating the success or failure of the operation.
         */
        cudnnStatus_t cudnnStatus = cudnnDestroyTensorDescriptor(_tensorDescriptor);
        CUDNNERROR(cudnnStatus, "Layer::~Layer: unable to delete _tensorDescriptor");

        /**
         * @brief Destroy the tensor descriptor for the odd batch size.
         *
         * @param _oddBatchTensorDescriptor The tensor descriptor to be destroyed.
         * @return The cudnnStatus_t indicating the success or failure of the operation.
         */
        cudnnStatus = cudnnDestroyTensorDescriptor(_oddBatchTensorDescriptor);
        CUDNNERROR(cudnnStatus, "Layer::~Layer: unable to delete _oddBatchTensorDescriptor");
    }

    if (_bBatchNormalization)
    {
        /**
         * @brief Destroy the tensor descriptor for scale, bias, mean, and variance of batch normalization.
         *
         * @param _scaleBiasMeanVarDescBN The tensor descriptor to be destroyed.
         * @return The cudnnStatus_t indicating the success or failure of the operation.
         */
        cudnnStatus_t cudnnStatus = cudnnDestroyTensorDescriptor(_scaleBiasMeanVarDescBN);
        CUDNNERROR(cudnnStatus, "Layer::~Layer: unable to delete _scaleBiasMeanVarDescBN");

        /**
         * @brief Destroy the tensor descriptor for batch normalization.
         *
         * @param _tensorDescriptorBN The tensor descriptor to be destroyed.
         * @return The cudnnStatus_t indicating the success or failure of the operation.
         */
        cudnnStatus = cudnnDestroyTensorDescriptor(_tensorDescriptorBN);
        CUDNNERROR(cudnnStatus, "Layer::~Layer: unable to delete _tensorDescriptorBN");

        /**
         * @brief Reset the unique pointer for pbScaleBN.
         */
        _pbScaleBN.reset();

        /**
         * @brief Reset the unique pointer for pbBiasBN.
         */
        _pbBiasBN.reset();

        /**
         * @brief Reset the unique pointer for pbScaleGradientBN.
         */
        _pbScaleGradientBN.reset();

        /**
         * @brief Reset the unique pointer for pbBiasGradientBN.
         */
        _pbBiasGradientBN.reset();

        /**
         * @brief Reset the unique pointer for pbRunningMeanBN.
         */
        _pbRunningMeanBN.reset();

        /**
         * @brief Reset the unique pointer for pbRunningVarianceBN.
         */
        _pbRunningVarianceBN.reset();

        /**
         * @brief Reset the unique pointer for pbSaveMeanBN.
         */
        _pbSaveMeanBN.reset();

        /**
         * @brief Reset the unique pointer for pbSaveInvVarianceBN.
         */
        _pbSaveInvVarianceBN.reset();

    }

    if (_type == Layer::Type::Pooling)
    {
        cudnnStatus_t cudnnStatus = cudnnDestroyPoolingDescriptor(_poolingDescriptor);
        CUDNNERROR(cudnnStatus, "Layer::~Layer: unable to destroy _poolingDescriptor");

        if (_poolingFunction == PoolingFunction::LRN)
        {
            cudnnStatus_t cudnnStatus = cudnnDestroyLRNDescriptor(_LRNDescriptor);
            CUDNNERROR(cudnnStatus, "Layer::~Layer: unable to delete _LRNDescriptor");
        }
    }
}

void Layer::Deallocate()
{
    if (getGpu()._id == 0)
        std::cout << "Layer::Deallocate: Deallocating all data for layer " << _name << std::endl;

    /**
     * @brief Reset the unique pointer for pbUnit.
     */
    _pbUnit.reset();

    /**
     * @brief Reset the unique pointer for pbUnitBN.
     */
    _pbUnitBN.reset();

    /**
     * @brief Reset the unique pointer for pbDelta.
     */
    _pbDelta.reset();

    /**
     * @brief Reset the unique pointer for pbDeltaBN.
     */
    _pbDeltaBN.reset();

    /**
     * @brief Reset the unique pointer for pbDropout.
     */
    _pbDropout.reset();

    /**
     * @brief Reset the unique pointer for pbBuffer1.
     */
    _pbBuffer1.reset();

    /**
     * @brief Reset the unique pointer for pbBuffer2.
     */
    _pbBuffer2.reset();

    /**
     * @brief Reset the unique pointer for pbScaleVelocityBN.
     */
    _pbScaleVelocityBN.reset();

    /**
     * @brief Reset the unique pointer for pbScaleGradientVelocityBN.
     */
    _pbScaleGradientVelocityBN.reset();

    /**
     * @brief Reset the unique pointer for pbBiasVelocityBN.
     */
    _pbBiasVelocityBN.reset();

    /**
     * @brief Reset the unique pointer for pbBiasGradientVelocityBN.
     */
    _pbBiasGradientVelocityBN.reset();

}

bool Layer::GetUnits(std::vector<Float>& vUnit)
{
    bool bValid = true;

    if (_pbUnit)
    {
        vUnit.reserve(_stride);
        _pbUnit->Download(vUnit);
    }
    else
    {
        std::cout << "Layer::GetUnits: Unit data not yet allocated.\n";
        bValid = false;
    }

    return bValid;
}

bool Layer::GetUnits(std::span<Float> pUnit)
{
    bool bValid = true;

    if (_pbUnit)
    {
        if (pUnit.size() < _stride)
        {
            std::cout << "Layer::GetUnits: Download span size is too small.\n";
            bValid = false;
        }
        else
        {
            _pbUnit->Download(pUnit.data());
        }
    }
    else
    {
        std::cout << "Layer::GetUnits: Unit data not yet allocated.\n";
        bValid = false;
    }

    return bValid;
}

bool Layer::GetDeltas(std::vector<Float>& vDelta)
{
    bool bValid = true;

    if (_pbDelta)
    {
        vDelta.reserve(_stride);
        _pbDelta->Download(vDelta);
    }
    else
    {
        std::cout << "Layer::GetDeltas: Deltas not yet allocated.\n";
        bValid = false;
    }

    return bValid;
}

bool Layer::GetDeltas(std::span<Float> pDelta)
{
    bool bValid = true;

    if (_pbDelta)
    {
        if (pDelta.size() < _stride)
        {
            std::cout << "Layer::GetDeltas: Download span size is too small.\n";
            bValid = false;
        }
        else
        {
            _pbDelta->Download(pDelta.data());
        }
    }
    else
    {
        std::cout << "Layer::GetDeltas: Deltas not yet allocated.\n";
        bValid = false;
    }

    return bValid;
}

bool Layer::SetUnits(const std::vector<Float>& vUnit)
{
    bool bValid = true;

    if (_pbUnit)
    {
        if (vUnit.size() < _stride)
        {
            std::cout << "Layer::SetUnits: Input unit data too small to set all units.\n";
            bValid = false;
        }

        _pbUnit->Upload(vUnit.data());
    }
    else
    {
        std::cout << "Layer::SetUnits: Unit data not yet allocated.\n";
        bValid = false;
    }

    return bValid;
}

bool Layer::SetDeltas(const std::vector<Float>& vDelta)
{
    bool bValid = true;

    if (_pbDelta)
    {
        if (vDelta.size() < _stride)
        {
            std::cout << "Layer::SetDeltas: Input delta data too small to set all deltas.\n";
            bValid = false;
        }

        _pbDelta->Upload(vDelta.data());
    }
    else
    {
        std::cout << "Layer::SetDeltas: Deltas not yet allocated.\n";
        bValid = false;
    }

    return bValid;
}

cudnnTensorDescriptor_t Layer::getTensorDescriptor(uint32_t batch)
{
    if (batch == _batch)
    {
        return _tensorDescriptor;
    }
    else if (batch != _oddBatch)
    {
        cudnnStatus_t cudnnStatus;
        std::array<int, 5> vDimensions{1, 1, 1, 1, 1};
        std::array<int, 5> vStride{1, 1, 1, 1, 1};

        switch (_dimensions)
        {
            case 2:
                vDimensions[0] = batch;
                vDimensions[1] = _Ny;
                vDimensions[2] = _Nx;
                vStride[2] = 1;
                vStride[1] = _Nx;
                vStride[0] = _Nx * _Ny;
                cudnnStatus = cudnnSetTensorNdDescriptor(_oddBatchTensorDescriptor, CUDNN_DATA_FLOAT, _dimensions + 1, vDimensions.data(), vStride.data());
                break;

            case 3:
                cudnnStatus = cudnnSetTensor4dDescriptor(_oddBatchTensorDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, _Nz, _Ny, _Nx);
                break;

            case 4:
                vDimensions[0] = batch;
                vDimensions[1] = _Nw;
                vDimensions[2] = _Nz;
                vDimensions[3] = _Ny;
                vDimensions[4] = _Nx;
                vStride[4] = 1;
                vStride[3] = _Nx;
                vStride[2] = _Nx * _Ny;
                vStride[1] = _Nx * _Ny * _Nz;
                vStride[0] = _Nx * _Ny * _Nz * _Nw;
                cudnnStatus = cudnnSetTensorNdDescriptor(_oddBatchTensorDescriptor, CUDNN_DATA_FLOAT, _dimensions + 1, vDimensions.data(), vStride.data());
                break;
        }
        CUDNNERROR(cudnnStatus, "Layer::getTensorDescriptor: Unable to set oddBatchTensorDescriptor");
        _oddBatch = batch;
    }

    return _oddBatchTensorDescriptor;
}

/**
 * @brief Retrieves the name of the layer.
 *
 * @return A constant reference to the name of the layer.
 */
const string& Layer::GetName() const {
    return _name;
}

/**
 * @brief Retrieves the name of the dataset associated with the layer.
 *
 * @return A constant reference to the name of the dataset.
 */
const string& Layer::GetDataSetName() const {
    return _dataSet;
}

/**
 * @brief Retrieves the kind of the layer.
 *
 * @return The kind of the layer.
 */
Layer::Kind Layer::GetKind() const {
    return _kind;
}

/**
 * @brief Retrieves the type of the layer.
 *
 * @return The type of the layer.
 */
Layer::Type Layer::GetType() const {
  return _type;
}

/**
 * @brief Retrieves the attributes of the layer.
 *
 * @return The attributes of the layer.
 */
uint32_t Layer::GetAttributes() const {
    return _attributes;
}

/**
 * @brief Retrieves the dataset associated with the layer.
 *
 * @return A pointer to the dataset associated with the layer.
 */
DataSetBase* Layer::GetDataSet() const {
    return _pDataSet;
}

/**
 * @brief Retrieves the number of dimensions of the layer.
 *
 * @return The number of dimensions of the layer.
 */
uint32_t Layer::GetNumDimensions() const {
    return _dimensions;
}

/**
 * @brief Retrieves the dimensions of the layer.
 *
 * @return A tuple representing the dimensions of the layer in the order (Nx, Ny, Nz, Nw).
 */
tuple<uint32_t, uint32_t, uint32_t, uint32_t> Layer::GetDimensions() const {
    return make_tuple(_Nx, _Ny, _Nz, _Nw);
}

/**
 * @brief Retrieves the local dimensions of the layer.
 *
 * @return A tuple representing the local dimensions of the layer in the order (maxX - minX, Ny, Nz, Nw).
 */
tuple<uint32_t, uint32_t, uint32_t, uint32_t> Layer::GetLocalDimensions() const {
    return make_tuple(_maxX - _minX, _Ny, _Nz, _Nw);
}

/**
 * @brief Retrieves the kernel dimensions of the layer.
 *
 * @return A tuple representing the kernel dimensions of the layer in the order (kernelX, kernelY, kernelZ).
 */
tuple<uint32_t, uint32_t, uint32_t> Layer::GetKernelDimensions() const {
    return make_tuple(_kernelX, _kernelY, _kernelZ);
}

/**
 * @brief Retrieves the kernel stride of the layer.
 *
 * @return A tuple representing the kernel stride of the layer in the order (kernelStrideX, kernelStrideY, kernelStrideZ).
 */
tuple<uint32_t, uint32_t, uint32_t> Layer::GetKernelStride() const {
    return make_tuple(_kernelStrideX, _kernelStrideY, _kernelStrideZ);
}

/**
 * @brief Retrieves the tensor descriptor and prints its details.
 *
 * @param t The tensor descriptor to retrieve.
 */
void Layer::DumpTensor(cudnnTensorDescriptor_t t)
{
    cudnnDataType_t dataType;
    int ndims;
    std::array<int, 16> vDim{};
    std::array<int, 16> vStride{};
    cudnnStatus_t cudnnStatus = cudnnGetTensorNdDescriptor(t, 8, &dataType, &ndims, vDim.data(), vStride.data());
    CUDNNERROR(cudnnStatus, "cudnnGetTensorNdDescriptor error");
    // Print tensor details
    std::cout << "Tensor:   " << ndims << " dimensions" << std::endl;
    std::cout << "DataType: " << dataType << std::endl;
    // Print each dimension and stride
    for (auto [i, dim, stride] : zipWithIndex(vDim, vStride))
        std::cout << i << ' ' << dim << ' ' << stride << std::endl;
    std::cout << std::endl;
}

void Layer::Allocate(bool validate)
{
    Deallocate();
    uint64_t size = static_cast<uint64_t>(_maxLocalStride) * static_cast<uint64_t>(_localBatch);

    if ((_type == Layer::Type::Pooling) && (_poolingFunction == PoolingFunction::Cosine))
    {
        _vBuffer1.resize(size);
        _pbBuffer1 = std::make_unique<GpuBuffer<Float>>(size);
        std::cout << std::format("Layer::Allocate: Allocating {} bytes ({}, {}) of auxilliary buffer 1 data for layer {}\n", size * sizeof(Float), _maxLocalStride, _localBatch, _name);
        _vBuffer2.resize(size);
        _pbBuffer2 = std::make_unique<GpuBuffer<Float>>(size);
        std::cout << std::format("Layer::Allocate: Allocating {} bytes ({}, {}) of auxilliary buffer 2 data for layer {}\n", size * sizeof(Float), _maxLocalStride, _localBatch, _name);
    }

    else if ((_type == Layer::Type::Pooling) || (_type == Layer::Type::Convolutional))
    {
        cudnnStatus_t cudnnStatus;
        std::array<int, 5> vDimensions{1, 1, 1, 1, 1};
        std::array<int, 5> vStride{1, 1, 1, 1, 1};

        switch (_dimensions)
        {
            /**
             * Set the tensor descriptor for case 2.
             *
             * @param _localBatch The local batch size.
             * @param _Ny The number of rows.
             * @param _Nx The number of columns.
             */
            case 2:
                vDimensions[0] = _localBatch;
                vDimensions[1] = _Ny;
                vDimensions[2] = _Nx;
                vStride[2] = 1;
                vStride[1] = _Nx;
                vStride[0] = _Nx * _Ny;
                cudnnStatus = cudnnSetTensorNdDescriptor(_tensorDescriptor, CUDNN_DATA_FLOAT, _dimensions + 1, vDimensions.data(), vStride.data());
                break;

            /**
             * Set the tensor descriptor for case 3.
             *
             * @param _localBatch The local batch size.
             * @param _Nz The number of channels.
             * @param _Ny The number of rows.
             * @param _Nx The number of columns.
             */
            case 3:
                cudnnStatus = cudnnSetTensor4dDescriptor(_tensorDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, _localBatch, _Nz, _Ny, _Nx);
                break;

            /**
             * Set the tensor descriptor for case 4.
             *
             * @param _localBatch The local batch size.
             * @param _Nw The number of samples.
             * @param _Nz The number of channels.
             * @param _Ny The number of rows.
             * @param _Nx The number of columns.
             */
            case 4:
                vDimensions[0] = _localBatch;
                vDimensions[1] = _Nw;
                vDimensions[2] = _Nz;
                vDimensions[3] = _Ny;
                vDimensions[4] = _Nx;
                vStride[4] = 1;
                vStride[3] = _Nx;
                vStride[2] = _Nx * _Ny;
                vStride[1] = _Nx * _Ny * _Nz;
                vStride[0] = _Nx * _Ny * _Nz * _Nw;
                cudnnStatus = cudnnSetTensorNdDescriptor(_tensorDescriptor, CUDNN_DATA_FLOAT, _dimensions + 1, vDimensions.data(), vStride.data());
                break;
        }
        CUDNNERROR(cudnnStatus, "Layer::Allocate: Unable to set tensor descriptor");
        DumpTensor(_tensorDescriptor);
    }

    if (!_bSparse || !_bFastSparse || (_kind != Input) || (_bSparse && (_kind == Input) && validate))
    {
        _vUnit.resize(size);
        _pbUnit = std::make_unique<GpuBuffer<Float>>(size);
        std::cout << std::format("Layer::Allocate: Allocating {} bytes ({}, {}) of unit data for layer {}\n", size * sizeof(Float), _maxLocalStride, _localBatch, _name);
    }

    if (_kind != Input)
    {
        _vDelta.resize(size);
        _pbDelta = std::make_unique<GpuBuffer<Float>>(size);
        std::cout << std::format("Layer::Allocate: Allocating {} bytes ({}, {}) of delta data for layer {}\n", size * sizeof(Float), _maxLocalStride, _localBatch, _name);

        if (_bBatchNormalization)
        {
            _pbUnitBN = std::make_unique<GpuBuffer<Float>>(size);
            _pbDeltaBN = std::make_unique<GpuBuffer<Float>>(size);
        }
    }

    if (_pDropout > static_cast<Float>(0.0))
    {
        _pbDropout = std::make_unique<GpuBuffer<Float>>(size);
        std::cout << std::format("Layer::Allocate: Allocating {} bytes ({}, {}) of dropout data for layer {}\n", size * sizeof(Float), _maxLocalStride, _localBatch, _name);
    }
    _bDirty = false;
}

void Layer::SetBatch(uint32_t batch)
{
    if (batch != _batch)
    {
        _batch = batch;
        if (_parallelization == Layer::Parallelization::Data)
            _localBatch = batch / getGpu()._numprocs;
        else
            _localBatch = batch;
        _bDirty = true;
    }
}

void Layer::RefreshParallelization()
{
/**
 * Count the number of layers of a specific type in the incoming layer vector.
 *
 * @tparam TLayer The layer type.
 * @param _vIncomingLayer The vector of incoming layers.
 * @param _type The layer type to count.
 * @return The count of layers of the specified type.
 */
constexpr uint32_t convolutionalInputs = countLayersOfType(_vIncomingLayer, Layer::Type::Convolutional);

/**
 * Count the number of layers of a specific type in the incoming layer vector.
 *
 * @tparam TLayer The layer type.
 * @param _vIncomingLayer The vector of incoming layers.
 * @param _type The layer type to count.
 * @return The count of layers of the specified type.
 */
constexpr uint32_t fullyConnectedInputs = countLayersOfType(_vIncomingLayer, Layer::Type::FullyConnected);

/**
 * Count the number of layers of a specific type in the incoming layer vector.
 *
 * @tparam TLayer The layer type.
 * @param _vIncomingLayer The vector of incoming layers.
 * @param _type The layer type to count.
 * @return The count of layers of the specified type.
 */
constexpr uint32_t poolingInputs = countLayersOfType(_vIncomingLayer, Layer::Type::Pooling);

/**
 * Count the number of layers of a specific type in the outgoing layer vector.
 *
 * @tparam TLayer The layer type.
 * @param _vOutgoingLayer The vector of outgoing layers.
 * @param _type The layer type to count.
 * @return The count of layers of the specified type.
 */
constexpr uint32_t convolutionalOutputs = countLayersOfType(_vOutgoingLayer, Layer::Type::Convolutional);

/**
 * Count the number of layers of a specific type in the outgoing layer vector.
 *
 * @tparam TLayer The layer type.
 * @param _vOutgoingLayer The vector of outgoing layers.
 * @param _type The layer type to count.
 * @return The count of layers of the specified type.
 */
constexpr uint32_t fullyConnectedOutputs = countLayersOfType(_vOutgoingLayer, Layer::Type::FullyConnected);

/**
 * Count the number of layers of a specific type in the outgoing layer vector.
 *
 * @tparam TLayer The layer type.
 * @param _vOutgoingLayer The vector of outgoing layers.
 * @param _type The layer type to count.
 * @return The count of layers of the specified type.
 */
constexpr uint32_t poolingOutputs = countLayersOfType(_vOutgoingLayer, Layer::Type::Pooling);

    const auto kindToParallelization = [](Layer::Kind kind) -> Layer::Parallelization {
        const std::unordered_map<Layer::Kind, Layer::Parallelization> kindToParallelizationMap = {
            {Layer::Kind::Input, convolutionalOutputs > 0 ? Layer::Parallelization::Data : Layer::Parallelization::Model},
            {Layer::Kind::Output, convolutionalInputs > 0 ? Layer::Parallelization::Data : Layer::Parallelization::Model},
            {Layer::Kind::Hidden, getHiddenParallelization(convolutionalInputs, fullyConnectedOutputs)}
        };

        return kindToParallelizationMap.at(kind);
    };

    _parallelization = kindToParallelization(_kind);
}

uint32_t Layer::countLayersOfType(const std::vector<Layer*>& layers, Layer::Type type)
{
    uint32_t count = 0;
    for (const auto& layer : layers)
    {
        if (layer->_type == type)
            count++;
    }
    return count;
}

Layer::Parallelization Layer::getHiddenParallelization(uint32_t convolutionalInputs, uint32_t fullyConnectedOutputs)
{
    if (_type == Layer::Type::FullyConnected)
    {
        _bTransposeParallelization = (convolutionalOutputs > 0);
        return Layer::Parallelization::Model;
    }
    else if (_type == Layer::Type::Pooling)
    {
        if (convolutionalInputs > 0)
        {
            _bTransposeParallelization = (fullyConnectedOutputs > 0);
            return Layer::Parallelization::Data;
        }
        else
        {
            _bTransposeParallelization = (convolutionalOutputs > 0);
            return Layer::Parallelization::Model;
        }
    }
    else
    {
        _bTransposeParallelization = (fullyConnectedOutputs > 0);
        return Layer::Parallelization::Data;
    }
}

void Layer::RefreshState(NNNetwork* pNetwork, TrainingMode trainingMode, bool validate)
{
    if (_bDirty)
    {
        _bFastSparse = false;
        if ((_kind == Input) && (_pDataSet != nullptr) && (_bSparse))
        {
            if (_pDataSet->_sparseDensity > 0.1)
            {
                if (getGpu()._id == 0)
                    printf("Layer::RefreshState: Sparse density per (%.2f) is too high to use fast sparse kernels on input layer %s\n", _pDataSet->_sparseDensity, _name.c_str());
            }
            else
            {
                _bFastSparse = true;
            }
        }

        if (getGpu()._numprocs > 1)
            RefreshParallelization();

        Allocate(validate);

        if (_bBatchNormalization)
        {
            if (trainingMode != TrainingMode::SGD)
            {
                _pbScaleVelocityBN = _bBatchNormalization ? std::make_unique<GpuBuffer<Float>>(_localStride) : nullptr;
                _pbBiasVelocityBN = _bBatchNormalization ? std::make_unique<GpuBuffer<Float>>(_localStride) : nullptr;

                if ((trainingMode == TrainingMode::AdaDelta) || (trainingMode == TrainingMode::Adam))
                {
                    _pbScaleGradientVelocityBN = _bBatchNormalization ? std::make_unique<GpuBuffer<Float>>(_localStride) : nullptr;
                    _pbBiasGradientVelocityBN = _bBatchNormalization ? std::make_unique<GpuBuffer<Float>>(_localStride) : nullptr;
                }
                else
                {
                    _pbScaleGradientVelocityBN.reset();
                    _pbBiasGradientVelocityBN.reset();
                }
            }
            else
            {
                _pbScaleVelocityBN.reset();
                _pbBiasVelocityBN.reset();
                _pbScaleGradientVelocityBN.reset();
                _pbBiasGradientVelocityBN.reset();
            }
        }

        if ((_kind != Hidden) && (_pDataSet != nullptr))
        {
            if (_parallelization == Layer::Parallelization::Model)
            {
                _pDataSet->Shard(DataSetEnums::Model);
            }
            else if (_parallelization == Layer::Parallelization::Data)
            {
                _pDataSet->Shard(DataSetEnums::Data);
            }
        }
        _bDirty = false;
    }

    if ((_kind == Input) && _pDataSet)
        _pDataSet->SetDenoising(_bDenoising);

    if ((_type == Layer::Type::Pooling) && (_poolingFunction == PoolingFunction::LRN))
    {
        cudnnStatus_t status = cudnnSetLRNDescriptor(_LRNDescriptor,
                                                    pNetwork->_LRN_n,
                                                    pNetwork->_LRN_alpha,
                                                    pNetwork->_LRN_beta,
                                                    pNetwork->_LRN_k);
        CUDNNERROR(status, "Layer::RefreshState: unable to set LRN descriptor");
    }
}

void Layer::ClearUpdates()
{
    _unitUpdateCount = 0;
    _deltaUpdateCount = 0;
    _bnCalls = 0;
}

void Layer::LoadPredictionBatch(uint32_t position, uint32_t batch)
{
    if (_kind == Input)
    {
        if (!_bSparse)
        {
            _pDataSet->LoadInputUnit(position, batch, _localStride, _pbUnit->_pDevData);
        }
        else if (!_bFastSparse)
        {
            _pDataSet->LoadSparseInputUnit(position, batch, _localStride, _pbUnit->_pDevData);
        }
    }
}

void Layer::LoadTrainingBatch(uint32_t position, uint32_t batch)
{
    if (_kind == Input)
    {
        if (_bSparse)
        {
            if (_bFastSparse)
            {
                if (_bDenoising)
                {
                    _pDataSet->CalculateSparseTransposedDenoisedMatrix(position, batch, this);
                }
                else
                {
                    _pDataSet->CalculateSparseTransposedMatrix(position, batch, this);
                }
            }
            else
            {
                if (_bDenoising)
                {
                    _pDataSet->LoadSparseDenoisedInputUnit(position, batch, _localStride, _pbUnit->_pDevData);
                }
                else
                {
                    _pDataSet->LoadSparseInputUnit(position, batch, _localStride, _pbUnit->_pDevData);
                }
            }
        }
        else
        {
            _pDataSet->LoadInputUnit(position, batch, _localStride, _pbUnit->_pDevData);

            if (_pDropout > 0.0)
                CalculateDropout(batch);
        }
    }
}

void Layer::LoadValidationBatch(uint32_t position, uint32_t batch)
{
    if (_kind == Input)
    {
        if (_bSparse)
        {
            _pDataSet->LoadSparseInputUnit(position, batch, _localStride, _pbUnit->_pDevData);
            _pDataSet->CalculateSparseTransposedMatrix(position, batch, this);
        }
        else
        {
            _pDataSet->LoadInputUnit(position, batch, _localStride, _pbUnit->_pDevData);
        }
    }
}

void Layer::GenerateDenoisingData()
{
    if (_pDataSet)
        _pDataSet->GenerateDenoisingData();
}

void Layer::ForwardPropagate(uint32_t position, uint32_t batch, bool bTraining)
{
    switch (_type)
    {
        case FullyConnected:
            ForwardPropagateFullyConnected(position, batch, bTraining);
            break;

        case Convolutional:
            ForwardPropagateConvolutional(position, batch, bTraining);
            break;

        case Pooling:
            ForwardPropagatePooling(position, batch, bTraining);
            break;
    }
}
    
    
void Layer::ForwardPropagateFullyConnected(uint32_t position, uint32_t batch, bool bTraining)
{    
    if (std::getGpu()._numprocs == 1)
    {
        if (_kind != Input)
        {         
            switch (_vIncomingLayer.size())
            {
                case 0:
                    std::cudaMemset(GetIncomingUnitBuffer(), 0, _stride * batch * sizeof(Float));
                    break;
                    
                case 1:
                    kClearUnit(GetIncomingUnitBuffer(), _vIncomingWeight[0]->_pbBias->_pDevData, _stride, batch);
                    break; 
                    
                case 2:
                    kClearDualSourceUnit(GetIncomingUnitBuffer(), _vIncomingWeight[0]->_pbBias->_pDevData, 
                                                                  _vIncomingWeight[1]->_pbBias->_pDevData, 
                                        _stride, batch);
                    break;                   
                    
                case 3:
                    kClearTripleSourceUnit(GetIncomingUnitBuffer(), _vIncomingWeight[0]->_pbBias->_pDevData, 
                                                                    _vIncomingWeight[1]->_pbBias->_pDevData, 
                                                                    _vIncomingWeight[2]->_pbBias->_pDevData, 
                                        _stride, batch);
                    break;      

                case 4:
                    kClearQuadSourceUnit(GetIncomingUnitBuffer(), _vIncomingWeight[0]->_pbBias->_pDevData, 
                                                                  _vIncomingWeight[1]->_pbBias->_pDevData, 
                                                                  _vIncomingWeight[2]->_pbBias->_pDevData, 
                                                                  _vIncomingWeight[3]->_pbBias->_pDevData, 
                                        _stride, batch);
                    break;                  
                    
                default:
                    if (std::getGpu()._id == 0)
                        printf("Layer::ForwardPropagate: Too many input layers for network layer %s\n", _name.c_str());          
                    std::getGpu().Shutdown();
                    std::exit(-1);
                    break; 
            }
        
        
            const Float sgemm_beta                = (Float)1.0;
            for (uint32_t i = 0; i < _vIncomingLayer.size(); i++)
            {
                if (_vIncomingLayer[i]->_bFastSparse)
                {
                    Float* pWeight                = _vIncomingWeight[i]->_bShared ? 
                                                      _vIncomingWeight[i]->_pSharedWeight->_pbWeight->_pDevData : 
                                                      _vIncomingWeight[i]->_pbWeight->_pDevData;
                    if (bTraining && _vIncomingLayer[i]->_bDenoising)
                        _vIncomingLayer[i]->_pDataSet->CalculateSparseDenoisedZ(position, batch, _stride, pWeight, GetIncomingUnitBuffer(), sgemm_beta);  
                    else
                        _vIncomingLayer[i]->_pDataSet->CalculateSparseZ(position, batch, _stride, pWeight, GetIncomingUnitBuffer(), sgemm_beta);
                }
                else      
                {
                    const Float sgemm_alpha       = (Float)1.0;
                    cublasStatus_t cstatus;
                    Float* pA                     = _vIncomingLayer[i]->GetUnitBuffer();
                    Float* pB                     = _vIncomingWeight[i]->_bShared ? 
                                                      _vIncomingWeight[i]->_pSharedWeight->_pbWeight->_pDevData : 
                                                      _vIncomingWeight[i]->_pbWeight->_pDevData;
                    Float* pC                     = GetIncomingUnitBuffer();
                    int m                           = batch;
                    int n                           = _localStride;
                    int k                           = _vIncomingLayer[i]->_stride;
                    int lda                         = _vIncomingWeight[i]->_bTransposed ? k : n;
                    int ldb                         = k;
                    int ldc                         = n;

                    cstatus                         =
                                                    cublasSgemm(std::std::getGpu()._cuBLASHandle, 
                                                    _vIncomingWeight[i]->_bTransposed ? CUBLAS_OP_T : CUBLAS_OP_N,
                                                    CUBLAS_OP_N,
                                                    n,
                                                    m,
                                                    k,
                                                    &sgemm_alpha,
                                                    pB,
                                                    lda,
                                                    pA,
                                                    ldb,
                                                    &sgemm_beta,
                                                    pC,
                                                    ldc);  

                    if (cstatus != CUBLAS_STATUS_SUCCESS)
                    {
                        if (std::std::getGpu()._id == 0)
                            printf("Layer::ForwardPropagate: SGEMM failure, aborting, status %d.\n", cstatus);
                        std::getGpu().Shutdown();
                        std::exit(-1);
                    }
                }
            }

            for (auto l : _vIncomingSkip)
            {
                kAddBuffers(GetIncomingUnitBuffer(), l->GetUnitBuffer(), batch * _stride);
            }
            
            if (_bBatchNormalization)
            {
                float alpha = 1;
                float beta = 0;
                cudnnStatus_t cudnnStatus;
                cudnnStatus = cudnnSetTensor4dDescriptor(_tensorDescriptorBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, _Nz, _Ny, _localStride);
                CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateFullyConnected: unable to create _tensorDescriptorBN");        
                cudnnStatus = cudnnSetTensor4dDescriptor(_scaleBiasMeanVarDescBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, _Nz, _Ny, _localStride);
                CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateFullyConnected: unable to create _scaleBiasMeanVarDescBN");        
                if (bTraining) {
                    cudnnStatus = cudnnBatchNormalizationForwardTraining(
                            std::std::getGpu()._cuDNNHandle,
                            CUDNN_BATCHNORM_PER_ACTIVATION,
                            &alpha,
                            &beta,
                            _tensorDescriptorBN,
                            GetIncomingUnitBuffer(),
                            _tensorDescriptorBN,
                            GetUnitBuffer(),
                            _scaleBiasMeanVarDescBN,
                            _pbScaleBN->_pDevData,
                            _pbBiasBN->_pDevData,
                            1.0/(_bnCalls + 1), 
                            _pbRunningMeanBN->_pDevData,
                            _pbRunningVarianceBN->_pDevData,
                            CUDNN_BN_MIN_EPSILON,
                            _pbSaveMeanBN->_pDevData,
                            _pbSaveInvVarianceBN->_pDevData);
                    CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateFullyConnected: cudnnBatchNormalizationForwardTraining Failed");
                    ++_bnCalls;
                } else {
                    cudnnStatus = cudnnBatchNormalizationForwardInference(
                            std::std::getGpu()._cuDNNHandle,
                            CUDNN_BATCHNORM_PER_ACTIVATION,
                            &alpha,
                            &beta,
                            _tensorDescriptorBN,
                            GetIncomingUnitBuffer(),
                            _tensorDescriptorBN,
                            GetUnitBuffer(),
                            _scaleBiasMeanVarDescBN,
                            _pbScaleBN->_pDevData,
                            _pbBiasBN->_pDevData,
                            _pbRunningMeanBN->_pDevData,
                            _pbRunningVarianceBN->_pDevData,
                            CUDNN_BN_MIN_EPSILON);
                    CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateFullyConnected: cudnnBatchNormalizationForwardInference Failed");
                }
            }
           
            CalculateActivation(batch);
            
            if (bTraining && (_pDropout > (Float)0.0))
                CalculateDropout(batch);             
       
#if 0
        std::string fname = "activation_" + _name;
        Dump(fname, _pbUnit->_pDevData);
#endif              
        }       
    }
    else
    {
        if (_kind != Input)
        {              
            if (_vIncomingLargerLayer.size() > 0)
            {
                Float sgemm_beta                  = (Float)0.0;
                for (uint32_t i = 0; i < _vIncomingLargerLayer.size(); i++)
                {
                    Layer* pInputLayer            = _vIncomingLargerLayer[i];
                    Float* pWeight                = _vIncomingLargerWeight[i]->_bShared ? 
                                                      _vIncomingLargerWeight[i]->_pSharedWeight->_pbWeight->_pDevData : 
                                                      _vIncomingLargerWeight[i]->_pbWeight->_pDevData;                                           

                    if (pInputLayer->_bFastSparse)
                    {
                        if (bTraining && pInputLayer->_bDenoising)
                            pInputLayer->_pDataSet->CalculateSparseDenoisedZ(position, batch, _stride, pWeight, std::std::getGpu()._pNetwork->GetP2PSendBuffer(), sgemm_beta);  
                        else
                            pInputLayer->_pDataSet->CalculateSparseZ(position, batch, _stride, pWeight, std::std::getGpu()._pNetwork->GetP2PSendBuffer(), sgemm_beta);  
                    }
                    else
                    {
                
                        const Float sgemm_alpha   = (Float)1.0;

                        Float* pA                 = pWeight;
                        Float* pB                 = pInputLayer->GetUnitBuffer();
                        Float* pC                 = std::std::getGpu()._pNetwork->GetP2PSendBuffer();
                        int m                       = _stride;
                        int n                       = batch;
                        int k                       = pInputLayer->_localStride;
                        int lda                     = _stride;
                        int ldb                     = pInputLayer->_localStride;
                        int ldc                     = _stride;

                        cublasStatus_t cstatus      =
                                                    cublasSgemm(std::std::getGpu()._cuBLASHandle, 
                                                    CUBLAS_OP_N,
                                                    CUBLAS_OP_N,
                                                    m,
                                                    n,
                                                    k,
                                                    &sgemm_alpha,
                                                    pA,
                                                    lda,
                                                    pB,
                                                    ldb,
                                                    &sgemm_beta,
                                                    pC,
                                                    ldc);  

                        if (cstatus != CUBLAS_STATUS_SUCCESS)
                        {
                            if (std::std::getGpu()._id == 0)
                                printf("Layer::ForwardPropagate: SGEMM failure, aborting, status %d.\n", cstatus);
                            std::std::getGpu().Shutdown();
                            std::exit(-1);
                        }                                     
                    }
                    
                    sgemm_beta                      = (Float)1.0;
                }

                Reduce(batch, _stride, GetIncomingUnitBuffer(), _localStride, _unitUpdateCount);
                _unitUpdateCount++;
            }
            
            for (auto l : _vIncomingSkip)
            {
                kAddBuffers(GetIncomingUnitBuffer(), l->GetUnitBuffer(), batch * _localStride);
            }            
                   
            switch (_vIncomingLayer.size())
            {
                case 0:
                    break;
                
                case 1:
                    kAddBias(GetIncomingUnitBuffer(), _vIncomingWeight[0]->_pbBias->_pDevData, _localStride, batch);
                    break; 
                        
                case 2:
                    kAddDualBias(GetIncomingUnitBuffer(), _vIncomingWeight[0]->_pbBias->_pDevData, 
                                                          _vIncomingWeight[1]->_pbBias->_pDevData, _localStride, batch);
                    break;                   
                        
                case 3:
                    kAddTripleBias(GetIncomingUnitBuffer(), _vIncomingWeight[0]->_pbBias->_pDevData, 
                                                            _vIncomingWeight[1]->_pbBias->_pDevData, 
                                                            _vIncomingWeight[2]->_pbBias->_pDevData, _localStride, batch);
                    break;      

                case 4:
                    kAddQuadBias(GetIncomingUnitBuffer(), _vIncomingWeight[0]->_pbBias->_pDevData, 
                                                          _vIncomingWeight[1]->_pbBias->_pDevData, 
                                                          _vIncomingWeight[2]->_pbBias->_pDevData, 
                                                          _vIncomingWeight[3]->_pbBias->_pDevData, _localStride, batch);
                    break;                  
                        
                default:
                    if (std::std::getGpu()._id == 0)
                        printf("Layer::ForwardPropagate: Too many input layers for network layer %s\n", _name.c_str());
                    std::std::getGpu().Shutdown();
                    std::exit(-1);
                    break; 
            }    
            
            if (_bBatchNormalization)
            {
                float alpha = 1;
                float beta = 0;
                cudnnStatus_t cudnnStatus;
                cudnnStatus = cudnnSetTensor4dDescriptor(_tensorDescriptorBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, _Nz, _Ny, _localStride);
                CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateFullyConnected: unable to create _tensorDescriptorBN");        
                cudnnStatus = cudnnSetTensor4dDescriptor(_scaleBiasMeanVarDescBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, _Nz, _Ny, _localStride);
                CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateFullyConnected: unable to create _scaleBiasMeanVarDescBN");        
                if (bTraining) {
                    cudnnStatus = cudnnBatchNormalizationForwardTraining(
                            std::std::getGpu()._cuDNNHandle,
                            CUDNN_BATCHNORM_PER_ACTIVATION,
                            &alpha,
                            &beta,
                            _tensorDescriptorBN,
                            GetIncomingUnitBuffer(),
                            _tensorDescriptorBN,
                            GetUnitBuffer(),
                            _scaleBiasMeanVarDescBN,
                            _pbScaleBN->_pDevData,
                            _pbBiasBN->_pDevData,
                            1.0/(_bnCalls + 1), 
                            _pbRunningMeanBN->_pDevData,
                            _pbRunningVarianceBN->_pDevData,
                            CUDNN_BN_MIN_EPSILON,
                            _pbSaveMeanBN->_pDevData,
                            _pbSaveInvVarianceBN->_pDevData);
                    CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateFullyConnected: cudnnBatchNormalizationForwardTraining Failed");
                } else {
                    cudnnStatus = cudnnBatchNormalizationForwardInference(
                            std::std::getGpu()._cuDNNHandle,
                            CUDNN_BATCHNORM_PER_ACTIVATION,
                            &alpha,
                            &beta,
                            _tensorDescriptorBN,
                            GetIncomingUnitBuffer(),
                            _tensorDescriptorBN,
                            GetUnitBuffer(),
                            _scaleBiasMeanVarDescBN,
                            _pbScaleBN->_pDevData,
                            _pbBiasBN->_pDevData,
                            _pbRunningMeanBN->_pDevData,
                            _pbRunningVarianceBN->_pDevData,
                            CUDNN_BN_MIN_EPSILON);
                    CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateFullyConnected: cudnnBatchNormalizationForwardInference Failed");
                }
            }
                                      
            CalculateActivation(batch);   
            
            if (bTraining && (_pDropout > (Float)0.0))
                CalculateDropout(batch);  
        }
        
#if 0
        std::string fname = "activation_" + _name;
        Dump(fname, _pbUnit->_pDevData);
#endif                                      
        if (_vOutgoingLargerLayer.size() > 0)
        {  
        
            if (_bFastSparse)
            {
                for (uint32_t i = 0; i < _vOutgoingLargerLayer.size(); i++)
                {
                    Layer* pOutputLayer       = _vOutgoingLargerLayer[i];
                    Float* pWeight            = _vOutgoingLargerWeight[i]->_bShared ? 
                                                  _vOutgoingLargerWeight[i]->_pSharedWeight->_pbWeight->_pDevData : 
                                                  _vOutgoingLargerWeight[i]->_pbWeight->_pDevData;
                    const Float sgemm_beta    = (pOutputLayer->_unitUpdateCount == 0) ? (Float)0.0 : (Float)1.0;
                    
                    if (bTraining && _bDenoising)
                        _pDataSet->CalculateSparseDenoisedZ(position, batch, pOutputLayer->_localStride, pWeight, pOutputLayer->GetIncomingUnitBuffer(), sgemm_beta);  
                    else
                        _pDataSet->CalculateSparseZ(position, batch, pOutputLayer->_localStride, pWeight, pOutputLayer->GetIncomingUnitBuffer(), sgemm_beta);
                }
            }
            else
            {
        
                Gather(batch, _stride, GetUnitBuffer(), _localStride);

                for (uint32_t i = 0; i < _vOutgoingLargerLayer.size(); i++)
                {
                    Layer* pOutputLayer       = _vOutgoingLargerLayer[i];
                    Weight* pWeight           = _vOutgoingLargerWeight[i];     
                    Weight* pSrcWeight        = pWeight->_bShared ? pWeight->_pSharedWeight : pWeight;
                    Float* pA                 = pSrcWeight->_pbWeight->_pDevData;
                    Float* pB                 = std::std::getGpu()._pNetwork->GetP2PSendBuffer();
                    Float* pC                 = pOutputLayer->GetIncomingUnitBuffer();
                    
                    int m                       = pOutputLayer->_localStride;
                    int n                       = batch;
                    int k                       = _stride;
                    int lda                     = pOutputLayer->_localStride;
                    int ldb                     = _stride;
                    int ldc                     = pOutputLayer->_localStride;
                    const Float sgemm_alpha   = 1.0;
                    const Float sgemm_beta    = (pOutputLayer->_unitUpdateCount == 0) ? (Float)0.0 : (Float)1.0;

                    cublasStatus_t cstatus      =
                                                cublasSgemm(std::std::getGpu()._cuBLASHandle, 
                                                CUBLAS_OP_T,
                                                CUBLAS_OP_N,
                                                m,
                                                n,
                                                k,
                                                &sgemm_alpha,
                                                pA,
                                                lda,
                                                pB,
                                                ldb,
                                                &sgemm_beta,
                                                pC,
                                                ldc);  

                    if (cstatus != CUBLAS_STATUS_SUCCESS)
                    {
                        if (std::std::getGpu()._id == 0)
                            printf("Layer::ForwardPropagate: SGEMM failure, aborting, status %d.\n", cstatus);
                        std::std::getGpu().Shutdown();
                        std::exit(-1);
                    }
                }
                
                pWeights                             = pWeights->_pNext;
                pbiases                             = pbiases->_pNext;        
            }
            _unitUpdateCount++;
        }
        
        if (_bFastSparse)
        {
            Gather(batch, _stride, GetUnitBuffer(), _localStride);
        }       
    }
}


void Layer::ForwardPropagateConvolutional(uint32_t position, uint32_t batch, bool bTraining)
{
    if (_kind != Layer::Kind::Input)
    {
        if (getGpu()._numprocs == 1)
        {
            Float alpha = static_cast<Float>(1.0);
            Float beta = static_cast<Float>(0.0);
            for (uint32_t i = 0; i < _vIncomingLayer.size(); i++)
            {
                Layer* pLayer = _vIncomingLayer[i];
                Weight* pWeight = _vIncomingWeight[i]->_bShared ?
                    _vIncomingWeight[i]->_pSharedWeight :
                    _vIncomingWeight[i];

                cudnnStatus_t cudnnStatus = cudnnConvolutionForward(
                    getGpu()._cuDNNHandle,
                    &alpha,
                    pLayer->getTensorDescriptor(batch),
                    pLayer->GetUnitBuffer(),
                    pWeight->_convFilterDesc,
                    pWeight->_pbWeight->_pDevData,
                    pWeight->_convDesc,
                    pWeight->_convFWAlgo,
                    getGpu()._pNetwork->_pbCUDNNWorkspace->_pDevData,
                    getGpu()._pNetwork->_CUDNNWorkspaceSize,
                    &beta,
                    getTensorDescriptor(batch),
                    GetIncomingUnitBuffer());
                CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateConvolutional: cudnnConvolutionForward Failed");

                cudnnStatus = cudnnAddTensor(
                    getGpu()._cuDNNHandle,
                    &alpha,
                    _vIncomingWeight[i]->_convBiasTensor,
                    _vIncomingWeight[i]->_pbBias->_pDevData,
                    &alpha,
                    getTensorDescriptor(batch),
                    GetIncomingUnitBuffer());
                CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateConvolutional: cudnnAddTensor Failed");
                beta = static_cast<Float>(1.0);
            }

            for (auto l : _vIncomingSkip)
            {
                kAddBuffers(GetIncomingUnitBuffer(), l->GetUnitBuffer(), batch * _stride);
            }

            if (_bBatchNormalization)
            {
                float alpha = 1.0f;
                float beta = 0.0f;
                cudnnStatus_t cudnnStatus;
                cudnnStatus = cudnnSetTensor4dDescriptor(
                    _tensorDescriptorBN,
                    CUDNN_TENSOR_NCHW,
                    CUDNN_DATA_FLOAT,
                    batch,
                    _Nz,
                    _Ny,
                    _Nx);
                CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateConvolutional: unable to create _tensorDescriptorBN");
                cudnnStatus = cudnnSetTensor4dDescriptor(
                    _scaleBiasMeanVarDescBN,
                    CUDNN_TENSOR_NCHW,
                    CUDNN_DATA_FLOAT,
                    1,
                    _Nz,
                    1,
                    1);
                CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateConvolutional: unable to create _scaleBiasMeanVarDescBN");
                if (bTraining)
                {
                    cudnnStatus = cudnnBatchNormalizationForwardTraining(
                        getGpu()._cuDNNHandle,
                        CUDNN_BATCHNORM_SPATIAL,
                        &alpha,
                        &beta,
                        _tensorDescriptorBN,
                        GetIncomingUnitBuffer(),
                        _tensorDescriptorBN,
                        GetUnitBuffer(),
                        _scaleBiasMeanVarDescBN,
                        _pbScaleBN->_pDevData,
                        _pbBiasBN->_pDevData,
                        1.0 / (_bnCalls + 1),
                        _pbRunningMeanBN->_pDevData,
                        _pbRunningVarianceBN->_pDevData,
                        CUDNN_BN_MIN_EPSILON,
                        _pbSaveMeanBN->_pDevData,
                        _pbSaveInvVarianceBN->_pDevData);
                    CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateConvolutional: cudnnBatchNormalizationForwardTraining Failed");
                    ++_bnCalls;
                }
                else
                {
                    cudnnStatus = cudnnBatchNormalizationForwardInference(
                        getGpu()._cuDNNHandle,
                        CUDNN_BATCHNORM_SPATIAL,
                        &alpha,
                        &beta,
                        _tensorDescriptorBN,
                        GetIncomingUnitBuffer(),
                        _tensorDescriptorBN,
                        GetUnitBuffer(),
                        _scaleBiasMeanVarDescBN,
                        _pbScaleBN->_pDevData,
                        _pbBiasBN->_pDevData,
                        _pbRunningMeanBN->_pDevData,
                        _pbRunningVarianceBN->_pDevData,
                        CUDNN_BN_MIN_EPSILON);
                    CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateConvolutional: cudnnBatchNormalizationForwardInference Failed");
                }
            }

            CalculateActivation(batch);

            if (bTraining && (_pDropout > static_cast<Float>(0.0)))
                CalculateDropout(batch);
        }
    }
}

void Layer::ForwardPropagatePooling(uint32_t position, uint32_t batch, bool bTraining)
{
    if (_kind != Layer::Kind::Input)
    {
        Float alpha = static_cast<Float>(1.0);
        Float beta = static_cast<Float>(0.0);
        for (int i = 0; i < _vIncomingLayer.size(); i++)
        {
            Layer* pLayer = _vIncomingLayer[i];
            cudnnStatus_t cudnnStatus;
            switch (_poolingFunction)
            {
                case PoolingFunction::Max:
                case PoolingFunction::Average:
                    cudnnStatus = cudnnPoolingForward(
                        getGpu()._cuDNNHandle,
                        _poolingDescriptor,
                        &alpha,
                        pLayer->getTensorDescriptor(batch),
                        pLayer->GetUnitBuffer(),
                        &beta,
                        getTensorDescriptor(batch),
                        GetIncomingUnitBuffer());
                    CUDNNERROR(cudnnStatus, "Layer::ForwardPropagatePooling: cudnnPoolingForward Failed");
                    break;

                case PoolingFunction::LRN:
                    cudnnStatus = cudnnLRNCrossChannelForward(
                        getGpu()._cuDNNHandle,
                        _LRNDescriptor,
                        CUDNN_LRN_CROSS_CHANNEL_DIM1,
                        &alpha,
                        pLayer->getTensorDescriptor(batch),
                        pLayer->GetUnitBuffer(),
                        &beta,
                        getTensorDescriptor(batch),
                        GetIncomingUnitBuffer());
                    CUDNNERROR(cudnnStatus, "Layer::ForwardPropagatePooling: cudnnLRNCrossChannelForward Failed");
                    break;

                case PoolingFunction::Cosine:
                    if (i >= 1)
                    {
                        Layer* p0Layer = _vIncomingLayer[0];
                        uint32_t offset = i - 1;
                        kCalculateCosine(
                            p0Layer->GetUnitBuffer(),
                            pLayer->GetUnitBuffer(),
                            batch,
                            pLayer->_localStride,
                            GetIncomingUnitBuffer() + offset,
                            _pbBuffer1->_pDevData + offset,
                            _pbBuffer2->_pDevData + offset,
                            _localStride);
                    }
                    break;

                case PoolingFunction::DotProduct:
                    if (i >= 1)
                    {
                        Layer* p0Layer = _vIncomingLayer[0];
                        uint32_t offset = i - 1;
                        kCalculateDotProduct(
                            p0Layer->GetUnitBuffer(),
                            pLayer->GetUnitBuffer(),
                            batch,
                            pLayer->_localStride,
                            GetIncomingUnitBuffer() + offset,
                            _localStride);
                    }
                    break;

                case PoolingFunction::Maxout:
                    if (beta != static_cast<Float>(0.0))
                    {
                        kCalculateMaxout(
                            pLayer->GetUnitBuffer(),
                            batch * _localStride,
                            GetIncomingUnitBuffer());
                    }
                    else
                    {
                        cudaError_t status = cudaMemcpy(
                            GetIncomingUnitBuffer(),
                            pLayer->GetUnitBuffer(),
                            batch * _localStride * sizeof(Float),
                            cudaMemcpyDefault);
                        RTERROR(status, "Layer::ForwardPropagate: Error calling cudaMemcpy for maxout pooling.");
                    }
                    break;
            }
            beta = static_cast<Float>(1.0);
        }

        for (auto l : _vIncomingSkip)
        {
            kAddBuffers(GetIncomingUnitBuffer(), l->GetUnitBuffer(), batch * _stride);
        }
    }
}

void Layer::CalculateActivation(uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(batch) * static_cast<uint64_t>(_localStride);
    switch (_activation)
    {
        case Sigmoid:
            kCalculateSigmoidActivation(GetUnitBuffer(), size);
            break;

        case Tanh:
            kCalculateTanhActivation(GetUnitBuffer(), size);
            break;

        case RectifiedLinear:
            kCalculateRELUActivation(GetUnitBuffer(), size);
            break;

        case LeakyRectifiedLinear:
            kCalculateLRELUActivation(GetUnitBuffer(), size, _RELUSlope);
            break;

        case ExponentialLinear:
            kCalculateELUActivation(GetUnitBuffer(), size, _ELUAlpha);
            break;

        case ScaledExponentialLinear:
            kCalculateSELUActivation(GetUnitBuffer(), size, _ELUAlpha, _SELULambda);
            break;

        case SoftMax:
            kCalculateSoftMaxActivation(GetUnitBuffer(), batch, _localStride);
            break;

        case Linear:
            break;
    }
}

#include <numbers>

void Layer::CalculateDropout(uint32_t batch)
{
    Float lambda = (_activation == ScaledExponentialLinear) ? _SELULambda : (Float)1.0;
    Float alpha = -lambda * _ELUAlpha;
    Float q = (Float)1.0 - _pDropout;
    Float a = 1.0 / std::sqrt(q + alpha * alpha * _pDropout * q);
    Float b = -a * _pDropout * alpha;
    Float target = (_activation == Sigmoid) ? 0.5 : 0.0;

    if (_activation == ExponentialLinear || _activation == ScaledExponentialLinear)
    {
        kCalculateScaledBiasedDropout(GetUnitBuffer(), _pbDropout->_pDevData, batch, _localStride, _pDropout, alpha, a, b);
    }
    else
    {
        kCalculateDropout(GetUnitBuffer(), _pbDropout->_pDevData, batch, _localStride, _pDropout, target);
    }
}

#include <cstdlib>

Float Layer::CalculateError(uint32_t position, uint32_t batch, ErrorFunction ef)
{
    if (_kind != Output)
    {
        if (getGpu()._id == 0)
            std::printf("Layer::CalculateError: Attempt to calculate error on non-output layer %s.\n", _name.c_str());
        getGpu().Shutdown();
        std::exit(-1);
    }

    switch (ef)
    {
        case L1:
            return _pDataSet->CalculateL1Error(position, batch, _localStride, GetUnitBuffer());

        case L2:
            return _pDataSet->CalculateL2Error(position, batch, _localStride, GetUnitBuffer());
            
        case L2Hinge:
            return _pDataSet->CalculateL2HingeError(position, batch, _localStride, GetUnitBuffer());            

        case Hinge:
            return _pDataSet->CalculateHingeError(position, batch, _localStride, GetUnitBuffer());              

        case CrossEntropy:
            if (_activation == SoftMax)
                return _pDataSet->CalculateMultinomialCrossEntropyError(position, batch, _localStride, GetUnitBuffer());
            else
                return _pDataSet->CalculateCrossEntropyError(position, batch, _localStride, GetUnitBuffer());

        case ScaledMarginalCrossEntropy:
            if (_activation == SoftMax)
                return _pDataSet->CalculateMultinomialScaledMarginalCrossEntropyError(position, batch, _localStride, GetUnitBuffer());
            else        
                return _pDataSet->CalculateScaledMarginalCrossEntropyError(position, batch, _localStride, GetUnitBuffer());

        case DataScaledMarginalCrossEntropy:
            if (_activation == SoftMax)
            {
                std::cout << "unsupported combination of activation with cost function" << std::endl;
                getGpu().Shutdown();
                std::exit(-1);
            }
            else
            {
                return _pDataSet->CalculateDataScaledMarginalCrossEntropyError(position, batch, _localStride, GetUnitBuffer());
            }
    }
    
    return 0.0;
}

#include <cstdlib>

void Layer::CalculateOutputDelta(uint32_t position, uint32_t batch, ErrorFunction ef)
{
    if (_kind != Output)
    {
        if (getGpu()._id == 0)
            std::printf("Layer::CalculateOutputDelta: Attempt to calculate output delta on non-output layer %s.\n", _name.c_str());
        getGpu().Shutdown();
        std::exit(-1);
    }

    switch (ef)
    {
        case L1:
            _pDataSet->CalculateL1OutputDelta(_activation, position, batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer(), _RELUSlope, _ELUAlpha, _SELULambda);
            break;

        case CrossEntropy:
            _pDataSet->CalculateCrossEntropyOutputDelta(_activation, position, batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer());
            break;

        case ScaledMarginalCrossEntropy:
            _pDataSet->CalculateScaledMarginalCrossEntropyOutputDelta(_activation, position, batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer());
            break;

        case L2:
            _pDataSet->CalculateOutputDelta(_activation, position, batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer(), _RELUSlope, _ELUAlpha, _SELULambda);
            break;
            
        case L2Hinge:
            _pDataSet->CalculateL2HingeOutputDelta(_activation, position, batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer(), _RELUSlope, _ELUAlpha, _SELULambda);
            break;            

        case Hinge:
            _pDataSet->CalculateHingeOutputDelta(_activation, position, batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer());
            break;            

        case DataScaledMarginalCrossEntropy:
            _pDataSet->CalculateDataScaledMarginalCrossEntropyOutputDelta(_activation, position, batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer());
            break;

        default:
            std::cout << "Unsupported cost function" << std::endl;
            std::exit(2);
    }
    
    if (_deltaNorm > 0.0)
    {
        if (getGpu()._numprocs == 1)
            kNormalizeDeltas(_deltaNorm, batch, _localStride, GetDeltaBuffer());
        else
        {
            Float* pMagnitude = getGpu()._pNetwork->GetScratchBuffer(batch);
            kCalculateDeltaMagnitudes(batch, _localStride, GetDeltaBuffer(), pMagnitude);
            getGpu()._pNetwork->P2P_Allreduce(pMagnitude, batch);
            kNormalizeDeltaMagnitudes(_deltaNorm, batch, _localStride, GetDeltaBuffer(), pMagnitude);            
        }
    }
}


#include <cstdlib> // for std::exit

void Layer::BackPropagate(uint32_t position, uint32_t batch)
{
    switch (_type)
    {
        case FullyConnected:
            BackPropagateFullyConnected(position, batch);
            break;
            
        case Convolutional:
            BackPropagateConvolutional(position, batch);
            break;
            
        case Pooling:
            BackPropagatePooling(position, batch);
            break;                        
    }
}

void Layer::BackPropagateConvolutional(uint32_t position, uint32_t batch)
{
    if (getGpu()._numprocs == 1)
    {
        if (_kind == Hidden)
        {
            if (_bSparse && getGpu()._data._bSparsenessPenalty)
            {
                Float p = (_sparsenessPenalty_p > 0.0) ? _sparsenessPenalty_p : getGpu()._pNetwork->_sparsenessPenalty_p;
                Float beta = (_sparsenessPenalty_beta > 0.0) ? _sparsenessPenalty_beta : getGpu()._pNetwork->_sparsenessPenalty_beta;
                kCalculateSparsenessPenalty(batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer(), p, beta);
            }   

            Float scale = 1.0 / (1.0 - _pDropout);
            kCalculateHadamardProduct(_activation, batch * _localStride, scale, GetUnitBuffer(), GetIncomingDeltaBuffer(), _RELUSlope, _ELUAlpha, _SELULambda);
            
            if (_deltaNorm > 0.0)
            {            
                kNormalizeDeltas(_deltaNorm, batch, _localStride, GetIncomingDeltaBuffer());
            }
            
            if (_bBatchNormalization)
            {
                cudnnStatus_t cudnnStatus;
                cudnnStatus = cudnnSetTensor4dDescriptor(_tensorDescriptorBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, _Nz, _Ny, _Nx);
                CUDNNERROR(cudnnStatus, "Layer::BackPropagateConvolutional: unable to create _tensorDescriptorBN");        
                cudnnStatus = cudnnSetTensor4dDescriptor(_scaleBiasMeanVarDescBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, _Nz, 1, 1);
                CUDNNERROR(cudnnStatus, "Layer::BackPropagateConvolutional: unable to create _scaleBiasMeanVarDescBN");        
                float alpha = 1;
                float beta = 0;
                cudnnStatus = cudnnBatchNormalizationBackward(
                        getGpu()._cuDNNHandle,
                        CUDNN_BATCHNORM_SPATIAL,
                        &alpha,
                        &beta,
                        &alpha,
                        &beta,
                        _tensorDescriptorBN,
                        GetIncomingUnitBuffer(),
                        _tensorDescriptorBN,
                        GetIncomingDeltaBuffer(),
                        _tensorDescriptorBN,
                        GetDeltaBuffer(),
                        _scaleBiasMeanVarDescBN,
                        _pbScaleBN->_pDevData,
                        _pbScaleGradientBN->_pDevData,
                        _pbBiasGradientBN->_pDevData,
                        CUDNN_BN_MIN_EPSILON,
                        _pbSaveMeanBN->_pDevData,
                        _pbSaveInvVarianceBN->_pDevData);
                CUDNNERROR(cudnnStatus, "Layer:BackPropagateConvolutional cudnnBatchNormalizationBackward Failed");
            }
        }


        for (uint32_t i = 0; i < _vIncomingLayer.size(); i++)
        {
            Layer* pInputLayer = _vIncomingLayer[i];

            Weight* pWeight = _vIncomingWeight[i];     
            Weight* pSrcWeight = pWeight->_bShared ? pWeight->_pSharedWeight : pWeight;
            Float gradient_alpha = -(1.0 / (pSrcWeight->_sharingCount * batch));            

            cudnnStatus_t cudnnStatus;
            if (!pWeight->_bLocked)
            {
                Float beta = (pSrcWeight->_updateCount == 0) ? 0.0 : 1.0;
                cudnnStatus = cudnnConvolutionBackwardFilter(getGpu()._cuDNNHandle,
                                                             &gradient_alpha,
                                                             pInputLayer->getTensorDescriptor(batch),
                                                             pInputLayer->GetUnitBuffer(),
                                                             getTensorDescriptor(batch),
                                                             GetDeltaBuffer(),
                                                             pSrcWeight->_convDesc,
                                                             pSrcWeight->_convBWWeightAlgo,
                                                             getGpu()._pNetwork->_pbCUDNNWorkspace->_pDevData,
                                                             getGpu()._pNetwork->_CUDNNWorkspaceSize,
                                                             &beta,
                                                             pSrcWeight->_convFilterDesc,
                                                             pSrcWeight->_pbWeightGradient->_pDevData);
                CUDNNERROR(cudnnStatus, "Layer::BackPropagateConvolutional: cudnnConvolutionBackwardFilter Failed"); 
                
                beta = 0.0;
                cudnnStatus = cudnnConvolutionBackwardBias(getGpu()._cuDNNHandle,
                                                           &gradient_alpha,
                                                           getTensorDescriptor(batch),
                                                           GetDeltaBuffer(),
                                                           &beta,
                                                           pWeight->_convBiasTensor,
                                                           pWeight->_pbBiasGradient->_pDevData);                
                

                pSrcWeight->_updateCount++;
            }
     
            if (pInputLayer->_kind != Input)
            {
                Float delta_alpha = 1.0;                
                Float beta = (pInputLayer->_deltaUpdateCount == 0) ? 0.0 : 1.0;
                cudnnStatus = cudnnConvolutionBackwardData(getGpu()._cuDNNHandle,
                                                           &delta_alpha,
                                                           pSrcWeight->_convFilterDesc,
                                                           pSrcWeight->_pbWeight->_pDevData,
                                                           getTensorDescriptor(batch),
                                                           GetDeltaBuffer(),
                                                           pSrcWeight->_convDesc, 
                                                           pSrcWeight->_convBWDeltaAlgo,
                                                           getGpu()._pNetwork->_pbCUDNNWorkspace->_pDevData,
                                                           getGpu()._pNetwork->_CUDNNWorkspaceSize,
                                                           &beta,
                                                           pInputLayer->getTensorDescriptor(batch),
                                                           pInputLayer->GetIncomingDeltaBuffer());
                CUDNNERROR(cudnnStatus, "Layer::BackPropagateConvolutional: cudnnConvolutionBackwardData Failed");

                pInputLayer->_deltaUpdateCount++; 
            }
        }    
        
        for (auto l : _vIncomingSkip)
        {
            if (l->_deltaUpdateCount > 0)
            {
                kAddBuffers(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), batch * _localStride);
            }
            else
            {
                cudaMemcpy(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), batch * _localStride * sizeof(Float), cudaMemcpyDefault);
            }
         
            l->_deltaUpdateCount++;
        }
    }
}

void Layer::BackPropagatePooling(uint32_t position, uint32_t batch)
{
    Float pooling_alpha = 1.0;

    for (auto* pInputLayer : _vIncomingLayer)
    {
        if (pInputLayer->_kind == Input)
            continue;

        cudnnStatus_t cudnnStatus;
        Float beta = pInputLayer->_deltaUpdateCount == 0 ? 0.0 : 1.0;

        switch (_poolingFunction)
        {
            case Max:
            case Average:
                cudnnStatus = cudnnPoolingBackward(
                    getGpu()._cuDNNHandle,
                    _poolingDescriptor,
                    &pooling_alpha,
                    getTensorDescriptor(batch),
                    GetUnitBuffer(),
                    getTensorDescriptor(batch),
                    GetDeltaBuffer(),
                    pInputLayer->getTensorDescriptor(batch),
                    pInputLayer->GetUnitBuffer(),
                    &beta,
                    pInputLayer->getTensorDescriptor(batch),
                    pInputLayer->GetIncomingDeltaBuffer()
                );
                CUDNNERROR(cudnnStatus, "Layer::BackPropagatePooling: cudnnPoolingBackward Failed");

                break;

            case LRN:
                cudnnStatus = cudnnLRNCrossChannelBackward(
                    getGpu()._cuDNNHandle,
                    _LRNDescriptor,
                    CUDNN_LRN_CROSS_CHANNEL_DIM1,
                    &pooling_alpha,
                    getTensorDescriptor(batch),
                    GetUnitBuffer(),
                    getTensorDescriptor(batch),
                    GetDeltaBuffer(),
                    pInputLayer->getTensorDescriptor(batch),
                    pInputLayer->GetUnitBuffer(),
                    &beta,
                    pInputLayer->getTensorDescriptor(batch),
                    pInputLayer->GetIncomingDeltaBuffer()
                );
                CUDNNERROR(cudnnStatus, "Layer::BackPropagatePooling: cudnnLRNCrossChannelBackward Failed");

                break;

            case Maxout:
                kCalculateMaxoutDelta(
                    GetUnitBuffer(),
                    GetDeltaBuffer(),
                    batch * _localStride,
                    beta,
                    pInputLayer->GetUnitBuffer(),
                    pInputLayer->GetIncomingDeltaBuffer()
                );
                break;

            case Cosine:
                if (auto* p0Layer = dynamic_cast<Layer*>(_vIncomingLayer[0]); p0Layer)
                {
                    Float beta0 = p0Layer->_deltaUpdateCount == 0 ? 0.0 : 1.0;
                    uint32_t offset = i - 1;

                    Float* pDPIn = GetUnitBuffer() + offset;
                    Float* pDPDeltaIn = GetDeltaBuffer() + offset;
                    Float* pAIn = _pbBuffer1->_pDevData + offset;
                    Float* pBIn = _pbBuffer2->_pDevData + offset;

                    kCalculateCosineDelta(
                        pDPDeltaIn,
                        pDPIn,
                        pAIn,
                        pBIn,
                        p0Layer->GetUnitBuffer(),
                        pInputLayer->GetUnitBuffer(),
                        batch,
                        _localStride,
                        p0Layer->GetIncomingDeltaBuffer(),
                        beta0,
                        pInputLayer->GetIncomingDeltaBuffer(),
                        beta,
                        pInputLayer->_localStride
                    );

                    p0Layer->_deltaUpdateCount++;
                }
                break;

            case DotProduct:
                if (auto* p0Layer = dynamic_cast<Layer*>(_vIncomingLayer[0]); p0Layer)
                {
                    Float beta0 = p0Layer->_deltaUpdateCount == 0 ? 0.0 : 1.0;
                    uint32_t offset = i - 1;
                    Float* pDPDeltaIn = GetDeltaBuffer() + offset;

                    kCalculateDotProductDelta(
                        pDPDeltaIn,
                        p0Layer->GetUnitBuffer(),
                        pInputLayer->GetUnitBuffer(),
                        batch,
                        _localStride,
                        p0Layer->GetIncomingDeltaBuffer(),
                        beta0,
                        pInputLayer->GetIncomingDeltaBuffer(),
                        beta,
                        pInputLayer->_localStride
                    );

                    p0Layer->_deltaUpdateCount++;
                }
                break;
        }

        pInputLayer->_deltaUpdateCount++;
    }

    for (auto* l : _vIncomingSkip)
    {
        if (l->_deltaUpdateCount > 0)
        {
            kAddBuffers(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), batch * _localStride);
        }
        else
        {
            cudaMemcpy(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), batch * _localStride * sizeof(Float), cudaMemcpyDefault);
        }

        l->_deltaUpdateCount++;
    }
}
#include <vector>
#include <array>
#include <algorithm>
#include <span>

void Layer::BackPropagateFullyConnected(uint32_t position, uint32_t batch)
{
    if (getGpu()._numprocs == 1)
    {
        if (_kind == Hidden)
        {
            if (_bSparse && getGpu()._data._bSparsenessPenalty)
            {
                Float p = (_sparsenessPenalty_p > (Float)0.0) ? _sparsenessPenalty_p : getGpu()._pNetwork->_sparsenessPenalty_p;
                Float beta = (_sparsenessPenalty_beta > (Float)0.0) ? _sparsenessPenalty_beta : getGpu()._pNetwork->_sparsenessPenalty_beta;
                kCalculateSparsenessPenalty(batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer(), p, beta);
            }

            Float scale = (Float)1.0 / ((Float)1.0 - _pDropout);
            kCalculateHadamardProduct(_activation, batch * _localStride, scale, GetUnitBuffer(), GetIncomingDeltaBuffer(), _RELUSlope, _ELUAlpha, _SELULambda);

            if (_deltaNorm > (Float)0.0)
            {
                kNormalizeDeltas(_deltaNorm, batch, _localStride, GetIncomingDeltaBuffer());
            }

            if (_bBatchNormalization)
            {
                cudnnStatus_t cudnnStatus;
                cudnnStatus = cudnnSetTensor4dDescriptor(_tensorDescriptorBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, _Nz, _Ny, _localStride);
                CUDNNERROR(cudnnStatus, "Layer::BackPropagateFullyConnected: unable to create _tensorDescriptorBN");
                cudnnStatus = cudnnSetTensor4dDescriptor(_scaleBiasMeanVarDescBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, _Nz, _Ny, _localStride);
                CUDNNERROR(cudnnStatus, "Layer::BackPropagateFullyConnected: unable to create _scaleBiasMeanVarDescBN");
                float alpha = 1;
                float beta = 0;
                cudnnStatus = cudnnBatchNormalizationBackward(
                    getGpu()._cuDNNHandle,
                    CUDNN_BATCHNORM_PER_ACTIVATION,
                    &alpha,
                    &beta,
                    &alpha,
                    &beta,
                    _tensorDescriptorBN,
                    GetIncomingUnitBuffer(),
                    _tensorDescriptorBN,
                    GetIncomingDeltaBuffer(),
                    _tensorDescriptorBN,
                    GetDeltaBuffer(),
                    _scaleBiasMeanVarDescBN,
                    _pbScaleBN->_pDevData,
                    _pbScaleGradientBN->_pDevData,
                    _pbBiasGradientBN->_pDevData,
                    CUDNN_BN_MIN_EPSILON,
                    _pbSaveMeanBN->_pDevData,
                    _pbSaveInvVarianceBN->_pDevData);
                CUDNNERROR(cudnnStatus, "Layer:BackPropagateFullyConnected cudnnBatchNormalizationBackward Failed");
            }
        }

#if 0
        if (_kind == Hidden)
        {
            string fname = "delta_" + _name;
            Dump(fname, _pbDelta->_pDevData);
        }
#endif

        for (const auto& [i, pInputLayer] : std::enumerate(_vIncomingLayer))
        {
            Layer* pInputLayer = _vIncomingLayer[i];
            cublasStatus_t cstatus;
            Weight* pWeight = _vIncomingWeight[i];
            Weight* pSrcWeight = pWeight->_bShared ? pWeight->_pSharedWeight : pWeight;

            if (!pWeight->_bLocked)
            {
                std::vector<Float> pDelta(GetDeltaBuffer(), GetDeltaBuffer() + batch * _localStride);
                std::vector<Float> pUnit(pInputLayer->GetUnitBuffer(), pInputLayer->GetUnitBuffer() + pInputLayer->_localStride);
                Float* pA = pWeight->_bTransposed ? pDelta.data() : pUnit.data();
                Float* pB = pWeight->_bTransposed ? pUnit.data() : pDelta.data();
                int m = pWeight->_bTransposed ? pInputLayer->_localStride : _localStride;
                int n = pWeight->_bTransposed ? _localStride : pInputLayer->_localStride;
                int k = batch;
                int lda = pWeight->_bTransposed ? pInputLayer->_localStride : _localStride;
                int ldb = pWeight->_bTransposed ? _localStride : pInputLayer->_localStride;
                int ldc = pWeight->_bTransposed ? pInputLayer->_localStride : _localStride;

                Float sgemm_alpha = -(Float)1.0 / (pSrcWeight->_sharingCount * (Float)batch);
                Float sgemm_beta = (pSrcWeight->_updateCount == 0) ? (Float)0.0 : (Float)1.0;
                Float* pC = pSrcWeight->_pbWeightGradient->_pDevData;

                if ((pInputLayer->_kind == Layer::Kind::Input) && pInputLayer->_bFastSparse && !pWeight->_bTransposed)
                {
                    pInputLayer->_pDataSet->CalculateSparseTransposedWeightGradient(sgemm_alpha, sgemm_beta, n, m, pB, pC);
                }
                else
                {
                    cstatus = cublasSgemm(getGpu()._cuBLASHandle,
                                          CUBLAS_OP_N,
                                          CUBLAS_OP_T,
                                          m,
                                          n,
                                          k,
                                          &sgemm_alpha,
                                          pB,
                                          lda,
                                          pA,
                                          ldb,
                                          &sgemm_beta,
                                          pC,
                                          ldc);

                    if (cstatus != CUBLAS_STATUS_SUCCESS)
                    {
                        if (getGpu()._id == 0)
                            printf("Layer::BackPropagate: SGEMM failure, aborting.\n");
                        getGpu().Shutdown();
                        exit(-1);
                    }
                }

                pSrcWeight->_updateCount++;
            }

            if (pInputLayer->_kind != Input)
            {
                Float sgemm_alpha = (Float)1.0;
                Float sgemm_beta = (pInputLayer->_deltaUpdateCount == 0) ? (Float)0.0 : (Float)1.0;
                int m = pInputLayer->_localStride;
                int n = batch;

                Float* pA = GetDeltaBuffer();
                Float* pB = pWeight->_bShared ?
                                pSrcWeight->_pbWeight->_pDevData :
                                pWeight->_pbWeight->_pDevData;

                Float* pC = pInputLayer->GetIncomingDeltaBuffer();
                int k = _localStride;
                int lda = pWeight->_bTransposed ? pInputLayer->_localStride : k;
                int ldb = k;
                int ldc = pInputLayer->_localStride;

                cstatus = cublasSgemm(getGpu()._cuBLASHandle,
                                      pWeight->_bTransposed ? CUBLAS_OP_N : CUBLAS_OP_T,
                                      CUBLAS_OP_N,
                                      m,
                                      n,
                                      k,
                                      &sgemm_alpha,
                                      pB,
                                      lda,
                                      pA,
                                      ldb,
                                      &sgemm_beta,
                                      pC,
                                      ldc);

                if (cstatus != CUBLAS_STATUS_SUCCESS)
                {
                    if (getGpu()._id == 0)
                        printf("Layer::BackPropagate: SGEMM failure, aborting.\n");
                    getGpu().Shutdown();
                    exit(-1);
                }

                pInputLayer->_deltaUpdateCount++;
            }
        }

        for (auto l : _vIncomingSkip)
        {
            if (l->_deltaUpdateCount > 0)
            {
                kAddBuffers(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), batch * _localStride);
            }
            else
            {
                cudaMemcpy(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), batch * _localStride * sizeof(Float), cudaMemcpyDefault);
            }

            l->_deltaUpdateCount++;
        }
    }
    else
    {
        if (_vOutgoingLargerLayer.size() > 0)
        {
            Gather(batch, _stride, GetUnitBuffer(), _localStride);

            for (int i = 0; i < _vOutgoingLargerLayer.size(); i++)
            {
                Layer* pOutputLayer = _vOutgoingLargerLayer[i];
                Weight* pWeight = _vOutgoingLargerWeight[i];
                Weight* pSrcWeight = pWeight->_bShared ? pWeight->_pSharedWeight : pWeight;

                Float* pA = pOutputLayer->GetDeltaBuffer();
                Float* pB = getGpu()._pNetwork->GetP2PSendBuffer();
                Float* pC = pSrcWeight->_pbWeightGradient->_pDevData;
                int m = pOutputLayer->_localStride;
                int n = _stride;
                int k = batch;
                int lda = pOutputLayer->_localStride;
                int ldb = _stride;
                int ldc = pOutputLayer->_localStride;

                Float sgemm_alpha = -(Float)1.0 / (pSrcWeight->_sharingCount * (Float)batch);
                Float sgemm_beta = (pSrcWeight->_updateCount == 0) ? (Float)0.0 : (Float)1.0;

                cublasStatus_t cstatus = cublasSgemm(getGpu()._cuBLASHandle,
                                                    CUBLAS_OP_N,
                                                    CUBLAS_OP_T,
                                                    m,
                                                    n,
                                                    k,
                                                    &sgemm_alpha,
                                                    pA,
                                                    lda,
                                                    pB,
                                                    ldb,
                                                    &sgemm_beta,
                                                    pC,
                                                    ldc);

                if (cstatus != CUBLAS_STATUS_SUCCESS)
                {
                    if (getGpu()._id == 0)
                        printf("Layer::BackPropagate: SGEMM failure, aborting.\n");
                    getGpu().Shutdown();
                    exit(-1);
                }

                pSrcWeight->_updateCount++;
            }

            Float sgemm_beta = (Float)0.0;
            for (uint32_t i = 0; i < _vOutgoingLargerLayer.size(); i++)
            {
                Layer* pOutputLayer = _vOutgoingLargerLayer[i];
                const Float sgemm_alpha = (Float)1.0;
                Float* pA = _vOutgoingLargerWeight[i]->_bShared ?
                                _vOutgoingLargerWeight[i]->_pSharedWeight->_pbWeight->_pDevData :
                                _vOutgoingLargerWeight[i]->_pbWeight->_pDevData;
                Float* pB = pOutputLayer->GetDeltaBuffer();
                Float* pC = getGpu()._pNetwork->GetP2PSendBuffer();
                int m = _stride;
                int n = batch;
                int k = pOutputLayer->_localStride;
                int lda = pOutputLayer->_localStride;
                int ldb = pOutputLayer->_localStride;
                int ldc = _stride;

                cublasStatus_t cstatus =
                    cublasSgemm(getGpu()._cuBLASHandle,
                                CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                m,
                                n,
                                k,
                                &sgemm_alpha,
                                pA,
                                lda,
                                pB,
                                ldb,
                                &sgemm_beta,
                                pC,
                                ldc);

                if (cstatus != CUBLAS_STATUS_SUCCESS)
                {
                    if (getGpu()._id == 0)
                        printf("Layer::BackPropagate: SGEMM failure, aborting, status %d.\n", cstatus);
                    getGpu().Shutdown();
                    exit(-1);
                }
#if 0
                Float* pD = pOutputLayer->_vDelta.data();
                Float* pW = _vOutgoingWeight[i]->_vWeight.data();

                pOutputLayer->_pbDelta->Download(pD);
                _vOutgoingLargerWeight[i]->_pbWeight->Download(pW);
                pW += pOutputLayer->_localStride;
                Float sum = 0.0f;
                for (int j = 0; j < pOutputLayer->_localStride; j++)
                {
                    sum += (*pD) * (*pW);
                    pD++;
                    pW++;
                }
                MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                if (getGpu()._id == 0)
                    printf("ZAG %16.12f\n", sum);
                MPI_Barrier(MPI_COMM_WORLD);
#endif

                sgemm_beta = (Float)1.0;
            }

            Reduce(batch, _stride, GetIncomingDeltaBuffer(), _localStride, _deltaUpdateCount);
            _deltaUpdateCount++;
        }

        if (_kind == Hidden)
        {
            if (_bSparse && getGpu()._data._bSparsenessPenalty)
            {
                Float p = (_sparsenessPenalty_p > (Float)0.0) ? _sparsenessPenalty_p : getGpu()._pNetwork->_sparsenessPenalty_p;
                Float beta = (_sparsenessPenalty_beta > (Float)0.0) ? _sparsenessPenalty_beta : getGpu()._pNetwork->_sparsenessPenalty_beta;
                kCalculateSparsenessPenalty(batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer(), p, beta);
            }

            Float scale = (Float)1.0 / ((Float)1.0 - _pDropout);
            kCalculateHadamardProduct(_activation, batch * _localStride, scale, GetUnitBuffer(), GetIncomingDeltaBuffer(), _RELUSlope, _ELUAlpha, _SELULambda);

            if (_deltaNorm > (Float)0.0)
            {
                Float* pMagnitude = getGpu()._pNetwork->GetScratchBuffer(batch);
                kCalculateDeltaMagnitudes(batch, _localStride, GetIncomingDeltaBuffer(), pMagnitude);
                getGpu()._pNetwork->P2P_Allreduce(pMagnitude, batch);
                kNormalizeDeltaMagnitudes(_deltaNorm, batch, _localStride, GetIncomingDeltaBuffer(), pMagnitude);
            }

            if (_bBatchNormalization)
            {
                cudnnStatus_t cudnnStatus;
                cudnnStatus = cudnnSetTensor4dDescriptor(_tensorDescriptorBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, _Nz, _Ny, _localStride);
                CUDNNERROR(cudnnStatus, "Layer::BackPropagateFullyConnected: unable to create _tensorDescriptorBN");
                cudnnStatus = cudnnSetTensor4dDescriptor(_scaleBiasMeanVarDescBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, _Nz, _Ny, _localStride);
                CUDNNERROR(cudnnStatus, "Layer::BackPropagateFullyConnected: unable to create _scaleBiasMeanVarDescBN");
                float alpha = 1;
                float beta = 0;
                cudnnStatus = cudnnBatchNormalizationBackward(
                    getGpu()._cuDNNHandle,
                    CUDNN_BATCHNORM_PER_ACTIVATION,
                    &alpha,
                    &beta,
                    &alpha,
                    &beta,
                    _tensorDescriptorBN,
                    GetIncomingUnitBuffer(),
                    _tensorDescriptorBN,
                    GetIncomingDeltaBuffer(),
                    _tensorDescriptorBN,
                    GetDeltaBuffer(),
                    _scaleBiasMeanVarDescBN,
                    _pbScaleBN->_pDevData,
                    _pbScaleGradientBN->_pDevData,
                    _pbBiasGradientBN->_pDevData,
                    CUDNN_BN_MIN_EPSILON,
                    _pbSaveMeanBN->_pDevData,
                    _pbSaveInvVarianceBN->_pDevData);
                CUDNNERROR(cudnnStatus, "Layer:BackPropagateFullyConnected cudnnBatchNormalizationBackward Failed");
            }
        }
    }
}

void Layer::UpdateWeights(TrainingMode trainingMode, uint32_t batch, Float alpha, Float lambda, Float lambda1, Float mu, Float mu1, Float t)
{
    if (_bBatchNormalization)
    {
        switch (trainingMode)
        {
            case TrainingMode::SGD:
                kSGDUpdateWeights(-alpha, lambda, lambda1, _localStride, _pbScaleGradientBN->_pDevData, _pbScaleBN->_pDevData);
                kSGDUpdateWeights(-alpha, lambda, lambda1, _localStride, _pbBiasGradientBN->_pDevData, _pbBiasBN->_pDevData);
                break;
                
            case TrainingMode::Momentum:
                kMomentumUpdateWeights(-alpha, lambda, lambda1, mu, _localStride, _pbScaleVelocityBN->_pDevData, _pbScaleGradientBN->_pDevData, _pbScaleBN->_pDevData);
                kMomentumUpdateWeights(-alpha, lambda, lambda1, mu, _localStride, _pbBiasVelocityBN->_pDevData, _pbBiasGradientBN->_pDevData, _pbBiasBN->_pDevData);
                break;
                        
            case TrainingMode::AdaGrad:
                kAdaGradUpdateWeights(-alpha, lambda, lambda1, _localStride, _pbScaleVelocityBN->_pDevData, _pbScaleGradientBN->_pDevData, _pbScaleBN->_pDevData);
                kAdaGradUpdateWeights(-alpha, lambda, lambda1, _localStride, _pbBiasVelocityBN->_pDevData, _pbBiasGradientBN->_pDevData, _pbBiasBN->_pDevData);
                break;
                        
            case TrainingMode::Nesterov:
                kNesterovUpdateWeights(-alpha, lambda, lambda1, mu, _localStride, _pbScaleVelocityBN->_pDevData, _pbScaleGradientBN->_pDevData, _pbScaleBN->_pDevData);
                kNesterovUpdateWeights(-alpha, lambda, lambda1, mu, _localStride, _pbBiasVelocityBN->_pDevData, _pbBiasGradientBN->_pDevData, _pbBiasBN->_pDevData);
                break;
                        
            case TrainingMode::RMSProp:
                kRMSPropUpdateWeights(-alpha, lambda, lambda1, mu, _localStride, _pbScaleVelocityBN->_pDevData, _pbScaleGradientBN->_pDevData, _pbScaleBN->_pDevData);
                kRMSPropUpdateWeights(-alpha, lambda, lambda1, mu, _localStride, _pbBiasVelocityBN->_pDevData, _pbBiasGradientBN->_pDevData, _pbBiasBN->_pDevData);
                break;

            case TrainingMode::AdaDelta:
                kAdaDeltaUpdateWeights(lambda, lambda1, mu, _localStride, _pbScaleVelocityBN->_pDevData, _pbScaleGradientBN->_pDevData, _pbScaleGradientVelocityBN->_pDevData, _pbScaleBN->_pDevData);
                kAdaDeltaUpdateWeights(lambda, lambda1, mu, _localStride, _pbBiasVelocityBN->_pDevData, _pbBiasGradientBN->_pDevData, _pbBiasGradientVelocityBN->_pDevData, _pbBiasBN->_pDevData);
                break;     

            case TrainingMode::Adam:
                kAdamUpdateWeights(-alpha, lambda, lambda1, mu, mu1, t, _localStride, _pbScaleVelocityBN->_pDevData, _pbScaleGradientBN->_pDevData, _pbScaleGradientVelocityBN->_pDevData, _pbScaleBN->_pDevData);
                kAdamUpdateWeights(-alpha, lambda, lambda1, mu, mu1, t, _localStride, _pbBiasVelocityBN->_pDevData, _pbBiasGradientBN->_pDevData, _pbBiasGradientVelocityBN->_pDevData, _pbBiasBN->_pDevData);
                break;   
        }
    }
}

void Layer::Reduce(std::span<Float> buffer, std::size_t localStride, uint32_t updateCount)
{
    if (getGpu()._numprocs > 1)
    {
        uint32_t stages = getGpu()._numprocs - 1;
        uint64_t pos = (getGpu()._id + 1) % getGpu()._numprocs; 
        uint32_t minX = (stride * pos) / getGpu()._numprocs;
        uint32_t maxX = (stride * (pos + 1)) / getGpu()._numprocs;
        uint32_t span = maxX - minX;
        Float* pSendBuffer = getGpu()._pNetwork->GetP2PSendBuffer();

        if (getGpu()._bP2P)
        {
            Float* pReceiveBuffer = getGpu()._pNetwork->GetP2PReceiveBuffer();
            Float* pPeerBuffer = getGpu()._pNetwork->GetPeerBuffer();

            for (uint32_t i = 0; i < stages; i++)
            {
                kCopy2D(pPeerBuffer + minX, stride, pSendBuffer + minX, stride, span, batch);
                // cudaDeviceSynchronize();   // Remove unnecessary synchronization
                std::scoped_lock lock(mutex_);
                MPI_Barrier(MPI_COMM_WORLD);
        
                pos = (pos + 1) % getGpu()._numprocs;
                minX = (stride * pos) / getGpu()._numprocs;
                maxX = (stride * (pos + 1)) / getGpu()._numprocs;
                span = maxX - minX;
                kAddBuffers2D(pSendBuffer + minX, stride, pReceiveBuffer + minX, stride, span, batch);
            }
        }
        else
        {
            std::vector<Float> pCPUBuffer(buffer.size());
            cudaError_t status = cudaMemcpy(pCPUBuffer.data(), pSendBuffer, buffer.size() * sizeof(Float), cudaMemcpyDefault);
            RTERROR(status, "Layer::Reduce1: cudaMemcpy download failed " + getGpu()._id);
            MPI_Allreduce(MPI_IN_PLACE, pCPUBuffer.data(), buffer.size(), MPI_Float, MPI_SUM, MPI_COMM_WORLD);

            status = cudaMemcpy(pSendBuffer, pCPUBuffer.data(), buffer.size() * sizeof(Float), cudaMemcpyDefault);
            RTERROR(status, "Layer::Reduce: cudaMemcpy upload failed" + getGpu()._id);
            minX = (stride * getGpu()._id) / getGpu()._numprocs;
            maxX = (stride * (getGpu()._id + 1)) / getGpu()._numprocs;
            span = maxX - minX;            
        }

        if (updateCount > 0) 
        {
            kAddBuffers2D(buffer.data(), localStride, pSendBuffer + minX, stride, span, batch);
        }
        else 
        {
            kCopy2D(buffer.data(), localStride, pSendBuffer + minX, stride, span, batch);
        }
    }
}

#include <vector>
#include <string>
#include <span>

void Layer::Gather(uint32_t batch, uint32_t stride, std::vector<Float>& pBuffer, uint32_t localStride)
{
    if (getGpu()._numprocs > 1)
    {
        uint32_t stages = getGpu()._numprocs - 1;
        uint64_t pos = getGpu()._id;
        std::vector<Float>& pSendBuffer = getGpu()._pNetwork->GetP2PSendBuffer();
        uint32_t minX = (stride * pos) / getGpu()._numprocs;
        uint32_t maxX = (stride * (pos + 1)) / getGpu()._numprocs;
        uint32_t span = maxX - minX;

        if (getGpu()._bP2P)
        {
            std::vector<Float>& pPeerBuffer = getGpu()._pNetwork->GetPeerBackBuffer();

            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);

            kCopy2D(pSendBuffer.data() + minX, stride, pBuffer.data(), localStride, span, batch);

            for (uint32_t i = 0; i < stages; i++)
            {
                kCopy2D(pPeerBuffer.data() + minX, stride, pSendBuffer.data() + minX, stride, span, batch);
                cudaDeviceSynchronize();
                MPI_Barrier(MPI_COMM_WORLD);
                pos = (pos + 1) % getGpu()._numprocs;
                minX = (stride * pos) / getGpu()._numprocs;
                maxX = (stride * (pos + 1)) / getGpu()._numprocs;
                span = maxX - minX;
            }
        }
        else
        {
            std::vector<Float>& pCPUBuffer = getGpu()._pNetwork->GetP2PCPUBuffer();

            cudaError_t status = cudaMemcpy2D(pCPUBuffer.data() + minX, stride * sizeof(Float), pBuffer.data(), localStride * sizeof(Float), localStride * sizeof(Float), batch, cudaMemcpyDefault);
            RTERROR(status, "Layer::Gather: cudaMemcpy download failed");

            for (uint32_t i = 0; i < getGpu()._numprocs; i++)
            {
                uint32_t minX = (stride * i) / getGpu()._numprocs;
                uint32_t maxX = (stride * (i + 1)) / getGpu()._numprocs;
                uint32_t span = maxX - minX;
                MPI_Datatype spanType;
                MPI_Type_vector(batch, span, stride, MPI_Float, &spanType);
                MPI_Type_commit(&spanType);
                MPI_Bcast(pCPUBuffer.data() + minX, 1, spanType, i, MPI_COMM_WORLD);
                MPI_Type_free(&spanType);
            }

            status = cudaMemcpy(pSendBuffer.data(), pCPUBuffer.data(), batch * stride * sizeof(Float), cudaMemcpyDefault);
            RTERROR(status, "Layer::Gather: cudaMemcpy upload failed");
        }
    }
}

void Layer::Dump(const std::string& fname, std::span<const Float> pBuffer)
{
    std::vector<Float> vData(_batch * _stride);
    if (getGpu()._numprocs == 1)
    {
        cudaMemcpy(vData.data(), pBuffer.data(), _batch * _stride * sizeof(Float), cudaMemcpyDefault);
    }
    else
    {
        if (getGpu()._id == 0)
        {
            Float* pData = vData.data();
            cudaMemcpy2D(pData, _stride * sizeof(Float), pBuffer.data(), _localStride * sizeof(Float), _localStride * sizeof(Float), _batch, cudaMemcpyDefault);
            pData += _localStride;
            for (uint32_t i = 1; i < getGpu()._numprocs; i++)
            {
                uint64_t size;
                MPI_Status status;
                MPI_Recv(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
                std::vector<Float> vTemp(size);
                MPI_Recv(vTemp.data(), size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
                uint64_t lstride = size / _batch;
                Float* pSrc = vTemp.data();
                Float* pDst = pData;
                for (uint32_t j = 0; j < _batch; j++)
                {
                    std::memcpy(pDst, pSrc, lstride * sizeof(Float));
                    pSrc += lstride;
                    pDst += _stride;
                }
                pData += lstride;
            }
        }
        else
        {
            uint64_t size = _batch * _localStride;
            std::vector<Float> vLocalData(size);
            cudaMemcpy(vLocalData.data(), pBuffer.data(), size * sizeof(Float), cudaMemcpyDefault);
            MPI_Send(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD);
            MPI_Send(vLocalData.data(), size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
    }

    if (getGpu()._id == 0)
    {
        FILE* fp = fopen(fname.c_str(), "w");
        Float* pData = vData.data();
        for (int i = 0; i < _batch; i++)
        {
            fprintf(fp, "%4d ", i);
            for (int j = 0; j < _stride; j++)
            {
                fprintf(fp, "%12.9f ", *pData);
                pData++;
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }
} // TODO


std::pair<Layer::Kind, string> Layer::_sKindPair[] =
{
    std::pair<Layer::Kind, string>(Layer::Kind::Input,      "Input"),
    std::pair<Layer::Kind, string>(Layer::Kind::Hidden,     "Hidden"),
    std::pair<Layer::Kind, string>(Layer::Kind::Output,     "Output"),
    std::pair<Layer::Kind, string>(Layer::Kind::Target,     "Target"),    
};

std::map<Layer::Kind, string> Layer::_sKindMap =
std::map<Layer::Kind, string>(_sKindPair, _sKindPair + sizeof(_sKindPair) / sizeof(_sKindPair[0]));


std::pair<Layer::Type, string> Layer::_sTypePair[] =
{
    std::pair<Layer::Type, string>(Layer::Type::FullyConnected, "FullyConnected"),
    std::pair<Layer::Type, string>(Layer::Type::Convolutional,  "Convolutional"),
    std::pair<Layer::Type, string>(Layer::Type::Pooling,        "Pooling"),    
};

std::map<Layer::Type, string> Layer::_sTypeMap =
std::map<Layer::Type, string>(_sTypePair, _sTypePair + sizeof(_sTypePair) / sizeof(_sTypePair[0]));

std::pair<Layer::Attributes, string> Layer::_sAttributesPair[] =
{
    std::pair<Layer::Attributes, string>(Layer::Attributes::None,               "None"),
    std::pair<Layer::Attributes, string>(Layer::Attributes::Sparse,             "Sparse"),
    std::pair<Layer::Attributes, string>(Layer::Attributes::Denoising,          "Denoising"),
    std::pair<Layer::Attributes, string>(Layer::Attributes::BatchNormalization, "BatchNormalization"),
};

std::map<Layer::Attributes, string> Layer::_sAttributesMap =
std::map<Layer::Attributes, string>(_sAttributesPair, _sAttributesPair + sizeof(_sAttributesPair) / sizeof(_sAttributesPair[0]));

std::pair<Layer::Parallelization, string> Layer::_sParallelizationPair[] =
{
    
    std::pair<Layer::Parallelization, string>(Layer::Parallelization::Data,     "Data"),
    std::pair<Layer::Parallelization, string>(Layer::Parallelization::Model,    "Model"),
    std::pair<Layer::Parallelization, string>(Layer::Parallelization::Serial,   "Serial"),
};

std::map<Layer::Parallelization, string> Layer::_sParallelizationMap =
std::map<Layer::Parallelization, string>(_sParallelizationPair, _sParallelizationPair + sizeof(_sParallelizationPair) / sizeof(_sParallelizationPair[0]));


ostream& operator<< (ostream& out, Layer::Kind& k)
{
    out << Layer::_sKindMap[k];
    return out;
}
ostream& operator<< (ostream& out, Layer::Type& t)
{
    out << Layer::_sTypeMap[t];
    return out;
}

ostream& operator<< (ostream& out, Layer::Parallelization& p)
{
    out << Layer::_sParallelizationMap[p];
    return out;
}

ostream& operator<< (ostream& out, Layer::Attributes& a)
{
    out << Layer::_sAttributesMap[a];
    return out;
}




LayerDescriptor::LayerDescriptor() :
_kind(Layer::Kind::Hidden),
_type(Layer::Type::FullyConnected),
_poolingFunction(None),
_Nx(1),
_Ny(1),
_Nz(1),
_Nw(1),
_dimensions(1),
_bDimensionsProvided(true),
_weightInit(Xavier),
_weightInitScale((Float)1.0),
_biasInit((Float)0.0),
_kernelX(1),
_kernelY(1),
_kernelZ(1),
_kernelStrideX(1),
_kernelStrideY(1),
_kernelStrideZ(1),
_kernelPaddingX(0),
_kernelPaddingY(0),
_kernelPaddingZ(0),
_kernelDimensions(1),
_weightNorm((Float)0.0),
_deltaNorm((Float)0.0),
_pDropout((Float)0.0),
_activation(Activation::Sigmoid),
_sparsenessPenalty_p((Float)0.0),
_sparsenessPenalty_beta((Float)0.0),
_RELUSlope(NAN),
_ELUAlpha(NAN),
_SELULambda(NAN),
_attributes(Layer::Attributes::None)
{

}

bool LoadLayerDescriptorNetCDF(const string& fname, netCDF::NcFile& nc, uint32_t index, LayerDescriptor& ld)
{
    bool bResult                                = true; 

    if (getGpu()._id == 0)
    {
        try {
            string lstring                      = "layer" + std::to_string(index) + "_";
            NcGroupAtt nameAtt                  = nc.getAtt(lstring + "name");
            if (nameAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No name supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            nameAtt.getValues(ld._name);

            NcGroupAtt kindAtt                  = nc.getAtt(lstring + "kind");
            if (kindAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No kind supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kindAtt.getValues(&ld._kind);

            NcGroupAtt typeAtt                  = nc.getAtt(lstring + "type");
            if (typeAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No type supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            typeAtt.getValues(&ld._type);
            
            NcGroupAtt poolingFunctionAtt       = nc.getAtt(lstring + "poolingfunction");
            if (poolingFunctionAtt.isNull())
            {
                if (ld._type == Layer::Type::Pooling)
                    throw NC_EXCEPTION("NcException", "Layer::Layer: No pooling function supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                ld._poolingFunction             = None;
            }
            else
                poolingFunctionAtt.getValues(&ld._poolingFunction);

            NcGroupAtt dataSetAtt               = nc.getAtt(lstring + "dataSet");
            if (dataSetAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No dataSet supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            dataSetAtt.getValues(ld._dataSet);

            NcGroupAtt NxAtt                    = nc.getAtt(lstring + "Nx");
            if (NxAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No Nx supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            NxAtt.getValues(&ld._Nx);

            NcGroupAtt NyAtt                    = nc.getAtt(lstring + "Ny");
            if (NyAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No Ny supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            NyAtt.getValues(&ld._Ny);

            NcGroupAtt NzAtt                    = nc.getAtt(lstring + "Nz");
            if (NzAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No Nz supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            NzAtt.getValues(&ld._Nz);

            NcGroupAtt NwAtt                    = nc.getAtt(lstring + "Nw");
            if (NwAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No Nw supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            NwAtt.getValues(&ld._Nw);

            NcGroupAtt dimensionsAtt            = nc.getAtt(lstring + "dimensions");
            if (dimensionsAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No dimensions supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            dimensionsAtt.getValues(&ld._dimensions);

            NcGroupAtt kernelXAtt               = nc.getAtt(lstring + "kernelX");
            if (kernelXAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No kernelX supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelXAtt.getValues(&ld._kernelX);

            NcGroupAtt kernelYAtt               = nc.getAtt(lstring + "kernelY");
            if (kernelYAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No kernelY supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelYAtt.getValues(&ld._kernelY);

            NcGroupAtt kernelZAtt               = nc.getAtt(lstring + "kernelZ");
            if (kernelZAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No kernelZ supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelZAtt.getValues(&ld._kernelZ);

            NcGroupAtt kernelStrideXAtt         = nc.getAtt(lstring + "kernelStrideX");
            if (kernelStrideXAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No kernelStrideX supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelStrideXAtt.getValues(&ld._kernelStrideX);

            NcGroupAtt kernelStrideYAtt         = nc.getAtt(lstring + "kernelStrideY");
            if (kernelStrideYAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No kernelStrideY supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelStrideYAtt.getValues(&ld._kernelStrideY);

            NcGroupAtt kernelStrideZAtt         = nc.getAtt(lstring + "kernelStrideZ");
            if (kernelStrideZAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No kernelStrideZ supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelStrideZAtt.getValues(&ld._kernelStrideZ);


            NcGroupAtt kernelPaddingXAtt        = nc.getAtt(lstring + "kernelPaddingX");
            if (kernelPaddingXAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No kernelPaddingX supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelPaddingXAtt.getValues(&ld._kernelPaddingX);

            NcGroupAtt kernelPaddingYAtt        = nc.getAtt(lstring + "kernelPaddingY");
            if (kernelPaddingYAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No kernelPaddingY supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelPaddingYAtt.getValues(&ld._kernelPaddingY);

            NcGroupAtt kernelPaddingZAtt        = nc.getAtt(lstring + "kernelPaddingZ");
            if (kernelPaddingZAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No kernelPaddingZ supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelPaddingZAtt.getValues(&ld._kernelPaddingZ);          

            NcGroupAtt kernelDimensionsAtt      = nc.getAtt(lstring + "kernelDimensions");
            if (kernelDimensionsAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No kernelDimensions supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelDimensionsAtt.getValues(&ld._kernelDimensions);
            
            NcGroupAtt weightInitAtt            = nc.getAtt(lstring + "weightInit");
            if (weightInitAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No weightInit supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                ld._weightInit                  = Xavier;
            }
            else
                weightInitAtt.getValues(&ld._weightInit);      
            
            NcGroupAtt weightInitScaleAtt       = nc.getAtt(lstring + "weightInitScale");
            if (weightInitScaleAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No weightInitScale supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                ld._weightInitScale             = (Float)1.0;
            }
            else
                weightInitScaleAtt.getValues(&ld._weightInitScale);   
                
            NcGroupAtt biasInitAtt              = nc.getAtt(lstring + "biasInit");
            if (biasInitAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No biasInit supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                ld._biasInit                    = (Float)0.0;
            }
            else
                biasInitAtt.getValues(&ld._biasInit);       
                      
            NcGroupAtt weightNormAtt            = nc.getAtt(lstring + "weightNorm");
            if (weightNormAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No weightNorm supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                ld._weightNorm                  = (Float)0.0;
            }
            else
                weightNormAtt.getValues(&ld._weightNorm);
            
            NcGroupAtt deltaNormAtt             = nc.getAtt(lstring + "deltaNorm");
            if (deltaNormAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No deltaNorm supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                ld._deltaNorm                   = (Float)0.0;
            }
            else
                deltaNormAtt.getValues(&ld._deltaNorm);
                
            NcGroupAtt pDropoutAtt              = nc.getAtt(lstring + "pDropout");
            if (pDropoutAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No pDropout supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            else
                pDropoutAtt.getValues(&ld._pDropout);

            NcGroupAtt activationAtt            = nc.getAtt(lstring + "activation");
            if (activationAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No activation supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            activationAtt.getValues(&ld._activation);

            NcGroupAtt RELUSlopeAtt             = nc.getAtt(lstring + "RELUSlope");
            if (RELUSlopeAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No RELUSlope supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            RELUSlopeAtt.getValues(&(ld._RELUSlope));

            
            NcGroupAtt ELUAlphaAtt              = nc.getAtt(lstring + "ELUAlpha");
            if (ELUAlphaAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No ELUAlpha supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            ELUAlphaAtt.getValues(&(ld._ELUAlpha));
            
            NcGroupAtt SELULambdaAtt            = nc.getAtt(lstring + "SELULambda");
            if (SELULambdaAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No SELULambda supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            SELULambdaAtt.getValues(&(ld._SELULambda)); 
            
            NcGroupAtt sparsenessPenalty_pAtt   = nc.getAtt("sparsenessPenalty_p");   
            if (sparsenessPenalty_pAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No sparsenessPenalty_p supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            else
            {
                sparsenessPenalty_pAtt.getValues(&(ld._sparsenessPenalty_p));
            }

            NcGroupAtt sparsenessPenalty_betaAtt= nc.getAtt("sparsenessPenalty_beta");
            if (sparsenessPenalty_betaAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No sparsenessPenalty_beta supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                ld._sparsenessPenalty_p = (Float)0.0;
            }
            else
            {
                sparsenessPenalty_betaAtt.getValues(&(ld._sparsenessPenalty_beta));
            }

            NcGroupAtt attributesAtt            = nc.getAtt(lstring + "attributes");
            if (attributesAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No attributes supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            attributesAtt.getValues(&ld._attributes);

            if (ld._attributes & Layer::Attributes::BatchNormalization)
            {
                NcDim bnDim                 = nc.getDim(lstring + "bnDim");
                NcVar scaleBNVar            = nc.getVar(lstring + "scaleBN");
                NcVar biasBNVar             = nc.getVar(lstring + "biasBN");
                NcVar runningMeanBNVar      = nc.getVar(lstring + "runningMeanBN");
                NcVar runningVarianceBNVar  = nc.getVar(lstring + "runningVarianceBN");

                ld._vScaleBN.resize(bnDim.getSize());
                ld._vBiasBN.resize(bnDim.getSize());
                ld._vRunningMeanBN.resize(bnDim.getSize());
                ld._vRunningVarianceBN.resize(bnDim.getSize());

                scaleBNVar.getVar(ld._vScaleBN.data());
                biasBNVar.getVar(ld._vBiasBN.data());
                runningMeanBNVar.getVar(ld._vRunningMeanBN.data());
                runningVarianceBNVar.getVar(ld._vRunningVarianceBN.data());
            }

            uint32_t sources                    = 0;
            NcGroupAtt sourcesAtt               = nc.getAtt(lstring + "sources");
            if (sourcesAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No sources supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            sourcesAtt.getValues(&sources);

            for (uint32_t i = 0; i < sources; i++)
            {
                string nstring                  = std::to_string(i);
                NcGroupAtt sourceAtt            = nc.getAtt(lstring + "source" + nstring);
                if (sourcesAtt.isNull())
                {
                    throw NC_EXCEPTION("NcException", "Layer::Layer: No source attributes supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                string source;
                sourceAtt.getValues(source);
                ld._vSource.push_back(source);        
            }   
            
            uint32_t skips                      = 0;
            NcGroupAtt skipsAtt                 = nc.getAtt(lstring + "skips");
            if (skipsAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "Layer::Layer: No skips supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            skipsAtt.getValues(&skips);

            for (uint32_t i = 0; i < skips; i++)
            {
                string nstring                  = std::to_string(i);
                NcGroupAtt skipAtt              = nc.getAtt(lstring + "skip" + nstring);
                if (skipAtt.isNull())
                {
                    throw NC_EXCEPTION("NcException", "Layer::Layer: No skip attributes supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                string skip;
                skipAtt.getValues(skip);
                ld._vSkip.push_back(skip);        
            }                    
        }
        catch (NcException& e)
        {
            cout << "Exception: " << e.what() << endl;
            bResult                             = false;
        }
    }
   
    return bResult;
}

ostream& operator<< (ostream& out, LayerDescriptor& d)
{
    out << "Name:                  " << d._name << endl;
    out << "Kind:                  " << d._kind << endl;
    out << "Type:                  " << d._type << endl;
    if (d._type != Layer::Type::Pooling)
        out << "Pooling Function:      " << d._poolingFunction << endl;
    out << "Nx:                    " << d._Nx << endl;
    out << "Ny:                    " << d._Ny << endl;
    out << "Nz:                    " << d._Nz << endl;
    out << "Nw:                    " << d._Nw << endl;
    if (d._type != Layer::Type::FullyConnected)
    {
        out << "kernelX:               " << d._kernelX << endl;
        out << "kernelY:               " << d._kernelY << endl;
        out << "kernelZ:               " << d._kernelZ << endl;
        out << "kernelStrideX:         " << d._kernelStrideX << endl;
        out << "kernelStrideY:         " << d._kernelStrideY << endl;
        out << "kernelStrideZ:         " << d._kernelStrideZ << endl;
        out << "kernelPaddingX:        " << d._kernelPaddingX << endl;
        out << "kernelPaddingY:        " << d._kernelPaddingY << endl;
        out << "kernelPaddingZ:        " << d._kernelPaddingZ << endl;
        out << "kernelDimensions:      " << d._kernelDimensions << endl;
    }
    if (d._type != Layer::Type::Pooling)
    {
        out << "pDropout:              " << d._pDropout << endl;
        out << "weightInit:            " << d._weightInit << endl;
        out << "weightInitScale:       " << d._weightInitScale << endl;
        out << "biasInit:              " << d._biasInit << endl;
        out << "weightNorm:            " << d._weightNorm << endl;
        out << "deltaNorm:             " << d._deltaNorm << endl;
        out << "activation:            " << d._activation << endl;
        out << "RELUSlope:             " << d._RELUSlope << endl;
        out << "ELUAlpha:              " << d._ELUAlpha << endl;
        out << "SELULambda:            " << d._SELULambda << endl; 
        out << "Sparse:                " << ((d._attributes & Layer::Attributes::Sparse) != 0) << endl;
        out << "batchNormalization:    " << ((d._attributes & Layer::Attributes::BatchNormalization) != 0) << endl;
        if (d._type == Layer::Type::FullyConnected)
        {
            if (d._sparsenessPenalty_p > (Float)0.0)
                out << "sparsenessPenalty_p    " << d._sparsenessPenalty_p << endl;
            if (d._sparsenessPenalty_beta > (Float)0.0)
                out << "sparsenessPenalty_beta " << d._sparsenessPenalty_beta << endl;
        }
        if (d._kind != Layer::Kind::Hidden)
            out << "DataSet:               " << d._dataSet << endl;
    }
    for (size_t i = 0 ; i < d._vSource.size(); i++)
    {
        out << "source " << setw(3) << i << ":            " << d._vSource[i] << endl;
    }
    for (size_t i = 0 ; i < d._vSkip.size(); i++)
    {
        out << "skip " << setw(3) << i << ":            " << d._vSkip[i] << endl;
    }     
    return out;
}

uint32_t MPI_Bcast_LayerDescriptor(LayerDescriptor& d)
{
    MPI_Bcast_string(d._name);
    MPI_Bcast(&d._kind, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._type, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._poolingFunction, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD); 
    MPI_Bcast(&d._Nx, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._Ny, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._Nz, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._Nw, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._dimensions, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._bDimensionsProvided, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelX, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelY, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelZ, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelStrideX, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelStrideY, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelStrideZ, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelPaddingX, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelPaddingY, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelPaddingZ, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._pDropout, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._weightInit, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._weightInitScale, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._biasInit, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._weightNorm, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._deltaNorm, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._activation, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._sparsenessPenalty_p, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._sparsenessPenalty_beta, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);    
    MPI_Bcast(&d._attributes, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._RELUSlope, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._ELUAlpha, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._SELULambda, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Bcast_string(d._dataSet);
    size_t size                         = d._vSource.size();
    MPI_Bcast(&size, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    d._vSource.resize(size);
    for (size_t i = 0; i < size; i++)
        MPI_Bcast_string(d._vSource[i]);
    size                                = d._vSkip.size();
    MPI_Bcast(&size, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    d._vSkip.resize(size);
    for (size_t i = 0; i < size; i++)
        MPI_Bcast_string(d._vSkip[i]);        
    return 0;
}

bool Layer::WriteNetCDF(NcFile& nc, uint32_t index)
{
    bool bResult                        = true;
    if (getGpu()._id == 0)
    {
        string lstring                  = "layer" + std::to_string(index) + "_";
        nc.putAtt(lstring + "name", _name);
        nc.putAtt(lstring + "kind", ncUint, _kind);
        nc.putAtt(lstring + "type", ncUint, _type);
        nc.putAtt(lstring + "poolingfunction", ncUint, _poolingFunction);
        nc.putAtt(lstring + "dataSet", _dataSet);
        nc.putAtt(lstring + "Nx", ncUint, _Nx);
        nc.putAtt(lstring + "Ny", ncUint, _Ny);
        nc.putAtt(lstring + "Nz", ncUint, _Nz);
        nc.putAtt(lstring + "Nw", ncUint, _Nw);
        nc.putAtt(lstring + "dimensions", ncUint, _dimensions);
        nc.putAtt(lstring + "kernelX", ncUint, _kernelX);
        nc.putAtt(lstring + "kernelY", ncUint, _kernelY);
        nc.putAtt(lstring + "kernelZ", ncUint, _kernelZ);
        nc.putAtt(lstring + "kernelDimensions", ncUint, _kernelDimensions);
        nc.putAtt(lstring + "kernelStrideX", ncUint, _kernelStrideX);
        nc.putAtt(lstring + "kernelStrideY", ncUint, _kernelStrideY);
        nc.putAtt(lstring + "kernelStrideZ", ncUint, _kernelStrideZ);
        nc.putAtt(lstring + "kernelPaddingX", ncUint, _kernelPaddingX);
        nc.putAtt(lstring + "kernelPaddingY", ncUint, _kernelPaddingY);
        nc.putAtt(lstring + "kernelPaddingZ", ncUint, _kernelPaddingZ);
        nc.putAtt(lstring + "pDropout", ncFloat, _pDropout);
        nc.putAtt(lstring + "weightInit", ncUint, _weightInit);
        nc.putAtt(lstring + "weightInitScale", ncFloat, _weightInitScale);
        nc.putAtt(lstring + "biasInit", ncFloat, _biasInit);
        nc.putAtt(lstring + "weightNorm", ncFloat, _weightNorm);
        nc.putAtt(lstring + "deltaNorm", ncFloat, _deltaNorm);
        nc.putAtt(lstring + "activation", ncUint, _activation);
        nc.putAtt(lstring + "sparsenessPenalty_p", ncFloat, _sparsenessPenalty_p);
        nc.putAtt(lstring + "sparsenessPenalty_beta", ncFloat, _sparsenessPenalty_beta);
        nc.putAtt(lstring + "RELUSlope", ncFloat, _RELUSlope);
        nc.putAtt(lstring + "ELUAlpha", ncFloat, _ELUAlpha);
        nc.putAtt(lstring + "SELULambda", ncFloat, _SELULambda);
                
        uint32_t attributes             = 0;
        if (_bSparse)
            attributes                 |= Layer::Attributes::Sparse;
        if (_bDenoising)
            attributes                 |= Layer::Attributes::Denoising;
        if (_bBatchNormalization)
            attributes                 |= Layer::Attributes::BatchNormalization;
        nc.putAtt(lstring + "attributes", ncUint, attributes);
        nc.putAtt(lstring + "sources", ncUint, (uint32_t)_vSource.size());
        for (size_t i = 0; i < _vSource.size(); i++)
        {
            string nstring             = std::to_string(i);
            nc.putAtt(lstring + "source" + nstring, _vSource[i]);
        }
        nc.putAtt(lstring + "skips", ncUint, (uint32_t)_vSkip.size());        
        for (size_t i = 0; i < _vSkip.size(); i++)
        {
            string nstring             = std::to_string(i);
            nc.putAtt(lstring + "skip" + nstring, _vSkip[i]);
        }

        if (_bBatchNormalization)
        {
            vector<Float>  bndata(_strideBN);
            size_t bytes = _strideBN * sizeof(Float);
            NcDim bnDim   = nc.addDim(lstring + "bnDim", _strideBN);

            cudaMemcpy(bndata.data(), _pbScaleBN->_pDevData, bytes, cudaMemcpyDeviceToHost);
            NcVar scaleVar  = nc.addVar(lstring + "scaleBN", "float", bnDim.getName());
            scaleVar.putVar(bndata.data());

            cudaMemcpy(bndata.data(), _pbBiasBN->_pDevData, bytes, cudaMemcpyDeviceToHost);
            NcVar biasVar  = nc.addVar(lstring + "biasBN", "float", bnDim.getName());
            biasVar.putVar(bndata.data());

            cudaMemcpy(bndata.data(), _pbRunningMeanBN->_pDevData, bytes, cudaMemcpyDeviceToHost);
            NcVar runningMeanVar  = nc.addVar(lstring + "runningMeanBN", "float", bnDim.getName());
            runningMeanVar.putVar(bndata.data());

            cudaMemcpy(bndata.data(), _pbRunningVarianceBN->_pDevData, bytes, cudaMemcpyDeviceToHost);
            NcVar runningVarianceVar  = nc.addVar(lstring + "runningVarianceBN", "float", bnDim.getName());
            runningVarianceVar.putVar(bndata.data());
        }
    }
    else
        bResult                     = false;

    return bResult;
}
