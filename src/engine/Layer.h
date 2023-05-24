#ifndef Layer_H
#define Layer_H

#ifndef __NVCC__
#include <memory>
#include <array>
#include <string_view>
#include <map>
#include <vector>
#include <ostream>
#include <netcdf>

class DataSetBase;
class Network;
class Weight;
class GpuBuffer;

using PoolingFunction = /* define the type of PoolingFunction */;
using Activation = /* define the type of Activation */;
using ErrorFunction = /* define the type of ErrorFunction */;

/**
 * @brief Class representing a layer in a neural network.
 */
class Layer {
public:
    friend class Network;
    friend class Weight;
    friend Network* LoadNeuralNetworkNetCDF(const std::string& fname, int batch);

    /**
     * @brief Enum defining the kind of layer.
     */
    enum Kind
    {
        Input,
        Hidden,
        Output,
        Target,
    };

    /**
     * @brief Enum defining the type of layer.
     */
    enum Type
    {
        FullyConnected,
        Convolutional,
        Pooling
    };

    /**
     * @brief Enum defining additional attributes of the layer.
     */
    enum Attributes
    {
        None                = 0x0,
        Sparse              = 0x1,
        Denoising           = 0x2,
        BatchNormalization  = 0x4,
    };

    /**
     * @brief Enum defining the parallelization strategy for the layer.
     */
    enum Parallelization {
        Data,
        Model,
        Serial,
    };

private:
    const std::string              _name; /**< Name of the layer. */
    const Kind                     _kind; /**< Kind of the layer. */
    const Type                     _type; /**< Type of the layer. */
    const uint32_t                 _attributes; /**< Additional attributes of the layer. */
    PoolingFunction                _poolingFunction; /**< Pooling function for the layer. */
    std::string                    _dataSet; /**< Data set associated with the layer. */
    DataSetBase*                 _pDataSet; /**< Pointer to the data set object. */
    std::vector<std::string>       _vSource; /**< Vector of source strings. */
    std::vector<std::string>       _vSkip; /**< Vector of skip strings. */
    uint32_t                       _Nx; /**< Size of dimension X. */
    uint32_t                       _Ny; /**< Size of dimension Y. */
    uint32_t                       _Nz; /**< Size of dimension Z. */
    uint32_t                       _Nw; /**< Size of dimension W. */
    uint32_t                       _stride; /**< Stride of the layer. */
    uint32_t                       _localStride; /**< Local stride of the layer. */
    uint32_t                       _maxLocalStride; /**< Maximum local stride of the layer. */
    uint32_t                       _strideBN; /**< Stride for batch normalization. */
    uint32_t                       _batch; /**< Batch size. */
    uint32_t                       _localBatch; /**< Local batch size. */
    uint32_t                       _deltaUpdateCount; /**< Delta update count. */
    uint32_t                       _unitUpdateCount; /**< Unit update count. */
    uint32_t                       _dimensions; /**< Number of dimensions of the layer. */
    uint32_t                       _minX; /**< Minimum value of dimension X. */
    uint32_t                       _maxX; /**< Maximum value of dimension X. */
    WeightInitialization           _weightInit; /**< Weight initialization strategy. */
    float                          _weightInitScale; /**< Weight initialization scale. */
    float                          _biasInit; /**< Bias initialization value. */
    float                          _RELUSlope; /**< Slope value for ReLU activation. */
    float                          _ELUAlpha; /**< Alpha value for ELU activation. */
    float                          _SELULambda; /**< Lambda value for SELU activation. */
    bool                           _bBatchNormalization; /**< Flag indicating batch normalization. */
    const uint32_t                 _kernelX; /**< Size of kernel in dimension X. */
    const uint32_t                 _kernelY; /**< Size of kernel in dimension Y. */
    const uint32_t                 _kernelZ; /**< Size of kernel in dimension Z. */
    const uint32_t                 _kernelStrideX; /**< Kernel stride in dimension X. */
    const uint32_t                 _kernelStrideY; /**< Kernel stride in dimension Y. */
    const uint32_t                 _kernelStrideZ; /**< Kernel stride in dimension Z. */
    const uint32_t                 _kernelPaddingX; /**< Kernel padding in dimension X. */
    const uint32_t                 _kernelPaddingY; /**< Kernel padding in dimension Y. */
    const uint32_t                 _kernelPaddingZ; /**< Kernel padding in dimension Z. */
    const uint32_t                 _kernelDimensions; /**< Number of kernel dimensions. */
    const Activation               _activation; /**< Activation function for the layer. */
    const float                    _pDropout; /**< Dropout probability. */
    bool                           _bSparse; /**< Flag indicating sparsity. */
    bool                           _bFastSparse; /**< Flag indicating fast sparsity. */
    float                          _sparsenessPenalty_p; /**< Sparseness penalty p value. */
    float                          _sparsenessPenalty_beta; /**< Sparseness penalty beta value. */
    const bool                     _bDenoising; /**< Flag indicating denoising. */
    float                          _weightNorm; /**< Weight norm value. */
    float                          _deltaNorm; /**< Delta norm value. */
    Parallelization                _parallelization; /**< Parallelization strategy. */
    bool                           _bTransposeParallelization; /**< Flag indicating transpose parallelization. */
    bool                           _bDirty; /**< Flag indicating if the layer is dirty. */
    cudnnTensorDescriptor_t        _scaleBiasMeanVarDescBN; /**< Descriptor for scale, bias, mean, and variance for batch normalization. */
    cudnnTensorDescriptor_t        _tensorDescriptorBN; /**< Tensor descriptor for batch normalization. */
    cudnnTensorDescriptor_t        _tensorDescriptor; /**< Tensor descriptor for the layer. */
    cudnnTensorDescriptor_t        _oddBatchTensorDescriptor; /**< Tensor descriptor for odd batch size. */
    uint32_t                       _oddBatch; /**< Odd batch size. */
    cudnnPoolingDescriptor_t       _poolingDescriptor; /**< Pooling descriptor for the layer. */
    cudnnLRNDescriptor_t           _LRNDescriptor; /**< Local Response Normalization (LRN) descriptor for the layer. */
    std::vector<Layer*>          _vIncomingLayer; /**< Vector of incoming layers. */
    std::vector<Weight*>         _vIncomingWeight; /**< Vector of incoming weights. */
    std::vector<Layer*>          _vOutgoingLayer; /**< Vector of outgoing layers. */
    std::vector<Weight*>         _vOutgoingWeight; /**< Vector of outgoing weights. */
    std::vector<Layer*>          _vIncomingLargerLayer; /**< Vector of incoming larger layers. */
    std::vector<Weight*>         _vIncomingLargerWeight; /**< Vector of incoming larger weights. */
    std::vector<Layer*>          _vOutgoingLargerLayer; /**< Vector of outgoing larger layers. */
    std::vector<Weight*>         _vOutgoingLargerWeight; /**< Vector of outgoing larger weights. */
    std::vector<Layer*>          _vIncomingSkip; /**< Vector of incoming skip layers. */
    std::vector<Layer*>          _vOutgoingSkip; /**< Vector of outgoing skip layers. */
    std::vector<float>             _vUnit; /**< Vector of unit values. */
    std::vector<float>             _vDelta; /**< Vector of delta values. */
    std::vector<float>             _vBuffer1; /**< Vector of buffer values 1. */
    std::vector<float>             _vBuffer2; /**< Vector of buffer values 2. */
    std::unique_ptr<GpuBuffer<float>> _pbUnit; /**< Pointer to GPU buffer for unit values. */
    std::unique_ptr<GpuBuffer<float>> _pbDelta; /**< Pointer to GPU buffer for delta values. */
    std::unique_ptr<GpuBuffer<float>> _pbDropout; /**< Pointer to GPU buffer for dropout values. */
    std::unique_ptr<GpuBuffer<float>> _pbBuffer1; /**< Pointer to GPU buffer for buffer values 1. */
    std::unique_ptr<GpuBuffer<float>> _pbBuffer2; /**< Pointer to GPU buffer for buffer values 2. */
    std::unique_ptr<GpuBuffer<float>> _pbDeltaBN; /**< Pointer to GPU buffer for batch normalized delta values. */
    std::unique_ptr<GpuBuffer<float>> _pbScaleGradientBN; /**< Pointer to GPU buffer for batch normalization scale gradient. */
    std::unique_ptr<GpuBuffer<float>> _pbBiasGradientBN; /**< Pointer to GPU buffer for batch normalization bias gradient. */
    std::unique_ptr<GpuBuffer<float>> _pbUnitBN; /**< Pointer to GPU buffer for batch normalized unit values. */
    std::unique_ptr<GpuBuffer<float>> _pbScaleBN; /**< Pointer to GPU buffer for batch normalization scale values. */
    std::unique_ptr<GpuBuffer<float>> _pbBiasBN; /**< Pointer to GPU buffer for batch normalization bias values. */
    std::unique_ptr<GpuBuffer<float>> _pbScaleVelocityBN; /**< Pointer to GPU buffer for batch normalization scale velocity. */
    std::unique_ptr<GpuBuffer<float>> _pbBiasVelocityBN; /**< Pointer to GPU buffer for batch normalization bias velocity. */
    std::unique_ptr<GpuBuffer<float>> _pbScaleGradientVelocityBN; /**< Pointer to GPU buffer for batch normalization scale gradient velocity. */
    std::unique_ptr<GpuBuffer<float>> _pbBiasGradientVelocityBN; /**< Pointer to GPU buffer for batch normalization bias gradient velocity. */
    std::unique_ptr<GpuBuffer<float>> _pbRunningMeanBN; /**< Pointer to GPU buffer for running mean in batch normalization. */
    std::unique_ptr<GpuBuffer<float>> _pbRunningVarianceBN; /**< Pointer to GPU buffer for running variance in batch normalization. */
    std::unique_ptr<GpuBuffer<float>> _pbSaveMeanBN; /**< Pointer to GPU buffer for saved mean in batch normalization. */
    std::unique_ptr<GpuBuffer<float>> _pbSaveInvVarianceBN; /**< Pointer to GPU buffer for saved inverse variance in batch normalization. */
    uint32_t                       _bnCalls; /**< Number of batch normalization calls. */
    int32_t                        _priority; /**< Priority of the layer. */

public:
    /**
     * @brief Constructor for the Layer class.
     *
     * @param l LayerDescriptor object containing the layer information.
     * @param batch Batch size.
     */
    Layer(LayerDescriptor& l, uint32_t batch);

    /**
     * @brief Destructor for the Layer class.
     */
    ~Layer();

    /**
     * @brief Allocates memory for the layer.
     *
     * @param validate Flag indicating whether to validate the allocation.
     */
    void Allocate(bool validate);

    /**
     * @brief Deallocates memory for the layer.
     */
    void Deallocate();

    /**
     * @brief Sets the batch size for the layer.
     *
     * @param batch Batch size.
     */
    void SetBatch(uint32_t batch);

    /**
     * @brief Refreshes the parallelization strategy of the layer.
     */
    void RefreshParallelization();

    /**
     * @brief Refreshes the state of the layer.
     *
     * @param pNetwork Pointer to the Network object.
     * @param trainingMode Training mode.
     * @param validate Flag indicating whether to validate the state.
     */
    void RefreshState(Network* pNetwork, TrainingMode trainingMode, bool validate);

    /**
     * @brief Loads the prediction batch for the layer.
     *
     * @param position Position of the batch.
     * @param batch Batch size.
     */
    void LoadPredictionBatch(uint32_t position, uint32_t batch);

    /**
     * @brief Loads the training batch for the layer.
     *
     * @param position Position of the batch.
     * @param batch Batch size.
     */
    void LoadTrainingBatch(uint32_t position, uint32_t batch);

    /**
     * @brief Loads the validation batch for the layer.
     *
     * @param position Position of the batch.
     * @param batch Batch size.
     */
    void LoadValidationBatch(uint32_t position, uint32_t batch);

    /**
     * @brief Performs forward propagation for the layer.
     *
     * @param position Position of the batch.
     * @param batch Batch size.
     * @param bTraining Flag indicating whether training is being performed.
     */
    void ForwardPropagate(uint32_t position, uint32_t batch, bool bTraining = false);

    /**
     * @brief Performs forward propagation for a fully connected layer.
     *
     * @param position Position of the batch.
     * @param batch Batch size.
     * @param bTraining Flag indicating whether training is being performed.
     */
    void ForwardPropagateFullyConnected(uint32_t position, uint32_t batch, bool bTraining);

    /**
     * @brief Performs forward propagation for a convolutional layer.
     *
     * @param position Position of the batch.
     * @param batch Batch size.
     * @param bTraining Flag indicating whether training is being performed.
     */
    void ForwardPropagateConvolutional(uint32_t position, uint32_t batch, bool bTraining);

    /**
     * @brief Performs forward propagation for a pooling layer.
     *
     * @param position Position of the batch.
     * @param batch Batch size.
     * @param bTraining Flag indicating whether training is being performed.
     */
    void ForwardPropagatePooling(uint32_t position, uint32_t batch, bool bTraining);

    /**
     * @brief Calculates the activation for the layer.
     *
     * @param batch Batch size.
     */
    void CalculateActivation(uint32_t batch);

    /**
     * @brief Calculates the dropout for the layer.
     *
     * @param batch Batch size.
     */
    void CalculateDropout(uint32_t batch);

    /**
     * @brief Calculates the error for the layer.
     *
     * @param position Position of the batch.
     * @param batch Batch size.
     * @param ef Error function.
     *
     * @return The calculated error.
     */
    float CalculateError(uint32_t position, uint32_t batch, ErrorFunction ef);

    /**
     * @brief Performs backpropagation for the layer.
     *
     * @param position Position of the batch.
     * @param batch Batch size.
     */
    void BackPropagate(uint32_t position, uint32_t batch);

    /**
     * @brief Performs backpropagation for a fully connected layer.
     *
     * @param position Position of the batch.
     * @param batch Batch size.
     */
    void BackPropagateFullyConnected(uint32_t position, uint32_t batch);

    /**
     * @brief Performs backpropagation for a convolutional layer.
     *
     * @param position Position of the batch.
     * @param batch Batch size.
     */
    void BackPropagateConvolutional(uint32_t position, uint32_t batch);

    /**
     * @brief Performs backpropagation for a pooling layer.
     *
     * @param position Position of the batch.
     * @param batch Batch size.
     */
    void BackPropagatePooling(uint32_t position, uint32_t batch);

    /**
     * @brief Calculates the output delta for the layer.
     *
     * @param position Position of the batch.
     * @param batch Batch size.
     * @param ef Error function.
     */
    void CalculateOutputDelta(uint32_t position, uint32_t batch, ErrorFunction ef);

    /**
     * @brief Updates the weights of the layer.
     *
     * @param trainingMode Training mode.
     * @param batch Batch size.
     * @param alpha Learning rate.
     * @param lambda L2 regularization factor.
     * @param lambda1 L1 regularization factor.
     * @param mu Momentum factor.
     * @param mu1 Nesterov momentum factor.
     * @param t Time step.
     */
    void UpdateWeights(TrainingMode trainingMode, uint32_t batch, float alpha, float lambda, float lambda1, float mu, float mu1, float t);

    /**
     * @brief Generates denoising data for the layer.
     */
    void GenerateDenoisingData();

    /**
     * @brief Reduces the size of the layer.
     *
     * @param batch Batch size.
     * @param stride Stride of the layer.
     * @param pBuffer Pointer to the buffer.
     * @param localStride Local stride of the layer.
     * @param updateCount Update count.
     */
    void Reduce(uint32_t batch, uint32_t stride, float* pBuffer, uint32_t localStride, uint32_t updateCount);

    /**
     * @brief Gathers the size of the layer.
     *
     * @param batch Batch size.
     * @param stride Stride of the layer.
     * @param pBuffer Pointer to the buffer.
     * @param localStride Local stride of the layer.
     */
    void Gather(uint32_t batch, uint32_t stride, float* pBuffer, uint32_t localStride);

    /**
     * @brief Clears the updates of the layer.
     */
    void ClearUpdates();

    /**
     * @brief Dumps the layer to a file.
     *
     * @param fname File name.
     * @param pData Pointer to the data.
     */
    void Dump(std::string fname, float* pData);

    /**
     * @brief Writes the layer to a NetCDF file.
     *
     * @param nc NetCDF file object.
     * @param index Index of the layer.
     *
     * @return True if writing is successful, false otherwise.
     */
    bool WriteNetCDF(netCDF::NcFile& nc, uint32_t index);

    /**
     * @brief Returns the buffer for incoming unit values.
     *
     * @return Pointer to the buffer.
     */
    float* GetIncomingUnitBuffer() 
    { 
        if (_bBatchNormalization)
            return _pbUnitBN ? _pbUnitBN->_pDevData : nullptr;
        else
            return _pbUnit ? _pbUnit->_pDevData : nullptr;
    }

    /**
     * @brief Returns the buffer for unit values.
     *
     * @return Pointer to the buffer.
     */
    float* GetUnitBuffer() { return _pbUnit ? _pbUnit->_pDevData : nullptr; }

    /**
     * @brief Returns the buffer for incoming delta values.
     *
     * @return Pointer to the buffer.
     */
    float* GetIncomingDeltaBuffer() 
    { 
        if (_bBatchNormalization)
            return _pbDeltaBN ? _pbDeltaBN->_pDevData : nullptr;
        else
            return _pbDelta ? _pbDelta->_pDevData : nullptr;
    }

    /**
     * @brief Returns the buffer for delta values.
     *
     * @return Pointer to the buffer.
     */
    float* GetDeltaBuffer() { return _pbDelta ? _pbDelta->_pDevData : nullptr; }

    /**
     * @brief Returns the size of the buffer.
     *
     * @return Size of the buffer.
     */
    uint64_t GetBufferSize() { return _batch * _stride; }

    /**
     * @brief Returns the tensor descriptor for the layer.
     *
     * @param batch Batch size.
     *
     * @return cudnnTensorDescriptor_t object.
     */
    cudnnTensorDescriptor_t getTensorDescriptor(uint32_t batch);

    /**
     * @brief Returns the tensor descriptor for batch normalization.
     *
     * @param batch Batch size.
     *
     * @return cudnnTensorDescriptor_t object.
     */
    cudnnTensorDescriptor_t getTensorDescriptorBN(uint32_t batch);
};

/**
 * @brief Overload of the output stream operator for Layer::Kind enum.
 *
 * @param out Reference to the output stream.
 * @param k Layer::Kind value.
 *
 * @return Reference to the output stream.
 */
std::ostream& operator<< (std::ostream& out, Layer::Kind& k);

/**
 * @brief Overload of the output stream operator for Layer::Type enum.
 *
 * @param out Reference to the output stream.
 * @param t Layer::Type value.
 *
 * @return Reference to the output stream.
 */
std::ostream& operator<< (std::ostream& out, Layer::Type& t);

/**
 * @brief Overload of the output stream operator for Layer::Parallelization enum.
 *
 * @param out Reference to the output stream.
 * @param p Layer::Parallelization value.
 *
 * @return Reference to the output stream.
 */
std::ostream& operator<< (std::ostream& out, Layer::Parallelization& p);

/**
 * @brief Overload of the output stream operator for Layer::Attributes enum.
 *
 * @param out Reference to the output stream.
 * @param a Layer::Attributes value.
 *
 * @return Reference to the output stream.
 */
std::ostream& operator<< (std::ostream& out, Layer::Attributes& a);

/**
 * @brief Struct representing the descriptor of a layer.
 */
struct LayerDescriptor
{
    std::string                  _name; /**< Name of the layer. */
    Layer::Kind                _kind; /**< Kind of the layer. */
    Layer::Type                _type; /**< Type of the layer. */
    PoolingFunction              _poolingFunction; /**< Pooling function for the layer. */
    std::string                  _dataSet; /**< Data set associated with the layer. */
    std::vector<std::string>     _vSource; /**< Vector of source strings. */
    std::vector<std::string>     _vSkip; /**< Vector of skip strings. */
    uint32_t                     _Nx; /**< Size of the layer in dimension X. */
    uint32_t                     _Ny; /**< Size of the layer in dimension Y. */
    uint32_t                     _Nz; /**< Size of the layer in dimension Z. */
    uint32_t                     _Nw; /**< Size of the layer in dimension W. */
    uint32_t                     _dimensions; /**< Number of dimensions of the layer. */
    bool                         _bDimensionsProvided; /**< Flag indicating whether dimensions are provided. */
    WeightInitialization         _weightInit; /**< Weight initialization method. */
    float                        _weightInitScale; /**< Weight initialization scale. */
    float                        _biasInit; /**< Bias initialization value. */
    uint32_t                     _kernelX; /**< Size of kernel in dimension X. */
    uint32_t                     _kernelY; /**< Size of kernel in dimension Y. */
    uint32_t                     _kernelZ; /**< Size of kernel in dimension Z. */
    uint32_t                     _kernelStrideX; /**< Kernel stride in dimension X. */
    uint32_t                     _kernelStrideY; /**< Kernel stride in dimension Y. */
    uint32_t                     _kernelStrideZ; /**< Kernel stride in dimension Z. */
    uint32_t                     _kernelPaddingX; /**< Kernel padding in dimension X. */
    uint32_t                     _kernelPaddingY; /**< Kernel padding in dimension Y. */
    uint32_t                     _kernelPaddingZ; /**< Kernel padding in dimension Z. */
    uint32_t                     _kernelDimensions; /**< Number of kernel dimensions. */
    std::vector<float>           _vScaleBN; /**< Vector of scale values for batch normalization. */
    std::vector<float>           _vBiasBN; /**< Vector of bias values for batch normalization. */
    std::vector<float>           _vRunningMeanBN; /**< Vector of running mean values for batch normalization. */
    std::vector<float>           _vRunningVarianceBN; /**< Vector of running variance values for batch normalization. */
    float                        _weightNorm; /**< Weight norm value. */
    float                        _deltaNorm; /**< Delta norm value. */
    float                        _pDropout; /**< Dropout probability. */
    Activation                   _activation; /**< Activation function for the layer. */
    float                        _sparsenessPenalty_p; /**< Sparseness penalty p value. */
    float                        _sparsenessPenalty_beta; /**< Sparseness penalty beta value. */
    uint32_t                     _attributes; /**< Attributes of the layer. */
    float                        _RELUSlope; /**< Slope value for ReLU activation. */
    float                        _ELUAlpha; /**< Alpha value for ELU activation. */
    float                        _SELULambda; /**< Lambda value for SELU activation. */

    /**
     * @brief Default constructor for the LayerDescriptor struct.
     */
    LayerDescriptor();
};

/**
 * @brief Loads a layer descriptor from a NetCDF file.
 *
 * @param fname File name.
 * @param nc NetCDF file object.
 * @param index Index of the layer.
 * @param ld Reference to the LayerDescriptor object.
 *
 * @return True if loading is successful, false otherwise.
 */
bool LoadLayerDescriptorNetCDF(const std::string& fname, netCDF::NcFile& nc, uint32_t index, LayerDescriptor& ld);

/**
 * @brief Overload of the output stream operator for LayerDescriptor struct.
 *
 * @param out Reference to the output stream.
 * @param d LayerDescriptor object.
 *
 * @return Reference to the output stream.
 */
std::ostream& operator<< (std::ostream& out, LayerDescriptor& d);

/**
 * @brief Broadcasts a LayerDescriptor object using MPI.
 *
 * @param d Reference to the LayerDescriptor object.
 *
 * @return The size of the serialized data.
 */
uint32_t MPI_Bcast_LayerDescriptor(LayerDescriptor& d);

#endif
#endif