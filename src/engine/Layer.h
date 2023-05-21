#ifndef LAYER_H
#define LAYER_H
#ifndef __NVCC__
#include <memory>
#include <string>
#include <array>
#include <map>
#include <vector>
#include <tuple>
#include <iostream>
#include <string_view>

class LayerDescriptor;
/**
 * @brief Class representing a Layer in a neural network.
 */
class Layer {
public:
    /**
     * @brief Friend class declaration for Network.
     *
     * Network class is a friend of Layer class, allowing access to its private and protected members.
     */
    friend class Network;

    /**
     * @brief Friend class declaration for Weight.
     *
     * Weight class is a friend of Layer class, allowing access to its private and protected members.
     */
    friend class Weight;

    /**
     * @brief Friend function declaration for LoadNeuralNetworkNetCDF.
     *
     * LoadNeuralNetworkNetCDF function is a friend of Layer class, allowing access to its private and protected members.
     *
     * @param fname The file name of the neural network in NetCDF format.
     * @param batch The batch size for the neural network.
     * @return A pointer to the loaded neural network.
     */
    friend Network* LoadNeuralNetworkNetCDF(const std::string_view fname, int batch);

    /**
     * @brief Enum representing the kind of layer.
     */
    enum class Kind {
        /**
         * @brief Input layer kind.
         */
        Input,

        /**
         * @brief Hidden layer kind.
         */
        Hidden,

        /**
         * @brief Output layer kind.
         */
        Output,

        /**
         * @brief Target layer kind.
         */
        Target,
    };

    /**
     * @brief Array of pairs mapping Layer::Kind enumeration to string representation.
     */
    static std::array<std::pair<Kind, std::string>, 4> _sKindPair;

    /**
     * @brief Map mapping Layer::Kind enumeration to string representation.
     */
    static std::map<Kind, std::string> _sKindMap;

    /**
     * @brief Enum representing the type of layer.
     */
    enum class Type {
        /**
         * @brief Fully connected layer type.
         */
        FullyConnected,

        /**
         * @brief Convolutional layer type.
         */
        Convolutional,

        /**
         * @brief Pooling layer type.
         */
        Pooling,
    };

    /**
     * @brief Array of pairs mapping Layer::Type enumeration to string representation.
     */
    static std::array<std::pair<Type, std::string>, 3> _sTypePair;

    /**
     * @brief Map mapping Layer::Type enumeration to string representation.
     */
    static std::map<Type, std::string> _sTypeMap;

    /**
     * @brief Enum representing the attributes of a layer.
     */
    enum class Attributes {
        /**
         * @brief No additional attributes.
         */
        None                = 0x0,

        /**
         * @brief Sparse attribute.
         */
        Sparse              = 0x1,

        /**
         * @brief Denoising attribute.
         */
        Denoising           = 0x2,

        /**
         * @brief Batch normalization attribute.
         */
        BatchNormalization  = 0x4,
    };

    /**
     * @brief Array of pairs mapping Layer::Attributes enumeration to string representation.
     */
    static std::array<std::pair<Attributes, std::string>, 5> _sAttributesPair;

    /**
     * @brief Map mapping Layer::Attributes enumeration to string representation.
     */
    static std::map<Attributes, std::string> _sAttributesMap;

    /**
     * @brief Enum representing the parallelization options for a layer.
     */
    enum class Parallelization {
        /**
         * @brief File containing data.
         */
        Data,

        /**
         * @brief File containing model.
         */
        Model,

        /**
         * @brief Serial file.
         */
        Serial,
    };

    /**
     * @brief Array of pairs mapping Layer::Parallelization enumeration to string representation.
     */
    static std::array<std::pair<Parallelization, std::string>, 3> _sParallelizationPair;

    /**
     * @brief Map mapping Layer::Parallelization enumeration to string representation.
     */
    static std::map<Parallelization, std::string> _sParallelizationMap;

private:
    /**
     * @brief The name of the class.
     */
    const std::string _name;

    /**
     * @brief The kind of the class.
     */
    const Kind _kind;

    /**
     * @brief The type of the class.
     */
    const Type _type;

    /**
     * @brief The attributes of the class.
     */
    const uint32_t _attributes;

    /**
     * @brief The pooling function used by the class.
     */
    PoolingFunction _poolingFunction;

    /**
     * @brief The data set associated with the class.
     */
    std::string _dataSet;

    /**
     * @brief A pointer to the base data set.
     */
    DataSetBase* _pDataSet;

    /**
     * @brief A vector of source strings.
     */
    std::vector<std::string> _vSource;

    /**
     * @brief A vector of strings to skip.
     */
    std::vector<std::string> _vSkip;

    /**
     * @brief The value of Nx.
     */
    uint32_t _Nx;

    /**
     * @brief The value of Ny.
     */
    uint32_t _Ny;

    /**
     * @brief The value of Nz.
     */
    uint32_t _Nz;

    /**
     * @brief The value of Nw.
     */
    uint32_t _Nw;

    /**
     * @brief The stride value.
     */
    uint32_t _stride;

    /**
     * @brief The local stride value.
     */
    uint32_t _localStride;

    /**
     * @brief The maximum local stride value.
     */
    uint32_t _maxLocalStride;

    /**
     * @brief The stride value used for batch normalization.
     */
    uint32_t _strideBN;

    /**
     * @brief The batch value.
     */
    uint32_t _batch;

    /**
     * @brief The local batch value.
     */
    uint32_t _localBatch;

    /**
     * @brief The delta update count value.
     */
    uint32_t _deltaUpdateCount;

    /**
     * @brief The unit update count value.
     */
    uint32_t _unitUpdateCount;

    /**
     * @brief The number of dimensions.
     */
    uint32_t _dimensions;

    /**
     * @brief The minimum value of X.
     */
    uint32_t _minX;

    /**
     * @brief The maximum value of X.
     */
    uint32_t _maxX;

    /**
     * @brief The weight initialization method.
     */
    WeightInitialization _weightInit;

    /**
     * @brief The weight initialization scale.
     */
    Float _weightInitScale;
    /**
     * @brief The bias initialization value.
     */
    Float _biasInit;

    /**
     * @brief The slope for the rectified linear unit (ReLU) activation function.
     */
    Float _RELUSlope;

    /**
     * @brief The alpha value for the exponential linear unit (ELU) activation function.
     */
    Float _ELUAlpha;

    /**
     * @brief The lambda value for the scaled exponential linear unit (SELU) activation function.
     */
    Float _SELULambda;

    /**
     * @brief Indicates whether batch normalization is enabled.
     */
    bool _bBatchNormalization;

    /**
     * @brief The X dimension of the kernel.
     */
    const uint32_t _kernelX;

    /**
     * @brief The Y dimension of the kernel.
     */
    const uint32_t _kernelY;

    /**
     * @brief The Z dimension of the kernel.
     */
    const uint32_t _kernelZ;

    /**
     * @brief The stride value along the X dimension of the kernel.
     */
    const uint32_t _kernelStrideX;

    /**
     * @brief The stride value along the Y dimension of the kernel.
     */
    const uint32_t _kernelStrideY;

    /**
     * @brief The stride value along the Z dimension of the kernel.
     */
    const uint32_t _kernelStrideZ;

    /**
     * @brief The padding value along the X dimension of the kernel.
     */
    const uint32_t _kernelPaddingX;

    /**
     * @brief The padding value along the Y dimension of the kernel.
     */
    const uint32_t _kernelPaddingY;

    /**
     * @brief The padding value along the Z dimension of the kernel.
     */
    const uint32_t _kernelPaddingZ;

    /**
     * @brief The number of dimensions in the kernel.
     */
    const uint32_t _kernelDimensions;

    /**
     * @brief The activation function used by the layer.
     */
    const Activation _activation;

    /**
     * @brief The dropout probability value.
     */
    const Float _pDropout;

    /**
     * @brief Indicates whether the layer is sparse.
     */
    bool _bSparse;

    /**
     * @brief Indicates whether fast sparsity is enabled.
     */
    bool _bFastSparse;

    /**
     * @brief The p value for sparseness penalty.
     */
    Float _sparsenessPenalty_p;

    /**
     * @brief The beta value for sparseness penalty.
     */
    Float _sparsenessPenalty_beta;

    /**
     * @brief Indicates whether denoising is enabled.
     */
    const bool _bDenoising;

    /**
     * @brief The weight normalization value.
     */
    Float _weightNorm;

    /**
     * @brief The delta normalization value.
     */
    Float _deltaNorm;

    /**
     * @brief The parallelization type.
     */
    Parallelization _parallelization;

    /**
     * @brief Indicates whether transpose parallelization is enabled.
     */
    bool _bTransposeParallelization;

    /**
     * @brief Indicates whether the layer is dirty and needs updating.
     */
    bool _bDirty;

    /**
     * @brief The descriptor for scale, bias, mean, and variance in batch normalization.
     */
    cudTensorDescriptor_t _scaleBiasMeanVarDescBN;

    /**
     * @brief The tensor descriptor for batch normalization.
     */
    cudTensorDescriptor_t _tensorDescriptorBN;

    /**
     * @brief The tensor descriptor for the layer.
     */
    cudTensorDescriptor_t _tensorDescriptor;

    /**
     * @brief The tensor descriptor for handling odd-sized batches.
     */
    cudTensorDescriptor_t _oddBatchTensorDescriptor;

    /**
     * @brief The number of odd-sized batches.
     */
    uint32_t _oddBatch;

    /**
     * @brief The descriptor for the pooling operation.
     */
    cudPoolingDescriptor_t _poolingDescriptor;

    /**
     * @brief The descriptor for the local response normalization (LRN) operation.
     */
    cudLRNDescriptor_t _LRNDescriptor;

    /**
     * @brief The vector of incoming layers.
     */
    std::vector<Layer*> _vIncomingLayer;

    /**
     * @brief The vector of incoming weights.
     */
    std::vector<Weight*> _vIncomingWeight;

    /**
     * @brief The vector of outgoing layers.
     */
    std::vector<Layer*> _vOutgoingLayer;

    /**
     * @brief The vector of outgoing weights.
     */
    std::vector<Weight*> _vOutgoingWeight;

    /**
     * @brief The vector of incoming larger layers.
     */
    std::vector<Layer*> _vIncomingLargerLayer;

    /**
     * @brief The vector of incoming larger weights.
     */
    std::vector<Weight*> _vIncomingLargerWeight;

    /**
     * @brief The vector of outgoing larger layers.
     */
    std::vector<Layer*> _vOutgoingLargerLayer;

    /**
     * @brief The vector of outgoing larger weights.
     */
    std::vector<Weight*> _vOutgoingLargerWeight;

    /**
     * @brief The vector of incoming skip layers.
     */
    std::vector<Layer*> _vIncomingSkip;

    /**
     * @brief The vector of outgoing skip layers.
     */
    std::vector<Layer*> _vOutgoingSkip;

    /**
     * @brief The vector of unit values.
     */
    std::vector<Float> _vUnit;

    /**
     * @brief The vector of delta values.
     */
    std::vector<Float> _vDelta;

    /**
     * @brief The first buffer vector.
     */
    std::vector<Float> _vBuffer1;

    /**
     * @brief The second buffer vector.
     */
    std::vector<Float> _vBuffer2;

    /**
     * @brief The unique pointer to the GPU buffer for unit values.
     */
    std::unique_ptr<GpuBuffer<Float>> _pbUnit;

    /**
     * @brief The unique pointer to the GPU buffer for delta values.
     */
    std::unique_ptr<GpuBuffer<Float>> _pbDelta;

    /**
     * @brief The unique pointer to the GPU buffer for dropout values.
     */
    std::unique_ptr<GpuBuffer<Float>> _pbDropout;

    /**
     * @brief The unique pointer to the GPU buffer for buffer1.
     */
    std::unique_ptr<GpuBuffer<Float>> _pbBuffer1;

    /**
     * @brief The unique pointer to the GPU buffer for buffer2.
     */
    std::unique_ptr<GpuBuffer<Float>> _pbBuffer2;

    /**
     * @brief The unique pointer to the GPU buffer for delta in batch normalization.
     */
    std::unique_ptr<GpuBuffer<Float>> _pbDeltaBN;

    /**
     * @brief The unique pointer to the GPU buffer for scale gradient in batch normalization.
     */
    std::unique_ptr<GpuBuffer<Float>> _pbScaleGradientBN;

    /**
     * @brief The unique pointer to the GPU buffer for bias gradient in batch normalization.
     */
    std::unique_ptr<GpuBuffer<Float>> _pbBiasGradientBN;

    /**
     * @brief The unique pointer to the GPU buffer for unit values in batch normalization.
     */
    std::unique_ptr<GpuBuffer<Float>> _pbUnitBN;

    /**
     * @brief The unique pointer to the GPU buffer for scale values in batch normalization.
     */
    std::unique_ptr<GpuBuffer<Float>> _pbScaleBN;

    /**
     * @brief The unique pointer to the GPU buffer for bias values in batch normalization.
     */
    std::unique_ptr<GpuBuffer<Float>> _pbBiasBN;

    /**
     * @brief The unique pointer to the GPU buffer for scale velocity in batch normalization.
     */
    std::unique_ptr<GpuBuffer<Float>> _pbScaleVelocityBN;

    /**
     * @brief The unique pointer to the GPU buffer for bias velocity in batch normalization.
     */
    std::unique_ptr<GpuBuffer<Float>> _pbBiasVelocityBN;

    /**
     * @brief The unique pointer to the GPU buffer for scale gradient velocity in batch normalization.
     */
    std::unique_ptr<GpuBuffer<Float>> _pbScaleGradientVelocityBN;

    /**
     * @brief The unique pointer to the GPU buffer for bias gradient velocity in batch normalization.
     */
    std::unique_ptr<GpuBuffer<Float>> _pbBiasGradientVelocityBN;

    /**
     * @brief The unique pointer to the GPU buffer for running mean in batch normalization.
     */
    std::unique_ptr<GpuBuffer<Float>> _pbRuingMeanBN;

    /**
     * @brief The unique pointer to the GPU buffer for running variance in batch normalization.
     */
    std::unique_ptr<GpuBuffer<Float>> _pbRuingVarianceBN;

    /**
     * @brief The unique pointer to the GPU buffer for saving mean in batch normalization.
     */
    std::unique_ptr<GpuBuffer<Float>> _pbSaveMeanBN;

    /**
     * @brief The unique pointer to the GPU buffer for saving inverse variance in batch normalization.
     */
    std::unique_ptr<GpuBuffer<Float>> _pbSaveInvVarianceBN;

    /**
     * @brief The number of batch normalization calls.
     */
    uint32_t _bnCalls;

    /**
     * @brief The priority of the layer.
     */
    int32_t _priority;

    Layer(LayerDescriptor& l, uint32_t batch);
    ~Layer();
    /**
     * @brief Allocates memory for the layer.
     * @param validate Flag indicating whether to validate the allocation.
     */
    void Allocate(bool validate);

    /**
     * @brief Deallocates memory for the layer.
     */
    void Deallocate();

    /**
     * @brief Sets the batch size for the layer.
     * @param batch The batch size to set.
     */
    void SetBatch(uint32_t batch);

    /**
     * @brief Refreshes the parallelization settings of the layer.
     */
    void RefreshParallelization();

    /**
     * @brief Refreshes the state of the layer.
     * @param pNetwork Pointer to the network containing the layer.
     * @param trainingMode The training mode.
     * @param validate Flag indicating whether to validate the refreshed state.
     */
    void RefreshState(Network* pNetwork, TrainingMode trainingMode, bool validate);

    /**
     * @brief Loads the input batch for prediction.
     * @param position The position of the batch.
     * @param batch The batch size.
     */
    void LoadPredictionBatch(uint32_t position, uint32_t batch);

    /**
     * @brief Loads the input batch for training.
     * @param position The position of the batch.
     * @param batch The batch size.
     */
    void LoadTrainingBatch(uint32_t position, uint32_t batch);

    /**
     * @brief Loads the input batch for validation.
     * @param position The position of the batch.
     * @param batch The batch size.
     */
    void LoadValidationBatch(uint32_t position, uint32_t batch);

    /**
     * @brief Performs forward propagation for the layer.
     * @param position The position of the input batch.
     * @param batch The batch size.
     * @param bTraining Flag indicating whether it is a training mode.
     */
    void ForwardPropagate(uint32_t position, uint32_t batch, bool bTraining = false);

    /**
     * @brief Performs forward propagation for the fully connected layer.
     * @param position The position of the input batch.
     * @param batch The batch size.
     * @param bTraining Flag indicating whether it is a training mode.
     */
    void ForwardPropagateFullyConnected(uint32_t position, uint32_t batch, bool bTraining);

    /**
     * @brief Performs forward propagation for the convolutional layer.
     * @param position The position of the input batch.
     * @param batch The batch size.
     * @param bTraining Flag indicating whether it is a training mode.
     */
    void ForwardPropagateConvolutional(uint32_t position, uint32_t batch, bool bTraining);

    /**
     * @brief Performs forward propagation for the pooling layer.
     * @param position The position of the input batch.
     * @param batch The batch size.
     * @param bTraining Flag indicating whether it is a training mode.
     */
    void ForwardPropagatePooling(uint32_t position, uint32_t batch, bool bTraining);

    /**
     * @brief Calculates the activation values for the layer.
     * @param batch The batch size.
     */
    void CalculateActivation(uint32_t batch);

    /**
     * @brief Calculates the dropout values for the layer.
     * @param batch The batch size.
     */
    void CalculateDropout(uint32_t batch);

    /**
     * @brief Calculates the error for the layer.
     * @param position The position of the input batch.
     * @param batch The batch size.
     * @param ef The error function to use.
     * @return The calculated error.
     */
    Float CalculateError(uint32_t position, uint32_t batch, ErrorFunction ef);

    /**
     * @brief Performs backpropagation for the layer.
     * @param position The position of the input batch.
     * @param batch The batch size.
     */
    void BackPropagate(uint32_t position, uint32_t batch);

    /**
     * @brief Performs backpropagation for the fully connected layer.
     * @param position The position of the input batch.
     * @param batch The batch size.
     */
    void BackPropagateFullyConnected(uint32_t position, uint32_t batch);

    /**
     * @brief Performs backpropagation for the convolutional layer.
     * @param position The position of the input batch.
     * @param batch The batch size.
     */
    void BackPropagateConvolutional(uint32_t position, uint32_t batch);

    /**
     * @brief Performs backpropagation for the pooling layer.
     * @param position The position of the input batch.
     * @param batch The batch size.
     */
    void BackPropagatePooling(uint32_t position, uint32_t batch);

    /**
     * @brief Calculates the output delta for the layer.
     * @param position The position of the input batch.
     * @param batch The batch size.
     * @param ef The error function to use.
     */
    void CalculateOutputDelta(uint32_t position, uint32_t batch, ErrorFunction ef);

    /**
     * @brief Updates the weights of the layer.
     * @param trainingMode The training mode.
     * @param batch The batch size.
     * @param alpha The learning rate.
     * @param lambda The weight decay factor.
     * @param lambda1 The L1 regularization factor.
     * @param mu The momentum factor.
     * @param mu1 The Nesterov momentum factor.
     * @param t The time step.
     */
    void UpdateWeights(TrainingMode trainingMode, uint32_t batch, Float alpha, Float lambda, Float lambda1, Float mu, Float mu1, Float t);

    /**
     * @brief Generates denoising data for the layer.
     */
    void GenerateDenoisingData();

    /**
     * @brief Reduces the data for parallelization.
     * @param batch The batch size.
     * @param stride The stride value.
     * @param pBuffer The data buffer.
     * @param localStride The local stride value.
     * @param updateCount The update count.
     */
    void Reduce(uint32_t batch, uint32_t stride, Float* pBuffer, uint32_t localStride, uint32_t updateCount);

    /**
     * @brief Gathers the data for parallelization.
     * @param batch The batch size.
     * @param stride The stride value.
     * @param pBuffer The data buffer.
     * @param localStride The local stride value.
     */
    void Gather(uint32_t batch, uint32_t stride, Float* pBuffer, uint32_t localStride);

    /**
     * @brief Clears the weight updates.
     */
    void ClearUpdates();

    /**
     * @brief Dumps the layer's data to a file.
     * @param fname The filename to dump the data.
     * @param pData The data to be dumped.
     */
    void Dump(std::string fname, Float* pData);

    /**
     * @brief Writes the layer's data to a netCDF file.
     * @param nc The netCDF file object.
     * @param index The index of the layer.
     * @return True if writing is successful, false otherwise.
     */
    bool WriteNetCDF(netCDF::NcFile& nc, uint32_t index);

    /**
     * @brief Returns a pointer to the incoming unit buffer.
     * @return A pointer to the incoming unit buffer.
     */
    Float* GetIncomingUnitBuffer() const;

    /**
     * @brief Returns a pointer to the unit buffer.
     * @return A pointer to the unit buffer.
     */
    Float* GetUnitBuffer() const;

    /**
     * @brief Returns a pointer to the incoming delta buffer.
     * @return A pointer to the incoming delta buffer.
     */
    Float* GetIncomingDeltaBuffer() const;

    /**
     * @brief Returns a pointer to the delta buffer.
     * @return A pointer to the delta buffer.
     */
    Float* GetDeltaBuffer() const;

    /**
     * @brief Returns the size of the buffer used by the layer.
     * @return The size of the buffer.
     */
    uint64_t GetBufferSize() const;

    /**
     * @brief Returns the tensor descriptor for the given batch size.
     * @param batch The batch size.
     * @return The tensor descriptor for the batch.
     */
    cudnnTensorDescriptor_t getTensorDescriptor(uint32_t batch);

    /**
     * @brief Returns the tensor descriptor for batch normalization for the given batch size.
     * @param batch The batch size.
     * @return The tensor descriptor for batch normalization for the batch.
     */
    cudnnTensorDescriptor_t getTensorDescriptorBN(uint32_t batch);

public:
    /**
     * @brief Returns the name of the layer.
     * @return The name of the layer.
     */
    const std::string& GetName() const;

    /**
     * @brief Returns the name of the associated data set.
     * @return The name of the associated data set.
     */
    const std::string& GetDataSetName() const;

    /**
     * @brief Returns the kind of the layer.
     * @return The kind of the layer.
     */
    Kind GetKind() const;

    /**
     * @brief Returns the type of the layer.
     * @return The type of the layer.
     */
    Type GetType() const;

    /**
     * @brief Returns the attributes of the layer.
     * @return The attributes of the layer.
     */
    uint32_t GetAttributes() const;

    /**
     * @brief Returns a pointer to the associated data set.
     * @return A pointer to the associated data set.
     */
    DataSetBase* GetDataSet() const;

    /**
     * @brief Returns the number of dimensions for the layer.
     * @return The number of dimensions for the layer.
     */
    uint32_t GetNumDimensions() const;

    /**
     * @brief Returns the dimensions of the layer.
     * @return A tuple representing the dimensions (Nx, Ny, Nz, Nw) of the layer.
     */
    std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> GetDimensions() const;

    /**
     * @brief Returns the local dimensions of the layer.
     * @return A tuple representing the local dimensions (Nx, Ny, Nz, Nw) of the layer.
     */
    std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> GetLocalDimensions() const;

    /**
     * @brief Returns the kernel dimensions of the layer.
     * @return A tuple representing the kernel dimensions (kernelX, kernelY, kernelZ) of the layer.
     */
    std::tuple<uint32_t, uint32_t, uint32_t> GetKernelDimensions() const;

    /**
     * @brief Returns the kernel stride of the layer.
     * @return A tuple representing the kernel stride (kernelStrideX, kernelStrideY, kernelStrideZ) of the layer.
     */
    std::tuple<uint32_t, uint32_t, uint32_t> GetKernelStride() const;

    /**
     * @brief Copies the unit values of the layer into a vector.
     * @param vUnit The vector to store the unit values.
     * @return True if the unit values were successfully copied, false otherwise.
     */
    bool GetUnits(std::vector<Float>& vUnit) const;

    /**
     * @brief Copies the unit values of the layer into an array.
     * @param pUnit The array to store the unit values.
     * @return True if the unit values were successfully copied, false otherwise.
     */
    bool GetUnits(Float* pUnit) const;

    /**
     * @brief Sets the unit values of the layer using a vector.
     * @param vUnit The vector containing the new unit values.
     * @return True if the unit values were successfully set, false otherwise.
     */
    bool SetUnits(const std::vector<Float>& vUnit);

    /**
     * @brief Copies the delta values of the layer into a vector.
     * @param vUnit The vector to store the delta values.
     * @return True if the delta values were successfully copied, false otherwise.
     */
    bool GetDeltas(std::vector<Float>& vUnit) const;

    /**
     * @brief Copies the delta values of the layer into an array.
     * @param pUnit The array to store the delta values.
     * @return True if the delta values were successfully copied, false otherwise.
     */
    bool GetDeltas(Float* pUnit) const;

    /**
     * @brief Sets the delta values of the layer using a vector.
     * @param vUnit The vector containing the new delta values.
     * @return True if the delta values were successfully set, false otherwise.
     */
    bool SetDeltas(const std::vector<Float>& vUnit);

};

    /**
     * @brief Overloaded output stream operator for Layer::Kind enumeration.
     * @param out The output stream.
     * @param k The Layer::Kind value to be printed.
     * @return The output stream after printing the Layer::Kind value.
     */
    std::ostream& operator<<(std::ostream& out, const Layer::Kind& k);

    /**
     * @brief Overloaded output stream operator for Layer::Type enumeration.
     * @param out The output stream.
     * @param t The Layer::Type value to be printed.
     * @return The output stream after printing the Layer::Type value.
     */
    std::ostream& operator<<(std::ostream& out, const Layer::Type& t);

    /**
     * @brief Overloaded output stream operator for Layer::Parallelization enumeration.
     * @param out The output stream.
     * @param p The Layer::Parallelization value to be printed.
     * @return The output stream after printing the Layer::Parallelization value.
     */
    std::ostream& operator<<(std::ostream& out, const Layer::Parallelization& p);

    /**
     * @brief Overloaded output stream operator for Layer::Attributes enumeration.
     * @param out The output stream.
     * @param a The Layer::Attributes value to be printed.
     * @return The output stream after printing the Layer::Attributes value.
     */
    std::ostream& operator<<(std::ostream& out, const Layer::Attributes& a);


/**
 * @brief Struct representing a Layer descriptor used for layer initialization.
 */
struct LayerDescriptor
{
    /**
     * @brief The name of the layer.
     */
    std::string _name;

    /**
     * @brief The kind of the layer.
     */
    Layer::Kind _kind;

    /**
     * @brief The type of the layer.
     */
    Layer::Type _type;

    /**
     * @brief The pooling function for the layer.
     */
    PoolingFunction _poolingFunction;

    /**
     * @brief The name of the associated data set.
     */
    std::string _dataSet;

    /**
     * @brief The vector of source names.
     */
    std::vector<std::string> _vSource;

    /**
     * @brief The vector of skip names.
     */
    std::vector<std::string> _vSkip;

    /**
     * @brief The number of units in the X dimension.
     */
    uint32_t _Nx;

    /**
     * @brief The number of units in the Y dimension.
     */
    uint32_t _Ny;

    /**
     * @brief The number of units in the Z dimension.
     */
    uint32_t _Nz;

    /**
     * @brief The number of units in the W dimension.
     */
    uint32_t _Nw;

    /**
     * @brief The number of dimensions for the layer.
     */
    uint32_t _dimensions;

    /**
     * @brief Flag indicating whether dimensions were provided.
     */
    bool _bDimensionsProvided;

    /**
     * @brief The weight initialization method.
     */
    WeightInitialization _weightInit;

    /**
     * @brief The scale factor for weight initialization.
     */
    Float _weightInitScale;

    /**
     * @brief The bias initialization value.
     */
    Float _biasInit;

    /**
     * @brief The size of the kernel in the X dimension.
     */
    uint32_t _kernelX;

    /**
     * @brief The size of the kernel in the Y dimension.
     */
    uint32_t _kernelY;

    /**
     * @brief The size of the kernel in the Z dimension.
     */
    uint32_t _kernelZ;

    /**
     * @brief The stride of the kernel in the X dimension.
     */
    uint32_t _kernelStrideX;

    /**
     * @brief The stride of the kernel in the Y dimension.
     */
    uint32_t _kernelStrideY;

    /**
     * @brief The stride of the kernel in the Z dimension.
     */
    uint32_t _kernelStrideZ;

    /**
     * @brief The padding size in the X dimension for the kernel.
     */
    uint32_t _kernelPaddingX;

    /**
     * @brief The padding size in the Y dimension for the kernel.
     */
    uint32_t _kernelPaddingY;

    /**
     * @brief The padding size in the Z dimension for the kernel.
     */
    uint32_t _kernelPaddingZ;

    /**
     * @brief The number of dimensions for the kernel.
     */
    uint32_t _kernelDimensions;

    /**
     * @brief The vector of scale values for batch normalization.
     */
    std::vector<Float> _vScaleBN;

    /**
     * @brief The vector of bias values for batch normalization.
     */
    std::vector<Float> _vBiasBN;

    /**
     * @brief The vector of running mean values for batch normalization.
     */
    std::vector<Float> _vRuingMeanBN;

    /**
     * @brief The vector of running variance values for batch normalization.
     */
    std::vector<Float> _vRuingVarianceBN;

    /**
     * @brief The weight normalization factor.
     */
    Float _weightNorm;

    /**
     * @brief The delta normalization factor.
     */
    Float _deltaNorm;

    /**
     * @brief The dropout probability.
     */
    Float _pDropout;

    /**
     * @brief The activation function.
     */
    Activation _activation;

    /**
     * @brief The p parameter for sparseness penalty.
     */
    Float _sparsenessPenalty_p;

    /**
     * @brief The beta parameter for sparseness penalty.
     */
    Float _sparsenessPenalty_beta;

    /**
     * @brief The attributes of the layer.
     */
    uint32_t _attributes;

    /**
     * @brief The slope for the Leaky ReLU activation function.
     */
    Float _RELUSlope;

    /**
     * @brief The alpha parameter for the ELU activation function.
     */
    Float _ELUAlpha;

    /**
     * @brief The lambda parameter for the SELU activation function.
     */
    Float _SELULambda;

    LayerDescriptor();
};

    /**
     * @brief Loads a layer descriptor from a NetCDF file.
     * @param fname The filename of the NetCDF file.
     * @param nc The netCDF::NcFile object representing the opened NetCDF file.
     * @param index The index of the layer descriptor in the NetCDF file.
     * @param ld The LayerDescriptor object to store the loaded layer descriptor.
     * @return True if the layer descriptor was successfully loaded, false otherwise.
     */
    bool LoadLayerDescriptorNetCDF(const std::string_view fname, netCDF::NcFile& nc, uint32_t index, LayerDescriptor& ld);

    /**
     * @brief Overloaded output stream operator for LayerDescriptor.
     * @param out The output stream.
     * @param d The LayerDescriptor object to be printed.
     * @return The output stream after printing the LayerDescriptor object.
     */
    std::ostream& operator<<(std::ostream& out, const LayerDescriptor& d);

    /**
     * @brief Broadcasts a LayerDescriptor object using MPI.
     * @param d The LayerDescriptor object to be broadcasted.
     * @return The rank of the process sending the broadcasted LayerDescriptor.
     */
    uint32_t MPI_Bcast_LayerDescriptor(LayerDescriptor& d);


#endif
#endif

