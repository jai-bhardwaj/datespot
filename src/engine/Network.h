#ifndef NETWORK_H
#define NETWORK_H
#ifndef __NVCC__
#include <memory>
#include <map>
#include <vector>
#include <tuple>
#include <string>
#include <iostream>

struct NetworkDescriptor;

class Network {
public:
    /**
     * Friend class declaration for Layer.
     */
    friend class Layer;

    /**
     * Friend class declaration for Weight.
     */
    friend class Weight;

    /**
     * Friend function declaration for GpuContext::SetNeuralNetwork.
     *
     * @param pNetwork Pointer to the neural network to set.
     */
    friend void GpuContext::SetNeuralNetwork(Network* pNetwork);

    /**
     * Enum class representing the kind of network.
     */
    enum class Kind {
        FeedForward, /**< Feed-forward network kind. */
        AutoEncoder /**< Autoencoder network kind. */
    };

    /**
     * Array of pairs mapping network kinds to their string representations.
     */
    inline static std::pair<Kind, std::string> _sKindPair[] = {
        {Kind::FeedForward, "FeedForward"},
        {Kind::AutoEncoder, "AutoEncoder"}
    };

    /**
     * Map associating network kinds with their string representations.
     */
    inline static std::map<Kind, std::string> _sKindMap = {
        {Kind::FeedForward, "FeedForward"},
        {Kind::AutoEncoder, "AutoEncoder"}
    };

private:
    /**
     * Loads a neural network from a JSON file.
     *
     * @param fname The filename of the JSON file.
     * @param batch The batch size.
     * @param vDataSet The vector of DataSetBase pointers.
     * @return A pointer to the loaded neural network.
     */
    friend Network* LoadNeuralNetworkJSON(const std::string& fname, const uint32_t batch, const std::vector<DataSetBase*>& vDataSet);

    /**
     * Loads a neural network from a NetCDF file.
     *
     * @param fname The filename of the NetCDF file.
     * @param batch The batch size.
     * @return A pointer to the loaded neural network.
     */
    friend Network* LoadNeuralNetworkNetCDF(const std::string& fname, const uint32_t batch);

    /**
     * Imports an autoencoder from a file.
     *
     * @param fname The filename of the autoencoder file.
     * @param batch The batch size.
     * @return A pointer to the imported autoencoder network.
     */
    friend Network* ImportAutoEncoder(const std::string& fname, uint32_t batch);

    /**
     * The name of the network.
     */
    std::string _name;

    /**
     * The batch size of the network.
     */
    uint32_t _batch;

    /**
     * The local batch size of the network.
     */
    uint32_t _localBatch;

    /**
     * The position in the network.
     */
    uint32_t _position;

    /**
     * The local position in the network.
     */
    uint32_t _localPosition;

    /**
     * Flag indicating if examples have been found in the network.
     */
    bool _bExamplesFound;

    /**
     * Flag indicating if all data has been loaded into the network.
     */
    bool _bAllDataLoaded;

    /**
     * The number of examples in the network.
     */
    uint32_t _examples;

    /**
     * The kind of the network.
     */
    const Kind _kind;

    /**
     * The error function used by the network.
     */
    ErrorFunction _errorFunction;

    /**
     * The training mode of the network.
     */
    TrainingMode _trainingMode;

    /**
     * The mode of the network.
     */
    Mode _mode;

    /**
     * The number of training epochs.
     */
    uint32_t _epochs;

    /**
     * The number of indices.
     */
    uint32_t _indices;

    /**
     * The number of batches.
     */
    uint32_t _batches;

    /**
     * The weight decay parameter of the network.
     */
    float _decay;

    /**
     * The k parameter of Local Response Normalization (LRN).
     */
    Float _LRN_k;

    /**
     * The n parameter of Local Response Normalization (LRN).
     */
    uint32_t _LRN_n;

    /**
     * The alpha parameter of Local Response Normalization (LRN).
     */
    Float _LRN_alpha;

    /**
     * The beta parameter of Local Response Normalization (LRN).
     */
    Float _LRN_beta;

    /**
     * The slope parameter of Rectified Linear Unit (ReLU).
     */
    Float _RELUSlope;

    /**
     * The alpha parameter of Exponential Linear Unit (ELU).
     */
    Float _ELUAlpha;

    /**
     * The lambda parameter of Scaled Exponential Linear Unit (SELU).
     */
    Float _SELULambda;

    /**
     * The k parameter of Maxout.
     */
    uint32_t _maxout_k;

    /**
     * Flag indicating if sparseness penalty is applied in the network.
     */
    bool _bSparsenessPenalty;

    /**
     * The p parameter of sparseness penalty.
     */
    Float _sparsenessPenalty_p;

    /**
     * The beta parameter of sparseness penalty.
     */
    Float _sparsenessPenalty_beta;

    /**
     * Flag indicating if denoising is applied in the network.
     */
    bool _bDenoising;

    /**
     * The p parameter of denoising.
     */
    Float _denoising_p;

    /**
     * The delta boost value for positive targets.
     */
    Float _deltaBoost_one;

    /**
     * The delta boost value for negative targets.
     */
    Float _deltaBoost_zero;

    /**
     * The target probability for positive targets in Scaled Multiclass Cross-Entropy (SMCE).
     */
    Float _SMCE_oneTarget;

    /**
     * The target probability for negative targets in Scaled Multiclass Cross-Entropy (SMCE).
     */
    Float _SMCE_zeroTarget;

    /**
     * The scaling factor for positive targets in Scaled Multiclass Cross-Entropy (SMCE).
     */
    Float _SMCE_oneScale;

    /**
     * The scaling factor for negative targets in Scaled Multiclass Cross-Entropy (SMCE).
     */
    Float _SMCE_zeroScale;

    /**
     * Flag indicating if shuffle indices are used in the network.
     */
    bool _bShuffleIndices;

    /**
     * The shuffle indices setting of the network.
     */
    uint32_t _shuffleIndices;

    /**
     * Pointer to the shuffle indices array.
     */
    uint32_t* _pShuffleIndex;

    /**
     * Unique pointer to the GpuBuffer for shuffle indices.
     */
    std::unique_ptr<GpuBuffer<uint32_t>> _pbShuffleIndex;

    /**
     * Unique pointer to the GpuSort for shuffle indices.
     */
    std::unique_ptr<GpuSort<uint32_t, uint32_t>> _pShuffleIndexSort;

    /**
     * The name of the checkpoint.
     */
    std::string _checkpoint_name;

    /**
     * The interval at which to save the checkpoint.
     */
    int32_t _checkpoint_interval;

    /**
     * The number of epochs between each checkpoint.
     */
    int32_t _checkpoint_epochs;

    /**
     * Vector containing pointers to the layers in the network.
     */
    std::vector<Layer*> _vLayer;

    /**
     * Vector containing pointers to the input layers in the network.
     */
    std::vector<Layer*> _vInputLayer;

    /**
     * Vector containing pointers to the output layers in the network.
     */
    std::vector<Layer*> _vOutputLayer;

    /**
     * Vector containing pointers to the weights in the network.
     */
    std::vector<Weight*> _vWeight;

    /**
     * Vector containing pointers to the shared weights in the network.
     */
    std::vector<Weight*> _vSharedWeight;

    /**
     * Vector containing pointers to the data sets in the network.
     */
    std::vector<DataSetBase*> _vData;

    /**
     * Vector containing pointers to the layers in forward propagation order.
     */
    std::vector<Layer*> _vFPOrder;

    /**
     * Vector containing pointers to the layers in backward propagation order.
     */
    std::vector<Layer*> _vBPOrder;

    /**
     * Map associating layer names with their respective layer objects.
     */
    std::map<std::string, Layer*> _mLayer;

    /**
     * Flag indicating if the network state has changed since the last update.
     */
    bool _bDirty;

    /**
     * Flag indicating if the velocity should be cleared during training.
     */
    bool _bClearVelocity;

    /**
     * The size of the scratch buffer used by the network.
     */
    size_t _scratchBufferSize;

    /**
     * Unique pointer to the GpuBuffer for the scratch buffer.
     */
    std::unique_ptr<GpuBuffer<Float>> _pbScratchBuffer;

    /**
     * The maximum stride in the network.
     */
    uint32_t _maxStride;

    /**
     * The index for sending data in peer-to-peer communication.
     */
    uint32_t _sendIndex;

    /**
     * The index for receiving data in peer-to-peer communication.
     */
    uint32_t _receiveIndex;

    /**
     * Array of unique pointers to GpuBuffers for peer-to-peer communication.
     */
    std::unique_ptr<GpuBuffer<Float>> _pbP2PBuffer[2];

    /**
     * Pointers to the peer buffers for peer-to-peer communication.
     */
    Float* _pPeerBuffer[2];

    /**
     * Unique pointer to the CPU buffer for peer-to-peer communication.
     */
    std::unique_ptr<Float[]> _pCPUBuffer;

    /**
     * The size of the CUDA workspace for GPU operations.
     */
    size_t _CUDWorkspaceSize;

    /**
     * The maximum size of the CUDA workspace for GPU operations.
     */
    size_t _maxCUDWorkspaceSize;

    /**
     * Unique pointer to the GpuBuffer for the CUDA workspace.
     */
    std::unique_ptr<GpuBuffer<uint8_t>> _pbCUDWorkspace;

    /**
     * Flag indicating if verbose debugging output is enabled.
     */
    bool _verbose;



public:

    /**
     * Destructor for the Network object.
     */
    ~Network();

    /**
     * Clears the data sets associated with the network.
     */
    void ClearDataSets();

    /**
     * Loads the provided data sets into the network.
     *
     * @param vData The vector of DataSetBase pointers.
     */
    void LoadDataSets(std::vector<DataSetBase*>& vData);

    /**
     * Randomizes the network's training data.
     */
    void Randomize();

    /**
     * Validates the network.
     *
     * @return True if the network validation is successful, false otherwise.
     */
    bool Validate();

    /**
     * Trains the network for the specified number of epochs using the provided learning parameters.
     *
     * @param epochs The number of training epochs (default: 1).
     * @param alpha The learning rate (default: 0.1).
     * @param lambda The regularization parameter (default: 0.001).
     * @param lambda1 The L1 regularization parameter (default: 0.0).
     * @param mu The momentum (default: 0.1).
     * @param mu1 The L1 momentum (default: 0.0).
     * @return The average training error.
     */
    float Train(uint32_t epochs = 1, Float alpha = (Float)0.1, Float lambda = (Float)0.001, Float lambda1 = (Float)0.0, Float mu = (Float)0.1,  Float mu1 = 0.0);

    /**
     * Predicts the output of the batch up to the specified number of layers.
     *
     * @param layers The number of layers to predict (default: 0 - predict all layers).
     */
    void PredictBatch(uint32_t layers = 0);

    /**
     * Calculates the output of the specified layer for the top-k values.
     *
     * @param layer The name of the layer.
     * @param k The number of top-k values to calculate.
     * @param pbKey The buffer to store the top-k values.
     * @param pbValue The buffer to store the indices of the top-k values.
     */
    void CalculateOutput(const std::string& layer, uint32_t k, GpuBuffer<Float>* pbKey, GpuBuffer<uint32_t>* pbValue);

    /**
     * Saves the batch to a file.
     *
     * @param fname The filename to save the batch to.
     */
    void SaveBatch(std::string fname);

    /**
     * Dumps the batch to a file.
     *
     * @param fp The file pointer to dump the batch to.
     */
    void DumpBatch(FILE* fp);

    /**
     * Saves the specified layer to a file.
     *
     * @param fname The filename to save the layer to.
     * @param layer The name of the layer.
     */
    void SaveLayer(const std::string& fname, const std::string& layer);

    /**
     * Dumps the specified layer to a file.
     *
     * @param fp The file pointer to dump the layer to.
     * @param layer The name of the layer.
     */
    void DumpLayer(FILE* fp, const std::string& layer);

    /**
     * Saves the weights between the specified input and output layers to a file.
     *
     * @param fname The filename to save the weights to.
     * @param inputLayer The name of the input layer.
     * @param outputLayer The name of the output layer.
     */
    void SaveWeights(const std::string& fname, const std::string& inputLayer, const std::string& outputLayer);

    /**
     * Locks the weights between the specified input and output layers.
     *
     * @param inputLayer The name of the input layer.
     * @param outputLayer The name of the output layer.
     * @return True if the weights were locked successfully, false otherwise.
     */
    bool LockWeights(const std::string& inputLayer, const std::string& outputLayer);

    /**
     * Unlocks the weights between the specified input and output layers.
     *
     * @param inputLayer The name of the input layer.
     * @param outputLayer The name of the output layer.
     * @return True if the weights were unlocked successfully, false otherwise.
     */
    bool UnlockWeights(const std::string& inputLayer, const std::string& outputLayer);

    /**
     * Sets the batch size for the network.
     *
     * @param batch The batch size.
     */
    void SetBatch(uint32_t batch);

    /**
     * Sets the position in the network.
     *
     * @param position The position in the network.
     */
    void SetPosition(uint32_t position);

    /**
     * Sets the weight decay parameter for the network.
     *
     * @param decay The weight decay parameter.
     * @return True if the weight decay parameter was set successfully, false otherwise.
     */
    bool SetDecay(Float decay);

    /**
     * Sets the training mode for the network.
     *
     * @param mode The training mode.
     */
    void SetTrainingMode(TrainingMode mode);

    /**
     * Sets the shuffle indices setting for the network.
     *
     * @param bShuffleIndices True to enable shuffle indices, false otherwise.
     */
    void SetShuffleIndices(bool bShuffleIndices);

    /**
     * Sets the CPU validation setting for the network.
     *
     * @param bValidate True to enable CPU validation, false otherwise.
     */
    void SetCPUValidate(bool bValidate);

    /**
     * Sets the clear velocity setting for the network.
     *
     * @param bClear True to enable clearing of velocity, false otherwise.
     */
    void SetClearVelocity(bool bClear) { _bClearVelocity = bClear; };

    /**
     * Saves the network to a NetCDF file.
     *
     * @param fname The filename to save the network to.
     * @return True if the network was saved successfully, false otherwise.
     */
    bool SaveNetCDF(const std::string& fname);


    /**
     * Returns the batch size of the network.
     *
     * @return The batch size of the network.
     */
    unsigned int GetBatch() const;

    /**
     * Returns the number of examples in the network.
     *
     * @return The number of examples in the network.
     */
    uint32_t GetExamples() const;

    /**
     * Returns the position in the network.
     *
     * @return The position in the network.
     */
    uint32_t GetPosition() const;

    /**
     * Returns a pointer to the weight object between the specified input and output layers.
     *
     * @param inputLayer The name of the input layer.
     * @param outputLayer The name of the output layer.
     * @return A pointer to the weight object between the specified input and output layers.
     */
    Weight* GetWeight(const std::string& inputLayer, const std::string& outputLayer) const;

    /**
     * Returns the buffer size of the specified layer.
     *
     * @param layer The name of the layer.
     * @return The buffer size of the specified layer.
     */
    uint64_t GetBufferSize(const std::string& layer) const;

    /**
     * Returns a pointer to the layer object with the specified name.
     *
     * @param layer The name of the layer.
     * @return A pointer to the layer object with the specified name.
     */
    Layer* GetLayer(const std::string &layer) const;


    /**
     * Returns an iterator to the layers of the specified kind.
     *
     * @param layerKind The kind of layer to retrieve.
     * @param layers The vector to store the layers in.
     * @return An iterator pointing to the layers of the specified kind.
     */
    std::vector<const Layer*>::iterator GetLayers(Layer::Kind layerKind, std::vector<const Layer*> &layers) const;

    /**
     * Returns a vector of names of all layers in the network.
     *
     * @return A vector of names of all layers in the network.
     */
    std::vector<std::string> GetLayers() const;

    /**
     * Returns the name of the network.
     *
     * @return The name of the network.
     */
    const std::string& GetName() const;

    /**
     * Returns the Local Response Normalization (LRN) parameters of the network.
     *
     * @return A tuple containing the k, n, alpha, and beta parameters of LRN.
     */
    std::tuple<Float, uint32_t, Float, Float> GetLRN() const;

    /**
     * Returns the weight decay parameter of the network.
     *
     * @return A tuple containing the weight decay parameter.
     */
    std::tuple<Float> GetDecay() const;

    /**
     * Returns the Maxout activation function parameters of the network.
     *
     * @return A tuple containing the k parameter of Maxout.
     */
    std::tuple<uint32_t> GetMaxout() const;

    /**
     * Returns the sparseness penalty parameters of the network.
     *
     * @return A tuple containing the sparseness penalty parameter and beta parameter.
     */
    std::tuple<Float, Float> GetSparsenessPenalty() const;

    /**
     * Returns the denoising parameter of the network.
     *
     * @return A tuple containing the denoising parameter.
     */
    std::tuple<Float> GetDenoising() const;

    /**
     * Returns the delta boost parameters of the network.
     *
     * @return A tuple containing the delta boost parameters for positive and negative targets.
     */
    std::tuple<Float, Float> GetDeltaBoost() const;

    /**
     * Returns the Scaled Multiclass Cross-Entropy (SMCE) parameters of the network.
     *
     * @return A tuple containing the target probabilities and scaling factors for positive and negative targets.
     */
    std::tuple<Float, Float, Float, Float> GetSMCE() const;

    /**
     * Returns the shuffle indices setting of the network.
     *
     * @return A tuple containing the shuffle indices setting.
     */
    std::tuple<bool> GetShuffleIndices() const;

    /**
     * Returns the checkpoint settings of the network.
     *
     * @return A tuple containing the name and interval of the checkpoint.
     */
    std::tuple<std::string, int32_t> GetCheckPoint() const;

    /**
     * Returns the debug level setting of the network.
     *
     * @return The debug level setting of the network.
     */
    bool GetDebugLevel() const {return _verbose;}


    /**
     * Returns a pointer to the unit buffer of the specified layer.
     *
     * @param layer The name of the layer.
     * @return A pointer to the unit buffer of the specified layer.
     */
    Float* GetUnitBuffer(const std::string& layer);

    /**
     * Returns a pointer to the delta buffer of the specified layer.
     *
     * @param layer The name of the layer.
     * @return A pointer to the delta buffer of the specified layer.
     */
    Float* GetDeltaBuffer(const std::string& layer);

    /**
     * Returns a pointer to the weight buffer between the specified input and output layers.
     *
     * @param inputLayer The name of the input layer.
     * @param outputLayer The name of the output layer.
     * @return A pointer to the weight buffer between the specified input and output layers.
     */
    Float* GetWeightBuffer(const std::string& inputLayer, const std::string& outputLayer);

    /**
     * Returns a pointer to a scratch buffer of the specified size.
     *
     * @param size The size of the scratch buffer (default: 0).
     * @return A pointer to the scratch buffer of the specified size.
     */
    Float* GetScratchBuffer(size_t size = 0);

    /**
     * Returns a pointer to the P2P send buffer.
     *
     * @return A pointer to the P2P send buffer.
     */
    Float* GetP2PSendBuffer();

    /**
     * Returns a pointer to the P2P receive buffer.
     *
     * @return A pointer to the P2P receive buffer.
     */
    Float* GetP2PReceiveBuffer();

    /**
     * Returns a pointer to the P2P CPU buffer.
     *
     * @return A pointer to the P2P CPU buffer.
     */
    Float* GetP2PCPUBuffer();

    /**
     * Returns a pointer to the peer buffer.
     *
     * @return A pointer to the peer buffer.
     */
    Float* GetPeerBuffer();

    /**
     * Returns a pointer to the peer back buffer.
     *
     * @return A pointer to the peer back buffer.
     */
    Float* GetPeerBackBuffer();

    /**
     * Broadcasts data using peer-to-peer communication.
     *
     * @param pBuffer The buffer to broadcast.
     * @param size The size of the buffer.
     * @return True if the broadcast was successful, false otherwise.
     */
    bool P2P_Bcast(void* pBuffer, size_t size);

    /**
     * Performs an allreduce operation using peer-to-peer communication.
     *
     * @param pBuffer The buffer to perform allreduce on.
     * @param size The size of the buffer.
     * @return True if the allreduce operation was successful, false otherwise.
     */
    bool P2P_Allreduce(Float* pBuffer, size_t size);



    /**
     * Sets Local Response Normalization (LRN) parameters for the network.
     *
     * @param k The k parameter for LRN (default: 2.0).
     * @param n The n parameter for LRN (default: 5).
     * @param alpha The alpha parameter for LRN (default: 0.0001).
     * @param beta The beta parameter for LRN (default: 0.75).
     * @return True if the LRN parameters were set successfully, false otherwise.
     */
    bool SetLRN(Float k = (Float)2.0, uint32_t n = 5, Float alpha = (Float)0.0001, Float beta = (Float)0.75);

    /**
     * Sets the Maxout activation function parameters for the network.
     *
     * @param k The k parameter for Maxout (default: 2).
     * @return True if the Maxout parameters were set successfully, false otherwise.
     */
    bool SetMaxout(uint32_t k = 2);

    /**
     * Sets the sparseness penalty parameters for the network.
     *
     * @param p The sparseness penalty parameter (default: 0.0).
     * @param beta The beta parameter for sparseness penalty (default: 0.0).
     * @return True if the sparseness penalty parameters were set successfully, false otherwise.
     */
    bool SetSparsenessPenalty(Float p = 0.0f, Float beta = 0.0f);

    /**
     * Sets the denoising parameters for the network.
     *
     * @param p The denoising parameter (default: 0.0).
     * @return True if the denoising parameters were set successfully, false otherwise.
     */
    bool SetDenoising(Float p = 0.0f);

    /**
     * Sets the delta boost parameters for the network.
     *
     * @param one The delta boost value for positive targets (default: 1.0).
     * @param zero The delta boost value for negative targets (default: 1.0).
     * @return True if the delta boost parameters were set successfully, false otherwise.
     */
    bool SetDeltaBoost(Float one = 1.0f, Float zero = 1.0f);

    /**
     * Sets the Scaled Multiclass Cross-Entropy (SMCE) parameters for the network.
     *
     * @param oneTarget The target probability for positive targets (default: 0.9).
     * @param zeroTarget The target probability for negative targets (default: 0.1).
     * @param oneScale The scaling factor for positive targets (default: 1.0).
     * @param zeroScale The scaling factor for negative targets (default: 1.0).
     * @return True if the SMCE parameters were set successfully, false otherwise.
     */
    bool SetSMCE(Float oneTarget = 0.9f, Float zeroTarget = 0.1f, Float oneScale = 1.0f, Float zeroScale = 1.0f);

    /**
     * Sets a checkpoint for the network to save its state during training.
     *
     * @param name The name of the checkpoint.
     * @param interval The interval (in iterations) at which to save the checkpoint.
     * @return True if the checkpoint was set successfully, false otherwise.
     */
    bool SetCheckpoint(std::string name, int32_t interval);

    /**
     * Sets the debug level for the network.
     *
     * @param verbose True to enable verbose debugging output, false otherwise.
     */
    void SetDebugLevel(bool verbose) {_verbose = verbose;}


private:
    /**
     * Calculates the propagation order of the network layers.
     */
    void CalculatePropagationOrder();

    /**
     * Generates a network graph.
     *
     * @return True if the network graph was generated successfully, false otherwise.
     */
    bool GenerateNetworkGraph();

    /**
     * Allocates buffers for peer-to-peer communication.
     */
    void AllocatePeerBuffers();

    /**
     * Deallocates buffers for peer-to-peer communication.
     */
    void DeallocatePeerBuffers();

    /**
     * Swaps peer buffers.
     */
    void SwapPeerBuffers();

    /**
     * Loads a batch of data for training or validation.
     */
    void LoadBatch();

    /**
     * Predicts the output of the training batch up to the specified number of layers.
     *
     * @param layers The number of layers to predict (default: 0 - predict all layers).
     */
    void PredictTrainingBatch(uint32_t layers = 0);

    /**
     * Predicts the output of the validation batch up to the specified number of layers.
     *
     * @param layers The number of layers to predict (default: 0 - predict all layers).
     */
    void PredictValidationBatch(uint32_t layers = 0);

    /**
     * Refreshes shuffle buffers.
     */
    void RefreshShuffleBuffers();

    /**
     * Shuffles indices.
     */
    void ShuffleIndices();

    /**
     * Calculates the error using the given regularization parameters.
     *
     * @param lambda The regularization parameter.
     * @param lambda1 The L1 regularization parameter.
     * @return A tuple containing the total error and the regularization error.
     */
    std::tuple<Float, Float> CalculateError(Float lambda, Float lambda1);

    /**
     * Clears update values for weights and biases.
     */
    void ClearUpdates();

    /**
     * Backpropagates the error through the network.
     */
    void BackPropagate();

    /**
     * Updates weights and biases using the specified learning parameters.
     *
     * @param alpha The learning rate.
     * @param lambda The regularization parameter.
     * @param lambda1 The L1 regularization parameter.
     * @param mu The momentum.
     * @param mu1 The L1 momentum.
     */
    void UpdateWeights(Float alpha, Float lambda, Float lambda1, Float mu, Float mu1);

    /**
     * Constructs a Network object with the given NetworkDescriptor and batch size.
     *
     * @param nd The NetworkDescriptor object.
     * @param batch The batch size (default: DefaultBatch).
     */
    Network(NetworkDescriptor& nd, uint32_t batch = DefaultBatch);

    /**
     * Refreshes the state of the network.
     */
    void RefreshState();

    /**
     * Shuffles the training data.
     */
    void Shuffle();

    /**
     * Sets the size of the CUDA workspace for GPU operations.
     *
     * @param size The size of the CUDA workspace.
     */
    void SetCUDWorkspace(size_t size);

};

std::ostream& operator<< (std::ostream& out, Network::Kind& k);


struct NetworkDescriptor
{
    /**
     * @brief The name of the network.
     */
    std::string _name;

    /**
     * @brief The kind of network.
     */
    Network::Kind _kind;

    /**
     * @brief The error function used for training.
     */
    ErrorFunction _errorFunction;

    /**
     * @brief Descriptors of layers in the network.
     */
    std::vector<LayerDescriptor> _vLayerDescriptor;

    /**
     * @brief Descriptors of weights in the network.
     */
    std::vector<WeightDescriptor> _vWeightDescriptor;

    /**
     * @brief Flag indicating whether to shuffle indices during training.
     */
    bool _bShuffleIndices;

    /**
     * @brief The decay factor used in training.
     */
    Float _decay;

    /**
     * @brief The value of k used in maxout layers.
     */
    uint32_t _maxout_k;

    /**
     * @brief The value of k used in Local Response Normalization (LRN) layers.
     */
    Float _LRN_k;

    /**
     * @brief The value of n used in LRN layers.
     */
    uint32_t _LRN_n;

    /**
     * @brief The value of alpha used in LRN layers.
     */
    Float _LRN_alpha;

    /**
     * @brief The value of beta used in LRN layers.
     */
    Float _LRN_beta;

    /**
     * @brief The slope of the rectified linear unit (ReLU) activation function.
     */
    Float _RELUSlope;

    /**
     * @brief The value of alpha used in exponential linear unit (ELU) activation function.
     */
    Float _ELUAlpha;

    /**
     * @brief The value of lambda used in scaled exponential linear unit (SELU) activation function.
     */
    Float _SELULambda;

    /**
     * @brief Flag indicating whether to apply sparseness penalty during training.
     */
    bool _bSparsenessPenalty;

    /**
     * @brief The value of p used in sparseness penalty.
     */
    Float _sparsenessPenalty_p;

    /**
     * @brief The value of beta used in sparseness penalty.
     */
    Float _sparsenessPenalty_beta;

    /**
     * @brief Flag indicating whether to apply denoising during training.
     */
    bool _bDenoising;

    /**
     * @brief The value of p used in denoising.
     */
    Float _denoising_p;

    /**
     * @brief The value of the delta boost for target one.
     */
    Float _deltaBoost_one;

    /**
     * @brief The value of the delta boost for target zero.
     */
    Float _deltaBoost_zero;

    /**
     * @brief The value of the scale for target one in Softmax Cross-Entropy (SMCE) loss function.
     */
    Float _SMCE_oneTarget;

    /**
     * @brief The value of the scale for target zero in SMCE loss function.
     */
    Float _SMCE_zeroTarget;

    /**
     * @brief The value of the scale for class one in SMCE loss function.
     */
    Float _SMCE_oneScale;

    /**
     * @brief The value of the scale for class zero in SMCE loss function.
     */
    Float _SMCE_zeroScale;

    /**
     * @brief The name of the checkpoint file for saving network weights.
     */
    std::string _checkpoint_name;

    /**
     * @brief The interval in epochs for saving checkpoints.
     */
    int32_t _checkpoint_interval;

    /**
     * @brief The maximum number of epochs for saving checkpoints.
     */
    int32_t _checkpoint_epochs;

    /**
     * @brief Flag indicating whether convolutional layers have been calculated.
     */
    bool _bConvLayersCalculated;

    NetworkDescriptor();
};

/**
 * Overloaded stream insertion operator to output a NetworkDescriptor object.
 *
 * @param out The output stream.
 * @param d The NetworkDescriptor object to be output.
 * @return The output stream after the NetworkDescriptor object has been inserted.
 */
std::ostream& operator<< (std::ostream& out, NetworkDescriptor& d);

/**
 * Loads a neural network from a NetCDF file.
 *
 * @param fname The filename of the NetCDF file.
 * @param batch The batch size for the neural network (default: DefaultBatch).
 * @return A pointer to the loaded Network object.
 */
Network* LoadNeuralNetworkNetCDF(const std::string& fname, const uint32_t batch = DefaultBatch);

/**
 * Loads a neural network from a JSON file.
 *
 * @param fname The filename of the JSON file.
 * @param batch The batch size for the neural network (default: DefaultBatch).
 * @param vDataSet The vector of DataSetBase pointers (default: empty vector).
 * @return A pointer to the loaded Network object.
 */
Network* LoadNeuralNetworkJSON(const std::string& fname, const uint32_t batch = DefaultBatch, const std::vector<DataSetBase*>& vDataSet = std::vector<DataSetBase*>());

/**
 * Saves a neural network to a JSON file.
 *
 * @param net The Network object to be saved.
 * @param fname The filename of the JSON file.
 * @return True if the network was saved successfully, false otherwise.
 */
bool SaveNeuralNetworkJSON(const Network& net, const std::string& fname);

/**
 * Saves a neural network to a NetCDF file.
 *
 * @param net The Network object to be saved.
 * @param jname The filename of the NetCDF file.
 * @return True if the network was saved successfully, false otherwise.
 */
bool SaveNeuralNetworkNetCDF(const Network& net, const std::string& jname);

/**
 * Imports an autoencoder neural network from a file.
 *
 * @param fname The filename of the file containing the autoencoder.
 * @param batch The batch size for the neural network (default: DefaultBatch).
 * @return A pointer to the imported Network object.
 */
Network* ImportAutoEncoder(const std::string& fname, uint32_t batch = DefaultBatch);

#endif
#endif
