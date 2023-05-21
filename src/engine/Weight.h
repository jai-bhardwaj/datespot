#ifndef WEIGHT_H
#define WEIGHT_H

#include <cstdint>
#include <array>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

class Layer;
class Network;

using Float = float;

/**
 * @brief Class representing a weight in a neural network.
 */
class Weight {
public:
    /**
     * @brief Enumeration for weight transform types.
     */
    enum class Transform {
        Convolution, /**< Convolutional transform */
        Linear       /**< Linear transform */
    };

    /**
     * @brief Pair array mapping transform types to string representations.
     */
    static constexpr std::array _sTransformPair{
        std::pair{Transform::Convolution, "Convolution"},
        std::pair{Transform::Linear, "Linear"}
    };

    /**
     * @brief Map mapping transform types to string representations.
     */
    static std::map<Transform, std::string> _sTransformMap;

private:
    friend class Network;
    friend class Layer;
    friend Network* LoadNeuralNetworkNetCDF(const std::string& fname, uint32_t batch);

    Layer& _inputLayer;
    Layer& _outputLayer;
    const bool _bShared;
    const bool _bTransposed;
    Transform _transform;
    bool _bLocked;
    Weight* _pSharedWeight;
    uint32_t _sharingCount;
    uint32_t _updateCount;
    uint32_t _dimensionality;
    uint64_t _width;
    uint64_t _height;
    uint64_t _length;
    uint64_t _depth;
    uint64_t _breadth;
    uint32_t _widthStride;
    uint32_t _heightStride;
    uint32_t _lengthStride;
    uint64_t _size;
    uint64_t _biasSize;
    uint64_t _localSize;
    uint64_t _localBiasSize;
    Float _norm;
    cudnnTensorDescriptor_t _convBiasTensor;
    cudnnFilterDescriptor_t _convFilterDesc;
    cudnnConvolutionDescriptor_t _convDesc;
    int _convPad[3];
    cudnnConvolutionFwdAlgo_t _convFWAlgo;
    cudnnConvolutionBwdFilterAlgo_t _convBWWeightAlgo;
    cudnnConvolutionBwdDataAlgo_t _convBWDeltaAlgo;
    std::vector<Float> _vWeight;
    std::vector<Float> _vBias;
    std::unique_ptr<GpuBuffer<Float>> _pbWeight;
    std::unique_ptr<GpuBuffer<Float>> _pbBias;
    std::unique_ptr<GpuBuffer<Float>> _pbWeightGradient;
    std::unique_ptr<GpuBuffer<Float>> _pbBiasGradient;
    std::unique_ptr<GpuBuffer<Float>> _pbWeightVelocity;
    std::unique_ptr<GpuBuffer<Float>> _pbBiasVelocity;
    std::unique_ptr<GpuBuffer<Float>> _pbWeightGradientVelocity;
    std::unique_ptr<GpuBuffer<Float>> _pbBiasGradientVelocity;

public:
    /**
     * @brief Constructs a Weight object.
     * @param inputLayer The input layer of the weight.
     * @param outputLayer The output layer of the weight.
     * @param bShared Indicates whether the weight is shared.
     * @param bTransposed Indicates whether the weight is transposed.
     * @param bLocked Indicates whether the weight is locked.
     * @param maxNorm The maximum norm value for weight regularization.
     */
    Weight(Layer& inputLayer, Layer& outputLayer, bool bShared = false, bool bTransposed = false, bool bLocked = false, Float maxNorm = 0.0f);

    /**
     * @brief Destructor for the Weight object.
     */
    ~Weight();

    /**
     * @brief Clears the shared gradient.
     */
    void ClearSharedGradient();

    /**
     * @brief Clears the gradient.
     */
    void ClearGradient();

    /**
     * @brief Calculates the regularization error for the weight.
     * @param lambda The L2 regularization coefficient.
     * @param lambda1 The L1 regularization coefficient.
     * @return The regularization error.
     */
    Float CalculateRegularizationError(Float lambda, Float lambda1);

    /**
     * @brief Clears the velocity of the weight.
     */
    void ClearVelocity();

    /**
     * @brief Randomizes the weight.
     */
    void Randomize();

    /**
     * @brief Locks the weight.
     */
    void Lock();

    /**
     * @brief Unlocks the weight.
     */
    void Unlock();

    /**
     * @brief Dumps the weight to a file.
     * @param fname The filename of the dump file.
     * @param pBuffer The buffer to dump the weight data.
     */
    void Dump(const std::string& fname, Float* pBuffer);

    /**
     * @brief Refreshes the state of the weight in the network.
     * @param pNetwork The network the weight belongs to.
     * @param trainingMode The training mode.
     */
    void RefreshState(Network* pNetwork, TrainingMode trainingMode);

    /**
     * @brief Updates the weights of the weight.
     * @param trainingMode The training mode.
     * @param batch The current batch size.
     * @param alpha The learning rate.
     * @param lambda The L2 regularization coefficient.
     * @param lambda1 The L1 regularization coefficient.
     * @param mu The momentum coefficient.
     * @param mu1 The Nesterov momentum coefficient.
     * @param t The current iteration count.
     */
    void UpdateWeights(TrainingMode trainingMode, uint32_t batch, Float alpha, Float lambda, Float lambda1, Float mu, Float mu1, Float t);

    /**
     * @brief Writes the weight to a NetCDF file.
     * @param nc The NetCDF file object.
     * @param index The index of the weight.
     * @param pWeight Pointer to the weight data. If nullptr, the weight data will be taken from the internal buffer.
     * @param pBias Pointer to the bias data. If nullptr, the bias data will be taken from the internal buffer.
     * @return True if the weight is written successfully, false otherwise.
     */
    bool WriteNetCDF(netCDF::NcFile& nc, uint32_t index, Float* pWeight = nullptr, Float* pBias = nullptr);

    /**
     * @brief Gets a pointer to the weight buffer.
     * @return Pointer to the weight buffer.
     */
    Float* GetWeightBuffer() { return _pbWeight ? _pbWeight->_pDevData : nullptr; }

    /**
     * @brief Gets a pointer to the weight gradient buffer.
     * @return Pointer to the weight gradient buffer.
     */
    Float* GetWeightGradientBuffer() { return _pbWeightGradient ? _pbWeightGradient->_pDevData : nullptr; }

    /**
     * @brief Gets the size of the weight buffer.
     * @return The size of the weight buffer.
     */
    uint64_t GetBufferSize() { return _localSize; }

    /**
     * @brief Copies the weights from another weight.
     * @param pWeight Pointer to the source weight.
     * @return True if the weights are copied successfully, false otherwise.
     */
    [[nodiscard]] bool CopyWeights(const Weight* pWeight);

    /**
     * @brief Sets the weights of the weight.
     * @param vWeight The weight values to set.
     * @return True if the weights are set successfully, false otherwise.
     */
    [[nodiscard]] bool SetWeights(const std::vector<Float>& vWeight);

    /**
     * @brief Sets the biases of the weight.
     * @param vBias The bias values to set.
     * @return True if the biases are set successfully, false otherwise.
     */
    [[nodiscard]] bool SetBiases(const std::vector<Float>& vBias);

    /**
     * @brief Gets the weights of the weight.
     * @param vWeight The output vector to store the weight values.
     * @return True if the weights are retrieved successfully, false otherwise.
     */
    [[nodiscard]] bool GetWeights(std::vector<Float>& vWeight);

    /**
     * @brief Gets the biases of the weight.
     * @param vBias The output vector to store the bias values.
     * @return True if the biases are retrieved successfully, false otherwise.
     */
    [[nodiscard]] bool GetBiases(std::vector<Float>& vBias);

    /**
     * @brief Gets the dimensions of the weight.
     * @param dimensions The output vector to store the dimensions.
     * @return True if the dimensions are retrieved successfully, false otherwise.
     */
    [[nodiscard]] bool GetDimensions(std::vector<uint64_t>& dimensions);

    /**
     * @brief Sets the norm of the weight.
     * @param norm The norm value to set.
     * @return True if the norm is set successfully, false otherwise.
     */
    bool SetNorm(Float norm) { _norm = norm; return true; }
};

/**
 * @brief Struct representing a weight descriptor.
 */
struct WeightDescriptor {
    std::string _inputLayer;        /**< The input layer name. */
    std::string _outputLayer;       /**< The output layer name. */
    uint64_t _width;                /**< The width of the weight. */
    uint64_t _height;               /**< The height of the weight. */
    uint64_t _length;               /**< The length of the weight. */
    uint64_t _depth;                /**< The depth of the weight. */
    uint64_t _breadth;              /**< The breadth of the weight. */
    std::vector<Float> _vWeight;    /**< The weight values. */
    std::vector<Float> _vBias;      /**< The bias values. */
    bool _bShared;                  /**< Indicates whether the weight is shared. */
    bool _bTransposed;              /**< Indicates whether the weight is transposed. */
    bool _bLocked;                  /**< Indicates whether the weight is locked. */
    Float _norm;                    /**< The norm value. */
    std::string _sourceInputLayer;   /**< The source input layer name. */
    std::string _sourceOutputLayer;  /**< The source output layer name. */

    /**
     * @brief Constructs a WeightDescriptor object.
     */
    WeightDescriptor();
};

/**
 * @brief Loads a weight descriptor from a NetCDF file.
 * @param fname The filename of the NetCDF file.
 * @param nc The NetCDF file object.
 * @param index The index of the weight descriptor.
 * @param wd The output weight descriptor.
 * @return True if the weight descriptor is loaded successfully, false otherwise.
 */
bool LoadWeightDescriptorNetCDF(const std::string& fname, netCDF::NcFile& nc, uint32_t index, WeightDescriptor& wd);

/**
 * @brief Overloaded stream insertion operator for WeightDescriptor.
 * @param out The output stream.
 * @param d The WeightDescriptor object to output.
 * @return The output stream.
 */
std::ostream& operator<< (std::ostream& out, WeightDescriptor& d);

/**
 * @brief Broadcasts a WeightDescriptor using MPI.
 * @param d The WeightDescriptor to broadcast.
 * @return The rank of the process.
 */
uint32_t MPI_Bcast_WeightDescriptor(WeightDescriptor& d);

#endif
