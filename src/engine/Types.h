#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <set>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <netcdf>
#include <tuple>
#include <json/json.h>
#include <cmath>
#include <memory>
#include <concepts>
#include <optional>

class DataSetBase;
class Layer;
class Network;
class Weight;

/**
 * Version of the code.
 */
constexpr float _VERSION = 0.9f;

/**
 * Minimum error value.
 */
constexpr float MIN_ERROR = 1.0e-12f;

/**
 * Minimum activation value.
 */
constexpr float MIN_ACTIVATION = 0.000001f;

/**
 * Maximum activation value.
 */
constexpr float MAX_ACTIVATION = 0.999999f;

/**
 * Maximum value.
 */
constexpr float MAX_VALUE = 999999999999999.0f;

/**
 * Buffer on GPU.
 * @tparam T Template type.
 */
template <typename T> struct GpuBuffer;

/**
 * Mode of operation.
 */
enum class Mode {
    Prediction = 0, /**< Prediction mode. */
    Training, /**< Training mode. */
    Validation, /**< Validation mode. */
    Unspecified /**< Unspecified mode. */
};

/**
 * Training mode.
 */
enum class TrainingMode {
    SGD = 0, /**< Stochastic Gradient Descent. */
    Momentum, /**< Momentum. */
    AdaGrad, /**< AdaGrad. */
    Nesterov, /**< Nesterov Accelerated Gradient. */
    RMSProp, /**< RMSProp. */
    AdaDelta, /**< AdaDelta. */
    Adam /**< Adam. */
};

/**
 * Output operator for TrainingMode enum.
 *
 * @param out Output stream.
 * @param e Training mode.
 * @return Reference to the output stream.
 */
std::ostream& operator<<(std::ostream& out, const TrainingMode& e);

/**
 * Error function.
 */
enum class ErrorFunction {
    L1, /**< L1 loss function. */
    L2, /**< L2 loss function. */
    CrossEntropy, /**< Cross-entropy loss function. */
    ScaledMarginalCrossEntropy, /**< Scaled Marginal Cross-Entropy loss function. */
    DataScaledMarginalCrossEntropy, /**< Data Scaled Marginal Cross-Entropy loss function. */
    Hinge, /**< Hinge loss function. */
    L2Hinge /**< L2 Hinge loss function. */
};

/**
 * Output operator for ErrorFunction enum.
 *
 * @param out Output stream.
 * @param e Error function.
 * @return Reference to the output stream.
 */
std::ostream& operator<<(std::ostream& out, const ErrorFunction& e);

/**
 * Activation function.
 */
enum class Activation {
    Sigmoid, /**< Sigmoid activation function. */
    Tanh, /**< Hyperbolic tangent activation function. */
    RectifiedLinear, /**< Rectified Linear Unit (ReLU) activation function. */
    Linear, /**< Linear activation function. */
    ParametricRectifiedLinear, /**< Parametric Rectified Linear Unit (PReLU) activation function. */
    SoftPlus, /**< SoftPlus activation function. */
    SoftSign, /**< SoftSign activation function. */
    SoftMax, /**< SoftMax activation function. */
    RELUMax, /**< ReLU-Max activation function. */
    LinearMax, /**< Linear-Max activation function. */
    ExponentialLinear, /**< Exponential Linear Unit (ELU) activation function. */
    LeakyRectifiedLinear, /**< Leaky Rectified Linear Unit (Leaky ReLU) activation function. */
    ScaledExponentialLinear /**< Scaled Exponential Linear Unit (SELU) activation function. */
};

/**
 * Output operator for Activation enum.
 *
 * @param out Output stream.
 * @param a Activation function.
 * @return Reference to the output stream.
 */
std::ostream& operator<<(std::ostream& out, const Activation& a);

/**
 * Weight initialization method.
 */
enum class WeightInitialization {
    Xavier, /**< Xavier weight initialization. */
    CaffeXavier, /**< Caffe Xavier weight initialization. */
    Gaussian, /**< Gaussian weight initialization. */
    Uniform, /**< Uniform weight initialization. */
    UnitBall, /**< Unit Ball weight initialization. */
    Constant, /**< Constant weight initialization. */
    SELU /**< SELU weight initialization. */
};

/**
 * Output operator for WeightInitialization enum.
 *
 * @param out Output stream.
 * @param w Weight initialization method.
 * @return Reference to the output stream.
 */
std::ostream& operator<<(std::ostream& out, const WeightInitialization& w);

/**
 * Pooling function.
 */
enum class PoolingFunction {
    None, /**< No pooling. */
    Max, /**< Max pooling. */
    Average, /**< Average pooling. */
    LRN, /**< Local Response Normalization (LRN) pooling. */
    Maxout, /**< Maxout pooling. */
    DotProduct, /**< Dot product pooling. */
    Cosine, /**< Cosine pooling. */
    Stochastic, /**< Stochastic pooling. */
    LCN, /**< Local Contrast Normalization (LCN) pooling. */
    GlobalTemporal /**< Global Temporal pooling. */
};

/**
 * Output operator for PoolingFunction enum.
 *
 * @param out Output stream.
 * @param p Pooling function.
 * @return Reference to the output stream.
 */
std::ostream& operator<<(std::ostream& out, const PoolingFunction& p);

#include "kernels.h"
#include "GpuSort.h"
#include "Enum.h"
#include "Weight.h"
#include "Layer.h"
#include "Network.h"

/**
 * Broadcasts a string using MPI.
 *
 * @param s String to be broadcasted.
 * @return Status code of the broadcast operation.
 */
int MPI_Bcast_string(std::string& s);

/**
 * Dimensions of the dataset.
 */
struct DataSetDimensions {
    uint32_t _dimensions; /**< Number of dimensions. */
    uint32_t _width; /**< Width of the dataset. */
    uint32_t _height; /**< Height of the dataset. */
    uint32_t _length; /**< Length of the dataset. */

    /**
     * Default constructor.
     */
    DataSetDimensions();

    /**
     * Constructor with specified width, height, and length.
     *
     * @param width Width of the dataset.
     * @param height Height of the dataset.
     * @param length Length of the dataset.
     */
    DataSetDimensions(uint32_t width, uint32_t height = 1, uint32_t length = 1);
};

/**
 * Descriptor of the dataset.
 */
struct DataSetDescriptor {
    std::string _name; /**< Name of the dataset. */
    DataSetEnums::DataType _dataType; /**< Data type of the dataset. */
    uint32_t _attributes; /**< Attributes of the dataset. */
    DataSetDimensions _dim; /**< Dimensions of the dataset. */
    uint32_t _examples; /**< Number of examples in the dataset. */
    float _sparseDensity; /**< Sparse density of the dataset. */

    /**
     * Check if the dataset attributes are supported.
     *
     * @param attributes Attributes of the dataset.
     * @return True if the attributes are supported, false otherwise.
     */
    static bool isSupported(uint32_t attributes) {
        using DataSetEnums::Attributes;
        static const std::vector<Attributes> SUPPORTED_ATTRIBUTES { Attributes::Sparse };
        for (auto mask : SUPPORTED_ATTRIBUTES) {
            if (attributes & mask) {
                attributes -= mask;
            }
        }
        return attributes == 0;
    }
};

DataSetBase* createDataSet(const DataSetDescriptor &descriptor);

struct DataSetBase {
    /**
     * @brief The name of the data set.
     */
    std::string _name;

    /**
     * @brief The data type of the data set.
     */
    DataSetEnums::DataType _dataType;

    /**
     * @brief The attributes of the data set.
     */
    uint32_t _attributes;

    /**
     * @brief The total number of examples in the data set.
     */
    uint32_t _examples;

    /**
     * @brief The number of unique examples in the data set.
     */
    uint32_t _uniqueExamples;

    /**
     * @brief The number of examples in the local partition of the data set.
     */
    uint32_t _localExamples;

    /**
     * @brief The number of dimensions in the data set.
     */
    uint32_t _dimensions;

    /**
     * @brief The width dimension of the data set.
     */
    uint32_t _width;

    /**
     * @brief The height dimension of the data set.
     */
    uint32_t _height;

    /**
     * @brief The length dimension of the data set.
     */
    uint32_t _length;

    /**
     * @brief The stride of the data set.
     */
    uint32_t _stride;

    /**
     * @brief The sharding type of the data set.
     */
    DataSetEnums::Sharding _sharding;

    /**
     * @brief The minimum X value in the data set.
     */
    uint32_t _minX;

    /**
     * @brief The maximum X value in the data set.
     */
    uint32_t _maxX;

    /**
     * @brief The size of the sparse data in bytes.
     */
    uint64_t _sparseDataSize;

    /**
     * @brief The density of the sparse data.
     */
    float _sparseDensity;

    /**
     * @brief The starting indices of sparse data.
     */
    std::vector<uint64_t> _vSparseStart;

    /**
     * @brief The GPU buffer for starting indices of sparse data.
     */
    std::optional<GpuBuffer<uint64_t>> _pbSparseStart;

    /**
     * @brief The ending indices of sparse data.
     */
    std::vector<uint64_t> _vSparseEnd;

    /**
     * @brief The GPU buffer for ending indices of sparse data.
     */
    std::optional<GpuBuffer<uint64_t>> _pbSparseEnd;

    /**
     * @brief The indices of sparse data.
     */
    std::vector<uint32_t> _vSparseIndex;

    /**
     * @brief The GPU buffer for indices of sparse data.
     */
    std::optional<GpuBuffer<uint32_t>> _pbSparseIndex;

    /**
     * @brief The weights of the data.
     */
    std::vector<Float> _vDataWeight;

    /**
     * @brief The GPU buffer for weights of the data.
     */
    std::optional<GpuBuffer<Float>> _pbDataWeight;

    /**
     * @brief The indices of the data.
     */
    std::vector<uint32_t> _vIndex;

    /**
     * @brief The GPU buffer for indices of the data.
     */
    std::optional<GpuBuffer<uint32_t>> _pbIndex;

    /**
     * @brief The GPU buffer for denoising random values.
     */
    std::optional<GpuBuffer<Float>> _pbDenoisingRandom;

    /**
     * @brief The count of sparse data points in each example.
     */
    std::vector<uint64_t> _vSparseDatapointCount;

    /**
     * @brief The maximum count of sparse data points in each example.
     */
    std::vector<uint32_t> _vSparseMaxDatapointCount;

    /**
     * @brief The count of multi-valued sparse data points in each example.
     */
    std::vector<uint32_t> _vSparseMultiDatapointCount;

    /**
     * @brief The transposed starting indices of sparse data.
     */
    std::vector<uint32_t> _vSparseTransposedStart;

    /**
     * @brief The transposed indices of sparse data.
     */
    uint64_t _sparseTransposedIndices;

    /**
     * @brief The GPU buffer for transposed starting indices of sparse data.
     */
    std::optional<GpuBuffer<uint32_t>> _pbSparseTransposedStart;

    /**
     * @brief The GPU buffer for transposed ending indices of sparse data.
     */
    std::optional<GpuBuffer<uint32_t>> _pbSparseTransposedEnd;

    /**
     * @brief The GPU buffer for transposed indices of sparse data.
     */
    std::optional<GpuBuffer<uint32_t>> _pbSparseTransposedIndex;

    /**
     * @brief The GPU buffer for transposed data of sparse data.
     */
    std::optional<GpuBuffer<Float>> _pbSparseTransposedData;

    /**
     * @brief Flag indicating whether denoising is enabled.
     */
    bool _bDenoising;

    /**
     * @brief Flag indicating whether the data set is dirty.
     */
    bool _bDirty;

    /**
     * @brief Flag indicating whether streaming is enabled.
     */
    bool _bStreaming;

    /**
     * @brief Flag indicating whether the data set is indexed.
     */
    bool _bIndexed;

    /**
     * @brief The batch size of the data set.
     */
    uint32_t _batch;

    /**
     * @brief Default constructor for DataSetBase.
     */
    DataSetBase();

    /**
     * @brief Get the dimensions of the data set.
     * @return The dimensions of the data set.
     */
    DataSetDimensions GetDimensions();

    /**
     * @brief Get the total number of examples in the data set.
     * @return The total number of examples.
     */
    uint32_t GetExamples() { return _examples; };

    /**
     * @brief Get the number of unique examples in the data set.
     * @return The number of unique examples.
     */
    uint32_t GetUniqueExamples() { return _uniqueExamples; };

    /**
     * @brief Save the data set in NetCDF format to a file.
     * @param fname The filename to save to.
     * @return True if successful, false otherwise.
     */
    virtual bool SaveNetCDF(const std::string& fname) = 0;

    /**
     * @brief Write a subset of the data set to an existing NetCDF file.
     * @param nfc The netCDF file object to write to.
     * @param fname The filename to write to.
     * @param n The subset index.
     * @return True if successful, false otherwise.
     */
    virtual bool WriteNetCDF(netCDF::NcFile& nfc, const std::string& fname, const uint32_t n) = 0;

    /**
     * @brief Virtual destructor for DataSetBase.
     */
    virtual ~DataSetBase() = 0;

    /**
     * @brief Refresh the state of the data set for a given batch.
     * @param batch The batch number.
     */
    virtual void RefreshState(uint32_t batch) = 0;

    /**
     * @brief Shard the data set using the specified sharding type.
     * @param sharding The sharding type.
     * @return True if successful, false otherwise.
     */
    virtual bool Shard(DataSetEnums::Sharding sharding) = 0;

    /**
     * @brief Unshard the data set.
     * @return True if successful, false otherwise.
     */
    virtual bool UnShard() = 0;

    /**
     * @brief Set the streaming flag for the data set.
     * @param flag The streaming flag.
     * @return True if successful, false otherwise.
     */
    virtual bool SetStreaming(bool flag) = 0;

    /**
     * @brief Get the streaming flag of the data set.
     * @return The streaming flag.
     */
    virtual bool GetStreaming() = 0;

    /**
     * @brief Get the memory usage of the data set.
     * @return A vector of memory usage tuples.
     */
    virtual std::vector<std::tuple<uint64_t, uint64_t>> getMemoryUsage() = 0;

    /**
     * @brief Calculate the counts of sparse data points in each example.
     * @return True if successful, false otherwise.
     */
    virtual bool CalculateSparseDatapointCounts() = 0;

    /**
     * @brief Generate the transposed matrix of sparse data for a given batch and layer.
     * @param batch The batch number.
     * @param pLayer The layer object.
     * @return True if successful, false otherwise.
     */
    virtual bool GenerateSparseTransposedMatrix(uint32_t batch, Layer* pLayer) = 0;

    /**
     * @brief Calculate the transposed matrix of sparse data for a given position, batch, and layer.
     * @param position The position index.
     * @param batch The batch number.
     * @param pLayer The layer object.
     * @return True if successful, false otherwise.
     */
    virtual bool CalculateSparseTransposedMatrix(uint32_t position, uint32_t batch, Layer* pLayer) = 0;

    /**
     * @brief Calculate the transposed denoised matrix of sparse data for a given position, batch, and layer.
     * @param position The position index.
     * @param batch The batch number.
     * @param pLayer The layer object.
     * @return True if successful, false otherwise.
     */
    virtual bool CalculateSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, Layer* pLayer) = 0;

    /**
     * @brief Calculate the weight gradient for sparse transposed data.
     * @param alpha The learning rate.
     * @param beta The momentum.
     * @param m The number of rows.
     * @param n The number of columns.
     * @param pDelta The delta values.
     * @param pWeightGradient The weight gradient values.
     * @return True if successful, false otherwise.
     */
    virtual bool CalculateSparseTransposedWeightGradient(Float alpha, Float beta, uint32_t m, uint32_t n, Float* pDelta, Float* pWeightGradient) = 0;

    /**
     * @brief Set the denoising flag for the data set.
     * @param flag The denoising flag.
     * @return True if successful, false otherwise.
     */
    virtual bool SetDenoising(bool flag) = 0;

    /**
     * @brief Generate the denoising data for the data set.
     * @return True if successful, false otherwise.
     */
    virtual bool GenerateDenoisingData() = 0;

    /**
     * @brief Load an input unit from the data set.
     * @param position The position index.
     * @param batch The batch number.
     * @param stride The stride value.
     * @param pUnit Pointer to store the loaded input unit.
     * @return True if successful, false otherwise.
     */
    virtual bool LoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit) = 0;

    /**
     * @brief Load a sparse input unit from the data set.
     * @param position The position index.
     * @param batch The batch number.
     * @param stride The stride value.
     * @param pUnit Pointer to store the loaded sparse input unit.
     * @return True if successful, false otherwise.
     */
    virtual bool LoadSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit) = 0;

    /**
     * @brief Load a sparse denoised input unit from the data set.
     * @param position The position index.
     * @param batch The batch number.
     * @param stride The stride value.
     * @param pUnit Pointer to store the loaded sparse denoised input unit.
     * @return True if successful, false otherwise.
     */
    virtual bool LoadSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit) = 0;

    /**
     * @brief Calculate the Z values for sparse data at a given position, batch, and stride.
     * @param position The position index.
     * @param batch The batch number.
     * @param stride The stride value.
     * @param pWeight Pointer to the weight values.
     * @param pUnit Pointer to the unit values.
     * @param beta The beta value.
     * @return True if successful, false otherwise.
     */
    virtual bool CalculateSparseZ(uint32_t position, uint32_t batch, uint32_t stride, Float* pWeight, Float* pUnit, Float beta = (Float)0.0) = 0;

    /**
     * @brief Calculate the Z values for sparse denoised data at a given position, batch, and stride.
     * @param position The position index.
     * @param batch The batch number.
     * @param stride The stride value.
     * @param pWeight Pointer to the weight values.
     * @param pUnit Pointer to the unit values.
     * @param beta The beta value.
     * @return True if successful, false otherwise.
     */
    virtual bool CalculateSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, Float* pWeight, Float* pUnit, Float beta = (Float)0.0) = 0;

    /**
     * @brief Calculate the L1 error for sparse data at a given position, batch, and stride.
     * @param position The position index.
     * @param batch The batch number.
     * @param stride The stride value.
     * @param pUnit Pointer to the unit values.
     * @return The calculated L1 error.
     */
    virtual float CalculateL1Error(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit) = 0;

    /**
     * @brief Calculate the L2 error for sparse data at a given position, batch, and stride.
     * @param position The position index.
     * @param batch The batch number.
     * @param stride The stride value.
     * @param pUnit Pointer to the unit values.
     * @return The calculated L2 error.
     */
    virtual float CalculateL2Error(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit) = 0;

    /**
     * @brief Calculate the L2 Hinge error for sparse data at a given position, batch, and stride.
     * @param position The position index.
     * @param batch The batch number.
     * @param stride The stride value.
     * @param pUnit Pointer to the unit values.
     * @return The calculated L2 Hinge error.
     */
    virtual float CalculateL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit) = 0;

    /**
     * @brief Calculate the cross-entropy error for sparse data at a given position, batch, and stride.
     * @param position The position index.
     * @param batch The batch number.
     * @param stride The stride value.
     * @param pUnit Pointer to the unit values.
     * @return The calculated cross-entropy error.
     */
    virtual float CalculateCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit) = 0;

    /**
     * @brief Calculate the scaled marginal cross-entropy error for sparse data at a given position, batch, and stride.
     * @param position The position index.
     * @param batch The batch number.
     * @param stride The stride value.
     * @param pUnit Pointer to the unit values.
     * @return The calculated scaled marginal cross-entropy error.
     */
    virtual float CalculateScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit) = 0;

    /**
     * @brief Calculate the multinomial cross-entropy error for sparse data at a given position, batch, and stride.
     * @param position The position index.
     * @param batch The batch number.
     * @param stride The stride value.
     * @param pUnit Pointer to the unit values.
     * @return The calculated multinomial cross-entropy error.
     */
    virtual float CalculateMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit) = 0;

    /**
     * @brief Calculate the multinomial scaled marginal cross-entropy error for sparse data at a given position, batch, and stride.
     * @param position The position index.
     * @param batch The batch number.
     * @param stride The stride value.
     * @param pUnit Pointer to the unit values.
     * @return The calculated multinomial scaled marginal cross-entropy error.
     */
    virtual float CalculateMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit) = 0;

    /**
     * @brief Calculate the data scaled marginal cross-entropy error for sparse data at a given position, batch, and stride.
     * @param position The position index.
     * @param batch The batch number.
     * @param stride The stride value.
     * @param pUnit Pointer to the unit values.
     * @return The calculated data scaled marginal cross-entropy error.
     */
    virtual float CalculateDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit) = 0;

    /**
     * @brief Calculate the hinge error for sparse data at a given position, batch, and stride.
     * @param position The position index.
     * @param batch The batch number.
     * @param stride The stride value.
     * @param pUnit Pointer to the unit values.
     * @return The calculated hinge error.
     */
    virtual float CalculateHingeError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit) = 0;

    /**
     * @brief Calculate the L1 output delta for sparse data with the specified activation function.
     * @param activation The activation function.
     * @param position The position index.
     * @param batch The batch number.
     * @param stride The stride value.
     * @param pUnit Pointer to the unit values.
     * @param pDelta Pointer to store the calculated output delta values.
     * @param slope The slope value.
     * @param alpha The alpha value.
     * @param lambda The lambda value.
     * @return True if successful, false otherwise.
     */
    virtual bool CalculateL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, Float slope, Float alpha, Float lambda) = 0;

    /**
     * @brief Calculate the cross-entropy output delta for sparse data with the specified activation function.
     * @param activation The activation function.
     * @param position The position index.
     * @param batch The batch number.
     * @param stride The stride value.
     * @param pUnit Pointer to the unit values.
     * @param pDelta Pointer to store the calculated output delta values.
     * @return True if successful, false otherwise.
     */
    virtual bool CalculateCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta) = 0;

    /**
     * @brief Calculate the scaled marginal cross-entropy output delta for sparse data with the specified activation function.
     * @param activation The activation function.
     * @param position The position index.
     * @param batch The batch number.
     * @param stride The stride value.
     * @param pUnit Pointer to the unit values.
     * @param pDelta Pointer to store the calculated output delta values.
     * @return True if successful, false otherwise.
     */
    virtual bool CalculateScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta) = 0;

    /**
     * @brief Calculate the output delta for sparse data with the specified activation function.
     * @param activation The activation function.
     * @param position The position index.
     * @param batch The batch number.
     * @param stride The stride value.
     * @param pUnit Pointer to the unit values.
     * @param pDelta Pointer to store the calculated output delta values.
     * @param slope The slope value.
     * @param alpha The alpha value.
     * @param lambda The lambda value.
     * @return True if successful, false otherwise.
     */
    virtual bool CalculateOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, Float slope, Float alpha, Float lambda) = 0;

    /**
     * @brief Calculate the L2 Hinge output delta for sparse data with the specified activation function.
     * @param activation The activation function.
     * @param position The position index.
     * @param batch The batch number.
     * @param stride The stride value.
     * @param pUnit Pointer to the unit values.
     * @param pDelta Pointer to store the calculated output delta values.
     * @param slope The slope value.
     * @param alpha The alpha value.
     * @param lambda The lambda value.
     * @return True if successful, false otherwise.
     */
    virtual bool CalculateL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, Float slope, Float alpha, Float lambda) = 0;

    /**
     * @brief Calculate the data scaled marginal cross-entropy output delta for sparse data with the specified activation function.
     * @param activation The activation function.
     * @param position The position index.
     * @param batch The batch number.
     * @param stride The stride value.
     * @param pUnit Pointer to the unit values.
     * @param pDelta Pointer to store the calculated output delta values.
     * @return True if successful, false otherwise.
     */
    virtual bool CalculateDataScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta) = 0;

    /**
     * @brief Calculate the hinge output delta for sparse data with the specified activation function.
     * @param activation The activation function.
     * @param position The position index.
     * @param batch The batch number.
     * @param stride The stride value.
     * @param pUnit Pointer to the unit values.
     * @param pDelta Pointer to store the calculated output delta values.
     * @return True if successful, false otherwise.
     */
    virtual bool CalculateHingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta) = 0;

    /**
     * @brief Load dense data into the data set.
     * @param srcData Pointer to the source dense data.
     */
    virtual void LoadDenseData(const void* srcData) = 0;

    /**
     * @brief Copy dense data into the data set.
     * @param srcData Pointer to the source dense data.
     */
    virtual void CopyDenseData(const void* srcData) = 0;

    /**
     * @brief Load sparse data into the data set.
     * @param srcSparseStart Pointer to the source sparse start indices.
     * @param srcSparseEnd Pointer to the source sparse end indices.
     * @param srcSparseData Pointer to the source sparse data.
     * @param srcSparseIndex Pointer to the source sparse indices.
     */
    virtual void LoadSparseData(const uint64_t* srcSparseStart, const uint64_t* srcSparseEnd, const void* srcSparseData, const uint32_t* srcSparseIndex) = 0;

    /**
     * @brief Copy sparse data into the data set.
     * @param srcSparseStart Pointer to the source sparse start indices.
     * @param srcSparseEnd Pointer to the source sparse end indices.
     * @param srcSparseData Pointer to the source sparse data.
     * @param srcSparseIndex Pointer to the source sparse indices.
     */
    virtual void CopySparseData(const uint64_t* srcSparseStart, const uint64_t* srcSparseEnd, const void* srcSparseData, const uint32_t* srcSparseIndex) = 0;

    /**
     * @brief Load sparse data into the data set with long indices.
     * @param srcSparseStart Pointer to the source sparse start indices.
     * @param srcSparseEnd Pointer to the source sparse end indices.
     * @param srcSparseData Pointer to the source sparse data.
     * @param srcSparseIndex Pointer to the source sparse indices.
     */
    virtual void LoadSparseData(const long* srcSparseStart, const long* srcSparseEnd, const void* srcSparseData, const long* srcSparseIndex) = 0;

    /**
     * @brief Copy sparse data into the data set with long indices.
     * @param srcSparseStart Pointer to the source sparse start indices.
     * @param srcSparseEnd Pointer to the source sparse end indices.
     * @param srcSparseData Pointer to the source sparse data.
     * @param srcSparseIndex Pointer to the source sparse indices.
     */
    virtual void CopySparseData(const long* srcSparseStart, const long* srcSparseEnd, const void* srcSparseData, const long* srcSparseIndex) = 0;

    /**
     * @brief Load indexed data into the data set.
     * @param srcIndexedData Pointer to the source indexed data.
     */
    virtual void LoadIndexedData(const uint32_t* srcIndexedData) = 0;

    /**
     * @brief Load data weights into the data set.
     * @param srcWeightData Pointer to the source data weights.
     */
    virtual void LoadDataWeight(const Float* srcWeightData) = 0;

protected:
    /**
     * @brief Base class for dataset.
     *
     * This class represents a base dataset and provides common properties and functionality.
     *
     * @param name The name of the dataset.
     * @param dataType The data type of the dataset.
     * @param examples The total number of examples in the dataset.
     * @param uniqueExamples The number of unique examples in the dataset.
     * @param datasetDim The dimensions of the dataset.
     */
    DataSetBase(const std::string& name, DataSetEnums::DataType dataType, uint32_t examples, uint32_t uniqueExamples,
                  const DataSetDimensions& datasetDim);
};

/**
 * @brief Overloaded output stream operator for DataSetEnums::Attributes enumeration.
 *
 * @param out The output stream object.
 * @param a The DataSetEnums::Attributes value to be printed.
 * @return The modified output stream object.
 */
std::ostream& operator<<(std::ostream& out, DataSetEnums::Attributes& a);

/**
 * @brief Overloaded output stream operator for DataSetEnums::Kind enumeration.
 *
 * @param out The output stream object.
 * @param k The DataSetEnums::Kind value to be printed.
 * @return The modified output stream object.
 */
std::ostream& operator<<(std::ostream& out, DataSetEnums::Kind& k);

/**
 * @brief Overloaded output stream operator for DataSetEnums::DataType enumeration.
 *
 * @param out The output stream object.
 * @param t The DataSetEnums::DataType value to be printed.
 * @return The modified output stream object.
 */
std::ostream& operator<<(std::ostream& out, DataSetEnums::DataType& t);

/**
 * @brief Overloaded output stream operator for DataSetEnums::Sharding enumeration.
 *
 * @param out The output stream object.
 * @param s The DataSetEnums::Sharding value to be printed.
 * @return The modified output stream object.
 */
std::ostream& operator<<(std::ostream& out, DataSetEnums::Sharding& s);

/**
 * @brief Template concept for numeric types.
 *
 * This concept checks if the given type is a numeric type (arithmetic).
 *
 * @tparam T The type to check.
 */
template <typename T>
concept NumericType = std::is_arithmetic_v<T>;

/**
 * @brief Dataset class representing a specific data type.
 *
 * This class represents a dataset of a specific data type and inherits from DataSetBase.
 *
 * @tparam T The data type of the dataset.
 */
template <NumericType T>
class DataSet : public DataSetBase {
public:
    /**
     * @brief Friend class declaration of `Network`.
     */
    friend class Network;

    /**
     * @brief Friend class declaration of `Layer`.
     */
    friend class Layer;

    /**
     * @brief Friend function declaration of `LoadNetCDF`.
     * 
     * @param fname The file name.
     * @return Vector of DataSetBase pointers.
     */
    friend std::vector<DataSetBase*> LoadNetCDF(const std::string& fname);

    /**
     * @brief Friend function declaration of `SaveNetCDF`.
     * 
     * @param fname The file name.
     * @param vDataSet Vector of DataSetBase pointers.
     * @return True if saving is successful, false otherwise.
     */
    friend bool SaveNetCDF(const std::string& fname, std::vector<DataSetBase*> vDataSet);

private:
    std::vector<T> _vData; /**< Vector containing data of type T. */
    std::unique_ptr<GpuBuffer<T>> _pbData; /**< Unique pointer to a GPU buffer of type T. */
    std::vector<T> _vSparseData; /**< Vector containing sparse data of type T. */
    std::unique_ptr<GpuBuffer<T>> _pbSparseData; /**< Unique pointer to a GPU buffer containing sparse data of type T. */

    std::string _name; /**< Name of the object. */

    /**
     * Get the memory usage of the object.
     *
     * @return A tuple containing the total memory usage and free memory.
     */
    std::tuple<uint64_t, uint64_t> getMemoryUsage();

    /**
     * Calculate the counts of sparse datapoints.
     *
     * @return True if the calculation is successful, false otherwise.
     */
    bool calculateSparseDatapointCounts();

    /**
     * Generate the sparse transposed matrix for a given batch and layer.
     *
     * @param batch The batch index.
     * @param pLayer Pointer to the Layer object.
     * @return True if the generation is successful, false otherwise.
     */
    bool generateSparseTransposedMatrix(uint32_t batch, Layer<T>* pLayer);

    /**
     * Calculate the sparse transposed matrix for a given position, batch, and layer.
     *
     * @param position The position index.
     * @param batch The batch index.
     * @param pLayer Pointer to the Layer object.
     * @return True if the calculation is successful, false otherwise.
     */
    bool calculateSparseTransposedMatrix(uint32_t position, uint32_t batch, Layer<T>* pLayer);

    /**
     * Calculate the sparse transposed denoised matrix for a given position, batch, and layer.
     *
     * @param position The position index.
     * @param batch The batch index.
     * @param pLayer Pointer to the Layer object.
     * @return True if the calculation is successful, false otherwise.
     */
    bool calculateSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, Layer<T>* pLayer);

    /**
     * Calculate the sparse transposed weight gradient.
     *
     * @param alpha Alpha parameter.
     * @param beta Beta parameter.
     * @param m Dimension m.
     * @param n Dimension n.
     * @param pDelta Pointer to the delta values.
     * @param pWeightGradient Pointer to the weight gradient.
     * @return True if the calculation is successful, false otherwise.
     */
    bool calculateSparseTransposedWeightGradient(T alpha, T beta, uint32_t m, uint32_t n, T* pDelta, T* pWeightGradient);

    /**
     * Set the streaming flag.
     *
     * @param flag The streaming flag.
     * @return True if the flag is set successfully, false otherwise.
     */
    bool setStreaming(bool flag);

    /**
     * Get the streaming flag.
     *
     * @return The streaming flag value.
     */
    bool getStreaming();

    /**
     * Set the denoising flag.
     *
     * @param flag The denoising flag.
     * @return True if the flag is set successfully, false otherwise.
     */
    bool setDenoising(bool flag);

    /**
     * Generate the denoising data.
     *
     * @return True if the generation is successful, false otherwise.
     */
    bool generateDenoisingData();

    /**
     * Load the input unit for a given position, batch, and stride.
     *
     * @param position The position index.
     * @param batch The batch index.
     * @param stride The stride value.
     * @param pUnit Pointer to the input unit.
     * @return True if the loading is successful, false otherwise.
     */
    bool loadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, T* pUnit);

    /**
     * Load the sparse input unit for a given position, batch, and stride.
     *
     * @param position The position index.
     * @param batch The batch index.
     * @param stride The stride value.
     * @param pUnit Pointer to the input unit.
     * @return True if the loading is successful, false otherwise.
     */
    bool loadSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, T* pUnit);

    /**
     * Load the sparse denoised input unit for a given position, batch, and stride.
     *
     * @param position The position index.
     * @param batch The batch index.
     * @param stride The stride value.
     * @param pUnit Pointer to the input unit.
     * @return True if the loading is successful, false otherwise.
     */
    bool loadSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, T* pUnit);

    /**
     * Calculate the sparse Z value for a given position, batch, stride, weight, unit, and beta.
     *
     * @param position The position index.
     * @param batch The batch index.
     * @param stride The stride value.
     * @param pWeight Pointer to the weight value.
     * @param pUnit Pointer to the input unit.
     * @param beta The beta value.
     * @return True if the calculation is successful, false otherwise.
     */
    bool calculateSparseZ(uint32_t position, uint32_t batch, uint32_t stride, T* pWeight, T* pUnit, T beta);

    /**
     * Calculate the sparse denoised Z value for a given position, batch, stride, weight, unit, and beta.
     *
     * @param position The position index.
     * @param batch The batch index.
     * @param stride The stride value.
     * @param pWeight Pointer to the weight value.
     * @param pUnit Pointer to the input unit.
     * @param beta The beta value.
     * @return True if the calculation is successful, false otherwise.
     */
    bool calculateSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, T* pWeight, T* pUnit, T beta);

    /**
     * Calculate the L1 error for a given position, batch, stride, and unit.
     *
     * @param position The position index.
     * @param batch The batch index.
     * @param stride The stride value.
     * @param pUnit Pointer to the input unit.
     * @return The calculated L1 error.
     */
    float calculateL1Error(uint32_t position, uint32_t batch, uint32_t stride, T* pUnit);

    /**
     * Calculate the L2 error for a given position, batch, stride, and unit.
     *
     * @param position The position index.
     * @param batch The batch index.
     * @param stride The stride value.
     * @param pUnit Pointer to the input unit.
     * @return The calculated L2 error.
     */
    float calculateL2Error(uint32_t position, uint32_t batch, uint32_t stride, T* pUnit);

    /**
     * Calculate the L2 hinge error for a given position, batch, stride, and unit.
     *
     * @param position The position index.
     * @param batch The batch index.
     * @param stride The stride value.
     * @param pUnit Pointer to the input unit.
     * @return The calculated L2 hinge error.
     */
    float calculateL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, T* pUnit);

    /**
     * Calculate the cross-entropy error for a given position, batch, stride, and unit.
     *
     * @param position The position index.
     * @param batch The batch index.
     * @param stride The stride value.
     * @param pUnit Pointer to the input unit.
     * @return The calculated cross-entropy error.
     */
    float calculateCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, T* pUnit);

    /**
     * Calculate the scaled marginal cross-entropy error for a given position, batch, stride, and unit.
     *
     * @param position The position index.
     * @param batch The batch index.
     * @param stride The stride value.
     * @param pUnit Pointer to the input unit.
     * @return The calculated scaled marginal cross-entropy error.
     */
    float calculateScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, T* pUnit);

    /**
     * Calculate the multinomial cross-entropy error for a given position, batch, stride, and unit.
     *
     * @param position The position index.
     * @param batch The batch index.
     * @param stride The stride value.
     * @param pUnit Pointer to the input unit.
     * @return The calculated multinomial cross-entropy error.
     */
    float calculateMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, T* pUnit);

    /**
     * Calculate the multinomial scaled marginal cross-entropy error for a given position, batch, stride, and unit.
     *
     * @param position The position index.
     * @param batch The batch index.
     * @param stride The stride value.
     * @param pUnit Pointer to the input unit.
     * @return The calculated multinomial scaled marginal cross-entropy error.
     */
    float calculateMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, T* pUnit);

    /**
     * Calculate the data scaled marginal cross-entropy error for a given position, batch, stride, and unit.
     *
     * @param position The position index.
     * @param batch The batch index.
     * @param stride The stride value.
     * @param pUnit Pointer to the input unit.
     * @return The calculated data scaled marginal cross-entropy error.
     */
    float calculateDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, T* pUnit);

    /**
     * Calculate the hinge error for a given position, batch, stride, and unit.
     *
     * @param position The position index.
     * @param batch The batch index.
     * @param stride The stride value.
     * @param pUnit Pointer to the input unit.
     * @return The calculated hinge error.
     */
    float calculateHingeError(uint32_t position, uint32_t batch, uint32_t stride, T* pUnit);

    /**
     * Calculate the L1 output delta for a given activation, position, batch, stride, unit, slope, alpha, and lambda.
     *
     * @param activation The activation function.
     * @param position The position index.
     * @param batch The batch index.
     * @param stride The stride value.
     * @param pUnit Pointer to the input unit.
     * @param pDelta Pointer to the delta values.
     * @param slope The slope value.
     * @param alpha The alpha value.
     * @param lambda The lambda value.
     * @return True if the calculation is successful, false otherwise.
     */
    bool calculateL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, T* pUnit, T* pDelta, T slope, T alpha, T lambda);

    /**
     * Calculate the cross-entropy output delta for a given activation, position, batch, stride, unit, and delta.
     *
     * @param activation The activation function.
     * @param position The position index.
     * @param batch The batch index.
     * @param stride The stride value.
     * @param pUnit Pointer to the input unit.
     * @param pDelta Pointer to the delta values.
     * @return True if the calculation is successful, false otherwise.
     */
    bool calculateCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, T* pUnit, T* pDelta);

    /**
     * Calculate the scaled marginal cross-entropy output delta for a given activation, position, batch, stride, unit, and delta.
     *
     * @param activation The activation function.
     * @param position The position index.
     * @param batch The batch index.
     * @param stride The stride value.
     * @param pUnit Pointer to the input unit.
     * @param pDelta Pointer to the delta values.
     * @return True if the calculation is successful, false otherwise.
     */
    bool calculateScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, T* pUnit, T* pDelta);

    /**
     * Calculate the output delta for a given activation, position, batch, stride, unit, slope, alpha, and lambda.
     *
     * @param activation The activation function.
     * @param position The position index.
     * @param batch The batch index.
     * @param stride The stride value.
     * @param pUnit Pointer to the input unit.
     * @param pDelta Pointer to the delta values.
     * @param slope The slope value.
     * @param alpha The alpha value.
     * @param lambda The lambda value.
     * @return True if the calculation is successful, false otherwise.
     */
    bool calculateOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, T* pUnit, T* pDelta, T slope, T alpha, T lambda);

    /**
     * Calculate the L2 hinge output delta for a given activation, position, batch, stride, unit, slope, alpha, and lambda.
     *
     * @param activation The activation function.
     * @param position The position index.
     * @param batch The batch index.
     * @param stride The stride value.
     * @param pUnit Pointer to the input unit.
     * @param pDelta Pointer to the delta values.
     * @param slope The slope value.
     * @param alpha The alpha value.
     * @param lambda The lambda value.
     * @return True if the calculation is successful, false otherwise.
     */
    bool calculateL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, T* pUnit, T* pDelta, T slope, T alpha, T lambda);

    /**
     * Calculate the data scaled marginal cross-entropy output delta for a given activation, position, batch, stride, unit, and delta.
     *
     * @param activation The activation function.
     * @param position The position index.
     * @param batch The batch index.
     * @param stride The stride value.
     * @param pUnit Pointer to the input unit.
     * @param pDelta Pointer to the delta values.
     * @return True if the calculation is successful, false otherwise.
     */
    bool calculateDataScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, T* pUnit, T* pDelta);

    /**
     * Calculate the hinge output delta for a given activation, position, batch, stride, unit, and delta.
     *
     * @param activation The activation function.
     * @param position The position index.
     * @param batch The batch index.
     * @param stride The stride value.
     * @param pUnit Pointer to the input unit.
     * @param pDelta Pointer to the delta values.
     * @return True if the calculation is successful, false otherwise.
     */
    bool calculateHingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, T* pUnit, T* pDelta);


public:
    /**
     * @brief Constructs a new DataSet object with the given number of examples, dimensions, and name.
     * 
     * @param examples The number of examples in the dataset.
     * @param dim The dimensions of the dataset.
     * @param name The name of the dataset (optional).
     */
    DataSet(uint32_t examples, const DataSetDimensions& dim, const std::string& name = "");

    /**
     * @brief Constructs a new DataSet object with the given number of examples, unique examples, dimensions, and name.
     * 
     * @param examples The number of examples in the dataset.
     * @param uniqueExamples The number of unique examples in the dataset.
     * @param dim The dimensions of the dataset.
     * @param name The name of the dataset (optional).
     */
    DataSet(uint32_t examples, uint32_t uniqueExamples, const DataSetDimensions& dim, const std::string& name = "");

    /**
     * @brief Constructs a new DataSet object with the given number of examples, sparse density, dimensions, weighted flag, and name.
     * 
     * @param examples The number of examples in the dataset.
     * @param sparseDensity The density of the sparse data.
     * @param dim The dimensions of the dataset.
     * @param isWeighted Specifies if the dataset is weighted (optional, default is false).
     * @param name The name of the dataset (optional).
     */
    DataSet(uint32_t examples, T sparseDensity, const DataSetDimensions& dim, bool isWeighted = false, const std::string& name = "");

    /**
     * @brief Constructs a new DataSet object with the given number of examples, unique examples, sparse data size, dimensions, indexed flag, weighted flag, and name.
     * 
     * @param examples The number of examples in the dataset.
     * @param uniqueExamples The number of unique examples in the dataset.
     * @param sparseDataSize The size of the sparse data.
     * @param dim The dimensions of the dataset.
     * @param isIndexed Specifies if the dataset is indexed (optional, default is false).
     * @param isWeighted Specifies if the dataset is weighted (optional, default is false).
     * @param name The name of the dataset (optional).
     */
    DataSet(uint32_t examples, uint32_t uniqueExamples, size_t sparseDataSize, const DataSetDimensions& dim,
            bool isIndexed = false, bool isWeighted = false, const std::string& name = "");

    /**
     * @brief Loads dense data into the dataset from the source data.
     * 
     * @param srcData Pointer to the source dense data.
     */
    void loadDenseData(const void* srcData);

    /**
     * @brief Copies dense data from the source data to the dataset.
     * 
     * @param srcData Pointer to the source dense data.
     */
    void copyDenseData(const void* srcData);

    /**
     * @brief Loads sparse data into the dataset from the source sparse data, start indices, end indices, and sparse indices.
     * 
     * @param srcSparseStart Pointer to the start indices of the source sparse data.
     * @param srcSparseEnd Pointer to the end indices of the source sparse data.
     * @param srcSparseData Pointer to the source sparse data.
     * @param srcSparseIndex Pointer to the sparse indices of the source sparse data.
     */
    void loadSparseData(const uint64_t* srcSparseStart, const uint64_t* srcSparseEnd, const void* srcSparseData,
                        const uint32_t* srcSparseIndex);

    /**
     * @brief Copies sparse data from the source sparse data, start indices, end indices, and sparse indices to the dataset.
     * 
     * @param srcSparseStart Pointer to the start indices of the source sparse data.
     * @param srcSparseEnd Pointer to the end indices of the source sparse data.
     * @param srcSparseData Pointer to the source sparse data.
     * @param srcSparseIndex Pointer to the sparse indices of the source sparse data.
     */
    void copySparseData(const uint64_t* srcSparseStart, const uint64_t* srcSparseEnd, const void* srcSparseData,
                        const uint32_t* srcSparseIndex);

    /**
     * @brief Loads sparse data into the dataset from the source sparse data, start indices, end indices, and sparse indices.
     * 
     * @param srcSparseStart Pointer to the start indices of the source sparse data.
     * @param srcSparseEnd Pointer to the end indices of the source sparse data.
     * @param srcSparseData Pointer to the source sparse data.
     * @param srcSparseIndex Pointer to the sparse indices of the source sparse data.
     */
    void loadSparseData(const long* srcSparseStart, const long* srcSparseEnd, const void* srcSparseData,
                        const long* srcSparseIndex);

    /**
     * @brief Copies sparse data from the source sparse data, start indices, end indices, and sparse indices to the dataset.
     * 
     * @param srcSparseStart Pointer to the start indices of the source sparse data.
     * @param srcSparseEnd Pointer to the end indices of the source sparse data.
     * @param srcSparseData Pointer to the source sparse data.
     * @param srcSparseIndex Pointer to the sparse indices of the source sparse data.
     */
    void copySparseData(const long* srcSparseStart, const long* srcSparseEnd, const void* srcSparseData,
                        const long* srcSparseIndex);

    /**
     * @brief Loads indexed data into the dataset from the source indexed data.
     * 
     * @param srcIndexedData Pointer to the source indexed data.
     */
    void loadIndexedData(const uint32_t* srcIndexedData);

    /**
     * @brief Loads data weights into the dataset from the source weight data.
     * 
     * @param srcWeightData Pointer to the source weight data.
     */
    void loadDataWeight(const T* srcWeightData);

    /**
     * @brief Destroys the DataSet object and frees the associated resources.
     */
    ~DataSet();

    /**
     * @brief Shuffles the data points in the dataset randomly.
     */
    void shuffle();

    /**
     * @brief Retrieves the data point at the specified indices.
     * 
     * @param n The example index.
     * @param x The X dimension index.
     * @param y The Y dimension index (optional, default is 0).
     * @param z The Z dimension index (optional, default is 0).
     * @return The value of the data point.
     */
    T getDataPoint(uint32_t n, uint32_t x, uint32_t y = 0, uint32_t z = 0);

    /**
     * @brief Sets the value of the data point at the specified indices.
     * 
     * @param v The new value of the data point.
     * @param n The example index.
     * @param x The X dimension index.
     * @param y The Y dimension index (optional, default is 0).
     * @param z The Z dimension index (optional, default is 0).
     * @return True if the data point was successfully set, false otherwise.
     */
    bool setDataPoint(T v, uint32_t n, uint32_t x, uint32_t y = 0, uint32_t z = 0);

    /**
     * @brief Retrieves the number of sparse data points for the specified example index.
     * 
     * @param n The example index.
     * @return The number of sparse data points.
     */
    uint64_t getSparseDataPoints(uint32_t n);

    /**
     * @brief Retrieves the sparse index at the specified example and sparse data point indices.
     * 
     * @param n The example index.
     * @param i The sparse data point index.
     * @return The sparse index.
     */
    uint32_t getSparseIndex(uint32_t n, uint32_t i);

    /**
     * @brief Sets the sparse index at the specified example and sparse data point indices.
     * 
     * @param n The example index.
     * @param i The sparse data point index.
     * @param v The new value of the sparse index.
     * @return True if the sparse index was successfully set, false otherwise.
     */
    bool setSparseIndex(uint32_t n, uint32_t i, uint32_t v);

    /**
     * @brief Retrieves the value of the sparse data point at the specified example and sparse data point indices.
     * 
     * @param n The example index.
     * @param i The sparse data point index.
     * @return The value of the sparse data point.
     */
    T getSparseDataPoint(uint32_t n, uint32_t i);

    /**
     * @brief Sets the value of the sparse data point at the specified example and sparse data point indices.
     * 
     * @param n The example index.
     * @param i The sparse data point index.
     * @param v The new value of the sparse data point.
     * @return True if the sparse data point was successfully set, false otherwise.
     */
    bool setSparseDataPoint(uint32_t n, uint32_t i, T v);

};

/**
 * @brief Loads the input unit from the dataset at the specified position and batch.
 *
 * @tparam T The data type of the dataset.
 * @param position The position of the input unit.
 * @param batch The batch index.
 * @param stride The stride of the input unit.
 * @param pUnit Pointer to the memory location to store the loaded input unit.
 * @return Returns true if the input unit is loaded successfully.
 */
template<typename T>
bool DataSet<T>::LoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit)
{
    if (_attributes & DataSetEnums::Indexed)
        kLoadIndexedInputUnit(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData);
    else
        kLoadInputUnit(position, batch, stride, pUnit, _pbData->_pDevData);
    
    return true;
}

/**
 * @brief Loads the sparse input unit from the dataset at the specified position and batch.
 *
 * @tparam T The data type of the dataset.
 * @param position The position of the input unit.
 * @param batch The batch index.
 * @param stride The stride of the input unit.
 * @param pUnit Pointer to the memory location to store the loaded sparse input unit.
 * @return Returns true if the sparse input unit is loaded successfully.
 */
template<typename T>
bool DataSet<T>::LoadSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit)
{
    Float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : NULL;

    if (_attributes & DataSetEnums::Boolean)
    {
        if (_attributes & DataSetEnums::Indexed)
            kLoadIndexedSparseInputUnit(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight);
        else
            kLoadSparseInputUnit(position, batch, stride, pUnit, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight);
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            kLoadIndexedSparseAnalogInputUnit(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData);
        else
            kLoadSparseAnalogInputUnit(position, batch, stride, pUnit, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData);
    }

    return true;
}

/**
 * @brief Loads the sparse denoised input unit from the dataset at the specified position and batch.
 *
 * @tparam T The data type of the dataset.
 * @param position The position of the input unit.
 * @param batch The batch index.
 * @param stride The stride of the input unit.
 * @param pUnit Pointer to the memory location to store the loaded sparse denoised input unit.
 * @return Returns true if the sparse denoised input unit is loaded successfully.
 */
template<typename T>
bool DataSet<T>::LoadSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit)
{
    Float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : NULL;

    if (_attributes & DataSetEnums::Boolean)
    {
        if (_attributes & DataSetEnums::Indexed)
            kLoadIndexedSparseDenoisedInputUnit(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbDenoisingRandom->_pDevData);
        else
            kLoadSparseDenoisedInputUnit(position, batch, stride, pUnit, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbDenoisingRandom->_pDevData);
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            kLoadIndexedSparseAnalogDenoisedInputUnit(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, _pbDenoisingRandom->_pDevData);
        else
            kLoadSparseAnalogDenoisedInputUnit(position, batch, stride, pUnit, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, _pbDenoisingRandom->_pDevData);
    }

    return true;
}

/**
 * @brief Calculates the sparse Z values for the given position, batch, weight, unit, and beta value.
 *
 * @tparam T The data type of the dataset.
 * @param position The position of the input unit.
 * @param batch The batch index.
 * @param stride The stride of the input unit.
 * @param pWeight Pointer to the weight values.
 * @param pUnit Pointer to the input unit.
 * @param beta The beta value.
 * @return Returns true if the sparse Z values are calculated successfully.
 */
template<typename T>
bool DataSet<T>::CalculateSparseZ(uint32_t position, uint32_t batch, uint32_t stride, Float* pWeight, Float* pUnit, Float beta)
{
    Float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : NULL;

    if (_attributes & DataSetEnums::Boolean)
    {
        if (_attributes & DataSetEnums::Indexed)
            kCalculateIndexedSparseZ(position, batch, stride, pWeight, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, pUnit, beta);
        else
            kCalculateSparseZ(position, batch, stride, pWeight, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, pUnit, beta);
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            kCalculateIndexedSparseAnalogZ(position, batch, stride, pWeight, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, pUnit, beta);
        else
            kCalculateSparseAnalogZ(position, batch, stride, pWeight, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, pUnit, beta);
    }

    return true;
}

/**
 * @brief Calculates the sparse denoised Z values for the given position, batch, weight, unit, and beta value.
 *
 * @tparam T The data type of the dataset.
 * @param position The position of the input unit.
 * @param batch The batch index.
 * @param stride The stride of the input unit.
 * @param pWeight Pointer to the weight values.
 * @param pUnit Pointer to the input unit.
 * @param beta The beta value.
 * @return Returns true if the sparse denoised Z values are calculated successfully.
 */
template<typename T>
bool DataSet<T>::CalculateSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, Float* pWeight, Float* pUnit, Float beta)
{
    Float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : NULL;

    if (_attributes & DataSetEnums::Boolean)
    {
        if (_attributes & DataSetEnums::Indexed)
            kCalculateIndexedSparseDenoisedZ(position, batch, stride, pWeight, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbDenoisingRandom->_pDevData, pUnit, beta);
        else
            kCalculateSparseDenoisedZ(position, batch, stride, pWeight, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbDenoisingRandom->_pDevData, pUnit, beta);
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            kCalculateIndexedSparseAnalogDenoisedZ(position, batch, stride, pWeight, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, _pbDenoisingRandom->_pDevData, pUnit, beta);
        else
            kCalculateSparseAnalogDenoisedZ(position, batch, stride, pWeight, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, _pbDenoisingRandom->_pDevData, pUnit, beta);
    }

    return true;
}

/**
 * @brief Calculates the sparse transposed matrix for the given position, batch, and layer.
 *
 * @tparam T The data type of the dataset.
 * @param position The position of the input unit.
 * @param batch The batch index.
 * @param pLayer Pointer to the layer.
 * @return Returns true if the sparse transposed matrix is calculated successfully.
 */
template<typename T>
bool DataSet<T>::CalculateSparseTransposedMatrix(uint32_t position, uint32_t batch, Layer* pLayer)
{
    if (_bDirty || (batch != _batch))
    {
        GenerateSparseTransposedMatrix(batch, pLayer);
    }

    _pbSparseTransposedEnd->Copy(_pbSparseTransposedStart->_pDevData);

    Float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : NULL;
    Float* pSparseTransposedData = ((_attributes & DataSetEnums::Weighted) || !(_attributes & DataSetEnums::Boolean)) ? _pbSparseTransposedData->_pDevData : NULL;

    if (_attributes & DataSetEnums::Boolean)
    {
        if (_attributes & DataSetEnums::Indexed)
            kCalculateIndexedSparseTransposedMatrix(position, batch, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData);
        else
            kCalculateSparseTransposedMatrix(position, batch, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData);
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            kCalculateIndexedSparseTransposedAnalogMatrix(position, batch, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData);
        else
            kCalculateSparseTransposedAnalogMatrix(position, batch, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData);
    }

    return true;
}
template<typename T>
bool DataSet<T>::CalculateSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, Layer* pLayer)
{
    if (_bDirty || (batch != _batch))
    {
        GenerateSparseTransposedMatrix(batch, pLayer);
    }

    _pbSparseTransposedEnd->Copy(_pbSparseTransposedStart->_pDevData);

    Float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    Float* pSparseTransposedData = ((_attributes & DataSetEnums::Weighted) || !(_attributes & DataSetEnums::Boolean)) ? _pbSparseTransposedData->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Boolean)
    {
        if (_attributes & DataSetEnums::Indexed)
        {
            kCalculateIndexedSparseTransposedDenoisedMatrix(position, batch, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbDenoisingRandom->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData);
        }
        else
        {
            kCalculateSparseTransposedDenoisedMatrix(position, batch, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbDenoisingRandom->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData);
        }
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
        {
            kCalculateIndexedSparseTransposedAnalogDenoisedMatrix(position, batch, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, _pbDenoisingRandom->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData);
        }
        else
        {
            kCalculateSparseTransposedAnalogDenoisedMatrix(position, batch, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, _pbDenoisingRandom->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData);
        }
    }

#if 0
    std::vector<uint32_t> vSparseTransposedStart(53120);
    std::vector<uint32_t> vSparseTransposedEnd(53120);
    _pbSparseTransposedStart->Download(&vSparseTransposedStart[0]);
    _pbSparseTransposedEnd->Download(&vSparseTransposedEnd[0]);
    for (uint32_t i = 0; i < 53120; i++)
        std::cout("%6u %9u %9u %9u %9u\n", i, vSparseTransposedStart[i], vSparseTransposedEnd[i], vSparseTransposedEnd[i] - vSparseTransposedStart[i], static_cast<uint32_t>(_vSparseDatapointCount[i]));
    exit(-1);
#endif
    return true;
}
template<typename T>
bool DataSet<T>::CalculateSparseTransposedWeightGradient(Float alpha, Float beta, uint32_t m, uint32_t n, Float* pDelta, Float* pWeightGradient)
{
    if ((_attributes & DataSetEnums::Boolean) && !(_attributes & DataSetEnums::Weighted))
        kCalculateSparseTransposedWeightGradient(alpha, beta, m, n, _pbSparseTransposedStart->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pDelta, pWeightGradient);
    else
        kCalculateSparseTransposedAnalogWeightGradient(alpha, beta, m, n, _pbSparseTransposedStart->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, _pbSparseTransposedData->_pDevData, pDelta, pWeightGradient);
    return true;
}

template<typename T>
float DataSet<T>::CalculateL1Error(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit)
{
    Float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Boolean)
        {
            if (_attributes & DataSetEnums::Indexed)
            {
                return kCalculateIndexedSparseL1Error(position, batch, stride, pUnit,
                    _pbIndex->_pDevData,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    bSparseIgnoreZero);
            }
            else
            {
                return kCalculateSparseL1Error(position, batch, stride, pUnit,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    bSparseIgnoreZero);
            }
        }
        else
        {
            if (_attributes & DataSetEnums::Indexed)
            {
                return kCalculateIndexedSparseAnalogL1Error(position, batch, stride, pUnit,
                    _pbIndex->_pDevData,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    _pbSparseData->_pDevData,
                    bSparseIgnoreZero);
            }
            else
            {
                return kCalculateSparseAnalogL1Error(position, batch, stride, pUnit,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    _pbSparseData->_pDevData,
                    bSparseIgnoreZero);
            }
        }
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            return kCalculateIndexedL1Error(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
        else
            return kCalculateL1Error(position, batch, stride, pUnit, _pbData->_pDevData, pDataWeight);
    }
}
template<typename T>
float DataSet<T>::CalculateL2Error(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit)
{
    Float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Boolean)
        {
            if (_attributes & DataSetEnums::Indexed)
            {
                return kCalculateIndexedSparseL2Error(position, batch, stride, pUnit,
                    _pbIndex->_pDevData,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    bSparseIgnoreZero);
            }
            else
            {
                return kCalculateSparseL2Error(position, batch, stride, pUnit,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    bSparseIgnoreZero);
            }
        }
        else
        {
            if (_attributes & DataSetEnums::Indexed)
            {
                return kCalculateIndexedSparseAnalogL2Error(position, batch, stride, pUnit,
                    _pbIndex->_pDevData,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    _pbSparseData->_pDevData,
                    bSparseIgnoreZero);
            }
            else
            {
                return kCalculateSparseAnalogL2Error(position, batch, stride, pUnit,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    _pbSparseData->_pDevData,
                    bSparseIgnoreZero);
            }
        }
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            return kCalculateIndexedL2Error(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
        else
            return kCalculateL2Error(position, batch, stride, pUnit, _pbData->_pDevData, pDataWeight);
    }
}
template<typename T>
float DataSet<T>::CalculateL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit)
{
    Float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Boolean)
        {
            if (_attributes & DataSetEnums::Indexed)
            {
                return kCalculateIndexedSparseL2HingeError(position, batch, stride, pUnit,
                    _pbIndex->_pDevData,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    bSparseIgnoreZero);
            }
            else
            {
                return kCalculateSparseL2HingeError(position, batch, stride, pUnit,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    bSparseIgnoreZero);
            }
        }
        else
        {
            if (_attributes & DataSetEnums::Indexed)
            {
                return kCalculateIndexedSparseAnalogL2HingeError(position, batch, stride, pUnit,
                    _pbIndex->_pDevData,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    _pbSparseData->_pDevData,
                    bSparseIgnoreZero);
            }
            else
            {
                return kCalculateSparseAnalogL2HingeError(position, batch, stride, pUnit,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    _pbSparseData->_pDevData,
                    bSparseIgnoreZero);
            }
        }
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            return kCalculateIndexedL2HingeError(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
        else
            return kCalculateL2HingeError(position, batch, stride, pUnit, _pbData->_pDevData, pDataWeight);
    }
}
template<typename T>
float DataSet<T>::CalculateCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit)
{
    Float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Indexed)
        {
            return kCalculateIndexedSparseCrossEntropyError(position, batch, stride, pUnit,
                _pbIndex->_pDevData,
                _pbSparseStart->_pDevData,
                _pbSparseEnd->_pDevData,
                _pbSparseIndex->_pDevData,
                pDataWeight,
                bSparseIgnoreZero);
        }
        else
        {
            return kCalculateSparseCrossEntropyError(position, batch, stride, pUnit,
                _pbSparseStart->_pDevData,
                _pbSparseEnd->_pDevData,
                _pbSparseIndex->_pDevData,
                pDataWeight,
                bSparseIgnoreZero);
        }
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            return kCalculateIndexedCrossEntropyError(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
        else
            return kCalculateCrossEntropyError(position, batch, stride, pUnit, _pbData->_pDevData, pDataWeight);
    }
}
template<typename T>
float DataSet<T>::CalculateScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit)
{
    Float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Indexed)
        {
            return kCalculateIndexedSparseScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                _pbIndex->_pDevData,
                _pbSparseStart->_pDevData,
                _pbSparseEnd->_pDevData,
                _pbSparseIndex->_pDevData,
                pDataWeight,
                bSparseIgnoreZero);
        }
        else
        {
            return kCalculateSparseScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                _pbSparseStart->_pDevData,
                _pbSparseEnd->_pDevData,
                _pbSparseIndex->_pDevData,
                pDataWeight,
                bSparseIgnoreZero);
        }
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            return kCalculateIndexedScaledMarginalCrossEntropyError(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
        else
            return kCalculateScaledMarginalCrossEntropyError(position, batch, stride, pUnit, _pbData->_pDevData, pDataWeight);
    }
}
template<typename T>
float DataSet<T>::CalculateMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit)
{
    Float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Sparse)
    {
        if (_attributes & DataSetEnums::Boolean)
        {
            if (_attributes & DataSetEnums::Indexed)
            {
                return kCalculateIndexedSparseMultinomialCrossEntropyError(position, batch, stride, pUnit,
                    _pbIndex->_pDevData,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight);
            }
            else
            {
                return kCalculateSparseMultinomialCrossEntropyError(position, batch, stride, pUnit,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight);
            }
        }
        else
        {
            if (_attributes & DataSetEnums::Indexed)
            {
                return kCalculateIndexedSparseAnalogMultinomialCrossEntropyError(position, batch, stride, pUnit,
                    _pbIndex->_pDevData,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    _pbSparseData->_pDevData);
            }
            else
            {
                return kCalculateSparseAnalogMultinomialCrossEntropyError(position, batch, stride, pUnit,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    _pbSparseData->_pDevData);
            }
        }
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            return kCalculateIndexedMultinomialCrossEntropyError(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
        else
            return kCalculateMultinomialCrossEntropyError(position, batch, stride, pUnit, _pbData->_pDevData, pDataWeight);
    }
}
template<typename T>
float DataSet<T>::CalculateMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit)
{
    Float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Sparse)
    {
        if (_attributes & DataSetEnums::Boolean)
        {
            if (_attributes & DataSetEnums::Indexed)
            {
                return kCalculateIndexedSparseMultinomialScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                    _pbIndex->_pDevData,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight);
            }
            else
            {
                return kCalculateSparseMultinomialScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight);
            }
        }
        else
        {
            if (_attributes & DataSetEnums::Indexed)
            {
                return kCalculateIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                    _pbIndex->_pDevData,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    _pbSparseData->_pDevData);
            }
            else
            {
                return kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    _pbSparseData->_pDevData);
            }
        }
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            return kCalculateIndexedMultinomialScaledMarginalCrossEntropyError(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
        else
            return kCalculateMultinomialScaledMarginalCrossEntropyError(position, batch, stride, pUnit, _pbData->_pDevData, pDataWeight);
    }
}
template<typename T>
float DataSet<T>::CalculateDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit)
{
    Float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Boolean)
        {
            if (_attributes & DataSetEnums::Indexed)
            {
                return kCalculateIndexedSparseScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                    _pbIndex->_pDevData,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    bSparseIgnoreZero);
            }
            else
            {
                return kCalculateSparseScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    bSparseIgnoreZero);
            }
        }
        else
        {
            if (_attributes & DataSetEnums::Indexed)
            {
                return kCalculateIndexedSparseDataScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                    _pbIndex->_pDevData,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    _pbSparseData->_pDevData,
                    bSparseIgnoreZero);
            }
            else
            {
                return kCalculateSparseDataScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    _pbSparseData->_pDevData,
                    bSparseIgnoreZero);
            }
        }
    }
    else
    {
        std::cout << "unsupported data format of this cost function" << std::endl;
        getGpu().Shutdown();
        std::exit(-1);
    }
}
template<typename T>
float DataSet<T>::CalculateHingeError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit)
{
    Float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Indexed)
        return kCalculateIndexedHingeError(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
    else
        return kCalculateHingeError(position, batch, stride, pUnit, _pbData->_pDevData, pDataWeight);
}

template<typename T>
bool DataSet<T>::CalculateL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, Float slope, Float alpha, Float lambda)
{
    Float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Indexed)
            kCalculateIndexedSparseL1OutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, bSparseIgnoreZero, slope, alpha, lambda);
        else
            kCalculateSparseL1OutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, bSparseIgnoreZero, slope, alpha, lambda);
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            kCalculateIndexedL1OutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight, slope, alpha, lambda);
        else
            kCalculateL1OutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbData->_pDevData, pDataWeight, slope, alpha, lambda);
    }
    return true;
}
template<typename T>
bool DataSet<T>::CalculateCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta)
{
    Float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Indexed)
            kCalculateIndexedSparseCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, bSparseIgnoreZero);
        else
            kCalculateSparseCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, bSparseIgnoreZero);

    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            kCalculateIndexedCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
        else
            kCalculateCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbData->_pDevData, pDataWeight);
    }
    return true;
}

template<typename T>
bool DataSet<T>::CalculateScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta)
{
    Float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Indexed)
            kCalculateIndexedSparseScaledMarginalCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, bSparseIgnoreZero);
        else
            kCalculateSparseScaledMarginalCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, bSparseIgnoreZero);
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            kCalculateIndexedScaledMarginalCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
        else
            kCalculateScaledMarginalCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbData->_pDevData, pDataWeight);
    }
    return true;
}
template<typename T>
bool DataSet<T>::CalculateOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, Float slope, Float alpha, Float lambda)
{
    Float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Sparse) {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Boolean) 
        {
            if (_attributes & DataSetEnums::Indexed)
                kCalculateIndexedSparseOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, bSparseIgnoreZero, slope, alpha, lambda);
            else
                kCalculateSparseOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, bSparseIgnoreZero, slope, alpha, lambda);
        } 
        else 
        {
            if (_attributes & DataSetEnums::Indexed)
                kCalculateIndexedSparseAnalogOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, bSparseIgnoreZero, slope, alpha, lambda);
            else
                kCalculateSparseAnalogOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, bSparseIgnoreZero, slope, alpha, lambda);
        }
    } 
    else 
    {
        if (_attributes & DataSetEnums::Indexed)
            kCalculateIndexedOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight, slope, alpha, lambda);
        else
            kCalculateOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbData->_pDevData, pDataWeight, slope, alpha, lambda);
    }
    return true;
}


template<typename T>
bool DataSet<T>::CalculateL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, Float slope, Float alpha, Float lambda)
{
    Float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Sparse) {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Boolean) 
        {
            if (_attributes & DataSetEnums::Indexed)
                kCalculateIndexedSparseL2HingeOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, bSparseIgnoreZero, slope, alpha, lambda);
            else
                kCalculateSparseL2HingeOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, bSparseIgnoreZero, slope, alpha, lambda);
        } 
        else 
        {
            if (_attributes & DataSetEnums::Indexed)
                kCalculateIndexedSparseAnalogL2HingeOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, bSparseIgnoreZero, slope, alpha, lambda);
            else
                kCalculateSparseAnalogL2HingeOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, bSparseIgnoreZero, slope, alpha, lambda);
        }
    } 
    else 
    {
        if (_attributes & DataSetEnums::Indexed)
            kCalculateIndexedL2HingeOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight, slope, alpha, lambda);
        else
            kCalculateL2HingeOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbData->_pDevData, pDataWeight, slope, alpha, lambda);
    }
    return true;
}
template<typename T>
bool DataSet<T>::CalculateDataScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta)
{
    if (_attributes & DataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Indexed)
        {
            kCalculateIndexedSparseDataScaledMarginalCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta,
                             _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData,
                            _pbSparseData->_pDevData, bSparseIgnoreZero);
        }
        else
        {
            kCalculateSparseDataScaledMarginalCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta,
                            _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData,
                            _pbSparseData->_pDevData, bSparseIgnoreZero);
        }
    } 
    else 
    {
        std::cout << "unsupported data format of this cost function" << std::endl;
        getGpu().Shutdown();
        std::exit(-1);
    }
    return true;
}

template<typename T>
bool DataSet<T>::CalculateHingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta)
{
    Float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Indexed)
        kCalculateIndexedHingeOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
    else
        kCalculateHingeOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbData->_pDevData, pDataWeight);
    return true;
}

std::vector<DataSetBase*> LoadNetCDF(const std::string& fname);
bool SaveNetCDF(const std::string& fname, std::vector<DataSetBase*> vDataset);
std::vector<DataSetBase*> LoadImageData(const std::string& fname);
std::vector<DataSetBase*> LoadCSVData(const std::string& fname);
std::vector<DataSetBase*> LoadJSONData(const std::string& fname);
std::vector<DataSetBase*> LoadAudioData(const std::string& name);

#endif
