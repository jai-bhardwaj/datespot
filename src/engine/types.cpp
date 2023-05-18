#include <stdexcept>
#include <sstream>
#include <map>
#include <string>
#include <iostream>
#include <concepts>

#include "GpuTypes.h"
#include "NcExcptionWrap.h"
#include "Types.h"
#include "kernels.h"

/**
 * @brief Template class for DataSet.
 *
 * @tparam T The data type of the DataSet.
 */
template <typename T>
class DataSet;

/**
 * @brief Instantiate DataSet with Float data type.
 */
template class DataSet<Float>;

/**
 * @brief Instantiate DataSet with double data type.
 */
template class DataSet<double>;

/**
 * @brief Instantiate DataSet with unsigned char data type.
 */
template class DataSet<unsigned char>;

/**
 * @brief Instantiate DataSet with char data type.
 */
template class DataSet<char>;

/**
 * @brief Instantiate DataSet with uint32_t data type.
 */
template class DataSet<uint32_t>;

/**
 * @brief Instantiate DataSet with uint64_t data type.
 */
template class DataSet<uint64_t>;

/**
 * @brief Instantiate DataSet with int32_t data type.
 */
template class DataSet<int32_t>;

/**
 * @brief Instantiate DataSet with int64_t data type.
 */
template class DataSet<int64_t>;

/**
 * @brief Namespace for netCDF.
 */
using namespace netCDF;

/**
 * @brief Namespace for netCDF exceptions.
 */
using namespace netCDF::exceptions;

/**
 * @brief Map to store string representations of TrainingMode enum values.
 */
static std::map<TrainingMode, std::string> sTrainingModeMap = {
    {TrainingMode::SGD,      "SGD"},
    {TrainingMode::Momentum, "Momentum"},
    {TrainingMode::AdaGrad,  "AdaGrad"},
    {TrainingMode::Nesterov, "Nesterov"},
    {TrainingMode::RMSProp,  "RMSProp"},
    {TrainingMode::AdaDelta, "AdaDelta"},
    {TrainingMode::Adam,     "Adam"}
};

/**
 * @brief Concept for checking if a type is an enum.
 *
 * @tparam T The type to check.
 */
template <typename T>
concept EnumType = std::is_enum_v<T>;

/**
 * @brief Overloaded operator<< for printing EnumType values.
 *
 * @tparam T The EnumType.
 * @param out The output stream.
 * @param e The EnumType value.
 * @return std::ostream& The output stream.
 */
template <EnumType T>
std::ostream& operator<<(std::ostream& out, const T& e)
{
    out << sTrainingModeMap[e];
    return out;
}

/**
 * @brief Map to store string representations of ErrorFunction enum values.
 */
static std::map<ErrorFunction, std::string> sErrorFunctionMap = {
    {ErrorFunction::L1,                             "L1"},
    {ErrorFunction::L2,                             "L2"},
    {ErrorFunction::CrossEntropy,                   "CrossEntropy"},
    {ErrorFunction::ScaledMarginalCrossEntropy,     "ScaledMarginalCrossEntropy"},
    {ErrorFunction::Hinge,                          "Hinge"},
    {ErrorFunction::L2Hinge,                        "L2Hinge"},
};

/**
 * @brief Overloaded operator<< for printing ErrorFunction values.
 *
 * @tparam T The ErrorFunction.
 * @param out The output stream.
 * @param e The ErrorFunction value.
 * @return std::ostream& The output stream.
 */
template <EnumType T>
std::ostream& operator<<(std::ostream& out, const T& e)
{
    out << sErrorFunctionMap[e];
    return out;
}

/**
 * @brief Map to store string representations of Activation enum values.
 */
static std::map<Activation, std::string> sActivationMap = {
    {Activation::Sigmoid,                              "Sigmoid"},
    {Activation::Tanh,                                 "Tanh"},
    {Activation::Linear,                               "Linear"},
    {Activation::ParametricRectifiedLinear,            "ParametricRectifiedLinear"},
    {Activation::SoftSign,                             "SoftSign"},
    {Activation::SoftPlus,                             "SoftPlus"},
    {Activation::SoftMax,                              "SoftMax"},
    {Activation::RELUMax,                              "RELUMax"},
    {Activation::LinearMax,                            "LinearMax"},
    {Activation::RectifiedLinear,                      "RectifiedLinear"},
    {Activation::LeakyRectifiedLinear,                 "LeakyRectifiedLinear"},
    {Activation::ExponentialLinear,                    "ExponentialLinear"},
    {Activation::ScaledExponentialLinear,              "ScaledExponentialLinear"}
};

/**
 * @brief Overloaded operator<< for printing Activation values.
 *
 * @tparam T The Activation.
 * @param out The output stream.
 * @param a The Activation value.
 * @return std::ostream& The output stream.
 */
template <EnumType T>
std::ostream& operator<<(std::ostream& out, const T& a)
{
    out << sActivationMap[a];
    return out;
}

/**
 * @brief Map to store string representations of WeightInitialization enum values.
 */
static std::map<WeightInitialization, std::string> sWeightInitializationMap = {
    {WeightInitialization::Xavier,           "Xavier"},
    {WeightInitialization::CaffeXavier,      "CaffeXavier"},
    {WeightInitialization::Gaussian,         "Gaussian"},
    {WeightInitialization::Uniform,          "Uniform"},
    {WeightInitialization::UnitBall,         "UnitBall"},
    {WeightInitialization::Constant,         "Constant"},
    {WeightInitialization::SELU,             "SELU"}
};

/**
 * @brief Overloaded operator<< for printing WeightInitialization values.
 *
 * @tparam T The WeightInitialization.
 * @param out The output stream.
 * @param w The WeightInitialization value.
 * @return std::ostream& The output stream.
 */
template <EnumType T>
std::ostream& operator<<(std::ostream& out, const T& w)
{
    out << sWeightInitializationMap[w];
    return out;
}

/**
 * @brief Map to store string representations of PoolingFunction enum values.
 */
static std::map<PoolingFunction, std::string> sPoolingFunctionMap = {
    {PoolingFunction::None,                       "None"},
    {PoolingFunction::Max,                        "Max"},
    {PoolingFunction::Average,                    "Average"},
    {PoolingFunction::Maxout,                     "Maxout"},
    {PoolingFunction::DotProduct,                 "DotProduct"},
    {PoolingFunction::Cosine,                     "Cosine"},
    {PoolingFunction::Stochastic,                 "Stochastic"},
    {PoolingFunction::LCN,                        "LocalContrastNormalization"},
    {PoolingFunction::LRN,                        "LocalResponseNormalization"},
    {PoolingFunction::GlobalTemporal,             "GlobalTemporal"}
};

/**
 * @brief Overloaded operator<< for printing PoolingFunction values.
 *
 * @tparam T The PoolingFunction.
 * @param out The output stream.
 * @param a The PoolingFunction value.
 * @return std::ostream& The output stream.
 */
template <EnumType T>
std::ostream& operator<<(std::ostream& out, const T& a)
{
    out << sPoolingFunctionMap[a];
    return out;
}

/**
 * @brief Map to store string representations of DataSetEnums::Kind enum values.
 */
static std::map<DataSetEnums::Kind, std::string> sKindMap = {
    {DataSetEnums::Numeric, "Numeric"},
    {DataSetEnums::Image,   "Image"},
    {DataSetEnums::Audio,   "Audio"}
};

/**
 * @brief Overloaded operator<< for printing DataSetEnums::Kind values.
 *
 * @param out The output stream.
 * @param k The DataSetEnums::Kind value.
 * @return std::ostream& The output stream.
 */
std::ostream& operator<<(std::ostream& out, DataSetEnums::Kind& k)
{
    out << sKindMap[k];
    return out;
}

/**
 * @brief Map to store string representations of DataSetEnums::Attributes enum values.
 */
static std::map<DataSetEnums::Attributes, std::string> sAttributesMap = {
    {DataSetEnums::Sparse,                       "Sparse"},
    {DataSetEnums::Boolean,                      "Boolean"},
    {DataSetEnums::Compressed,                   "Compressed"},
    {DataSetEnums::Recurrent,                    "Recurrent"},
    {DataSetEnums::Mutable,                      "Mutable"},
    {DataSetEnums::Attributes::SparseIgnoreZero, "SparseIgnoreZero"},
    {DataSetEnums::Attributes::Indexed,          "Indexed"},
    {DataSetEnums::Attributes::Weighted,         "Weighted"},
};

/**
 * @brief Overloaded operator<< for printing DataSetEnums::Attributes values.
 *
 * @param out The output stream.
 * @param a The DataSetEnums::Attributes value.
 * @return std::ostream& The output stream.
 */
std::ostream& operator<<(std::ostream& out, DataSetEnums::Attributes& a)
{
    out << sAttributesMap[a];
    return out;
}

/**
 * @brief Map to store string representations of DataSetEnums::Sharding enum values.
 */
static std::map<DataSetEnums::Sharding, std::string> sShardingMap = {
    {DataSetEnums::None,  "None"},
    {DataSetEnums::Model, "Model"},
    {DataSetEnums::Data,  "Data"}
};

/**
 * @brief Overloaded operator<< for printing DataSetEnums::Sharding values.
 *
 * @param out The output stream.
 * @param s The DataSetEnums::Sharding value.
 * @return std::ostream& The output stream.
 */
std::ostream& operator<<(std::ostream& out, DataSetEnums::Sharding& s)
{
    out << sShardingMap[s];
    return out;
}

/**
 * @brief Map to store string representations of DataSetEnums::DataType enum values.
 */
static std::map<DataSetEnums::DataType, std::string> sDataTypeMap = {
    {DataSetEnums::UInt,   "UInt"},
    {DataSetEnums::Int,    "Int"},
    {DataSetEnums::LLInt,  "LLInt"},
    {DataSetEnums::ULLInt, "ULLInt"},
    {DataSetEnums::Float,  "Float"},
    {DataSetEnums::Double, "Double"},
    {DataSetEnums::RGB8,   "RGB8"},
    {DataSetEnums::RGB16,  "RGB16"},
    {DataSetEnums::UChar,  "UChar"},
    {DataSetEnums::Char,   "Char"}
};

/**
 * @brief Overloaded operator<< for printing DataSetEnums::DataType values.
 *
 * @param out The output stream.
 * @param t The DataSetEnums::DataType value.
 * @return std::ostream& The output stream.
 */
std::ostream& operator<<(std::ostream& out, DataSetEnums::DataType& t)
{
    out << sDataTypeMap[t];
    return out;
}

/**
 * @brief Function to get the MPI_Datatype corresponding to a DataSetEnums::DataType.
 *
 * @param datatype The DataSetEnums::DataType.
 * @return MPI_Datatype The corresponding MPI_Datatype.
 */
static MPI_Datatype getMPIDataType(DataSetEnums::DataType datatype)
{
    static const std::unordered_map<DataSetEnums::DataType, std::optional<MPI_Datatype>> dataTypeMap = {
        {DataSetEnums::UInt, MPI_UINT32_T},
        {DataSetEnums::Int, MPI_INT32_T},
        {DataSetEnums::ULLInt, MPI_UINT64_T},
        {DataSetEnums::LLInt, MPI_INT64_T},
        {DataSetEnums::Float, MPI_FLOAT},
        {DataSetEnums::Double, MPI_DOUBLE}
    };

    const auto it = dataTypeMap.find(datatype);
    if (it != dataTypeMap.end()) {
        if (it->second.has_value()) {
            return it->second.value();
        }
    }

    return MPI_DATATYPE_NULL;
}

/**
 * @brief Function to get the NcType corresponding to a DataSetEnums::DataType.
 *
 * @param datatype The DataSetEnums::DataType.
 * @return NcType The corresponding NcType.
 */
static NcType getNetCDFDataType(DataSetEnums::DataType datatype)
{
    static const std::unordered_map<DataSetEnums::DataType, NcType> dataTypeMap = {
        {DataSetEnums::UInt, ncUint},
        {DataSetEnums::Int, ncInt},
        {DataSetEnums::ULLInt, ncUint64},
        {DataSetEnums::LLInt, ncInt64},
        {DataSetEnums::Float, ncFloat},
        {DataSetEnums::Double, ncDouble}
    };

    const auto it = dataTypeMap.find(datatype);
    if (it != dataTypeMap.end()) {
        return it->second;
    }

    return NcType();
}

/**
 * @brief Function to check if a string has a specific suffix.
 *
 * @param str The string to check.
 * @param suffix The suffix to check.
 * @return bool True if the string has the suffix, false otherwise.
 */
inline bool has_suffix(const std::string& str, const std::string& suffix)
{
    return str.ends_with(suffix);
}

/**
 * @brief Function to broadcast a string using MPI_Bcast.
 *
 * @param s The string to broadcast.
 * @return int MPI_SUCCESS if successful.
 */
int MPI_Bcast_string(std::string& s)
{
    int length = static_cast<int>(s.size());
    MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<char> buff(length + 1);
    std::memcpy(buff.data(), s.data(), length);
    MPI_Bcast(buff.data(), length, MPI_CHAR, 0, MPI_COMM_WORLD);
    buff[length] = '\0';
    s = buff.data();
    return MPI_SUCCESS;
}

/**
 * @brief Default constructor for DataSetDimensions.
 */
DataSetDimensions::DataSetDimensions() : DataSetDimensions(1, 1, 1) {}

/**
 * @brief Constructor for DataSetDimensions.
 *
 * @param width The width dimension.
 * @param height The height dimension.
 * @param length The length dimension.
 */
DataSetDimensions::DataSetDimensions(uint32_t width, uint32_t height, uint32_t length) :
    _width(width),
    _height(height),
    _length(length),
    _dimensions(std::count_if(
        {width, height, length},
        [](uint32_t dimension) { return dimension > 1; }
    ))
{}

/**
 * @brief Template function to create a DataSet of a specific data type.
 *
 * @tparam T The data type of the DataSet.
 * @param descriptor The DataSetDescriptor for the DataSet.
 * @return std::unique_ptr<DataSetBase> A unique pointer to the created DataSet.
 */
template<typename T>
std::unique_ptr<DataSetBase> createDataSet(const DataSetDescriptor& descriptor)
{
    using DataSetEnums::Attributes;

    uint32_t attributes = descriptor._attributes;
    if (!DataSetDescriptor::isSupported(attributes))
    {
        std::stringstream msg;
        msg << "Unsupported attributes " << attributes << " for dataset " << descriptor._name;
        throw std::runtime_error(msg.str());
    }

    if (attributes & Attributes::Sparse)
    {
        return std::make_unique<DataSet<T>>(descriptor._examples, descriptor._sparseDensity, descriptor._dim, false, descriptor._name);
    }
    else
    {
        return std::make_unique<DataSet<T>>(descriptor._examples, descriptor._dim, descriptor._name);
    }
}

/**
 * @brief Map to store functions for creating DataSets of different data types.
 */
std::unordered_map<DataSetEnums::DataType, std::function<std::unique_ptr<DataSetBase>(const DataSetDescriptor&)>> datasetCreationMap = {
    {DataSetEnums::DataType::UInt, [](const DataSetDescriptor& descriptor) { return std::make_unique<DataSet<uint32_t>>(descriptor); }},
    {DataSetEnums::DataType::Int, [](const DataSetDescriptor& descriptor) { return std::make_unique<DataSet<int>>(descriptor); }},
    {DataSetEnums::DataType::Float, [](const DataSetDescriptor& descriptor) { return std::make_unique<DataSet<float>>(descriptor); }},
    {DataSetEnums::DataType::Double, [](const DataSetDescriptor& descriptor) { return std::make_unique<DataSet<double>>(descriptor); }},
    {DataSetEnums::DataType::Char, [](const DataSetDescriptor& descriptor) { return std::make_unique<DataSet<char>>(descriptor); }},
    {DataSetEnums::DataType::UChar, [](const DataSetDescriptor& descriptor) { return std::make_unique<DataSet<uint8_t>>(descriptor); }},
    {DataSetEnums::DataType::RGB8, [](const DataSetDescriptor& descriptor) { return std::make_unique<DataSet<uint8_t>>(descriptor); }}
};

/**
 * @brief Function to create a DataSet based on a DataSetDescriptor.
 *
 * @param descriptor The DataSetDescriptor for the DataSet.
 * @return std::unique_ptr<DataSetBase> A unique pointer to the created DataSet.
 */
std::unique_ptr<DataSetBase> createDataSet(const DataSetDescriptor& descriptor)
{
    const auto dataType = descriptor._dataType;
    const auto it = datasetCreationMap.find(dataType);
    if (it != datasetCreationMap.end())
    {
        return it->second(descriptor);
    }
    else
    {
        std::stringstream msg;
        msg << "Unsupported data type: " << static_cast<int>(dataType)
            << ". DataType must be one of: UInt, Int, Float, Double, Char, UChar, RGB8";
        throw std::runtime_error(msg.str());
    }
}

/**
 * @brief Default constructor for DataSetBase.
 */
DataSetBase::DataSetBase() :
    _name(""),
    _attributes(DataSetEnums::None),
    _examples(0),
    _uniqueExamples(0),
    _dimensions(0),
    _width(0),
    _height(0),
    _length(0),
    _stride(0),
    _sharding(DataSetEnums::Sharding::None),
    _minX(0),
    _maxX(0),
    _sparseDataSize(0),
    _sparseTransposedIndices(0),
    _sparseDensity(0),
    _bDenoising(false),
    _pbSparseStart(),
    _pbSparseEnd(),
    _pbSparseIndex(),
    _pbIndex(),
    _pbSparseTransposedStart(),
    _pbSparseTransposedEnd(),
    _pbSparseTransposedIndex(),
    _pbSparseTransposedData(),
    _batch(0),
    _pbDenoisingRandom(),
    _bStreaming(false),
    _bIndexed(false),
    _bDirty(true)
{
}

/**
 * @brief Constructor for DataSetBase.
 *
 * @param name The name of the DataSet.
 * @param dataType The data type of the DataSet.
 * @param examples The number of examples in the DataSet.
 * @param uniqueExamples The number of unique examples in the DataSet.
 * @param datasetDim The dimensions of the DataSet.
 */
DataSetBase::DataSetBase(const string& name, DataSetEnums::DataType dataType, uint32_t examples,
                         uint32_t uniqueExamples, const DataSetDimensions& datasetDim)
    : _name(name),
      _dataType(dataType),
      _attributes(DataSetEnums::None),
      _examples(examples),
      _uniqueExamples(uniqueExamples),
      _localExamples(examples),
      _stride(0),
      _sharding(DataSetEnums::Sharding::None),
      _minX(0),
      _maxX(0),
      _sparseDataSize(0),
      _sparseTransposedIndices(0),
      _sparseDensity(0),
      _bDenoising(false),
      _pbSparseStart(),
      _pbSparseEnd(),
      _pbSparseIndex(),
      _pbIndex(),
      _pbSparseTransposedStart(),
      _pbSparseTransposedEnd(),
      _pbSparseTransposedIndex(),
      _pbSparseTransposedData(),
      _batch(0),
      _pbDenoisingRandom(),
      _bStreaming(false),
      _bIndexed(false),
      _bDirty(true)
{
    _dimensions = datasetDim._dimensions;
    _width = datasetDim._width;
    _height = datasetDim._height;
    _length = datasetDim._length;
}

/**
 * @brief Destructor for DataSetBase.
 */
DataSetBase::~DataSetBase() = default;

/**
 * @brief Get the dimensions of the DataSet.
 *
 * @return DataSetDimensions The dimensions of the DataSet.
 */
DataSetDimensions DataSetBase::GetDimensions()
{
    return { _width, _height, _length };
}

/**
 * \brief Retrieves the memory usage of the DataSet.
 *
 * This function calculates the memory usage of the DataSet, including both CPU and GPU memory. It returns a vector of tuples, where each tuple contains the CPU and GPU memory usage of a process in the MPI world.
 *
 * \return A vector of tuples containing the CPU and GPU memory usage of each process.
 */
template<typename T>
vector<tuple<uint64_t, uint64_t>> DataSet<T>::getMemoryUsage()
{
    uint64_t cpuMemory = 0;
    uint64_t gpuMemory = 0;

    if (_attributes & DataSetEnums::Sparse)
    {
        cpuMemory += _uniqueExamples * 2 * sizeof(uint64_t);
        gpuMemory += _uniqueExamples * 2 * sizeof(uint64_t);
        cpuMemory += _vSparseIndex.size() * sizeof(uint32_t);
        gpuMemory += _vSparseIndex.size() * sizeof(uint32_t);

        if (!(_attributes & DataSetEnums::Boolean))
        {
            cpuMemory += _vSparseData.size() * sizeof(T);
            gpuMemory += _vSparseData.size() * sizeof(T);
        }
    }
    else
    {
        cpuMemory += _vData.size() * sizeof(T);
        gpuMemory += _vData.size() * sizeof(T);
    }

    if (_bIndexed)
    {
        cpuMemory += _examples * sizeof(uint32_t);
        gpuMemory += _examples * sizeof(uint32_t);
    }

    vector<tuple<uint64_t, uint64_t>> vResult(getGpu()._numprocs);
    vResult[getGpu()._id] = make_tuple(cpuMemory, gpuMemory);

    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, vResult.data(), sizeof(tuple<uint64_t, uint64_t>), MPI_BYTE, MPI_COMM_WORLD);
    
    return vResult;
}

/**
 * \brief Constructor for a DataSet object with a specified number of examples.
 *
 * This constructor initializes a DataSet object with a specified number of examples. The dimensions of the DataSet are specified by the dim parameter. The name parameter specifies the name of the DataSet.
 *
 * \param examples The number of examples in the DataSet.
 * \param dim The dimensions of the DataSet.
 * \param name The name of the DataSet.
 */
template<typename T>
DataSet<T>::DataSet(uint32_t examples, const DataSetDimensions& dim, const string& name)
    : DataSetBase(name, DataSetEnums::getDataType<T>(), examples, examples, dim),
      _sparseDensity(1.0f),
      _stride(_width * _height * _length),
      _vData(_stride * _examples),
      _pbData(make_unique<GpuBuffer<T>>(_vData.size(), false, _bStreaming))
{
}

/**
 * \brief Constructor for a DataSet object with a specified number of examples and unique examples.
 *
 * This constructor initializes a DataSet object with a specified number of examples and unique examples. The dimensions of the DataSet are specified by the dim parameter. The name parameter specifies the name of the DataSet.
 *
 * \param examples The number of examples in the DataSet.
 * \param uniqueExamples The number of unique examples in the DataSet.
 * \param dim The dimensions of the DataSet.
 * \param name The name of the DataSet.
 */
template<typename T>
DataSet<T>::DataSet(uint32_t examples, uint32_t uniqueExamples, const DataSetDimensions& dim, const string& name)
    : DataSetBase(name, DataSetEnums::getDataType<T>(), examples, uniqueExamples, dim),
      _sparseDensity(1.0f),
      _stride(_width * _height * _length),
      _attributes(DataSetEnums::Attributes::Indexed),
      _bIndexed(true),
      _vData(_stride * _uniqueExamples),
      _vIndex(_examples, 0),
      _pbData(make_unique<GpuBuffer<T>>(_vData.size(), false, _bStreaming)),
      _pbIndex(make_unique<GpuBuffer<uint32_t>>(_vIndex.size(), false, _bStreaming))
{
}

/**
 * \brief Constructor for a sparse DataSet object with a specified number of examples, sparse density, dimensions, and optional weighting.
 *
 * This constructor initializes a sparse DataSet object with a specified number of examples, sparse density, dimensions, and optional weighting. The name parameter specifies the name of the DataSet.
 *
 * \param examples The number of examples in the DataSet.
 * \param sparseDensity The density of the sparse data.
 * \param dim The dimensions of the DataSet.
 * \param isWeighted Specifies whether the DataSet is weighted.
 * \param name The name of the DataSet.
 */
template<typename T>
DataSet<T>::DataSet(uint32_t examples, Float sparseDensity, const DataSetDimensions& dim, bool isWeighted, const string& name)
    : DataSet(examples, examples, (size_t)(((double)(dim._width * dim._height * dim._length * examples)) * sparseDensity), dim, false, isWeighted, name)
{
    _attributes = DataSetEnums::Attributes::Sparse;
    if (isWeighted) {
        _attributes |= DataSetEnums::Attributes::Weighted;
    }
}
/**
 * \brief Constructor for a DataSet object with sparse data.
 *
 * This constructor initializes a DataSet object with sparse data. It sets the attributes, sparse data size, and other necessary members of the DataSet. The dimensions of the DataSet are specified by the dim parameter. The isIndexed and isWeighted parameters indicate whether the DataSet is indexed or weighted, respectively. The name parameter specifies the name of the DataSet.
 *
 * \param examples The number of examples in the DataSet.
 * \param uniqueExamples The number of unique examples in the DataSet.
 * \param sparseDataSize The size of the sparse data.
 * \param dim The dimensions of the DataSet.
 * \param isIndexed Specifies whether the DataSet is indexed.
 * \param isWeighted Specifies whether the DataSet is weighted.
 * \param name The name of the DataSet.
 */
template<typename T>
DataSet<T>::DataSet(uint32_t examples, uint32_t uniqueExamples, size_t sparseDataSize,
                   const DataSetDimensions& dim, bool isIndexed, bool isWeighted, const string& name)
    : DataSetBase(name, DataSetEnums::getDataType<T>(), examples, uniqueExamples, dim),
      _attributes(DataSetEnums::Attributes::Sparse),
      _sparseDataSize(sparseDataSize),
      _vSparseStart(_uniqueExamples, 0),
      _vSparseEnd(_uniqueExamples, 0),
      _vSparseData(_sparseDataSize),
      _vSparseIndex(_sparseDataSize, 0),
      _pbSparseStart(make_unique<GpuBuffer<uint64_t>>(_vSparseStart.size(), false, _bStreaming)),
      _pbSparseEnd(make_unique<GpuBuffer<uint64_t>>(_vSparseEnd.size(), false, _bStreaming)),
      _pbSparseData(make_unique<GpuBuffer<T>>(_vSparseData.size(), false, _bStreaming)),
      _pbSparseIndex(make_unique<GpuBuffer<uint32_t>>(_vSparseIndex.size(), false, _bStreaming))
{
    size_t sparseStride = (_sparseDataSize + _uniqueExamples - 1) / _uniqueExamples;

    _vSparseStart[0] = 0;
    _vSparseEnd[0] = sparseStride;
    for (uint32_t i = 1; i < _uniqueExamples; ++i)
    {
        _vSparseStart[i] = _vSparseEnd[i - 1];
        _vSparseEnd[i] = _vSparseStart[i] + sparseStride;
    }

    if (isIndexed)
    {
        _attributes |= DataSetEnums::Attributes::Indexed;
        _bIndexed = true;
        _vIndex.resize(_examples, 0);
        _pbIndex = make_unique<GpuBuffer<uint32_t>>(_vIndex.size(), false, _bStreaming);
    }

    if (isWeighted)
    {
        _attributes |= DataSetEnums::Attributes::Weighted;
        _vDataWeight.resize(_examples);
        _pbDataWeight = make_unique<GpuBuffer<Float>>(_vDataWeight.size(), false, _bStreaming);
    }
}

/**
 * \brief Loads dense data into the DataSet.
 *
 * This function loads dense data into the DataSet. It expects the source dense data to be provided as an array: srcData. The function checks if the DataSet is a sparse dataset and throws a runtime_error if it is. It then copies the dense data from srcData to the internal data vector of the DataSet and uploads it to the GPU buffer for efficient access during computation.
 *
 * \param srcData Pointer to the source array containing the dense data values.
 * \throw std::runtime_error If the DataSet is a sparse dataset.
 */
template<typename T>
void DataSet<T>::LoadDenseData(const void* srcData)
{
    const T* srcDataTyped = static_cast<const T*>(srcData);

    if (_attributes & DataSetEnums::Attributes::Sparse)
    {
        throw std::runtime_error("can't set dense data on a sparse DataSet");
    }
    else
    {
        std::copy(srcDataTyped, srcDataTyped + _vData.size(), _vData.data());
        _pbData->Upload(_vData.data());
    }
}

/**
 * \brief Copies dense data into the DataSet.
 *
 * This function copies dense data into the DataSet. It expects the source dense data to be provided as an array: srcData. The function checks if the DataSet is a sparse dataset and throws a runtime_error if it is. It then copies the dense data from srcData to the internal data vector of the DataSet.
 *
 * \param srcData Pointer to the source array containing the dense data values.
 * \throw std::runtime_error If the DataSet is a sparse dataset.
 */
template<typename T>
void DataSet<T>::CopyDenseData(const void* srcData)
{
    const T* srcDataTyped = static_cast<const T*>(srcData);

    if (_attributes & DataSetEnums::Attributes::Sparse)
    {
        throw std::runtime_error("can't set dense data on a sparse DataSet");
    }
    else
    {
        std::copy(srcDataTyped, srcDataTyped + _vData.size(), _vData.data());
    }
}
/**
 * \brief Loads sparse data into the DataSet.
 *
 * This function loads sparse data into the DataSet. It expects the source sparse data to be provided as arrays: srcSparseStart, srcSparseEnd, srcSparseData, and srcSparseIndex. The function checks if the DataSet is a sparse dataset and throws a runtime_error if it is not. It verifies that the srcSparseStart array starts with index 0, and then copies the sparse start, end, data, and index arrays to the corresponding vectors in the DataSet. If the required space to store the sparse data exceeds the capacity of the DataSet's internal vectors, a length_error exception is thrown.
 * 
 * The function uploads the sparse start, end, index, and data vectors to the GPU buffers for efficient access during computation.
 *
 * \param srcSparseStart Pointer to the source array containing the start indices of the sparse data for each example.
 * \param srcSparseEnd Pointer to the source array containing the end indices of the sparse data for each example.
 * \param srcSparseData Pointer to the source array containing the sparse data values.
 * \param srcSparseIndex Pointer to the source array containing the indices of the sparse data.
 * \throw std::runtime_error If the DataSet is not a sparse dataset or if the srcSparseStart array does not start with index 0.
 * \throw std::length_error If the required space to store the sparse data exceeds the capacity of the DataSet's internal vectors.
 */
template<typename T>
void DataSet<T>::LoadSparseData(const uint64_t* srcSparseStart, const uint64_t* srcSparseEnd,
                               const void* srcSparseData, const uint32_t* srcSparseIndex)
{
    const T* srcSparseDataTyped = static_cast<const T*>(srcSparseData);

    if (_attributes & DataSetEnums::Attributes::Sparse)
    {
        if (srcSparseStart[0] != 0)
        {
            throw std::runtime_error("Sparse data should be zero indexed; srcSparseStart[0] != 0");
        }

        size_t dataLength = srcSparseEnd[_uniqueExamples - 1];
        if (dataLength > _vSparseData.size() || dataLength > _vSparseIndex.size())
        {
            std::stringstream msg;
            msg << "Not enough space to store sparse data. Allocated: " << _vSparseData.size()
                << " Required: " << dataLength;
            throw std::length_error(msg.str());
        }

        std::copy_n(srcSparseStart, _uniqueExamples, _vSparseStart.begin());
        std::copy_n(srcSparseEnd, _uniqueExamples, _vSparseEnd.begin());
        std::copy_n(srcSparseDataTyped, dataLength, _vSparseData.begin());
        std::copy_n(srcSparseIndex, dataLength, _vSparseIndex.begin());

        _pbSparseStart->Upload(_vSparseStart.data());
        _pbSparseEnd->Upload(_vSparseEnd.data());
        _pbSparseIndex->Upload(_vSparseIndex.data());
        _pbSparseData->Upload(_vSparseData.data());
    }
    else
    {
        throw std::runtime_error("can't set sparse data on a non-sparse DataSet");
    }
}

/**
 * \brief Copies sparse data into the DataSet.
 *
 * This function copies sparse data into the DataSet. It expects the source sparse data to be provided as arrays: srcSparseStart, srcSparseEnd, srcSparseData, and srcSparseIndex. The function checks if the DataSet is a sparse dataset and throws a runtime_error if it is not. It verifies that the srcSparseStart array starts with index 0, and then copies the sparse start, end, data, and index arrays to the corresponding vectors in the DataSet. If the required space to store the sparse data exceeds the capacity of the DataSet's internal vectors, a length_error exception is thrown.
 * 
 * The function uploads the sparse start, end, index, and data vectors to the GPU buffers for efficient access during computation.
 *
 * \param srcSparseStart Pointer to the source array containing the start indices of the sparse data for each example.
 * \param srcSparseEnd Pointer to the source array containing the end indices of the sparse data for each example.
 * \param srcSparseData Pointer to the source array containing the sparse data values.
 * \param srcSparseIndex Pointer to the source array containing the indices of the sparse data.
 * \throw std::runtime_error If the DataSet is not a sparse dataset or if the srcSparseStart array does not start with index 0.
 * \throw std::length_error If the required space to store the sparse data exceeds the capacity of the DataSet's internal vectors.
 */
template<typename T>
void DataSet<T>::CopySparseData(const uint64_t* srcSparseStart, const uint64_t* srcSparseEnd,
                               const void* srcSparseData, const uint32_t* srcSparseIndex)
{
    const T* srcSparseDataTyped = static_cast<const T*>(srcSparseData);

    if (_attributes & DataSetEnums::Attributes::Sparse)
    {
        if (srcSparseStart[0] != 0)
        {
            throw std::runtime_error("Sparse data should be zero indexed; srcSparseStart[0] != 0");
        }

        size_t dataLength = srcSparseEnd[_uniqueExamples - 1];
        if (dataLength > _vSparseData.size() || dataLength > _vSparseIndex.size())
        {
            std::stringstream msg;
            msg << "Not enough space to store sparse data. Allocated: " << _vSparseData.size()
                << " Required: " << dataLength;
            throw std::length_error(msg.str());
        }

        std::copy_n(srcSparseStart, _uniqueExamples, _vSparseStart.begin());
        std::copy_n(srcSparseEnd, _uniqueExamples, _vSparseEnd.begin());
        std::copy_n(srcSparseDataTyped, dataLength, _vSparseData.begin());
        std::copy_n(srcSparseIndex, dataLength, _vSparseIndex.begin());

        _pbSparseStart->Upload(_vSparseStart.data());
        _pbSparseEnd->Upload(_vSparseEnd.data());
        _pbSparseIndex->Upload(_vSparseIndex.data());
        _pbSparseData->Upload(_vSparseData.data());
    }
    else
    {
        throw std::runtime_error("can't set sparse data on a non-sparse DataSet");
    }
}

/**
 * \brief Loads sparse data into the DataSet.
 *
 * This function loads sparse data into the DataSet. It expects the source sparse data to be provided as arrays: srcSparseStart, srcSparseEnd, srcSparseData, and srcSparseIndex. The function checks if the DataSet is a sparse dataset and throws a runtime_error if it is not. It verifies that the srcSparseStart array starts with index 0, and then copies the sparse start, end, data, and index arrays to the corresponding vectors in the DataSet. If the required space to store the sparse data exceeds the capacity of the DataSet's internal vectors, a length_error exception is thrown.
 * 
 * The function uploads the sparse start, end, index, and data vectors to the GPU buffers for efficient access during computation.
 *
 * \param srcSparseStart Pointer to the source array containing the start indices of the sparse data for each example.
 * \param srcSparseEnd Pointer to the source array containing the end indices of the sparse data for each example.
 * \param srcSparseData Pointer to the source array containing the sparse data values.
 * \param srcSparseIndex Pointer to the source array containing the indices of the sparse data.
 * \throw std::runtime_error If the DataSet is not a sparse dataset or if the srcSparseStart array does not start with index 0.
 * \throw std::length_error If the required space to store the sparse data exceeds the capacity of the DataSet's internal vectors.
 */
template<typename T>
void DataSet<T>::LoadSparseData(const long* srcSparseStart, const long* srcSparseEnd,
                               const void* srcSparseData, const long* srcSparseIndex)
{
    const T* srcSparseDataTyped = static_cast<const T*>(srcSparseData);

    if (_attributes & DataSetEnums::Attributes::Sparse)
    {
        if (srcSparseStart[0] != 0)
        {
            throw std::runtime_error("Sparse data should be zero indexed; srcSparseStart[0] != 0");
        }

        size_t dataLength = srcSparseEnd[_uniqueExamples - 1];
        if (dataLength > _vSparseData.size() || dataLength > _vSparseIndex.size())
        {
            std::stringstream msg;
            msg << "Not enough space to store sparse data. Allocated: " << _vSparseData.size()
                << " Required: " << dataLength;
            throw std::length_error(msg.str());
        }

        for (size_t i = 0; i < _uniqueExamples; ++i)
        {
            _vSparseStart[i] = static_cast<uint64_t>(srcSparseStart[i]);
            _vSparseEnd[i] = static_cast<uint64_t>(srcSparseEnd[i]);
        }
        for (size_t i = 0; i < dataLength; ++i)
        {
            _vSparseData[i] = srcSparseDataTyped[i];
            _vSparseIndex[i] = static_cast<uint32_t>(srcSparseIndex[i]);
        }

        _pbSparseStart->Upload(_vSparseStart.data());
        _pbSparseEnd->Upload(_vSparseEnd.data());
        _pbSparseIndex->Upload(_vSparseIndex.data());
        _pbSparseData->Upload(_vSparseData.data());
    }
    else
    {
        throw std::runtime_error("can't set sparse data on a non-sparse DataSet");
    }
}
/**
 * \brief Copies sparse data to the DataSet.
 *
 * This function copies sparse data to the DataSet. It expects the source sparse data to be provided as arrays: srcSparseStart, srcSparseEnd, srcSparseData, and srcSparseIndex. The function checks if the DataSet is a sparse dataset and throws a runtime_error if it is not. It verifies that the srcSparseStart array starts with index 0, and then copies the sparse start, end, data, and index arrays to the corresponding vectors in the DataSet. If the required space to store the sparse data exceeds the capacity of the DataSet's internal vectors, a length_error exception is thrown.
 *
 * \param srcSparseStart Pointer to the source array containing the start indices of the sparse data for each example.
 * \param srcSparseEnd Pointer to the source array containing the end indices of the sparse data for each example.
 * \param srcSparseData Pointer to the source array containing the sparse data values.
 * \param srcSparseIndex Pointer to the source array containing the indices of the sparse data.
 * \throw std::runtime_error If the DataSet is not a sparse dataset or if the srcSparseStart array does not start with index 0.
 * \throw std::length_error If the required space to store the sparse data exceeds the capacity of the DataSet's internal vectors.
 */
template<typename T>
void DataSet<T>::CopySparseData(const long* srcSparseStart, const long* srcSparseEnd,
                               const void* srcSparseData, const long* srcSparseIndex)
{
    const T* srcSparseDataTyped = static_cast<const T*>(srcSparseData);

    if (_attributes & DataSetEnums::Attributes::Sparse)
    {
        if (srcSparseStart[0] != 0)
        {
            throw std::runtime_error("Sparse data should be zero indexed; srcSparseStart[0] != 0");
        }

        size_t dataLength = srcSparseEnd[_uniqueExamples - 1];
        if (dataLength > _vSparseData.size() || dataLength > _vSparseIndex.size())
        {
            std::stringstream msg;
            msg << "Not enough space to store sparse data. Allocated: " << _vSparseData.size()
                << " Required: " << dataLength;
            throw std::length_error(msg.str());
        }

        std::copy_n(srcSparseStart, _uniqueExamples, _vSparseStart.begin());
        std::copy_n(srcSparseEnd, _uniqueExamples, _vSparseEnd.begin());
        std::copy_n(srcSparseDataTyped, dataLength, _vSparseData.begin());
        std::copy_n(srcSparseIndex, dataLength, _vSparseIndex.begin());
    }
    else
    {
        throw std::runtime_error("can't set sparse data on a non-sparse DataSet");
    }
}

/**
 * \brief Loads indexed data into the DataSet.
 *
 * This function loads indexed data into the DataSet. If the DataSet is an indexed dataset, it copies the indexed data from the source array to the DataSet's internal index vector. It then uploads the index data to the GPU buffer. If the DataSet is not an indexed dataset, it throws a runtime_error.
 *
 * \param srcIndexedData Pointer to the source array containing the indexed data.
 * \throw std::runtime_error If the DataSet is not an indexed dataset.
 */
template<typename T>
void DataSet<T>::LoadIndexedData(const uint32_t* srcIndexedData)
{
    if (_attributes & DataSetEnums::Attributes::Indexed)
    {
        std::copy(srcIndexedData, srcIndexedData + _vIndex.size(), _vIndex.begin());
        _pbIndex->Upload(_vIndex.data());
    }
    else
    {
        throw std::runtime_error("can't set indexed data on a non-indexed DataSet");
    }
}

/**
 * \brief Loads data weights into the DataSet.
 *
 * This function loads data weights into the DataSet. If the DataSet is a weighted dataset, it copies the weight data from the source array to the DataSet's internal weight vector. It then uploads the weight data to the GPU buffer. If the DataSet is not a weighted dataset, it throws a runtime_error. 
 *
 * \param srcWeightData Pointer to the source array containing the weight data.
 * \throw std::runtime_error If the DataSet is not a weighted dataset.
 */
template<typename T>
void DataSet<T>::LoadDataWeight(const Float* srcWeightData)
{
    if (_attributes & DataSetEnums::Attributes::Weighted)
    {
        std::copy(srcWeightData, srcWeightData + _vDataWeight.size(), _vDataWeight.begin());
        _pbDataWeight->Upload(_vDataWeight.data());
    }
    else
    {
        throw std::runtime_error("can't set weight data on a non-weighted DataSet");
    }
}

/**
 * \brief Retrieves the value of a data point from the DataSet.
 *
 * This function retrieves the value of a data point from the DataSet. It checks if the DataSet is a sparse dataset and throws a runtime_error if it is. It verifies the example index and the coordinates of the data point to ensure they are within the valid range. If the DataSet is indexed, it converts the example index accordingly. Finally, it returns the value of the data point.
 *
 * \param n The index of the example.
 * \param x The x-coordinate of the data point.
 * \param y The y-coordinate of the data point.
 * \param z The z-coordinate of the data point.
 * \return The value of the data point.
 * \throw std::runtime_error If the DataSet is a sparse dataset, the example index is illegal, or the data point coordinates are illegal.
 */
template<typename T>
T DataSet<T>::GetDataPoint(uint32_t n, uint32_t x, uint32_t y, uint32_t z)
{
    if (_attributes & DataSetEnums::Attributes::Sparse)
    {
        throw std::runtime_error("attempt to read non-sparse data from a sparse DataSet");
    }

    if (n >= _examples)
    {
        throw std::runtime_error("illegal example index");
    }

    if (_bIndexed)
    {
        n = _vIndex[n];
    }

    if ((x >= _width) || (y >= _height) || (z >= _length))
    {
        throw std::runtime_error("illegal data point coordinates");
    }

    return _vData[(n * _stride) + x + _width * (y + z * _height)];
}
/**
 * \brief Sets the value of a data point in the DataSet.
 *
 * This function sets the value of a data point in the DataSet. It checks if the DataSet is a sparse dataset and returns an error if it is. It verifies the example index and the coordinates of the data point to ensure they are within the valid range. If the DataSet is indexed, it converts the example index accordingly. Finally, it updates the value of the data point in the DataSet.
 *
 * \param v The new value for the data point.
 * \param n The index of the example.
 * \param x The x-coordinate of the data point.
 * \param y The y-coordinate of the data point.
 * \param z The z-coordinate of the data point.
 */
template<typename T>
void DataSet<T>::SetDataPoint(T v, uint32_t n, uint32_t x, uint32_t y, uint32_t z)
{
    if (_attributes & DataSetEnums::Attributes::Sparse)
    {
        assert(false && "attempt to read non-sparse data from a sparse dataset");
    }

    if (n >= _examples)
    {
        assert(false && "illegal example index");
    }

    if (_bIndexed)
    {
        n = _vIndex[n];
    }

    if ((x >= _width) || (y >= _height) || (z >= _length))
    {
        assert(false && "illegal datapoint coordinates");
    }

    _vData[(n * _stride) + x + _width * (y + z * _height)] = v;
}

/**
 * \brief Retrieves the number of sparse data points in an example of the DataSet.
 *
 * This function retrieves the number of sparse data points in an example of the DataSet. It checks if the DataSet is a sparse dataset and returns an error if it is not. It verifies the example index to ensure it is within the valid range. If the DataSet is indexed, it converts the example index accordingly. Finally, it returns the number of sparse data points in the example.
 *
 * \param n The index of the example.
 * \return The number of sparse data points in the example.
 */
template<typename T>
uint64_t DataSet<T>::GetSparseDataPoints(uint32_t n)
{
    if (!(_attributes & DataSetEnums::Attributes::Sparse))
    {
        assert(false && "attempt to read sparse data from a non-sparse dataset");
    }

    if (n >= _examples)
    {
        assert(false && "illegal example index");
    }

    if (_bIndexed)
    {
        n = _vIndex[n];
    }

    return _vSparseEnd[n] - _vSparseStart[n];
}

/**
 * \brief Retrieves the index of a sparse data point in the DataSet.
 *
 * This function retrieves the index of a sparse data point in the DataSet. It checks if the DataSet is a sparse dataset and returns an error if it is not. It verifies the example index and sparse index to ensure they are within the valid range. If the DataSet is indexed, it converts the example index accordingly. Finally, it returns the index of the sparse data point.
 *
 * \param n The index of the example.
 * \param i The index of the sparse data point within the example.
 * \return The index of the sparse data point.
 */
template<typename T>
uint32_t DataSet<T>::GetSparseIndex(uint32_t n, uint32_t i)
{
    if (!(_attributes & DataSetEnums::Attributes::Sparse))
    {
        assert(false && "attempt to read sparse data from a non-sparse dataset");
    }

    if (n >= _examples)
    {
        assert(false && "illegal example index");
    }

    if (_bIndexed)
    {
        n = _vIndex[n];
    }

    if (i >= _vSparseEnd[n] - _vSparseStart[n])
    {
        assert(false && "sparse index out of range");
    }

    return _vSparseIndex[_vSparseStart[n] + i];
}

/**
 * \brief Sets the index of a sparse data point in the DataSet.
 *
 * This function sets the index of a sparse data point in the DataSet. It checks if the DataSet is a sparse dataset and returns an error if it is not. It verifies the example index and sparse index to ensure they are within the valid range. If the DataSet is indexed, it converts the example index accordingly. It then updates the index of the sparse data point, marks the DataSet as dirty, and returns true.
 *
 * \param n The index of the example.
 * \param i The index of the sparse data point within the example.
 * \param v The new index for the sparse data point.
 * \return Returns true if the sparse index is successfully set, or false otherwise.
 */
template<typename T>
bool DataSet<T>::SetSparseIndex(uint32_t n, uint32_t i, uint32_t v)
{
    if (!(_attributes & DataSetEnums::Attributes::Sparse))
    {
        assert(false && "attempt to read sparse data from a non-sparse dataset");
    }

    if (n >= _examples)
    {
        assert(false && "illegal example index");
    }

    if (_bIndexed)
    {
        n = _vIndex[n];
    }

    if (i >= _vSparseEnd[n] - _vSparseStart[n])
    {
        assert(false && "sparse index out of range");
    }

    _vSparseIndex[_vSparseStart[n] + i] = v;
    _bDirty = true;
    return true;
}

/**
 * \brief Retrieves the value of a sparse data point in the DataSet.
 *
 * This function retrieves the value of a sparse data point in the DataSet. It checks if the DataSet is a sparse dataset and returns an error if it is not. It verifies the example index and sparse index to ensure they are within the valid range. If the DataSet is indexed, it converts the example index accordingly. Finally, it returns the value of the sparse data point.
 *
 * \param n The index of the example.
 * \param i The index of the sparse data point within the example.
 * \return The value of the sparse data point.
 */
template<typename T>
T DataSet<T>::GetSparseDataPoint(uint32_t n, uint32_t i)
{
    if (!(_attributes & DataSetEnums::Attributes::Sparse))
    {
        assert(false && "attempt to read sparse data from a non-sparse dataset");
    }

    if (n >= _examples)
    {
        assert(false && "illegal example index");
    }

    if (_bIndexed)
    {
        n = _vIndex[n];
    }

    if (i >= _vSparseEnd[n] - _vSparseStart[n])
    {
        assert(false && "sparse index out of range");
    }

    return _vSparseData[_vSparseStart[n] + i];
}

/**
 * \brief Sets the value of a sparse data point in the DataSet.
 *
 * This function sets the value of a sparse data point in the DataSet. It checks if the DataSet is a sparse dataset and returns an error if it is not. It verifies the example index and sparse index to ensure they are within the valid range. If the DataSet is indexed, it converts the example index accordingly. It then updates the value of the sparse data point, marks the DataSet as dirty, and returns true.
 *
 * \param n The index of the example.
 * \param i The index of the sparse data point within the example.
 * \param v The new value for the sparse data point.
 * \return Returns true if the sparse data point is successfully set, or false otherwise.
 */
template<typename T>
bool DataSet<T>::SetSparseDataPoint(uint32_t n, uint32_t i, T v)
{
    if (!(_attributes & DataSetEnums::Attributes::Sparse))
    {
        assert(false && "attempt to read sparse data from a non-sparse dataset");
    }

    if (n >= _examples)
    {
        assert(false && "illegal example index");
    }

    if (_bIndexed)
    {
        n = _vIndex[n];
    }

    if (i >= _vSparseEnd[n] - _vSparseStart[n])
    {
        assert(false && "sparse index out of range");
    }

    _vSparseData[_vSparseStart[n] + i] = v;
    _bDirty = true;
    return true;
}
/**
 * \brief Constructs a DataSet object from a NetCDF input file.
 *
 * This constructor initializes a DataSet object by reading data from a NetCDF input file. It opens the file and retrieves various attributes and dimensions required for constructing the DataSet. It checks if the current process is the root process (ID 0) and performs file operations and attribute retrieval only on the root process. The retrieved data is then broadcasted to all processes using MPI. The constructor handles different attributes and data types based on the provided NetCDF file and initializes the DataSet accordingly.
 *
 * \param fname The path to the NetCDF input file.
 * \param n The index of the DataSet in the NetCDF file.
 */
template<typename T>
DataSet<T>::DataSet(const std::string& fname, uint32_t n)
    : _pbData(),
      _pbSparseData()
{
    bool bResult = true;
    if (getGpu()._id == 0)
    {
        bool bOpened = false;
        try
        {
            NcFile nfc(fname.c_str(), NcFile::read);
            bOpened = true;

            std::string nstring = std::to_string(n);
            std::string vname = "name" + nstring;
            NcGroupAtt nameAtt = nfc.getAtt(vname);
            if (nameAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "DataSet::DataSet: No dataset name supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            nameAtt.getValues(_name);
            std::cout << "DataSet<T>::DataSet: Name of data set: " << _name << std::endl;

            vname = "dataType" + nstring;
            NcGroupAtt dataTypeAtt = nfc.getAtt(vname);
            if (dataTypeAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "DataSet::DataSet: No datatype supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            int dataType;
            dataTypeAtt.getValues(&dataType);
            _dataType = std::scoped_enum<DataSetEnums::DataType>(dataType);

            vname = "attributes" + nstring;
            NcGroupAtt attributesAtt = nfc.getAtt(vname);
            if (attributesAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "DataSet::DataSet: No attributes supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            attributesAtt.getValues(&_attributes);
            if (_attributes != 0)
            {
                int tempAtt = _attributes;
                std::cout << "DataSet<T>::DataSet: Attributes:";
                while (tempAtt != 0)
                {
                    DataSetEnums::Attributes a = std::scoped_enum<DataSetEnums::Attributes>(1 << (std::ffs(tempAtt) - 1));
                    std::cout << " " << a;
                    tempAtt ^= 1 << (std::ffs(tempAtt) - 1);
                }
                std::cout << std::endl;
            }

            vname = "examplesDim" + nstring;
            NcDim examplesDim = nfc.getDim(vname);
            if (examplesDim.isNull())
            {
                throw NC_EXCEPTION("NcException", "DataSet::DataSet: No examples count supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            _examples = examplesDim.getSize();

            if (_examples == 0)
            {
                throw NC_EXCEPTION("NcException", "DataSet::DataSet: Zero-valued Examples count in NetCDF input file " + fname, __FILE__, __LINE__);
            }

            vname = "uniqueExamplesDim" + nstring;
            NcDim uniqueExamplesDim = nfc.getDim(vname);
            if (uniqueExamplesDim.isNull())
            {
                _uniqueExamples = _examples;
            }
            else
            {
                _uniqueExamples = uniqueExamplesDim.getSize();
            }

            vname = "dimensions" + nstring;
            NcGroupAtt dimensionsAtt = nfc.getAtt(vname);
            if (dimensionsAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "DataSet::DataSet: No dimension count supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            dimensionsAtt.getValues(&_dimensions);

            if ((_dimensions < 1) || (_dimensions > 3))
            {
                throw NC_EXCEPTION("NcException", "DataSet::DataSet: Invalid dimension count (" + std::to_string(_dimensions) + ") supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }

            vname = "width" + nstring;
            NcGroupAtt widthAtt = nfc.getAtt(vname);
            if (widthAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "DataSet::DataSet: No datapoint width supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            widthAtt.getValues(&_width);

            if (_dimensions > 1)
            {
                vname = "height" + nstring;
                NcGroupAtt heightAtt = nfc.getAtt(vname);
                if (heightAtt.isNull())
                {
                    throw NC_EXCEPTION("NcException", "DataSet::DataSet: No datapoint height supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                heightAtt.getValues(&_height);
            }
            else
            {
                _height = 1;
            }

            if (_dimensions > 2)
            {
                vname = "length" + nstring;
                NcGroupAtt lengthAtt = nfc.getAtt(vname);
                if (lengthAtt.isNull())
                {
                    throw NC_EXCEPTION("NcException", "DataSet::DataSet: No datapoint length supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                lengthAtt.getValues(&_length);
            }
            else
            {
                _length = 1;
            }
            std::cout << "DataSet<T>::DataSet: " << _dimensions << "-dimensional data comprised of (" << _width << ", " << _height << ", " << _length << ") datapoints." << std::endl;

            if ((_width == 0) || (_height == 0) || (_length == 0))
            {
                throw NC_EXCEPTION("NcException", "DataSet::DataSet: Invalid dataset dimensions in NetCDF input file " + fname, __FILE__, __LINE__);
            }

            if (_attributes & DataSetEnums::Sparse)
            {
                _vSparseStart.resize(_uniqueExamples);
                _vSparseEnd.resize(_uniqueExamples);
                vname = "sparseDataDim" + nstring;
                NcDim sparseDataDim = nfc.getDim(vname);
                if (sparseDataDim.isNull())
                {
                    throw NC_EXCEPTION("NcException", "DataSet::DataSet: No sparse data dimensions supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                _sparseDataSize = sparseDataDim.getSize();

                if (_sparseDataSize == 0)
                {
                    throw NC_EXCEPTION("NcException", "DataSet::DataSet: Sparse data set with no actual data in NetCDF input file " + fname, __FILE__, __LINE__);
                }

                _vSparseIndex.resize(_sparseDataSize);
                std::cout << "DataSet<T>::DataSet: " << _sparseDataSize << " total datapoints." << std::endl;
                vname = "sparseStart" + nstring;
                NcVar sparseStartVar = nfc.getVar(vname);
                if (sparseStartVar.isNull())
                {
                    throw NC_EXCEPTION("NcException", "DataSet::DataSet: No sparse offset start supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                vname = "sparseEnd" + nstring;
                NcVar sparseEndVar = nfc.getVar(vname);
                if (sparseEndVar.isNull())
                {
                    throw NC_EXCEPTION("NcException", "DataSet::DataSet: No sparse data end supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                vname = "sparseIndex" + nstring;
                NcVar sparseIndexVar = nfc.getVar(vname);
                if (sparseIndexVar.isNull())
                {
                    throw NC_EXCEPTION("NcException", "DataSet::DataSet: No sparse data indices supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }

                NcType vStartType = sparseStartVar.getType();
                if (vStartType == ncUint)
                {
                    std::vector<uint32_t> vTempSparseStart(_uniqueExamples);
                    sparseStartVar.getVar((uint32_t*)vTempSparseStart.data());
                    std::copy(vTempSparseStart.begin(), vTempSparseStart.end(), _vSparseStart.begin());
                }
                else
                {
                    sparseStartVar.getVar((uint64_t*)_vSparseStart.data());
                }

                NcType vEndType = sparseEndVar.getType();
                if (vEndType == ncUint)
                {
                    std::vector<uint32_t> vTempSparseEnd(_uniqueExamples);
                    sparseEndVar.getVar((uint32_t*)vTempSparseEnd.data());
                    std::copy(vTempSparseEnd.begin(), vTempSparseEnd.end(), _vSparseEnd.begin());
                }
                else
                {
                    sparseEndVar.getVar((uint64_t*)_vSparseEnd.data());
                }
                sparseIndexVar.getVar((uint32_t*)_vSparseIndex.data());

                if (!(_attributes & DataSetEnums::Boolean))
                {
                    vname = "sparseData" + nstring;
                    NcVar sparseDataVar = nfc.getVar(vname);
                    if (sparseDataVar.isNull())
                    {
                        throw NC_EXCEPTION("NcException", "DataSet::DataSet: No sparse data located in NetCDF input file " + fname, __FILE__, __LINE__);
                    }
                    _vSparseData.resize(sparseDataDim.getSize());
                    sparseDataVar.getVar(_vSparseData.data());
                }
            }
            else
            {
                _stride = _width * _height * _length;
                vname = "dataDim" + nstring;
                NcDim dataDim = nfc.getDim(vname);
                if (dataDim.isNull())
                {
                    throw NC_EXCEPTION("NcException", "DataSet::DataSet: No data dimensions located in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                vname = "data" + nstring;
                NcVar dataVar = nfc.getVar(vname);

                if (_attributes & DataSetEnums::Boolean)
                {
                    uint64_t size = static_cast<uint64_t>(_width) * static_cast<uint64_t>(_height) * static_cast<uint64_t>(_length);
                    _vData.resize(dataDim.getSize() * size);
                    std::fill_n(_vData.begin(), _vData.size(), T(0));
                    std::vector<T> vData(dataDim.getSize());
                    dataVar.getVar(vData.data());
                    for (int i = 0; i < dataDim.getSize(); i++)
                    {
                        _vData[i * size + vData[i]] = T(1.0);
                    }
                }
                else
                {
                    _vData.resize(dataDim.getSize());
                    dataVar.getVar(_vData.data());
                }
            }

            if (_attributes & DataSetEnums::Weighted)
            {
                vname = "dataWeight" + nstring;
                NcVar DataWeightVar = nfc.getVar(vname);
                if (DataWeightVar.isNull())
                {
                    throw NC_EXCEPTION("NcException", "DataSet::DataSet: No data weights located in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                _vDataWeight.resize(_examples);
                DataWeightVar.getVar(_vDataWeight.data());
            }

            if (_attributes & DataSetEnums::Indexed)
            {
                vname = "index" + nstring;
                NcVar indexVar = nfc.getVar(vname);
                if (indexVar.isNull())
                {
                    throw NC_EXCEPTION("NcException", "DataSet::DataSet: No indexed data located in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                _vIndex.resize(_examples);
                indexVar.getVar(_vIndex.data());
            }

            std::cout << "DataSet<T>::DataSet: " << _examples << " examples." << std::endl;
            std::cout << "DataSet<T>::DataSet: " << _uniqueExamples << " unique examples." << std::endl;
        }
        catch (NcException& e)
        {
            if (!bOpened)
            {
                std::cout << "Exception: DataSet::DataSet: Error opening NetCDF input file " << fname << std::endl;
            }
            else
            {
                std::cout << "Exception: " << e.what() << std::endl;
            }
            bResult = false;
        }
    }

    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!bResult)
    {
        getGpu().Shutdown();
        exit(-1);
    }

    MPI_Bcast_string(_name);
    MPI_Bcast(&_dataType, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_attributes, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_examples, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_uniqueExamples, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_dimensions, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_width, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_height, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_length, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_sparseDataSize, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    if (getGpu()._id != 0)
    {
        _vData.resize(0);
        _vSparseStart.resize(_uniqueExamples, 0);
        _vSparseEnd.resize(_uniqueExamples, 0);
        _vSparseIndex.resize(0);
        _vSparseData.resize(0);
    }

    if (_attributes & DataSetEnums::Indexed)
    {
        _vIndex.resize(_examples);
        MPI_Bcast(_vIndex.data(), _examples, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    }

    if (_attributes & DataSetEnums::Weighted)
    {
        _vDataWeight.resize(_examples);
        MPI_Bcast(_vDataWeight.data(), _examples, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    if (_attributes & DataSetEnums::Sparse)
    {
        CalculateSparseDatapointCounts();
    }
}
/**
 * \brief Renames the DataSet.
 *
 * This function renames the DataSet with the specified name.
 *
 * \param name The new name for the DataSet.
 * \return Returns true if the DataSet is successfully renamed, or false otherwise.
 */
template<typename T>
bool DataSet<T>::Rename(const std::string& name)
{
    _name = name;
    return true;
}
/**
 * \brief Calculates the sparse datapoint counts for the DataSet.
 *
 * This function calculates the sparse datapoint counts for the DataSet. It checks if the DataSet is a sparse dataset and returns an error if it is not. The function initializes necessary containers for storing the sparse datapoint counts and clears them. It then calculates the counts based on the sparse indices and values, considering the unique examples and their counts. Finally, it computes the sparse density of the dataset.
 *
 * \return Returns true if the sparse datapoint counts are successfully calculated, or false otherwise.
 */
template<typename T>
bool DataSet<T>::CalculateSparseDatapointCounts()
{
    if (!(_attributes & DataSetEnums::Sparse))
    {
        if (getGpu()._id == 0)
        {
            std::printf("DataSet::CalculateSparseDatapointCounts: Attempt to calculate sparse datapoint counts on non-sparse dataset %s.\n", _name.c_str());
        }
        return false;
    }

    const uint64_t N = _width * _height * _length;
    _vSparseDatapointCount.resize(N);
    _vSparseMaxDatapointCount.resize(N);
    _vSparseMultiDatapointCount.resize(N);
    std::fill(_vSparseDatapointCount.begin(), _vSparseDatapointCount.end(), 0);
    std::fill(_vSparseMaxDatapointCount.begin(), _vSparseMaxDatapointCount.end(), 0);
    std::fill(_vSparseMultiDatapointCount.begin(), _vSparseMultiDatapointCount.end(), 0);

    std::vector<uint32_t> vCount(N, 0);
    std::vector<uint32_t> vExampleCount(_uniqueExamples, 0);
    if (_attributes & DataSetEnums::Indexed)
    {
        for (size_t i = 0; i < _examples; i++)
        {
            vExampleCount[_vIndex[i]]++;
        }
    }
    else
    {
        std::fill(vExampleCount.begin(), vExampleCount.end(), 1);
    }

    for (size_t i = 0; i < _uniqueExamples; i++)
    {
        const uint64_t count = _vSparseEnd[i] - _vSparseStart[i];
        for (size_t j = _vSparseStart[i]; j < _vSparseEnd[i]; j++)
        {
            const uint32_t x = _vSparseIndex[j];

            if (x >= N)
            {
                std::cout << "DataSet::CalculateSparseDatapointCounts: vCount address = " << x << " >= vCount size = " << N << std::endl;
                std::terminate();
            }

            vCount[x]++;
        }

        for (size_t j = _vSparseStart[i]; j < _vSparseEnd[i]; j++)
        {
            const uint32_t x = _vSparseIndex[j];

            if (vCount[x] > 0)
            {
                _vSparseMaxDatapointCount[x] = std::max(_vSparseMaxDatapointCount[x], vCount[x]);
                if (vCount[x] > 1)
                    _vSparseMultiDatapointCount[x] += vExampleCount[i];
                _vSparseDatapointCount[x] += vExampleCount[i] * vCount[x];
                vCount[x] = 0;
            }
        }
    }

    size_t sz = 0;
    const size_t batch = 2048;
    size_t active = 0;
    for (size_t i = 0; i < N; i++)
    {
        size_t size1 = _vSparseDatapointCount[i];
        size1 = std::min(batch, size1);
        active += (_vSparseDatapointCount[i] > 0);
        if (_vSparseMaxDatapointCount[i] > 1)
        {
            size_t size2 = std::min(_vSparseMaxDatapointCount[i] * batch, batch + (_vSparseMaxDatapointCount[i] - 1) * _vSparseMultiDatapointCount[i]);
            size1 = std::max(size1, size2);
        }
        sz += size1;
    }

    _sparseDensity = static_cast<double>(_sparseDataSize) / static_cast<double>(_uniqueExamples * N);
    return true;
}
/**
 * \brief Generates a sparse transposed matrix for a given batch and layer.
 *
 * This function generates a sparse transposed matrix for the specified batch and layer. The function calculates the sparse datapoint counts if the DataSet is dirty (modified), and then initializes the necessary buffers for the sparse transposed matrix. The batch size, offset, and sizes for each datapoint are computed based on the dataset's properties and the provided layer's dimensions. Finally, the function uploads the sparse transposed start indices to the GPU buffer and allocates memory for the sparse transposed weight gradient index and value matrices if necessary.
 *
 * \param batch The batch size for the sparse transposed matrix.
 * \param pLayer A pointer to the Layer object for which the sparse transposed matrix is generated.
 * \return Returns true if the sparse transposed matrix is successfully generated, or false if an error occurs.
 */
template<typename T>
bool DataSet<T>::GenerateSparseTransposedMatrix(uint32_t batch, Layer* pLayer)
{
    if (_bDirty)
    {
        CalculateSparseDatapointCounts();
        _bDirty = false;
    }

    const uint64_t NData = _width * _height * _length;
    const auto [Nx, Ny, Nz, Nw] = pLayer->GetLocalDimensions();
    const uint64_t NLayer = Nx * Ny * Nz * Nw;
    const uint64_t N = std::max(NData, NLayer);
    _vSparseTransposedStart.resize(N);
    _pbSparseTransposedStart = std::make_unique<GpuBuffer<uint32_t>>(N);
    _pbSparseTransposedEnd = std::make_unique<GpuBuffer<uint32_t>>(N);

    _batch = batch;
    uint32_t offset = 0;
    for (size_t i = 0; i < _vSparseDatapointCount.size(); i++)
    {
        _vSparseTransposedStart[i] = offset;
        size_t size1 = _vSparseDatapointCount[i];
        size1 = std::min(static_cast<size_t>(batch), size1);
        if (_vSparseMaxDatapointCount[i] > 1)
        {
            size_t size2 = std::min(_vSparseMaxDatapointCount[i] * batch, batch + (_vSparseMaxDatapointCount[i] - 1) * _vSparseMultiDatapointCount[i]);
            size1 = std::max(size1, size2);
        }
        offset += size1;
        offset = ((offset + 31) >> 5) << 5;
    }
    _pbSparseTransposedStart->Upload(_vSparseTransposedStart.data());

    if (offset > _sparseTransposedIndices)
    {
        _sparseTransposedIndices = offset;
        std::printf("DataSet::GenerateSparseTransposedMatrix: Allocating %lu bytes for sparse transposed weight gradient index matrix %s.\n", _sparseTransposedIndices * sizeof(uint32_t), _name.c_str());
        _pbSparseTransposedIndex = std::make_unique<GpuBuffer<uint32_t>>(_sparseTransposedIndices);
        if (!(_attributes & DataSetEnums::Boolean) || (_attributes & DataSetEnums::Weighted))
        {
            std::printf("DataSet::GenerateSparseTransposedMatrix: Allocating %lu bytes for sparse transposed weight gradient value matrix %s.\n", _sparseTransposedIndices * sizeof(Float), _name.c_str());
            _pbSparseTransposedData = std::make_unique<GpuBuffer<Float>>(_sparseTransposedIndices);
        }
    }
    return true;
}

/**
 * \brief Sets the denoising flag for the DataSet.
 *
 * This function sets the denoising flag for the DataSet. Denoising can only be set for sparse datasets. If the DataSet is not sparse, an error message is printed, and the function returns false. If the flag is set to false and denoising was previously enabled, the denoising random buffer is reset. If the flag is set to true and denoising was previously disabled, memory is allocated for the denoising random buffer.
 *
 * \param flag The value indicating whether denoising is enabled (true) or disabled (false).
 * \return Returns true if the denoising flag is successfully set, or false otherwise.
 */
template<typename T>
bool DataSet<T>::SetDenoising(bool flag)
{
    if (!(_attributes & DataSetEnums::Sparse))
    {
        if (getGpu()._id == 0)
        {
            std::printf("DataSet::SetDenoising: Attempt to set denoising on non-sparse data set.\n");
        }
        return false;
    }
    
    if (!flag && _bDenoising)
    {
        _pbDenoisingRandom.reset();
    }
    else if (flag && !_bDenoising)
    {
        _pbDenoisingRandom = std::make_unique<GpuBuffer<Float>>(_vSparseIndex.size());
    }
    
    _bDenoising = flag;
    return true;
}

/**
 * \brief Sets the streaming flag for the DataSet.
 *
 * This function sets the streaming flag for the DataSet. Streaming datasets are supported only if unified memory is enabled on the GPU. If streaming is not supported, an error message is printed, and the function returns false.
 *
 * \param flag The value indicating whether streaming is enabled (true) or disabled (false).
 * \return Returns true if the streaming flag is successfully set, or false otherwise.
 */
template<typename T>
bool DataSet<T>::SetStreaming(bool flag)
{
    if (!getGpu()._bUnifiedMemory)
    {
        std::printf("DataSet::SetStreaming: Streaming datasets not supported on GPU %d\n", getGpu()._id);
        return false;
    }

    if (flag != _bStreaming)
    {
        _bStreaming = flag;
        _bDirty = true;
    }

    return true;
}

/**
 * \brief Gets the streaming flag of the DataSet.
 *
 * This function returns the value of the streaming flag, indicating whether streaming is enabled or disabled for the DataSet.
 *
 * \return Returns true if streaming is enabled, or false if streaming is disabled.
 */
template<typename T>
bool DataSet<T>::GetStreaming() const
{
    return _bStreaming;
}

/**
 * \brief Generates denoising data for the DataSet.
 *
 * This function generates denoising randoms for the DataSet. Denoising randoms can only be generated for sparse data sets. If the DataSet is not sparse, an error message is printed, and the function returns false.
 *
 * \return Returns true if the denoising data is successfully generated, or false if an error occurs.
 */
template<typename T>
bool DataSet<T>::GenerateDenoisingData()
{
    if (!(_attributes & DataSetEnums::Sparse))
    {
        if (getGpu()._id == 0)
        {
            std::printf("DataSet::GenerateDenoisingData: Attempt to generate denoising randoms on non-sparse data set.\n");
        }
        return false;
    }
    curandGenerateUniform(getGpu()._RNG, _pbDenoisingRandom->_pDevData, _vSparseIndex.size());
    return true;
}
/**
 * @brief Unshards the dataset and restores it to the original state.
 *
 * @tparam T The data type of the dataset.
 * @return True if the unsharding is successful, false otherwise.
 */
template<typename T>
bool DataSet<T>::UnShard()
{
    if (_sharding == DataSetEnums::Model)
    {
        if (_attributes & DataSetEnums::Sparse)
        {
            _pbSparseStart->Download(_vSparseStart.data());
            _pbSparseEnd->Download(_vSparseEnd.data());
            _pbSparseIndex->Download(_vSparseIndex.data());
            _pbSparseStart.reset();
            _pbSparseEnd.reset();
            _pbSparseIndex.reset();

            if (!(_attributes & DataSetEnums::Boolean))
            {
                _pbSparseData->Download(_vSparseData.data());
                _pbSparseData.reset();
            }

            const int32_t xmin = (_width * getGpu()._id) / getGpu()._numprocs;
            const int32_t xmax = (_width * (getGpu()._id + 1)) / getGpu()._numprocs;

            for (auto& index : _vSparseIndex)
                index -= xmin;

            std::vector<uint32_t> vSparseCount(_uniqueExamples);
            for (uint32_t i = 0; i < _uniqueExamples; i++)
            {
                vSparseCount[i] = _vSparseEnd[i] - _vSparseStart[i];
            }

            uint64_t datapoints = _vSparseIndex.size();
            MPI_Reduce((getGpu()._id == 0) ? MPI_IN_PLACE : &datapoints, &datapoints, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce((getGpu()._id == 0) ? MPI_IN_PLACE : vSparseCount.data(), vSparseCount.data(), _uniqueExamples, MPI_UINT32_T, MPI_SUM, 0, MPI_COMM_WORLD);

            if (getGpu()._id == 0)
            {
                std::vector<uint64_t> vTempSparseStart(_uniqueExamples);
                std::vector<uint64_t> vTempSparseEnd(_uniqueExamples);
                std::vector<uint32_t> vTempSparseIndex(datapoints);
                std::vector<T> vTempSparseData;

                if (!(_attributes & DataSetEnums::Boolean))
                    vTempSparseData.resize(datapoints);

                vTempSparseStart[0] = 0;
                uint64_t start = 0;

                for (int i = 0; i < _uniqueExamples; i++)
                {
                    vTempSparseStart[i] = start;
                    vTempSparseEnd[i] = start;

                    for (uint64_t j = _vSparseStart[i]; j < _vSparseEnd[i]; j++)
                    {
                        vTempSparseIndex[vTempSparseEnd[i]] = _vSparseIndex[vTempSparseEnd[i]];

                        if (!(_attributes & DataSetEnums::Boolean))
                        {
                            vTempSparseData[vTempSparseEnd[i]] = _vSparseData[vTempSparseEnd[i]];
                        }

                        vTempSparseEnd[i]++;
                    }

                    start += vSparseCount[i];
                }

                for (uint32_t i = 1; i < getGpu()._numprocs; i++)
                {
                    uint64_t size;
                    MPI_Status status;
                    MPI_Recv(vSparseCount.data(), _uniqueExamples, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
                    MPI_Recv(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
                    std::vector<uint32_t> vPeerSparseIndex(size);
                    MPI_Recv(vPeerSparseIndex.data(), size, MPI_UINT32_T, i, 0, MPI_COMM_WORLD, &status);

                    std::vector<T> vPeerSparseData;
                    if (!(_attributes & DataSetEnums::Boolean))
                    {
                        vPeerSparseData.resize(size);
                        MPI_Recv(vPeerSparseData.data(), size, getMPIDataType(_dataType), i, 0, MPI_COMM_WORLD, &status);
                    }

                    for (uint32_t i = 0; i < _uniqueExamples; i++)
                    {
                        uint64_t start = 0;
                        for (int j = 0; j < vSparseCount[i]; j++)
                        {
                            vTempSparseIndex[vTempSparseEnd[i]] = vPeerSparseIndex[start];
                            if (!(_attributes & DataSetEnums::Boolean))
                            {
                                vTempSparseData[vTempSparseEnd[i]] = vPeerSparseData[start];
                            }
                            vTempSparseEnd[i]++;
                            start++;
                        }
                    }
                }

                _vSparseStart = std::move(vTempSparseStart);
                _vSparseEnd = std::move(vTempSparseEnd);
                _vSparseIndex = std::move(vTempSparseIndex);
                if (!(_attributes & DataSetEnums::Boolean))
                    _vSparseData = std::move(vTempSparseData);

                _pbSparseStart.reset(new GpuBuffer<uint64_t>(_uniqueExamples, false, _bStreaming));
                _pbSparseEnd.reset(new GpuBuffer<uint64_t>(_uniqueExamples, false, _bStreaming));
                _pbSparseIndex.reset(new GpuBuffer<uint32_t>(_vSparseIndex.size(), false, _bStreaming));
                _pbSparseStart->Upload(_vSparseStart.data());
                _pbSparseEnd->Upload(_vSparseEnd.data());
                _pbSparseIndex->Upload(_vSparseIndex.data());

                if (!(_attributes & DataSetEnums::Boolean))
                {
                    _pbSparseData.reset(new GpuBuffer<T>(_vSparseData.size(), false, _bStreaming));
                    _pbSparseData->Upload(_vSparseData.data());
                }
            }
            else
            {
                uint64_t size = _vSparseIndex.size();
                MPI_Send(vSparseCount.data(), _uniqueExamples, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD);
                MPI_Send(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD);
                MPI_Send(_vSparseIndex.data(), size, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD);

                if (!(_attributes & DataSetEnums::Boolean))
                {
                    MPI_Send(_vSparseData.data(), size, getMPIDataType(_dataType), 0, 0, MPI_COMM_WORLD);
                }
            }
        }
        else
        {
            _pbData->Download(_vData.data());
            _pbData.reset();

            if (getGpu()._id == 0)
            {
                std::vector<T> vTempData(_vData);
                _vData.resize(_uniqueExamples * _width);

                const uint32_t xmax = _width / getGpu()._numprocs;
                for (uint64_t i = 0; i < _uniqueExamples; i++)
                {
                    for (uint64_t j = 0; j < xmax; j++)
                    {
                        _vData[i * _width + j] = vTempData[i * xmax + j];
                    }
                }

                for (int i = 1; i < getGpu()._numprocs; i++)
                {
                    const int xmin = (i * _width) / getGpu()._numprocs;
                    const uint32_t xmax = ((i + 1) * _width) / getGpu()._numprocs;
                    const int slice = xmax - xmin;
                    const int size = _uniqueExamples * slice;
                    vTempData.resize(size);
                    MPI_Status status;
                    MPI_Recv(vTempData.data(), size, getMPIDataType(_dataType), i, 0, MPI_COMM_WORLD, &status);
                    for (int j = 0; j < _uniqueExamples; j++)
                    {
                        for (int k = 0; k < slice; k++)
                        {
                            _vData[j * _width + xmin + k] = vTempData[j * slice + k];
                        }
                    }
                }

                _pbData.reset(new GpuBuffer<T>(_vData.size(), false, _bStreaming));
                _pbData->Upload(_vData.data());
            }
            else
            {
                MPI_Send(_vData.data(), _vData.size(), getMPIDataType(_dataType), 0, 0, MPI_COMM_WORLD);
            }
        }
    }
    else if (_sharding == DataSetEnums::Data)
    {
        // Code for Data sharding
    }

    _sharding = DataSetEnums::Sharding::None;

    if (_attributes & DataSetEnums::Indexed)
    {
        _pbIndex.reset(new GpuBuffer<uint32_t>(_vIndex.size(), false, _bStreaming));
        _pbIndex->Upload(_vIndex.data());
    }

    if (_attributes & DataSetEnums::Weighted)
    {
        _pbDataWeight.reset(new GpuBuffer<Float>(_vDataWeight.size(), false, _bStreaming));
        _pbDataWeight->Upload(_vDataWeight.data());
    }

    return true;
}
/**
 * @brief Shards the dataset based on the specified sharding type.
 *
 * @tparam T The data type of the dataset.
 * @param sharding The sharding type.
 * @return True if the sharding is successful, false otherwise.
 */
template<typename T>
bool Shard(DataSetEnums::Sharding sharding,) {
    if (sharding == _sharding)
        return true;

    UnShard();

    if (sharding == DataSetEnums::Model) {
        _sharding = DataSetEnums::Model;
        _minX = static_cast<size_t>(_width) * static_cast<size_t>(getGpu()._id) / static_cast<size_t>(getGpu()._numprocs);
        _maxX = static_cast<size_t>(_width) * static_cast<size_t>(getGpu()._id + 1) / static_cast<size_t>(getGpu()._numprocs);

        if (_attributes & DataSetEnums::Sparse) {
            if (getGpu()._id == 0) {
                std::cout << "DataSet<T>::Shard: Model Sharding sparse dataset " << _name << " across all GPUs.\n";

                for (size_t i = 1; i < getGpu()._numprocs; i++) {
                    uint32_t xmin = static_cast<size_t>(_width) * i / static_cast<size_t>(getGpu()._numprocs);
                    uint32_t xmax = static_cast<size_t>(_width) * (i + 1) / static_cast<size_t>(getGpu()._numprocs);

                    std::vector<uint64_t> vLocalSparseStart(_uniqueExamples);
                    std::vector<uint64_t> vLocalSparseEnd(_uniqueExamples);
                    std::vector<uint32_t> vLocalSparseIndex;
                    std::vector<T> vLocalSparseData;

                    for (int j = 0; j < _uniqueExamples; j++) {
                        vLocalSparseStart[j] = vLocalSparseIndex.size();

                        for (uint64_t k = _vSparseStart[j]; k < _vSparseEnd[j]; k++) {
                            if ((_vSparseIndex[k] >= xmin) && (_vSparseIndex[k] < xmax)) {
                                vLocalSparseIndex.push_back(_vSparseIndex[k] - xmin);

                                if (!(_attributes & DataSetEnums::Boolean)) {
                                    vLocalSparseData.push_back(_vSparseData[k]);
                                }
                            }
                        }

                        vLocalSparseEnd[j] = vLocalSparseIndex.size();
                    }

                    uint64_t size = vLocalSparseIndex.size();
                    MPI_Send(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                    MPI_Send(vLocalSparseStart.data(), _uniqueExamples, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                    MPI_Send(vLocalSparseEnd.data(), _uniqueExamples, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                    MPI_Send(vLocalSparseIndex.data(), size, MPI_UINT32_T, i, 0, MPI_COMM_WORLD);

                    if (!(_attributes & DataSetEnums::Boolean)) {
                        MPI_Datatype mpiType = getMPIDataType(_dataType);
                        MPI_Send(vLocalSparseData.data(), size, mpiType, i, 0, MPI_COMM_WORLD);
                    }
                }

                std::vector<uint64_t> vTempSparseStart = _vSparseStart;
                std::vector<uint64_t> vTempSparseEnd = _vSparseEnd;
                std::vector<uint32_t> vTempSparseIndex = _vSparseIndex;
                std::vector<T> vTempSparseData = _vSparseData;

                _vSparseIndex.resize(0);
                _vSparseData.resize(0);
                _vSparseStart.resize(_localExamples);
                _vSparseEnd.resize(_localExamples);

                for (uint32_t j = 0; j < _uniqueExamples; j++) {
                    _vSparseStart[j] = _vSparseIndex.size();

                    for (uint64_t k = vTempSparseStart[j]; k < vTempSparseEnd[j]; k++) {
                        if ((vTempSparseIndex[k] >= _minX) && (vTempSparseIndex[k] < _maxX)) {
                            _vSparseIndex.push_back(vTempSparseIndex[k]);

                            if (!(_attributes & DataSetEnums::Boolean)) {
                                _vSparseData.push_back(vTempSparseData[k]);
                            }
                        }
                    }

                    _vSparseEnd[j] = _vSparseIndex.size();
                }
            }
            else {
                uint64_t size;
                MPI_Status status;
                MPI_Recv(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &status);
                _vSparseStart.resize(_uniqueExamples);
                _vSparseEnd.resize(_uniqueExamples);
                _vSparseIndex.resize(size);
                MPI_Recv(_vSparseStart.data(), _uniqueExamples, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(_vSparseEnd.data(), _uniqueExamples, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(_vSparseIndex.data(), size, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD, &status);

                if (!(_attributes & DataSetEnums::Boolean)) {
                    MPI_Datatype mpiType = getMPIDataType(_dataType);
                    _vSparseData.resize(size);
                    MPI_Recv(_vSparseData.data(), size, mpiType, 0, 0, MPI_COMM_WORLD, &status);
                }
            }

            _pbSparseStart.reset(new GpuBuffer<uint64_t>(_uniqueExamples, false, _bStreaming));
            _pbSparseEnd.reset(new GpuBuffer<uint64_t>(_uniqueExamples, false, _bStreaming));
            _pbSparseIndex.reset(new GpuBuffer<uint32_t>(static_cast<uint64_t>(_vSparseIndex.size()), false, _bStreaming));
            _pbSparseStart->Upload(_vSparseStart.data());
            _pbSparseEnd->Upload(_vSparseEnd.data());
            _pbSparseIndex->Upload(_vSparseIndex.data());

            if (!(_attributes & DataSetEnums::Boolean)) {
                _pbSparseData.reset(new GpuBuffer<T>(static_cast<uint64_t>(_vSparseData.size()), false, _bStreaming));
                _pbSparseData->Upload(_vSparseData.data());
            }
        }
        else {
            if (getGpu()._id == 0) {
                std::cout << "DataSet<T>::Shard: Model Sharding dataset " << _name << " across all GPUs.\n";

                for (size_t i = 1; i < getGpu()._numprocs; i++) {
                    uint32_t xmin = static_cast<size_t>(_width) * i / static_cast<size_t>(getGpu()._numprocs);
                    uint32_t xmax = static_cast<size_t>(_width) * (i + 1) / static_cast<size_t>(getGpu()._numprocs);
                    uint32_t slice = xmax - xmin;

                    std::vector<T> vLocalData(_uniqueExamples * slice);

                    for (size_t j = 0; j < _uniqueExamples; j++) {
                        for (size_t k = 0; k < slice; k++) {
                            vLocalData[j * slice + k] = _vData[j * _width + xmin + k];
                        }
                    }

                    size_t size = vLocalData.size();
                    MPI_Send(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                    MPI_Datatype mpiType = getMPIDataType(_dataType);
                    MPI_Send(vLocalData.data(), _uniqueExamples * slice, mpiType, i, 0, MPI_COMM_WORLD);
                }

                std::vector<T> vTempData = _vData;
                uint64_t xmax = _width / getGpu()._numprocs;
                _vData.resize(_uniqueExamples * xmax);

                for (uint64_t j = 0; j < _uniqueExamples; j++) {
                    for (uint64_t k = 0; k < xmax; k++) {
                        _vData[j * xmax + k] = vTempData[j * _width + k];
                    }
                }
            }
            else {
                uint64_t size;
                MPI_Status status;
                MPI_Recv(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &status);
                _vData.resize(size);
                MPI_Datatype mpiType = getMPIDataType(_dataType);
                MPI_Recv(_vData.data(), size, mpiType, 0, 0, MPI_COMM_WORLD, &status);
            }

            _pbData.reset(new GpuBuffer<T>(static_cast<uint64_t>(_vData.size()), false, _bStreaming));
            _pbData->Upload(_vData.data());
        }
    }
    else if (sharding == DataSetEnums::Data) {
        _sharding = DataSetEnums::Data;
        size_t segment = _uniqueExamples / getGpu()._numprocs;
        size_t remainder = _uniqueExamples % getGpu()._numprocs;
        _localExamples = segment + (remainder > getGpu()._id);

        if (_attributes & DataSetEnums::Sparse) {
            if (getGpu()._id == 0) {
                std::cout << "DataSet<T>::Shard: Data Sharding sparse dataset " << _name << " across all GPUs.\n";

                for (size_t i = 1; i < getGpu()._numprocs; i++) {
                    size_t localExamples = segment + (remainder > i);
                    std::vector<uint64_t> vLocalSparseStart(localExamples);
                    std::vector<uint64_t> vLocalSparseEnd(localExamples);
                    std::vector<uint32_t> vLocalSparseIndex;
                    std::vector<T> vLocalSparseData;
                    size_t position = i;

                    for (size_t j = position; j < _uniqueExamples; j += getGpu()._numprocs) {
                        vLocalSparseStart[j] = vLocalSparseIndex.size();

                        for (size_t k = _vSparseStart[j]; k < _vSparseEnd[j]; k++) {
                            vLocalSparseIndex.push_back(_vSparseIndex[k]);

                            if (!(_attributes & DataSetEnums::Boolean)) {
                                vLocalSparseData.push_back(_vSparseData[k]);
                            }
                        }

                        vLocalSparseEnd[j] = vLocalSparseIndex.size();
                    }

                    uint64_t size = vLocalSparseIndex.size();
                    MPI_Send(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                    MPI_Send(vLocalSparseStart.data(), localExamples, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                    MPI_Send(vLocalSparseEnd.data(), localExamples, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                    MPI_Send(vLocalSparseIndex.data(), size, MPI_UINT32_T, i, 0, MPI_COMM_WORLD);

                    if (!(_attributes & DataSetEnums::Boolean)) {
                        MPI_Datatype mpiType = getMPIDataType(_dataType);
                        MPI_Send(vLocalSparseData.data(), size, mpiType, i, 0, MPI_COMM_WORLD);
                    }
                }

                std::vector<uint64_t> vTempSparseStart = _vSparseStart;
                std::vector<uint64_t> vTempSparseEnd = _vSparseEnd;
                std::vector<uint32_t> vTempSparseIndex = _vSparseIndex;
                std::vector<T> vTempSparseData = _vSparseData;

                _vSparseIndex.resize(0);
                _vSparseData.resize(0);
                _vSparseStart.resize(_localExamples);
                _vSparseEnd.resize(_localExamples);

                for (uint32_t j = 0; j < _uniqueExamples; j++) {
                    _vSparseStart[j] = _vSparseIndex.size();

                    for (uint64_t k = vTempSparseStart[j]; k < vTempSparseEnd[j]; k++) {
                        if ((vTempSparseIndex[k] >= _minX) && (vTempSparseIndex[k] < _maxX)) {
                            _vSparseIndex.push_back(vTempSparseIndex[k]);

                            if (!(_attributes & DataSetEnums::Boolean)) {
                                _vSparseData.push_back(vTempSparseData[k]);
                            }
                        }
                    }

                    _vSparseEnd[j] = _vSparseIndex.size();
                }
            }
            else {
                // TODO: Implement the receive part for sparse data sharding
            }
        }
        else {
            if (getGpu()._id == 0) {
                std::cout << "DataSet<T>::Shard: Data Sharding dataset " << _name << " across all GPUs.\n";

                for (size_t i = 1; i < getGpu()._numprocs; i++) {
                    size_t localExamples = segment + (remainder > i);
                    std::vector<T> vLocalData(localExamples * _stride);
                    T* pData = vLocalData.data();
                    size_t position = i;

                    for (size_t j = position; j < _uniqueExamples; j += getGpu()._numprocs) {
                        std::copy(&_vData[j * _stride], &_vData[j * _stride] + _stride, pData);
                        pData += _stride;
                    }

                    size_t size = vLocalData.size();
                    MPI_Send(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                    MPI_Datatype mpiType = getMPIDataType(_dataType);
                    MPI_Send(vLocalData.data(), localExamples * _stride, mpiType, i, 0, MPI_COMM_WORLD);
                }

                std::vector<T> vLocalData(_localExamples * _stride);
                T* pData = vLocalData.data();
                size_t position = 0;

                for (size_t j = position; j < _uniqueExamples; j += getGpu()._numprocs) {
                    std::copy(&_vData[j * _stride], &_vData[j * _stride] + _stride, pData);
                    pData += _stride;
                }

                _vData.resize(_localExamples * _stride);
                _vData = vLocalData;
            }
            else {
                // TODO: Implement the receive part for non-sparse data sharding
            }

            _pbData.reset(new GpuBuffer<T>(_vData.size(), false, _bStreaming));
            _pbData->Upload(_vData.data());
        }
    }

    if (_attributes & DataSetEnums::Indexed) {
        _pbIndex.reset(new GpuBuffer<uint32_t>(_vIndex.size(), false, _bStreaming));
        _pbIndex->Upload(_vIndex.data());
    }

    return true;
}
/**
 * @brief Saves the DataSet to a NetCDF file.
 *
 * This function saves the DataSet to a NetCDF file. It creates a new NetCDF file with the specified filename,
 * writes the necessary attributes, and calls the WriteNetCDF function to write the DataSet's data to the file.
 *
 * @tparam T The data type of the DataSet.
 * @param fname The name of the NetCDF output file.
 * @return A boolean indicating whether the saving operation was successful (true) or not (false).
 */
template<typename T>
bool SaveNetCDF(const std::string& fname) {
    bool bResult = true;

    DataSetEnums::Sharding oldSharding = _sharding;
    UnShard();

    if (getGpu()._id == 0) {
        bool bOpened = false;
        try {
            NcFile nfc(fname, NcFile::replace);
            bOpened = true;

            NcGroupAtt datasetsAtt = nfc.putAtt("datasets", ncUint, 1);
            if (datasetsAtt.isNull()) {
                throw NC_EXCEPTION("NcException", "SaveNetCDF: Unable to write datasets attribute to NetCDF file " + fname, __FILE__, __LINE__);
            }

            bool bResult = WriteNetCDF(nfc, fname, 0);
            if (!bResult) {
                throw NC_EXCEPTION("NcException", "SaveNetCDF: Unable to write dataset to NetCDF file " + fname, __FILE__, __LINE__);
            }
        }
        catch (NcException& e) {
            if (!bOpened) {
                std::cout << "SaveNetCDF: Unable to create NetCDF output file " << fname << std::endl;
            }
            else {
                std::cout << e.what() << std::endl;
            }
            bResult = false;
        }
    }

    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (!bResult) {
        getGpu().Shutdown();
        std::exit(-1);
    }

    Shard(oldSharding);

    return bResult;
}
/**
 * @brief Saves the DataSet to a NetCDF file.
 *
 * This function saves the DataSet to a NetCDF file. It creates a new NetCDF file with the specified filename,
 * writes the necessary attributes, and calls the WriteNetCDF function to write the DataSet's data to the file.
 *
 * @param fname The name of the NetCDF output file.
 * @return A boolean indicating whether the saving operation was successful (true) or not (false).
 */
template<typename T>
bool SaveNetCDF(const std::string& fname) {
    bool bResult = true;

    DataSetEnums::Sharding oldSharding = _sharding;
    UnShard();

    if (getGpu()._id == 0) {
        bool bOpened = false;
        try {
            NcFile nfc(fname, NcFile::replace);
            bOpened = true;

            NcGroupAtt datasetsAtt = nfc.putAtt("datasets", ncUint, 1);
            if (datasetsAtt.isNull()) {
                throw NC_EXCEPTION("NcException", "SaveNetCDF: Unable to write datasets attribute to NetCDF file " + fname, __FILE__, __LINE__);
            }

            bool bResult = WriteNetCDF(nfc, fname, 0);
            if (!bResult) {
                throw NC_EXCEPTION("NcException", "SaveNetCDF: Unable to write dataset to NetCDF file " + fname, __FILE__, __LINE__);
            }
        }
        catch (NcException& e) {
            if (!bOpened) {
                std::cout << "SaveNetCDF: Unable to create NetCDF output file " << fname << std::endl;
            }
            else {
                std::cout << e.what() << std::endl;
            }
            bResult = false;
        }
    }

    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (!bResult) {
        getGpu().Shutdown();
        std::exit(-1);
    }

    Shard(oldSharding);

    return bResult;
}
/**
 * @brief Writes the DataSet to a NetCDF file.
 *
 * This function writes the DataSet object to a NetCDF file. It stores various attributes and variables
 * of the DataSet in the NetCDF file.
 *
 * @param nfc The NetCDF file object to write to.
 * @param fname The name of the NetCDF file.
 * @param n The index of the DataSet.
 * @return A boolean indicating whether the writing operation was successful (true) or not (false).
 */
template<typename T>
bool WriteNetCDF(NcFile& nfc, const std::string& fname, const uint32_t n) {
    try {
        if (getGpu()._id == 0) {
            std::string nstring = std::to_string(n);
            std::string vname = "name" + nstring;
            NcGroupAtt nameAtt = nfc.putAtt(vname, _name);
            if (nameAtt.isNull()) {
                throw NC_EXCEPTION("NcException", "DataSet::WriteNetCDF: Failed to write dataset name to NetCDF file " + fname, __FILE__, __LINE__);
            }

            vname = "attributes" + nstring;
            NcGroupAtt attributesAtt = nfc.putAtt(vname, ncUint, _attributes);
            if (attributesAtt.isNull()) {
                throw NC_EXCEPTION("NcException", "DataSet::WriteNetCDF: Failed to write dataset attributes to NetCDF file " + fname, __FILE__, __LINE__);
            }

            vname = "kind" + nstring;
            NcGroupAtt kindAtt = nfc.putAtt(vname, ncUint, 0);
            if (kindAtt.isNull()) {
                throw NC_EXCEPTION("NcException", "DataSet::WriteNetCDF: Failed to write dataset kind to NetCDF file " + fname, __FILE__, __LINE__);
            }

            vname = "datatype" + nstring;
            NcGroupAtt datatypeAtt = nfc.putAtt(vname, ncUint, _dataType);
            if (datatypeAtt.isNull()) {
                throw NC_EXCEPTION("NcException", "DataSet::WriteNetCDF: Failed to write dataset type to NetCDF file " + fname, __FILE__, __LINE__);
            }

            vname = "dimensions" + nstring;
            NcGroupAtt dimensionsAtt = nfc.putAtt(vname, ncUint, _dimensions);
            if (dimensionsAtt.isNull()) {
                throw NC_EXCEPTION("NcException", "DataSet::WriteNetCDF: Failed to write dataset dimensions to NetCDF file " + fname, __FILE__, __LINE__);
            }

            vname = "width" + nstring;
            NcGroupAtt widthAtt = nfc.putAtt(vname, ncUint, _width);
            if (widthAtt.isNull()) {
                throw NC_EXCEPTION("NcException", "DataSet::WriteNetCDF: Failed to write dataset width to NetCDF file " + fname, __FILE__, __LINE__);
            }

            if (_dimensions > 1) {
                vname = "height" + nstring;
                NcGroupAtt heightAtt = nfc.putAtt(vname, ncUint, _height);
                if (heightAtt.isNull()) {
                    throw NC_EXCEPTION("NcException", "DataSet::WriteNetCDF: Failed to write dataset height to NetCDF file " + fname, __FILE__, __LINE__);
                }

                if (_dimensions > 2) {
                    vname = "length" + nstring;
                    NcGroupAtt lengthAtt = nfc.putAtt(vname, ncUint, _length);
                    if (lengthAtt.isNull()) {
                        throw NC_EXCEPTION("NcException", "DataSet::WriteNetCDF: Failed to write dataset length to NetCDF file " + fname, __FILE__, __LINE__);
                    }
                }
            }

            vname = "uniqueExamplesDim" + nstring;
            NcDim uniqueExamplesDim = nfc.addDim(vname, static_cast<size_t>(_uniqueExamples));
            if (uniqueExamplesDim.isNull()) {
                throw NC_EXCEPTION("NcException", "DataSet::WriteNetCDF: Failed to write dataset unique example count to NetCDF file " + fname, __FILE__, __LINE__);
            }

            vname = "examplesDim" + nstring;
            NcDim examplesDim = nfc.addDim(vname, static_cast<size_t>(_examples));
            if (examplesDim.isNull()) {
                throw NC_EXCEPTION("NcException", "DataSet::WriteNetCDF: Failed to write dataset example count to NetCDF file " + fname, __FILE__, __LINE__);
            }

            if (_attributes & DataSetEnums::Sparse) {
                vname = "sparseDataDim" + nstring;
                NcDim sparseDataDim = nfc.addDim(vname, _vSparseIndex.size());
                if (sparseDataDim.isNull()) {
                    throw NC_EXCEPTION("NcException", "DataSet::WriteNetCDF: Failed to write dataset sparse datapoint count to NetCDF file " + fname, __FILE__, __LINE__);
                }

                vname = "sparseStart" + nstring;
                NcVar sparseStartVar = nfc.addVar(vname, "uint", uniqueExamplesDim.getName());
                if (sparseStartVar.isNull()) {
                    throw NC_EXCEPTION("NcException", "DataSet::WriteNetCDF: Failed to write dataset sparse start variable to NetCDF file " + fname, __FILE__, __LINE__);
                }
                sparseStartVar.putVar(_vSparseStart.data());

                vname = "sparseEnd" + nstring;
                NcVar sparseEndVar = nfc.addVar(vname, "uint", uniqueExamplesDim.getName());
                if (sparseEndVar.isNull()) {
                    throw NC_EXCEPTION("NcException", "DataSet::WriteNetCDF: Failed to write dataset sparse end variable to NetCDF file " + fname, __FILE__, __LINE__);
                }
                sparseEndVar.putVar(_vSparseEnd.data());

                vname = "sparseIndex" + nstring;
                NcVar sparseIndexVar = nfc.addVar(vname, "uint64", sparseDataDim.getName());
                if (sparseIndexVar.isNull()) {
                    throw NC_EXCEPTION("NcException", "DataSet::WriteNetCDF: Failed to write dataset sparse index variable to NetCDF file " + fname, __FILE__, __LINE__);
                }
                sparseIndexVar.putVar(_vSparseIndex.data());

                if (!(_attributes & DataSetEnums::Boolean)) {
                    vname = "sparseData" + nstring;
                    NcType sparseType = getNetCDFDataType(_dataType);
                    NcVar sparseDataVar = nfc.addVar(vname, sparseType.getName(), sparseDataDim.getName());
                    if (sparseDataVar.isNull()) {
                        throw NC_EXCEPTION("NcException", "DataSet::WriteNetCDF: Failed to write dataset sparse data variable to NetCDF file " + fname, __FILE__, __LINE__);
                    }
                    sparseDataVar.putVar(_vSparseData.data());
                }
            }
            else {
                // TODO: Implement non-sparse data writing if needed
            }

            if (_attributes & DataSetEnums::Weighted) {
                vname = "dataWeight" + nstring;
                NcVar DataWeightVar = nfc.addVar(vname, "float", uniqueExamplesDim.getName());
                if (DataWeightVar.isNull()) {
                    throw NC_EXCEPTION("NcException", "DataSet::DataSet: Failed to write data weights to NetCDF file " + fname, __FILE__, __LINE__);
                }
                DataWeightVar.putVar(_vDataWeight.data());
            }

            if (_attributes & DataSetEnums::Indexed) {
                vname = "index" + nstring;
                NcVar indexVar = nfc.addVar(vname, "uint32", examplesDim.getName());
                if (indexVar.isNull()) {
                    throw NC_EXCEPTION("NcException", "DataSet::WriteNetCDF: Failed to create dataset index variable to NetCDF file " + fname, __FILE__, __LINE__);
                }
                indexVar.putVar(_vIndex.data());
            }
        }
    }
    catch (NcException& e) {
        std::cout << e.what() << std::endl;
        return false;
    }

    return true;
}
template<typename T>
DataSet<T>::~DataSet()
{
    // Destructor implementation goes here (if any)
}
/**
 * @brief Saves NetCDF data sets to a file.
 *
 * This function saves the given vector of DataSetBase objects to a NetCDF output file.
 * It retrieves information about the data sets and writes them to the file.
 *
 * @param fname The name of the NetCDF output file.
 * @param vDataSet A vector of DataSetBase pointers representing the data sets to be saved.
 * @return A boolean indicating whether the saving operation was successful (true) or not (false).
 */
bool SaveNetCDF(const std::string& fname, const std::vector<DataSetBase*>& vDataSet)
{
    std::vector<DataSetEnums::Sharding> vSharding(vDataSet.size());
    for (size_t i = 0; i < vDataSet.size(); i++)
    {
        vSharding[i] = vDataSet[i]->_sharding;
        vDataSet[i]->UnShard();
    }

    bool bResult = true;
    try
    {
        if (getGpu()._id == 0)
        {
            NcFile nfc(fname, NcFile::replace);
            if (!nfc.is_valid())
            {
                std::cout << "SaveNetCDF: Unable to create NetCDF output file " << fname << std::endl;
                return false;
            }

            NcGroupAtt datasetsAtt = nfc.putAtt("datasets", ncUint, static_cast<unsigned int>(vDataSet.size()));
            if (datasetsAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "SaveNetCDF: Unable to write datasets attribute to NetCDF file " + fname, __FILE__, __LINE__);
            }

            for (size_t i = 0; i < vDataSet.size(); i++)
            {
                bool bResult = vDataSet[i]->WriteNetCDF(nfc, fname, static_cast<uint32_t>(i));
                if (!bResult)
                {
                    throw NC_EXCEPTION("NcException", "SaveNetCDF: Unable to write dataset to NetCDF file " + fname, __FILE__, __LINE__);
                }
            }
        }
    }
    catch (NcException& e)
    {
        std::cout << e.what() << std::endl;
        bResult = false;
    }

    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!bResult)
    {
        getGpu().Shutdown();
        exit(-1);
    }

    for (size_t i = 0; i < vDataSet.size(); i++)
    {
        vDataSet[i]->Shard(vSharding[i]);
    }

    return bResult;
}
/**
 * @brief Loads NetCDF data sets from a file.
 *
 * This function reads NetCDF input file and retrieves information about the data sets and their types.
 * It creates corresponding DataSetBase objects for each data set and returns them in a vector.
 *
 * @param fname The name of the NetCDF input file.
 * @return A vector of DataSetBase pointers representing the loaded data sets.
 */
std::vector<DataSetBase*> LoadNetCDF(const std::string& fname)
{
    std::vector<DataSetBase*> vDataSet;
    std::vector<DataSetEnums::DataType> vDataType;
    bool bResult = true;

    if (getGpu()._id == 0)
    {
        bool bOpened = false;
        try
        {
            NcFile rnc(fname, NcFile::read);
            bOpened = true;

            NcGroupAtt dataSetsAtt = rnc.getAtt("datasets");
            if (dataSetsAtt.isNull())
            {
                throw NC_EXCEPTION("NcException", "LoadNetCDF: No datasets count supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            uint32_t datasets;
            dataSetsAtt.getValues(&datasets);

            for (uint32_t i = 0; i < datasets; i++)
            {
                std::string nstring = std::to_string(i);
                std::string vname = "dataType" + nstring;
                NcGroupAtt dataTypeAtt = rnc.getAtt(vname);
                if (dataTypeAtt.isNull())
                {
                    throw NC_EXCEPTION("NcException", "LoadNetCDF: No " + vname + " attribute located in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                uint32_t dataType;
                dataTypeAtt.getValues(&dataType);
                switch (dataType)
                {
                    case DataSetEnums::UInt:
                    case DataSetEnums::Int:
                    case DataSetEnums::LLInt:
                    case DataSetEnums::ULLInt:
                    case DataSetEnums::Float:
                    case DataSetEnums::Double:
                    case DataSetEnums::RGB8:
                    case DataSetEnums::RGB16:
                    case DataSetEnums::UChar:
                    case DataSetEnums::Char:
                        vDataType.push_back(static_cast<DataSetEnums::DataType>(dataType));
                        break;

                    default:
                        std::cout << "LoadNetCDF: Invalid data type in binary input file " << fname << std::endl;
                }
            }
        }
        catch (NcException& e)
        {
            if (!bOpened)
            {
                std::cout << "NcException: LoadNetCDF: Error opening NetCDF input file " << fname << std::endl;
            }
            else
            {
                std::cout << "Exception: " << e.what() << std::endl;
            }
            bResult = false;
        }
    }

    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!bResult)
    {
        getGpu().Shutdown();
        exit(-1);
    }

    uint32_t size = static_cast<uint32_t>(vDataType.size());
    MPI_Bcast(&size, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    vDataType.resize(size);
    MPI_Bcast(vDataType.data(), size, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    for (int i = 0; i < vDataType.size(); i++)
    {
        DataSetBase* pDataSet = nullptr;
        if (getGpu()._id == 0)
            std::cout << "LoadNetCDF: Loading " << vDataType[i] << " data set" << std::endl;
        switch (vDataType[i])
        {
            case DataSetEnums::UInt:
                pDataSet = new DataSet<uint32_t>(fname, i);
                break;

            case DataSetEnums::Int:
                pDataSet = new DataSet<long>(fname, i);
                break;

            case DataSetEnums::Float:
                pDataSet = new DataSet<float>(fname, i);
                break;

            case DataSetEnums::Double:
                pDataSet = new DataSet<double>(fname, i);
                break;

            case DataSetEnums::Char:
                pDataSet = new DataSet<char>(fname, i);
                break;

            case DataSetEnums::UChar:
            case DataSetEnums::RGB8:
                pDataSet = new DataSet<uint8_t>(fname, i);
                break;

            default:
                std::cout << "LoadNetCDF: Invalid dataset type in binary input file " << fname << std::endl;
                getGpu().Shutdown();
                exit(-1);
        }
        vDataSet.push_back(pDataSet);
    }

    return vDataSet;
}

/**
 * @brief Loads image data sets from a file.
 *
 * This function reads image data from the specified file and creates corresponding DataSetBase objects
 * for each image data set found in the file. The data sets are returned in a vector.
 *
 * @param fname The name of the image file.
 * @return A vector of DataSetBase pointers representing the loaded data sets.
 */
std::vector<DataSetBase*> LoadImageData(const std::string& fname) {}

/**
 * @brief Loads CSV data sets from a file.
 *
 * This function reads CSV data from the specified file and creates corresponding DataSetBase objects
 * for each CSV data set found in the file. The data sets are returned in a vector.
 *
 * @param fname The name of the CSV file.
 * @return A vector of DataSetBase pointers representing the loaded data sets.
 */
std::vector<DataSetBase*> LoadCSVData(const std::string& fname) {}

/**
 * @brief Loads JSON data sets from a file.
 *
 * This function reads JSON data from the specified file and creates corresponding DataSetBase objects
 * for each JSON data set found in the file. The data sets are returned in a vector.
 *
 * @param fname The name of the JSON file.
 * @return A vector of DataSetBase pointers representing the loaded data sets.
 */
std::vector<DataSetBase*> LoadJSONData(const std::string& fname) {}

/**
 * @brief Loads audio data sets from a file.
 *
 * This function reads audio data from the specified file and creates corresponding DataSetBase objects
 * for each audio data set found in the file. The data sets are returned in a vector.
 *
 * @param name The name of the audio file.
 * @return A vector of DataSetBase pointers representing the loaded data sets.
 */
std::vector<DataSetBase*> LoadAudioData(const std::string& name) {}
