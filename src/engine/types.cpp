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

template <typename T>
class DataSet;

template class DataSet<Float>;
template class DataSet<double>;
template class DataSet<unsigned char>;
template class DataSet<char>;
template class DataSet<uint32_t>;
template class DataSet<uint64_t>;
template class DataSet<int32_t>;
template class DataSet<int64_t>;

using namespace netCDF;
using namespace netCDF::exceptions;

static std::map<TrainingMode, std::string> sTrainingModeMap = {
    {TrainingMode::SGD,      "SGD"},
    {TrainingMode::Momentum, "Momentum"},
    {TrainingMode::AdaGrad,  "AdaGrad"},
    {TrainingMode::Nesterov, "Nesterov"},
    {TrainingMode::RMSProp,  "RMSProp"},
    {TrainingMode::AdaDelta, "AdaDelta"},
    {TrainingMode::Adam,     "Adam"}
};

template <typename T>
concept EnumType = std::is_enum_v<T>;

template <EnumType T>
std::ostream& operator<<(std::ostream& out, const T& e)
{
    out << sTrainingModeMap[e];
    return out;
}

static std::map<ErrorFunction, std::string> sErrorFunctionMap = {
    {ErrorFunction::L1,                             "L1"},
    {ErrorFunction::L2,                             "L2"},
    {ErrorFunction::CrossEntropy,                   "CrossEntropy"},
    {ErrorFunction::ScaledMarginalCrossEntropy,     "ScaledMarginalCrossEntropy"},
    {ErrorFunction::Hinge,                          "Hinge"},
    {ErrorFunction::L2Hinge,                        "L2Hinge"},
};

template <EnumType T>
std::ostream& operator<<(std::ostream& out, const T& e)
{
    out << sErrorFunctionMap[e];
    return out;
}

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

template <EnumType T>
std::ostream& operator<<(std::ostream& out, const T& a)
{
    out << sActivationMap[a];
    return out;
}

static std::map<WeightInitialization, std::string> sWeightInitializationMap = {
    {WeightInitialization::Xavier,           "Xavier"},
    {WeightInitialization::CaffeXavier,      "CaffeXavier"},
    {WeightInitialization::Gaussian,         "Gaussian"},
    {WeightInitialization::Uniform,          "Uniform"},
    {WeightInitialization::UnitBall,         "UnitBall"},
    {WeightInitialization::Constant,         "Constant"},
    {WeightInitialization::SELU,             "SELU"}
};

template <EnumType T>
std::ostream& operator<<(std::ostream& out, const T& w)
{
    out << sWeightInitializationMap[w];
    return out;
}

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

template <EnumType T>
std::ostream& operator<<(std::ostream& out, const T& a)
{
    out << sPoolingFunctionMap[a];
    return out;
}

static std::map<DataSetEnums::Kind, std::string> sKindMap = {
    {DataSetEnums::Numeric, "Numeric"},
    {DataSetEnums::Image,   "Image"},
    {DataSetEnums::Audio,   "Audio"}
};

std::ostream& operator<<(std::ostream& out, DataSetEnums::Kind& k)
{
    out << sKindMap[k];
    return out;
}

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

std::ostream& operator<<(std::ostream& out, DataSetEnums::Attributes& a)
{
    out << sAttributesMap[a];
    return out;
}

static std::map<DataSetEnums::Sharding, std::string> sShardingMap = {
    {DataSetEnums::None,  "None"},
    {DataSetEnums::Model, "Model"},
    {DataSetEnums::Data,  "Data"}
};

std::ostream& operator<<(std::ostream& out, DataSetEnums::Sharding& s)
{
    out << sShardingMap[s];
    return out;
}

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

std::ostream& operator<<(std::ostream& out, DataSetEnums::DataType& t)
{
    out << sDataTypeMap[t];
    return out;
}

static std::map<DataSetEnums::Attributes, std::string> sAttributesMap = {
    {DataSetEnums::Attributes::Sparse, "Sparse"},
    {DataSetEnums::Attributes::Boolean, "Boolean"},
    {DataSetEnums::Attributes::Compressed, "Compressed"},
    {DataSetEnums::Attributes::Recurrent, "Recurrent"},
    {DataSetEnums::Attributes::Mutable, "Mutable"},
    {DataSetEnums::Attributes::SparseIgnoreZero, "SparseIgnoreZero"},
    {DataSetEnums::Attributes::Indexed, "Indexed"},
    {DataSetEnums::Attributes::Weighted, "Weighted"},
};

std::ostream& operator<< (std::ostream& out, const DataSetEnums::Attributes& a)
{
    if (const auto it = sAttributesMap.find(a); it != sAttributesMap.end()) {
        out << it->second;
    } else {
        out << "Unknown";
    }
    return out;
}

std::map<DataSetEnums::Sharding, std::string> sShardingMap = {
    {DataSetEnums::None, "None"},
    {DataSetEnums::Model, "Model"},
    {DataSetEnums::Data, "Data"}
};

std::ostream& operator<< (std::ostream& out, const DataSetEnums::Sharding& s)
{
    if (const auto it = sShardingMap.find(s); it != sShardingMap.end()) {
        out << it->second;
    } else {
        out << "Unknown";
    }
    return out;
}

std::map<DataSetEnums::DataType, std::string> sDataTypeMap = {
    {DataSetEnums::UInt, "UInt"},
    {DataSetEnums::Int, "Int"},
    {DataSetEnums::LLInt, "LLInt"},
    {DataSetEnums::ULLInt, "ULLInt"},
    {DataSetEnums::Float, "Float"},
    {DataSetEnums::Double, "Double"},
    {DataSetEnums::RGB8, "RGB8"},
    {DataSetEnums::RGB16, "RGB16"},
    {DataSetEnums::UChar, "UChar"},
    {DataSetEnums::Char, "Char"}
};

std::ostream& operator<< (std::ostream& out, const DataSetEnums::DataType& t)
{
    if (const auto it = sDataTypeMap.find(t); it != sDataTypeMap.end()) {
        out << it->second;
    } else {
        out << "Unknown";
    }
    return out;
}

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

inline bool has_suffix(const std::string& str, const std::string& suffix)
{
    return str.ends_with(suffix);
}

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

DataSetDimensions::DataSetDimensions() : DataSetDimensions(1, 1, 1) {}

DataSetDimensions::DataSetDimensions(uint32_t width, uint32_t height, uint32_t length) :
    _width(width),
    _height(height),
    _length(length),
    _dimensions(std::count_if(
        {width, height, length},
        [](uint32_t dimension) { return dimension > 1; }
    ))
{}

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

std::unordered_map<DataSetEnums::DataType, std::function<std::unique_ptr<DataSetBase>(const DataSetDescriptor&)>> datasetCreationMap = {
    {DataSetEnums::DataType::UInt, [](const DataSetDescriptor& descriptor) { return std::make_unique<DataSet<uint32_t>>(descriptor); }},
    {DataSetEnums::DataType::Int, [](const DataSetDescriptor& descriptor) { return std::make_unique<DataSet<int>>(descriptor); }},
    {DataSetEnums::DataType::Float, [](const DataSetDescriptor& descriptor) { return std::make_unique<DataSet<float>>(descriptor); }},
    {DataSetEnums::DataType::Double, [](const DataSetDescriptor& descriptor) { return std::make_unique<DataSet<double>>(descriptor); }},
    {DataSetEnums::DataType::Char, [](const DataSetDescriptor& descriptor) { return std::make_unique<DataSet<char>>(descriptor); }},
    {DataSetEnums::DataType::UChar, [](const DataSetDescriptor& descriptor) { return std::make_unique<DataSet<uint8_t>>(descriptor); }},
    {DataSetEnums::DataType::RGB8, [](const DataSetDescriptor& descriptor) { return std::make_unique<DataSet<uint8_t>>(descriptor); }}
};

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