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

DataSetBase::~DataSetBase() = default;

DataSetDimensions DataSetBase::GetDimensions()
{
    return { _width, _height, _length };
}

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

template<typename T>
DataSet<T>::DataSet(uint32_t examples, const DataSetDimensions& dim, const string& name)
    : DataSetBase(name, DataSetEnums::getDataType<T>(), examples, examples, dim),
      _sparseDensity(1.0f),
      _stride(_width * _height * _length),
      _vData(_stride * _examples),
      _pbData(make_unique<GpuBuffer<T>>(_vData.size(), false, _bStreaming))
{
}

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

template<typename T>
DataSet<T>::DataSet(uint32_t examples, Float sparseDensity, const DataSetDimensions& dim, bool isWeighted, const string& name)
    : DataSet(examples, examples, (size_t)(((double)(dim._width * dim._height * dim._length * examples)) * sparseDensity), dim, false, isWeighted, name)
{
    _attributes = DataSetEnums::Attributes::Sparse;
    if (isWeighted) {
        _attributes |= DataSetEnums::Attributes::Weighted;
    }
}

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
template<typename T>
bool DataSet<T>::Rename(const std::string& name)
{
    _name = name;
    return true;
}

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
template<typename T>
bool DataSet<T>::GetStreaming() const
{
    return _bStreaming;
}

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