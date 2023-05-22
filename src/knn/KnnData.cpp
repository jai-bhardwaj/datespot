#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <span>

#include "cudautil.h"
#include "KnnData.h"
#include "MathUtil.h"

namespace astdl::knn {

    /**
     * @brief Number of rows to pad the matrix for alignment.
     */
    inline static constexpr int ROW_PADDING = 8;

    /**
     * @brief Mapping from string to DataType enumeration.
     */
    inline static const std::unordered_map<std::string, DataType> STRING_TO_DATA_TYPE = {
        { "fp32", DataType::FP32 },
        { "fp16", DataType::FP16 }
    };

    Matrix::Matrix() noexcept
        : data(nullptr),
          numRows(0),
          numColumns(0),
          elementSize(0),
          memoryType(cudaMemoryTypeHost)
    {
    }

    Matrix::Matrix(void* data, uint32_t numRows, int numColumns, size_t elementSize, cudaMemoryType memoryType) noexcept
        : data(data),
          numRows(numRows),
          numColumns(numColumns),
          elementSize(elementSize),
          memoryType(memoryType)
    {
    }

    size_t Matrix::getSizeInBytes() const noexcept
    {
        return numRows * numColumns * elementSize;
    }

    size_t Matrix::getLength() const noexcept
    {
        return numRows * numColumns;
    }

    std::string getDataTypeString(DataType dataType)
    {
        switch (dataType) {
            case DataType::FP32:
                return "fp32";
            case DataType::FP16:
                return "fp16";
            default:
                return "unknown";
        }
    }

    DataType getDataTypeFromString(const std::string& dataTypeLiteral)
    {
        if (const auto entry = STRING_TO_DATA_TYPE.find(dataTypeLiteral); entry != STRING_TO_DATA_TYPE.end())
            return entry->second;

        std::stringstream msg;
        msg << "Unknown DataType " << dataTypeLiteral;
        throw std::invalid_argument(msg.str());
    }

    Matrix loadDataOnHost(DataReader* dataReader)
    {
        uint32_t rows = dataReader->getRows();
        int columns = dataReader->getColumns();
        size_t elementSize = sizeof(float);

        Matrix matrix = allocateMatrixOnHost(rows, columns, elementSize);

        std::string ignored;
        for (int rowNum = 0; dataReader->readRow(&ignored, static_cast<float*>(matrix.data) + (rowNum * columns)); ++rowNum)
        {
        }
        return matrix;
    }

    KnnData::KnnData(int numGpus, int batchSize, int maxK, DataType dataType)
        : numGpus(numGpus),
          batchSize(batchSize),
          maxK(maxK),
          dataType(dataType),
          dCollectionPartitions(numGpus),
          dInputBatches(numGpus),
          dProducts(numGpus),
          dResultScores(numGpus),
          dResultIndexes(numGpus),
          hResultScores(numGpus),
          hResultIndexes(numGpus),
          dInputBatchTmpBuffers(numGpus),
          collectionRowsPadded(numGpus),
          hKeys(numGpus),
          elapsedSgemm(numGpus),
          elapsedTopK(numGpus)
    {
        int deviceCount = astdl::cuda_util::getDeviceCount();
        if (deviceCount < 1)
        {
            std::stringstream msg;
            msg << "No GPU device found on host. Device count is " << deviceCount;
            throw std::runtime_error(msg.str());
        }

        if (deviceCount < numGpus)
        {
            std::stringstream msg;
            msg << "Not enough GPUs on host. Required " << numGpus << ", found " << deviceCount;
            throw std::runtime_error(msg.str());
        }

        std::cout << "INFO: Initializing KnnData with numGpus = " << numGpus << ", batchSize = " << batchSize
                  << ", maxK = " << maxK << ", dataType = " << getDataTypeString(dataType) << '\n';

        if (dataType == DataType::FP16)
        {
            cudaDeviceProp deviceProp;
            CHECK_ERR(cudaGetDeviceProperties(&deviceProp, 0));
            int smMajor = deviceProp.major;
            int smMinor = deviceProp.minor;

            if (smMajor < 7)
            {
                std::cout << "WARNING: fp16 compute is not supported in sm " << smMajor << '.' << smMinor
                          << " < 7. Only storing data in fp16.\n";
            }
        }

        cublasHandles.reserve(numGpus);
        for (int i = 0; i < numGpus; ++i)
        {
            CHECK_ERR(cudaSetDevice(i));
            cublasHandle_t handle;
            STATUS_ERR(cublasCreate(&handle));

            if (dataType == DataType::FP16)
            {
                std::cout << "INFO: On device " << i << ", setting cublas mode to CUBLAS_TENSOR_OP_MATH\n";
                STATUS_ERR(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
            }

            cublasHandles.push_back(handle);
        }
    }

    int KnnData::getFeatureSize() const noexcept
    {
        return dCollectionPartitions[0].numColumns;
    }

    void KnnData::load(int device, DataReader* dataReader)
    {
        CHECK_ERR(cudaSetDevice(device));

        uint32_t actualRows = dataReader->getRows();
        uint32_t rows = ((actualRows + (ROW_PADDING - 1)) / ROW_PADDING) * ROW_PADDING;
        uint32_t rowsPadded = rows - actualRows;
        int columns = dataReader->getColumns();

        Matrix hTmpMatrix = allocateMatrixOnHost(rows, columns, sizeof(float));
        size_t hTmpDataBytes = hTmpMatrix.getSizeInBytes();
        float* hTmpData = static_cast<float*>(hTmpMatrix.data);

        collectionRowsPadded[device] = rowsPadded;

        hKeys[device].reserve(actualRows);
        std::string key;
        float vector[columns];
        for (const auto& [key, vector] : std::span(hKeys[device]).zip(std::span(hTmpData, rows * columns).subspan(0, actualRows * columns)))
        {
            key.push_back(key);
            std::copy_n(vector.data(), columns, hTmpData);
            hTmpData += columns;
        }

        if (dataType == DataType::FP16)
        {
            dCollectionPartitions[device] = allocateMatrixOnDevice(rows, columns, sizeof(half));
            astdl::math::kFloatToHalf(hTmpMatrix.data, hTmpDataBytes, static_cast<half*>(dCollectionPartitions[device].data));

            dInputBatches[device] = allocateMatrixOnDevice(batchSize, columns, sizeof(half));
            dInputBatchTmpBuffers[device] = allocateMatrixOnDevice(batchSize, columns, sizeof(float));
        }
        else
        {
            dCollectionPartitions[device] = allocateMatrixOnDevice(rows, columns, sizeof(float));
            CHECK_ERR(cudaMemcpy(dCollectionPartitions[device].data, hTmpMatrix.data, hTmpDataBytes, cudaMemcpyHostToDevice));
            dInputBatches[device] = allocateMatrixOnDevice(batchSize, columns, sizeof(float));
        }

        freeMatrix(hTmpMatrix);

        dProducts[device] = allocateMatrixOnDevice(batchSize, rows, sizeof(float));
        dResultScores[device] = allocateMatrixOnDevice(batchSize, maxK, sizeof(float));
        dResultIndexes[device] = allocateMatrixOnDevice(batchSize, maxK, sizeof(uint32_t));

        hResultScores[device] = allocateMatrixOnHost(batchSize, maxK, sizeof(float));
        hResultIndexes[device] = allocateMatrixOnHost(batchSize, maxK, sizeof(uint32_t));

        size_t totalMemory;
        size_t freeMemory;
        astdl::cuda_util::getDeviceMemoryInfoInMb(device, &totalMemory, &freeMemory);

        std::cout << "INFO: loaded " << actualRows << " (" << rows << " padded) rows and " << columns
                  << " columns into device " << device << ". Used: " << totalMemory - freeMemory << " MB, Free: "
                  << freeMemory << " MB, Total: " << totalMemory << " MB\n";
    }

    void KnnData::load(const std::vector<std::pair<int, DataReader*>>& deviceToData)
    {
        #pragma omp parallel for num_threads(numGpus)
        for (size_t i = 0; i < deviceToData.size(); ++i)
        {
            int device = deviceToData[i].first;
            DataReader* dataReader = deviceToData[i].second;
            load(device, dataReader);
        }
    }

    void KnnData::load(const std::map<int, std::string>& deviceToFile, char keyValDelim, char vecDelim)
    {
        std::vector<std::pair<int, DataReader*>> deviceToData;
        for (const auto& entry : deviceToFile)
        {
            int device = entry.first;
            std::string file = entry.second;
            DataReader* dataReader = new TextFileDataReader(file, keyValDelim, vecDelim);
            deviceToData.emplace_back(device, dataReader);
        }

        load(deviceToData);

        for (const auto& [device, dataReader] : deviceToData)
        {
            delete dataReader;
        }
        deviceToData.clear();
    }

    KnnData::~KnnData()
    {
        for (auto handle : cublasHandles)
        {
            cublasDestroy(handle);
        }
        for (auto& dCollection : dCollectionPartitions)
        {
            freeMatrix(dCollection);
        }
        for (auto& dInputBatch : dInputBatches)
        {
            freeMatrix(dInputBatch);
        }
        for (auto& dProduct : dProducts)
        {
            freeMatrix(dProduct);
        }
        for (auto& dResultScore : dResultScores)
        {
            freeMatrix(dResultScore);
        }
        for (auto& dResultIndex : dResultIndexes)
        {
            freeMatrix(dResultIndex);
        }
        for (auto& hResultScore : hResultScores)
        {
            freeMatrix(hResultScore);
        }
        for (auto& hResultIndex : hResultIndexes)
        {
            freeMatrix(hResultIndex);
        }
        for (auto& hKey : hKeys)
        {
            hKey.clear();
        }

        for (auto& dInputBatchTmpBuffer : dInputBatchTmpBuffers)
        {
            freeMatrix(dInputBatchTmpBuffer);
        }

        cublasHandles.clear();
        dCollectionPartitions.clear();
        dInputBatches.clear();
        dProducts.clear();
        dResultScores.clear();
        dResultIndexes.clear();
        hKeys.clear();
        elapsedSgemm.clear();
        elapsedTopK.clear();
    }

    Matrix allocateMatrixOnHost(uint32_t numRows, int numColumns, size_t elementSize)
    {
        void* data = malloc(numRows * numColumns * elementSize);
        return Matrix(data, numRows, numColumns, elementSize, cudaMemoryTypeHost);
    }

    Matrix allocateMatrixOnDevice(uint32_t numRows, int numColumns, size_t elementSize)
    {
        void* data;
        CHECK_ERR(cudaMalloc(&data, numRows * numColumns * elementSize));
        return Matrix(data, numRows, numColumns, elementSize, cudaMemoryTypeDevice);
    }

    void freeMatrix(const Matrix& matrix)
    {
        if (matrix.data != nullptr)
        {
            switch (matrix.memoryType) {
                case cudaMemoryTypeDevice:
                    CHECK_ERR(cudaFree(matrix.data))
                    break;
                case cudaMemoryTypeHost:
                    free(matrix.data);
                    break;
                default:
                    std::stringstream msg;
                    msg << "Unknown memory type " << matrix.memoryType;
                    throw std::invalid_argument(msg.str());
            }
        }
    }
}  // namespace astdl::knn
