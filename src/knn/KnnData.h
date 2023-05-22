#ifndef LIBKNN_KNN_HANDLE_H_
#define LIBKNN_KNN_HANDLE_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string_view>
#include <vector>
#include <span>
#include <string>
#include <array>

#include <cublas_v2.h>

#include "DataReader.h"

namespace astdl::knn
{

/**
 * \brief Enumeration representing the data type.
 */
enum class DataType
{
  FP32 = 0, /**< Single-precision floating-point data type */
  FP16 = 1  /**< Half-precision floating-point data type */
};

/**
 * \brief Get the string representation of the DataType.
 * \param dataType The DataType value.
 * \return The string representation of the DataType.
 */
[[nodiscard]] std::string_view getDataTypeString(DataType dataType);

/**
 * \brief Get the DataType value from its string representation.
 * \param dataTypeLiteral The string representation of the DataType.
 * \return The DataType value.
 */
[[nodiscard]] DataType getDataTypeFromString(std::string_view dataTypeLiteral);

/**
 * \brief Structure representing a matrix.
 */
struct Matrix
{
    void* data;                   /**< Pointer to the data */
    uint32_t numRows;             /**< Number of rows */
    int numColumns;               /**< Number of columns */
    size_t elementSize;           /**< Size of each element */
    cudaMemoryType_t memoryType;  /**< Memory type */

    /**
     * \brief Default constructor for Matrix.
     */
    Matrix();

    /**
     * \brief Constructor for Matrix.
     * \param data Pointer to the data.
     * \param numRows Number of rows.
     * \param numColumns Number of columns.
     * \param elementSize Size of each element.
     * \param memoryType Memory type.
     */
    Matrix(void* data, uint32_t numRows, int numColumns, size_t elementSize, cudaMemoryType_t memoryType);

    /**
     * \brief Get the size of the matrix in bytes.
     * \return The size of the matrix in bytes.
     */
    [[nodiscard]] size_t getSizeInBytes() const;

    /**
     * \brief Get the total number of elements in the matrix.
     * \return The total number of elements in the matrix.
     */
    [[nodiscard]] size_t getLength() const;
};

/**
 * \brief Load data on the host using a DataReader.
 * \param dataReader Pointer to the DataReader object.
 * \return The loaded Matrix object.
 */
Matrix loadDataOnHost(DataReader* dataReader);

/**
 * \brief Structure representing the KNN data.
 */
struct KnnData
{
    const int numGpus;                        /**< Number of GPUs */
    const int batchSize;                       /**< Batch size */
    const int maxK;                            /**< Maximum K value */

    std::vector<cublasHandle_t> cublasHandles; /**< Cublas handles */

    std::vector<Matrix> dCollectionPartitions;  /**< Collection partitions stored on device */
    std::vector<Matrix> dInputBatches;          /**< Input batches stored on device */
    std::vector<Matrix> dProducts;              /**< Intermediate products stored on device */

    std::vector<Matrix> dResultScores;          /**< Result scores stored on device */
    std::vector<Matrix> dResultIndexes;         /**< Result indexes stored on device */

    std::vector<Matrix> hResultScores;          /**< Result scores stored on host */
    std::vector<Matrix> hResultIndexes;         /**< Result indexes stored on host */
    std::vector<std::vector<std::string>> hKeys;/**< Keys stored on host */

    std::vector<Matrix> dInputBatchTmpBuffers;  /**< Temporary input batch buffers stored on device */

    std::vector<uint32_t> collectionRowsPadded; /**< Collection rows padded */

    std::vector<float> elapsedSgemm;            /**< Elapsed time for sgemm */
    std::vector<float> elapsedTopK;             /**< Elapsed time for top-K */

    const DataType dataType;                   /**< Data type */

    /**
     * \brief Constructor for KnnData.
     * \param numGpus Number of GPUs.
     * \param batchSize Batch size.
     * \param maxK Maximum K value.
     * \param dataType Data type.
     */
    KnnData(int numGpus, int batchSize, int maxK, DataType dataType);

    /**
     * \brief Load data for a specific device using a DataReader.
     * \param device The device number.
     * \param dataReader Pointer to the DataReader object.
     */
    void load(int device, DataReader* dataReader);

    /**
     * \brief Load data for multiple devices using a map of device to DataReader pointers.
     * \param deviceToData Map of device number to DataReader pointers.
     */
    void load(const std::map<int, DataReader*>& deviceToData);

    /**
     * \brief Load data for multiple devices using a map of device to file paths and delimiters.
     * \param deviceToFile Map of device number to file paths.
     * \param keyValDelim Key-value delimiter character.
     * \param vecDelim Vector delimiter character.
     */
    void load(const std::map<int, std::string_view>& deviceToFile, char keyValDelim, char vecDelim);

    /**
     * \brief Get the feature size.
     * \return The feature size.
     */
    [[nodiscard]] int getFeatureSize() const;

    /**
     * \brief Destructor for KnnData.
     */
    ~KnnData();
};

/**
 * \brief Allocate a matrix on the host.
 * \param numRows Number of rows.
 * \param numColumns Number of columns.
 * \param elementSize Size of each element.
 * \return The allocated Matrix object.
 */
[[nodiscard]] Matrix allocateMatrixOnHost(uint32_t numRows, int numColumns, size_t elementSize);

/**
 * \brief Allocate a matrix on the device.
 * \param numRows Number of rows.
 * \param numColumns Number of columns.
 * \param elementSize Size of each element.
 * \return The allocated Matrix object.
 */
[[nodiscard]] Matrix allocateMatrixOnDevice(uint32_t numRows, int numColumns, size_t elementSize);

/**
 * \brief Free the memory occupied by a matrix.
 * \param matrix The Matrix object to free.
 */
void freeMatrix(const Matrix& matrix);

}  // namespace astdl::knn

#endif  // LIBKNN_KNN_HANDLE_H_
