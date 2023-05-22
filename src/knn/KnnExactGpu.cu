#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <omp.h>
#include <sstream>
#include <array>
#include <vector>

#include "cudautil.h"
#include "KnnExactGpu.h"
#include "MathUtil.h"
#include "Output.h"

namespace astdl {
namespace knn {

    /**
     * @brief Constructor for the Knn class.
     * @param data Pointer to the KnnData object.
     */
    Knn::Knn(KnnData* data) : data(data) {}

    /**
     * @brief Constructor for the KnnExactGpu class.
     * @param data Pointer to the KnnData object.
     */
    KnnExactGpu::KnnExactGpu(KnnData* data) : Knn(data) {}

    /**
     * @brief Performs k-nearest neighbor search on the GPU.
     * @param k Number of nearest neighbors to retrieve.
     * @param inputs Pointer to the input data.
     * @param size Size of the input data.
     * @param keys Pointer to store the keys of the nearest neighbors.
     * @param scores Pointer to store the scores of the nearest neighbors.
     */
    void KnnExactGpu::search(int k, const float* inputs, int size, std::string* keys, float* scores) {
        int maxK = data->maxK;
        int batchSize = data->batchSize;
        int numGpus = data->numGpus;

        // Check if k exceeds maxK
        if (k > maxK) {
            std::stringstream msg;
            msg << "k = " << k << " is > maxK = " << maxK;
            throw std::invalid_argument(msg.str());
        }

        // Check if size exceeds batchSize
        if (size > batchSize) {
            std::stringstream msg;
            msg << "size = " << size << " is > batchSize = " << batchSize;
            throw std::invalid_argument(msg.str());
        }

        batchSize = size;

        std::array<float*, numGpus> allScores{};
        std::array<uint32_t*, numGpus> allIndexes{};

        #pragma omp parallel num_threads(numGpus)
        {
            int device = omp_get_thread_num();
            cudaSetDevice(device);

            cublasHandle_t handle = data->cublasHandles[device];
            Matrix dCollectionPartition = data->dCollectionPartitions[device];
            Matrix dInputBatch = data->dInputBatches[device];
            Matrix dProducts = data->dProducts[device];
            Matrix dResultScores = data->dResultScores[device];
            Matrix dResultIndexes = data->dResultIndexes[device];
            Matrix hResultScores = data->hResultScores[device];
            Matrix hResultIndexes = data->hResultIndexes[device];
            uint32_t paddedRows = data->collectionRowsPadded[device];

            void* dA = dCollectionPartition.data;
            void* dB = dInputBatch.data;
            void* dC = dProducts.data;

            float* dScores = (float*)dResultScores.data;
            uint32_t* dIndexes = (uint32_t*)dResultIndexes.data;

            float* hScores = (float*)hResultScores.data;
            uint32_t* hIndexes = (uint32_t*)hResultIndexes.data;

            uint32_t aRows = dCollectionPartition.numRows;
            uint32_t bRows = batchSize;
            uint32_t cRows = batchSize;
            int aColumns = dCollectionPartition.numColumns;
            int bColumns = dInputBatch.numColumns;
            int cColumns = dProducts.numColumns;

            cudaDataType aType;
            cudaDataType bType;
            cudaDataType cType = CUDA_R_32F;

            if (data->dataType == astdl::knn::DataType::FP16) {
                aType = CUDA_R_16F;
                bType = CUDA_R_16F;

                Matrix tmpBuffer = data->dInputBatchTmpBuffers[device];
                astdl::math::kFloatToHalf(inputs, dInputBatch.getLength() * sizeof(float), (half*)dB, (float*)tmpBuffer.data,
                    tmpBuffer.getSizeInBytes());
            }
            else if (data->dataType == astdl::knn::DataType::FP32) {
                aType = CUDA_R_32F;
                bType = CUDA_R_32F;
                cudaMemcpy(dB, inputs, dInputBatch.getSizeInBytes(), cudaMemcpyHostToDevice);
            }
            else {
                throw std::runtime_error("Unknown data type");
            }

            static const cublasOperation_t transa = CUBLAS_OP_N;
            static const cublasOperation_t transb = CUBLAS_OP_N;
            static const float alpha = 1.0f;
            static const float beta = 0.0f;

            cudaEvent_t start, stop;
            float elapsed;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);
            cublasSgemmEx(handle, transa, transb, aRows, bRows, aColumns, &alpha, dA, aType, aRows, dB, bType,
                bColumns, &beta, dC, cType, cColumns);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);
            data->elapsedSgemm[device] = elapsed;

            cudaEventRecord(start, 0);
            kCalculateOutput((float*)dC, dScores, dIndexes, cRows, cColumns, paddedRows, maxK);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);
            data->elapsedOutput[device] = elapsed;

            cudaMemcpy(hScores, dScores, hResultScores.getSizeInBytes(), cudaMemcpyDeviceToHost);
            cudaMemcpy(hIndexes, dIndexes, hResultIndexes.getSizeInBytes(), cudaMemcpyDeviceToHost);

            allScores[device] = hScores;
            allIndexes[device] = hIndexes;
        }

        mergeKnn(k, batchSize, maxK, numGpus, allScores, allIndexes, data->hKeys, scores, keys);
    }

    /**
     * @brief Merges the results from multiple GPUs into the final k-nearest neighbors.
     * @param k Number of nearest neighbors to retrieve.
     * @param batchSize Batch size.
     * @param width Width of the results.
     * @param numGpus Number of GPUs.
     * @param allScores Array of score pointers from all GPUs.
     * @param allIndexes Array of index pointers from all GPUs.
     * @param allKeys Vector of key vectors from all GPUs.
     * @param scores Pointer to store the final scores.
     * @param keys Pointer to store the final keys.
     */
    void mergeKnn(int k, int batchSize, int width, int numGpus, const std::array<float*, numGpus>& allScores,
        const std::array<uint32_t*, numGpus>& allIndexes, const std::vector<std::vector<std::string>>& allKeys, float* scores,
        std::string* keys) {

        #pragma omp parallel for
        for (int i = 0; i < batchSize; ++i) {
            std::array<int, numGpus> posIdxs{};
            for (int n = 0; n < numGpus; n++) {
                posIdxs[n] = i * width;
            }
            for (int col = 0; col < k; col++) {
                int deviceId_0 = 0;
                int posIdx_0 = posIdxs[deviceId_0];
                float maxVal = allScores[deviceId_0][posIdx_0];
                uint32_t maxIdx = allIndexes[deviceId_0][posIdx_0];
                int maxDeviceId = deviceId_0;

                // Find the maximum score and its index across all GPUs
                for (int deviceId = 0; deviceId < numGpus; deviceId++) {
                    int posIdx = posIdxs[deviceId];
                    if (maxVal < allScores[deviceId][posIdx]) {
                        maxVal = allScores[deviceId][posIdx];
                        maxIdx = allIndexes[deviceId][posIdx];
                        maxDeviceId = deviceId;
                    }
                }
                ++posIdxs[maxDeviceId];
                scores[i * k + col] = maxVal;
                keys[i * k + col] = allKeys[maxDeviceId][maxIdx];
            }
        }
    }

} // namespace knn
} // namespace astdl
