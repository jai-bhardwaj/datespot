#include <iostream>
#include <algorithm>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <memory>

#include "Generator.h"
#include "GpuTypes.h"
#include "Utils.h"
#include "Filters.h"

using std::string;
using std::vector;
using std::unique_ptr;
using std::tie;
using std::chrono::steady_clock;

/**
 * @brief Class for generating recommendations.
 */
class RecsGenerator {
public:
    /**
     * @brief Default layer label for recommendations generation.
     */
    static constexpr string_view DEFAULT_LAYER_RECS_GEN_LABEL = "Output";

    /**
     * @brief Default score precision for recommendations generation.
     */
    static constexpr string_view DEFAULT_SCORE_PRECISION = "4.3f";

    /**
     * @brief Scalar value for output.
     */
    static constexpr unsigned int Output_SCALAR = 5;

    /**
     * @brief Constructs a RecsGenerator object.
     * @param xBatchSize The batch size.
     * @param xK The number of recommendations to generate.
     * @param xOutputBufferSize The size of the output buffer.
     * @param layer The label of the layer for recommendations generation.
     * @param precision The score precision for recommendations.
     */
    RecsGenerator(unsigned int xBatchSize, unsigned int xK, unsigned int xOutputBufferSize, const string& layer, const string& precision);

    /**
     * @brief Generates recommendations based on the given network and filters.
     * @param xNetwork The network.
     * @param xK The number of recommendations to generate.
     * @param xFilterSet The filter configuration.
     * @param xCustomerIndex The customer index.
     * @param xFeatureIndex The feature index.
     */
    void generateRecs(Network* xNetwork, unsigned int xK, const FilterConfig* xFilterSet, const vector<string>& xCustomerIndex, const vector<string>& xFeatureIndex);

private:
    unique_ptr<GpuBuffer<Float>> pbKey;            ///< GpuBuffer for recommendation keys.
    unique_ptr<GpuBuffer<unsigned int>> pbUIValue;   ///< GpuBuffer for recommendation UI values.
    unique_ptr<GpuBuffer<Float>> pFilteredOutput;  ///< GpuBuffer for filtered output.
    string recsGenLayerLabel;                        ///< Layer label for recommendations generation.
    string scorePrecision;                            ///< Score precision for recommendations.

};

// Implementation of RecsGenerator methods...

/**
 * @brief Constructs a RecsGenerator object.
 * @param xBatchSize The batch size.
 * @param xK The number of recommendations to generate.
 * @param xOutputBufferSize The size of the output buffer.
 * @param layer The label of the layer for recommendations generation.
 * @param precision The score precision for recommendations.
 */
RecsGenerator::RecsGenerator(unsigned int xBatchSize, unsigned int xK, unsigned int xOutputBufferSize, const string& layer, const string& precision)
    : pbKey(make_unique<GpuBuffer<Float>>(xBatchSize * xK * Output_SCALAR, true)),
      pbUIValue(make_unique<GpuBuffer<unsigned int>>(xBatchSize * xK * Output_SCALAR, true)),
      pFilteredOutput(make_unique<GpuBuffer<Float>>(xOutputBufferSize, true)),
      recsGenLayerLabel(layer),
      scorePrecision(precision)
{
}

/**
 * @brief Generates recommendations based on the given network and filters.
 * @param xNetwork The network.
 * @param xK The number of recommendations to generate.
 * @param xFilterSet The filter configuration.
 * @param xCustomerIndex The customer index.
 * @param xFeatureIndex The feature index.
 */
void RecsGenerator::generateRecs(Network* xNetwork, unsigned int xK, const FilterConfig* xFilterSet, const vector<string>& xCustomerIndex, const vector<string>& xFeatureIndex)
{
    int lBatch = xNetwork->GetBatch();
    int lExamples = xNetwork->GetExamples();
    int lPosition = xNetwork->GetPosition();
    if (lPosition + lBatch > lExamples)
    {
        lBatch = lExamples - lPosition;
    }

    bool bMultiGPU = (getGpu()._numprocs > 1);
    unique_ptr<GpuBuffer<Float>> pbMultiKey;
    unique_ptr<GpuBuffer<unsigned int>> pbMultiUIValue;
    unique_ptr<GpuBuffer<unsigned int>> pbUIValueCache;
    Float* pMultiKey = nullptr;
    unsigned int* pMultiUIValue = nullptr;
    unsigned int* pUIValueCache = nullptr;

    cudaIpcMemHandle_t keyMemHandle;
    cudaIpcMemHandle_t valMemHandle;
    const Float* dOutput = xNetwork->GetUnitBuffer(recsGenLayerLabel);
    const Layer* pLayer = xNetwork->GetLayer(recsGenLayerLabel);
    unsigned int lx, ly, lz, lw;
    tie(lx, ly, lz, lw) = pLayer->GetDimensions();
    int lOutputStride = lx * ly * lz * lw;
    unsigned int llx, lly, llz, llw;
    tie(llx, lly, llz, llw) = pLayer->GetLocalDimensions();
    int lLocalOutputStride = llx * lly * llz * llw;
    unsigned int outputBufferSize = lLocalOutputStride * lBatch;
    if (!bMultiGPU)
    {
        outputBufferSize = xNetwork->GetBufferSize(recsGenLayerLabel);
    }

    vector<float> hOutputBuffer(outputBufferSize);

    if (bMultiGPU)
    {
        if (getGpu()._id == 0)
        {
            const size_t bufferSize = getGpu()._numprocs * lBatch * xK * Output_SCALAR;
            pbMultiKey = make_unique<GpuBuffer<Float>>(bufferSize, true);
            pbMultiUIValue = make_unique<GpuBuffer<unsigned int>>(bufferSize, true);
            pMultiKey = pbMultiKey->_pDevData;
            pMultiUIValue = pbMultiUIValue->_pDevData;
            cudaError_t status = cudaIpcGetMemHandle(&keyMemHandle, pMultiKey);
            RTERROR(status, "cudaIpcGetMemHandle: Failed to get IPC mem handle on pMultiKey");
            status = cudaIpcGetMemHandle(&valMemHandle, pMultiUIValue);
            RTERROR(status, "cudaIpcGetMemHandle: Failed to get IPC mem handle on pMultiUIValue");
        }

        MPI_Bcast(&keyMemHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&valMemHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        if (getGpu()._id != 0)
        {
            cudaError_t status = cudaIpcOpenMemHandle(reinterpret_cast<void**>(&pMultiKey), keyMemHandle, cudaIpcMemLazyEnablePeerAccess);
            RTERROR(status, "cudaIpcOpenMemHandle: Unable to open key IPCMemHandle");
            status = cudaIpcOpenMemHandle(reinterpret_cast<void**>(&pMultiUIValue), valMemHandle, cudaIpcMemLazyEnablePeerAccess);
            RTERROR(status, "cudaIpcOpenMemHandle: Unable to open value IPCMemHandle");
        }
    }

    cudaMemcpy(hOutputBuffer.data(), dOutput, outputBufferSize * sizeof(Float), cudaMemcpyDeviceToHost);

    const auto start = steady_clock::now();
    for (int j = 0; j < lBatch; j++)
    {
        int sampleIndex = lPosition + j;
        int offset = getGpu()._id * lLocalOutputStride;
        xFilterSet->applySamplesFilter(hOutputBuffer.data() + j * lLocalOutputStride, sampleIndex, offset, lLocalOutputStride);
    }

    pFilteredOutput->Upload(hOutputBuffer.data());
    kCalculateOutput(pFilteredOutput->_pDevData, pbKey->_pDevData, pbUIValue->_pDevData, lBatch, lLocalOutputStride, xK * Output_SCALAR);

    if (bMultiGPU)
    {
        uint32_t offset = xK * Output_SCALAR * getGpu()._id;
        uint32_t kstride = xK * Output_SCALAR * getGpu()._numprocs;
        cudaMemcpy2D(pMultiKey + offset, kstride * sizeof(Float), pbKey->_pDevData, xK * Output_SCALAR * sizeof(Float), xK * Output_SCALAR * sizeof(Float), lBatch, cudaMemcpyDefault);
        cudaMemcpy2D(pMultiUIValue + offset, kstride * sizeof(unsigned int), pbUIValue->_pDevData, xK * Output_SCALAR * sizeof(unsigned int), xK * Output_SCALAR * sizeof(unsigned int), lBatch, cudaMemcpyDefault);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        if (getGpu()._id == 0)
        {
            kCalculateOutput(pbMultiKey->_pDevData, pbKey->_pDevData, pbUIValue->_pDevData, lBatch, kstride, xK * Output_SCALAR);
            pbUIValueCache = make_unique<GpuBuffer<unsigned int>>(getGpu()._numprocs * lBatch * xK * Output_SCALAR, true);
            kCalculateOutput(pbMultiKey->_pDevData, pbMultiUIValue->_pDevData, pbKey->_pDevData, pbUIValueCache->_pDevData, lBatch, kstride, xK * Output_SCALAR);
        }
    }

    if (getGpu()._id == 0)
    {
        const char* fileName = xFilterSet->getOutputFileName().c_str();
        const auto now = steady_clock::now();
        std::cout << "Time Elapsed for Filtering and selecting Top " << xK << " recs: " << elapsed_seconds(start, now) << std::endl;
        std::cout << "Writing to " << fileName << std::endl;
        std::ofstream outputFile(fileName, std::ios::app);
        pbKey->Download();
        pbUIValue->Download();
        Float* pKey = pbKey->_pSysData;
        unsigned int* pIndex = pbUIValue->_pSysData;

        if (bMultiGPU)
        {
            pbUIValueCache->Download();
            pUIValueCache = pbUIValueCache->_pSysData;
        }

        string strFormat = "%s,%" + scorePrecision + ":";
        for (int j = 0; j < lBatch; j++)
        {
            outputFile << xCustomerIndex[lPosition + j] << '\t';
            for (int x = 0; x < xK; ++x)
            {
                const size_t bufferPos = j * xK * Output_SCALAR + x;
                int finalIndex = pIndex[bufferPos];
                float value = pKey[bufferPos];
                if (bMultiGPU)
                {
                    int gpuId = finalIndex / (xK * Output_SCALAR);
                    int localIndex = pUIValueCache[bufferPos];
                    int globalIndex = gpuId * lLocalOutputStride + localIndex;
                    if (globalIndex < xFeatureIndex.size())
                    {
                        outputFile << string_format(strFormat, xFeatureIndex[globalIndex], value);
                    }
                }
                else if (finalIndex < xFeatureIndex.size())
                {
                    outputFile << string_format(strFormat, xFeatureIndex[finalIndex], value);
                }
            }
            outputFile << std::endl;
        }
        outputFile.close();
        const auto end = steady_clock::now();
        std::cout << "Time Elapsed for Writing to file: " << elapsed_seconds(start, end) << std::endl;
    }

    if (bMultiGPU)
    {
        if (getGpu()._id != 0)
        {
            cudaError_t status = cudaIpcCloseMemHandle(pMultiKey);
            RTERROR(status, "cudaIpcCloseMemHandle: Error closing MultiKey IpcMemHandle");
            status = cudaIpcCloseMemHandle(pMultiUIValue);
            RTERROR(status, "cudaIpcCloseMemHandle: Error closing MultiFValue IpcMemHandle");
        }
    }
}
