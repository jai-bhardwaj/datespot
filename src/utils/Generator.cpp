#include <cstdio>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <vector>
#include <string>
#include <cstdio>
#include <chrono>

#include "Generator.h"
#include "GpuTypes.h"
#include "Utils.h"
#include "Filters.h"

/**
 * @brief Default label for the recs generator layer.
 */
const string RecsGenerator::DEFAULT_LAYER_RECS_GEN_LABEL = "Output";

/**
 * @brief Default precision for the score value.
 */
const string RecsGenerator::DEFAULT_SCORE_PRECISION = "4.3f";

/**
 * @brief Scalar value for the output.
 */
const unsigned int RecsGenerator::Output_SCALAR = 5;

/**
 * @brief Constructor for RecsGenerator.
 *
 * This constructor initializes the RecsGenerator object with the specified batch size,
 * top record count (K), output buffer size, layer label, and score precision.
 *
 * @param xBatchSize The batch size.
 * @param xK The number of top records to select.
 * @param xOutputBufferSize The size of the output buffer.
 * @param layer The label of the recs generator layer.
 * @param precision The precision for the score value.
 */
RecsGenerator::RecsGenerator(unsigned int xBatchSize,
                             unsigned int xK,
                             unsigned int xOutputBufferSize,
                             const string &layer,
                             const string &precision)
    : pbKey(new GpuBuffer<NNFloat>(xBatchSize * xK * Output_SCALAR, true)),
      pbUIValue(new GpuBuffer<unsigned int>(xBatchSize * xK * Output_SCALAR, true)),
      pFilteredOutput(new GpuBuffer<NNFloat>(xOutputBufferSize, true)),
      recsGenLayerLabel(layer),
      scorePrecision(precision)
{
}

/**
 * @brief Generate recommendations using the specified network and filter settings.
 *
 * This function generates recommendations using the specified network and filter settings.
 * It takes into account the batch size, examples, position, and the availability of multiple GPUs.
 *
 * @param xNetwork Pointer to the network object.
 * @param xK Number of top records to select.
 * @param xFilterSet Pointer to the filter set configuration.
 * @param xCustomerIndex The customer index.
 * @param xFeatureIndex The feature index.
 */
void RecsGenerator::generateRecs(Network *xNetwork,
                                unsigned int xK,
                                const FilterConfig *xFilterSet,
                                const vector<string> &xCustomerIndex,
                                const vector<string> &xFeatureIndex)
{
    int lBatch    = xNetwork->GetBatch();
    int lExamples = xNetwork->GetExamples();
    int lPosition = xNetwork->GetPosition();
    if (lPosition + lBatch > lExamples)
    {
        lBatch = lExamples - lPosition;
    }

    bool bMultiGPU                           = (getGpu()._numprocs > 1);
    unique_ptr<GpuBuffer<NNFloat>> pbMultiKey;
    unique_ptr<GpuBuffer<unsigned int>> pbMultiUIValue;
    unique_ptr<GpuBuffer<unsigned int>> pbUIValueCache;
    NNFloat* pMultiKey                       = NULL;
    unsigned int* pMultiUIValue              = NULL;
    unsigned int* pUIValueCache              = NULL;

    cudaIpcMemHandle_t keyMemHandle;
    cudaIpcMemHandle_t valMemHandle;
    const NNFloat* dOutput         = xNetwork->GetUnitBuffer(recsGenLayerLabel);
    const Layer* pLayer          = xNetwork->GetLayer(recsGenLayerLabel);
    unsigned int lx, ly, lz, lw;
    tie(lx, ly, lz, lw)            = pLayer->GetDimensions();
    int lOutputStride              = lx * ly * lz * lw;
    unsigned int llx, lly, llz, llw;
    tie(llx, lly, llz, llw)        = pLayer->GetLocalDimensions();

    int lLocalOutputStride         = llx * lly * llz * llw;
    unsigned int outputBufferSize  = lLocalOutputStride * lBatch;
    if (!bMultiGPU)
    {
        outputBufferSize = xNetwork->GetBufferSize(recsGenLayerLabel);
    }

    unique_ptr<float[]> hOutputBuffer(new float[outputBufferSize]);
    
    /**
     * @brief Perform multi-GPU operations and calculate output.
     *
     * If bMultiGPU is true, this function performs multi-GPU operations and calculates
     * the output. It handles the allocation and transfer of memory buffers, IPC memory
     * handle operations, and data synchronization between GPUs using MPI.
     *
     * If the current GPU ID is 0, it allocates and initializes the multi-GPU buffers,
     * obtains IPC memory handles for pMultiKey and pMultiUIValue, and broadcasts the
     * memory handles to other GPUs using MPI. For other GPUs, it opens the IPC memory
     * handles using cudaIpcOpenMemHandle.
     *
     * After the multi-GPU operations, it transfers the output from the device to the
     * host memory buffer hOutputBuffer. Then, for each sample in the batch, it applies
     * the filter set to the corresponding output values. Finally, it uploads the filtered
     * output to the device and calculates the final output using pbKey and pbUIValue.
     *
     * @param bMultiGPU Flag indicating whether multiple GPUs are used.
     * @param xK Number of top records to select.
     * @param pbKey Pointer to the key memory buffer.
     * @param pbUIValue Pointer to the UIValue memory buffer.
     * @param pbMultiKey Pointer to the multi-GPU key memory buffer.
     * @param pbMultiUIValue Pointer to the multi-GPU UIValue memory buffer.
     * @param lBatch The batch size.
     * @param lPosition The position in the customer index.
     * @param lLocalOutputStride The stride for local output.
     * @param hOutputBuffer Pointer to the host output buffer.
     * @param xFilterSet Pointer to the filter set.
     * @param pFilteredOutput Pointer to the filtered output memory buffer.
     * @param dOutput Pointer to the device output buffer.
     */
    if (bMultiGPU)
    {
        if (getGpu()._id == 0)
        {
            const size_t bufferSize = getGpu()._numprocs * lBatch * xK * Output_SCALAR;

            pbMultiKey.reset(new GpuBuffer<NNFloat>(bufferSize, true));
            pbMultiUIValue.reset(new GpuBuffer<unsigned int>(bufferSize, true));

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
            cudaError_t status = cudaIpcOpenMemHandle((void**)&pMultiKey, keyMemHandle, cudaIpcMemLazyEnablePeerAccess);
            RTERROR(status, "cudaIpcOpenMemHandle: Unable to open key IPCMemHandle");
            status = cudaIpcOpenMemHandle((void**)&pMultiUIValue, valMemHandle, cudaIpcMemLazyEnablePeerAccess);
            RTERROR(status, "cudaIpcOpenMemHandle: Unable to open value IPCMemHandle");
        }

    }
    cudaMemcpy(hOutputBuffer.get(), dOutput, outputBufferSize * sizeof(NNFloat), cudaMemcpyDeviceToHost);

    auto const start = std::chrono::steady_clock::now();
    for (int j = 0; j < lBatch; j++)
    {
        int sampleIndex = lPosition + j;

        int offset = getGpu()._id * lLocalOutputStride;
        xFilterSet->applySamplesFilter(hOutputBuffer.get() + j * lLocalOutputStride, sampleIndex, offset, lLocalOutputStride);
    }

    pFilteredOutput->Upload(hOutputBuffer.get());
    kCalculateOutput(pFilteredOutput->_pDevData, pbKey->_pDevData, pbUIValue->_pDevData, lBatch, lLocalOutputStride, xK * Output_SCALAR);

    /**
     * @brief Perform data transfer and synchronization for multi-GPU processing.
     *
     * If bMultiGPU is true, this function performs data transfer and synchronization
     * for multi-GPU processing. It transfers the key and UIValue data from the device
     * buffers pbKey and pbUIValue to the multi-GPU buffers pMultiKey and pMultiUIValue.
     * It then synchronizes the CUDA devices and waits for the MPI barrier.
     *
     * If the current GPU ID is 0, it also performs additional calculations and allocates
     * memory for the UIValue cache.
     *
     * @param bMultiGPU Flag indicating whether multiple GPUs are used.
     * @param xK Number of top records to select.
     * @param pbKey Pointer to the key memory buffer.
     * @param pbUIValue Pointer to the UIValue memory buffer.
     * @param pMultiKey Pointer to the multi-GPU key memory buffer.
     * @param pMultiUIValue Pointer to the multi-GPU UIValue memory buffer.
     * @param pbMultiKey Pointer to the multi-GPU key memory buffer.
     * @param pbMultiUIValue Pointer to the multi-GPU UIValue memory buffer.
     * @param lBatch The batch size.
     */
    if (bMultiGPU)
    {
        uint32_t offset = xK * Output_SCALAR * getGpu()._id;
        uint32_t kstride = xK * Output_SCALAR * getGpu()._numprocs;
        cudaMemcpy2D(pMultiKey + offset, kstride * sizeof(NNFloat), pbKey->_pDevData, xK * Output_SCALAR * sizeof(NNFloat), xK * Output_SCALAR * sizeof(NNFloat), lBatch, cudaMemcpyDefault);
        cudaMemcpy2D(pMultiUIValue + offset, kstride * sizeof(unsigned int), pbUIValue->_pDevData, xK * Output_SCALAR * sizeof(unsigned int), xK * Output_SCALAR * sizeof(unsigned int), lBatch, cudaMemcpyDefault);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        if (getGpu()._id == 0)
        {
            kCalculateOutput(pbMultiKey->_pDevData, pbKey->_pDevData, pbUIValue->_pDevData, lBatch, kstride, xK * Output_SCALAR);

            pbUIValueCache.reset(new GpuBuffer<unsigned int>(getGpu()._numprocs * lBatch * xK * Output_SCALAR, true));

            kCalculateOutput(pbMultiKey->_pDevData, pbMultiUIValue->_pDevData, pbKey->_pDevData, pbUIValueCache->_pDevData, lBatch, kstride, xK * Output_SCALAR);
        }
    }

    /**
     * @brief Write filtered and selected records to a file and print time elapsed.
     *
     * If the GPU ID is 0, this function writes the filtered and selected records to
     * a file specified by xFilterSet's output file name. It also prints the time elapsed
     * for filtering and selecting the top xK records and the time elapsed for writing
     * to the file.
     *
     * @param xFilterSet Pointer to the filter set.
     * @param xK Number of top records to select.
     * @param lPosition The position in the customer index.
     * @param lBatch The batch size.
     * @param bMultiGPU Flag indicating whether multiple GPUs are used.
     * @param pbKey Pointer to the key memory buffer.
     * @param pbUIValue Pointer to the UIValue memory buffer.
     * @param pbUIValueCache Pointer to the UIValue cache memory buffer.
     * @param xCustomerIndex The customer index.
     * @param xFeatureIndex The feature index.
     * @param xLocalOutputStride The stride for local output.
     * @param scorePrecision The precision of the score value.
     */
    if (getGpu()._id == 0)
    {
        const char *fileName = xFilterSet->getOutputFileName().c_str();
        auto const now = std::chrono::steady_clock::now();
        std::cout << "Time Elapsed for Filtering and selecting Top " << xK << " recs: " << elapsed_seconds(start, now) << endl;
        std::cout << "Writing to " << fileName << endl;
        FILE *fp = fopen(fileName, "a");
        pbKey->Download();
        pbUIValue->Download();
        NNFloat* pKey                   = pbKey->_pSysData;
        unsigned int* pIndex            = pbUIValue->_pSysData;

        if (bMultiGPU)
        {
            pbUIValueCache->Download();
            pUIValueCache               = pbUIValueCache->_pSysData;
        }

        string strFormat = "%s,%" + scorePrecision + ":";
        for (int j = 0; j < lBatch; j++)
        {
            fprintf(fp, "%s%c", xCustomerIndex[lPosition + j].c_str(), '\t');
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
                        fprintf(fp, strFormat.c_str(), xFeatureIndex[globalIndex].c_str(), value);
                    } 
                }
                else if (finalIndex < xFeatureIndex.size())
                {
                    fprintf(fp, strFormat.c_str(), xFeatureIndex[finalIndex].c_str(), value);
                }
            }

            fprintf(fp, "\n");
        }
        fclose(fp);
        auto const end = std::chrono::steady_clock::now();
        std::cout << "Time Elapsed for Writing to file: " << elapsed_seconds(start, end) << endl;
    }

    /**
     * @brief Close the CUDA IPC memory handles for MultiKey and MultiFValue.
     *
     * If bMultiGPU is true and the current GPU is not the first GPU, this function
     * closes the CUDA IPC memory handles for pMultiKey and pMultiUIValue.
     *
     * @param bMultiGPU Flag indicating whether multiple GPUs are used.
     * @param pMultiKey Pointer to the MultiKey memory handle.
     * @param pMultiUIValue Pointer to the MultiFValue memory handle.
     */
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
