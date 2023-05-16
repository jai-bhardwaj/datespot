#ifndef __CALCULATE_H__
#define __CALCULATE_H__

#include <algorithm>
#include <optional>
#include <cstdint>
#include <vector>
#include <array>
#include <string>
#include <memory>
#include <stdexcept>
/**
 * Predicts the output of a network based on the given inputs and calculates precision, recall, and NDCG.
 *
 * @param pNetwork Pointer to the Network object.
 * @param vDataSet Vector of DataSetBase pointers.
 * @param cdl Reference to the CDL object.
 * @param K The number of top-K predictions to consider.
 * @return A PyObject pointer representing the predicted output as a Python list.
 */
static PyObject* tensorhubcalculate_PredictOutput(Network* pNetwork, std::vector<DataSetBase*>& vDataSet, CDL& cdl, unsigned int K) {
    bool bFilterPast = false;
    const auto pLayer = pNetwork->GetLayer("Output");
    uint32_t Nx, Ny, Nz, Nw;
    std::tie(Nx, Ny, Nz, Nw) = pLayer->GetLocalDimensions();
    const uint32_t STRIDE = Nx * Ny * Nz * Nw;

    /**
     * Lambda function to find a dataset with the given name.
     *
     * @param name The name of the dataset to find.
     * @return The index of the dataset in the vector vDataSet.
     */
    auto findDataSet = [&vDataSet](const std::string& name) {
        auto it = std::find_if(vDataSet.begin(), vDataSet.end(),
            [&name](const auto& dataSet) { return dataSet->_name == name; });
        if (it == vDataSet.end()) {
            printf("Unable to find %s dataset, exiting.\n", name.c_str());
            exit(-1);
        }
        return it - vDataSet.begin();
    };

    size_t inputIndex = findDataSet("input");
    size_t outputIndex = findDataSet("output");

    int cdlBatch = cdl._batch;

    std::vector<NNFloat> vPrecision(K);
    std::vector<NNFloat> vRecall(K);
    std::vector<NNFloat> vNDCG(K);
    std::vector<uint32_t> vDataPoints(cdlBatch);
    auto pbTarget = std::make_unique<GpuBuffer<NNFloat>>(cdlBatch * STRIDE, true);
    auto pbOutput = std::make_unique<GpuBuffer<NNFloat>>(cdlBatch * STRIDE, true);
    auto pInputDataSet = dynamic_cast<DataSet<NNFloat>*>(vDataSet[inputIndex]);
    auto pOutputDataSet = dynamic_cast<DataSet<NNFloat>*>(vDataSet[outputIndex]);
    auto pbKey = std::make_unique<GpuBuffer<NNFloat>>(cdlBatch * K, true);
    auto pbUIValue = std::make_unique<GpuBuffer<unsigned int>>(cdlBatch * K, true);
    auto pbFValue = std::make_unique<GpuBuffer<NNFloat>>(cdlBatch * K, true);
    NNFloat* pOutputValue = pbOutput->_pSysData;
    bool bMultiGPU = (getGpu()._numprocs > 1);
    std::optional<std::unique_ptr<GpuBuffer<NNFloat>>> pbMultiKey;
    std::optional<std::unique_ptr<GpuBuffer<NNFloat>>> pbMultiFValue;
    NNFloat* pMultiKey = nullptr;
    NNFloat* pMultiFValue = nullptr;
    cudaIpcMemHandle_t keyMemHandle;
    cudaIpcMemHandle_t valMemHandle;

    if (bMultiGPU)
    {
        if (getGpu()._id == 0)
        {
            pbMultiKey = std::make_unique<GpuBuffer<NNFloat>>(getGpu()._numprocs * cdlBatch * K, true);
            pbMultiFValue = std::make_unique<GpuBuffer<NNFloat>>(getGpu()._numprocs * cdlBatch * K, true);
            pMultiKey = pbMultiKey->_pDevData;
            pMultiFValue = pbMultiFValue->_pDevData;

            cudaError_t status = cudaIpcGetMemHandle(&keyMemHandle, pMultiKey);
            RTERROR(status, "cudaIpcGetMemHandle: Failed to get IPC mem handle on pMultiKey");

            status = cudaIpcGetMemHandle(&valMemHandle, pMultiFValue);
            RTERROR(status, "cudaIpcGetMemHandle: Failed to get IPC mem handle on pMultiFValue");
        }

        /**
         * Broadcasts the CUDA IPC memory handles to all GPUs in the multi-GPU setup.
         */
        MPI_Bcast(&keyMemHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&valMemHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        if (getGpu()._id != 0)
        {
            cudaError_t status = cudaIpcOpenMemHandle((void**)&pMultiKey, keyMemHandle, cudaIpcMemLazyEnablePeerAccess);
            RTERROR(status, "cudaIpcOpenMemHandle: Unable to open key IPCMemHandle");

            status = cudaIpcOpenMemHandle((void**)&pMultiFValue, valMemHandle, cudaIpcMemLazyEnablePeerAccess);
            RTERROR(status, "cudaIpcOpenMemHandle: Unable to open value IPCMemHandle");
        }
    }

    for (unsigned long long int pos = 0; pos < pNetwork->GetExamples(); pos += pNetwork->GetBatch())
    {
        pNetwork->SetPosition(pos);
        pNetwork->PredictBatch();

        auto batch = std::min(pNetwork->GetBatch(), pNetwork->GetExamples() - pos);
        auto pTarget = pbTarget->_pSysData;
        
        std::fill_n(pTarget, STRIDE * batch, 0);            
        const auto pOutputKey = pNetwork->GetUnitBuffer("Output");
        auto pOut = pOutputValue;
        
        cudaError_t status = cudaMemcpy(pOut, pOutputKey, batch * STRIDE * sizeof(NNFloat), cudaMemcpyDeviceToHost);
        RTERROR(status, "cudaMemcpy GpuBuffer::Download failed");
            
        for (unsigned int i = 0; i < batch; ++i)
        {
            auto j = pos + i;
            vDataPoints[i] = pOutputDataSet->_vSparseEnd[j] - pOutputDataSet->_vSparseStart[j];
                
            for (auto k = pOutputDataSet->_vSparseStart[j]; k < pOutputDataSet->_vSparseEnd[j]; ++k)
            {
                pTarget[pOutputDataSet->_vSparseIndex[k]] = 1.0f;
            }
                
            if (bFilterPast)
            {
                for (auto k = pInputDataSet->_vSparseStart[j]; k < pInputDataSet->_vSparseEnd[j]; ++k)
                {
                    pOut[pInputDataSet->_vSparseIndex[k]] = 0.0f;
                }
            }

            std::advance(pTarget, STRIDE);
            std::advance(pOut, STRIDE);
        }
        
        pbTarget->Upload();
        pbOutput->Upload();
        kCalculateOutput(pbOutput->_pDevData, pbTarget->_pDevData, pbKey->_pDevData, pbFValue->_pDevData, batch, STRIDE, K);
        pbKey->Download();
        pbFValue->Download();
            
        if (bMultiGPU)
        {
            MPI_Reduce((getGpu()._id == 0) ? MPI_IN_PLACE : vDataPoints.data(), vDataPoints.data(), batch, MPI_UINT32_T, MPI_SUM, 0, MPI_COMM_WORLD);
            
            const auto offset = K * getGpu()._id;
            const auto kstride = K * getGpu()._numprocs;
            const auto copy_size = K * sizeof(NNFloat);

            cudaMemcpy2D(pMultiKey + offset, kstride * sizeof(NNFloat), pbKey->_pDevData, copy_size, copy_size, batch, cudaMemcpyDefault);
            cudaMemcpy2D(pMultiFValue + offset, kstride * sizeof(NNFloat), pbFValue->_pDevData, copy_size, copy_size, batch, cudaMemcpyDefault);
            
            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);
            
            if (getGpu()._id == 0)
            {
                kCalculateOutput(pbMultiKey->_pDevData, pbMultiFValue->_pDevData, pbKey->_pDevData, pbFValue->_pDevData, batch, getGpu()._numprocs * K, K);
            }
        }

        if (getGpu()._id == 0)
        {
            pbKey->Download();
            pbFValue->Download();
            auto* pKey = pbKey->_pSysData;
            auto* pValue = pbFValue->_pSysData;
            for (unsigned int i = 0; i < batch; i++)
            {
                const auto p = static_cast<NNFloat>(vDataPoints[i]);
                auto tp = 0.0f;
                auto fp = 0.0f;
                auto idcg = 0.0f;
                for (auto pp = 0.0f; pp < p; pp++)
                {
                    idcg += 1.0f / log2(pp + 2.0f);
                }
                auto dcg = 0.0f;
                for (unsigned int j = 0; j < K; j++)
                {
                    if (pValue[j] == 1.0f)
                    {
                        tp++;
                        dcg += 1.0f / log2(static_cast<float>(j + 2));
                    }
                    else
                    {
                        fp++;
                    }
                    vPrecision[j] += tp / (tp + fp);
                    vRecall[j] += tp / p;
                    vNDCG[j] += dcg / idcg;
                }
                pKey += K;
                pValue += K;
            }
        }

    std::vector<GpuBuffer<NNFloat>*> gpuBuffers = {pbKey, pbFValue, pbUIValue, pbTarget, pbOutput};
    for (auto* buffer : gpuBuffers) {
        delete buffer;
    }
        
    if (bMultiGPU)
    {
        if (getGpu()._id != 0)
        {
            const auto closeMemHandle = [](auto& handle, const char* errorMessage) {
                const auto status = cudaIpcCloseMemHandle(handle);
                RTERROR(status, errorMessage);
            };

            closeMemHandle(pMultiKey, "cudaIpcCloseMemHandle: Error closing MultiKey IpcMemHandle");
            closeMemHandle(pMultiFValue, "cudaIpcCloseMemHandle: Error closing MultiFValue IpcMemHandle");
        }

        std::vector<GpuBuffer<NNFloat>*> gpuBuffers = {pbMultiKey, pbMultiFValue};
        for (auto* buffer : gpuBuffers) {
            delete buffer;
        }
    }

    auto python_list = PyList_New(K);
    if (getGpu()._id == 0)
    {
        const auto examples = static_cast<float>(pNetwork->GetExamples());
        for (unsigned int i = 0; i < K; i++)
        {
            PyList_SetItem(python_list, i, Py_BuildValue("[fff]",
                                                        vPrecision[i] / examples,
                                                        vRecall[i] / examples,
                                                        vNDCG[i] / examples));
        }
    }
    return python_list;
}

/**
 * @brief Transposes the vEmbedding matrix and stores the result in vASINWeight.
 *
 * This function transposes the vEmbedding matrix and stores the result in the vASINWeight matrix.
 *
 * @param vASINWeight Pointer to the vASINWeight PyArrayObject.
 * @param vEmbedding Pointer to the vEmbedding PyArrayObject.
 * @return Py_RETURN_NONE on success, or NULL on failure.
 */
static PyObject* tensorhubcalculate_Transpose(PyArrayObject* vASINWeight, PyArrayObject* vEmbedding) {
    CheckNumPyArray(vASINWeight);
    CheckNumPyArray(vEmbedding);

    const std::array<npy_intp, 2> vASINWeightDims = {PyArray_DIM(vASINWeight, 0), PyArray_DIM(vASINWeight, 1)};
    const std::array<npy_intp, 2> vEmbeddingDims = {PyArray_DIM(vEmbedding, 0), PyArray_DIM(vEmbedding, 1)};

    if (vASINWeightDims[0] != 2) {
        const std::string message = "Normalize received incorrect vASINWeight matrix dimensionality; expected = 2  received = " +
            std::to_string(vASINWeightDims[0]);
        PyErr_SetString(PyExc_RuntimeError, message.c_str());
        return nullptr;
    }
    if (vEmbeddingDims[0] != 2) {
        const std::string message = "Normalize received incorrect vEmbedding matrix dimensionality; expected = 2  received = " +
            std::to_string(vEmbeddingDims[0]);
        PyErr_SetString(PyExc_RuntimeError, message.c_str());
        return nullptr;
    }
    if (vASINWeightDims[0] > vEmbeddingDims[1] || vASINWeightDims[1] > vEmbeddingDims[0]) {
        const std::string message = "Normalize received vASINWeight dimensions that exceed vEmbedding dimensions; vASINWeight dimensions = (" +
            std::to_string(vASINWeightDims[0]) + ", " + std::to_string(vASINWeightDims[1]) + ")  vEmbedding dimensions = (" +
            std::to_string(vEmbeddingDims[0]) + ", " + std::to_string(vEmbeddingDims[1]) + ")";
        PyErr_SetString(PyExc_RuntimeError, message.c_str());
        return nullptr;
    }

    auto* pASINWeight = reinterpret_cast<NNFloat*>(PyArray_DATA(vASINWeight));
    auto* pEmbedding = reinterpret_cast<NNFloat*>(PyArray_DATA(vEmbedding));
    const npy_intp STRIDE = vASINWeightDims[0];
    const npy_intp EmbeddingExamples = vASINWeightDims[1];

    for (int i = 0; i < EmbeddingExamples; i++) {
        const auto embeddingOffset = i * STRIDE;
        const auto asinWeightOffset = i;
        std::copy(pEmbedding + embeddingOffset, pEmbedding + embeddingOffset + STRIDE, pASINWeight + asinWeightOffset);
    }

    Py_RETURN_NONE;
}
/**
 * @brief Calculates the Mean Reciprocal Rank (MRR) for a given network.
 *
 * The function calculates the Mean Reciprocal Rank (MRR) using the given network, data set, and output layer.
 * It iterates over the data set and for each example, it sets the position in the network, predicts the batch,
 * and computes the MRR based on the predictions.
 *
 * @param pNetwork Pointer to the network to be used for prediction.
 * @param vDataSet Vector of pointers to DataSetBase. The data set to use for prediction is indicated by outputIndex.
 * @param pOutputLayer Pointer to the output layer of the network to get the units from.
 * @param outputIndex Index in the vDataSet vector to the data set to be used for prediction.
 * @param NxOutput Number of outputs per example in the data set.
 *
 * @return PyObject* Returns a Python float object representing the calculated MRR.
 */
static PyObject* tensorhubcalculate_CalculateMRR(Network* pNetwork, std::vector<DataSetBase*>& vDataSet, 
                                                  Layer* pOutputLayer, uint32_t outputIndex, uint32_t NxOutput) {
    
    auto pOutputDataSet  = dynamic_cast<DataSet<uint32_t>*>(vDataSet[outputIndex]);
    if (!pOutputDataSet) {
    }
    
    double MRR = 0.0;
    std::vector<double> vOutput(NxOutput * pNetwork->GetBatch());
    
    for (size_t pos = 0; pos < pNetwork->GetExamples(); pos += pNetwork->GetBatch()) {
        pNetwork->SetPosition(pos);
        pNetwork->PredictBatch();    
        
        unsigned int batch = pNetwork->GetBatch();
        if (pos + batch > pNetwork->GetExamples()) {
            batch = pNetwork->GetExamples() - pos;
        }
        
        pOutputLayer->GetUnits(vOutput);

        for (size_t i = 0; i < batch; ++i) {
            size_t offset = i * NxOutput;
            double asinValue = -99999.0;
            
            for (size_t j = pOutputDataSet->_vSparseStart[i + pos]; j < pOutputDataSet->_vSparseEnd[i + pos]; ++j) {
                asinValue = std::max(asinValue, vOutput[offset + pOutputDataSet->_vSparseIndex[j]]);
            }
            
            size_t asinCount = 1;
            auto[minValue, maxValue] = std::minmax_element(vOutput.begin() + offset, vOutput.begin() + offset + NxOutput);
            
            asinCount += std::count_if(vOutput.begin() + offset, vOutput.begin() + offset + NxOutput, 
                                       [asinValue](double val) { return val > asinValue; });
            
            MRR += (1.0 / static_cast<double>(asinCount));
        }
    }
    
    MRR /= static_cast<double>(pNetwork->GetExamples());
    return Py_BuildValue("f", MRR);
}
