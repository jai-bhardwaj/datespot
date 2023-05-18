#include "GpuTypes.h"
#include "Types.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <string>
#include <memory>
#include <filesystem>

namespace fs = std::filesystem;
/**
 * @brief The main function of the program.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line arguments.
 * @return The exit status of the program.
 */
int main(int argc, char** argv)
{
    getGpu().Startup(argc, argv);

    CDL cdl;

    if (argc == 2)
    {
        int err = cdl.Load_JSON(argv[1]);
        if (err != 0)
        {
            std::cerr << "*** Error, " << argv[0] << " could not parse CDC file " << argv[1] << '\n';
            return -1;
        }
    }
    else
    {
        cdl._mode = Prediction;
        cdl._optimizer = TrainingMode::Nesterov;
        cdl._networkFileName = "network.nc";
        cdl._alphaInterval = 20;
        cdl._alphaMultiplier = 0.8f;
        cdl._alpha = 0.025f;
        cdl._lambda = 0.0001f;
        cdl._mu = 0.5f;
        cdl._randomSeed = 12345;
        cdl._epochs = 60;
        cdl._dataFileName = "../../data/data_test.nc";
    }

    getGpu().SetRandomSeed(cdl._randomSeed);

    float lambda1 = 0.0f;
    float mu1 = 0.0f;

    auto pNetwork = std::make_unique<Network>();

    std::vector<std::unique_ptr<DataSetBase>> vDataSet;

    vDataSet = LoadNetCDF(cdl._dataFileName);

#if 0
    auto [cpuMemory, gpuMemory] = vDataSet[0]->getMemoryUsage();
    std::cout << "CPUMem: " << cpuMemory << " GPUMem: " << gpuMemory << '\n';
    exit(-1);
#endif

    if (cdl._mode == Prediction)
        pNetwork = LoadNeuralNetworkNetCDF(cdl._networkFileName, cdl._batch);
    else
        pNetwork = LoadNeuralNetworkJSON(cdl._networkFileName, cdl._batch, vDataSet);

    int totalGPUMemory;
    int totalCPUMemory;
    getGpu().GetMemoryUsage(&totalGPUMemory, &totalCPUMemory);

    std::cout << "GPU Memory Usage: " << totalGPUMemory << " KB\n";
    std::cout << "CPU Memory Usage: " << totalCPUMemory << " KB\n";

    pNetwork->LoadDataSets(vDataSet);

    pNetwork->SetCheckpoint(cdl._checkpointFileName, cdl._checkpointInterval);

    if (cdl._mode == Mode::Validation)
    {
        pNetwork->SetTrainingMode(Nesterov);
        pNetwork->Validate();
    }
    else if (cdl._mode == Training)
    {
        pNetwork->SetTrainingMode(cdl._optimizer);

        float alpha = cdl._alpha;
        int epochs = 0;

        while (epochs < cdl._epochs)
        {
            pNetwork->Train(cdl._alphaInterval, alpha, cdl._lambda, lambda1, cdl._mu, mu1);
            alpha *= cdl._alphaMultiplier;
            epochs += cdl._alphaInterval;
        }

        pNetwork->SaveNetCDF(cdl._resultsFileName);
    }
    else
    {
        bool bFilterPast = false;

        const Layer* pLayer = pNetwork->GetLayer("Output");
        uint32_t Nx, Ny, Nz, Nw;
        std::tie(Nx, Ny, Nz, Nw) = pLayer->GetLocalDimensions();

        const uint32_t STRIDE = Nx * Ny * Nz * Nw;

        unsigned int K = 10;

        size_t inputIndex = 0;
        while ((inputIndex < vDataSet.size()) && (vDataSet[inputIndex]->_name != "input"))
            inputIndex++;
        if (inputIndex == vDataSet.size())
        {
            std::cerr << "Unable to find input dataset, exiting.\n";
            exit(-1);
        }

        size_t outputIndex = 0;
        while ((outputIndex < vDataSet.size()) && (vDataSet[outputIndex]->_name != "output"))
            outputIndex++;
        if (outputIndex == vDataSet.size())
        {
            std::cerr << "Unable to find output dataset, exiting.\n";
            exit(-1);
        }

        int batch = cdl._batch;

        std::vector<NNFloat> vPrecision(K);
        std::vector<NNFloat> vRecall(K);
        std::vector<NNFloat> vNDCG(K);

        std::vector<uint32_t> vDataPoints(batch);

        auto pbTarget = std::make_unique<GpuBuffer<NNFloat>>(batch * STRIDE, true);
        auto pbOutput = std::make_unique<GpuBuffer<NNFloat>>(batch * STRIDE, true);

        auto pInputDataSet = dynamic_cast<DataSet<NNFloat>*>(vDataSet[inputIndex].get());
        auto pOutputDataSet = dynamic_cast<DataSet<NNFloat>*>(vDataSet[outputIndex].get());
        auto pbKey = std::make_unique<GpuBuffer<NNFloat>>(batch * K, true);
        auto pbUIValue = std::make_unique<GpuBuffer<unsigned int>>(batch * K, true);
        auto pbFValue = std::make_unique<GpuBuffer<NNFloat>>(batch * K, true);

        NNFloat* pOutputValue = pbOutput->_pSysData;

        bool bMultiGPU = (getGpu()._numprocs > 1);

        std::unique_ptr<GpuBuffer<NNFloat>> pbMultiKey = nullptr;
        std::unique_ptr<GpuBuffer<NNFloat>> pbMultiFValue = nullptr;
        NNFloat* pMultiKey = nullptr;
        NNFloat* pMultiFValue = nullptr;

        cudaIpcMemHandle_t keyMemHandle;
        cudaIpcMemHandle_t valMemHandle;

        if (bMultiGPU)
        {
            if (getGpu()._id == 0)
            {
                pbMultiKey = std::make_unique<GpuBuffer<NNFloat>>(getGpu()._numprocs * batch * K, true);
                pbMultiFValue = std::make_unique<GpuBuffer<NNFloat>>(getGpu()._numprocs * batch * K, true);
                pMultiKey = pbMultiKey->_pDevData;
                pMultiFValue = pbMultiFValue->_pDevData;

                cudaError_t status = cudaIpcGetMemHandle(&keyMemHandle, pMultiKey);
                RTERROR(status, "cudaIpcGetMemHandle: Failed to get IPC mem handle on pMultiKey");

                status = cudaIpcGetMemHandle(&valMemHandle, pMultiFValue);
                RTERROR(status, "cudaIpcGetMemHandle: Failed to get IPC mem handle on pMultiFValue");
            }

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

            unsigned int batch = pNetwork->GetBatch();

            if (pos + batch > pNetwork->GetExamples())
                batch = pNetwork->GetExamples() - pos;

            NNFloat* pTarget = pbTarget->_pSysData;
            memset(pTarget, 0, STRIDE * batch * sizeof(NNFloat));

            const NNFloat* pOutputKey = pNetwork->GetUnitBuffer("Output");
            NNFloat* pOut = pOutputValue;
            cudaError_t status = cudaMemcpy(pOut, pOutputKey, batch * STRIDE * sizeof(NNFloat), cudaMemcpyDeviceToHost);
            RTERROR(status, "cudaMemcpy GpuBuffer::Download failed");

            for (int i = 0; i < batch; i++)
            {
                int j = pos + i;
                vDataPoints[i] = pOutputDataSet->_vSparseEnd[j] - pOutputDataSet->_vSparseStart[j];

                for (size_t k = pOutputDataSet->_vSparseStart[j]; k < pOutputDataSet->_vSparseEnd[j]; k++)
                {
                    pTarget[pOutputDataSet->_vSparseIndex[k]] = 1.0f;
                }

                if (bFilterPast)
                {
                    for (size_t k = pInputDataSet->_vSparseStart[j]; k < pInputDataSet->_vSparseEnd[j]; k++)
                    {
                        pOut[pInputDataSet->_vSparseIndex[k]] = 0.0f;
                    }
                }

                pTarget += STRIDE;
                pOut += STRIDE;
            }
            pbTarget->Upload();

            pbOutput->Upload();

            kCalculateOutput(pbOutput->_pDevData, pbTarget->_pDevData, pbKey->_pDevData, pbFValue->_pDevData, batch, STRIDE, K);

            pbKey->Download();

            pbFValue->Download();

            if (bMultiGPU)
            {

                MPI_Reduce((getGpu()._id == 0) ? MPI_IN_PLACE : vDataPoints.data(), vDataPoints.data(), batch, MPI_UINT32_T, MPI_SUM, 0, MPI_COMM_WORLD);

                uint32_t offset = K * getGpu()._id;
                uint32_t kstride = K * getGpu()._numprocs;

                cudaMemcpy2D(pMultiKey + offset, kstride * sizeof(NNFloat), pbKey->_pDevData, K * sizeof(NNFloat), K * sizeof(NNFloat), batch, cudaMemcpyDefault);
                cudaMemcpy2D(pMultiFValue + offset, kstride * sizeof(NNFloat), pbFValue->_pDevData, K * sizeof(NNFloat), K * sizeof(NNFloat), batch, cudaMemcpyDefault);

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

                NNFloat* pKey = pbKey->_pSysData;
                NNFloat* pValue = pbFValue->_pSysData;

                for (int i = 0; i < batch; i++)
                {
                    NNFloat p = vDataPoints[i];

                    NNFloat tp = 0.0f;
                    NNFloat fp = 0.0f;
                    NNFloat idcg = 0.0f;
                    NNFloat dcg = 0.0f;

                    for (NNFloat pp = 0.0f; pp < p; pp++)
                    {
                        idcg += 1.0f / log2(pp + 2.0f);
                    }

                    for (int j = 0; j < K; j++)
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
        }

        pbKey.reset();
        pbFValue.reset();
        pbUIValue.reset();
        pbTarget.reset();
        pbOutput.reset();

        if (bMultiGPU)
        {
            if (getGpu()._id != 0)
            {
                cudaError_t status = cudaIpcCloseMemHandle(pMultiKey);
                RTERROR(status, "cudaIpcCloseMemHandle: Error closing MultiKey IpcMemHandle");

                status = cudaIpcCloseMemHandle(pMultiFValue);
                RTERROR(status, "cudaIpcCloseMemHandle: Error closing MultiFValue IpcMemHandle");
            }

            pbMultiKey.reset();
            pbMultiFValue.reset();
        }

        if (getGpu()._id == 0)
        {
            for (int i = 0; i < K; i++)
                std::cout << i + 1 << "," << vPrecision[i] / pNetwork->GetExamples() << "," << vRecall[i] / pNetwork->GetExamples() << "," << vNDCG[i] / pNetwork->GetExamples() << '\n';
        }
    }

    getGpu().GetMemoryUsage(&totalGPUMemory, &totalCPUMemory);

    if (getGpu()._id == 0)
    {
        std::cout << "GPU Memory Usage: " << totalGPUMemory << " KB\n";
        std::cout << "CPU Memory Usage: " << totalCPUMemory << " KB\n";
    }

    pNetwork.reset();
    vDataSet.clear();
    getGpu().Shutdown();

    return 0;
}
