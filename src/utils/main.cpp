#include "GpuTypes.h"  // Include file for GPU types
#include "Types.h"  // Include file for general types
#include <chrono>  // Include file for time-related functions
#include <iostream>  // Include file for standard input/output
#include <fstream>  // Include file for file input/output
#include <vector>  // Include file for vector container
#include <tuple>  // Include file for tuple container
#include <string_view>  // Include file for string_view type
#include <memory>  // Include file for smart pointers
#include <filesystem>  // Include file for filesystem operations

namespace fs = std::filesystem;  // Alias for std::filesystem namespace

int main(int argc, char** argv)
{
    getGpu().Startup(argc, argv);  // Initialize GPU

    CDL cdl;  // Create CDL object

    if (argc == 2)  // Check if command-line argument is provided
    {
        int err = cdl.Load_JSON(argv[1]);  // Load JSON file
        if (err != 0)
        {
            std::cerr << "*** Error, " << argv[0] << " could not parse CDC file " << argv[1] << '\n';  // Print error message
            return -1;
        }
    }
    else  // If no command-line argument is provided
    {
        // Set default values for CDL object
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

    getGpu().SetRandomSeed(cdl._randomSeed);  // Set random seed for GPU

    float lambda1 = 0.0f;  // Initialize lambda1 variable
    float mu1 = 0.0f;  // Initialize mu1 variable

    std::unique_ptr<Network> pNetwork;  // Create unique pointer to Network object

    std::vector<std::unique_ptr<DataSetBase>> vDataSet;  // Create vector of unique pointers to DataSetBase objects

    vDataSet = LoadNetCDF(cdl._dataFileName);  // Load NetCDF data into the vector

#if 0
    std::vector<std::tuple<uint64_t, uint64_t>> vMemory = vDataSet[0]->getMemoryUsage();

    uint64_t cpuMemory, gpuMemory;
    std::tie(cpuMemory, gpuMemory) = vMemory[0];

    std::cout << "CPUMem: " << cpuMemory << " GPUMem: " << gpuMemory << '\n';

    exit(-1);
#endif

    if (cdl._mode == Prediction)  // Check if mode is Prediction
        pNetwork = LoadNeuralNetworkNetCDF(cdl._networkFileName, cdl._batch);  // Load neural network from NetCDF file
    else  // If mode is not Prediction
        pNetwork = LoadNeuralNetworkJSON(cdl._networkFileName, cdl._batch, vDataSet);  // Load neural network from JSON file with given dataset

    int totalGPUMemory;  // Initialize variable for total GPU memory
    int totalCPUMemory;  // Initialize variable for total CPU memory
    getGpu().GetMemoryUsage(&totalGPUMemory, &totalCPUMemory);  // Get memory usage from GPU

    std::cout << "GPU Memory Usage: " << totalGPUMemory << " KB\n";  // Print GPU memory usage
    std::cout << "CPU Memory Usage: " << totalCPUMemory << " KB\n";  // Print CPU memory usage

    pNetwork->LoadDataSets(vDataSet);  // Load datasets into the network

    pNetwork->SetCheckpoint(cdl._checkpointFileName, cdl._checkpointInterval);  // Set checkpoint parameters

    if (cdl._mode == Mode::Validation)  // If mode is Validation
    {
        pNetwork->SetTrainingMode(Nesterov);  // Set training mode to Nesterov
        pNetwork->Validate();  // Perform validation
    }
    else if (cdl._mode == Training)  // If mode is Training
    {
        pNetwork->SetTrainingMode(cdl._optimizer);  // Set training mode to specified optimizer

        float alpha = cdl._alpha;  // Initialize alpha variable
        int epochs = 0;  // Initialize epochs variable

        while (epochs < cdl._epochs)  // Train for specified number of epochs
        {
            pNetwork->Train(cdl._alphaInterval, alpha, cdl._lambda, lambda1, cdl._mu, mu1);  // Train the network
            alpha *= cdl._alphaMultiplier;  // Update alpha
            epochs += cdl._alphaInterval;  // Increment epochs
        }

        pNetwork->SaveNetCDF(cdl._resultsFileName);  // Save results to NetCDF file
    }
    else  // If mode is not Validation or Training
    {
        bool bFilterPast = false;  // Initialize bFilterPast variable

        const Layer* pLayer = pNetwork->GetLayer("Output");  // Get output layer
        uint32_t Nx, Ny, Nz, Nw;  // Initialize variables for dimensions
        std::tie(Nx, Ny, Nz, Nw) = pLayer->GetLocalDimensions();  // Get dimensions of output layer

        const uint32_t STRIDE = Nx * Ny * Nz * Nw;  // Calculate stride

        unsigned int K = 10;  // Set K value

        size_t inputIndex = 0;  // Initialize inputIndex variable
        while ((inputIndex < vDataSet.size()) && (vDataSet[inputIndex]->_name != "input"))  // Find index of input dataset
            inputIndex++;
        if (inputIndex == vDataSet.size())
        {
            std::cerr << "Unable to find input dataset, exiting.\n";  // Print error message and exit
            exit(-1);
        }

        size_t outputIndex = 0;  // Initialize outputIndex variable
        while ((outputIndex < vDataSet.size()) && (vDataSet[outputIndex]->_name != "output"))  // Find index of output dataset
            outputIndex++;
        if (outputIndex == vDataSet.size())
        {
            std::cerr << "Unable to find output dataset, exiting.\n";  // Print error message and exit
            exit(-1);
        }

        int batch = cdl._batch;  // Set batch size

        std::vector<NNFloat> vPrecision(K);  // Create vector for precision values
        std::vector<NNFloat> vRecall(K);  // Create vector for recall values
        std::vector<NNFloat> vNDCG(K);  // Create vector for NDCG values

        std::vector<uint32_t> vDataPoints(batch);  // Create vector for data points

        std::unique_ptr<GpuBuffer<NNFloat>> pbTarget = std::make_unique<GpuBuffer<NNFloat>>(batch * STRIDE, true);  // Create unique pointer for target buffer
        std::unique_ptr<GpuBuffer<NNFloat>> pbOutput = std::make_unique<GpuBuffer<NNFloat>>(batch * STRIDE, true);  // Create unique pointer for output buffer

        DataSet<NNFloat>* pInputDataSet = dynamic_cast<DataSet<NNFloat>*>(vDataSet[inputIndex].get());  // Get input dataset
        DataSet<NNFloat>* pOutputDataSet = dynamic_cast<DataSet<NNFloat>*>(vDataSet[outputIndex].get());  // Get output dataset
        std::unique_ptr<GpuBuffer<NNFloat>> pbKey = std::make_unique<GpuBuffer<NNFloat>>(batch * K, true);  // Create unique pointer for key buffer
        std::unique_ptr<GpuBuffer<unsigned int>> pbUIValue = std::make_unique<GpuBuffer<unsigned int>>(batch * K, true);  // Create unique pointer for UIValue buffer
        std::unique_ptr<GpuBuffer<NNFloat>> pbFValue = std::make_unique<GpuBuffer<NNFloat>>(batch * K, true);  // Create unique pointer for FValue buffer

        NNFloat* pOutputValue = pbOutput->_pSysData;  // Get pointer to output value data

        bool bMultiGPU = (getGpu()._numprocs > 1);  // Check if multiple GPUs are being used

        std::unique_ptr<GpuBuffer<NNFloat>> pbMultiKey = nullptr;  // Create unique pointer for multi-key buffer
        std::unique_ptr<GpuBuffer<NNFloat>> pbMultiFValue = nullptr;  // Create unique pointer for multi-FValue buffer
        NNFloat* pMultiKey = nullptr;  // Initialize pointer to multi-key data
        NNFloat* pMultiFValue = nullptr;  // Initialize pointer to multi-FValue data

        cudaIpcMemHandle_t keyMemHandle;  // Create IPC memory handle for key
        cudaIpcMemHandle_t valMemHandle;  // Create IPC memory handle for value

        if (bMultiGPU)  // If multiple GPUs are being used
        {
            if (getGpu()._id == 0)
            {
                pbMultiKey = std::make_unique<GpuBuffer<NNFloat>>(getGpu()._numprocs * batch * K, true);  // Create unique pointer for multi-key buffer
                pbMultiFValue = std::make_unique<GpuBuffer<NNFloat>>(getGpu()._numprocs * batch * K, true);  // Create unique pointer for multi-FValue buffer
                pMultiKey = pbMultiKey->_pDevData;  // Get pointer to multi-key data
                pMultiFValue = pbMultiFValue->_pDevData;  // Get pointer to multi-FValue data

                cudaError_t status = cudaIpcGetMemHandle(&keyMemHandle, pMultiKey);  // Get IPC memory handle for multi-key data
                RTERROR(status, "cudaIpcGetMemHandle: Failed to get IPC mem handle on pMultiKey");

                status = cudaIpcGetMemHandle(&valMemHandle, pMultiFValue);  // Get IPC memory handle for multi-FValue data
                RTERROR(status, "cudaIpcGetMemHandle: Failed to get IPC mem handle on pMultiFValue");
            }

            MPI_Bcast(&keyMemHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, MPI_COMM_WORLD);  // Broadcast IPC memory handle for multi-key data
            MPI_Bcast(&valMemHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, MPI_COMM_WORLD);  // Broadcast IPC memory handle for multi-FValue data

            if (getGpu()._id != 0)
            {
                cudaError_t status = cudaIpcOpenMemHandle((void**)&pMultiKey, keyMemHandle, cudaIpcMemLazyEnablePeerAccess);  // Open IPC memory handle for multi-key data
                RTERROR(status, "cudaIpcOpenMemHandle: Unable to open key IPCMemHandle");

                status = cudaIpcOpenMemHandle((void**)&pMultiFValue, valMemHandle, cudaIpcMemLazyEnablePeerAccess);  // Open IPC memory handle for multi-FValue data
                RTERROR(status, "cudaIpcOpenMemHandle: Unable to open value IPCMemHandle");
            }
        }

        for (unsigned long long int pos = 0; pos < pNetwork->GetExamples(); pos += pNetwork->GetBatch())  // Loop over examples in batches
        {
            pNetwork->SetPosition(pos);  // Set current position in the network

            pNetwork->PredictBatch();  // Perform batch prediction

            unsigned int batch = pNetwork->GetBatch();  // Get batch size

            if (pos + batch > pNetwork->GetExamples())  // Adjust batch size if necessary
                batch = pNetwork->GetExamples() - pos;

            NNFloat* pTarget = pbTarget->_pSysData;  // Get pointer to target data

            memset(pTarget, 0, STRIDE * batch * sizeof(NNFloat));  // Initialize target data to zero

            const NNFloat* pOutputKey = pNetwork->GetUnitBuffer("Output");  // Get output key data
            NNFloat* pOut = pOutputValue;  // Get pointer to output value data
            cudaError_t status = cudaMemcpy(pOut, pOutputKey, batch * STRIDE * sizeof(NNFloat), cudaMemcpyDeviceToHost);  // Copy output value data from device to host
            RTERROR(status, "cudaMemcpy GpuBuffer::Download failed");

            for (int i = 0; i < batch; i++)  // Loop over examples in the batch
            {
                int j = pos + i;
                vDataPoints[i] = pOutputDataSet->_vSparseEnd[j] - pOutputDataSet->_vSparseStart[j];  // Get number of data points for the example

                for (size_t k = pOutputDataSet->_vSparseStart[j]; k < pOutputDataSet->_vSparseEnd[j]; k++)  // Loop over data points in the example
                {
                    pTarget[pOutputDataSet->_vSparseIndex[k]] = 1.0f;  // Set target value to 1.0 for the data point
                }

                if (bFilterPast)  // If filtering past data points
                {
                    for (size_t k = pInputDataSet->_vSparseStart[j]; k < pInputDataSet->_vSparseEnd[j]; k++)  // Loop over data points in the input dataset
                    {
                        pOut[pInputDataSet->_vSparseIndex[k]] = 0.0f;  // Set output value to 0.0 for the data point
                    }
                }

                pTarget += STRIDE;
                pOut += STRIDE;
            }
            pbTarget->Upload();  // Upload target data to GPU

            pbOutput->Upload();  // Upload output data to GPU

            kCalculateOutput(pbOutput->_pDevData, pbTarget->_pDevData, pbKey->_pDevData, pbFValue->_pDevData, batch, STRIDE, K);  // Calculate output values

            pbKey->Download();  // Download key data from GPU

            pbFValue->Download();  // Download FValue data from GPU

            if (bMultiGPU)  // If multiple GPUs are being used
            {

                MPI_Reduce((getGpu()._id == 0) ? MPI_IN_PLACE : vDataPoints.data(), vDataPoints.data(), batch, MPI_UINT32_T, MPI_SUM, 0, MPI_COMM_WORLD);  // Reduce data points using MPI

                uint32_t offset = K * getGpu()._id;  // Calculate offset for multi-key and multi-FValue data
                uint32_t kstride = K * getGpu()._numprocs;  // Calculate stride for multi-key and multi-FValue data

                cudaMemcpy2D(pMultiKey + offset, kstride * sizeof(NNFloat), pbKey->_pDevData, K * sizeof(NNFloat), K * sizeof(NNFloat), batch, cudaMemcpyDefault);  // Copy key data to multi-key data
                cudaMemcpy2D(pMultiFValue + offset, kstride * sizeof(NNFloat), pbFValue->_pDevData, K * sizeof(NNFloat), K * sizeof(NNFloat), batch, cudaMemcpyDefault);  // Copy FValue data to multi-FValue data

                cudaDeviceSynchronize();  // Synchronize GPUs

                MPI_Barrier(MPI_COMM_WORLD);  // Synchronize MPI processes

                if (getGpu()._id == 0)  // If GPU ID is 0
                {
                    kCalculateOutput(pbMultiKey->_pDevData, pbMultiFValue->_pDevData, pbKey->_pDevData, pbFValue->_pDevData, batch, getGpu()._numprocs * K, K);  // Calculate output values using multi-key and multi-FValue data
                }
            }

            if (getGpu()._id == 0)  // If GPU ID is 0
            {
                pbKey->Download();  // Download key data from GPU
                pbFValue->Download();  // Download FValue data from GPU

                NNFloat* pKey = pbKey->_pSysData;  // Get pointer to key data
                NNFloat* pValue = pbFValue->_pSysData;  // Get pointer to FValue data

                for (int i = 0; i < batch; i++)  // Loop over examples in the batch
                {
                    NNFloat p = vDataPoints[i];  // Get number of data points for the example

                    NNFloat tp = 0.0f;  // Initialize true positives
                    NNFloat fp = 0.0f;  // Initialize false positives
                    NNFloat idcg = 0.0f;  // Initialize ideal DCG
                    NNFloat dcg = 0.0f;  // Initialize DCG

                    for (NNFloat pp = 0.0f; pp < p; pp++)  // Calculate ideal DCG
                    {
                        idcg += 1.0f / log2(pp + 2.0f);
                    }

                    for (int j = 0; j < K; j++)  // Loop over top K values
                    {
                        if (pValue[j] == 1.0f)  // If value is 1.0
                        {
                            tp++;  // Increment true positives
                            dcg += 1.0f / log2(static_cast<float>(j + 2));  // Calculate DCG
                        }
                        else
                        {
                            fp++;  // Increment false positives
                        }

                        vPrecision[j] += tp / (tp + fp);  // Update precision
                        vRecall[j] += tp / p;  // Update recall
                        vNDCG[j] += dcg / idcg;  // Update NDCG
                    }

                    pKey += K;
                    pValue += K;
                }
            }
        }

        pbKey.reset();  // Reset key buffer
        pbFValue.reset();  // Reset FValue buffer
        pbUIValue.reset();  // Reset UIValue buffer
        pbTarget.reset();  // Reset target buffer
        pbOutput.reset();  // Reset output buffer

        if (bMultiGPU)  // If multiple GPUs are being used
        {
            if (getGpu()._id != 0)
            {
                cudaError_t status = cudaIpcCloseMemHandle(pMultiKey);  // Close IPC memory handle for multi-key data
                RTERROR(status, "cudaIpcCloseMemHandle: Error closing MultiKey IpcMemHandle");

                status = cudaIpcCloseMemHandle(pMultiFValue);  // Close IPC memory handle for multi-FValue data
                RTERROR(status, "cudaIpcCloseMemHandle: Error closing MultiFValue IpcMemHandle");
            }

            pbMultiKey.reset();  // Reset multi-key buffer
            pbMultiFValue.reset();  // Reset multi-FValue buffer
        }

        if (getGpu()._id == 0)  // If GPU ID is 0
        {
            for (int i = 0; i < K; i++)  // Loop over top K values
                std::cout << i + 1 << "," << vPrecision[i] / pNetwork->GetExamples() << "," << vRecall[i] / pNetwork->GetExamples() << "," << vNDCG[i] / pNetwork->GetExamples() << '\n';  // Print precision, recall, and NDCG values
        }
    }

    getGpu().GetMemoryUsage(&totalGPUMemory, &totalCPUMemory);  // Get memory usage from GPU

    if (getGpu()._id == 0)  // If GPU ID is 0
    {
        std::cout << "GPU Memory Usage: " << totalGPUMemory << " KB\n";  // Print GPU memory usage
        std::cout << "CPU Memory Usage: " << totalCPUMemory << " KB\n";  // Print CPU memory usage
    }

    pNetwork.reset();  // Reset network

    vDataSet.clear();  // Clear dataset vector

    getGpu().Shutdown();  // Shutdown GPU

    return 0;  // Return from main function
}
