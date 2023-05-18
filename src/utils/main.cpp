#include "GpuTypes.h"
#include "Types.h"
#include <climits>
#include <ctime>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <tuple>

int main(int argc, char** argv)
{
    /**
     * @brief Initializes the GPU.
     *
     * This statement initializes the GPU by calling the Startup function of the GPU object (getGpu()).
     *
     * @param[in] argc The number of command line arguments.
     * @param[in] argv The array of command line arguments.
     */
    getGpu().Startup(argc, argv);

    /**
     * @brief Creates a CDL object.
     *
     * A CDL object (cdl) is created to store the configuration settings.
     */
    CDL cdl;

    /**
     * @brief Loads the configuration from a JSON file.
     *
     * This statement checks if the number of command line arguments (argc) is 2.
     * If so, it attempts to load the configuration from the specified JSON file (argv[1]) using the Load_JSON function of the CDL object.
     * If an error occurs during loading, an error message is displayed, and the program exits.
     *
     * @param[in] argv[1] The path to the JSON file.
     * @return The error code, indicating the success or failure of loading the JSON file.
     */
    if (argc == 2)
    {
        int err = cdl.Load_JSON(argv[1]);
        if (err != 0)
        {
            std::cerr << "*** Error, " << argv[0] << " could not parse CDC file " << argv[1] << std::endl;
            return -1;
        }
    }
    else
    {
        /**
         * @brief Sets the mode to Prediction.
         *
         * This statement sets the mode (cdl._mode) to Prediction, indicating that the network is being used for prediction.
         */
        cdl._mode = Prediction;

        /**
         * @brief Sets the optimizer to Nesterov.
         *
         * This statement sets the optimizer (cdl._optimizer) to Nesterov for training the neural network.
         */
        cdl._optimizer = TrainingMode::Nesterov;

        /**
         * @brief Sets the network file name.
         *
         * This statement sets the network file name (cdl._networkFileName) to "network.nc".
         *
         * @note Adjust the file name according to your specific network file.
         */
        cdl._networkFileName = "network.nc";

        /**
         * @brief Sets the alpha interval.
         *
         * This statement sets the alpha interval (cdl._alphaInterval) to 20, which represents the number of epochs before updating the learning rate.
         */
        cdl._alphaInterval = 20;

        /**
         * @brief Sets the alpha multiplier.
         *
         * This statement sets the alpha multiplier (cdl._alphaMultiplier) to 0.8, which is used to decrease the learning rate after each alpha interval.
         */
        cdl._alphaMultiplier = 0.8f;

        /**
         * @brief Sets the initial learning rate (alpha).
         *
         * This statement sets the initial learning rate (cdl._alpha) to 0.025, which is used in the training process to adjust the weights of the neural network.
         */
        cdl._alpha = 0.025f;

        /**
         * @brief Sets the lambda regularization parameter.
         *
         * This statement sets the lambda regularization parameter (cdl._lambda) to 0.0001, which controls the impact of weight decay in the training process.
         */
        cdl._lambda = 0.0001f;

        /**
         * @brief Sets the mu regularization parameter.
         *
         * This statement sets the mu regularization parameter (cdl._mu) to 0.5, which controls the impact of sparse penalty in the training process.
         */
        cdl._mu = 0.5f;

        /**
         * @brief Sets the random seed.
         *
         * This statement sets the random seed (cdl._randomSeed) to 12345, which initializes the random number generator used in the training process.
         */
        cdl._randomSeed = 12345;

        /**
         * @brief Sets the number of epochs.
         *
         * This statement sets the number of epochs (cdl._epochs) to 60, which represents the total number of training iterations.
         */
        cdl._epochs = 60;

        /**
         * @brief Sets the data file name.
         *
         * This statement sets the data file name (cdl._dataFileName) to "../../data/data_test.nc", which is the file containing the input data for training or prediction.
         *
         * @note Adjust the file name according to your specific data file.
         */
        cdl._dataFileName = "../../data/data_test.nc";
    }

    /**
     * @brief Sets the random seed for the GPU.
     *
     * This statement sets the random seed for the GPU using the specified random seed value (cdl._randomSeed).
     *
     * @param[in] cdl._randomSeed The random seed value.
     */
    getGpu().SetRandomSeed(cdl._randomSeed);

    /**
     * @brief Initializes lambda1 and mu1.
     *
     * The variables lambda1 and mu1 are set to 0.0f.
     */
    float lambda1 = 0.0f;
    float mu1 = 0.0f;

    /**
     * @brief Pointer to the neural network.
     *
     * The pointer pNetwork is declared to store the neural network object.
     */
    Network* pNetwork;

    /**
     * @brief Vector of dataset pointers.
     *
     * The vector vDataSet is declared to store pointers to dataset objects.
     */
    std::vector<DataSetBase*> vDataSet;

    /**
     * @brief Loads datasets from a NetCDF file.
     *
     * This statement loads datasets from the specified NetCDF file (cdl._dataFileName) using the LoadNetCDF function.
     * The loaded datasets are assigned to the vDataSet vector.
     *
     * @param[in] cdl._dataFileName The file name of the NetCDF file.
     * @return The vector of loaded datasets.
     */
    vDataSet = LoadNetCDF(cdl._dataFileName);

#if 0
    /**
     * @brief Retrieves the memory usage of the first dataset.
     *
     * This statement retrieves the memory usage of the first dataset (vDataSet[0]) and stores it in the vector vMemory.
     * The memory usage is represented as a tuple of CPU memory and GPU memory.
     *
     * @param[out] vMemory The vector of memory usage tuples.
     */
    std::vector<std::tuple<uint64_t, uint64_t>> vMemory = vDataSet[0]->getMemoryUsage();

    /**
     * @brief Retrieves the CPU and GPU memory usage from the first tuple.
     *
     * This statement extracts the CPU memory and GPU memory values from the first tuple in vMemory and assigns them to cpuMemory and gpuMemory respectively.
     *
     * @param[out] cpuMemory The CPU memory usage.
     * @param[out] gpuMemory The GPU memory usage.
     */
    uint64_t cpuMemory, gpuMemory;
    std::tie(cpuMemory, gpuMemory) = vMemory[0];

    /**
     * @brief Prints the CPU and GPU memory usage.
     *
     * These statements print the CPU and GPU memory usage to the console.
     */
    std::cout << "CPUMem: " << cpuMemory << " GPUMem: " << gpuMemory << std::endl;

    /**
     * @brief Exits the program with an error code.
     *
     * This statement exits the program with an error code (-1).
     */
    exit(-1);
#endif
    /**
     * @brief Loads the neural network from a NetCDF or JSON file.
     *
     * Depending on the specified mode (cdl._mode), this statement loads the neural network (pNetwork) from either a NetCDF file or a JSON file.
     * If the mode is Prediction, the neural network is loaded from a NetCDF file using the LoadNeuralNetworkNetCDF function.
     * If the mode is not Prediction, the neural network is loaded from a JSON file using the LoadNeuralNetworkJSON function, along with the specified batch size (cdl._batch) and dataset vector (vDataSet).
     *
     * @param[in] cdl._mode The mode of operation.
     * @param[in] cdl._networkFileName The file name of the neural network.
     * @param[in] cdl._batch The batch size.
     * @param[in] vDataSet The vector of datasets.
     */
    if (cdl._mode == Prediction)
        pNetwork = LoadNeuralNetworkNetCDF(cdl._networkFileName, cdl._batch);
    else
        pNetwork = LoadNeuralNetworkJSON(cdl._networkFileName, cdl._batch, vDataSet);

    /**
     * @brief Retrieves the GPU and CPU memory usage.
     *
     * This statement retrieves the GPU and CPU memory usage from the getGpu() object and stores the values in totalGPUMemory and totalCPUMemory respectively.
     *
     * @param[out] totalGPUMemory The total GPU memory usage.
     * @param[out] totalCPUMemory The total CPU memory usage.
     */
    int totalGPUMemory;
    int totalCPUMemory;
    getGpu().GetMemoryUsage(&totalGPUMemory, &totalCPUMemory);

    /**
     * @brief Prints the GPU and CPU memory usage.
     *
     * These statements print the GPU and CPU memory usage to the console.
     */
    std::cout << "GPU Memory Usage: " << totalGPUMemory << " KB" << std::endl;
    std::cout << "CPU Memory Usage: " << totalCPUMemory << " KB" << std::endl;

    /**
     * @brief Loads the datasets into the neural network.
     *
     * This statement loads the datasets (vDataSet) into the neural network (pNetwork).
     *
     * @param[in] vDataSet The vector of datasets.
     */
    pNetwork->LoadDataSets(vDataSet);

    /**
     * @brief Sets the checkpoint file and interval.
     *
     * This statement sets the checkpoint file name (cdl._checkpointFileName) and the checkpoint interval (cdl._checkpointInterval) for the neural network (pNetwork).
     *
     * @param[in] cdl._checkpointFileName The file name for saving the checkpoint.
     * @param[in] cdl._checkpointInterval The interval for saving the checkpoint.
     */
    pNetwork->SetCheckpoint(cdl._checkpointFileName, cdl._checkpointInterval);

    /**
     * @brief Performs validation if the mode is Validation.
     *
     * This statement checks if the mode is set to Validation (cdl._mode == Mode::Validation) and performs validation on the neural network (pNetwork).
     * The validation process evaluates the performance of the trained network on a validation dataset.
     */
    if (cdl._mode == Mode::Validation)
    {
        /**
         * @brief Sets the training mode of the neural network to Nesterov.
         *
         * This statement sets the training mode of the neural network (pNetwork) to Nesterov.
         */
        pNetwork->SetTrainingMode(Nesterov);

        /**
         * @brief Performs network validation.
         *
         * This statement performs network validation using the Validate() function of the neural network (pNetwork).
         * The validation process evaluates the performance of the trained network on a validation dataset.
         */
        pNetwork->Validate();

    }
    else if (cdl._mode == Training)
    {
        /**
         * @brief Sets the training mode of the neural network.
         *
         * This statement sets the training mode of the neural network (pNetwork) to the specified optimizer (cdl._optimizer).
         *
         * @param[in] cdl._optimizer The optimizer to use for training.
         */
        pNetwork->SetTrainingMode(cdl._optimizer);

        /**
         * @brief Initializes variables for training.
         *
         * The variable alpha is set to cdl._alpha, and the epochs counter is initialized to 0.
         */
        float alpha = cdl._alpha;
        int epochs = 0;

        /**
         * @brief Training loop.
         *
         * This loop continues training the neural network until the number of epochs (epochs) reaches cdl._epochs.
         * Each iteration trains the network for cdl._alphaInterval epochs using the specified learning rate (alpha), regularization parameters (cdl._lambda, lambda1, cdl._mu, mu1), and updates the learning rate for the next iteration (alpha *= cdl._alphaMultiplier).
         *
         * @note The epochs counter is incremented by cdl._alphaInterval in each iteration.
         */
        while (epochs < cdl._epochs)
        {
            pNetwork->Train(cdl._alphaInterval, alpha, cdl._lambda, lambda1, cdl._mu, mu1);
            alpha *= cdl._alphaMultiplier;
            epochs += cdl._alphaInterval;
        }

        /**
         * @brief Saves the neural network to a NetCDF file.
         *
         * This statement saves the trained neural network (pNetwork) to the specified NetCDF file (cdl._resultsFileName).
         *
         * @param[in] cdl._resultsFileName The file name for saving the neural network.
         */
        pNetwork->SaveNetCDF(cdl._resultsFileName);
    }
    else
    {
        /**
         * @brief Flag indicating whether to filter past values.
         *
         * This flag determines whether to filter past values based on bFilterPast.
         */
        bool bFilterPast = false;

        /**
         * @brief Retrieves the local dimensions of the "Output" layer.
         *
         * This statement retrieves the local dimensions (Nx, Ny, Nz, Nw) of the "Output" layer from pNetwork.
         */
        const Layer* pLayer = pNetwork->GetLayer("Output");
        uint32_t Nx, Ny, Nz, Nw;
        std::tie(Nx, Ny, Nz, Nw) = pLayer->GetLocalDimensions();

        /**
         * @brief Stride value for indexing data.
         *
         * The stride value (STRIDE) is calculated based on the local dimensions (Nx, Ny, Nz, Nw).
         */
        const uint32_t STRIDE = Nx * Ny * Nz * Nw;

        /**
         * @brief Number of top-K values to consider.
         *
         * The value of K is set to 10.
         */
        unsigned int K = 10;

        /**
         * @brief Finds the index of the input dataset.
         *
         * This loop searches for the index of the "input" dataset in vDataSet and stores it in inputIndex.
         * If the dataset is not found, an error message is displayed, and the program exits.
         */
        size_t inputIndex = 0;
        while ((inputIndex < vDataSet.size()) && (vDataSet[inputIndex]->_name != "input"))
            inputIndex++;
        if (inputIndex == vDataSet.size())
        {
            std::cerr << "Unable to find input dataset, exiting." << std::endl;
            exit(-1);
        }

        /**
         * @brief Finds the index of the output dataset.
         *
         * This loop searches for the index of the "output" dataset in vDataSet and stores it in outputIndex.
         * If the dataset is not found, an error message is displayed, and the program exits.
         */
        size_t outputIndex = 0;
        while ((outputIndex < vDataSet.size()) && (vDataSet[outputIndex]->_name != "output"))
            outputIndex++;
        if (outputIndex == vDataSet.size())
        {
            std::cerr << "Unable to find output dataset, exiting." << std::endl;
            exit(-1);
        }

        /**
         * @brief Batch size for evaluation.
         *
         * The batch size is set to the value stored in cdl._batch.
         */
        int batch = cdl._batch;

        /**
         * @brief Vectors for evaluation metrics.
         *
         * These vectors (vPrecision, vRecall, and vNDCG) are used to store the evaluation metrics for each K value.
         * Each vector is initialized with a size of K.
         */
        std::vector<NNFloat> vPrecision(K);
        std::vector<NNFloat> vRecall(K);
        std::vector<NNFloat> vNDCG(K);

        /**
         * @brief Vector for storing the data point counts.
         *
         * The vDataPoints vector is used to store the count of data points for each batch.
         * It is initialized with a size of batch.
         */
        std::vector<uint32_t> vDataPoints(batch);

        /**
         * @brief GpuBuffers for target and output data.
         *
         * These GpuBuffers (pbTarget and pbOutput) are created to store the target and output data for each batch.
         * Each GpuBuffer is initialized with a size of batch * STRIDE.
         *
         * @note The GpuBuffers are created with the 'true' parameter to indicate ownership of device memory.
         */
        GpuBuffer<NNFloat>* pbTarget = new GpuBuffer<NNFloat>(batch * STRIDE, true);
        GpuBuffer<NNFloat>* pbOutput = new GpuBuffer<NNFloat>(batch * STRIDE, true);

        /**
         * @brief Pointers and buffers for input and output datasets.
         *
         * These statements initialize pointers and buffers for the input and output datasets.
         * The pbKey buffer stores the key data, pbUIValue stores the unsigned integer data, and pbFValue stores the float data.
         * The pbKey and pbFValue buffers are initialized with a size of batch * K.
         *
         * @note The GpuBuffers are created with the 'true' parameter to indicate ownership of device memory.
         */
        DataSet<NNFloat>* pInputDataSet = dynamic_cast<DataSet<NNFloat>*>(vDataSet[inputIndex]);
        DataSet<NNFloat>* pOutputDataSet = dynamic_cast<DataSet<NNFloat>*>(vDataSet[outputIndex]);
        GpuBuffer<NNFloat>* pbKey = new GpuBuffer<NNFloat>(batch * K, true);
        GpuBuffer<unsigned int>* pbUIValue = new GpuBuffer<unsigned int>(batch * K, true);
        GpuBuffer<NNFloat>* pbFValue = new GpuBuffer<NNFloat>(batch * K, true);

        /**
         * @brief Pointer to the output data on the CPU.
         *
         * The pointer pOutputValue is assigned the value of pbOutput->_pSysData.
         */
        NNFloat* pOutputValue = pbOutput->_pSysData;

        /**
         * @brief Flag indicating multi-GPU setup.
         *
         * The flag bMultiGPU is set to true if the number of GPUs (getGpu()._numprocs) is greater than 1.
         */
        bool bMultiGPU = (getGpu()._numprocs > 1);

        /**
         * @brief Buffers for multi-GPU communication.
         *
         * These pointers and GpuBuffers (pbMultiKey and pbMultiFValue) are used for multi-GPU communication.
         * They are initialized to nullptr.
         *
         * @note The GpuBuffers are created with the 'true' parameter to indicate ownership of device memory.
         */
        GpuBuffer<NNFloat>* pbMultiKey = nullptr;
        GpuBuffer<NNFloat>* pbMultiFValue = nullptr;
        NNFloat* pMultiKey = nullptr;
        NNFloat* pMultiFValue = nullptr;

        /**
         * @brief IPC memory handles for multi-GPU communication.
         *
         * These variables (keyMemHandle and valMemHandle) store the IPC memory handles for pMultiKey and pMultiFValue respectively.
         */
        cudaIpcMemHandle_t keyMemHandle;
        cudaIpcMemHandle_t valMemHandle;

        if (bMultiGPU)
        {
            if (getGpu()._id == 0)
            {
                /**
                 * @brief Creates GpuBuffers for multi-GPU communication.
                 *
                 * This block of code is executed only by the GPU with ID 0.
                 * It creates GpuBuffers (pbMultiKey and pbMultiFValue) to store the data for multi-GPU communication.
                 * The size of each GpuBuffer is getGpu()._numprocs * batch * K.
                 * The pointers to the device data (pMultiKey and pMultiFValue) are retrieved from the GpuBuffers.
                 *
                 * @note The GpuBuffers are created with the 'true' parameter to indicate ownership of device memory.
                 */
                pbMultiKey = new GpuBuffer<NNFloat>(getGpu()._numprocs * batch * K, true);
                pbMultiFValue = new GpuBuffer<NNFloat>(getGpu()._numprocs * batch * K, true);
                pMultiKey = pbMultiKey->_pDevData;
                pMultiFValue = pbMultiFValue->_pDevData;

                /**
                 * @brief Gets the IPC memory handles for pMultiKey and pMultiFValue.
                 *
                 * This statement gets the IPC memory handles for pMultiKey and pMultiFValue using cudaIpcGetMemHandle.
                 * The obtained memory handles are stored in keyMemHandle and valMemHandle respectively.
                 *
                 * @param[out] keyMemHandle The IPC memory handle for pMultiKey.
                 * @param[in] pMultiKey The pointer to the device memory for pMultiKey.
                 */
                cudaError_t status = cudaIpcGetMemHandle(&keyMemHandle, pMultiKey);
                RTERROR(status, "cudaIpcGetMemHandle: Failed to get IPC mem handle on pMultiKey");

                /**
                 * @brief Gets the IPC memory handle for pMultiFValue.
                 *
                 * This statement gets the IPC memory handle for pMultiFValue using cudaIpcGetMemHandle.
                 * The obtained memory handle is stored in valMemHandle.
                 *
                 * @param[out] valMemHandle The IPC memory handle for pMultiFValue.
                 * @param[in] pMultiFValue The pointer to the device memory for pMultiFValue.
                 */
                status = cudaIpcGetMemHandle(&valMemHandle, pMultiFValue);
                RTERROR(status, "cudaIpcGetMemHandle: Failed to get IPC mem handle on pMultiFValue");
            }

            /**
             * @brief Broadcasts the IPC memory handles to all MPI processes.
             *
             * These statements broadcast the IPC memory handles (keyMemHandle and valMemHandle) from the GPU with ID 0 to all other MPI processes using MPI_Bcast.
             * The memory handles are broadcasted as byte arrays of size sizeof(cudaIpcMemHandle_t).
             */
            MPI_Bcast(&keyMemHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&valMemHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, MPI_COMM_WORLD);

            if (getGpu()._id != 0)
            {
                /**
                 * @brief Opens the IPC memory handles on non-zero GPU processes.
                 *
                 * This block of code is executed only by GPU processes with IDs other than 0.
                 * It opens the IPC memory handles (keyMemHandle and valMemHandle) to access the shared memory from the GPU with ID 0 using cudaIpcOpenMemHandle.
                 * The opened memory handles are assigned to pMultiKey and pMultiFValue pointers respectively.
                 *
                 * @note The cudaIpcMemLazyEnablePeerAccess flag is used to enable peer access between GPUs.
                 *
                 * @param[out] pMultiKey The pointer to the device memory for pMultiKey.
                 * @param[in] keyMemHandle The IPC memory handle for pMultiKey.
                 */
                cudaError_t status = cudaIpcOpenMemHandle((void**)&pMultiKey, keyMemHandle, cudaIpcMemLazyEnablePeerAccess);
                RTERROR(status, "cudaIpcOpenMemHandle: Unable to open key IPCMemHandle");

                /**
                 * @brief Opens the IPC memory handle for pMultiFValue.
                 *
                 * This statement opens the IPC memory handle (valMemHandle) to access the shared memory from the GPU with ID 0 using cudaIpcOpenMemHandle.
                 * The opened memory handle is assigned to pMultiFValue pointer.
                 *
                 * @param[out] pMultiFValue The pointer to the device memory for pMultiFValue.
                 * @param[in] valMemHandle The IPC memory handle for pMultiFValue.
                 */
                status = cudaIpcOpenMemHandle((void**)&pMultiFValue, valMemHandle, cudaIpcMemLazyEnablePeerAccess);
                RTERROR(status, "cudaIpcOpenMemHandle: Unable to open value IPCMemHandle");
            }
        }

        for (unsigned long long int pos = 0; pos < pNetwork->GetExamples(); pos += pNetwork->GetBatch())
        {
            /**
             * @brief Sets the position of the neural network to the given value.
             *
             * This statement sets the position of the neural network (pNetwork) to the given value (pos).
             *
             * @param[in] pos The desired position of the neural network.
             */
            pNetwork->SetPosition(pos);

            /**
             * @brief Makes predictions for the current batch of data.
             *
             * This statement makes predictions for the current batch of data using the neural network (pNetwork).
             */
            pNetwork->PredictBatch();

            /**
             * @brief Retrieves the batch size.
             *
             * This statement retrieves the batch size from the neural network (pNetwork) and stores it in the variable batch.
             */
            unsigned int batch = pNetwork->GetBatch();

            /**
             * @brief Adjusts the batch size if it exceeds the available examples.
             *
             * If the sum of the current position (pos) and the batch size (batch) exceeds the total number of examples in the neural network (pNetwork),
             * the batch size is adjusted to the remaining number of examples.
             *
             * @note This adjustment is necessary to prevent accessing data beyond the available examples.
             */
            if (pos + batch > pNetwork->GetExamples())
                batch = pNetwork->GetExamples() - pos;

            /**
             * @brief Retrieves a pointer to the target data on the CPU.
             *
             * This statement retrieves a pointer to the target data stored in pbTarget on the CPU and assigns it to pTarget.
             */
            NNFloat* pTarget = pbTarget->_pSysData;

            /**
             * @brief Initializes the target data with zeros.
             *
             * This statement initializes the target data pointed by pTarget with zeros.
             * The total number of elements initialized is STRIDE * batch.
             */
            memset(pTarget, 0, STRIDE * batch * sizeof(NNFloat));

            /**
             * @brief Copies the output data from the device to the host.
             *
             * This statement copies the output data from the device (pOutputKey) to the host (pOut).
             * The size of the data being copied is batch * STRIDE * sizeof(NNFloat).
             *
             * @param[out] pOut The pointer to the host memory where the output data will be copied.
             * @param[in] pOutputKey The pointer to the device memory containing the output data.
             */
            const NNFloat* pOutputKey = pNetwork->GetUnitBuffer("Output");
            NNFloat* pOut = pOutputValue;
            cudaError_t status = cudaMemcpy(pOut, pOutputKey, batch * STRIDE * sizeof(NNFloat), cudaMemcpyDeviceToHost);
            RTERROR(status, "cudaMemcpy GpuBuffer::Download failed");

            for (int i = 0; i < batch; i++)
            {
                /**
                 * @brief Calculates the data point count and assigns it to vDataPoints.
                 *
                 * This statement calculates the count of data points for the current index (pos + i) and assigns it to vDataPoints[i].
                 * The count is determined by subtracting the sparse start position (pOutputDataSet->_vSparseStart[j]) from the sparse end position (pOutputDataSet->_vSparseEnd[j]).
                 *
                 * @param[in] i The loop index representing the current data point.
                 * @param[in] j The calculated index based on pos and i.
                 */
                int j = pos + i;
                vDataPoints[i] = pOutputDataSet->_vSparseEnd[j] - pOutputDataSet->_vSparseStart[j];

                /**
                 * @brief Sets the target values to 1.0 for the corresponding indices.
                 *
                 * This loop iterates over the sparse indices for the current index (j) in pOutputDataSet.
                 * For each sparse index (pOutputDataSet->_vSparseIndex[k]), it sets the corresponding position in pTarget to 1.0f.
                 *
                 * @param[in] k The loop index representing the current sparse index.
                 */
                for (size_t k = pOutputDataSet->_vSparseStart[j]; k < pOutputDataSet->_vSparseEnd[j]; k++)
                {
                    pTarget[pOutputDataSet->_vSparseIndex[k]] = 1.0f;
                }

                /**
                 * @brief Filters past values by setting them to 0.0 in pOut.
                 *
                 * This loop iterates over the sparse indices for the current index (j) in pInputDataSet.
                 * For each sparse index (pInputDataSet->_vSparseIndex[k]), it sets the corresponding position in pOut to 0.0f.
                 * This operation is performed if bFilterPast is true.
                 *
                 * @param[in] k The loop index representing the current sparse index.
                 */
                if (bFilterPast)
                {
                    for (size_t k = pInputDataSet->_vSparseStart[j]; k < pInputDataSet->_vSparseEnd[j]; k++)
                    {
                        pOut[pInputDataSet->_vSparseIndex[k]] = 0.0f;
                    }
                }

                /**
                 * @brief Advances the pointers pTarget and pOut.
                 *
                 * These statements advance the pointers pTarget and pOut by STRIDE positions.
                 */
                pTarget += STRIDE;
                pOut += STRIDE;
            }
            /**
             * @brief Uploads pbTarget data from the CPU to the GPU.
             *
             * This statement uploads the data stored in pbTarget from the CPU to the GPU.
             */
            pbTarget->Upload();

            /**
             * @brief Uploads pbOutput data from the CPU to the GPU.
             *
             * This statement uploads the data stored in pbOutput from the CPU to the GPU.
             */
            pbOutput->Upload();

            /**
             * @brief Calculates the output using kCalculateOutput function.
             *
             * This statement calls the kCalculateOutput function to calculate the output using the provided input data.
             * The function takes pbOutput->_pDevData, pbTarget->_pDevData, pbKey->_pDevData, pbFValue->_pDevData, batch, STRIDE, and K as arguments.
             */
            kCalculateOutput(pbOutput->_pDevData, pbTarget->_pDevData, pbKey->_pDevData, pbFValue->_pDevData, batch, STRIDE, K);

            /**
             * @brief Downloads pbKey data from the GPU to the CPU.
             *
             * This statement downloads the data stored in pbKey from the GPU to the CPU.
             */
            pbKey->Download();

            /**
             * @brief Downloads pbFValue data from the GPU to the CPU.
             *
             * This statement downloads the data stored in pbFValue from the GPU to the CPU.
             */
            pbFValue->Download();

            if (bMultiGPU)
            {

                /**
                 * @brief Performs an MPI reduction operation on vDataPoints.
                 *
                 * This statement performs an MPI reduction operation on the vDataPoints vector, summing the values across all MPI processes and storing the result in vDataPoints.
                 * The reduction operation is performed using MPI_SUM.
                 *
                 * @note This operation is performed only by the GPU with ID 0.
                 */
                MPI_Reduce((getGpu()._id == 0) ? MPI_IN_PLACE : vDataPoints.data(), vDataPoints.data(), batch, MPI_UINT32_T, MPI_SUM, 0, MPI_COMM_WORLD);

                /**
                 * @brief Calculates the offset and stride for data transfer.
                 *
                 * These statements calculate the offset and stride values to be used for data transfer.
                 * The offset is based on the GPU ID, and the stride is calculated as K multiplied by the number of GPUs.
                 */
                uint32_t offset = K * getGpu()._id;
                uint32_t kstride = K * getGpu()._numprocs;

                /**
                 * @brief Copies pbKey data from device to host.
                 *
                 * This statement copies the data stored in pbKey from the device to the host, storing it in pMultiKey.
                 */
                cudaMemcpy2D(pMultiKey + offset, kstride * sizeof(NNFloat), pbKey->_pDevData, K * sizeof(NNFloat), K * sizeof(NNFloat), batch, cudaMemcpyDefault);

                /**
                 * @brief Copies pbFValue data from device to host.
                 *
                 * This statement copies the data stored in pbFValue from the device to the host, storing it in pMultiFValue.
                 */
                cudaMemcpy2D(pMultiFValue + offset, kstride * sizeof(NNFloat), pbFValue->_pDevData, K * sizeof(NNFloat), K * sizeof(NNFloat), batch, cudaMemcpyDefault);

                /**
                 * @brief Synchronizes the device.
                 *
                 * This statement synchronizes the device, ensuring that all preceding CUDA calls are completed.
                 */
                cudaDeviceSynchronize();

                /**
                 * @brief Synchronizes MPI processes.
                 *
                 * This statement synchronizes all MPI processes within MPI_COMM_WORLD, ensuring that all processes have reached this point before continuing execution.
                 */
                MPI_Barrier(MPI_COMM_WORLD);

                if (getGpu()._id == 0)
                {
                    /**
                     * @brief Calculates the output using kCalculateOutput function.
                     *
                     * This statement calls the kCalculateOutput function to calculate the output using the provided input data.
                     * The function takes pbMultiKey->_pDevData, pbMultiFValue->_pDevData, pbKey->_pDevData, pbFValue->_pDevData, batch, getGpu()._numprocs * K, and K as arguments.
                     *
                     * @note This operation is performed only by the GPU with ID 0.
                     */
                    kCalculateOutput(pbMultiKey->_pDevData, pbMultiFValue->_pDevData, pbKey->_pDevData, pbFValue->_pDevData, batch, getGpu()._numprocs * K, K);
                }
            }

            if (getGpu()._id == 0)
            {
                /**
                 * @brief Downloads pbKey and pbFValue data from the GPU to the CPU.
                 *
                 * This statement downloads the data stored in pbKey and pbFValue from the GPU to the CPU.
                 */
                pbKey->Download();
                pbFValue->Download();

                /**
                 * @brief Retrieves pointers to the downloaded data.
                 *
                 * These statements retrieve pointers to the downloaded data stored in pbKey and pbFValue.
                 */
                NNFloat* pKey = pbKey->_pSysData;
                NNFloat* pValue = pbFValue->_pSysData;

                for (int i = 0; i < batch; i++)
                {
                    /**
                     * @brief Retrieves the data point value.
                     *
                     * This statement retrieves the value of the data point at index i from the vDataPoints vector and stores it in the variable p.
                     */
                    NNFloat p = vDataPoints[i];

                    /**
                     * @brief Initializes variables for evaluation metrics.
                     *
                     * These statements initialize variables tp, fp, idcg, and dcg used for evaluation metrics calculation.
                     */
                    NNFloat tp = 0.0f;
                    NNFloat fp = 0.0f;
                    NNFloat idcg = 0.0f;
                    NNFloat dcg = 0.0f;

                    for (NNFloat pp = 0.0f; pp < p; pp++)
                    {
                        /**
                         * @brief Calculates the ideal discounted cumulative gain (IDCG).
                         *
                         * This loop calculates the IDCG value by summing the inverse of the logarithm (base 2) of pp + 2.
                         */
                        idcg += 1.0f / log2(pp + 2.0f);
                    }

                    for (int j = 0; j < K; j++)
                    {
                        if (pValue[j] == 1.0f)
                        {
                            /**
                             * @brief Updates true positive (tp) and discounted cumulative gain (DCG).
                             *
                             * If the value at index j in pValue is 1.0, it increments tp and updates the DCG value by summing the inverse of the logarithm (base 2) of j + 2.
                             * Otherwise, it increments false positive (fp).
                             */
                            tp++;
                            dcg += 1.0f / log2(static_cast<float>(j + 2));
                        }
                        else
                        {
                            fp++;
                        }

                        /**
                         * @brief Updates evaluation metrics.
                         *
                         * These statements update the evaluation metrics by adding tp / (tp + fp) to vPrecision[j], tp / p to vRecall[j], and dcg / idcg to vNDCG[j].
                         */
                        vPrecision[j] += tp / (tp + fp);
                        vRecall[j] += tp / p;
                        vNDCG[j] += dcg / idcg;
                    }

                    pKey += K;
                    pValue += K;
                }
            }
        }

        /**
         * @brief Deletes the pbKey object.
         *
         * This statement deletes the pbKey object, freeing the allocated memory.
         */
        delete pbKey;

        /**
         * @brief Deletes the pbFValue object.
         *
         * This statement deletes the pbFValue object, freeing the allocated memory.
         */
        delete pbFValue;

        /**
         * @brief Deletes the pbUIValue object.
         *
         * This statement deletes the pbUIValue object, freeing the allocated memory.
         */
        delete pbUIValue;

        /**
         * @brief Deletes the pbTarget object.
         *
         * This statement deletes the pbTarget object, freeing the allocated memory.
         */
        delete pbTarget;

        /**
         * @brief Deletes the pbOutput object.
         *
         * This statement deletes the pbOutput object, freeing the allocated memory.
         */
        delete pbOutput;

        if (bMultiGPU)
        {
            if (getGpu()._id != 0)
            {
                /**
                 * @brief Closes MultiKey IpcMemHandle.
                 *
                 * This statement closes the IpcMemHandle associated with the pMultiKey object.
                 *
                 * @note This operation is performed only for GPUs other than the GPU with ID 0.
                 */
                cudaError_t status = cudaIpcCloseMemHandle(pMultiKey);
                RTERROR(status, "cudaIpcCloseMemHandle: Error closing MultiKey IpcMemHandle");

                /**
                 * @brief Closes MultiFValue IpcMemHandle.
                 *
                 * This statement closes the IpcMemHandle associated with the pMultiFValue object.
                 *
                 * @note This operation is performed only for GPUs other than the GPU with ID 0.
                 */
                status = cudaIpcCloseMemHandle(pMultiFValue);
                RTERROR(status, "cudaIpcCloseMemHandle: Error closing MultiFValue IpcMemHandle");
            }

            /**
             * @brief Deletes the pbMultiKey object.
             *
             * This statement deletes the pbMultiKey object, freeing the allocated memory.
             */
            delete pbMultiKey;

            /**
             * @brief Deletes the pbMultiFValue object.
             *
             * This statement deletes the pbMultiFValue object, freeing the allocated memory.
             */
            delete pbMultiFValue;
        }

        if (getGpu()._id == 0)
        {
            /**
             * @brief Prints evaluation metrics.
             *
             * This loop prints the evaluation metrics, such as precision, recall, and NDCG, to the standard output.
             *
             * @note This operation is performed only for the GPU with ID 0.
             */
            for (int i = 0; i < K; i++)
                std::cout << i + 1 << "," << vPrecision[i] / pNetwork->GetExamples() << "," << vRecall[i] / pNetwork->GetExamples() << "," << vNDCG[i] / pNetwork->GetExamples() << std::endl;
        }
    }

    /**
     * @brief Retrieves the GPU and CPU memory usage.
     *
     * This function retrieves the memory usage of the GPU and CPU and stores the values in the variables totalGPUMemory and totalCPUMemory respectively.
     *
     * @param[out] totalGPUMemory The variable to store the GPU memory usage.
     * @param[out] totalCPUMemory The variable to store the CPU memory usage.
     */
    getGpu().GetMemoryUsage(&totalGPUMemory, &totalCPUMemory);

    if (getGpu()._id == 0)
    {
        /**
         * @brief Prints the GPU memory usage.
         *
         * This statement prints the GPU memory usage to the standard output.
         */
        std::cout << "GPU Memory Usage: " << totalGPUMemory << " KB" << std::endl;

        /**
         * @brief Prints the CPU memory usage.
         *
         * This statement prints the CPU memory usage to the standard output.
         */
        std::cout << "CPU Memory Usage: " << totalCPUMemory << " KB" << std::endl;
    }

    /**
     * @brief Deletes the pNetwork object.
     *
     * This statement deletes the pNetwork object, freeing the allocated memory.
     */
    delete pNetwork;

    /**
     * @brief Deletes the elements of vDataSet.
     *
     * This loop iterates over the elements of vDataSet and deletes each element, freeing the allocated memory.
     */
    for (auto p : vDataSet)
        delete p;

    /**
     * @brief Shuts down the GPU.
     *
     * This statement shuts down the GPU, performing any necessary cleanup operations.
     */
    getGpu().Shutdown();

    /**
     * @brief Returns 0.
     *
     * This statement returns 0 to indicate successful execution of the function.
     */
    return 0;
}
