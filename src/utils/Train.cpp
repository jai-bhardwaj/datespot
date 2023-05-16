#include <cstdio>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <netcdf>
#include <values.h>
#include <stdexcept>
#include <unordered_map>

#include "GpuTypes.h"
#include "NetCDFhelper.h"
#include "Types.h"
#include "Utils.h"

using namespace netCDF;
using namespace netCDF::exceptions;

/**
 * Prints the usage information for the 'train' command.
 */
void printUsageTrain() {
    std::cout << "Train: Trains a neural network given a config and dataset." << std::endl;
    std::cout << "Usage: train -d <dataset_name> -c <config_file> -n <network_file> -i <input_netcdf> -o <output_netcdf> [-b <batch_size>] [-e <num_epochs>]" << std::endl;
    std::cout << "    -c config_file: (required) the JSON config files with network training parameters." << std::endl;
    std::cout << "    -i input_netcdf: (required) path to the netcdf with dataset for the input of the network." << std::endl;
    std::cout << "    -o output_netcdf: (required) path to the netcdf with dataset for expected output of the network." << std::endl;
    std::cout << "    -n network_file: (required) the output trained neural network in NetCDF file." << std::endl;
    std::cout << "    -b batch_size: (default = 1024) the number records/input rows to process in a batch." << std::endl;
    std::cout << "    -e num_epochs: (default = 40) the number passes on the full dataset." << std::endl;
    std::cout << std::endl;
}

/**
 * The entry point of the program.
 *
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line arguments.
 * @return The exit code of the program.
 */
int main(int argc, char** argv)
{
    /**
     * The learning rate.
     */
    constexpr float alpha = std::stof(getOptionalArgValue(argc, argv, "-alpha", "0.025f"));

    /**
     * The weight decay parameter.
     */
    constexpr float lambda = std::stof(getOptionalArgValue(argc, argv, "-lambda", "0.0001f"));

    /**
     * The regularization parameter for L1 regularization.
     */
    constexpr float lambda1 = std::stof(getOptionalArgValue(argc, argv, "-lambda1", "0.0f"));

    /**
     * The momentum factor.
     */
    constexpr float mu = std::stof(getOptionalArgValue(argc, argv, "-mu", "0.5f"));

    /**
     * The regularization parameter for L2 regularization.
     */
    constexpr float mu1 = std::stof(getOptionalArgValue(argc, argv, "-mu1", "0.0f"));

    /**
     * Checks if the help flag is set. If set, prints the usage and exits the program.
     */
    if (isArgSet(argc, argv, "-h")) {
        printUsageTrain();
        std::exit(1);
    }

    /**
     * Retrieves the required argument value specified by the given flag.
     *
     * @param argc The number of command-line arguments.
     * @param argv The array of command-line arguments.
     * @param flag The flag to search for.
     * @param errorMessage The error message to display if the flag is not found.
     * @param printUsageTrain A pointer to the function that prints the train usage.
     * @return The path of the required argument value.
     */
    std::filesystem::path configFileName = getRequiredArgValue(argc, argv, "-c", "config file was not specified.", &printUsageTrain);

    if (!std::filesystem::exists(configFileName)) {
        std::cerr << "Error: Cannot read config file: " << configFileName << std::endl;
        return 1;
    } else {
        std::cout << "Train will use configuration file: " << configFileName << std::endl;
    }

    /**
     * Checks if the specified file exists. If not, displays an error message.
     *
     * @param configFileName The path of the file to check.
     */
    std::filesystem::path inputDataFile = getRequiredArgValue(argc, argv, "-i", "input data file is not specified.", &printUsageTrain);

    if (!std::filesystem::exists(inputDataFile)) {
        std::cerr << "Error: Cannot read input feature index file: " << inputDataFile << std::endl;
        return 1;
    } else {
        std::cout << "Train will use input data file: " << inputDataFile << std::endl;
    }

    /**
     * Checks if the specified file exists. If not, displays an error message.
     *
     * @param inputDataFile The path of the file to check.
     */
    std::filesystem::path outputDataFile = getRequiredArgValue(argc, argv, "-o", "output data file is not specified.", &printUsageTrain);

    if (!std::filesystem::exists(outputDataFile)) {
        std::cerr << "Error: Cannot read output feature index file: " << outputDataFile << std::endl;
        return 1;
    } else {
        std::cout << "Train will use output data file: " << outputDataFile << std::endl;
    }

    /**
     * Checks if the specified file already exists. If it does, displays an error message.
     *
     * @param networkFileName The path of the file to check.
     */
    std::filesystem::path networkFileName = getRequiredArgValue(argc, argv, "-n", "the output network file path is not specified.", &printUsageTrain);

    if (std::filesystem::exists(networkFileName)) {
        std::cerr << "Error: Network file already exists: " << networkFileName << std::endl;
        return 1;
    } else {
        std::cout << "Train will produce network file: " << networkFileName << std::endl;
    }

    /**
     * Set the batch size for training.
     *
     * @param batchSize The batch size to use for training.
     */
    unsigned int batchSize = std::stoi(getOptionalArgValue(argc, argv, "-b", "1024"));
    std::cout << "Train will use batchSize: " << batchSize << std::endl;

    /**
     * Set the number of epochs for training.
     *
     * @param epoch The number of epochs to train for.
     */
    unsigned int epoch = std::stoi(getOptionalArgValue(argc, argv, "-e", "40"));
    std::cout << "Train will use number of epochs: " << epoch << std::endl;
    std::cout << "Train alpha " << alpha << ", lambda " << lambda << ", mu " << mu << ". Please check CDL.txt for meanings" << std::endl;
    std::cout << "Train alpha " << alpha << ", lambda " << lambda << ", lambda1 " << lambda1 << ", mu " << mu << ", mu1 " << mu1 << ". Please check CDL.txt for meanings" << std::endl;    

    /**
     * Startup the GPU and set the random seed.
     *
     * @param argc The number of command-line arguments.
     * @param argv The command-line arguments.
     */
    getGpu().Startup(argc, argv);
    getGpu().SetRandomSeed(FIXED_SEED);

    /**
     * Load the input and output data sets from NetCDF files.
     *
     * @param inputDataFile The file name of the input data.
     * @param outputDataFile The file name of the output data.
     */
    std::vector<DataSetBase*> vDataSetInput = LoadNetCDF(inputDataFile);
    std::vector<DataSetBase*> vDataSetOutput = LoadNetCDF(outputDataFile);

    /**
     * Combine the input and output data sets.
     */
    vDataSetInput.insert(vDataSetInput.end(), vDataSetOutput.begin(), vDataSetOutput.end());

    /**
     * Load the neural network from a JSON configuration file.
     *
     * @param configFileName The file name of the neural network configuration.
     * @param batchSize The batch size to use for training.
     * @param vDataSetInput The vector of input data sets.
     *
     * @return A pointer to the loaded neural network.
     */
    Network* pNetwork = LoadNeuralNetworkJSON(configFileName, batchSize, vDataSetInput);

    /**
     * Load the input and output data sets into the neural network.
     *
     * @param vDataSetInput The vector of input data sets.
     */
    pNetwork->LoadDataSets(vDataSetInput);
    pNetwork->LoadDataSets(vDataSetOutput);

    /**
     * Set the checkpoint for saving the neural network during training.
     *
     * @param networkFileName The file name to save the network checkpoint to.
     * @param checkpointInterval The interval at which to save the checkpoint.
     */
    pNetwork->SetCheckpoint(networkFileName, 10);

    /**
     * Set the position in the data sets to start training from.
     */
    pNetwork->SetPosition(0);

    /**
     * Perform a batch prediction before training.
     */
    pNetwork->PredictBatch();

    /**
     * Save the initial neural network to a NetCDF file.
     *
     * @param networkFileName The file name to save the initial network to.
     */
    pNetwork->SaveNetCDF("initial_network.nc");

    /**
     * Set the training mode of the neural network.
     *
     * @param mode The training mode to set.
     */
    TrainingMode mode = SGD;
    pNetwork->SetTrainingMode(mode);

    /**
     * Perform training using a neural network for a given number of epochs.
     *
     * @param epoch The number of epochs to train for.
     * @param alpha The learning rate.
     * @param lambda The regularization parameter.
     * @param lambda1 The regularization parameter for the first layer.
     * @param mu The momentum rate.
     * @param mu1 The momentum rate for the first layer.
     */
    auto start = std::chrono::steady_clock::now();
    for (unsigned int x = 0; x < epoch; ++x) {
        float error = pNetwork->Train(1, alpha, lambda, lambda1, mu, mu1);
        CWMetric::updateMetrics("Average_Error", error);
        CWMetric::updateMetrics("Epochs", x + 1);
    }

    auto end = std::chrono::steady_clock::now();
    double elapsedTime = std::chrono::duration<double>(end - start).count();

    /**
     * Print the elapsed time for the training process.
     */
    std::cout << "Elapsed time: " << elapsedTime << " seconds" << std::endl;
    }

    auto const end = std::chrono::steady_clock::now();

    /**
     * Update the training time metric.
     */
    CWMetric::updateMetrics("Training_Time", elapsed_seconds(start, end));

    /**
     * Print the total training time.
     */
    std::cout << "Total Training Time: " << elapsed_seconds(start, end) << " seconds" << std::endl;

    int totalGPUMemory;
    int totalCPUMemory;

    /**
     * Get the GPU and CPU memory usage.
     */
    getGpu().GetMemoryUsage(&totalGPUMemory, &totalCPUMemory);

    /**
     * Print the GPU memory usage.
     */
    std::cout << "GPU Memory Usage: " << totalGPUMemory << " KB" << std::endl;

    /**
     * Print the CPU memory usage.
     */
    std::cout << "CPU Memory Usage: " << totalCPUMemory << " KB" << std::endl;

    /**
     * Update the training GPU usage metric.
     */
    CWMetric::updateMetrics("Training_GPU_usage", totalGPUMemory);

    /**
     * Save the neural network to a NetCDF file.
     *
     * @param networkFileName The file name to save the network to.
     */
    pNetwork->SaveNetCDF(networkFileName);

    /**
     * Reset the neural network.
     */
    pNetwork.reset();

    /**
     * Shut down the GPU.
     */
    getGpu().Shutdown();

    return 0;
}

