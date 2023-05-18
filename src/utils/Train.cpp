#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <netcdf>

#include "GpuTypes.h"
#include "NetCDFhelper.h"
#include "Types.h"
#include "Utils.h"

using namespace netCDF;
using namespace netCDF::exceptions;
using namespace std::chrono;
using namespace std::filesystem;

// Function to print the usage of the 'train' program
void printUsageTrain() {
    std::cout << "Train: Trains a neural network given a config and dataset." << std::endl
              << "Usage: train -d <dataset_name> -c <config_file> -n <network_file> -i <input_netcdf> -o <output_netcdf> [-b <batch_size>] [-e <num_epochs>]" << std::endl
              << "    -c config_file: (required) the JSON config files with network training parameters." << std::endl
              << "    -i input_netcdf: (required) path to the netcdf with dataset for the input of the network." << std::endl
              << "    -o output_netcdf: (required) path to the netcdf with dataset for expected output of the network." << std::endl
              << "    -n network_file: (required) the output trained neural network in NetCDF file." << std::endl
              << "    -b batch_size: (default = 1024) the number records/input rows to process in a batch." << std::endl
              << "    -e num_epochs: (default = 40) the number passes on the full dataset." << std::endl
              << std::endl;
}

int main(int argc, char** argv) {
    // Constants with default values for training parameters
    constexpr float alpha = std::stof(getOptionalArgValue(argc, argv, "-alpha", "0.025f"));
    constexpr float lambda = std::stof(getOptionalArgValue(argc, argv, "-lambda", "0.0001f"));
    constexpr float lambda1 = std::stof(getOptionalArgValue(argc, argv, "-lambda1", "0.0f"));
    constexpr float mu = std::stof(getOptionalArgValue(argc, argv, "-mu", "0.5f"));
    constexpr float mu1 = std::stof(getOptionalArgValue(argc, argv, "-mu1", "0.0f"));

    // Check if the '-h' option is set, print usage and exit if true
    if (isArgSet(argc, argv, "-h")) {
        printUsageTrain();
        std::exit(1);
    }

    // Variables for storing command line arguments
    path configFileName;
    path inputDataFile;
    path outputDataFile;
    path networkFileName;
    unsigned int batchSize = 1024;
    unsigned int epoch = 40;

    // Parse command line arguments
    for (int i = 1; i < argc; i += 2) {
        std::string option = argv[i];
        std::string value = argv[i + 1];
        if (option == "-c") {
            configFileName = value;
        } else if (option == "-i") {
            inputDataFile = value;
        } else if (option == "-o") {
            outputDataFile = value;
        } else if (option == "-n") {
            networkFileName = value;
        } else if (option == "-b") {
            batchSize = std::stoi(value);
        } else if (option == "-e") {
            epoch = std::stoi(value);
        }
    }

    // Check if any required file paths are missing, print usage and exit if true
    if (configFileName.empty() || inputDataFile.empty() || outputDataFile.empty() || networkFileName.empty()) {
        printUsageTrain();
        std::exit(1);
    }

    // Check if the config file exists, print error message and exit if not
    if (!exists(configFileName)) {
        std::cerr << "Error: Cannot read config file: " << configFileName << std::endl;
        return 1;
    } else {
        std::cout << "Train will use configuration file: " << configFileName << std::endl;
    }

    // Check if the input data file exists, print error message and exit if not
    if (!exists(inputDataFile)) {
        std::cerr << "Error: Cannot read input feature index file: " << inputDataFile << std::endl;
        return 1;
    } else {
        std::cout << "Train will use input data file: " << inputDataFile << std::endl;
    }

    // Check if the output data file exists, print error message and exit if not
    if (!exists(outputDataFile)) {
        std::cerr << "Error: Cannot read output feature index file: " << outputDataFile << std::endl;
        return 1;
    } else {
        std::cout << "Train will use output data file: " << outputDataFile << std::endl;
    }

    // Check if the network file already exists, print error message and exit if true
    if (exists(networkFileName)) {
        std::cerr << "Error: Network file already exists: " << networkFileName << std::endl;
        return 1;
    } else {
        std::cout << "Train will produce network file: " << networkFileName << std::endl;
    }

    // Print the batch size and number of epochs to be used for training
    std::cout << "Train will use batchSize: " << batchSize << std::endl;
    std::cout << "Train will use number of epochs: " << epoch << std::endl;

    // Print training parameters
    std::cout << "Train alpha " << alpha << ", lambda " << lambda << ", mu " << mu << ". Please check CDL.txt for meanings" << std::endl;
    std::cout << "Train alpha " << alpha << ", lambda " << lambda << ", lambda1 " << lambda1 << ", mu " << mu << ", mu1 " << mu1 << ". Please check CDL.txt for meanings" << std::endl;

    // Startup GPU and set random seed
    getGpu().Startup(argc, argv);
    getGpu().SetRandomSeed(FIXED_SEED);

    // Load input and output datasets from NetCDF files
    std::vector<DataSetBase*> vDataSetInput = LoadNetCDF(inputDataFile);
    std::vector<DataSetBase*> vDataSetOutput = LoadNetCDF(outputDataFile);

    // Combine input and output datasets into a single vector
    vDataSetInput.insert(vDataSetInput.end(), vDataSetOutput.begin(), vDataSetOutput.end());

    // Load neural network from JSON config file with specified batch size and input datasets
    Network* pNetwork = LoadNeuralNetworkJSON(configFileName, batchSize, vDataSetInput);

    // Load input and output datasets into the neural network
    pNetwork->LoadDataSets(vDataSetInput);
    pNetwork->LoadDataSets(vDataSetOutput);

    // Set the checkpoint file for saving the network periodically during training
    pNetwork->SetCheckpoint(networkFileName, 10);

    // Set the starting position in the dataset
    pNetwork->SetPosition(0);

    // Predict the initial batch
    pNetwork->PredictBatch();

    // Save the initial network to a NetCDF file
    pNetwork->SaveNetCDF("initial_network.nc");

    // Set the training mode to SGD (Stochastic Gradient Descent)
    TrainingMode mode = SGD;
    pNetwork->SetTrainingMode(mode);

    // Start the timer
    auto start = steady_clock::now();

    // Training loop for the specified number of epochs
    for (unsigned int x = 0; x < epoch; ++x) {
        // Train the network for one epoch and get the average error
        float error = pNetwork->Train(1, alpha, lambda, lambda1, mu, mu1);

        // Update metrics with the average error and the current epoch
        CWMetric::updateMetrics("Average_Error", error);
        CWMetric::updateMetrics("Epochs", x + 1);
    }

    // Stop the timer
    auto end = steady_clock::now();

    // Calculate the elapsed time in seconds
    double elapsedTime = duration<double>(end - start).count();

    // Print the elapsed time
    std::cout << "Elapsed time: " << elapsedTime << " seconds" << std::endl;

    // Update metrics with the training time
    CWMetric::updateMetrics("Training_Time", elapsedTime);

    // Print the total training time
    std::cout << "Total Training Time: " << elapsedTime << " seconds" << std::endl;

    // Variables for storing GPU and CPU memory usage
    int totalGPUMemory;
    int totalCPUMemory;

    // Get GPU and CPU memory usage
    getGpu().GetMemoryUsage(&totalGPUMemory, &totalCPUMemory);

    // Print GPU and CPU memory usage
    std::cout << "GPU Memory Usage: " << totalGPUMemory << " KB" << std::endl;
    std::cout << "CPU Memory Usage: " << totalCPUMemory << " KB" << std::endl;

    // Update metrics with GPU memory usage
    CWMetric::updateMetrics("Training_GPU_usage", totalGPUMemory);

    // Save the trained network to a NetCDF file
    pNetwork->SaveNetCDF(networkFileName);
    pNetwork.reset();

    // Shutdown the GPU
    getGpu().Shutdown();

    return 0;
}
