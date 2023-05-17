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
#include <stdexcept>
#include <filesystem>
#include <unordered_map>
#include <iostream>
#include <chrono>
#include <memory>
#include <values.h>
#include "Utils.h"
#include "Filters.h"
#include "GpuTypes.h"
#include "Types.h"
#include "ranges"
#include "Generator.h"
#include "CDFhelper.h"

using namespace netCDF;
using namespace netCDF::exceptions;

unsigned int INTERVAL_REPORT_PROGRESS = 1000000;

/**
 * @brief Extracts the contents of an unordered map into a vector.
 *
 * This function iterates over the given unordered map and populates a vector
 * with the keys from the map. The values from the map are used as indices
 * to assign the keys to the vector.
 *
 * @param vVectors The vector to store the extracted keys.
 * @param mMaps The unordered map containing key-value pairs.
 */
void extractNNMapsToVectors(vector<string>& vVectors, unordered_map<string, unsigned int>& mMaps)
{
    for (const auto& [key, value] : mMaps)
    {
        vVectors[value] = key;
    }
}

/**
 * @brief Converts a text file to a NetCDF file.
 *
 * This function reads a text file and converts it to a NetCDF file format.
 * It uses the provided feature and signal indexes to generate sparse data,
 * which is then written to the NetCDF file.
 *
 * @param inputTextFile The path to the input text file.
 * @param dataSetName The name of the data set in the NetCDF file.
 * @param outputNCDFFile The path to the output NetCDF file.
 * @param mFeatureIndex The feature index unordered map.
 * @param mSignalIndex The signal index unordered map.
 * @param featureIndexFile The path to the feature index file.
 * @param sampleIndexFile The path to the sample index file.
 */
void convertTextToNetCDF(const std::string& inputTextFile,
                         const std::string& dataSetName,
                         const std::string& outputNCDFFile,
                         const std::unordered_map<std::string, unsigned int>& mFeatureIndex,
                         const std::unordered_map<std::string, unsigned int>& mSignalIndex,
                         const std::string& featureIndexFile,
                         const std::string& sampleIndexFile)
{
    std::vector<unsigned int> vSparseStart;
    std::vector<unsigned int> vSparseEnd;
    std::vector<unsigned int> vSparseIndex;
    std::vector<float> vSparseData;

    if (!generateNetCDFIndexes(inputTextFile, false, featureIndexFile, sampleIndexFile, mFeatureIndex, mSignalIndex, vSparseStart, vSparseEnd, vSparseIndex, vSparseData, std::cout))
    {
        exit(1);
    }

    if (getGpu()._id == 0)
    {
        writeNetCDFFile(vSparseStart, vSparseEnd, vSparseIndex, outputNCDFFile, dataSetName, mFeatureIndex.size());
    }
}

/**
 * @brief Prints the usage information for the predict command.
 *
 * This function outputs the usage information for the predict command,
 * including the available command-line options and their descriptions.
 */
void printUsagePredict()
{
    std::cout << R"(
Predict: Generates predictions from a trained neural network given a signals/input dataset.
Usage: predict -d <dataset_name> -n <network_file> -r <input_text_file> -i <input_feature_index> -o <output_feature_index> -f <filters_json> [-b <batch_size>] [-k <num_recs>] [-l layer] [-s input_signals_index] [-p score_precision]
    -b batch_size: (default = 1024) the number records/input rows to process in a batch.
    -d dataset_name: (required) name for the dataset within the netcdf file.
    -f samples filterFileName .
    -i input_feature_index: (required) path to the feature index file, used to transform input signals to correct input feature vector.
    -k num_recs: (default = 100) The number of predictions (sorted by score to generate). Ignored if -l flag is used.
    -l layer: (default = Output) the network layer to use for predictions. If specified, the raw scores for each node in the layer is output in order.
    -n network_file: (required) the trained neural network in NetCDF file.
    -o output_feature_index: (required) path to the feature index file, used to transform the network output feature vector to appropriate features.
    -p score_precision: (default = 4.3f) precision of the scores in output
    -r input_text_file: (required) path to the file with input signal to use to generate predictions (i.e. recommendations).
    -s filename (required) . to put the output recs to.
)" << std::endl;
}

namespace fs = std::filesystem;
/**
 * @brief The main function of the program.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line argument strings.
 * @return The exit status of the program.
 */
int main(int argc, char** argv)
{
    if (isArgSet(argc, argv, "-h")) {
        printUsagePredict();
        return 1;
    }

    // Get the dataset name from command-line arguments and append the input dataset suffix
    const std::string dataSetName = getRequiredArgValue(argc, argv, "-d", "dataset_name is not specified.", &printUsagePredict) + INPUT_DATASET_SUFFIX;

    // Get the filters file name from command-line arguments
    const std::string filtersFileName = getRequiredArgValue(argc, argv, "-f", "filters_json is not specified.", &printUsagePredict);

    // Check if the filters file exists, if not, display an error message and return an error code
    if (!fs::exists(filtersFileName)) {
        std::cout << "Error: Cannot read filter file: " << filtersFileName << std::endl;
        return 1;
    }

    // Get the input feature index file name from command-line arguments
    const std::string inputIndexFileName = getRequiredArgValue(argc, argv, "-i", "input features index file is not specified.", &printUsagePredict);

    // Check if the input feature index file exists, if not, display an error message and return an error code
    if (!fs::exists(inputIndexFileName)) {
        std::cout << "Error: Cannot read input feature index file: " << inputIndexFileName << std::endl;
        return 1;
    }

    // Get the network file name from command-line arguments
    const std::string networkFileName = getRequiredArgValue(argc, argv, "-n", "network file is not specified.", &printUsagePredict);

    // Check if the network file exists, if not, display an error message and return an error code
    if (!fs::exists(networkFileName)) {
        std::cout << "Error: Cannot read network file: " << networkFileName << std::endl;
        return 1;
    }

    // Get the output feature index file name from command-line arguments
    const std::string outputIndexFileName = getRequiredArgValue(argc, argv, "-o", "output features index file is not specified.", &printUsagePredict);

    // Check if the output feature index file exists, if not, display an error message and return an error code
    if (!fs::exists(outputIndexFileName)) {
        std::cout << "Error: Cannot read output feature index file: " << outputIndexFileName << std::endl;
        return 1;
    }

    // Get the input text file name from command-line arguments
    const std::string recsFileName = getRequiredArgValue(argc, argv, "-r", "input_text_file is not specified.", &printUsagePredict);

    // Check if the input text file exists, if not, display an error message and return an error code
    if (!fs::exists(recsFileName)) {
        std::cout << "Error: Cannot read input_text_file: " << recsFileName << std::endl;
        return 1;
    }

    // Get the output recs file name from command-line arguments
    const std::string recsOutputFileName = getRequiredArgValue(argc, argv, "-s", "filename to put the output recs to.", &printUsagePredict);

    // Get the batch size from command-line arguments and convert it to an unsigned integer
    const unsigned int batchSize = std::stoi(getOptionalArgValue(argc, argv, "-b", "1024"));

    // Get the output count from command-line arguments and convert it to an unsigned integer
    const unsigned int Output = std::stoi(getOptionalArgValue(argc, argv, "-k", "100"));

    // Check if the output count is greater than or equal to 128, display an error message and return an error code
    if (Output >= 128) {
        std::cout << "Error: Optimized Output Only works for top 128. " << Output << " is greater" << std::endl;
        return 1;
    }

    // Get the score format from command-line arguments, use the default value if not specified
    const std::string scoreFormat = getOptionalArgValue(argc, argv, "-p", RecsGenerator::DEFAULT_SCORE_PRECISION);

    // Perform GPU startup and set the random seed
    getGpu().Startup(argc, argv);
    getGpu().SetRandomSeed(FIXED_SEED);

    // Start measuring preprocessing time
    const auto preProcessingStart = std::chrono::steady_clock::now();

    // Create an unordered map to store input feature index
    std::unordered_map<std::string, unsigned int> mInput;

    // Print the input feature index file name being loaded
    std::cout << "Loading input feature index from: " << inputIndexFileName << std::endl;

    // Load the input feature index from file and check if it was successful
    if (!loadIndexFromFile(mInput, inputIndexFileName, std::cout)) {
        return 1; // Return an error code if loading fails
    }

    // Create an unordered map to store signal data
    std::unordered_map<std::string, unsigned int> mSignals;

    // Declare a string variable to store the input NetCDF file name
    std::string inputNetCDFFileName;

    // Create a string representing the prefix for dataset files
    const std::string dataSetFilesPrefix = dataSetName + "_predict";

    // Append the file extension to the prefix to get the input NetCDF file name
    inputNetCDFFileName = dataSetFilesPrefix + NETCDF_FILE_EXTENTION;

    // Create strings for feature and sample index files
    const std::string featureIndexFile = dataSetFilesPrefix + ".featuresIndex";
    const std::string sampleIndexFile = dataSetFilesPrefix + ".samplesIndex";

    // Convert text files to NetCDF format using provided conversion function
    convertTextToNetCDF(recsFileName, dataSetName, inputNetCDFFileName, mInput, mSignals, featureIndexFile, sampleIndexFile);

    // Check if the current process is running on GPU with id 0
    if (getGpu()._id == 0) {
        // Print the number of network input nodes and entries to generate predictions for
        std::cout << "Number of network input nodes: " << mInput.size() << std::endl;
        std::cout << "Number of entries to generate predictions for: " << mSignals.size() << std::endl;

        // Update metrics using CWMetric::updateMetrics() function
        CWMetric::updateMetrics("Signals_Size", mSignals.size());
    }

    // Load NetCDF datasets into a vector of DataSetBase pointers
    std::vector<DataSetBase*> vDataSetInput = LoadNetCDF(inputNetCDFFileName);

    // Load the neural network from NetCDF file and create a unique pointer to it
    std::unique_ptr<Network> pNetwork(LoadNeuralNetworkNetCDF(networkFileName, batchSize));

    // Load datasets into the network
    pNetwork->LoadDataSets(vDataSetInput);

    // Create a vector of strings to store signals
    std::vector<std::string> vSignals;

    // Extract signal maps to vectors using extractNNMapsToVectors() function
    extractNNMapsToVectors(vSignals, mSignals);

    // Create an unordered map to store output feature index
    std::unordered_map<std::string, unsigned int> mOutput;

    // Print the output feature index file name being loaded
    std::cout << "Loading output feature index from: " << outputIndexFileName << std::endl;

    // Load the output feature index from file and check if it was successful
    if (!loadIndexFromFile(mOutput, outputIndexFileName, std::cout)) {
        return 1; // Return an error code if loading fails
    }

    // Create a vector of strings to store output feature index
    std::vector<std::string> vOutput;

    // Extract output feature maps to vectors using extractNNMapsToVectors() function
    extractNNMapsToVectors(vOutput, mOutput);

    // Load filters using provided function, storing the result in vFilterSet
    FilterConfig* vFilterSet = loadFilters(filtersFileName, recsOutputFileName, mOutput, mSignals);

    // Clear the input, output, and signal maps
    mInput.clear();
    mOutput.clear();
    mSignals.clear();

    // Measure the start time of pre-processing
    const auto preProcessingEnd = std::chrono::steady_clock::now();

    // Print the total time for loading the network and data
    std::cout << "Total time for loading network and data is: " << elapsed_seconds(preProcessingStart, preProcessingEnd) << std::endl;

    // Label for the layer used for generating recommendations
    const std::string recsGenLayerLabel = "Output";

    // Get the layer with the specified label from the network
    const Layer* pLayer = pNetwork->GetLayer(recsGenLayerLabel);

    // Get the dimensions of the layer
    const auto [lx, ly, lz, lw] = pLayer->GetDimensions();

    // Get the batch size used for prediction
    const unsigned int lBatch = pNetwork->GetBatch();

    // Get the buffer size for the output of the specified layer
    const unsigned int outputBufferSize = pNetwork->GetBufferSize(recsGenLayerLabel);

    // Create a RecsGenerator object using the obtained parameters
    std::unique_ptr<RecsGenerator> recsGenerator(new RecsGenerator(lBatch, Output, outputBufferSize, recsGenLayerLabel, scoreFormat));

    // Measure the start time of recommendation generation
    const auto recsGenerationStart = std::chrono::steady_clock::now();

    // Measure the start time for the progress reporter
    auto progressReporterStart = std::chrono::steady_clock::now();

    // Iterate over positions in the network's examples
    for (unsigned long long int pos = 0; pos < pNetwork->GetExamples(); pos += pNetwork->GetBatch())
    {
        // Print the current prediction position
        std::cout << "Predicting from position " << pos << std::endl;

        // Set the position in the network
        pNetwork->SetPosition(pos);

        // Perform batch prediction
        pNetwork->PredictBatch();

        // Generate recommendations using the RecsGenerator
        recsGenerator->generateRecs(pNetwork.get(), Output, vFilterSet, vSignals, vOutput);

        // Check if it's time to report progress
        if ((pos % INTERVAL_REPORT_PROGRESS) < pNetwork->GetBatch() && (pos / INTERVAL_REPORT_PROGRESS) > 0 && getGpu()._id == 0) {
            // Measure the end time of the progress reporting
            const auto progressReporterEnd = std::chrono::steady_clock::now();

            // Calculate the duration for progress reporting
            const auto progressReportDuration = elapsed_seconds(progressReporterStart, progressReporterEnd);

            // Print the elapsed time and update metrics
            std::cout << "Elapsed time after " << pos << " is " << progressReportDuration << std::endl;
            CWMetric::updateMetrics("Prediction_Time_Progress", progressReportDuration);
            CWMetric::updateMetrics("Prediction_Progress", static_cast<unsigned int>(pos));

            // Reset the start time of the progress reporter
            progressReporterStart = std::chrono::steady_clock::now();
        }
    }

    // Measure the end time of recommendation generation
    const auto recsGenerationEnd = std::chrono::steady_clock::now();

    // Calculate the duration of recommendation generation
    const auto recsGenerationDuration = elapsed_seconds(recsGenerationStart, recsGenerationEnd);

    // Check if it's the main GPU and update metrics
    if (getGpu()._id == 0) {
        CWMetric::updateMetrics("Prediction_Time", recsGenerationDuration);
        std::cout << "Total time for Generating recs for " << pNetwork->GetExamples() << " was " << recsGenerationDuration << std::endl;
    }

    // Shutdown the GPU
    getGpu().Shutdown();

    // Return 0 to indicate
    return 0;
}
