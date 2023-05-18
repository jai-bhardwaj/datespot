#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <netcdf>
#include <filesystem>

#include "Generator.h"
#include "CDFhelper.h"
#include "Utils.h"
#include "Filters.h"
#include "GpuTypes.h"
#include "Types.h"

namespace fs = std::filesystem;
using namespace netCDF;
using namespace netCDF::exceptions;

constexpr unsigned int INTERVAL_REPORT_PROGRESS = 1000000;

/**
 * @brief Extracts the values from a map and stores them in a vector.
 * @param vVectors The vector to store the values.
 * @param mMaps The map containing the values.
 */
void extractNNMapsToVectors(std::vector<std::string>& vVectors, const std::map<std::string, unsigned int>& mMaps)
{
    for (const auto& [key, value] : mMaps)
    {
        vVectors[value] = key;
    }
}

/**
 * @brief Converts a text file to NetCDF format using the provided parameters.
 * @param inputTextFile The path to the input text file.
 * @param dataSetName The name for the dataset within the NetCDF file.
 * @param outputNCDFFile The path to the output NetCDF file.
 * @param mFeatureIndex The map containing the feature index.
 * @param mSignalIndex The map containing the signal index.
 * @param featureIndexFile The path to the feature index file.
 * @param sampleIndexFile The path to the sample index file.
 */
void convertTextToNetCDF(const std::string& inputTextFile,
                         const std::string& dataSetName,
                         const std::string& outputNCDFFile,
                         const std::map<std::string, unsigned int>& mFeatureIndex,
                         const std::map<std::string, unsigned int>& mSignalIndex,
                         const std::string& featureIndexFile,
                         const std::string& sampleIndexFile)
{
    std::vector<unsigned int> vSparseStart;
    std::vector<unsigned int> vSparseEnd;
    std::vector<unsigned int> vSparseIndex;
    std::vector<float> vSparseData;

    if (!generateNetCDFIndexes(inputTextFile, false, featureIndexFile, sampleIndexFile, mFeatureIndex, mSignalIndex, vSparseStart, vSparseEnd, vSparseIndex, vSparseData, std::cout))
    {
        throw std::runtime_error("Failed to generate NetCDF indexes.");
    }

    if (getGpu()._id == 0)
    {
        writeNetCDFFile(vSparseStart, vSparseEnd, vSparseIndex, outputNCDFFile, dataSetName, mFeatureIndex.size());
    }
}

/**
 * @brief Prints the usage instructions for the "predict" function.
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

/**
 * @brief Parses a string as an unsigned integer.
 * @param str The string to parse.
 * @return The parsed unsigned integer value.
 */
unsigned int parseUInt(const std::string_view str)
{
    return std::stoi(std::string(str));
}

/**
 * @brief Retrieves the value of a required argument from the command-line arguments.
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line arguments.
 * @param argName The name of the required argument.
 * @param errorMessage The error message to throw if the argument is not found.
 * @return The value of the required argument.
 * @throws std::runtime_error if the required argument is not found.
 */
std::string getRequiredArgValue(int argc, char** argv, const std::string_view argName, const std::string_view errorMessage)
{
    for (int i = 1; i < argc - 1; ++i)
    {
        if (argName == argv[i])
        {
            return std::string(argv[i + 1]);
        }
    }
    throw std::runtime_error(std::string(errorMessage));
}

/**
 * @brief Retrieves the value of an optional argument from the command-line arguments.
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line arguments.
 * @param argName The name of the optional argument.
 * @param defaultValue The default value to return if the argument is not found.
 * @return The value of the optional argument, or the default value if not found.
 */
std::string getOptionalArgValue(int argc, char** argv, const std::string_view argName, const std::string_view defaultValue)
{
    for (int i = 1; i < argc - 1; ++i)
    {
        if (argName == argv[i])
        {
            return std::string(argv[i + 1]);
        }
    }
    return std::string(defaultValue);
}

/**
 * @brief Parses the command-line arguments and returns a map of argument-value pairs.
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line arguments.
 * @return A map containing the argument-value pairs.
 */
std::unordered_map<std::string_view, std::string_view> parseCommandLineArgs(int argc, char** argv)
{
    std::unordered_map<std::string_view, std::string_view> argsMap;
    for (int i = 1; i < argc - 1; ++i)
    {
        argsMap[argv[i]] = argv[i + 1];
    }
    return argsMap;
}

/**
 * @brief Validates that a file exists at the specified path.
 * @param filePath The path to the file.
 * @throws std::runtime_error if the file does not exist.
 */
void validateFileExists(const std::string& filePath)
{
    if (!fs::exists(filePath))
    {
        throw std::runtime_error("Error: Cannot read file: " + filePath);
    }
}

/**
 * @brief The main function for generating predictions from a trained neural network.
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line arguments.
 * @return 0 if the predictions were generated successfully, 1 otherwise.
 */
int main(int argc, char** argv)
{
    if (argc < 2 || std::string_view(argv[1]) == "-h")
    {
        printUsagePredict();
        return 1;
    }

    const std::unordered_map<std::string_view, std::string_view> argsMap = parseCommandLineArgs(argc, argv);

    const std::string dataSetName = std::string(getRequiredArgValue(argc, argv, "-d", "dataset_name is not specified.")) + INPUT_DATASET_SUFFIX;

    const std::string filtersFileName = std::string(getRequiredArgValue(argc, argv, "-f", "filters_json is not specified."));

    validateFileExists(filtersFileName);

    const std::string inputIndexFileName = std::string(getRequiredArgValue(argc, argv, "-i", "input features index file is not specified."));

    validateFileExists(inputIndexFileName);

    const std::string networkFileName = std::string(getRequiredArgValue(argc, argv, "-n", "network file is not specified."));

    validateFileExists(networkFileName);

    const std::string outputIndexFileName = std::string(getRequiredArgValue(argc, argv, "-o", "output features index file is not specified."));

    validateFileExists(outputIndexFileName);

    const std::string recsFileName = std::string(getRequiredArgValue(argc, argv, "-r", "input_text_file is not specified."));

    validateFileExists(recsFileName);

    const std::string recsOutputFileName = std::string(getRequiredArgValue(argc, argv, "-s", "filename to put the output recs to."));

    const unsigned int batchSize = parseUInt(getOptionalArgValue(argc, argv, "-b", "1024"));

    const unsigned int Output = parseUInt(getOptionalArgValue(argc, argv, "-k", "100"));

    if (Output >= 128) {
        std::cout << "Error: Optimized Output Only works for top 128. " << Output << " is greater" << std::endl;
        return 1;
    }

    const std::string scoreFormat = std::string(getOptionalArgValue(argc, argv, "-p", RecsGenerator::DEFAULT_SCORE_PRECISION));

    getGpu().Startup(argc, argv);
    getGpu().SetRandomSeed(FIXED_SEED);

    const auto preProcessingStart = std::chrono::steady_clock::now();

    std::map<std::string, unsigned int> mInput;
    std::map<std::string, unsigned int> mSignals;

    std::cout << "Loading input feature index from: " << inputIndexFileName << std::endl;

    if (!loadIndexFromFile(mInput, inputIndexFileName, std::cout)) {
        return 1;
    }

    std::string inputNetCDFFileName;

    const std::string dataSetFilesPrefix = dataSetName + "_predict";

    inputNetCDFFileName = dataSetFilesPrefix + NETCDF_FILE_EXTENTION;

    const std::string featureIndexFile = dataSetFilesPrefix + ".featuresIndex";
    const std::string sampleIndexFile = dataSetFilesPrefix + ".samplesIndex";

    convertTextToNetCDF(recsFileName, dataSetName, inputNetCDFFileName, mInput, mSignals, featureIndexFile, sampleIndexFile);

    if (getGpu()._id == 0) {
        std::cout << "Number of network input nodes: " << mInput.size() << std::endl;
        std::cout << "Number of entries to generate predictions for: " << mSignals.size() << std::endl;

        CWMetric::updateMetrics("Signals_Size", mSignals.size());
    }

    std::vector<DataSetBase*> vDataSetInput = LoadNetCDF(inputNetCDFFileName);

    std::unique_ptr<Network> pNetwork(LoadNeuralNetworkNetCDF(networkFileName, batchSize));

    pNetwork->LoadDataSets(vDataSetInput);

    std::vector<std::string> vSignals;

    extractNNMapsToVectors(vSignals, mSignals);

    std::map<std::string, unsigned int> mOutput;

    std::cout << "Loading output feature index from: " << outputIndexFileName << std::endl;

    if (!loadIndexFromFile(mOutput, outputIndexFileName, std::cout)) {
        return 1;
    }

    std::vector<std::string> vOutput;

    extractNNMapsToVectors(vOutput, mOutput);

    FilterConfig* vFilterSet = loadFilters(filtersFileName, recsOutputFileName, mOutput, mSignals);

    const auto preProcessingEnd = std::chrono::steady_clock::now();

    std::cout << "Total time for loading network and data is: " << elapsed_seconds(preProcessingStart, preProcessingEnd) << std::endl;

    const std::string recsGenLayerLabel = "Output";

    const Layer* pLayer = pNetwork->GetLayer(recsGenLayerLabel);

    const auto [lx, ly, lz, lw] = pLayer->GetDimensions();

    const unsigned int lBatch = pNetwork->GetBatch();

    const unsigned int outputBufferSize = pNetwork->GetBufferSize(recsGenLayerLabel);

    std::unique_ptr<RecsGenerator> recsGenerator(new RecsGenerator(lBatch, Output, outputBufferSize, recsGenLayerLabel, scoreFormat));

    const auto recsGenerationStart = std::chrono::steady_clock::now();

    auto progressReporterStart = std::chrono::steady_clock::now();

    for (unsigned long long int pos = 0; pos < pNetwork->GetExamples(); pos += pNetwork->GetBatch())
    {
        std::cout << "Predicting from position " << pos << std::endl;

        pNetwork->SetPosition(pos);

        pNetwork->PredictBatch();

        recsGenerator->generateRecs(pNetwork.get(), Output, vFilterSet, vSignals, vOutput);

        if ((pos % INTERVAL_REPORT_PROGRESS) < pNetwork->GetBatch() && (pos / INTERVAL_REPORT_PROGRESS) > 0 && getGpu()._id == 0) {
            const auto progressReporterEnd = std::chrono::steady_clock::now();

            const auto progressReportDuration = elapsed_seconds(progressReporterStart, progressReporterEnd);

            std::cout << "Elapsed time after " << pos << " is " << progressReportDuration << std::endl;
            CWMetric::updateMetrics("Prediction_Time_Progress", progressReportDuration);
            CWMetric::updateMetrics("Prediction_Progress", static_cast<unsigned int>(pos));

            progressReporterStart = std::chrono::steady_clock::now();
        }
    }

    const auto recsGenerationEnd = std::chrono::steady_clock::now();

    const auto recsGenerationDuration = elapsed_seconds(recsGenerationStart, recsGenerationEnd);

    if (getGpu()._id == 0) {
        CWMetric::updateMetrics("Prediction_Time", recsGenerationDuration);
        std::cout << "Total time for Generating recs for " << pNetwork->GetExamples() << " was " << recsGenerationDuration << std::endl;
    }

    getGpu().Shutdown();

    return 0;
}
