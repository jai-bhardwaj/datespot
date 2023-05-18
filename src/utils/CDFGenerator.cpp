#include <chrono>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <iomanip>
#include <filesystem>

#include <optional>
#include <string>

#include "CDFhelper.h"
#include "Utils.h"

namespace fs = std::filesystem;

const std::string DATASET_TYPE_INDICATOR = "indicator";
const std::string DATASET_TYPE_ANALOG = "analog";

/**
 * Prints the usage information for the NetCDFGenerator.
 */
void printUsageNetCDFGenerator() {
    std::cout << "NetCDFGenerator: Converts a text dataset file into a more compressed NetCDF file.\n";
    std::cout << "Usage: generateNetCDF -d <dataset_name> -i <input_text_file> -o <output_netcdf_file> -f <features_index> -s <samples_index> [-c] [-m]\n";
    std::cout << "    -d dataset_name: (required) name for the dataset within the netcdf file.\n";
    std::cout << "    -i input_text_file: (required) path to the input text file with records in data format.\n";
    std::cout << "    -o output_netcdf_file: (required) path to the output netcdf file that we generate.\n";
    std::cout << "    -f features_index: (required) path to the features index file to read-from/write-to.\n";
    std::cout << "    -s samples_index: (required) path to the samples index file to read-from/write-to.\n";
    std::cout << "    -m : if set, we'll merge the feature index with new features found in the input_text_file. (Cannot be used with -c).\n";
    std::cout << "    -c : if set, we'll create a new feature index from scratch. (Cannot be used with -m).\n";
    std::cout << "    -t type: (default = 'indicator') the type of dataset to generate. Valid values are: [" << std::quoted("indicator") << ", " << std::quoted("analog") << "].\n";
    std::cout << '\n';
}

/**
 * Entry point of the program.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line argument strings.
 * @return The exit code of the program.
 */
int main(int argc, char** argv) {
    if (isArgSet(argc, argv, "-h")) {
        printUsageNetCDFGenerator();
        std::exit(1);
    }

    auto inputFile = getRequiredArgValue<std::string>(argc, argv, "-i", "input text file to convert.", [](){
        printUsageNetCDFGenerator();
        std::exit(1);
    });

    auto outputFile = getRequiredArgValue<std::string>(argc, argv, "-o", "output NetCDF file to generate.", [](){
        printUsageNetCDFGenerator();
        std::exit(1);
    });

    auto datasetName = getRequiredArgValue<std::string>(argc, argv, "-d", "dataset name for the NetCDF metadata.", [](){
        printUsageNetCDFGenerator();
        std::exit(1);
    });

    auto featureIndexFile = getRequiredArgValue<std::string>(argc, argv, "-f", "feature index file.", [](){
        printUsageNetCDFGenerator();
        std::exit(1);
    });

    auto sampleIndexFile = getRequiredArgValue<std::string>(argc, argv, "-s", "samples index file.", [](){
        printUsageNetCDFGenerator();
        std::exit(1);
    });

    bool createFeatureIndex = isArgSet(argc, argv, "-c");
    if (createFeatureIndex) {
        std::cout << "Flag -c is set. Will create a new feature file and overwrite: " << featureIndexFile << '\n';
    }

    bool mergeFeatureIndex = isArgSet(argc, argv, "-m");
    if (mergeFeatureIndex) {
        std::cout << "Flag -m is set. Will merge with existing feature file and overwrite: " << featureIndexFile << '\n';
    }

    if (createFeatureIndex && mergeFeatureIndex) {
        std::cout << "Error: Cannot create (-c) and update existing (-u) feature index. Please select only one.\n";
        printUsageNetCDFGenerator();
        std::exit(1);
    }

    bool updateFeatureIndex = createFeatureIndex || mergeFeatureIndex;

    std::string dataType = getOptionalArgValue(argc, argv, "-t", "indicator");
    if (dataType != DATASET_TYPE_INDICATOR && dataType != DATASET_TYPE_ANALOG) {
        std::cout << "Error: Unknown dataset type [" << dataType << "].";
        std::cout << " Please select one of {" << DATASET_TYPE_INDICATOR << "," << DATASET_TYPE_ANALOG << "}\n";
        std::exit(1);
    }
    std::cout << "Generating dataset of type: " << dataType << '\n';

    std::unordered_map<std::string, unsigned int> mFeatureIndex;
    std::unordered_map<std::string, unsigned int> mSampleIndex;

    auto const start = std::chrono::steady_clock::now();

    if (!fs::exists(sampleIndexFile)) {
        std::cout << "Will create a new samples index file: " << sampleIndexFile << '\n';
    } else {
        std::cout << "Loading sample index from: " << sampleIndexFile << '\n';
        if (!loadIndexFromFile(mSampleIndex, sampleIndexFile, std::cout)) {
            std::exit(1);
        }
    }

    if (createFeatureIndex) {
        std::cout << "Will create a new features index file: " << featureIndexFile << '\n';
    } else if (!fs::exists(featureIndexFile)) {
        std::cout << "Error: Cannot find a valid feature index file: " << featureIndexFile << '\n';
        std::exit(1);
    } else {
        std::cout << "Loading feature index from: " << featureIndexFile << '\n';
        if (!loadIndexFromFile(mFeatureIndex, featureIndexFile, std::cout)) {
            std::exit(1);
        }
    }

    std::vector<unsigned int> vSparseStart;
    std::vector<unsigned int> vSparseEnd;
    std::vector<unsigned int> vSparseIndex;
    std::vector<float> vSparseData;

    auto [success, result] = generateNetCDFIndexes(
        inputFile,
        updateFeatureIndex,
        featureIndexFile,
        sampleIndexFile,
        mFeatureIndex,
        mSampleIndex,
        vSparseStart,
        vSparseEnd,
        vSparseIndex,
        vSparseData,
        std::cout
    );

    if (!success) {
        std::exit(1);
    }

    if (dataType == DATASET_TYPE_ANALOG) {
        writeNetCDFFile(vSparseStart, vSparseEnd, vSparseIndex, vSparseData, outputFile, datasetName, mFeatureIndex.size());
    } else {
        writeNetCDFFile(vSparseStart, vSparseEnd, vSparseIndex, outputFile, datasetName, mFeatureIndex.size());
    }

    const auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    std::cout << "Total time for generating NetCDF: " << elapsed.count() << " secs.\n";
}
