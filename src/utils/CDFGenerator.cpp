#include <chrono>
#include <vector>
#include <cstdio>
#include <iostream>
#include <unordered_map>
#include "CDFhelper.h"
#include "Utils.h"
#include <iomanip>
#include <filesystem>

namespace fs = std::filesystem;

string DATASET_TYPE_INDICATOR("indicator");
string DATASET_TYPE_ANALOG("analog");

/**
 * Prints the usage information for the NetCDFGenerator.
 */
void printUsageNetCDFGenerator() {
    std::cout << "NetCDFGenerator: Converts a text dataset file into a more compressed NetCDF file." << '\n';
    std::cout << "Usage: generateNetCDF -d <dataset_name> -i <input_text_file> -o <output_netcdf_file> -f <features_index> -s <samples_index> [-c] [-m]" << '\n';
    std::cout << "    -d dataset_name: (required) name for the dataset within the netcdf file." << '\n';
    std::cout << "    -i input_text_file: (required) path to the input text file with records in data format." << '\n';
    std::cout << "    -o output_netcdf_file: (required) path to the output netcdf file that we generate." << '\n';
    std::cout << "    -f features_index: (required) path to the features index file to read-from/write-to." << '\n';
    std::cout << "    -s samples_index: (required) path to the samples index file to read-from/write-to." << '\n';
    std::cout << "    -m : if set, we'll merge the feature index with new features found in the input_text_file. (Cannot be used with -c)." << '\n';
    std::cout << "    -c : if set, we'll create a new feature index from scratch. (Cannot be used with -m)." << '\n';
    std::cout << "    -t type: (default = 'indicator') the type of dataset to generate. Valid values are: [" << std::quoted("indicator") << ", " << std::quoted("analog") << "]." << '\n';
    std::cout << '\n';
}
/**
 * The entry point of the program.
 *
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line arguments.
 * @return The exit code of the program.
 */
int main(int argc, char **argv) {
    /**
     * Checks if the "-h" argument is set and prints the usage of the NetCDF generator.
     * Exits the program with a return code of 1.
     */
    if (isArgSet(argc, argv, "-h")) {
        printUsageNetCDFGenerator();
        std::exit(1);
    }

    /**
     * Retrieves the value of the "-i" argument, which specifies the input text file to convert.
     * If the argument is not provided, prints the usage of the NetCDF generator and exits the program with a return code of 1.
     */
    std::optional<std::string> inputFile = getRequiredArgValue(argc, argv, "-i", "input text file to convert.", [](){
        printUsageNetCDFGenerator();
        std::exit(1);
    });

    /**
     * Retrieves the value of the "-o" argument, which specifies the output NetCDF file to generate.
     * If the argument is not provided, prints the usage of the NetCDF generator and exits the program with a return code of 1.
     */
    std::optional<std::string> outputFile = getRequiredArgValue(argc, argv, "-o", "output NetCDF file to generate.", [](){
        printUsageNetCDFGenerator();
        std::exit(1);
    });

    /**
     * Retrieves the value of the "-d" argument, which specifies the dataset name for the NetCDF metadata.
     * If the argument is not provided, prints the usage of the NetCDF generator and exits the program with a return code of 1.
     */
    std::optional<std::string> datasetName = getRequiredArgValue(argc, argv, "-d", "dataset name for the NetCDF metadata.", [](){
        printUsageNetCDFGenerator();
        std::exit(1);
    });

    /**
     * Retrieves the value of the "-f" argument, which specifies the feature index file.
     * If the argument is not provided, prints the usage of the NetCDF generator and exits the program with a return code of 1.
     */
    std::optional<std::string> featureIndexFile = getRequiredArgValue(argc, argv, "-f", "feature index file.", [](){
        printUsageNetCDFGenerator();
        std::exit(1);
    });

    /**
     * Retrieves the value of the "-s" argument, which specifies the samples index file.
     * If the argument is not provided, prints the usage of the NetCDF generator and exits the program with a return code of 1.
     */
    std::optional<std::string> sampleIndexFile = getRequiredArgValue(argc, argv, "-s", "samples index file.", [](){
        printUsageNetCDFGenerator();
        std::exit(1);
    });

    /**
     * Checks if the "-c" argument is set to create a new feature index.
     * If the argument is set, displays a message indicating the file that will be created or overwritten.
     */
    bool createFeatureIndex = isArgSet(argc, argv, "-c");
    if (createFeatureIndex) {
        std::cout << "Flag -c is set. Will create a new feature file and overwrite: " << featureIndexFile.value() << std::endl;
    }

    /**
     * Checks if the "-m" argument is set to merge with an existing feature index.
     * If the argument is set, displays a message indicating the file that will be merged and overwritten.
     */
    bool mergeFeatureIndex = isArgSet(argc, argv, "-m");
    if (mergeFeatureIndex) {
        std::cout << "Flag -m is set. Will merge with existing feature file and overwrite: " << featureIndexFile.value() << std::endl;
    }

    /**
     * Checks if both createFeatureIndex and mergeFeatureIndex are set, which is an invalid combination.
     * Displays an error message, prints the usage of the NetCDF generator, and exits the program with a return code of 1.
     */
    if (createFeatureIndex && mergeFeatureIndex) {
        std::cout << "Error: Cannot create (-c) and update existing (-u) feature index. Please select only one.";
        printUsageNetCDFGenerator();
        std::exit(1);
    }

    /**
     * Determines if the feature index needs to be updated, either by creating a new one or merging with an existing one.
     */
    bool updateFeatureIndex = createFeatureIndex || mergeFeatureIndex;

    /**
     * Retrieves the value of the "-t" argument, which specifies the dataset type.
     * If the value is not "indicator" or "analog", displays an error message and exits the program with a return code of 1.
     */
    std::string dataType = getOptionalArgValue(argc, argv, "-t", "indicator");
    if (dataType != DATASET_TYPE_INDICATOR && dataType != DATASET_TYPE_ANALOG) {
        std::cout << "Error: Unknown dataset type [" << dataType << "].";
        std::cout << " Please select one of {" << DATASET_TYPE_INDICATOR << "," << DATASET_TYPE_ANALOG << "}" << std::endl;
        std::exit(1);
    }
    std::cout << "Generating dataset of type: " << dataType << std::endl;

    /**
     * Initializes two unordered maps to store the feature and sample indexes.
     */
    std::unordered_map<std::string, unsigned int> mFeatureIndex;
    std::unordered_map<std::string, unsigned int> mSampleIndex;

    /**
     * Retrieves the current time before performing the file operations.
     */
    auto const start = std::chrono::steady_clock::now();

    /**
     * Checks if the sample index file exists.
     * If the file does not exist, displays a message indicating that a new file will be created.
     * If the file exists, displays a message indicating that the sample index will be loaded from the file.
     * If the loading fails, exits the program with a return code of 1.
     */
    if (!fs::exists(sampleIndexFile)) {
        std::cout << "Will create a new samples index file: " << sampleIndexFile << std::endl;
    } else {
        std::cout << "Loading sample index from: " << sampleIndexFile << std::endl;
        if (!loadIndexFromFile(mSampleIndex, sampleIndexFile, std::cout)) {
            std::exit(1);
        }
    }

    /**
     * Checks if the createFeatureIndex flag is set to true.
     * If true, displays a message indicating that a new feature index file will be created.
     */
    if (createFeatureIndex) {
        std::cout << "Will create a new features index file: " << featureIndexFile << std::endl;
    } 
    /**
     * Checks if the feature index file does not exist.
     * If true, displays an error message indicating that a valid feature index file could not be found and exits the program with a return code of 1.
     * If false, displays a message indicating that the feature index will be loaded from the file.
     * If loading the feature index fails, exits the program with a return code of 1.
     */
    else if (!fs::exists(featureIndexFile)) {
        std::cout << "Error: Cannot find a valid feature index file: " << featureIndexFile << std::endl;
        std::exit(1);
    } else {
        std::cout << "Loading feature index from: " << featureIndexFile << std::endl;
        if (!loadIndexFromFile(mFeatureIndex, featureIndexFile, std::cout)) {
            std::exit(1);
        }
    }

    /**
     * Vector to store the start positions of sparse data.
     */
    std::vector<unsigned int> vSparseStart;

    /**
     * Vector to store the end positions of sparse data.
     */
    std::vector<unsigned int> vSparseEnd;

    /**
     * Vector to store the indexes of sparse data.
     */
    std::vector<unsigned int> vSparseIndex;

    /**
     * Vector to store the sparse data values.
     */
    std::vector<float> vSparseData;

    /**
     * Generates the NetCDF indexes and assigns the result to 'success' and 'result' variables.
     * If the generation fails, exits the program with a return code of 1.
     */
    auto [success, result] = generateNetCDFIndexes(inputFile, updateFeatureIndex, featureIndexFile, sampleIndexFile, mFeatureIndex, mSampleIndex, vSparseStart, vSparseEnd, vSparseIndex, vSparseData, std::cout);
    if (!success) {
        std::exit(1);
    }

    /**
     * Checks the dataset type and writes the NetCDF file accordingly.
     * If the dataset type is "analog", writes the NetCDF file with the sparse data.
     * If the dataset type is "indicator", writes the NetCDF file without the sparse data.
     */
    if (dataType.compare(DATASET_TYPE_ANALOG) == 0) {
        writeNetCDFFile(vSparseStart, vSparseEnd, vSparseIndex, vSparseData, outputFile, datasetName, mFeatureIndex.size());
    } else {
        writeNetCDFFile(vSparseStart, vSparseEnd, vSparseIndex, outputFile, datasetName, mFeatureIndex.size());
    }

    /**
     * Calculates the elapsed time for generating the NetCDF file.
     */
    const auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    /**
     * Displays the total time taken for generating the NetCDF file.
     */
    std::cout << "Total time for generating NetCDF: " << elapsed.count() << " secs. " << std::endl;

}
