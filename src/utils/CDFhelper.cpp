#include <cstdio>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <netcdf>
#include <unordered_map>
#include <stdexcept>
#include "Enum.h"
#include "Utils.h"
#include "CDFhelper.h"

using namespace netCDF;
using namespace netCDF::exceptions;

int gLoggingRate = 10000;

/**
 * @brief Loads an index from a file into the provided unordered_map.
 * @param labelsToIndices The unordered_map to store the loaded index.
 * @param inputFile The path to the input file containing the index data.
 * @param outputStream The output stream to write status messages and errors.
 * @return True if the index was successfully loaded, false otherwise.
 */
bool loadIndexFromFile(std::unordered_map<std::string, unsigned int>& labelsToIndices, const std::string& inputFile,
                       std::ostream& outputStream) {
    std::ifstream inputStream(inputFile);
    if (!inputStream.is_open()) {
        outputStream << "Error: Failed to open index file" << std::endl;
        return false;
    }

    return loadIndex(labelsToIndices, inputStream, outputStream);
}

/**
 * @brief Exports the contents of an unordered_map index to a file.
 * @param mLabelToIndex The unordered_map representing the index to export.
 * @param indexFileName The name of the output index file.
 */
void exportIndex(const std::unordered_map<std::string, unsigned int>& mLabelToIndex, const std::string& indexFileName) {
    std::ofstream outputIndexStream(indexFileName);
    for (const auto& entry : mLabelToIndex) {
        outputIndexStream << entry.first << "\t" << entry.second << std::endl;
    }
    outputIndexStream.close();
}

/**
 * @brief Loads an index from the provided input stream into the unordered_map.
 * @param labelsToIndices The unordered_map to store the loaded index.
 * @param inputStream The input stream containing the index data.
 * @param outputStream The output stream to write status messages and errors.
 * @return True if the index was successfully loaded, false otherwise.
 */
bool loadIndex(std::unordered_map<std::string, unsigned int>& labelsToIndices, std::istream& inputStream,
               std::ostream& outputStream) {
    std::string line;
    unsigned int linesProcessed = 0;
    const std::size_t initialIndexSize = labelsToIndices.size();

    while (std::getline(inputStream, line)) {
        std::vector<std::string> vData = split(line, '\t');
        linesProcessed++;

        if (vData.size() == 2 && !vData[0].empty()) {
            labelsToIndices[vData[0]] = std::stoi(vData[1]);
        } else {
            outputStream << "Error: line " << linesProcessed << " contains invalid data" << std::endl;
            return false;
        }
    }

    const std::size_t numEntriesAdded = labelsToIndices.size() - initialIndexSize;
    outputStream << "Number of lines processed: " << linesProcessed << std::endl;
    outputStream << "Number of entries added to index: " << numEntriesAdded << std::endl;

    if (linesProcessed != numEntriesAdded) {
        outputStream << "Error: Number of entries added to index not equal to number of lines processed" << std::endl;
        return false;
    }

    if (inputStream.bad()) {
        outputStream << "Error: " << std::strerror(errno) << std::endl;
        return false;
    }

    return true;
}

/**
 * @brief Loads an index from a file into the provided unordered_map.
 *
 * This function reads the index data from the specified input file and populates
 * the provided unordered_map with the loaded index values. The index file should
 * contain one entry per line in the format "label\tindex", where "label" is a string
 * representing the index label and "index" is an unsigned int representing the index value.
 *
 * @param labelsToIndices The unordered_map to store the loaded index.
 * @param inputFile The path to the input file containing the index data.
 * @param outputStream The output stream to write status messages and errors.
 * @return True if the index was successfully loaded, false otherwise.
 */
bool loadIndexFromFile(std::unordered_map<std::string, unsigned int>& labelsToIndices, const std::string& inputFile,
                       std::ostream& outputStream) {
    std::ifstream inputStream(inputFile);
    if (!inputStream.is_open()) {
        outputStream << "Error: Failed to open index file" << std::endl;
        return false;
    }

    return loadIndex(labelsToIndices, inputStream, outputStream);
}

/**
 * @brief Exports the contents of an unordered_map index to a file.
 *
 * This function writes the contents of the provided unordered_map index to the specified
 * output file. Each entry in the index is written as a separate line in the format "label\tindex",
 * where "label" is the index label and "index" is the associated index value.
 *
 * @param mLabelToIndex The unordered_map representing the index to export.
 * @param indexFileName The name of the output index file.
 */
void exportIndex(const std::unordered_map<std::string, unsigned int>& mLabelToIndex, const std::string& indexFileName) {
    std::ofstream outputIndexStream(indexFileName);
    for (const auto& entry : mLabelToIndex) {
        outputIndexStream << entry.first << "\t" << entry.second << std::endl;
    }
    outputIndexStream.close();
}
/**
 * @brief Parses samples from an input stream and populates the feature and sample indices.
 *
 * This function reads samples from the provided input stream and extracts the feature
 * and sample information. It populates the feature and sample indices based on the parsed data.
 * The function supports updating the feature index if enabled. It also keeps track of progress
 * and reports it to the provided output stream.
 *
 * @param inputStream The input stream containing the samples to parse.
 * @param enableFeatureIndexUpdates Flag indicating whether feature index updates are enabled.
 * @param mFeatureIndex The unordered_map representing the feature index.
 * @param mSampleIndex The unordered_map representing the sample index.
 * @param featureIndexUpdated Output parameter indicating whether the feature index was updated.
 * @param sampleIndexUpdated Output parameter indicating whether the sample index was updated.
 * @param mSignals The map to store the parsed signals for each sample.
 * @param mSignalValues The map to store the signal values for each sample.
 * @param outputStream The output stream for status messages and errors.
 * @return True if the samples were parsed successfully, false otherwise.
 */
bool parseSamples(std::istream& inputStream,
                  const bool enableFeatureIndexUpdates,
                  std::unordered_map<std::string, unsigned int>& mFeatureIndex,
                  std::unordered_map<std::string, unsigned int>& mSampleIndex,
                  bool& featureIndexUpdated,
                  bool& sampleIndexUpdated,
                  std::map<unsigned int, std::vector<unsigned int>>& mSignals,
                  std::map<unsigned int, std::vector<float>>& mSignalValues,
                  std::ostream& outputStream) {
    const auto start = std::chrono::steady_clock::now();
    auto reported = start;
    std::string line;
    int lineNumber = 0;

    while (std::getline(inputStream, line)) {
        lineNumber++;
        if (line.empty()) {
            continue;
        }

        const size_t index = line.find('\t');
        if (index == std::string::npos) {
            outputStream << "Warning: Skipping over malformed line (" << line << ") at line " << lineNumber << std::endl;
            continue;
        }

        const std::string sampleLabel = line.substr(0, index);
        const std::string dataString = line.substr(index + 1);

        unsigned int sampleIndex = 0;
        auto sampleIndexIter = mSampleIndex.find(sampleLabel);
        if (sampleIndexIter != mSampleIndex.end()) {
            sampleIndex = sampleIndexIter->second;
        } else {
            unsigned int newIndex = mSampleIndex.size();
            sampleIndex = newIndex;
            mSampleIndex.emplace(sampleLabel, newIndex);
            sampleIndexUpdated = true;
        }

        std::vector<unsigned int> signals;
        std::vector<float> signalValue;

        std::vector<std::string> dataPointTuples = split(dataString, ':');
        for (const auto& dataPoint : dataPointTuples) {
            std::vector<std::string> dataElems = split(dataPoint, ',');

            if (dataElems.empty() || dataElems[0].empty()) {
                continue;
            }

            const size_t numDataElems = dataElems.size();
            if (numDataElems > 2) {
                outputStream << "Warning: Data point [" << dataPoint << "] at line " << lineNumber << " has more "
                            << "than 1 value for feature (actual value: " << numDataElems << "). "
                            << "Keeping the first value and ignoring subsequent values." << std::endl;
            }

            const std::string& featureName = dataElems[0];
            float featureValue = 0.0;
            if (numDataElems > 1) {
                featureValue = std::stof(dataElems[1]);
            }

            unsigned int featureIndex = 0;
            auto featureIndexIter = mFeatureIndex.find(featureName);
            if (featureIndexIter != mFeatureIndex.end()) {
                featureIndex = featureIndexIter->second;
            } else {
                if (enableFeatureIndexUpdates) {
                    unsigned int index = mFeatureIndex.size();
                    featureIndex = index;
                    mFeatureIndex.emplace(featureName, index);
                    featureIndexUpdated = true;
                } else {
                    continue;
                }
            }
            signals.push_back(featureIndex);
            signalValue.push_back(featureValue);
        }

        mSignals.emplace(sampleIndex, signals);
        mSignalValues.emplace(sampleIndex, signalValue);
        if (mSampleIndex.size() % gLoggingRate == 0) {
            const auto now = std::chrono::steady_clock::now();
            outputStream << "Progress Parsing (Sample " << mSampleIndex.size() << ", ";
            outputStream << "Time " << elapsed_seconds(reported, now) << ", ";
            outputStream << "Total " << elapsed_seconds(start, now) << ")" << std::endl;
            reported = now;
        }
    }

    if (inputStream.bad()) {
        outputStream << "Error: " << strerror(errno) << endl;
        return false;
    }

    return true;
}
/**
 * @brief Imports samples from a given path and populates the feature and sample indices.
 *
 * This function imports samples from the specified path and populates the feature and sample indices.
 * It supports updating the feature index if enabled. The function processes each file in the path
 * and extracts the samples. The extracted samples are then indexed and stored in the appropriate
 * data structures.
 *
 * @param samplesPath The path containing the samples to import.
 * @param enableFeatureIndexUpdates Flag indicating whether feature index updates are enabled.
 * @param mFeatureIndex The unordered_map representing the feature index.
 * @param mSampleIndex The unordered_map representing the sample index.
 * @param featureIndexUpdated Output parameter indicating whether the feature index was updated.
 * @param sampleIndexUpdated Output parameter indicating whether the sample index was updated.
 * @param vSparseStart The vector to store the start indices of sparse samples.
 * @param vSparseEnd The vector to store the end indices of sparse samples.
 * @param vSparseIndex The vector to store the sparse indices of the samples.
 * @param vSparseData The vector to store the sparse data values of the samples.
 * @param outputStream The output stream for status messages and errors.
 * @return True if the samples were imported successfully, false otherwise.
 * @throws std::runtime_error if there are missing signal values for a sample key.
 */
bool importSamplesFromPath(const std::string& samplesPath,
                           const bool enableFeatureIndexUpdates,
                           std::unordered_map<std::string, unsigned int>& mFeatureIndex,
                           std::unordered_map<std::string, unsigned int>& mSampleIndex,
                           bool& featureIndexUpdated,
                           bool& sampleIndexUpdated,
                           std::vector<unsigned int>& vSparseStart,
                           std::vector<unsigned int>& vSparseEnd,
                           std::vector<unsigned int>& vSparseIndex,
                           std::vector<float>& vSparseData,
                           std::ostream& outputStream) {

    featureIndexUpdated = false;
    sampleIndexUpdated = false;

    if (!fileExists(samplesPath)) {
        outputStream << "Error: " << samplesPath << " not found." << std::endl;
        return false;
    }

    std::vector<std::string> files;

    std::map<unsigned int, std::vector<unsigned int>> mSignals;
    std::map<unsigned int, std::vector<float>> mSignalValues;

    if (listFiles(samplesPath, false, files) == 0) {
        outputStream << "Indexing " << files.size() << " files" << std::endl;

        for (const auto& file : files) {
            outputStream << "\tIndexing file: " << file << std::endl;

            std::ifstream inputStream(file);
            if (!inputStream.is_open()) {
                outputStream << "Error: Failed to open index file" << std::endl;
                return false;
            }

            if (!parseSamples(inputStream,
                            enableFeatureIndexUpdates,
                            mFeatureIndex,
                            mSampleIndex,
                            featureIndexUpdated,
                            sampleIndexUpdated,
                            mSignals,
                            mSignalValues,
                            outputStream)) {
                return false;
            }
        }
    }

    for (const auto& [key, signals] : mSignals) {
        vSparseStart.push_back(vSparseIndex.size());

        auto signalValuesIter = mSignalValues.find(key);
        if (signalValuesIter != mSignalValues.end()) {
            const auto& signalValues = signalValuesIter->second;

            for (size_t i = 0; i < signals.size(); ++i) {
                vSparseIndex.push_back(signals[i]);
                vSparseData.push_back(signalValues[i]);
            }
        } else {
            throw std::runtime_error("Missing signal values for key: " + std::to_string(key));
        }

        vSparseEnd.push_back(vSparseIndex.size());
    }

    return true;
}
/**
 * @brief Generates NetCDF indexes for the samples at the specified path.
 *
 * This function generates NetCDF indexes for the samples located at the given path.
 * It imports the samples, populates the feature and sample indices, and stores the
 * sparse representation of the samples. The function also supports updating the
 * feature index if enabled. After generating the indexes, it exports them to the
 * specified feature and sample index files. Progress and status messages are output
 * to the provided output stream.
 *
 * @param samplesPath The path containing the samples to generate NetCDF indexes for.
 * @param enableFeatureIndexUpdates Flag indicating whether feature index updates are enabled.
 * @param outFeatureIndexFileName The file name to export the feature index.
 * @param outSampleIndexFileName The file name to export the sample index.
 * @param mFeatureIndex The unordered_map representing the feature index.
 * @param mSampleIndex The unordered_map representing the sample index.
 * @param vSparseStart The vector to store the start indices of sparse samples.
 * @param vSparseEnd The vector to store the end indices of sparse samples.
 * @param vSparseIndex The vector to store the sparse indices of the samples.
 * @param vSparseData The vector to store the sparse data values of the samples.
 * @param outputStream The output stream for status messages and errors.
 * @return True if the NetCDF indexes were generated successfully, false otherwise.
 */
bool generateNetCDFIndexes(const std::string& samplesPath,
                           const bool enableFeatureIndexUpdates,
                           const std::string& outFeatureIndexFileName,
                           const std::string& outSampleIndexFileName,
                           std::unordered_map<std::string, unsigned int>& mFeatureIndex,
                           std::unordered_map<std::string, unsigned int>& mSampleIndex,
                           std::vector<unsigned int>& vSparseStart,
                           std::vector<unsigned int>& vSparseEnd,
                           std::vector<unsigned int>& vSparseIndex,
                           std::vector<float>& vSparseData,
                           std::ostream& outputStream) {
    bool featureIndexUpdated = false;
    bool sampleIndexUpdated = false;

    if (!importSamplesFromPath(samplesPath,
                               enableFeatureIndexUpdates,
                               mFeatureIndex,
                               mSampleIndex,
                               featureIndexUpdated,
                               sampleIndexUpdated,
                               vSparseStart,
                               vSparseEnd,
                               vSparseIndex,
                               vSparseData,
                               outputStream)) {
        return false;
    }

    if (featureIndexUpdated) {
        exportIndex(mFeatureIndex, outFeatureIndexFileName);
        outputStream << "Exported " << outFeatureIndexFileName << " with " << mFeatureIndex.size() << " entries." << std::endl;
    }

    if (sampleIndexUpdated) {
        exportIndex(mSampleIndex, outSampleIndexFileName);
        outputStream << "Exported " << outSampleIndexFileName << " with " << mSampleIndex.size() << " entries." << std::endl;
    }

    return true;
}
/**
 * @brief Rounds up the maximum feature index to the nearest multiple of 128.
 *
 * This function takes the maximum feature index and rounds it up to the nearest
 * multiple of 128 by adding 127 and then right-shifting by 7 bits.
 *
 * @param maxFeatureIndex The maximum feature index to be rounded up.
 * @return The rounded-up maximum feature index.
 */
unsigned int roundUpMaxIndex(unsigned int maxFeatureIndex) {
    return ((maxFeatureIndex + 127u) >> 7u) << 7u;
}
/**
 * @brief Writes the data to a NetCDF file.
 *
 * This function writes the sparse data to a NetCDF file with the specified file name
 * and dataset name. It also takes the maximum feature index and performs rounding up
 * using the roundUpMaxIndex function. The function creates the necessary dimensions
 * and variables in the NetCDF file and stores the sparse start, end, index, and data.
 * Progress and status messages are printed to the standard output.
 *
 * @param vSparseStart The vector of sparse start indices.
 * @param vSparseEnd The vector of sparse end indices.
 * @param vSparseIndex The vector of sparse indices.
 * @param vSparseData The vector of sparse data values.
 * @param fileName The name of the output NetCDF file.
 * @param datasetName The name of the dataset.
 * @param maxFeatureIndex The maximum feature index.
 * @throws std::runtime_error if there is an error creating or writing to the NetCDF file.
 */
void writeNetCDFFile(const std::vector<unsigned int>& vSparseStart,
                     const std::vector<unsigned int>& vSparseEnd,
                     const std::vector<unsigned int>& vSparseIndex,
                     const std::vector<float>& vSparseData,
                     const std::string& fileName,
                     const std::string& datasetName,
                     unsigned int maxFeatureIndex) {
    std::cout << "Raw max index is: " << maxFeatureIndex << std::endl;
    maxFeatureIndex = roundUpMaxIndex(maxFeatureIndex);
    std::cout << "Rounded up max index to: " << maxFeatureIndex << std::endl;

    try {
        NcFile nc(fileName, NcFile::replace);
        if (nc.isNull()) {
            std::cout << "Error creating output file: " << fileName << std::endl;
            throw std::runtime_error("Error creating NetCDF file.");
        }
        nc.putAtt("datasets", ncUint, 1);
        nc.putAtt("name0", datasetName);
        nc.putAtt("attributes0", ncUint, DataSetEnums::Sparse);
        nc.putAtt("kind0", ncUint, DataSetEnums::Numeric);
        nc.putAtt("dataType0", ncUint, DataSetEnums::Float);
        nc.putAtt("dimensions0", ncUint, 1);
        nc.putAtt("width0", ncUint, maxFeatureIndex);
        NcDim examplesDim = nc.addDim("examplesDim0", vSparseStart.size());
        NcDim sparseDataDim = nc.addDim("sparseDataDim0", vSparseIndex.size());
        NcVar sparseStartVar = nc.addVar("sparseStart0", "uint", "examplesDim0");
        NcVar sparseEndVar = nc.addVar("sparseEnd0", "uint", "examplesDim0");
        NcVar sparseIndexVar = nc.addVar("sparseIndex0", "uint", "sparseDataDim0");
        NcVar sparseDataVar = nc.addVar("sparseData0", ncFloat, sparseDataDim);
        sparseStartVar.putVar(vSparseStart.data());
        sparseEndVar.putVar(vSparseEnd.data());
        sparseIndexVar.putVar(vSparseIndex.data());
        sparseDataVar.putVar(vSparseData.data());

        std::cout << "Created NetCDF file " << fileName << " for dataset " << datasetName << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Caught exception: " << e.what() << "\n";
        throw std::runtime_error("Error writing to NetCDF file.");
    }
}
/**
 * @brief Writes the sparse data to a NetCDF file.
 *
 * This function writes the sparse data to a NetCDF file with the specified file name
 * and dataset name. It takes the sparse start, end, and index vectors along with the
 * maximum feature index. The maximum feature index is aligned using the align function.
 * The function creates the necessary dimensions and variables in the NetCDF file and
 * stores the sparse start, end, and index vectors. Progress and status messages are
 * printed to the standard output.
 *
 * @param vSparseStart The vector of sparse start indices.
 * @param vSparseEnd The vector of sparse end indices.
 * @param vSparseIndex The vector of sparse indices.
 * @param fileName The name of the output NetCDF file.
 * @param datasetName The name of the dataset.
 * @param maxFeatureIndex The maximum feature index.
 * @throws std::runtime_error if there is an error creating or writing to the NetCDF file.
 */
void writeNetCDFFile(const std::vector<unsigned int>& vSparseStart,
                     const std::vector<unsigned int>& vSparseEnd,
                     const std::vector<unsigned int>& vSparseIndex,
                     const std::string& fileName,
                     const std::string& datasetName,
                     unsigned int maxFeatureIndex) {
    std::cout << "Raw max index is: " << maxFeatureIndex << std::endl;
    maxFeatureIndex = align(maxFeatureIndex);
    std::cout << "Rounded up max index to: " << maxFeatureIndex << std::endl;

    try {
        NcFile nc(fileName, NcFile::replace);
        if (nc.isNull()) {
            std::cout << "Error creating output file: " << fileName << std::endl;
            throw std::runtime_error("Error creating NetCDF file.");
        }
        nc.putAtt("datasets", ncUint, 1);
        nc.putAtt("name0", datasetName);
        nc.putAtt("attributes0", ncUint, (DataSetEnums::Sparse + DataSetEnums::Boolean));
        nc.putAtt("kind0", ncUint, DataSetEnums::Numeric);
        nc.putAtt("dataType0", ncUint, DataSetEnums::UInt);
        nc.putAtt("dimensions0", ncUint, 1);
        nc.putAtt("width0", ncUint, maxFeatureIndex);
        NcDim examplesDim = nc.addDim("examplesDim0", vSparseStart.size());
        NcDim sparseDataDim = nc.addDim("sparseDataDim0", vSparseIndex.size());
        NcVar sparseStartVar = nc.addVar("sparseStart0", "uint", "examplesDim0");
        NcVar sparseEndVar = nc.addVar("sparseEnd0", "uint", "examplesDim0");
        NcVar sparseIndexVar = nc.addVar("sparseIndex0", "uint", "sparseDataDim0");
        sparseStartVar.putVar(vSparseStart.data());
        sparseEndVar.putVar(vSparseEnd.data());
        sparseIndexVar.putVar(vSparseIndex.data());

        std::cout << "Created NetCDF file " << fileName << " for dataset " << datasetName << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Caught exception: " << e.what() << "\n";
        throw std::runtime_error("Error writing to NetCDF file.");
    }
}
/**
 * @brief Aligns the given size to the nearest multiple of 128.
 *
 * This function takes a size value and aligns it to the nearest multiple of 128
 * by adding 127 and then right-shifting by 7 bits. The aligned value ensures that
 * it is divisible evenly by 128.
 *
 * @param size The size value to be aligned.
 * @return The aligned size value.
 */
unsigned int align(size_t size) {
    return static_cast<unsigned int>(((size + 127) >> 7) << 7);
}

    /**
     * @brief Adds data to the NetCDF file for a given data index.
     *
     * This function adds the data to the NetCDF file for the specified data index. It takes
     * input samples, their associated features, and other parameters and stores them in the
     * NetCDF file. It also handles aligning the feature dimensionality if specified.
     *
     * @param nc The NetCDF file object.
     * @param dataIndex The index of the data to be added.
     * @param dataName The name of the data.
     * @param mFeatureNameToIndex The map of feature names to their indices.
     * @param vInputSamples The input samples.
     * @param vInputSamplesTime The time information associated with the input samples.
     * @param vInputSamplesData The data values associated with the input samples.
     * @param alignFeatureDimensionality Flag indicating whether to align the feature dimensionality.
     * @param minDate The minimum date value.
     * @param maxDate The maximum date value.
     * @param featureDimensionality The dimensionality of the features.
     * @return True if the data was added successfully, false otherwise.
     */
    bool addDataToNetCDF(NcFile& nc, const long long dataIndex, const std::string& dataName,
                        const std::map<std::string, unsigned int>& mFeatureNameToIndex,
                        const std::vector<std::vector<unsigned int>>& vInputSamples,
                        const std::vector<std::vector<unsigned int>>& vInputSamplesTime,
                        const std::vector<std::vector<float>>& vInputSamplesData,
                        const bool alignFeatureDimensionality, int& minDate, int& maxDate,
                        const int featureDimensionality) {
        std::vector<std::string> vFeatureName(mFeatureNameToIndex.size());
        std::vector<std::string> vFeatureNameC;
        if (!mFeatureNameToIndex.empty()) {
            vFeatureNameC.reserve(mFeatureNameToIndex.size());
            for (const auto& pair : mFeatureNameToIndex) {
                vFeatureNameC.push_back(pair.first);
            }
            std::sort(vFeatureNameC.begin(), vFeatureNameC.end());
            std::transform(vFeatureNameC.begin(), vFeatureNameC.end(), vFeatureName.begin(),
                        [](const std::string& str) { return str; });
        }

        std::string sDataIndex = std::to_string(dataIndex);

        NcDim indToFeatureDim;
        NcVar indToFeatureVar;
        if (!vFeatureNameC.empty()) {
            indToFeatureDim = nc.addDim((std::string("indToFeatureDim") + sDataIndex).c_str(), vFeatureNameC.size());
            indToFeatureVar = nc.addVar((std::string("indToFeature") + sDataIndex).c_str(), "string",
                                        (std::string("indToFeatureDim") + sDataIndex).c_str());
            std::vector<const char*> vFeatureNameCChars;
            vFeatureNameCChars.reserve(vFeatureNameC.size());
            for (const auto& str : vFeatureNameC) {
                vFeatureNameCChars.push_back(str.c_str());
            }
            indToFeatureVar.putVar({0}, {vFeatureNameC.size()}, vFeatureNameCChars.data());
        }

        return !vFeatureNameC.empty();
    }

    /**
     * @brief Calculates the total number of samples from the input samples.
     *
     * This function calculates the total number of samples from the input samples provided
     * in the vector of vectors. It iterates over each vector of samples and accumulates the
     * count of samples.
     *
     * @param vInputSamples The vector of vectors containing the input samples.
     * @return The total number of samples.
     */
    unsigned long long calculateTotalNumberOfSamples(const std::vector<std::vector<unsigned int>>& vInputSamples) {
        unsigned long long numberSamples = 0;
        for (const auto& samples : vInputSamples) {
            numberSamples += samples.size();
        }
        return numberSamples;
    }

    if (numberSamples) {
        std::vector<unsigned int> vSparseInputStart(vInputSamples.size());
        std::vector<unsigned int> vSparseInputEnd(vInputSamples.size());
        std::vector<unsigned int> vSparseInputIndex;
        std::vector<unsigned int> vSparseInputTime;

        for (int i = 0; i < vInputSamples.size(); i++) {
            vSparseInputStart[i] = static_cast<unsigned int>(vSparseInputIndex.size());

            for (unsigned int sample : vInputSamples[i]) {
                vSparseInputIndex.push_back(sample);

                if (!vInputSamplesTime.empty() && !vInputSamplesTime[i].empty()) {
                    unsigned int time = vInputSamplesTime[i][vSparseInputIndex.size() - 1];
                    vSparseInputTime.push_back(time);
                    minDate = std::min(minDate, static_cast<int>(time));
                    maxDate = std::max(maxDate, static_cast<int>(time));
                }
            }

            vSparseInputEnd[i] = static_cast<unsigned int>(vSparseInputIndex.size());
        }

        std::vector<float> vSparseData(vSparseInputIndex.size(), 1.f);
        if (!vInputSamplesData.empty()) {
            int cnt = 0;
            for (const auto& inputData : vInputSamplesData) {
                for (float value : inputData) {
                    vSparseData[cnt++] = value;
                }
            }
        }
        std::cout << vSparseInputIndex.size() << " total input data points." << std::endl;
        std::cout << "write " << dataName << " " << sDataIndex << std::endl;

        unsigned int width = 0;

        if (featureDimensionality > 0 && mFeatureNameToIndex.empty()) {
            width = featureDimensionality;
        } else {
            width = static_cast<unsigned int>(mFeatureNameToIndex.size());
        }

        width = (alignFeatureDimensionality) ? align(width) : width;
        nc.putAtt((std::string("name") + sDataIndex).c_str(), dataName.c_str());
        if (!vInputSamplesData.empty()) {
            nc.putAtt((std::string("attributes") + sDataIndex).c_str(), ncUint, 1);
            nc.putAtt((std::string("kind") + sDataIndex).c_str(), ncUint, 0);
            nc.putAtt((std::string("dataType") + sDataIndex).c_str(), ncUint, 4);
        } else {
            nc.putAtt((std::string("attributes") + sDataIndex).c_str(), ncUint, 3);
            nc.putAtt((std::string("kind") + sDataIndex).c_str(), ncUint, 0);
            nc.putAtt((std::string("dataType") + sDataIndex).c_str(), ncUint, 0);
        }

        nc.putAtt((std::string("dimensions") + sDataIndex).c_str(), ncUint, 1);
        nc.putAtt((std::string("width") + sDataIndex).c_str(), ncUint, width);
        NcDim examplesDim = nc.addDim((std::string("examplesDim") + sDataIndex).c_str(), vSparseInputStart.size());
        NcDim sparseDataDim = nc.addDim((std::string("sparseDataDim") + sDataIndex).c_str(), vSparseInputIndex.size());
        NcVar sparseStartVar = nc.addVar((std::string("sparseStart") + sDataIndex).c_str(), "uint", (std::string("examplesDim") + sDataIndex).c_str());
        NcVar sparseEndVar = nc.addVar((std::string("sparseEnd") + sDataIndex).c_str(), "uint", (std::string("examplesDim") + sDataIndex).c_str());
        NcVar sparseIndexVar = nc.addVar((std::string("sparseIndex") + sDataIndex).c_str(), "uint", (std::string("sparseDataDim") + sDataIndex).c_str());
        NcVar sparseTimeVar;
        if (!vSparseInputTime.empty()) {
            sparseTimeVar = nc.addVar((std::string("sparseTime") + sDataIndex).c_str(), "uint", (std::string("sparseDataDim") + sDataIndex).c_str());
        }

        NcVar sparseDataVar;
        if (!vInputSamplesData.empty()) {
            sparseDataVar = nc.addVar((std::string("sparseData") + sDataIndex).c_str(), ncFloat, sparseDataDim);
        }

        sparseStartVar.putVar(vSparseInputStart.data());
        sparseEndVar.putVar(vSparseInputEnd.data());
        sparseIndexVar.putVar(vSparseInputIndex.data());
        if (!vSparseInputTime.empty()) {
            sparseTimeVar.putVar(vSparseInputTime.data());
        }
        if (!vInputSamplesData.empty()) {
            sparseDataVar.putVar(vSparseData.data());
        }
        return true;
    } else {
        return false;
    }
}
/**
 * @brief Reads the indToFeature data from a NetCDF file.
 *
 * This function reads the indToFeature data from the specified NetCDF file. It retrieves
 * the feature strings associated with the given index and populates them into the provided
 * vector. The function opens the NetCDF file, reads the dimensions and variables, and retrieves
 * the feature strings.
 *
 * @param fname The name of the NetCDF file to read.
 * @param n The index associated with the indToFeature data.
 * @param vFeaturesStr The vector to store the retrieved feature strings.
 */
void readNetCDFindToFeature(const std::string& fname, int n, std::vector<std::string>& vFeaturesStr) {
    NcFile nc(fname, NcFile::read);
    if (nc.isNull()) {
        std::cout << "Error opening binary output file " << fname << std::endl;
        return;
    }

    std::string nstring = std::to_string(n);
    vFeaturesStr.clear();

    NcDim indToFeatureDim = nc.getDim("indToFeatureDim" + nstring);
    if (indToFeatureDim.isNull()) {
        std::cout << "reading error indToFeatureDim" << std::endl;
        return;
    }

    NcVar indToFeatureVar = nc.getVar("indToFeature" + nstring);
    if (indToFeatureVar.isNull()) {
        std::cout << "reading error indToFeature" << std::endl;
        return;
    }

    std::vector<char*> vFeaturesChars(indToFeatureDim.getSize());
    indToFeatureVar.getVar(vFeaturesChars.data());
    vFeaturesStr.resize(indToFeatureDim.getSize());
    for (int i = 0; i < vFeaturesStr.size(); i++) {
        vFeaturesStr[i] = vFeaturesChars[i];
    }
}
/**
 * @brief Reads the samples names from a NetCDF file.
 *
 * This function reads the samples names from the specified NetCDF file. It opens the file,
 * retrieves the necessary dimensions and variables, and populates the provided vector with
 * the samples names. The function reads the samples names from the "samples" variable in the
 * NetCDF file.
 *
 * @param fname The name of the NetCDF file to read.
 * @param vSamplesName The vector to store the retrieved samples names.
 */
void readNetCDFsamplesName(const std::string& fname, std::vector<std::string>& vSamplesName) {

    NcFile nc(fname, NcFile::read);
    if (nc.isNull()) {
        std::cout << "Error opening binary output file " << fname << std::endl;
        return;
    }

    vSamplesName.clear();

    NcDim samplesDim = nc.getDim("samplesDim");
    if (samplesDim.isNull()) {
        std::cout << "reading error examplesDim" << std::endl;
        return;
    }
    NcVar sparseSamplesVar = nc.getVar("samples");
    if (sparseSamplesVar.isNull()) {
        std::cout << "reading error sparseSamplesVar" << std::endl;
        return;
    }
    std::vector<char*> vSamplesChars(samplesDim.getSize());
    vSamplesName.resize(samplesDim.getSize());
    sparseSamplesVar.getVar(vSamplesChars.data());
    for (int i = 0; i < vSamplesChars.size(); i++) {
        vSamplesName[i] = vSamplesChars[i];
    }
}
/**
 * @brief Writes data to a NetCDF file.
 *
 * This function writes the specified input and output data to a NetCDF file. It creates a new NetCDF file
 * with the given file name and populates it with the provided data. The input and output data are associated
 * with the corresponding feature indices and samples names. The function supports writing multiple datasets
 * based on the value of `datasetNum`.
 *
 * @param fileName The name of the NetCDF file to write.
 * @param vSamplesName The vector of samples names.
 * @param mInputFeatureNameToIndex The map of input feature names to indices.
 * @param vInputSamples The vector of input samples data.
 * @param vInputSamplesTime The vector of input samples time data.
 * @param vInputSamplesData The vector of input samples data values.
 * @param mOutputFeatureNameToIndex The map of output feature names to indices.
 * @param vOutputSamples The vector of output samples data.
 * @param vOutputSamplesTime The vector of output samples time data.
 * @param vOutputSamplesData The vector of output samples data values.
 * @param minInpDate The minimum input date.
 * @param maxInpDate The maximum input date.
 * @param minOutDate The minimum output date.
 * @param maxOutDate The maximum output date.
 * @param alignFeatureDimensionality Flag indicating whether to align the feature dimensionality.
 * @param datasetNum The number of datasets to write.
 */
void writeNETCDF(const std::string& fileName, const std::vector<std::string>& vSamplesName,
                const std::map<std::string, unsigned int>& mInputFeatureNameToIndex, std::vector<std::vector<unsigned int>>& vInputSamples,
                const std::vector<std::vector<unsigned int>>& vInputSamplesTime, std::vector<std::vector<float>>& vInputSamplesData,
                const std::map<std::string, unsigned int>& mOutputFeatureNameToIndex, const std::vector<std::vector<unsigned int>>& vOutputSamples,
                const std::vector<std::vector<unsigned int>>& vOutputSamplesTime,
                const std::vector<std::vector<float>>& vOutputSamplesData, int& minInpDate, int& maxInpDate, int& minOutDate,
                int& maxOutDate, const bool alignFeatureDimensionality, const int datasetNum) {

    NcFile nc(fileName, NcFile::replace);
    if (nc.isNull()) {
        std::cout << "Error opening binary output file" << std::endl;
        std::exit(2);
    }

    int countData = 0;
    if (datasetNum >= 1) {
        if (addDataToNetCDF(nc, 0, "input", mInputFeatureNameToIndex, vInputSamples, vInputSamplesTime, vInputSamplesData,
                        alignFeatureDimensionality, minInpDate, maxInpDate)) {
            countData++;
        } else {
            std::cout << "failed to write input data";
            std::exit(1);
        }
    }
    if (datasetNum >= 2) {
        if (addDataToNetCDF(nc, 1, "output", mOutputFeatureNameToIndex, vOutputSamples, vOutputSamplesTime, vOutputSamplesData,
                            alignFeatureDimensionality, minOutDate, maxOutDate)) {
            countData++;
        } else {
            std::cout << "failed to write output data";
            std::exit(1);
        }
    } else {
        std::cout << "number of data sets datasetNum " << datasetNum << " is not implemented";
        std::exit(1);
    }
    nc.putAtt("datasets", std::ncUint, countData);
    std::vector<const char*> vSamplesChars;
    vSamplesChars.reserve(vSamplesName.size());
    for (const auto& sampleName : vSamplesName) {
        vSamplesChars.push_back(sampleName.c_str());
    }
    NcDim samplesDim = nc.addDim("samplesDim", vSamplesName.size());
    NcVar sparseSamplesVar = nc.addVar("samples", "string", "samplesDim");
    sparseSamplesVar.putVar({0}, {vSamplesChars.size()}, vSamplesChars.data());

}
