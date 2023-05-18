#pragma once

#include <iosfwd>
#include <string>
#include <vector>
#include <unordered_map>
#include <netcdf>

/**
 * @brief Struct representing index data.
 */
struct IndexData {
    std::unordered_map<std::string, unsigned int>& labelsToIndices; /**< Map of labels to indices. */
    std::ostream& outputStream; /**< Output stream for logging. */
};

/**
 * @brief Loads the index data from an input stream.
 * 
 * @param indexData The IndexData struct containing the labels-to-indices map and the output stream.
 * @param inputStream The input stream to read the index data from.
 * @return True if the index data is successfully loaded, false otherwise.
 */
bool loadIndex(IndexData& indexData, std::istream& inputStream);

/**
 * @brief Loads the index data from a file.
 * 
 * @param indexData The IndexData struct containing the labels-to-indices map and the output stream.
 * @param inputFile The path to the input file containing the index data.
 * @return True if the index data is successfully loaded, false otherwise.
 */
bool loadIndexFromFile(IndexData& indexData, const std::string& inputFile);

/**
 * @brief Exports the index data to a file.
 * 
 * @param labelToIndex The labels-to-indices map to export.
 * @param indexFileName The name of the output index file.
 */
void exportIndex(std::unordered_map<std::string, unsigned int>& labelToIndex, std::string indexFileName);

/**
 * @brief Struct representing data for parsing samples.
 */
struct ParseSamplesData {
    std::istream& inputStream; /**< Input stream containing the sample data. */
    bool enableFeatureIndexUpdates; /**< Flag indicating whether to update the feature index. */
    std::unordered_map<std::string, unsigned int>& featureIndex; /**< Map of feature names to indices. */
    std::unordered_map<std::string, unsigned int>& sampleIndex; /**< Map of sample names to indices. */
    bool& featureIndexUpdated; /**< Flag indicating whether the feature index was updated. */
    bool& sampleIndexUpdated; /**< Flag indicating whether the sample index was updated. */
    std::map<unsigned int, std::vector<unsigned int>>& signals; /**< Map of signal indices to signal data. */
    std::map<unsigned int, std::vector<float>>& signalValues; /**< Map of signal indices to signal values. */
    std::ostream& outputStream; /**< Output stream for logging. */
};

/**
 * @brief Parses the sample data.
 * 
 * @param parseData The ParseSamplesData struct containing the sample data and the output stream.
 * @return True if the sample data is successfully parsed, false otherwise.
 */
bool parseSamples(ParseSamplesData& parseData);

/**
 * @brief Struct representing data for importing samples.
 */
struct ImportSamplesData {
    std::string samplesPath; /**< Path to the samples data. */
    bool enableFeatureIndexUpdates; /**< Flag indicating whether to update the feature index. */
    std::unordered_map<std::string, unsigned int>& featureIndex; /**< Map of feature names to indices. */
    std::unordered_map<std::string, unsigned int>& sampleIndex; /**< Map of sample names to indices. */
    bool& featureIndexUpdated; /**< Flag indicating whether the feature index was updated. */
    bool& sampleIndexUpdated; /**< Flag indicating whether the sample index was updated. */
    std::vector<unsigned int>& vSparseStart; /**< Vector containing the start indices of sparse data. */
    std::vector<unsigned int>& vSparseEnd; /**< Vector containing the end indices of sparse data. */
    std::vector<unsigned int>& vSparseIndex; /**< Vector containing the indices of sparse data. */
    std::vector<float>& vSparseData; /**< Vector containing the sparse data. */
    std::ostream& outputStream; /**< Output stream for logging. */
};

/**
 * @brief Imports the samples data from the specified path.
 * 
 * @param importData The ImportSamplesData struct containing the samples data and the output stream.
 * @return True if the samples data is successfully imported, false otherwise.
 */
bool importSamplesFromPath(ImportSamplesData& importData);

/**
 * @brief Struct representing data for generating NetCDF indexes.
 */
struct GenerateNetCDFIndexesData {
    std::string samplesPath; /**< Path to the samples data. */
    bool enableFeatureIndexUpdates; /**< Flag indicating whether to update the feature index. */
    std::string outFeatureIndexFileName; /**< Output file name for the feature index. */
    std::string outSampleIndexFileName; /**< Output file name for the sample index. */
    std::unordered_map<std::string, unsigned int>& featureIndex; /**< Map of feature names to indices. */
    std::unordered_map<std::string, unsigned int>& sampleIndex; /**< Map of sample names to indices. */
    std::vector<unsigned int>& vSparseStart; /**< Vector containing the start indices of sparse data. */
    std::vector<unsigned int>& vSparseEnd; /**< Vector containing the end indices of sparse data. */
    std::vector<unsigned int>& vSparseIndex; /**< Vector containing the indices of sparse data. */
    std::vector<float>& vSparseData; /**< Vector containing the sparse data. */
    std::ostream& outputStream; /**< Output stream for logging. */
};

/**
 * @brief Generates NetCDF indexes from the samples data.
 * 
 * @param generateData The GenerateNetCDFIndexesData struct containing the samples data and the output stream.
 * @return True if the NetCDF indexes are successfully generated, false otherwise.
 */
bool generateNetCDFIndexes(GenerateNetCDFIndexesData& generateData);

/**
 * @brief Writes a NetCDF file with the specified data.
 * 
 * @param vSparseStart Vector containing the start indices of sparse data.
 * @param vSparseEnd Vector containing the end indices of sparse data.
 * @param vSparseIndex Vector containing the indices of sparse data.
 * @param vSparseValue Vector containing the values of sparse data.
 * @param fileName The name of the output NetCDF file.
 * @param datasetName The name of the dataset in the NetCDF file.
 * @param maxFeatureIndex The maximum feature index.
 */
void writeNetCDFFile(const std::vector<unsigned int>& vSparseStart,
                     const std::vector<unsigned int>& vSparseEnd,
                     const std::vector<unsigned int>& vSparseIndex,
                     const std::vector<float>& vSparseValue,
                     std::string fileName,
                     std::string datasetName,
                     unsigned int maxFeatureIndex);

/**
 * @brief Writes a NetCDF file with the specified data.
 * 
 * @param vSparseStart Vector containing the start indices of sparse data.
 * @param vSparseEnd Vector containing the end indices of sparse data.
 * @param vSparseIndex Vector containing the indices of sparse data.
 * @param fileName The name of the output NetCDF file.
 * @param datasetName The name of the dataset in the NetCDF file.
 * @param maxFeatureIndex The maximum feature index.
 */
void writeNetCDFFile(const std::vector<unsigned int>& vSparseStart,
                     const std::vector<unsigned int>& vSparseEnd,
                     const std::vector<unsigned int>& vSparseIndex,
                     std::string fileName,
                     std::string datasetName,
                     unsigned int maxFeatureIndex);

/**
 * @brief Rounds up the maximum feature index to the nearest multiple of 8.
 * 
 * @param maxFeatureIndex The maximum feature index.
 * @return The rounded up maximum feature index.
 */
unsigned int roundUpMaxIndex(unsigned int maxFeatureIndex);

/**
 * @brief Lists files in a directory.
 * 
 * @param dirname The name of the directory.
 * @param recursive Flag indicating whether to list files recursively.
 * @param files Vector to store the list of files.
 * @return The number of files listed.
 */
int listFiles(const std::string& dirname, bool recursive, std::vector<std::string>& files);

/**
 * @brief Writes data to a NETCDF file.
 * 
 * @param fileName The name of the output NETCDF file.
 * @param vSamplesName Vector containing the sample names.
 * @param mInputFeatureNameToIndex Map of input feature names to indices.
 * @param vInputSamples Vector of input samples.
 * @param vInputSamplesTime Vector of input sample times.
 * @param vInputSamplesData Vector of input sample data.
 * @param mOutputFeatureNameToIndex Map of output feature names to indices.
 * @param vOutputSamples Vector of output samples.
 * @param vOutputSamplesTime Vector of output sample times.
 * @param vOutputSamplesData Vector of output sample data.
 * @param minInpDate The minimum input date.
 * @param maxInpDate The maximum input date.
 * @param minOutDate The minimum output date.
 * @param maxOutDate The maximum output date.
 * @param alignFeatureDimensionality Flag indicating whether to align the feature dimensionality.
 * @param datasetNum The dataset number.
 */
void writeNETCDF(const std::string& fileName,
                 const std::vector<std::string>& vSamplesName,
                 const std::map<std::string, unsigned int>& mInputFeatureNameToIndex,
                 std::vector<std::vector<unsigned int>>& vInputSamples,
                 const std::vector<std::vector<unsigned int>>& vInputSamplesTime,
                 std::vector<std::vector<float>>& vInputSamplesData,
                 const std::map<std::string, unsigned int>& mOutputFeatureNameToIndex,
                 const std::vector<std::vector<unsigned int>>& vOutputSamples,
                 const std::vector<std::vector<unsigned int>>& vOutputSamplesTime,
                 const std::vector<std::vector<float>>& vOutputSamplesData,
                 int& minInpDate,
                 int& maxInpDate,
                 int& minOutDate,
                 int& maxOutDate,
                 bool alignFeatureDimensionality,
                 int datasetNum);

/**
 * @brief Reads the sample names from a NetCDF file.
 * 
 * @param fname The name of the NetCDF file.
 * @param vSamplesName Vector to store the sample names.
 */
void readNetCDFsamplesName(const std::string& fname, std::vector<std::string>& vSamplesName);

/**
 * @brief Reads the index-to-feature mapping from a NetCDF file.
 * 
 * @param fname The name of the NetCDF file.
 * @param n The number of features.
 * @param vFeaturesStr Vector to store the feature strings.
 */
void readNetCDFindToFeature(const std::string& fname, int n, std::vector<std::string>& vFeaturesStr);

/**
 * @brief Aligns a size to the nearest multiple of 8.
 * 
 * @param size The size to align.
 * @return The aligned size.
 */
unsigned int align(size_t size);

/**
 * @brief Adds data to a NetCDF file.
 * 
 * @param nc The NetCDF file object.
 * @param dataIndex The data index.
 * @param dataName The name of the data.
 * @param mFeatureNameToIndex Map of feature names to indices.
 * @param vInputSamples Vector of input samples.
 * @param vInputSamplesTime Vector of input sample times.
 * @param vInputSamplesData Vector of input sample data.
 * @param alignFeatureDimensionality Flag indicating whether to align the feature dimensionality.
 * @param minDate The minimum date.
 * @param maxDate The maximum date.
 * @param featureDimensionality The feature dimensionality.
 * @return True if the data is successfully added to the NetCDF file, false otherwise.
 */
bool addDataToNetCDF(netCDF::NcFile& nc,
                     const long long dataIndex,
                     const std::string& dataName,
                     const std::map<std::string, unsigned int>& mFeatureNameToIndex,
                     const std::vector<std::vector<unsigned int>>& vInputSamples,
                     const std::vector<std::vector<unsigned int>>& vInputSamplesTime,
                     const std::vector<std::vector<float>>& vInputSamplesData,
                     bool alignFeatureDimensionality,
                     int& minDate,
                     int& maxDate,
                     int featureDimensionality = -1);
