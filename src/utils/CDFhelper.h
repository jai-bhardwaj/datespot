#pragma once

#include <iosfwd>
#include <map>
#include <string>
#include <vector>
#include <unordered_map>
#include <netcdf>

/**
 * @brief Loads an index from an input stream.
 *
 * This function loads an index from the specified `inputStream`.
 * The loaded index data is stored in the `mLabelToIndex` unordered map.
 * The function outputs information to the specified `outputStream`.
 *
 * @param mLabelToIndex The unordered map to store the loaded index data.
 * @param inputStream The input stream to load the index from.
 * @param outputStream The output stream for informational messages.
 * @return True if the index loading is successful, false otherwise.
 */
bool loadIndex(std::unordered_map<std::string, unsigned int> &mLabelToIndex, std::istream &inputStream,
               std::ostream &outputStream);

/**
 * @brief Loads an index from a file.
 *
 * This function loads an index from the specified `inputFile`.
 * The loaded index data is stored in the `labelsToIndices` unordered map.
 * The function outputs information to the specified `outputStream`.
 *
 * @param labelsToIndices The unordered map to store the loaded index data.
 * @param inputFile The name of the file to load the index from.
 * @param outputStream The output stream for informational messages.
 * @return True if the index loading is successful, false otherwise.
 */
bool loadIndexFromFile(std::unordered_map<std::string, unsigned int> &labelsToIndices, const std::string &inputFile,
                       std::ostream &outputStream);

/**
 * @brief Exports an index to a file.
 *
 * This function exports the index stored in the `mLabelToIndex` unordered map to the specified `indexFileName` file.
 *
 * @param mLabelToIndex The unordered map containing the index data to export.
 * @param indexFileName The name of the file to export the index to.
 */
void exportIndex(std::unordered_map<std::string, unsigned int> &mLabelToIndex, std::string indexFileName);

/**
 * @brief Parses samples from an input stream and generates signal data.
 *
 * This function parses samples from the specified `inputStream` and generates signal data.
 * It can optionally update feature indexes if `enableFeatureIndexUpdates` is set to true.
 * The updated feature indexes are stored in the `mFeatureIndex` unordered map.
 * The generated sample indexes are stored in the `mSampleIndex` unordered map.
 * The function sets the `featureIndexUpdated` flag to true if feature indexes were updated,
 * and sets the `sampleIndexUpdated` flag to true if sample indexes were updated.
 * The generated signal data, including signal IDs and corresponding values,
 * are stored in the `mSignals` and `mSignalValues` maps, respectively.
 * The function also outputs information to the specified `outputStream`.
 *
 * @param inputStream The input stream to parse samples from.
 * @param enableFeatureIndexUpdates Flag indicating whether to update feature indexes.
 * @param mFeatureIndex The unordered map to store the updated feature indexes.
 * @param mSampleIndex The unordered map to store the generated sample indexes.
 * @param featureIndexUpdated Flag indicating whether feature indexes were updated.
 * @param sampleIndexUpdated Flag indicating whether sample indexes were updated.
 * @param mSignals The map to store the generated signal IDs and their corresponding indices.
 * @param mSignalValues The map to store the generated signal values.
 * @param outputStream The output stream for informational messages.
 * @return True if the sample parsing and signal data generation is successful, false otherwise.
 */
bool parseSamples(std::istream &inputStream,
                  const bool enableFeatureIndexUpdates,
                  std::unordered_map<std::string, unsigned int> &mFeatureIndex,
                  std::unordered_map<std::string, unsigned int> &mSampleIndex,
                  bool &featureIndexUpdated,
                  bool &sampleIndexUpdated,
                  std::map<unsigned int, std::vector<unsigned int>> &mSignals,
                  std::map<unsigned int, std::vector<float>> &mSignalValues,
                  std::ostream &outputStream);
/**
 * @brief Imports samples from a specified path and generates sparse data.
 *
 * This function imports samples from the specified `samplesPath` and generates sparse data.
 * It can optionally update feature indexes if `enableFeatureIndexUpdates` is set to true.
 * The updated feature indexes are stored in the `mFeatureIndex` unordered map.
 * The generated sample indexes are stored in the `mSampleIndex` unordered map.
 * The function sets the `featureIndexUpdated` flag to true if feature indexes were updated,
 * and sets the `sampleIndexUpdated` flag to true if sample indexes were updated.
 * The generated sparse data, including start indices, end indices, index values, and data values,
 * are stored in the vectors `vSparseStart`, `vSparseEnd`, `vSparseIndex`, and `vSparseData`, respectively.
 * The function also outputs information to the specified `outputStream`.
 *
 * @param samplesPath The path to import samples from.
 * @param enableFeatureIndexUpdates Flag indicating whether to update feature indexes.
 * @param mFeatureIndex The unordered map to store the updated feature indexes.
 * @param mSampleIndex The unordered map to store the generated sample indexes.
 * @param featureIndexUpdated Flag indicating whether feature indexes were updated.
 * @param sampleIndexUpdated Flag indicating whether sample indexes were updated.
 * @param vSparseStart The vector to store the start indices of the generated sparse data.
 * @param vSparseEnd The vector to store the end indices of the generated sparse data.
 * @param vSparseIndex The vector to store the index values of the generated sparse data.
 * @param vSparseData The vector to store the data values of the generated sparse data.
 * @param outputStream The output stream for informational messages.
 * @return True if the import and sparse data generation is successful, false otherwise.
 */
bool importSamplesFromPath(const std::string &samplesPath,
                           const bool enableFeatureIndexUpdates,
                           std::unordered_map<std::string, unsigned int> &mFeatureIndex,
                           std::unordered_map<std::string, unsigned int> &mSampleIndex,
                           bool &featureIndexUpdated,
                           bool &sampleIndexUpdated,
                           std::vector<unsigned int> &vSparseStart,
                           std::vector<unsigned int> &vSparseEnd,
                           std::vector<unsigned int> &vSparseIndex,
                           std::vector<float> &vSparseData,
                           std::ostream &outputStream);
/**
 * @brief Generates NetCDF indexes and writes sparse data.
 *
 * This function generates NetCDF indexes based on the data from the specified `samplesPath`.
 * It can optionally update feature indexes and write them to the `outFeatureIndexFileName` file.
 * The generated sample indexes are written to the `outSampleIndexFileName` file.
 * The updated feature indexes are stored in the `mFeatureIndex` unordered map.
 * The generated sample indexes are stored in the `mSampleIndex` unordered map.
 * The generated sparse data, including start indices, end indices, index values, and data values,
 * are stored in the vectors `vSparseStart`, `vSparseEnd`, `vSparseIndex`, and `vSparseData`, respectively.
 * The function also outputs information to the specified `outputStream`.
 *
 * @param samplesPath The path to the data samples.
 * @param enableFeatureIndexUpdates Flag indicating whether to update feature indexes.
 * @param outFeatureIndexFileName The name of the file to write the updated feature indexes (optional).
 * @param outSampleIndexFileName The name of the file to write the generated sample indexes.
 * @param mFeatureIndex The unordered map to store the updated feature indexes.
 * @param mSampleIndex The unordered map to store the generated sample indexes.
 * @param vSparseStart The vector to store the start indices of the generated sparse data.
 * @param vSparseEnd The vector to store the end indices of the generated sparse data.
 * @param vSparseIndex The vector to store the index values of the generated sparse data.
 * @param vSparseData The vector to store the data values of the generated sparse data.
 * @param outputStream The output stream for informational messages.
 * @return True if the NetCDF indexes generation is successful, false otherwise.
 */
bool generateNetCDFIndexes(const std::string &samplesPath,
                           const bool enableFeatureIndexUpdates,
                           const std::string &outFeatureIndexFileName,
                           const std::string &outSampleIndexFileName,
                           std::unordered_map<std::string, unsigned int> &mFeatureIndex,
                           std::unordered_map<std::string, unsigned int> &mSampleIndex,
                           std::vector<unsigned int> &vSparseStart,
                           std::vector<unsigned int> &vSparseEnd,
                           std::vector<unsigned int> &vSparseIndex,
                           std::vector<float> &vSparseData,
                           std::ostream &outputStream);

/**
 * @brief Writes sparse data to a NetCDF file.
 *
 * This function writes sparse data to a NetCDF file specified by `fileName`.
 * The sparse data includes start indices, end indices, index values, and corresponding values
 * stored in the vectors `vSparseStart`, `vSparseEnd`, `vSparseIndex`, and `vSparseValue`, respectively.
 * The `datasetName` parameter specifies the name of the dataset in the NetCDF file.
 * The `maxFeatureIndex` parameter represents the maximum feature index.
 *
 * @param vSparseStart The vector of sparse start indices.
 * @param vSparseEnd The vector of sparse end indices.
 * @param vSparseIndex The vector of sparse index values.
 * @param vSparseValue The vector of sparse values.
 * @param fileName The name of the NetCDF file to write.
 * @param datasetName The name of the dataset in the NetCDF file.
 * @param maxFeatureIndex The maximum feature index.
 */
void writeNetCDFFile(std::vector<unsigned int> &vSparseStart,
                     std::vector<unsigned int> &vSparseEnd,
                     std::vector<unsigned int> &vSparseIndex,
                     std::vector<float> &vSparseValue,
                     std::string fileName,
                     std::string datasetName,
                     unsigned int maxFeatureIndex);

/**
 * @brief Writes data to a NetCDF file.
 *
 * This function writes sparse data to a NetCDF file specified by `fileName`.
 * The sparse data includes start indices, end indices, and index values stored in the vectors
 * `vSparseStart`, `vSparseEnd`, and `vSparseIndex`, respectively.
 * The `datasetName` parameter specifies the name of the dataset in the NetCDF file.
 * The `maxFeatureIndex` parameter represents the maximum feature index.
 *
 * @param vSparseStart The vector of sparse start indices.
 * @param vSparseEnd The vector of sparse end indices.
 * @param vSparseIndex The vector of sparse index values.
 * @param fileName The name of the NetCDF file to write.
 * @param datasetName The name of the dataset in the NetCDF file.
 * @param maxFeatureIndex The maximum feature index.
 */
void writeNetCDFFile(std::vector<unsigned int> &vSparseStart,
                     std::vector<unsigned int> &vSparseEnd,
                     std::vector<unsigned int> &vSparseIndex,
                     std::string fileName,
                     std::string datasetName,
                     unsigned int maxFeatureIndex);

/**
 * @brief Rounds up the maximum feature index to the nearest multiple of a specified value.
 *
 * This function takes the maximum feature index `maxFeatureIndex` and rounds it up to the nearest
 * multiple of the specified value. The result is returned as an unsigned integer.
 *
 * @param maxFeatureIndex The maximum feature index to round up.
 * @return The rounded up maximum feature index as an unsigned integer.
 */
unsigned int roundUpMaxIndex(unsigned int maxFeatureIndex);

/**
 * @brief Lists files in a directory.
 *
 * This function lists the files in the specified directory `dirname` and stores their paths in the vector `files`.
 * If the `recursive` flag is set to true, it recursively searches subdirectories as well.
 * The function returns the number of files found.
 *
 * @param dirname The name of the directory.
 * @param recursive Flag indicating whether to search subdirectories recursively.
 * @param files The vector to store the file paths.
 * @return The number of files found.
 */
int listFiles(const std::string &dirname, const bool recursive, std::vector<std::string> &files);

/**
 * @brief Writes data to a NetCDF file.
 *
 * This function writes various data arrays to a NetCDF file specified by `fileName`.
 * The data includes sample names, input feature indices, input samples, input sample times,
 * input sample data, output feature indices, output samples, output sample times, and output sample data.
 * The minimum and maximum dates for input and output samples are also updated.
 * The `alignFeatureDimensionality` flag determines whether the feature dimensionality should be aligned.
 *
 * @param fileName The name of the NetCDF file to write.
 * @param vSamplesName The vector of sample names.
 * @param mInputFeatureNameToIndex The map of input feature names to indices.
 * @param vInputSamples The vector of input samples.
 * @param vInputSamplesTime The vector of input sample times.
 * @param vInputSamplesData The vector of input sample data.
 * @param mOutputFeatureNameToIndex The map of output feature names to indices.
 * @param vOutputSamples The vector of output samples.
 * @param vOutputSamplesTime The vector of output sample times.
 * @param vOutputSamplesData The vector of output sample data.
 * @param minInpDate The minimum date for input samples (updated by the function).
 * @param maxInpDate The maximum date for input samples (updated by the function).
 * @param minOutDate The minimum date for output samples (updated by the function).
 * @param maxOutDate The maximum date for output samples (updated by the function).
 * @param alignFeatureDimensionality Flag indicating whether to align feature dimensionality.
 * @param datasetNum The dataset number.
 */
void writeNETCDF(const std::string& fileName, const std::vector<std::string>& vSamplesName,
                const std::map<std::string, unsigned int>& mInputFeatureNameToIndex, std::vector<std::vector<unsigned int> >& vInputSamples,
                const std::vector<std::vector<unsigned int> >& vInputSamplesTime, std::vector<std::vector<float> >& vInputSamplesData,
                const std::map<std::string, unsigned int>& mOutputFeatureNameToIndex, const std::vector<std::vector<unsigned int> >& vOutputSamples,
                const std::vector<std::vector<unsigned int> >& vOutputSamplesTime,
                const std::vector<std::vector<float> >& vOutputSamplesData, int& minInpDate, int& maxInpDate,
                int& minOutDate, int& maxOutDate, const bool alignFeatureDimensionality, const int datasetNum);

/**
 * @brief Reads the names of samples from a NetCDF file.
 *
 * This function reads the names of samples from a NetCDF file specified by `fname`
 * and stores them in the vector `vSamplesName`.
 *
 * @param fname The name of the NetCDF file.
 * @param vSamplesName The vector to store the sample names.
 */
void readNetCDFsamplesName(const std::string& fname, std::vector<std::string>& vSamplesName);

/**
 * @brief Reads feature strings from a NetCDF file.
 *
 * This function reads feature strings from a NetCDF file specified by `fname`,
 * starting from the feature at index `n`, and stores them in the vector `vFeaturesStr`.
 *
 * @param fname The name of the NetCDF file.
 * @param n The index of the starting feature.
 * @param vFeaturesStr The vector to store the feature strings.
 */
void readNetCDFindToFeature(const std::string& fname, const int n, std::vector<std::string>& vFeaturesStr);

/**
 * @brief Aligns the size to the nearest multiple of the alignment value.
 *
 * This function takes a size value and aligns it to the nearest multiple of the alignment value.
 * The aligned size is returned as an unsigned integer.
 *
 * @param size The size to align.
 * @return The aligned size as an unsigned integer.
 */
unsigned int align(size_t size);

/**
 * @brief Adds data to a NetCDF file.
 *
 * This function adds data to a NetCDF file using the specified parameters.
 *
 * @param nc The NetCDF file to add data to.
 * @param dataIndex The index of the data.
 * @param dataName The name of the data.
 * @param mFeatureNameToIndex A map that maps feature names to their indices.
 * @param vInputSamples A vector of input samples.
 * @param vInputSamplesTime A vector of input sample times.
 * @param vInputSamplesData A vector of input sample data.
 * @param alignFeatureDimensionality A flag indicating whether to align feature dimensionality.
 * @param minDate The minimum date value.
 * @param maxDate The maximum date value.
 * @param featureDimensionality The dimensionality of the feature (default: -1).
 * @return A boolean value indicating whether the data was successfully added.
 */
bool addDataToNetCDF(netCDF::NcFile& nc, const long long dataIndex, const std::string& dataName,
                     const std::map<std::string, unsigned int>& mFeatureNameToIndex,
                     const std::vector<std::vector<unsigned int>>& vInputSamples,
                     const std::vector<std::vector<unsigned int>>& vInputSamplesTime,
                     const std::vector<std::vector<float>>& vInputSamplesData,
                     const bool alignFeatureDimensionality, int& minDate, int& maxDate,
                     const int featureDimensionality = -1);
