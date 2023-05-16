#include <json/json.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <chrono>

#include "Filters.h"
#include "Utils.h"

using namespace Json;

/**
 * @brief The interval at which samples are logged.
 */
const static int gSamplesLoggingInterval = 10000;

/**
 * @brief Updates the records in the given array using the filter.
 *
 * This function multiplies each element in the `xArray` with the corresponding value
 * from the `xFilter`, if present.
 *
 * @param xArray   The array of records to be updated.
 * @param xFilter  The filter containing the values to multiply the records with.
 */
void AbstractFilter::updateRecords(float *xArray, const std::unordered_map<int, float> *xFilter) const
{
    if (xFilter && xFilter->size() > 0)
    {
        std::unordered_map<int, float>::const_iterator filterIter;
        for (filterIter = xFilter->begin(); filterIter != xFilter->end(); ++filterIter)
        {
            int index = filterIter->first;
            float value = filterIter->second;
            xArray[index] = value * xArray[index];
        }
    }
}

/**
 * @brief Updates the records in the given array using the provided filter.
 *
 * This function updates the records in the given array using the provided filter.
 * The update is applied only to the portion of the array specified by the offset and width parameters.
 *
 * @param xArray The array to be updated.
 * @param xFilter The filter to be applied, represented as an std::unordered_map<int, float>.
 * @param offset The starting index of the portion to be updated.
 * @param width The width of the portion to be updated.
 */
void AbstractFilter::updateRecords(float *xArray, const std::unordered_map<int, float> *xFilter, int offset, int width) const
{
    if (xFilter && xFilter->size() > 0)
    {
        std::unordered_map<int, float>::const_iterator filterIter;
        for (filterIter = xFilter->begin(); filterIter != xFilter->end(); ++filterIter)
        {
            int index = filterIter->first;
            float value = filterIter->second;
            if (index >= offset && index < offset + width)
            { 
                xArray[index - offset] = value * xArray[index - offset];
            }
        }
    }
}

/**
 * @brief Loads a single filter from a file and populates the sampleFilters vector.
 *
 * This function reads a single filter from the specified file and populates the sampleFilters vector
 * with the loaded filter data. It utilizes the xMInput and xMSamples maps for key-value lookups.
 *
 * @param xMInput The input map containing string keys and unsigned integer values.
 * @param xMSamples The samples map containing string keys and unsigned integer values.
 * @param sampleFilters The vector of unique pointers to unordered_map<int, float> representing the sample filters.
 * @param filePath The path to the file containing the filter data.
 */
void SamplesFilter::loadSingleFilter(std::unordered_map<std::string, unsigned int> &xMInput,
                                     std::unordered_map<std::string, unsigned int> &xMSamples,
                                     std::vector<std::unique_ptr<std::unordered_map<int, float>>> &sampleFilters,
                                     const std::string &filePath)
{
    std::ifstream samplesFile(filePath);
    auto start = std::chrono::steady_clock::now();
    std::unordered_map<int, float> *sampleFilter = nullptr;
    int samplesFilterCount = 0;
    std::vector<std::string> filters;
    if (samplesFile.good())
    {
        std::string line;
        int sample = -1;
        while (std::getline(samplesFile, line))
        {
            filters = split(line, ':');
            if (filters.size() > 0)
            {
                std::vector<std::string> vals = split(filters[0], '\t');
                if (vals.size() > 0)
                {
                    try
                    {
                        sample = xMSamples.at(vals[0]);
                        if (vals.size() > 1)
                        {
                            filters[0] = vals[1];
                        }
                    }
                    catch (const std::out_of_range& oor)
                    {
                        continue;
                    }
                }
            }

            sampleFilter = new std::unordered_map<int, float>();
            for (int i = 0; i < filters.size(); ++i)
            {
                std::vector<std::string> vals = split(filters[i], ',');
                if (vals.size() > 0)
                {
                    try
                    {
                        int key = xMInput.at(vals[0]);
                        float value = 0.0f;
                        if (vals.size() > 1)
                        {
                            value = std::atof(vals[1].c_str());
                        }
                        (*sampleFilter)[key] = value;
                    }
                    catch (const std::out_of_range& oor)
                    {
                        continue;
                    }
                }
            }
            if (sample != -1)
            {
                sampleFilters[sample].reset(sampleFilter);
                ++samplesFilterCount;
                if (samplesFilterCount % gSamplesLoggingInterval == 0)
                {
                    auto const end = std::chrono::steady_clock::now();
                    std::cout << "Progress Parsing Filter " << samplesFilterCount;
                    std::cout << "Time " << elapsed_seconds(start, end) << std::endl;
                    start = std::chrono::steady_clock::now();
                }
            }
        }
    }
    else
    {
        std::cout << "Unable to read the file " << filePath << std::endl;
        throw std::invalid_argument("invalid sample filters " + filePath + ", exiting...");
    }
}

/**
 * @brief Loads filter data from multiple files and populates the sample filters.
 *
 * This function loads filter data from multiple files located at the specified filter file path.
 * It populates the sample filters with the loaded data.
 *
 * @param xMInput The input map containing string keys and unsigned integer values.
 * @param xMSamples The samples map containing string keys and unsigned integer values.
 * @param filterFilePath The path to the directory containing the filter files.
 */
void SamplesFilter::loadFilter(std::unordered_map<std::string, unsigned int>& xMInput,
                               std::unordered_map<std::string, unsigned int>& xMSamples,
                               const std::string& filterFilePath)
{
    samplefilters.reset(new std::vector<std::unique_ptr<std::unordered_map<int, float>>>(xMSamples.size()));

    std::vector<std::string> files;
    if (listFiles(filterFilePath, false, files) == 0)
    {
        std::cout << "Loading " << files.size() << " filter files" << std::endl;

        for (const auto& file : files)
        {
            std::cout << "\tLoading filter: " << file << std::endl;
            loadSingleFilter(xMInput, xMSamples, *samplefilters.get(), file);
        }
    }

    std::cout << "Info:SamplesFilter " << samplefilters->size() << std::endl;
}

/**
 * @brief Applies a filter to a portion of the given array.
 *
 * This function applies a filter to a portion of the given array, starting from the specified offset and spanning the specified width.
 *
 * @param xArray         The array to be filtered.
 * @param xSamplesIndex  The index of the filter to be used from the sample filters.
 * @param offset         The starting index of the portion to be filtered.
 * @param width          The width of the portion to be filtered.
 */
void SamplesFilter::applyFilter(float *xArray, int xSamplesIndex, int offset, int width) const
{
    std::unordered_map<int, float> *filter = (*samplefilters)[xSamplesIndex].get();
    updateRecords(xArray, filter, offset, width);
}

/**
 * @brief Applies a filter to the entire given array.
 *
 * This function applies a filter to the entire given array using the specified filter index.
 *
 * @param xArray         The array to be filtered.
 * @param xSamplesIndex  The index of the filter to be used from the sample filters.
 */
void SamplesFilter::applyFilter(float *xArray, int xSamplesIndex) const
{
    std::unordered_map<int, float> *filter = (*samplefilters)[xSamplesIndex].get();
    updateRecords(xArray, filter);
}

/**
 * @brief Loads filters based on the provided configuration.
 *
 * @param samplesFilterFileName The file name of the samples filter.
 * @param outputFileName The file name of the output.
 * @param xMInput The input map to populate with data.
 * @param xMSamples The samples map to populate with data.
 * @return A pointer to the loaded filter configuration.
 */
FilterConfig* loadFilters(const std::string &samplesFilterFileName,
                          const std::string &outputFileName,
                          std::unordered_map<std::string, unsigned int>& xMInput,
                          std::unordered_map<std::string, unsigned int>& xMSamples)
{
    Json::Value index;
    Json::Reader reader;
    FilterConfig *filterConfig = new FilterConfig();
    SamplesFilter *samplesFilter = new SamplesFilter();
    samplesFilter->loadFilter(xMInput, xMSamples, samplesFilterFileName);
    filterConfig->setSamplesFilter(samplesFilter);
    filterConfig->setOutputFileName(outputFileName);
    FILE *fp = std::fopen(outputFileName.c_str(), "w");
    std::fclose(fp);
    return filterConfig;
}
