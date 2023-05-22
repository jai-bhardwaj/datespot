#include <cstdio>
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <format>
#include <array>
#include <memory>
#include <span>

#include <json/json.h>

#include "Filters.h"
#include "Utils.h"

using namespace Json;
using namespace std::chrono;

constexpr int gSamplesLoggingInterval = 10000;

/**
 * @brief Updates the records in an array using a filter.
 *
 * @param xArray Pointer to the array of records.
 * @param xFilter Pointer to the filter map.
 */
void AbstractFilter::updateRecords(std::span<float> xArray, const std::unordered_map<int, float>& xFilter) const
{
    for (const auto& [index, value] : xFilter)
    {
        xArray[index] = value * xArray[index];
    }
}

/**
 * @brief Updates a portion of the records in an array using a filter.
 *
 * @param xArray Pointer to the array of records.
 * @param xFilter Pointer to the filter map.
 * @param offset Starting index of the portion to update.
 * @param width Width of the portion to update.
 */
void AbstractFilter::updateRecords(std::span<float> xArray, const std::unordered_map<int, float>& xFilter,
                                   int offset, int width) const
{
    for (const auto& [index, value] : xFilter)
    {
        if (index >= offset && index < offset + width)
        {
            xArray[index - offset] = value * xArray[index - offset];
        }
    }
}

/**
 * @brief Loads a single filter from a file.
 *
 * @param xMInput Reference to the input map.
 * @param xMSamples Reference to the samples map.
 * @param sampleFilters Reference to the vector of sample filters.
 * @param filePath Path to the filter file.
 */
void SamplesFilter::loadSingleFilter(std::unordered_map<std::string, unsigned int>& xMInput,
                                     std::unordered_map<std::string, unsigned int>& xMSamples,
                                     std::vector<std::unique_ptr<std::unordered_map<int, float>>>& sampleFilters,
                                     const std::string& filePath)
{
    std::ifstream samplesFile(filePath);
    auto start = steady_clock::now();
    std::unordered_map<int, float> sampleFilter;
    int samplesFilterCount = 0;
    std::vector<std::string> filters;
    if (samplesFile.good())
    {
        std::string line;
        int sample = -1;
        while (std::getline(samplesFile, line))
        {
            filters = split(line, ':');
            if (filters.empty())
            {
                continue;
            }
            std::vector<std::string> vals = split(filters[0], '\t');
            if (!vals.empty())
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

            sampleFilter.clear();
            for (const std::string& filter : filters)
            {
                std::vector<std::string> vals = split(filter, ',');
                if (!vals.empty())
                {
                    try
                    {
                        int key = xMInput.at(vals[0]);
                        float value = 0.0f;
                        if (vals.size() > 1)
                        {
                            value = std::stof(vals[1]);
                        }
                        sampleFilter[key] = value;
                    }
                    catch (const std::out_of_range& oor)
                    {
                        continue;
                    }
                }
            }
            if (sample != -1)
            {
                sampleFilters[sample] = std::make_unique<std::unordered_map<int, float>>(sampleFilter);
                ++samplesFilterCount;
                if (samplesFilterCount % gSamplesLoggingInterval == 0)
                {
                    const auto end = steady_clock::now();
                    std::cout << std::format("Progress Parsing Filter {}: Time {}\n",
                                             samplesFilterCount, elapsed_seconds(start, end));
                    start = steady_clock::now();
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
 * @brief Loads filters from a directory.
 *
 * @param xMInput Reference to the input map.
 * @param xMSamples Reference to the samples map.
 * @param filterFilePath Path to the directory containing filter files.
 */
void SamplesFilter::loadFilter(std::unordered_map<std::string, unsigned int>& xMInput,
                               std::unordered_map<std::string, unsigned int>& xMSamples,
                               const std::string& filterFilePath)
{
    samplefilters = std::make_unique<std::vector<std::unique_ptr<std::unordered_map<int, float>>>>(xMSamples.size());

    std::vector<std::string> files;
    for (const auto& entry : std::filesystem::directory_iterator(filterFilePath))
    {
        if (entry.is_regular_file())
        {
            files.push_back(entry.path().string());
        }
    }

    std::cout << "Loading " << files.size() << " filter files" << std::endl;

    for (const auto& file : files)
    {
        std::cout << "\tLoading filter: " << file << std::endl;
        loadSingleFilter(xMInput, xMSamples, *samplefilters, file);
    }

    std::cout << "Info:SamplesFilter " << samplefilters->size() << std::endl;
}

/**
 * @brief Applies a filter to a portion of the records in an array.
 *
 * @param xArray Pointer to the array of records.
 * @param xSamplesIndex Index of the sample filter to apply.
 * @param offset Starting index of the portion to update.
 * @param width Width of the portion to update.
 */
void SamplesFilter::applyFilter(std::span<float> xArray, int xSamplesIndex, int offset, int width) const
{
    const auto& filter = *(*samplefilters)[xSamplesIndex];
    updateRecords(xArray, filter, offset, width);
}

/**
 * @brief Applies a filter to all records in an array.
 *
 * @param xArray Pointer to the array of records.
 * @param xSamplesIndex Index of the sample filter to apply.
 */
void SamplesFilter::applyFilter(std::span<float> xArray, int xSamplesIndex) const
{
    const auto& filter = *(*samplefilters)[xSamplesIndex];
    updateRecords(xArray, filter);
}

/**
 * @brief Loads filters and creates a FilterConfig object.
 *
 * @param samplesFilterFileName Path to the samples filter file.
 * @param outputFileName Output file name.
 * @param xMInput Reference to the input map.
 * @param xMSamples Reference to the samples map.
 * @return Pointer to the created FilterConfig object.
 */
FilterConfig* loadFilters(const std::string& samplesFilterFileName,
                          const std::string& outputFileName,
                          std::unordered_map<std::string, unsigned int>& xMInput,
                          std::unordered_map<std::string, unsigned int>& xMSamples)
{
    Json::Value index;
    Json::Reader reader;
    FilterConfig* filterConfig = new FilterConfig();
    std::unique_ptr<SamplesFilter> samplesFilter = std::make_unique<SamplesFilter>();
    samplesFilter->loadFilter(xMInput, xMSamples, samplesFilterFileName);
    filterConfig->setSamplesFilter(std::move(samplesFilter));
    filterConfig->setOutputFileName(outputFileName);
    std::ofstream ofs(outputFileName);
    ofs.close();
    return filterConfig;
}
