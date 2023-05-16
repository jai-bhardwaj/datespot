#ifndef FILTERS_H
#define FILTERS_H

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

/**
 * @class AbstractFilter
 * @brief The base abstract class for filters.
 */
class AbstractFilter
{
public:
    /**
     * @brief Virtual destructor for AbstractFilter.
     */
    virtual ~AbstractFilter() = default;

    /**
     * @brief Loads the filter data from a file.
     *
     * @param xMInput The input map to populate with data.
     * @param xMSamples The samples map to populate with data.
     * @param filePath The path of the file to load.
     */
    virtual void loadFilter(std::unordered_map<std::string, unsigned int> &xMInput,
                            std::unordered_map<std::string, unsigned int> &xMSamples,
                            const std::string &filePath) = 0;

    /**
     * @brief Applies the filter to an array of floats.
     *
     * @param xArray The array to apply the filter to.
     * @param xSamplesIndex The index of the samples.
     */
    virtual void applyFilter(float *xArray, int xSamplesIndex) const = 0;

    /**
     * @brief Applies the filter to a subset of an array of floats.
     *
     * @param xArray The array to apply the filter to.
     * @param xSamplesIndex The index of the samples.
     * @param offset The offset within the array.
     * @param width The width of the subset to apply the filter to.
     */
    virtual void applyFilter(float *xArray, int xSamplesIndex, int offset, int width) const = 0;

    /**
     * @brief Gets the type of the filter.
     *
     * @return The filter type as a string.
     */
    virtual std::string getFilterType() const = 0;

protected:
    /**
     * @brief Updates the records using the given array and filter.
     *
     * @param xArray The array to update the records from.
     * @param xFilter The filter to apply to the records.
     */
    void updateRecords(float *xArray, const std::unordered_map<int, float> *xFilter) const;

    /**
     * @brief Updates a subset of the records using the given array, filter, offset, and width.
     *
     * @param xArray The array to update the records from.
     * @param xFilter The filter to apply to the records.
     * @param offset The offset within the array.
     * @param width The width of the subset to update the records.
     */
    void updateRecords(float *xArray, const std::unordered_map<int, float> *xFilter, int offset, int width) const;
};

class SamplesFilter : public AbstractFilter
{
    std::unique_ptr<std::vector<std::unique_ptr<std::unordered_map<int, float>>>> samplefilters;

    /**
     * @brief Loads a single filter from a file.
     *
     * @param xMInput The input map to populate with data.
     * @param xMSamples The samples map to populate with data.
     * @param sampleFilters The vector of sample filters to populate.
     * @param filePath The path of the file to load.
     */
    void loadSingleFilter(std::unordered_map<std::string, unsigned int> &xMInput,
                          std::unordered_map<std::string, unsigned int> &xMSamples,
                          std::vector<std::unique_ptr<std::unordered_map<int, float>>> &sampleFilters,
                          const std::string &filePath);

public:
    /**
     * @brief Loads the filter data from a file.
     *
     * @param xMInput The input map to populate with data.
     * @param xMSamples The samples map to populate with data.
     * @param filePath The path of the file to load.
     */
    void loadFilter(std::unordered_map<std::string, unsigned int> &xMInput,
                    std::unordered_map<std::string, unsigned int> &xMSamples,
                    const std::string &filePath);

    /**
     * @brief Applies the filter to an array of floats.
     *
     * @param xArray The array to apply the filter to.
     * @param xSamplesIndex The index of the samples.
     */
    void applyFilter(float *xArray, int xSamplesIndex) const;

    /**
     * @brief Applies the filter to a subset of an array of floats.
     *
     * @param xArray The array to apply the filter to.
     * @param xSamplesIndex The index of the samples.
     * @param offset The offset within the array.
     * @param width The width of the subset to apply the filter to.
     */
    void applyFilter(float *xArray, int xSamplesIndex, int offset, int width) const;

    /**
     * @brief Gets the type of the filter.
     *
     * @return The filter type as a string.
     */
    std::string getFilterType() const
    {
        return "samplesFilterType";
    }
};

/**
 * @class FilterConfig
 * @brief Configuration class for filters.
 */
class FilterConfig
{
    std::unique_ptr<SamplesFilter> sampleFilter; /**< Unique pointer to the SamplesFilter object. */
    std::string outputFileName; /**< Output file name. */

public:
    /**
     * @brief Sets the output file name.
     * @param xOutputFileName The output file name to set.
     */
    void setOutputFileName(const std::string &xOutputFileName)
    {
        outputFileName = xOutputFileName;
    }

    /**
     * @brief Gets the output file name.
     * @return The output file name.
     */
    std::string getOutputFileName() const
    {
        return outputFileName;
    }

    /**
     * @brief Sets the SamplesFilter object.
     * @param xSampleFilter A pointer to the SamplesFilter object to set.
     */
    void setSamplesFilter(SamplesFilter *xSampleFilter)
    {
        sampleFilter.reset(xSampleFilter);
    }

    /**
     * @brief Applies the samples filter.
     * @param xInput The input array of samples.
     * @param xSampleIndex The index of the sample.
     * @param offset The offset value.
     * @param width The width value.
     */
    void applySamplesFilter(float *xInput, int xSampleIndex, int offset, int width) const
    {
        if (sampleFilter)
        {
            sampleFilter->applyFilter(xInput, xSampleIndex, offset, width);
        }
    }
};

/**
 * @brief Loads filters from a file.
 * @param samplesFilterFileName The file name containing the samples filter.
 * @param outputFileName The output file name.
 * @param xMInput Reference to the unordered map for input.
 * @param xMSamples Reference to the unordered map for samples.
 * @return A pointer to the loaded FilterConfig object.
 */
FilterConfig* loadFilters(const std::string &samplesFilterFileName,
                          const std::string &outputFileName,
                          std::unordered_map<std::string, unsigned int> &xMInput,
                          std::unordered_map<std::string, unsigned int> &xMSamples);

#endif
