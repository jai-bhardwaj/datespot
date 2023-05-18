#ifndef FILTERS_H
#define FILTERS_H

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

/**
 * @brief Abstract base class for filters.
 */
class AbstractFilter
{
public:
    virtual ~AbstractFilter() = default;

    /**
     * @brief Loads a filter from a file.
     *
     * @param xMInput The input map.
     * @param xMSamples The samples map.
     * @param filePath The path to the filter file.
     */
    virtual void loadFilter(std::unordered_map<std::string, unsigned int> &xMInput,
                            std::unordered_map<std::string, unsigned int> &xMSamples,
                            const std::string &filePath) = 0;

    /**
     * @brief Applies the filter to an array of samples at the given index.
     *
     * @param xArray The array of samples.
     * @param xSamplesIndex The index of the samples.
     */
    virtual void applyFilter(float *xArray, int xSamplesIndex) const = 0;

    /**
     * @brief Applies the filter to a subset of samples within an array.
     *
     * @param xArray The array of samples.
     * @param xSamplesIndex The index of the samples.
     * @param offset The offset within the array.
     * @param width The width of the subset to apply the filter to.
     */
    virtual void applyFilter(float *xArray, int xSamplesIndex, int offset, int width) const = 0;

    /**
     * @brief Gets the type of the filter.
     *
     * @return The filter type.
     */
    virtual std::string getFilterType() const = 0;

protected:
    /**
     * @brief Updates records using the provided filter.
     *
     * @param xArray The array of samples.
     * @param xFilter The filter to apply.
     */
    void updateRecords(float *xArray, const std::unordered_map<int, float> *xFilter) const;

    /**
     * @brief Updates a subset of records using the provided filter.
     *
     * @param xArray The array of samples.
     * @param xFilter The filter to apply.
     * @param offset The offset within the array.
     * @param width The width of the subset to update.
     */
    void updateRecords(float *xArray, const std::unordered_map<int, float> *xFilter, int offset, int width) const;
};

/**
 * @brief Filter class for processing samples.
 */
class SamplesFilter : public AbstractFilter
{
    std::vector<std::unique_ptr<std::unordered_map<int, float>>> samplefilters;

    /**
     * @brief Loads a single filter from a file.
     *
     * @param xMInput The input map.
     * @param xMSamples The samples map.
     * @param sampleFilters The vector to store the loaded filters.
     * @param filePath The path to the filter file.
     */
    void loadSingleFilter(std::unordered_map<std::string, unsigned int> &xMInput,
                          std::unordered_map<std::string, unsigned int> &xMSamples,
                          std::vector<std::unique_ptr<std::unordered_map<int, float>>> &sampleFilters,
                          const std::string &filePath);

public:
    void loadFilter(std::unordered_map<std::string, unsigned int> &xMInput,
                    std::unordered_map<std::string, unsigned int> &xMSamples,
                    const std::string &filePath) override;

    void applyFilter(float *xArray, int xSamplesIndex) const override;

    void applyFilter(float *xArray, int xSamplesIndex, int offset, int width) const override;

    std::string getFilterType() const override
    {
        return "samplesFilterType";
    }
};

/**
 * @brief Configuration class for filters.
 */
class FilterConfig
{
    std::unique_ptr<SamplesFilter> sampleFilter;
    std::string_view outputFileName;

public:
    /**
     * @brief Sets the output file name.
     *
     * @param xOutputFileName The output file name.
     */
    void setOutputFileName(std::string_view xOutputFileName)
    {
        outputFileName = xOutputFileName;
    }

    /**
     * @brief Gets the output file name.
     *
     * @return The output file name.
     */
    std::string_view getOutputFileName() const
    {
        return outputFileName;
    }

    /**
     * @brief Sets the samples filter.
     *
     * @param xSampleFilter The samples filter.
     */
    void setSamplesFilter(SamplesFilter *xSampleFilter)
    {
        sampleFilter.reset(xSampleFilter);
    }

    /**
     * @brief Applies the samples filter to an input array.
     *
     * @param xInput The input array.
     * @param xSampleIndex The index of the samples.
     * @param offset The offset within the array.
     * @param width The width of the subset to apply the filter to.
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
 * @brief Loads filters from file.
 *
 * @param samplesFilterFileName The file name for the samples filter.
 * @param outputFileName The output file name.
 * @param xMInput The input map.
 * @param xMSamples The samples map.
 *
 * @return The loaded filter configuration.
 */
FilterConfig* loadFilters(const std::string &samplesFilterFileName,
                          const std::string &outputFileName,
                          std::unordered_map<std::string, unsigned int> &xMInput,
                          std::unordered_map<std::string, unsigned int> &xMSamples);

#endif
