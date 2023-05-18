#ifndef GENERATOR_H
#define GENERATOR_H

#include <vector>
#include <string>
#include <memory>
#include "GpuTypes.h"
#include "Types.h"

class FilterConfig;
class Network;

/**
 * @brief Class for generating recommendations.
 */
class RecsGenerator
{
    std::unique_ptr<GpuBuffer<NNFloat>> pbKey;
    std::unique_ptr<GpuBuffer<unsigned int>> pbUIValue;
    std::unique_ptr<GpuBuffer<NNFloat>> pFilteredOutput;
    std::vector<GpuBuffer<NNFloat>*> vNodeFilters;
    std::string recsGenLayerLabel;
    std::string scorePrecision;

public:
    static inline const std::string DEFAULT_LAYER_RECS_GEN_LABEL = "DefaultLayer"; /**< Default label for the recommendation generation layer. */
    static inline const unsigned int Output_SCALAR = 1; /**< Scalar value for output. */
    static inline const std::string DEFAULT_SCORE_PRECISION = "DefaultPrecision"; /**< Default score precision. */

    /**
     * @brief Constructs a RecsGenerator object.
     * @param xBatchSize The batch size.
     * @param xK The number of recommendations to generate.
     * @param xOutputBufferSize The size of the output buffer.
     * @param layer The label for the recommendation generation layer.
     * @param precision The score precision.
     */
    RecsGenerator(unsigned int xBatchSize,
                  unsigned int xK,
                  unsigned int xOutputBufferSize,
                  const std::string& layer = DEFAULT_LAYER_RECS_GEN_LABEL,
                  const std::string& precision = DEFAULT_SCORE_PRECISION);

    /**
     * @brief Generates recommendations.
     * @param network The network used for recommendation generation.
     * @param Output The output value.
     * @param filters The filter configuration.
     * @param customerIndex The customer index.
     * @param featureIndex The feature index.
     */
    void generateRecs(Network* network,
                      unsigned int Output,
                      const FilterConfig* filters,
                      const std::vector<std::string>& customerIndex,
                      const std::vector<std::string>& featureIndex);
};

#endif
