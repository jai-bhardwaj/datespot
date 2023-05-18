#pragma once

#include <string>
#include "GpuTypes.h"
#include "Types.h"

/**
 * @brief Struct representing CDL (Compressed Deep Learning) configuration.
 */
struct CDL
{
    std::string     _networkFileName;       /**< The file name of the network. */
    int             _randomSeed;            /**< The random seed value. */
    Mode            _mode;                  /**< The mode of operation. */
    std::string     _dataFileName;          /**< The file name of the data. */

    int             _epochs;                /**< The number of epochs. */
    int             _batch;                 /**< The batch size. */
    float           _alpha;                 /**< The learning rate. */
    float           _lambda;                /**< The regularization parameter. */
    float           _mu;                    /**< The momentum parameter. */
    int             _alphaInterval;         /**< The interval for changing the learning rate. */
    float           _alphaMultiplier;       /**< The multiplier for changing the learning rate. */
    TrainingMode    _optimizer;             /**< The optimization algorithm. */
    std::string     _checkpointFileName;    /**< The file name for checkpointing. */
    int             _checkpointInterval;    /**< The interval for checkpointing. */
    bool            _shuffleIndexes;        /**< Flag indicating whether to shuffle indexes. */
    std::string     _resultsFileName;       /**< The file name for storing results. */

    /**
     * @brief Default constructor for CDL.
     */
    CDL() = default;

    /**
     * @brief Load JSON configuration from file.
     * @param fname The name of the JSON file to load.
     * @return The status of the load operation.
     */
    int Load_JSON(const std::string& fname);
};
