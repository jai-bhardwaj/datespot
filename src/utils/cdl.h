#ifndef __CDL_H__
#define __CDL_H__

#include "GpuTypes.h"
#include "Types.h"

/**
 * @brief The CDL struct represents a configuration for a CDL (Convolutional Deep Learning) model.
 */
struct CDL
{
    /**
     * @brief Default constructor for CDL.
     */
    CDL();

    /**
     * @brief Load the CDL configuration from a JSON file.
     * @param fname The filename of the JSON file.
     * @return The status of the loading operation.
     */
    int Load_JSON(const string& fname);

    std::string     _networkFileName;       // The filename of the network file.
    int             _randomSeed;            // The random seed value.
    Mode            _mode;                  // The mode of operation.
    std::string     _dataFileName;          // The filename of the data file.

    int             _epochs;                // The number of training epochs.
    int             _batch;                 // The batch size.
    float           _alpha;                 // The learning rate.
    float           _lambda;                // The regularization parameter.
    float           _mu;                    // The momentum parameter.
    int             _alphaInterval;         // The interval at which to update the learning rate.
    float           _alphaMultiplier;       // The multiplier to adjust the learning rate.
    TrainingMode    _optimizer;             // The optimizer to use for training.
    std::string     _checkpointFileName;    // The filename to save checkpoints.
    int             _checkpointInterval;    // The interval at which to save checkpoints.
    bool            _shuffleIndexes;        // Whether to shuffle the indexes during training.
    std::string     _resultsFileName;       // The filename to save training results.
};

#endif
