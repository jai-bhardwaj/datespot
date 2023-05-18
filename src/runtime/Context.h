#ifndef CONTEXT_H_
#define CONTEXT_H_

#include <map>
#include <string>
#include <stdexcept>
#include <vector>

#include "engine/GpuTypes.h"
#include "engine/Types.h"
#include "engine/Layer.h"

/**
 * @brief The Context class represents the execution context for a neural network.
 */
class Context
{
private:
    //Constant representing all elements.
    inline static constexpr int ALL = -1;
    // Filename of the neural network.
    onst std::string networkFilename;
    // Batch size for inference.
    const uint32_t batchSize;
    // Maximum K value.
    const uint32_t maxK;

    // Map of output scores buffers for each layer.
    std::map<std::string, GpuBuffer<Float>*> dOutputScores;
    // Map of output indexes buffers for each layer.
    std::map<std::string, GpuBuffer<uint32_t>*> dOutputIndexes;

public:
    /**
     * @brief Constructs a new Context object.
     * @param networkFilename Filename of the neural network.
     * @param batchSize Batch size for inference.
     * @param maxK Maximum K value.
     */
    Context(const std::string& networkFilename, uint32_t batchSize, int maxK = ALL)
        : networkFilename(networkFilename), batchSize(batchSize), maxK(maxK)
    {
    }

    /**
     * @brief Destroys the Context object.
     */
    ~Context() = default;

    /**
     * @brief Retrieves the neural network associated with this context.
     * @return Pointer to the neural network.
     */
    Network* getNetwork() const;

    /**
     * @brief Initializes input layer data sets.
     * @param datasetDescriptors Vector of dataset descriptors.
     */
    void initInputLayerDataSets(const std::vector<DataSetDescriptor>& datasetDescriptors);

    /**
     * @brief Retrieves the output scores buffer for a specific layer.
     * @param layerName Name of the layer.
     * @return Pointer to the output scores buffer.
     */
    GpuBuffer<Float>* getOutputScoresBuffer(const std::string& layerName);

    /**
     * @brief Retrieves the output indexes buffer for a specific layer.
     * @param layerName Name of the layer.
     * @return Pointer to the output indexes buffer.
     */
    GpuBuffer<uint32_t>* getOutputIndexesBuffer(const std::string& layerName);

    /**
     * @brief Converts a pointer to Context.
     * @param ptr Pointer to convert.
     * @return Pointer to the Context object.
     * @throws std::runtime_error if the input pointer is nullptr.
     */
    static Context* fromPtr(long ptr)
    {
        Context* dc = reinterpret_cast<Context*>(ptr);
        if (dc == nullptr)
        {
            throw std::runtime_error("Cannot convert nullptr to Context");
        }
        return dc;
    }
};

#endif
