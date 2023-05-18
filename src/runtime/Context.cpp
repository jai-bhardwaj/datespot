#include <Context.h>
#include <vector>
#include <stdexcept>

namespace {
    const int ARGC = 1;
    char *ARGV = "tensorhub-faux-process";
    const unsigned long SEED = 12134ULL;
}

/**
 * @brief Constructs a Context object.
 *
 * @param networkFilename The filename of the neural network.
 * @param batchSize The batch size.
 * @param maxK The maximum K value.
 */
Context::Context(const std::string &networkFilename, uint32_t batchSize, int maxK)
    : networkFilename(networkFilename), batchSize(batchSize), maxK(maxK)
{
    getGpu().Startup(ARGC, &ARGV);
    getGpu().SetRandomSeed(SEED);
    Network *network = LoadNeuralNetworkNetCDF(networkFilename, batchSize);
    getGpu().SetNeuralNetwork(network);

    auto outputLayers = network->GetLayers(Layer::Kind::Output);

    for (const auto *layer : outputLayers)
    {
        const std::string &layerName = layer->GetName();

        if (maxK != ALL)
        {
            if (layer->GetNumDimensions() > 1)
            {
                throw std::runtime_error("topK only supported on 1-D output layers");
            }
            size_t outputBufferLength = maxK * batchSize;
            printf("Context::Context: Allocating output score and index buffers, each of size %zu for output layer %s\n",
                   outputBufferLength, layerName.c_str());
            auto outputScores = std::make_unique<GpuBuffer<Float>>(outputBufferLength, false, false);
            auto outputIndexes = std::make_unique<GpuBuffer<uint32_t>>(outputBufferLength, false, false);

            dOutputScores[layerName] = std::move(outputScores);
            dOutputIndexes[layerName] = std::move(outputIndexes);
        }
    }
}

/**
 * @brief Returns the output scores buffer for the specified layer.
 *
 * @param layerName The name of the layer.
 * @return GpuBuffer<Float>* The output scores buffer.
 */
GpuBuffer<Float>* Context::getOutputScoresBuffer(const std::string &layerName)
{
    return dOutputScores.at(layerName).get();
}

/**
 * @brief Returns the output indexes buffer for the specified layer.
 *
 * @param layerName The name of the layer.
 * @return GpuBuffer<uint32_t>* The output indexes buffer.
 */
GpuBuffer<uint32_t>* Context::getOutputIndexesBuffer(const std::string &layerName)
{
    return dOutputIndexes.at(layerName).get();
}

/**
 * @brief Destroys the Context object.
 */
Context::~Context()
{
    const std::string networkName = getNetwork()->GetName();
    dOutputScores.clear();
    dOutputIndexes.clear();

    delete getNetwork();
    getGpu().Shutdown();
    printf("Context::~Context: Destroyed context for network %s\n", networkName.c_str());
}

/**
 * @brief Returns the Network associated with the Context.
 *
 * @return Network* The Network object.
 */
Network* Context::getNetwork() const
{
    return getGpu()._pNetwork;
}

/**
 * @brief Initializes the input layer data sets.
 *
 * @param datasetDescriptors The descriptors of the data sets.
 */
void Context::initInputLayerDataSets(const std::vector<DataSetDescriptor> datasetDescriptors)
{
    std::vector<std::unique_ptr<DataSetBase>> datasets;
    for (const auto &descriptor : datasetDescriptors)
    {
        std::unique_ptr<DataSetBase> dataset(createDataSet(descriptor));
        datasets.push_back(std::move(dataset));
    }

    Network *network = getNetwork();
    network->LoadDataSets(datasets);
    network->PredictBatch();
    network->SetPosition(0);
}
