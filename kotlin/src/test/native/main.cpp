#include <iostream>
#include <utility>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <memory>
#include <charconv>
#include <format>

#include <mpi.h>

#include "engine/Context.h"
#include "engine/GpuTypes.h"
#include "engine/Types.h"
#include "engine/Layer.h"
#include "engine/Network.h"

namespace fs = std::filesystem;

void readIndexFile(const std::string& indexFile, std::unordered_map<std::string, uint32_t>& indexMap,
                   std::unordered_map<uint32_t, std::string>& rIndexMap)
{
    std::ifstream in(indexFile);
    if (!in)
    {
        throw std::runtime_error(indexFile + " does not exist");
    }
    std::string line;
    while (std::getline(in, line))
    {
        size_t idx = line.find_first_of('\t');
        std::string key = line.substr(0, idx);
        uint32_t index;
        std::from_chars(line.data() + idx + 1, line.data() + line.size(), index);
        indexMap[key] = index;
        rIndexMap[index] = key;
    }
    in.close();
}

void readInputFile(const std::string& inputFile, const std::unordered_map<std::string, uint32_t>& indexMap,
                   std::vector<uint64_t>& sparseStart, std::vector<uint64_t>& sparseEnd,
                   std::vector<uint32_t>& sparseIndex, std::vector<int>& sparseData)
{
    std::ifstream in(inputFile);
    if (!in)
    {
        throw std::runtime_error(inputFile + " does not exist");
    }

    std::string line;
    int i = 0;

    sparseStart[0] = 0;
    while (std::getline(in, line))
    {
        if (i != 0)
        {
            sparseStart[i] = sparseEnd[i - 1];
        }

        size_t idx = line.find_first_of('\t');
        std::string v = line.substr(idx + 1, line.size());

        int j = 0;
        std::stringstream vs(v);
        std::string elem;
        while (std::getline(vs, elem, ':'))
        {
            size_t vidx = elem.find_first_of(',');
            std::string key = elem.substr(0, vidx);

            sparseIndex[sparseStart[i] + j] = indexMap.at(key);
            sparseData[sparseStart[i] + j] = 1;
            ++j;
        }
        sparseEnd[i] = sparseStart[i] + j;
        ++i;
    }
    in.close();
}

int main(int argc, char** argv)
{
    uint32_t k = 10;
    uint32_t batchSize = 32;
    float sparseDensity = 0.09;

    if (argc < 4)
    {
        std::cout << "Usage: ./program networkFile indexFile inputFile" << std::endl;
        return 1;
    }

    const std::string networkFile = argv[1];
    const std::string indexFile = argv[2];
    const std::string inputFile = argv[3];

    if (!fs::exists(networkFile))
    {
        std::cout << networkFile << " does not exist" << std::endl;
        return 1;
    }

    Context dc(networkFile, batchSize);

    Network* network = dc.getNetwork();
    std::cout << "main.o: loaded network " << network->GetName() << std::endl;

    std::vector<const Layer*> inputLayers;
    std::vector<DataSetDescriptor> datasets;

    auto it = network->GetLayers(Layer::Kind::Input, inputLayers);
    for (const Layer* layer : inputLayers)
    {
        DataSetDescriptor desc;

        uint32_t x, y, z, w;
        std::tie(x, y, z, w) = layer->GetDimensions();
        desc._dim = DataSetDimensions(x, y, z);
        desc._name = layer->GetDataSetName();
        desc._dataType = DataSetEnums::DataType::Int;
        desc._attributes = DataSetEnums::Attributes::Sparse;
        desc._examples = network->GetBatch();
        desc._sparseDensity = sparseDensity;

        datasets.push_back(desc);
    }

    dc.initInputLayerDataSets(datasets);

    std::unordered_map<std::string, uint32_t> indexes;
    std::unordered_map<uint32_t, std::string> rIndexes;
    readIndexFile(indexFile, indexes, rIndexes);
    std::cout << "Read " << indexes.size() << " indexes" << std::endl;

    const Layer* inputLayer = network->GetLayer("Input");
    uint32_t x, y, z, w;
    std::tie(x, y, z, w) = inputLayer->GetDimensions();
    size_t sparseDataLength = static_cast<size_t>((x * y * z * batchSize) * sparseDensity);
    DataSetBase* inputDataset = inputLayer->GetDataSet();

    auto outputScores = std::make_unique<float[]>(k * batchSize);
    auto outputIndexes = std::make_unique<uint32_t[]>(k * batchSize);

    for (int i = 0; i < 1; ++i)
    {
        std::vector<uint64_t> sparseStart(batchSize);
        std::vector<uint64_t> sparseEnd(batchSize);
        std::vector<uint32_t> sparseIndex(sparseDataLength);
        std::vector<int> sparseData(sparseDataLength);

        readInputFile(inputFile, indexes, sparseStart, sparseEnd, sparseIndex, sparseData);

        inputDataset->LoadSparseData(sparseStart.data(), sparseEnd.data(), sparseData.data(), sparseIndex.data());

        network->SetPosition(0);
        network->PredictBatch();

        const Layer* outputLayer = network->GetLayer("Output");
        float* dUnitBuffer = network->GetUnitBuffer("Output");

        std::tie(x, y, z, w) = outputLayer->GetDimensions();

        size_t width = x * y * z;

        kCalculateOutput(dUnitBuffer, outputScores.get(), outputIndexes.get(), batchSize, width, k);
        cudaDeviceSynchronize();

        for (size_t i = 0; i < batchSize; ++i)
        {
            std::cout << std::format("{}\t", i + 1);
            for (size_t j = 0; j < k; ++j)
            {
                const std::string& idx = rIndexes.at(outputIndexes[i * k + j]);
                std::cout << std::format("{}:{:5.3f},", idx, outputScores[i * k + j]);
            }
            std::cout << std::endl;
        }
    }

    return 0;
}
