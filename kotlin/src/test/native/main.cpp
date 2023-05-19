#include <iostream>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <mpi.h>
#include <string_view>
#include "src/engine/Context.h"
#include "src/engine/GpuTypes.h"
#include "src/engine/Types.h"
#include "src/engine/Layer.h"
#include "src/engine/Network.h"

/**
 * Reads an index file and maps the keys to indices and vice versa.
 * 
 * @param indexFile The path to the index file.
 * @param indexMap A map from keys to indices.
 * @param rIndexMap A map from indices to keys.
 * 
 * @throws std::runtime_error if the index file does not exist.
 */
void readIndexFile(std::string_view indexFile, std::map<std::string, uint32_t> &indexMap, std::map<uint32_t, std::string> &rIndexMap)
{
    std::ifstream in(indexFile.data());
    if (!in) {
        throw std::runtime_error(std::string(indexFile) + " does not exist");
    }

    std::string line;
    while (std::getline(in, line))
    {
        auto idx = line.find_first_of('\t');
        auto key = line.substr(0, idx);
        uint32_t index = std::stoul(line.substr(idx + 1, line.size()));
        indexMap[key] = index;
        rIndexMap[index] = key;
    }

    in.close();
}

/**
 * Reads an input file and maps the keys to indices and stores the data in sparse format.
 * 
 * @param inputFile The path to the input file.
 * @param indexMap A map from keys to indices.
 * @param sparseStart An array that stores the start indices for each input in the sparse data.
 * @param sparseEnd An array that stores the end indices for each input in the sparse data.
 * @param sparseIndex An array that stores the indices of the non-zero elements in the sparse data.
 * @param sparseData An array that stores the values of the non-zero elements in the sparse data.
 * 
 * @throws std::runtime_error if the input file does not exist.
 */
void readInputFile(std::string_view inputFile, const std::map<std::string, uint32_t> &indexMap, uint64_t *sparseStart,
                   uint64_t *sparseEnd, uint32_t *sparseIndex, int *sparseData)
{
    std::ifstream in(inputFile.data());
    if (!in) {
        throw std::runtime_error(std::string(inputFile) + " does not exist");
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

        auto idx = line.find_first_of('\t');
        auto v = line.substr(idx + 1, line.size());

        int j = 0;
        std::stringstream vs(v);
        std::string elem;

        while (std::getline(vs, elem, ':'))
        {
            auto vidx = elem.find_first_of(',');
            auto key = elem.substr(0, vidx);

            sparseIndex[sparseStart[i] + j] = indexMap.at(key);
            sparseData[sparseStart[i] + j] = 1;
            ++j;
        }

        sparseEnd[i] = sparseStart[i] + j;
        ++i;
    }

    in.close();
}

/**
 * The main function for a program that predicts the top-k elements of a sparse input data.
 * 
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line arguments.
 * 
 * @return 0 if the program executes successfully, non-zero otherwise.
 */
int main(int argc, char** argv)
{
    uint32_t k = 10;
    uint32_t batchSize = 32;
    float sparseDensity = 0.09;

    const std::string networkFile = argv[1];
    const std::string indexFile = argv[2];
    const std::string inputFile = argv[3];

    Context dc(networkFile, batchSize);

    auto network = dc.getNetwork();
    std::cout << "main.o: loaded network " << network->GetName() << std::endl;

    std::map<std::string, uint32_t> indexes;
    std::map<uint32_t, std::string> rIndexes;
    readIndexFile(indexFile, indexes, rIndexes);
    std::cout << "Read " << indexes.size() << " indexes" << std::endl;

    const auto& inputLayer = network->GetLayer("Input");
    auto [x, y, z, w] = inputLayer.GetDimensions();
    size_t sparseDataLength = ((float) (x * y * z * batchSize)) * sparseDensity;
    auto inputDataset = inputLayer.GetDataSet();

    Float *outputScores;
    uint32_t *outputIndexes;
    cudaMallocManaged(&outputScores, k * batchSize * sizeof(Float));
    cudaMallocManaged(&outputIndexes, k * batchSize * sizeof(uint32_t));

    for (int i = 0; i < 1; ++i)
    {
        uint64_t *sparseStart = (uint64_t*) std::calloc(batchSize, sizeof(uint64_t));
        uint64_t *sparseEnd = (uint64_t*) std::calloc(batchSize, sizeof(uint64_t));
        uint32_t *sparseIndex = (uint32_t*) std::calloc(sparseDataLength, sizeof(uint32_t));
        int *sparseData = (int*) std::calloc(sparseDataLength, sizeof(int));

        readInputFile(inputFile, indexes, sparseStart, sparseEnd, sparseIndex, sparseData);

        inputDataset->LoadSparseData(sparseStart, sparseEnd, sparseData, sparseIndex);

        std::free(sparseStart);
        std::free(sparseEnd);
        std::free(sparseIndex);
        std::free(sparseData);

        network->SetPosition(0);
        network->PredictBatch();

        const auto& outputLayer = network->GetLayer("Output");
        Float *dUnitBuffer = network->GetUnitBuffer("Output");

        auto [x, y, z, w] = outputLayer.GetDimensions();

        size_t width = x * y * z;

        kCalculateOutput(dUnitBuffer, outputScores, outputIndexes, batchSize, width, k);
        cudaDeviceSynchronize();

        for (size_t i = 0; i < batchSize; ++i)
        {
            std::("%d\t", i+1);
            for (size_t j = 0; j < k; ++j)
            {

                const string &idx = rIndexes.at(outputIndexes[i * k + j]);
                std::("%s:%5.3f,", idx.c_str(), outputScores[i * k + j]);
            }
            std::("\n");
        }

    }

    cudaFree(outputScores);
    cudaFree(outputIndexes);
}
