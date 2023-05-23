#include "MultiHeadAttentionLayer.h"
#include <cmath>
#include <ranges>
#include <numeric>

MultiHeadAttentionLayer::MultiHeadAttentionLayer(const std::string& name, const std::vector<std::vector<float>>& queryWeights, const std::vector<float>& queryBiases, const std::vector<std::vector<float>>& keyWeights, const std::vector<float>& keyBiases, const std::vector<std::vector<float>>& valueWeights, const std::vector<float>& valueBiases, int headSize, int numHeads)
    : queryWeights(queryWeights), queryBiases(queryBiases), keyWeights(keyWeights), keyBiases(keyBiases), valueWeights(valueWeights), valueBiases(valueBiases), headSize(headSize), numHeads(numHeads) {
}

void MultiHeadAttentionLayer::initialize() {
}

void MultiHeadAttentionLayer::forward() {
    const std::vector<float>& queries = getInputQueries();
    const std::vector<float>& keys = getInputKeys();
    const std::vector<float>& values = getInputValues();

    std::vector<float> transformedQueries = linearTransform(queries, queryWeights, queryBiases);
    std::vector<float> transformedKeys = linearTransform(keys, keyWeights, keyBiases);
    std::vector<float> transformedValues = linearTransform(values, valueWeights, valueBiases);

    std::vector<float> transposedKeys = transpose(transformedKeys);

    std::vector<float> attentionScores = computeAttentionScores(transformedQueries, transposedKeys, headSize);

    std::vector<std::vector<float>> attentionMask = getAttentionMask(queries.size());
    std::vector<float> maskedAttentionScores = applyAttentionMask(attentionScores, attentionMask);

    std::vector<float> attentionWeights = maskedSoftmax(maskedAttentionScores);

    std::vector<float> weightedSum = weightedSumValues(attentionWeights, transformedValues, headSize);

    std::vector<float> combinedOutput = combineAttentionOutput(transformedValues, weightedSum, numHeads);

    setOutput(combinedOutput);
}

std::vector<float> MultiHeadAttentionLayer::linearTransform(const std::vector<float>& input, const std::vector<std::vector<float>>& weights, const std::vector<float>& biases) {
    std::vector<float> output(weights.size(), 0.0f);
    for (int i = 0; i < weights.size(); ++i) {
        output[i] = std::inner_product(input.begin(), input.end(), weights[i].begin(), biases[i]);
    }
    return output;
}

std::vector<float> MultiHeadAttentionLayer::transpose(const std::vector<std::vector<float>>& matrix) {
    std::vector<float> transposedMatrix;
    if (!matrix.empty()) {
        int numRows = matrix.size();
        int numCols = matrix[0].size();
        transposedMatrix.resize(numCols * numRows);

        for (int i = 0; i < numCols; ++i) {
            for (int j = 0; j < numRows; ++j) {
                transposedMatrix[i * numRows + j] = matrix[j][i];
            }
        }
    }
    return transposedMatrix;
}

std::vector<float> MultiHeadAttentionLayer::computeAttentionScores(const std::vector<float>& queries, const std::vector<float>& keys, int headSize) {
    std::vector<float> attentionScores;
    int numHeads = queries.size() / headSize;
    attentionScores.resize(numHeads * headSize);

    for (int head = 0; head < numHeads; ++head) {
        for (int i = 0; i < headSize; ++i) {
            float score = std::inner_product(queries.begin() + head * headSize, queries.begin() + (head + 1) * headSize, keys.begin() + head * headSize, 0.0f);
            attentionScores[head * headSize + i] = score / std::sqrt(headSize);
        }
    }

    return attentionScores;
}

std::vector<float> MultiHeadAttentionLayer::maskedSoftmax(const std::vector<float>& input, const std::vector<std::vector<float>>& mask) {
    std::vector<float> softmaxOutput(input.size(), 0.0f);

    for (int i = 0; i < input.size(); ++i) {
        if (mask[i / mask.size()][i % mask.size()] == 0.0f) {
            softmaxOutput[i] = std::exp(input[i]);
        }
    }

    float sum = std::reduce(softmaxOutput.begin(), softmaxOutput.end());

    for (float& value : softmaxOutput) {
        value /= sum;
    }

    return softmaxOutput;
}

std::vector<float> MultiHeadAttentionLayer::weightedSumValues(const std::vector<float>& attentionWeights, const std::vector<float>& values, int headSize) {
    std::vector<float> weightedSum(headSize, 0.0f);

    for (int i = 0; i < headSize; ++i) {
        for (int j = 0; j < attentionWeights.size(); ++j) {
            weightedSum[i] += attentionWeights[j] * values[j * headSize + i];
        }
    }

    return weightedSum;
}

std::vector<float> MultiHeadAttentionLayer::combineAttentionOutput(const std::vector<float>& attentionOutput, const std::vector<float>& weightedSum, int numHeads) {
    std::vector<float> combinedOutput(attentionOutput.size(), 0.0f);
    for (int i = 0; i < numHeads; ++i) {
        if (i == head) {
            for (int j = 0; j < weightedSum.size(); ++j) {
                combinedOutput[head * weightedSum.size() + j] = weightedSum[j];
            }
        }
        else {
            for (int j = 0; j < attentionOutput.size() / numHeads; ++j) {
                combinedOutput[i * attentionOutput.size() / numHeads + j] = attentionOutput[i * attentionOutput.size() / numHeads + j];
            }
        }
    }
    return combinedOutput;
}

std::vector<std::vector<float>> MultiHeadAttentionLayer::getAttentionMask(int inputSize) const {
    std::vector<std::vector<float>> attentionMask(inputSize, std::vector<float>(inputSize, 0.0f));
    return attentionMask;
}

void MultiHeadAttentionLayer::setQueryWeights(const std::vector<std::vector<float>>& queryWeights) {
    this->queryWeights = queryWeights;
}

void MultiHeadAttentionLayer::setQueryBiases(const std::vector<float>& queryBiases) {
    this->queryBiases = queryBiases;
}

void MultiHeadAttentionLayer::setKeyWeights(const std::vector<std::vector<float>>& keyWeights) {
    this->keyWeights = keyWeights;
}

void MultiHeadAttentionLayer::setKeyBiases(const std::vector<float>& keyBiases) {
    this->keyBiases = keyBiases;
}

void MultiHeadAttentionLayer::setValueWeights(const std::vector<std::vector<float>>& valueWeights) {
    this->valueWeights = valueWeights;
}

void MultiHeadAttentionLayer::setValueBiases(const std::vector<float>& valueBiases) {
    this->valueBiases = valueBiases;
}

std::vector<float> MultiHeadAttentionLayer::getOutput() const {
    return output;
}

std::vector<float> MultiHeadAttentionLayer::getTransformedQueries() const {
    return transformedQueries;
}

std::vector<float> MultiHeadAttentionLayer::getTransformedKeys() const {
    return transformedKeys;
}

std::vector<float> MultiHeadAttentionLayer::getTransformedValues() const {
    return transformedValues;
}

void MultiHeadAttentionLayer::backward() {
}


void MultiHeadAttentionLayer::setHeadSize(int headSize) {
    this->headSize = headSize;
}

void MultiHeadAttentionLayer::setNumHeads(int numHeads) {
    this->numHeads = numHeads;
}

int MultiHeadAttentionLayer::getHeadSize() const {
    return headSize;
}

int MultiHeadAttentionLayer::getNumHeads() const {
    return numHeads;
}

void MultiHeadAttentionLayer::setInputQueries(const std::vector<float>& queries) {
    inputQueries = queries;
}

void MultiHeadAttentionLayer::setInputKeys(const std::vector<float>& keys) {
    inputKeys = keys;
}

void MultiHeadAttentionLayer::setInputValues(const std::vector<float>& values) {
    inputValues = values;
}

std::vector<float> MultiHeadAttentionLayer::getInputQueries() const {
    return inputQueries;
}

std::vector<float> MultiHeadAttentionLayer::getInputKeys() const {
    return inputKeys;
}

std::vector<float> MultiHeadAttentionLayer::getInputValues() const {
    return inputValues;
}
