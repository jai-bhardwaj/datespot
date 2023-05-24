#include "MultiHeadAttentionLayer.h"
#include <algorithm>
#include <cmath>
#include <ranges>
#include <numeric>
#include <vector>
#include <Layer.h>
#include <SparseVector.h>
#include <Random.h>
#include <Initializer.h>
#include <Activations.h>
#include <Optimizers.h>
#include <random>
#include <functional>
#include <span>
#include <execution>

class MultiHeadAttentionLayer : public Layer {
private:
    std::vector<SparseVector> queryWeights;
    std::vector<float> queryBiases;
    std::vector<SparseVector> keyWeights;
    std::vector<float> keyBiases;
    std::vector<SparseVector> valueWeights;
    std::vector<float> valueBiases;
    int headSize;
    int numHeads;

    SparseVector inputQueries;
    SparseVector inputKeys;
    SparseVector inputValues;

    SparseVector transformedQueries;
    SparseVector transformedKeys;
    SparseVector transformedValues;

    SparseVector attentionMask;
    SparseVector attentionScores;
    SparseVector maskedAttentionScores;
    SparseVector attentionWeights;
    SparseVector weightedSum;
    SparseVector combinedOutput;

    SparseVector inputQueriesGrad;
    SparseVector inputKeysGrad;
    SparseVector inputValuesGrad;
    SparseVector transformedQueriesGrad;
    SparseVector transformedKeysGrad;
    SparseVector transformedValuesGrad;
    SparseVector attentionMaskGrad;
    SparseVector attentionScoresGrad;
    SparseVector maskedAttentionScoresGrad;
    SparseVector attentionWeightsGrad;
    SparseVector weightedSumGrad;
    SparseVector combinedOutputGrad;

public:
    MultiHeadAttentionLayer(const std::vector<std::vector<float>>& queryWeights,
                            const std::vector<float>& queryBiases,
                            const std::vector<std::vector<float>>& keyWeights,
                            const std::vector<float>& keyBiases,
                            const std::vector<std::vector<float>>& valueWeights,
                            const std::vector<float>& valueBiases,
                            int headSize, int numHeads)
        : queryWeights(queryWeights.size()), queryBiases(queryBiases),
          keyWeights(keyWeights.size()), keyBiases(keyBiases),
          valueWeights(valueWeights.size()), valueBiases(valueBiases),
          headSize(headSize), numHeads(numHeads)
    {
        for (int i = 0; i < queryWeights.size(); ++i) {
            this->queryWeights[i] = SparseVector(queryWeights[i]);
            this->keyWeights[i] = SparseVector(keyWeights[i]);
            this->valueWeights[i] = SparseVector(valueWeights[i]);
        }
    }

    void initialize() {
        std::for_each(std::execution::par, queryWeights.begin(), queryWeights.end(), [](SparseVector& weights) {
            weights.initialize();
        });
        queryBiases.initialize();
        std::for_each(std::execution::par, keyWeights.begin(), keyWeights.end(), [](SparseVector& weights) {
            weights.initialize();
        });
        keyBiases.initialize();
        std::for_each(std::execution::par, valueWeights.begin(), valueWeights.end(), [](SparseVector& weights) {
            weights.initialize();
        });
        valueBiases.initialize();
    }

    void forward() {
        transformedQueries = std::transform_reduce(std::execution::par, queryWeights.begin(), queryWeights.end(), SparseVector(), [&](const SparseVector& sum, const SparseVector& weights) {
            return sum + inputQueries.dot(weights);
        }) + queryBiases;

        transformedKeys = std::transform_reduce(std::execution::par, keyWeights.begin(), keyWeights.end(), SparseVector(), [&](const SparseVector& sum, const SparseVector& weights) {
            return sum + inputKeys.dot(weights);
        }) + keyBiases;

        transformedValues = std::transform_reduce(std::execution::par, valueWeights.begin(), valueWeights.end(), SparseVector(), [&](const SparseVector& sum, const SparseVector& weights) {
            return sum + inputValues.dot(weights);
        }) + valueBiases;

        SparseVector transposedKeys = transformedKeys.transpose();

        attentionScores = transformedQueries.dot(transposedKeys) / std::sqrt(static_cast<float>(headSize));

        attentionMask = getAttentionMask(inputQueries.size());
        maskedAttentionScores = attentionScores * attentionMask;

        attentionWeights = maskedAttentionScores.softmax();

        weightedSum = std::transform_reduce(std::execution::par, valueWeights.begin(), valueWeights.end(), SparseVector(), std::plus<>(), [&](const SparseVector& weights, float weight) {
            return weights * weight;
        });

        combinedOutput = std::transform_reduce(std::execution::par, valueWeights.begin(), valueWeights.end(), weightedSum, SparseVector(), std::plus<>(), [&](const SparseVector& weights, const SparseVector& sum) {
            return sum.combine(weights, numHeads);
        });
    }

    void backward() {
        transformedValuesGrad = std::transform_reduce(std::execution::par, valueWeights.begin(), valueWeights.end(), attentionScoresGrad, SparseVector(), std::plus<>(), [&](const SparseVector& weights, float grad) {
            return weights * grad;
        }) * (1.0f / headSize);

        transformedKeysGrad = std::transform_reduce(std::execution::par, keyWeights.begin(), keyWeights.end(), SparseVector(), [&](const SparseVector& sum, const SparseVector& weights) {
            return sum + inputKeys.dotGrad(transposedKeys, attentionScoresGrad);
        });

        transformedQueriesGrad = std::transform_reduce(std::execution::par, queryWeights.begin(), queryWeights.end(), SparseVector(), [&](const SparseVector& sum, const SparseVector& weights) {
            return sum + inputQueries.dotGrad(weights, attentionScoresGrad);
        });

        SparseVector transposedKeys = transformedKeys.transpose();
        inputQueriesGrad = transformedQueriesGrad.dotGrad(queryWeights, attentionScoresGrad);
        inputKeysGrad = transposedKeys.dotGrad(keyWeights, attentionScoresGrad);
        inputValuesGrad = transformedValuesGrad.dotGrad(valueWeights, attentionScoresGrad);
    }

    SparseVector getAttentionMask(int inputSize) const {
        SparseVector attentionMask(inputSize, 1.0f);
        return attentionMask;
    }

    void setInputQueries(const SparseVector& queries) {
        inputQueries = queries;
    }

    void setInputKeys(const SparseVector& keys) {
        inputKeys = keys;
    }

    void setInputValues(const SparseVector& values) {
        inputValues = values;
    }

    const SparseVector& getInputQueries() const {
        return inputQueries;
    }

    const SparseVector& getInputKeys() const {
        return inputKeys;
    }

    const SparseVector& getInputValues() const {
        return inputValues;
    }

    const SparseVector& getOutput() const {
        return combinedOutput;
    }

    void setAttentionWeightsGrad(const SparseVector& grad) {
        attentionWeightsGrad = grad;
    }

    void setAttentionScoresGrad(const SparseVector& grad) {
        attentionScoresGrad = grad;
    }

    void setMaskedAttentionScoresGrad(const SparseVector& grad) {
        maskedAttentionScoresGrad = grad;
    }

    void setWeightedSumGrad(const SparseVector& grad) {
        weightedSumGrad = grad;
    }

    const SparseVector& getInputQueriesGrad() const {
        return inputQueriesGrad;
    }

    const SparseVector& getInputKeysGrad() const {
        return inputKeysGrad;
    }

    const SparseVector& getInputValuesGrad() const {
        return inputValuesGrad;
    }
};