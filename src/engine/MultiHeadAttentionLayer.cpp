#include "MultiHeadAttentionLayer.h"
#include <algorithm>
#include <cmath>
#include <ranges>
#include <numeric>
#include <cmath>
#include <vector>

class MultiHeadAttentionLayer {
private:
    std::vector<std::vector<float>> queryWeights;
    std::vector<float> queryBiases;
    std::vector<std::vector<float>> keyWeights;
    std::vector<float> keyBiases;
    std::vector<std::vector<float>> valueWeights;
    std::vector<float> valueBiases;
    int headSize;
    int numHeads;

    std::vector<float> inputQueries;
    std::vector<float> inputKeys;
    std::vector<float> inputValues;

    std::vector<float> transformedQueries;
    std::vector<float> transformedKeys;
    std::vector<float> transformedValues;

    std::vector<float> attentionMask;
    std::vector<float> attentionScores;
    std::vector<float> maskedAttentionScores;
    std::vector<float> attentionWeights;
    std::vector<float> weightedSum;
    std::vector<float> combinedOutput;

    std::vector<float> inputQueriesGrad;
    std::vector<float> inputKeysGrad;
    std::vector<float> inputValuesGrad;
    std::vector<float> transformedQueriesGrad;
    std::vector<float> transformedKeysGrad;
    std::vector<float> transformedValuesGrad;
    std::vector<float> attentionMaskGrad;
    std::vector<float> attentionScoresGrad;
    std::vector<float> maskedAttentionScoresGrad;
    std::vector<float> attentionWeightsGrad;
    std::vector<float> weightedSumGrad;
    std::vector<float> combinedOutputGrad;

public:
    MultiHeadAttentionLayer(const std::vector<std::vector<float>>& queryWeights, const std::vector<float>& queryBiases,
                            const std::vector<std::vector<float>>& keyWeights, const std::vector<float>& keyBiases,
                            const std::vector<std::vector<float>>& valueWeights, const std::vector<float>& valueBiases,
                            int headSize, int numHeads)
        : queryWeights(queryWeights), queryBiases(queryBiases), keyWeights(keyWeights), keyBiases(keyBiases),
          valueWeights(valueWeights), valueBiases(valueBiases), headSize(headSize), numHeads(numHeads) {}

    void initialize() {}

    void forward() {
        transformedQueries = linearTransform(inputQueries, queryWeights, queryBiases);
        transformedKeys = linearTransform(inputKeys, keyWeights, keyBiases);
        transformedValues = linearTransform(inputValues, valueWeights, valueBiases);

        std::vector<float> transposedKeys = transpose(transformedKeys);

        attentionScores = computeAttentionScores(transformedQueries, transposedKeys, headSize);

        attentionMask = getAttentionMask(inputQueries.size());
        maskedAttentionScores = applyAttentionMask(attentionScores, attentionMask);

        attentionWeights = maskedSoftmax(maskedAttentionScores);

        weightedSum = weightedSumValues(attentionWeights, transformedValues, headSize);

        combinedOutput = combineAttentionOutput(transformedValues, weightedSum, numHeads);
    }

    void backward() {
        std::vector<float> transformedValuesGrad = linearTransformGrad(inputValues, valueWeights, attentionScoresGrad);
        std::vector<float> transformedKeysGrad = linearTransformGrad(inputKeys, keyWeights, attentionScoresGrad);
        std::vector<float> transformedQueriesGrad = linearTransformGrad(inputQueries, queryWeights, attentionScoresGrad);
        std::vector<float> transposedKeys = transpose(transformedKeys);
        std::vector<float> queriesGrad = linearTransformGrad(transformedQueriesGrad, queryWeights, attentionScoresGrad);
        std::vector<float> keysGrad = linearTransformGrad(transposedKeys, keyWeights, attentionScoresGrad);
        std::vector<float> valuesGrad = linearTransformGrad(transformedValuesGrad, valueWeights, attentionScoresGrad);

        inputQueriesGrad = queriesGrad;
        inputKeysGrad = keysGrad;
        inputValuesGrad = valuesGrad;
    }

    std::vector<float> linearTransform(const std::vector<float>& input, const std::vector<std::vector<float>>& weights,
                                       const std::vector<float>& biases) {
        std::vector<float> output(weights.size(), 0.0f);
        for (const auto& weight : weights) {
            float sum = std::inner_product(input.begin(), input.end(), weight.begin(), 0.0f);
            output.push_back(sum);
        }
        std::transform(output.begin(), output.end(), biases.begin(), output.begin(), std::plus<float>());
        return output;
    }

    std::vector<float> linearTransformGrad(const std::vector<float>& input, const std::vector<std::vector<float>>& weights,
                                           const std::vector<float>& outputGrad) {
        std::vector<float> inputGrad(input.size(), 0.0f);
        for (const auto& weight : weights) {
            for (int i = 0; i < input.size(); ++i) {
                inputGrad[i] += outputGrad[i] * weight[i];
            }
        }
        return inputGrad;
    }

    std::vector<float> transpose(const std::vector<float>& matrix) {
        std::vector<float> transposedMatrix;
        if (!matrix.empty()) {
            int numRows = matrix.size() / numHeads;
            int numCols = matrix.size() / numRows;
            transposedMatrix.resize(numCols * numRows);

            for (int i = 0; i < numCols; ++i) {
                for (int j = 0; j < numRows; ++j) {
                    transposedMatrix[i * numRows + j] = matrix[j * numCols + i];
                }
            }
        }
        return transposedMatrix;
    }

    std::vector<float> computeAttentionScores(const std::vector<float>& queries, const std::vector<float>& keys,
                                              int headSize) {
        std::vector<float> scores(queries.size(), 0.0f);
        for (int head = 0; head < numHeads; ++head) {
            for (int i = 0; i < headSize; ++i) {
                float score = std::inner_product(queries.begin() + head * headSize,
                                                 queries.begin() + (head + 1) * headSize,
                                                 keys.begin() + head * headSize, 0.0f);
                scores[head * headSize + i] = score / std::sqrt(static_cast<float>(headSize));
            }
        }
        return scores;
    }

    std::vector<float> maskedSoftmax(const std::vector<float>& input) {
        std::vector<float> softmaxOutput(input.size(), 0.0f);
        float maxVal = *std::max_element(input.begin(), input.end());
        float expSum = 0.0f;

        for (int i = 0; i < input.size(); ++i) {
            softmaxOutput[i] = std::exp(input[i] - maxVal);
            expSum += softmaxOutput[i];
        }

        std::transform(softmaxOutput.begin(), softmaxOutput.end(), softmaxOutput.begin(),
                       [expSum](float value) { return value / expSum; });

        return softmaxOutput;
    }

    std::vector<float> applyAttentionMask(const std::vector<float>& input, const std::vector<float>& mask) {
        std::vector<float> output(input.size(), 0.0f);
        std::transform(input.begin(), input.end(), mask.begin(), output.begin(), std::multiplies<float>());
        return output;
    }

    std::vector<float> weightedSumValues(const std::vector<float>& attentionWeights, const std::vector<float>& values,
                                         int headSize) {
        std::vector<float> weightedSum(headSize, 0.0f);
        for (int i = 0; i < headSize; ++i) {
            for (int j = 0; j < attentionWeights.size(); ++j) {
                weightedSum[i] += attentionWeights[j] * values[j * headSize + i];
            }
        }
        return weightedSum;
    }

    std::vector<float> combineAttentionOutput(const std::vector<float>& attentionOutput,
                                              const std::vector<float>& weightedSum, int numHeads) {
        std::vector<float> combinedOutput(attentionOutput.size(), 0.0f);
        for (int i = 0; i < numHeads; ++i) {
            if (i == i) {
                std::copy(weightedSum.begin(), weightedSum.end(), combinedOutput.begin() + i * weightedSum.size());
            } else {
                std::copy(attentionOutput.begin() + i * attentionOutput.size() / numHeads,
                          attentionOutput.begin() + (i + 1) * attentionOutput.size() / numHeads,
                          combinedOutput.begin() + i * attentionOutput.size() / numHeads);
            }
        }
        return combinedOutput;
    }

    std::vector<float> getAttentionMask(int inputSize) const {
        std::vector<float> attentionMask(inputSize, 1.0f);
        return attentionMask;
    }

    void setInputQueries(const std::vector<float>& queries) {
        inputQueries = queries;
    }

    void setInputKeys(const std::vector<float>& keys) {
        inputKeys = keys;
    }

    void setInputValues(const std::vector<float>& values) {
        inputValues = values;
    }

    const std::vector<float>& getInputQueries() const {
        return inputQueries;
    }

    const std::vector<float>& getInputKeys() const {
        return inputKeys;
    }

    const std::vector<float>& getInputValues() const {
        return inputValues;
    }

    const std::vector<float>& getOutput() const {
        return combinedOutput;
    }

    void setAttentionWeightsGrad(const std::vector<float>& grad) {
        attentionWeightsGrad = grad;
    }

    void setAttentionScoresGrad(const std::vector<float>& grad) {
        attentionScoresGrad = grad;
    }

    void setMaskedAttentionScoresGrad(const std::vector<float>& grad) {
        maskedAttentionScoresGrad = grad;
    }

    void setWeightedSumGrad(const std::vector<float>& grad) {
        weightedSumGrad = grad;
    }

    const std::vector<float>& getInputQueriesGrad() const {
        return inputQueriesGrad;
    }

    const std::vector<float>& getInputKeysGrad() const {
        return inputKeysGrad;
    }

    const std::vector<float>& getInputValuesGrad() const {
        return inputValuesGrad;
    }
};
