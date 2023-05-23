#ifndef MULTI_HEAD_ATTENTION_LAYER_H
#define MULTI_HEAD_ATTENTION_LAYER_H

#include "Layer.h"
#include <string>
#include <vector>

class MultiHeadAttentionLayer : public Layer {
private:
    std::vector<std::vector<float>> queryWeights;
    std::vector<std::vector<float>> keyWeights;
    std::vector<std::vector<float>> valueWeights;
    std::vector<float> queryBiases;
    std::vector<float> keyBiases;
    std::vector<float> valueBiases;
    int headSize;
    int numHeads;

    std::vector<float> inputQueries;
    std::vector<float> inputKeys;
    std::vector<float> inputValues;

public:
    MultiHeadAttentionLayer(const std::string& name, const std::vector<std::vector<float>>& queryWeights, const std::vector<float>& queryBiases, const std::vector<std::vector<float>>& keyWeights, const std::vector<float>& keyBiases, const std::vector<std::vector<float>>& valueWeights, const std::vector<float>& valueBiases, int headSize, int numHeads);

    void initialize() override;
    void forward() override;
    void backward(); // Placeholder for backward propagation

    // Additional functionality
    void setHeadSize(int headSize);
    void setNumHeads(int numHeads);
    int getHeadSize() const;
    int getNumHeads() const;

    void setInputQueries(const std::vector<float>& queries);
    void setInputKeys(const std::vector<float>& keys);
    void setInputValues(const std::vector<float>& values);
    std::vector<float> getInputQueries() const;
    std::vector<float> getInputKeys() const;
    std::vector<float> getInputValues() const;

    std::vector<float> getOutput() const;
    std::vector<float> getTransformedQueries() const;
    std::vector<float> getTransformedKeys() const;
    std::vector<float> getTransformedValues() const;

private:
    std::vector<float> linearTransform(const std::vector<float>& input, const std::vector<std::vector<float>>& weights, const std::vector<float>& biases);
    std::vector<float> transpose(const std::vector<std::vector<float>>& matrix);
    std::vector<float> computeAttentionScores(const std::vector<float>& queries, const std::vector<float>& keys, int headSize);
    std::vector<float> maskedSoftmax(const std::vector<float>& input, const std::vector<std::vector<float>>& mask);
    std::vector<float> weightedSumValues(const std::vector<float>& attentionWeights, const std::vector<float>& values, int headSize);
    std::vector<float> combineAttentionOutput(const std::vector<float>& attentionOutput, const std::vector<float>& weightedSum, int numHeads);
    std::vector<std::vector<float>> getAttentionMask(int inputSize) const;
};

#endif // MULTI_HEAD_ATTENTION_LAYER_H
