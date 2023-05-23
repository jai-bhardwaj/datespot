#ifndef MULTI_HEAD_ATTENTION_LAYER_H
#define MULTI_HEAD_ATTENTION_LAYER_H

#include "Layer.h"
#include <string>

class MultiHeadAttentionLayer : public Layer {
private:
    std::vector<std::vector<float>> weightQueries_;
    std::vector<std::vector<float>> weightKeys_;
    std::vector<std::vector<float>> weightValues_;
    std::vector<float> biasQueries_;
    std::vector<float> biasKeys_;
    std::vector<float> biasValues_;

public:
    MultiHeadAttentionLayer(const std::string& name);

    void initialize() override;
    void forward() override;

private:
    std::vector<float> linearTransform(const std::vector<float>& input, const std::vector<std::vector<float>>& weights, const std::vector<float>& biases);
    std::vector<float> transpose(const std::vector<std::vector<float>>& matrix);
    std::vector<float> computeAttentionScores(const std::vector<float>& queries, const std::vector<float>& keys, int headSize);
    std::vector<float> maskedSoftmax(const std::vector<float>& input, const std::vector<std::vector<float>>& mask);
    std::vector<float> weightedSumValues(const std::vector<float>& attentionWeights, const std::vector<float>& values, int headSize);
    std::vector<float> combineAttentionOutput(const std::vector<float>& attentionOutput, const std::vector<float>& weightedSum, int head, int numHeads);
    std::vector<std::vector<float>> getAttentionMask(int inputSize) const;
};

#endif // MULTI_HEAD_ATTENTION_LAYER_H
