#include "TransformerEncoderLayer.h"

TransformerEncoderLayer::TransformerEncoderLayer(const std::string& name) : multiHeadAttentionLayer_(name), feedForwardNetworkLayer_(name) {
    // Implementation of the constructor
}

void TransformerEncoderLayer::initialize() {
    multiHeadAttentionLayer_.initialize();
    feedForwardNetworkLayer_.initialize();
}

void TransformerEncoderLayer::forward() {
    const std::vector<float>& input = input_;

    std::vector<float> attentionOutput = multiHeadAttentionLayer_.getOutput();
    std::vector<float> attentionOutputNorm = layerNormalization_(attentionOutput);

    feedForwardNetworkLayer_.setInput(attentionOutputNorm);
    feedForwardNetworkLayer_.forward();
    std::vector<float> feedForwardOutput = feedForwardNetworkLayer_.getOutput();
    std::vector<float> feedForwardOutputNorm = layerNormalization_(feedForwardOutput);

    std::vector<float> output = ResidualConnection::add(attentionOutputNorm, feedForwardOutputNorm);

    output_ = output;
}
