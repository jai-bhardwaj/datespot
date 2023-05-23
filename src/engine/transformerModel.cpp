#include "TransformerModel.h"

TransformerModel::TransformerModel(int numLayers, int numHeads, int hiddenSize, int feedForwardSize)
    : transformerEncoderLayers_(numLayers), positionalEncoding_(1000, hiddenSize) {
}

void TransformerModel::initialize() {
    for (auto& layer : transformerEncoderLayers_) {
        layer.initialize();
    }
}

void TransformerModel::forward() {
    const std::vector<float>& input = input_;

    std::vector<float> embeddedInput = embeddingLayer_(input);
    std::vector<float> positionalEncodedInput = applyPositionalEncoding(embeddedInput);

    std::vector<std::vector<float>> paddingMask = Masking::createPaddingMask(input);

    for (auto& layer : transformerEncoderLayers_) {
        layer.setInput(positionalEncodedInput);
        layer.setAttentionMask(paddingMask);
        layer.forward();
        positionalEncodedInput = layer.getOutput();
    }

    std::vector<float> output = layerNormalization_(positionalEncodedInput);

    output_ = output;
}

std::vector<float> TransformerModel::embeddingLayer_(const std::vector<float>& input) {
    std::vector<float> embeddedInput(input.size());
    return embeddedInput;
}

std::vector<float> TransformerModel::applyPositionalEncoding(const std::vector<float>& input) {
    const int inputSize = input.size();
    std::vector<float> output(inputSize);

    for (int i = 0; i < inputSize; ++i) {
        std::vector<float> positionalEncoding = positionalEncoding_(i);
        output[i] = input[i] + positionalEncoding[i];
    }

    return output;
}

std::vector<float> TransformerModel::layerNormalization_(const std::vector<float>& input) {
    std::vector<float> output;
    return output;
}
