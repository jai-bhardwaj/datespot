#include "TransformerModel.h"

TransformerModel::TransformerModel(int numLayers, int numHeads, int hiddenSize, int feedForwardSize)
    : transformerEncoderLayers_(numLayers), positionalEncoding_(1000, hiddenSize) {
    // Initialize the transformer encoder layers
    for (int i = 0; i < numLayers; ++i) {
        transformerEncoderLayers_[i] = TransformerEncoderLayer(numHeads, hiddenSize, feedForwardSize);
    }
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
    const int inputSize = input.size();
    const int embeddingSize = 512; // Define the embedding size

    std::vector<float> embeddedInput(inputSize * embeddingSize);

    // Implement the embedding layer logic
    for (int i = 0; i < inputSize; ++i) {
        // Get the input value at index i
        float inputValue = input[i];

        // Perform embedding for the input value
        for (int j = 0; j < embeddingSize; ++j) {
            // Generate the embedding value based on the input value and index j
            float embeddingValue = inputValue * j;

            // Assign the embedding value to the corresponding position in the embeddedInput vector
            embeddedInput[i * embeddingSize + j] = embeddingValue;
        }
    }

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
    // Implementation of the layer normalization
    std::vector<float> output;
    // ... (apply layer normalization to the input)
    return output;
}
