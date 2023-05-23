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
    const int inputSize = input.size();
    const float epsilon = 1e-6; // Small value to avoid division by zero

    std::vector<float> output(inputSize);

    // Calculate the mean of the input sequence
    float mean = 0.0f;
    for (int i = 0; i < inputSize; ++i) {
        mean += input[i];
    }
    mean /= inputSize;

    // Calculate the variance of the input sequence
    float variance = 0.0f;
    for (int i = 0; i < inputSize; ++i) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    variance /= inputSize;

    // Calculate the standard deviation
    float stdDev = sqrt(variance);

    // Apply layer normalization to the input sequence
    for (int i = 0; i < inputSize; ++i) {
        float normalizedValue = (input[i] - mean) / (stdDev + epsilon);
        output[i] = normalizedValue;
    }

    return output;
}
