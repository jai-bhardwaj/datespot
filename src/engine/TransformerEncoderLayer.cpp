#include "TransformerEncoderLayer.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <execution>

void applyMaskKernel(const float* input, const float* mask, float* maskedInput, int sequenceLength, int hiddenSize) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < sequenceLength * hiddenSize) {
        auto row = idx / hiddenSize;
        auto col = idx % hiddenSize;
        maskedInput[idx] = input[idx] * mask[row * hiddenSize + col];
    }
}

void TransformerEncoderLayer::forward(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& mask) {
    const auto& inputBatch = input;

    auto positionalEncodedInput = positionalEncodingLayer_.forward(inputBatch);
    auto preprocessedInput = layerNorm1_.forward(positionalEncodedInput);

    auto attentionOutput = multiHeadAttentionLayer_.forward(preprocessedInput, mask);
    auto attentionOutputWithDropout = dropout_.forward(attentionOutput);
    auto attentionOutputResidual = residualConnection(preprocessedInput, attentionOutputWithDropout);
    auto attentionOutputNorm = layerNorm2_.forward(attentionOutputResidual);

    auto feedForwardOutput = feedForwardNetworkLayer_.forward(attentionOutputNorm, mask);
    auto feedForwardOutputWithDropout = dropout_.forward(feedForwardOutput);
    auto feedForwardOutputResidual = residualConnection(attentionOutputNorm, feedForwardOutputWithDropout);
    auto feedForwardOutputNorm = layerNorm1_.forward(feedForwardOutputResidual);

    output_ = feedForwardOutputNorm;
}

std::vector<std::vector<float>> TransformerEncoderLayer::applyMask(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& mask) {
    const auto& inputBatch = input;
    auto batchSize = static_cast<int>(inputBatch.size());
    auto sequenceLength = static_cast<int>(inputBatch[0].size());
    auto hiddenSize = static_cast<int>(inputBatch[0][0].size());

    std::vector<std::vector<float>> maskedInput(batchSize, std::vector<float>(sequenceLength * hiddenSize));

    static std::unique_ptr<float[]> d_input = nullptr;
    static std::unique_ptr<float[]> d_mask = nullptr;
    static std::unique_ptr<float[]> d_maskedInput = nullptr;
    static bool initialized = false;

    if (!initialized) {
        d_input = std::make_unique<float[]>(batchSize * sequenceLength * hiddenSize);
        d_mask = std::make_unique<float[]>(batchSize * sequenceLength * hiddenSize);
        d_maskedInput = std::make_unique<float[]>(batchSize * sequenceLength * hiddenSize);
        initialized = true;
    }

    cudaMemcpyAsync(d_input.get(), inputBatch.data(), batchSize * sequenceLength * hiddenSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_mask.get(), mask.data(), batchSize * sequenceLength * hiddenSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    auto threadsPerBlock = 256;
    auto numBlocks = (batchSize * sequenceLength * hiddenSize + threadsPerBlock - 1) / threadsPerBlock;
    applyMaskKernel<<<numBlocks, threadsPerBlock>>>(d_input.get(), d_mask.get(), d_maskedInput.get(), sequenceLength, hiddenSize);

    cudaMemcpyAsync(maskedInput.data(), d_maskedInput.get(), batchSize * sequenceLength * hiddenSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    return maskedInput;
}

std::vector<std::vector<float>> TransformerEncoderLayer::getOutput() const {
    return output_;
}

std::vector<std::vector<float>> TransformerEncoderLayer::applyMask(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& mask) {
    const auto& inputBatch = input;
    auto batchSize = static_cast<int>(inputBatch.size());
    auto sequenceLength = static_cast<int>(inputBatch[0].size());
    auto hiddenSize = static_cast<int>(inputBatch[0][0].size());

    std::vector<std::vector<float>> maskedInput(batchSize, std::vector<float>(sequenceLength * hiddenSize));

    #pragma omp parallel for collapse(2)
    for (auto i = 0; i < batchSize; i++) {
        for (auto j = 0; j < sequenceLength; j++) {
            std::transform(std::execution::par, inputBatch[i][j].begin(), inputBatch[i][j].end(), mask[i][j].begin(), maskedInput[i].begin() + (j * hiddenSize),
                           [](float x, float y) { return x * y; });
        }
    }

    return maskedInput;
}

std::vector<std::vector<float>> TransformerEncoderLayer::residualConnection(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& output) {
    const auto& inputBatch = input;
    auto batchSize = static_cast<int>(inputBatch.size());
    auto sequenceLength = static_cast<int>(inputBatch[0].size());
    auto hiddenSize = static_cast<int>(inputBatch[0][0].size());

    std::vector<std::vector<float>> residualOutput(batchSize, std::vector<float>(sequenceLength, std::vector<float>(hiddenSize)));

    #pragma omp parallel for collapse(3)
    for (auto i = 0; i < batchSize; i++) {
        for (auto j = 0; j < sequenceLength; j++) {
            std::transform(std::execution::par, inputBatch[i][j].begin(), inputBatch[i][j].end(), output[i][j].begin(), residualOutput[i][j].begin(),
                           [](float x, float y) { return x + y; });
        }
    }

    return residualOutput;
}

void TransformerEncoderLayer::setInput(const std::vector<std::vector<float>>& input) {
    input_ = input;
}

std::vector<std::vector<float>> TransformerEncoderLayer::getInput() const {
    return input_;
}

std::vector<std::vector<float>> TransformerEncoderLayer::getAttentionWeights() const {
    return multiHeadAttentionLayer_.getAttentionWeights();
}

std::vector<std::vector<float>> TransformerEncoderLayer::getOutputMask() const {
    return multiHeadAttentionLayer_.getOutputMask();
}

void TransformerEncoderLayer::backward(const std::vector<std::vector<float>>& inputGrad, const std::vector<std::vector<float>>& outputGrad) {
    std::vector<std::vector<float>> feedForwardNetworkGrads = feedForwardNetworkLayer_.backward(outputGrad, attentionOutputNorm);
    feedForwardNetworkLayer_.updateParameters(feedForwardNetworkGrads);

    std::vector<std::vector<float>> feedForwardOutputNormGrads = layerNorm1_.backward(feedForwardOutputResidual, feedForwardNetworkGrads);
    std::vector<std::vector<float>> feedForwardOutputResidualGrads = residualConnection(attentionOutputNorm, feedForwardOutputNormGrads);

    std::vector<std::vector<float>> multiHeadAttentionGrads = multiHeadAttentionLayer_.backward(feedForwardOutputResidualGrads, preprocessedInput);
    multiHeadAttentionLayer_.updateParameters(multiHeadAttentionGrads);

    std::vector<std::vector<float>> attentionOutputNormGrads = layerNorm2_.backward(attentionOutputResidual, multiHeadAttentionGrads);
    std::vector<std::vector<float>> attentionOutputResidualGrads = residualConnection(preprocessedInput, attentionOutputNormGrads);

    std::vector<std::vector<float>> positionalEncodingGrads = positionalEncodingLayer_.backward(attentionOutputResidualGrads, inputGrad);
}

std::vector<std::vector<float>> TransformerEncoderLayer::getMultiHeadAttentionParameters() const {
    return multiHeadAttentionLayer_.getParameters();
}

std::vector<std::vector<float>> TransformerEncoderLayer::getFeedForwardNetworkParameters() const {
    return feedForwardNetworkLayer_.getParameters();
}

void TransformerEncoderLayer::setMultiHeadAttentionParameters(const std::vector<std::vector<float>>& parameters) {
    multiHeadAttentionLayer_.setParameters(parameters);
}

void TransformerEncoderLayer::setFeedForwardNetworkParameters(const std::vector<std::vector<float>>& parameters) {
    feedForwardNetworkLayer_.setParameters(parameters);
}
