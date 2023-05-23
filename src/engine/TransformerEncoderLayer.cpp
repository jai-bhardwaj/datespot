#include "TransformerEncoderLayer.h"
#include <cuda_runtime.h>

__global__ void applyMaskKernel(const float* input, const float* mask, float* maskedInput, int sequenceLength, int hiddenSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < sequenceLength * hiddenSize) {
        int row = idx / hiddenSize;
        int col = idx % hiddenSize;
        maskedInput[idx] = input[idx] * mask[row * hiddenSize + col];
    }
}

void TransformerEncoderLayer::forward(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& mask) {
    const std::vector<std::vector<float>>& inputBatch = input;

    std::vector<std::vector<float>> positionalEncodedInput = positionalEncodingLayer_.forward(inputBatch);
    std::vector<std::vector<float>> preprocessedInput = layerNorm1_.forward(positionalEncodedInput);

    std::vector<std::vector<float>> attentionOutput = multiHeadAttentionLayer_.forward(preprocessedInput, mask);
    std::vector<std::vector<float>> attentionOutputWithDropout = dropout_.forward(attentionOutput);
    std::vector<std::vector<float>> attentionOutputResidual = residualConnection(preprocessedInput, attentionOutputWithDropout);
    std::vector<std::vector<float>> attentionOutputNorm = layerNorm2_.forward(attentionOutputResidual);

    std::vector<std::vector<float>> feedForwardOutput = feedForwardNetworkLayer_.forward(attentionOutputNorm, mask);
    std::vector<std::vector<float>> feedForwardOutputWithDropout = dropout_.forward(feedForwardOutput);
    std::vector<std::vector<float>> feedForwardOutputResidual = residualConnection(attentionOutputNorm, feedForwardOutputWithDropout);
    std::vector<std::vector<float>> feedForwardOutputNorm = layerNorm1_.forward(feedForwardOutputResidual);

    output_ = feedForwardOutputNorm;
}

std::vector<std::vector<float>> TransformerEncoderLayer::applyMask(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& mask) {
    const std::vector<std::vector<float>>& inputBatch = input;
    int batchSize = inputBatch.size();
    int sequenceLength = inputBatch[0].size();
    int hiddenSize = inputBatch[0][0].size();

    std::vector<std::vector<float>> maskedInput(batchSize, std::vector<float>(sequenceLength * hiddenSize));

    float* d_input;
    float* d_mask;
    float* d_maskedInput;
    cudaMalloc((void**)&d_input, batchSize * sequenceLength * hiddenSize * sizeof(float));
    cudaMalloc((void**)&d_mask, batchSize * sequenceLength * hiddenSize * sizeof(float));
    cudaMalloc((void**)&d_maskedInput, batchSize * sequenceLength * hiddenSize * sizeof(float));

    cudaMemcpy(d_input, inputBatch.data(), batchSize * sequenceLength * hiddenSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask.data(), batchSize * sequenceLength * hiddenSize * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (batchSize * sequenceLength * hiddenSize + threadsPerBlock - 1) / threadsPerBlock;
    applyMaskKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_mask, d_maskedInput, sequenceLength, hiddenSize);

    cudaMemcpy(maskedInput.data(), d_maskedInput, batchSize * sequenceLength * hiddenSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_mask);
    cudaFree(d_maskedInput);

    return maskedInput;
}

std::vector<std::vector<float>> TransformerEncoderLayer::getOutput() const {
    return output_;
}

std::vector<std::vector<float>> TransformerEncoderLayer::applyMask(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& mask) {
    const std::vector<std::vector<float>>& inputBatch = input;
    int batchSize = inputBatch.size();
    int sequenceLength = inputBatch[0].size();
    int hiddenSize = inputBatch[0][0].size();

    std::vector<std::vector<float>> maskedInput(batchSize, std::vector<float>(sequenceLength * hiddenSize));

    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < sequenceLength; j++) {
            for (int k = 0; k < hiddenSize; k++) {
                maskedInput[i][j * hiddenSize + k] = inputBatch[i][j][k] * mask[i][j][k];
            }
        }
    }

    return maskedInput;
}

std::vector<std::vector<float>> TransformerEncoderLayer::residualConnection(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& output) {
    const std::vector<std::vector<float>>& inputBatch = input;
    int batchSize = inputBatch.size();
    int sequenceLength = inputBatch[0].size();
    int hiddenSize = inputBatch[0][0].size();

    std::vector<std::vector<float>> residualOutput(batchSize, std::vector<float>(sequenceLength, std::vector<float>(hiddenSize)));

    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < sequenceLength; j++) {
            for (int k = 0; k < hiddenSize; k++) {
                residualOutput[i][j][k] = inputBatch[i][j][k] + output[i][j][k];
            }
        }
    }

    return residualOutput;
}


std::vector<std::vector<float>> TransformerEncoderLayer::residualConnection(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& output) {
    const std::vector<std::vector<float>>& inputBatch = input;
    int batchSize = inputBatch.size();
    int sequenceLength = inputBatch[0].size();
    int hiddenSize = inputBatch[0][0].size();

    std::vector<std::vector<float>> residualOutput(batchSize, std::vector<float>(sequenceLength, std::vector<float>(hiddenSize)));

    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < sequenceLength; j++) {
            for (int k = 0; k < hiddenSize; k++) {
                residualOutput[i][j][k] = inputBatch[i][j][k] + output[i][j][k];
            }
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
