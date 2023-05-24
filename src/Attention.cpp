#include "Attention.h"

void Attention::applyAttentionMask(const std::vector<float>& input, const std::vector<float>& mask, std::vector<float>& maskedInput) {
    const auto& inputBatch = input;
    auto batchSize = static_cast<int>(inputBatch.size());
    auto sequenceLength = static_cast<int>(inputBatch[0].size());
    auto hiddenSize = static_cast<int>(inputBatch[0][0].size());

    for (auto i = 0; i < batchSize; i++) {
        for (auto j = 0; j < sequenceLength; j++) {
            maskedInput[i * sequenceLength + j] = input[i * sequenceLength + j] * mask[i * sequenceLength + j];
        }
    }
}