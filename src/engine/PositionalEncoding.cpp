#include "PositionalEncoding.h"
#include <cmath>
#include <algorithm>

PositionalEncoding::PositionalEncoding(int maxSeqLength, int hiddenSize) {
    encoding_ = generatePositionalEncoding(maxSeqLength, hiddenSize);
}

std::vector<float> PositionalEncoding::operator()(int position) const {
    return encoding_[position];
}

std::vector<std::vector<float>> PositionalEncoding::generatePositionalEncoding(int maxSeqLength, int hiddenSize) {
    std::vector<std::vector<float>> encoding(maxSeqLength, std::vector<float>(hiddenSize));
    for (int pos = 0; pos < maxSeqLength; ++pos) {
        for (int i = 0; i < hiddenSize; ++i) {
            float angle = pos / std::pow(10000.0f, (2.0f * i) / hiddenSize);
            encoding[pos][i] = (pos % 2 == 0) ? std::sin(angle) : std::cos(angle);
        }
    }
    return encoding;
}

int PositionalEncoding::reverseEncoding(const std::vector<float>& encodingVector) const {
    auto iter = std::find(encoding_.begin(), encoding_.end(), encodingVector);
    if (iter != encoding_.end()) {
        int position = std::distance(encoding_.begin(), iter);
        return position;
    }
    return -1; // Return -1 if the encoding vector is not found
}

