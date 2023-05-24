#include "PositionalEncoding.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>

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
    for (int pos = 0; pos < encoding_.size(); ++pos) {
        if (encoding_[pos] == encodingVector) {
            return pos;
        }
    }
    return -1;
}

int PositionalEncoding::getMaxSequenceLength() const {
    return encoding_.size();
}

int PositionalEncoding::getHiddenSize() const {
    return encoding_.empty() ? 0 : encoding_[0].size();
}

void PositionalEncoding::printEncoding() const {
    for (const auto& encodingVector : encoding_) {
        for (const auto& value : encodingVector) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
}

std::vector<std::vector<float>> PositionalEncoding::getEncodingMatrix() const {
    return encoding_;
}

std::vector<float> PositionalEncoding::getPositionalEncoding(int position) const {
    if (position < 0 || position >= encoding_.size()) {
        throw std::out_of_range("Invalid position");
    }
    return encoding_[position];
}

int PositionalEncoding::findClosestPosition(const std::vector<float>& encodingVector) const {
    float closestDistance = std::numeric_limits<float>::max();
    int closestPosition = -1;

    for (int pos = 0; pos < encoding_.size(); ++pos) {
        float distance = calculateDistance(encodingVector, encoding_[pos]);
        if (distance < closestDistance) {
            closestDistance = distance;
            closestPosition = pos;
        }
    }

    return closestPosition;
}

float PositionalEncoding::calculateDistance(const std::vector<float>& encodingVector1, const std::vector<float>& encodingVector2) const {
    if (encodingVector1.size() != encodingVector2.size()) {
        throw std::invalid_argument("Encoding vectors must have the same size");
    }

    float distance = 0.0f;
    for (int i = 0; i < encodingVector1.size(); ++i) {
        distance += std::pow(encodingVector1[i] - encodingVector2[i], 2);
    }
    return std::sqrt(distance);
}

std::vector<int> PositionalEncoding::findClosestPositions(const std::vector<std::vector<float>>& encodingVectors, int k) const {
    if (k <= 0 || k > encoding_.size()) {
        throw std::invalid_argument("Invalid value of k");
    }

    std::vector<int> closestPositions;

    for (const auto& encodingVector : encodingVectors) {
        int closestPosition = findClosestPosition(encodingVector);
        closestPositions.push_back(closestPosition);
    }

    std::sort(closestPositions.begin(), closestPositions.end());
    closestPositions.resize(k);

    return closestPositions;
}

std::vector<float> PositionalEncoding::interpolatePositionalEncoding(int start, int end, int numSteps) const {
    if (start < 0 || start >= encoding_.size() || end < 0 || end >= encoding_.size()) {
        throw std::out_of_range("Invalid start or end position");
    }
    if (numSteps <= 0) {
        throw std::invalid_argument("Invalid number of steps");
    }

    std::vector<float> interpolatedEncoding(encoding_[start].size());

    for (int i = 0; i < encoding_[start].size(); ++i) {
        float startValue = encoding_[start][i];
        float endValue = encoding_[end][i];
        float stepSize = (endValue - startValue) / numSteps;

        for (int step = 0; step <= numSteps; ++step) {
            interpolatedEncoding[i] = startValue + step * stepSize;
        }
    }

    return interpolatedEncoding;
}

std::vector<std::vector<float>> PositionalEncoding::getSubsequenceEncoding(int start, int end) const {
    if (start < 0 || start >= encoding_.size() || end < 0 || end >= encoding_.size() || start > end) {
        throw std::out_of_range("Invalid start or end position");
    }

    std::vector<std::vector<float>> subsequence(encoding_.begin() + start, encoding_.begin() + end + 1);

    return subsequence;
}

std::vector<float> PositionalEncoding::getAverageEncoding() const {
    std::vector<float> averageEncoding(encoding_[0].size(), 0.0f);

    for (const auto& encodingVector : encoding_) {
        for (int i = 0; i < encodingVector.size(); ++i) {
            averageEncoding[i] += encodingVector[i];
        }
    }

    float normalizationFactor = 1.0f / encoding_.size();

    for (int i = 0; i < averageEncoding.size(); ++i) {
        averageEncoding[i] *= normalizationFactor;
    }

    return averageEncoding;
}

