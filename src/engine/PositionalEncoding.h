#ifndef POSITIONAL_ENCODING_H
#define POSITIONAL_ENCODING_H

#include <vector>

class PositionalEncoding {
public:
    PositionalEncoding(int maxSeqLength, int hiddenSize);
    std::vector<float> operator()(int position) const;
    int reverseEncoding(const std::vector<float>& encodingVector) const;
    int getMaxSequenceLength() const;
    int getHiddenSize() const;
    void printEncoding() const;
    std::vector<std::vector<float>> getEncodingMatrix() const;
    std::vector<float> getPositionalEncoding(int position) const;
    int findClosestPosition(const std::vector<float>& encodingVector) const;
    std::vector<int> findClosestPositions(const std::vector<std::vector<float>>& encodingVectors, int k) const;
    std::vector<float> interpolatePositionalEncoding(int start, int end, int numSteps) const;
    std::vector<std::vector<float>> getSubsequenceEncoding(int start, int end) const;
    std::vector<float> getAverageEncoding() const;

private:
    std::vector<std::vector<float>> encoding_;
    std::vector<std::vector<float>> generatePositionalEncoding(int maxSeqLength, int hiddenSize);
    float calculateDistance(const std::vector<float>& encodingVector1, const std::vector<float>& encodingVector2) const;
};

#endif  // POSITIONAL_ENCODING_H
