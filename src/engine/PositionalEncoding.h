#ifndef POSITIONAL_ENCODING_H
#define POSITIONAL_ENCODING_H

#include <vector>

class PositionalEncoding {
public:
    PositionalEncoding(int maxSeqLength, int hiddenSize);
    std::vector<float> operator()(int position) const;
    int reverseEncoding(const std::vector<float>& encodingVector) const;

private:
    std::vector<std::vector<float>> encoding_;
    std::vector<std::vector<float>> generatePositionalEncoding(int maxSeqLength, int hiddenSize);
};

#endif  // POSITIONAL_ENCODING_H
