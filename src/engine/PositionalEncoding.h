#ifndef POSITIONAL_ENCODING_H
#define POSITIONAL_ENCODING_H

#include <vector>

class PositionalEncoding {
private:
    std::vector<std::vector<float>> encoding_;

public:
    PositionalEncoding(int maxSeqLength, int hiddenSize);

    std::vector<float> operator()(int position) const;

private:
    std::vector<std::vector<float>> generatePositionalEncoding(int maxSeqLength, int hiddenSize);
};

#endif // POSITIONAL_ENCODING_H
