#ifndef TRANSFORMER_MODEL_H
#define TRANSFORMER_MODEL_H

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "Masking.h"
#include "TransformerEncoderLayer.h"
#include "PositionalEncoding.h"
#include "FeedForwardNetworkLayer.h"

class TransformerModel {
private:
    std::vector<TransformerEncoderLayer> transformerEncoderLayers_;
    PositionalEncoding positionalEncoding_;
    std::vector<float> input_;
    std::vector<float> output_;

public:
    TransformerModel(int numLayers, int numHeads, int hiddenSize, int feedForwardSize);

    void initialize();
    void forward();

    void setInput(const std::vector<float>& input);
    const std::vector<float>& getOutput() const;

private:
    std::vector<float> embeddingLayer(const std::vector<float>& input);
    std::vector<float> applyPositionalEncoding(const std::vector<float>& input);
    std::vector<float> layerNormalization(std::vector<float>&& input);
};

#endif // TRANSFORMER_MODEL_H
