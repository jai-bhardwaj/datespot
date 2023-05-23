#ifndef TRANSFORMER_MODEL_H
#define TRANSFORMER_MODEL_H

#include "FeedForwardNetworkLayer.h"
#include "TransformerEncoderLayer.h"
#include "PositionalEncoding.h"
#include <vector>

class TransformerModel : public FeedForwardNetworkLayer {
private:
    std::vector<TransformerEncoderLayer> transformerEncoderLayers_;
    PositionalEncoding positionalEncoding_;

public:
    TransformerModel(int numLayers, int numHeads, int hiddenSize, int feedForwardSize);

    void initialize() override;
    void forward() override;

private:
    std::vector<float> embeddingLayer_(const std::vector<float>& input);
    std::vector<float> applyPositionalEncoding(const std::vector<float>& input);
    std::vector<float> layerNormalization_(const std::vector<float>& input);
};

#endif // TRANSFORMER_MODEL_H
