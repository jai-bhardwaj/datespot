#ifndef TRANSFORMER_ENCODER_LAYER_H
#define TRANSFORMER_ENCODER_LAYER_H

#include "Layer.h"
#include "MultiHeadAttentionLayer.h"
#include "FeedForwardNetworkLayer.h"
#include <string>
#include <vector>

class TransformerEncoderLayer : public Layer {
private:
    MultiHeadAttentionLayer multiHeadAttentionLayer_;
    FeedForwardNetworkLayer feedForwardNetworkLayer_;
    std::vector<float> input_;
    std::vector<float> output_;

public:
    TransformerEncoderLayer(const std::string& name);

    void initialize() override;
    void forward() override;

private:
    std::vector<float> layerNormalization_(const std::vector<float>& input);
};

#endif // TRANSFORMER_ENCODER_LAYER_H