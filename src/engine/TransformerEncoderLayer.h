#ifndef TRANSFORMER_ENCODER_LAYER_H
#define TRANSFORMER_ENCODER_LAYER_H

#include "Layer.h"
#include "MultiHeadAttentionLayer.h"
#include "FeedForwardNetworkLayer.h"
#include <string>

class TransformerEncoderLayer : public Layer {
private:
    MultiHeadAttentionLayer multiHeadAttentionLayer_;
    FeedForwardNetworkLayer feedForwardNetworkLayer_;

public:
    TransformerEncoderLayer(const std::string& name);

    void initialize() override;
    void forward() override;
};

#endif // TRANSFORMER_ENCODER_LAYER_H