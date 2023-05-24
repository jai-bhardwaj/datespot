#ifndef FEED_FORWARD_NETWORK_LAYER_H
#define FEED_FORWARD_NETWORK_LAYER_H

#include "Layer.h"
#include <string>

class FeedForwardNetworkLayer : public Layer {
private:
    std::vector<std::vector<float>> weight1_;
    std::vector<float> bias1_;
    std::vector<std::vector<float>> weight2_;
    std::vector<float> bias2_;

public:
    FeedForwardNetworkLayer(const std::string& name);

    void initialize() override;
    void forward() override;

private:
    std::vector<float> linearTransform(const std::vector<float>& input, const std::vector<std::vector<float>>& weights, const std::vector<float>& biases);
    std::vector<float> relu(const std::vector<float>& input);
};

#endif // FEED_FORWARD_NETWORK_LAYER_H
