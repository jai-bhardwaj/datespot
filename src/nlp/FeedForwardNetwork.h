#ifndef FEED_FORWARD_NETWORK_H
#define FEED_FORWARD_NETWORK_H

#include <vector>

class FeedForwardNetwork {
public:
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& input);
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& grads);
    void updateParameters(const std::vector<std::vector<float>>& gradients);

    std::vector<std::vector<float>> getParameters() const;
    void setParameters(const std::vector<std::vector<float>>& parameters);

private:
    // TODO Private member functions and variables

};

#endif  // FEED_FORWARD_NETWORK_H