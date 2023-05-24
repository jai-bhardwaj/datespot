#ifndef LAYER_NORM_H
#define LAYER_NORM_H

#include <vector>

class LayerNorm {
public:
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& input);
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& grads);

private:
    // TODO Private member functions and variables

};

#endif  // LAYER_NORM_H