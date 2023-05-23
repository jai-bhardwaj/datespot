#ifndef LAYER_NORMALIZATION_H
#define LAYER_NORMALIZATION_H

#include "Layer.h"
#include <string>

class LayerNormalization : public Layer {
private:
    std::vector<float> scale_;
    std::vector<float> bias_;

public:
    LayerNormalization(const std::string& name);

    void initialize() override;
    void forward() override;

private:
    float mean();
    float stdDev();
};

#endif // LAYER_NORMALIZATION_H
