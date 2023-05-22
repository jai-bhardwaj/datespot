#ifndef NEURON_H
#define NEURON_H

#include <vector>

class Neuron {
private:
    std::vector<float> weights_;
    float bias_;

public:
    Neuron(int numInputs);

    float activate(const std::vector<float>& inputs) const;
};

#endif  // NEURON_H