#ifndef NEURON_H
#define NEURON_H

#include "Layer.h"
#include <string>

class Neuron : public Layer {
protected:
    std::vector<float> weights_;
    float bias_;

public:
    Neuron(const std::string& name);

    void initialize() override;
};

class LinearNeuron : public Neuron {
public:
    LinearNeuron(const std::string& name);

    void forward() override;
};

class ActivationNeuron : public Neuron {
public:
    ActivationNeuron(const std::string& name);

    void forward() override;

private:
    float activate(float x);
};

#endif // NEURON_H

