#include "Neuron.h"
#include <stdexcept>
#include <cmath>

Neuron::Neuron(const std::string& name)
  : name(name), weight(0.0f), input(0.0f), output(0.0f) {
}

void Neuron::setWeight(float weight) {
  if (weight >= 0) {
    this->weight = weight;
  } else {
    throw std::invalid_argument("Weight must be a non-negative value.");
  }
}

float Neuron::getWeight() const {
  return weight;
}

void Neuron::setInput(float input) {
  this->input = input;
}

float Neuron::getInput() const {
  return input;
}

float Neuron::getOutput() const {
  return output;
}

void LinearNeuron::forward() {
  output = getInput() * getWeight();
}

void LinearNeuron::backward(float learningRate) {
  float gradient = getInput();
  float weightUpdate = learningRate * gradient;
  setWeight(getWeight() - weightUpdate);
}

void ActivationNeuron::forward() {
  output = activationFunc(getInput());
}

void ActivationNeuron::backward(float learningRate) {
  float gradient = getInput() * derivative;
  float weightUpdate = learningRate * gradient;
  setWeight(getWeight() - weightUpdate);
}

float sigmoidActivationFunction(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}
