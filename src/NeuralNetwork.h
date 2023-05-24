#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Neuron.h"
#include <unordered_map>
#include <memory>
#include <vector>

class NeuralNetwork {
private:
  std::unordered_map<std::string, std::shared_ptr<Neuron>> neurons;

public:
  void addNeuron(const std::string& name, const std::shared_ptr<Neuron>& neuron);
  Neuron* getNeuron(const std::string& name) const;
  void forwardPropagation();
  void backwardPropagation(float learningRate);
  void randomizeWeights(float minWeight, float maxWeight);
  void printNetworkState() const;
  void setInputValues(const std::unordered_map<std::string, float>& inputValues);
  std::vector<float> getOutputValues() const;
  void clearNeurons();
  void removeNeuron(const std::string& name);
  size_t getNumNeurons() const;
  void optimizeWeights(float targetValue, float learningRate);
};

#endif  // NEURALNETWORK_H
