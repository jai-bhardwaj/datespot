#include "NeuralNetwork.h"
#include <iostream>
#include <random>

void NeuralNetwork::addNeuron(const std::string& name, const std::shared_ptr<Neuron>& neuron) {
  neurons[name] = neuron;
}

Neuron* NeuralNetwork::getNeuron(const std::string& name) const {
  if (const auto it = neurons.find(name); it != neurons.end()) {
    return it->second.get();
  }
  return nullptr;
}

void NeuralNetwork::forwardPropagation() {
  for (const auto& [name, neuron] : neurons) {
    neuron->forward();
  }
}

void NeuralNetwork::backwardPropagation(float learningRate) {
  for (auto& neuron : neurons) {
    neuron.second->backward(learningRate);
  }
}

void NeuralNetwork::randomizeWeights(float minWeight, float maxWeight) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(minWeight, maxWeight);

  for (auto& [name, neuron] : neurons) {
    float weight = dist(gen);
    neuron->setWeight(weight);
  }
}

void NeuralNetwork::printNetworkState() const {
  for (const auto& [name, neuron] : neurons) {
    std::cout << "Neuron: " << name << ", Weight: " << neuron->getWeight()
              << ", Input: " << neuron->getInput() << ", Output: " << neuron->getOutput() << std::endl;
  }
}

void NeuralNetwork::setInputValues(const std::unordered_map<std::string, float>& inputValues) {
  for (const auto& [name, value] : inputValues) {
    Neuron* neuron = getNeuron(name);
    if (neuron) {
      neuron->setInput(value);
    }
  }
}

std::vector<float> NeuralNetwork::getOutputValues() const {
  std::vector<float> outputValues;
  for (const auto& [name, neuron] : neurons) {
    outputValues.push_back(neuron->getOutput());
  }
  return outputValues;
}

void NeuralNetwork::clearNeurons() {
  neurons.clear();
}

void NeuralNetwork::removeNeuron(const std::string& name) {
  neurons.erase(name);
}

size_t NeuralNetwork::getNumNeurons() const {
  return neurons.size();
}

void NeuralNetwork::optimizeWeights(float targetValue, float learningRate) {
  for (auto& neuron : neurons) {
    float input = neuron.second->getInput();
    float output = neuron.second->getOutput();
    float gradient = (output - targetValue) * input;
    neuron.second->setWeight(neuron.second->getWeight() - learningRate * gradient);
  }
}
