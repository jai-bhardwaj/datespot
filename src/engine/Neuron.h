#ifndef NEURON_H
#define NEURON_H

#include <string>
#include <functional>

class Neuron {
protected:
  std::string name;
  float weight;
  float input;
  float output;

public:
  Neuron() = default;
  explicit Neuron(const std::string& name);

  void setWeight(float weight);
  [[nodiscard]] float getWeight() const;

  void setInput(float input);
  [[nodiscard]] float getInput() const;

  [[nodiscard]] float getOutput() const;

  virtual void forward() = 0;
  virtual void backward(float learningRate) = 0;
};

class LinearNeuron : public Neuron {
public:
  using Neuron::Neuron;
  void forward() override;
  void backward(float learningRate) override;
};

class ActivationNeuron : public Neuron {
private:
  std::function<float(float)> activationFunc;
  float derivative;

public:
  ActivationNeuron(const std::string& name, std::function<float(float)> activationFunc);
  void forward() override;
  void backward(float learningRate) override;
};

ActivationNeuron::ActivationNeuron(const std::string& name, std::function<float(float)> activationFunc)
  : Neuron(name), activationFunc(activationFunc) {
}

[[nodiscard]] float sigmoidActivationFunction(float x);

#endif  // NEURON_H
