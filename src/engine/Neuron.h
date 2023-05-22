#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iostream>

enum class ActivationFunction {
    Sigmoid,
    ReLU,
    Step,
    Tanh
};

class Neuron {
private:
    std::vector<float> weights_;
    float bias_;
    ActivationFunction activation_;

public:
    Neuron(int numInputs, ActivationFunction activation = ActivationFunction::Sigmoid);

    float activate(const std::vector<float>& inputs) const;

    void initializeWeights(float mean = 0.0f, float stddev = 1.0f);

    void setWeights(const std::vector<float>& weights);

    void setBias(float bias);

    void setActivationFunction(ActivationFunction activation);

    int getNumInputs() const;

    std::vector<float> getWeights() const;

    float getBias() const;

    ActivationFunction getActivationFunction() const;

    static float sigmoid(float x);

    static float relu(float x);

    static float step(float x);

    static float tanh(float x);

    static std::vector<float> elementwiseMultiply(const std::vector<float>& vec1, const std::vector<float>& vec2);

    static float dotProduct(const std::vector<float>& vec1, const std::vector<float>& vec2);

    static std::vector<float> normalize(const std::vector<float>& input);

    static std::vector<float> softmax(const std::vector<float>& input);

    static float crossEntropyLoss(const std::vector<float>& predicted, const std::vector<float>& target);

    static std::vector<float> meanPooling(const std::vector<std::vector<float>>& inputs);

    static std::vector<float> maxPooling(const std::vector<std::vector<float>>& inputs);

    static std::vector<float> applyFunction(const std::vector<float>& input, float (*function)(float));

    static std::vector<float> convolve(const std::vector<float>& input, const std::vector<float>& kernel, int stride = 1, int padding = 0);

    static std::vector<std::vector<float>> im2col(const std::vector<std::vector<float>>& input, int kernelSize, int stride = 1, int padding = 0);

    static std::vector<float> col2im(const std::vector<std::vector<float>>& input, const std::vector<float>& outputShape, int kernelSize, int stride = 1, int padding = 0);

    static float meanSquaredError(const std::vector<float>& predicted, const std::vector<float>& target);

    static std::vector<float> elementwiseAdd(const std::vector<float>& vec1, const std::vector<float>& vec2);

    static std::vector<float> elementwiseSubtract(const std::vector<float>& vec1, const std::vector<float>& vec2);

    static std::vector<float> elementwiseMultiply(const std::vector<float>& vec1, const std::vector<float>& vec2);

    static std::vector<float> elementwiseDivide(const std::vector<float>& vec1, const std::vector<float>& vec2);

    static float sum(const std::vector<float>& vec);

    static float mean(const std::vector<float>& vec);

    static float variance(const std::vector<float>& vec);

    static float standardDeviation(const std::vector<float>& vec);

    static float median(const std::vector<float>& vec);

    static std::vector<float> sort(const std::vector<float>& vec);

    static std::vector<float> abs(const std::vector<float>& vec);

    static std::vector<float> sqrt(const std::vector<float>& vec);

    static std::vector<float> pow(const std::vector<float>& vec, float exponent);

    static std::vector<float> log(const std::vector<float>& vec);

    static std::vector<float> exp(const std::vector<float>& vec);

    static std::vector<float> round(const std::vector<float>& vec);

    static std::vector<float> floor(const std::vector<float>& vec);

    static std::vector<float> ceil(const std::vector<float>& vec);

    static std::vector<float> clamp(const std::vector<float>& vec, float minVal, float maxVal);

    static std::vector<float> reverse(const std::vector<float>& vec);

    static std::vector<float> diff(const std::vector<float>& vec);

    static std::vector<float> cumulativeSum(const std::vector<float>& vec);

    static std::vector<float> rollingSum(const std::vector<float>& vec, int windowSize);

    static std::vector<float> rollingMean(const std::vector<float>& vec, int windowSize);

    static std::vector<float> rollingMax(const std::vector<float>& vec, int windowSize);

    static std::vector<float> rollingMin(const std::vector<float>& vec, int windowSize);

    static std::vector<float> rollingMedian(const std::vector<float>& vec, int windowSize);

    static std::vector<float> differentiate(const std::vector<float>& vec, float dt = 1.0f);

    static std::vector<float> integrate(const std::vector<float>& vec, float dt = 1.0f);

    static void printVector(const std::vector<float>& vec);
};

#endif  // NEURON_H
