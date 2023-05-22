#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>

enum class ActivationFunction {
    Sigmoid,
    ReLU,
    Tanh
};

class FeedForwardNetwork {
private:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    ActivationFunction activationFunction;

public:
    FeedForwardNetwork(const std::vector<int>& layerSizes, ActivationFunction activation = ActivationFunction::Sigmoid)
        : activationFunction(activation) {
        initializeRandomWeights(layerSizes);
    }

    void train(const std::vector<std::vector<double>>& trainingData, double learningRate, int numIterations) {
        for (int iter = 0; iter < numIterations; ++iter) {
            for (const auto& dataPoint : trainingData) {
                std::vector<double> input(dataPoint.begin(), dataPoint.end() - 1);
                std::vector<double> targetOutput(dataPoint.end() - 1, dataPoint.end());

                std::vector<double> output = forward(input);
                backpropagate(output, targetOutput, learningRate);
            }
        }
    }

    std::vector<double> forward(const std::vector<double>& input) {
        std::vector<double> output = input;

        for (int i = 0; i < weights.size(); ++i) {
            std::vector<double> newOutput;
            const std::vector<double>& currentLayerWeights = weights[i];
            int currentLayerSize = currentLayerWeights.size() / output.size();

            for (int j = 0; j < currentLayerSize; ++j) {
                double neuronSum = biases[i];
                for (int k = 0; k < output.size(); ++k) {
                    neuronSum += output[k] * currentLayerWeights[j * output.size() + k];
                }
                newOutput.push_back(activate(neuronSum));
            }

            output = newOutput;
        }

        return output;
    }

    void saveWeights(const std::string& filePath) {
        std::ofstream weightFile(filePath);
        if (weightFile.is_open()) {
            for (const auto& layerWeights : weights) {
                for (double weight : layerWeights) {
                    weightFile << weight << " ";
                }
                weightFile << std::endl;
            }
            weightFile.close();
            std::cout << "Weights saved to " << filePath << std::endl;
        } else {
            std::cerr << "Unable to save weights file." << std::endl;
        }
    }

    void loadWeights(const std::string& filePath) {
        std::ifstream weightFile(filePath);
        if (weightFile.is_open()) {
            for (auto& layerWeights : weights) {
                for (double& weight : layerWeights) {
                    weightFile >> weight;
                }
            }
            weightFile.close();
            std::cout << "Weights loaded from " << filePath << std::endl;
        } else {
            std::cerr << "Unable to load weights file." << std::endl;
        }
    }

private:
    double activate(double x) {
        switch (activationFunction) {
            case ActivationFunction::Sigmoid:
                return 1.0 / (1.0 + std::exp(-x));
            case ActivationFunction::ReLU:
                return std::max(0.0, x);
            case ActivationFunction::Tanh:
                return std::tanh(x);
            default:
                return x;
        }
    }

    void initializeRandomWeights(const std::vector<int>& layerSizes) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        for (int i = 1; i < layerSizes.size(); ++i) {
            int currentLayerSize = layerSizes[i];
            int prevLayerSize = layerSizes[i - 1];

            std::vector<double> currentLayerWeights;
            for (int j = 0; j < currentLayerSize * prevLayerSize; ++j) {
                currentLayerWeights.push_back(dist(gen));
            }

            weights.push_back(currentLayerWeights);
            biases.push_back(0.0);
        }
    }

    void backpropagate(const std::vector<double>& output, const std::vector<double>& targetOutput, double learningRate) {
        std::vector<double> errorGradient = calculateOutputErrorGradient(output, targetOutput);
        for (int i = weights.size() - 1; i >= 0; --i) {
            std::vector<double> currentLayerInputs;
            if (i == 0) {
                currentLayerInputs = output;
            } else {
                const std::vector<double>& prevLayerOutputs = (i == weights.size() - 1) ? output : weights[i + 1];
                currentLayerInputs = prevLayerOutputs;
            }

            std::vector<double> currentLayerGradients;
            for (int j = 0; j < currentLayerInputs.size(); ++j) {
                double gradient = errorGradient[j] * activateDerivative(output[j]);
                currentLayerGradients.push_back(gradient);
            }

            updateWeights(i, currentLayerGradients, currentLayerInputs, learningRate);

            errorGradient = calculateHiddenLayerErrorGradient(i, currentLayerGradients);
        }
    }

    std::vector<double> calculateOutputErrorGradient(const std::vector<double>& output, const std::vector<double>& targetOutput) {
        std::vector<double> errorGradient;
        for (int i = 0; i < output.size(); ++i) {
            double gradient = (targetOutput[i] - output[i]) * activateDerivative(output[i]);
            errorGradient.push_back(gradient);
        }
        return errorGradient;
    }

    std::vector<double> calculateHiddenLayerErrorGradient(int layerIndex, const std::vector<double>& prevLayerGradients) {
        const std::vector<double>& currentLayerWeights = weights[layerIndex];
        int currentLayerSize = currentLayerWeights.size() / prevLayerGradients.size();

        std::vector<double> errorGradient;
        for (int i = 0; i < currentLayerSize; ++i) {
            double gradientSum = 0.0;
            for (int j = 0; j < prevLayerGradients.size(); ++j) {
                gradientSum += prevLayerGradients[j] * currentLayerWeights[i + j * currentLayerSize];
            }
            errorGradient.push_back(gradientSum * activateDerivative(prevLayerGradients[i]));
        }
        return errorGradient;
    }

    double activateDerivative(double x) {
        switch (activationFunction) {
            case ActivationFunction::Sigmoid: {
                double sigmoid = activate(x);
                return sigmoid * (1.0 - sigmoid);
            }
            case ActivationFunction::ReLU:
                return x > 0.0 ? 1.0 : 0.0;
            case ActivationFunction::Tanh: {
                double tanh = std::tanh(x);
                return 1.0 - tanh * tanh;
            }
            default:
                return 1.0;
        }
    }

    void updateWeights(int layerIndex, const std::vector<double>& gradients, const std::vector<double>& inputs, double learningRate) {
        std::vector<double>& currentLayerWeights = weights[layerIndex];
        int currentLayerSize = currentLayerWeights.size() / inputs.size();

        for (int i = 0; i < currentLayerSize; ++i) {
            for (int j = 0; j < inputs.size(); ++j) {
                int weightIndex = i + j * currentLayerSize;
                currentLayerWeights[weightIndex] += learningRate * gradients[i] * inputs[j];
            }
        }

        biases[layerIndex] += learningRate * gradients[0];
    }
};

void saveNetworkConfig(const std::string& filePath, const std::vector<int>& layerSizes) {
    std::ofstream configFile(filePath);
    if (configFile.is_open()) {
        for (int i = 1; i < layerSizes.size(); ++i) {
            configFile << "layer_" << i << "_size=" << layerSizes[i] << std::endl;
        }
        configFile.close();
        std::cout << "Network configuration saved to " << filePath << std::endl;
    } else {
        std::cerr << "Unable to save network configuration file." << std::endl;
    }
}

void saveTrainingData(const std::string& filePath, const std::vector<std::vector<double>>& trainingData) {
    std::ofstream dataFile(filePath);
    if (dataFile.is_open()) {
        for (const auto& dataPoint : trainingData) {
            for (double value : dataPoint) {
                dataFile << value << "\t";
            }
            dataFile << std::endl;
        }
        dataFile.close();
        std::cout << "Training data saved to " << filePath << std::endl;
    } else {
        std::cerr << "Unable to save training data file." << std::endl;
    }
}

std::vector<std::vector<double>> loadTrainingData(const std::string& filePath) {
    std::vector<std::vector<double>> trainingData;
    std::ifstream dataFile(filePath);
    if (dataFile.is_open()) {
        std::string line;
        while (std::getline(dataFile, line)) {
            std::vector<double> dataPoint;
            std::istringstream iss(line);
            double value;
            while (iss >> value) {
                dataPoint.push_back(value);
            }
            trainingData.push_back(dataPoint);
        }
        dataFile.close();
        std::cout << "Training data loaded from " << filePath << std::endl;
    } else {
        std::cerr << "Unable to load training data file." << std::endl;
    }
    return trainingData;
}

int main() {
    std::vector<int> layerSizes = {2, 3, 1};
    std::string networkConfigFile = "network.config";
    std::string trainingDataFile = "training.data";
    std::string weightsFile = "weights.txt";
    std::vector<std::vector<double>> trainingData = {{0.5, 0.8, 0.9}};

    saveNetworkConfig(networkConfigFile, layerSizes);
    saveTrainingData(trainingDataFile, trainingData);

    FeedForwardNetwork network(layerSizes, ActivationFunction::Sigmoid);
    network.train(trainingData, 0.1, 1000);

    std::vector<double> input = {0.5, 0.8};
    std::vector<double> output = network.forward(input);

    std::cout << "Input: " << input[0] << ", " << input[1] << std::endl;
    std::cout << "Output: " << output[0] << std::endl;

    network.saveWeights(weightsFile);

    FeedForwardNetwork newNetwork(layerSizes, ActivationFunction::Sigmoid);
    newNetwork.loadWeights(weightsFile);

    std::vector<double> newOutput = newNetwork.forward(input);
    std::cout << "New Output: " << newOutput[0] << std::endl;

    std::vector<std::vector<double>> loadedTrainingData = loadTrainingData(trainingDataFile);

    return 0;
}

