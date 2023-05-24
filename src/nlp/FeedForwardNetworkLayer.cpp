#include <cmath>
#include <random>
#include <string>
#include <chrono>
#include <iostream>
#include <vector>


class FeedForwardNetworkLayer {
private:
    std::string name;
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
    std::vector<float> output;
    std::mt19937 randomEngine;

public:
    FeedForwardNetworkLayer(const std::string& name) : name(name), randomEngine(std::random_device{}()) {
    }

    void initialize() {
        initializeRandomMatrix(0.0f, 0.1f);
        initializeRandomVector(0.0f, 0.1f);
    }

    void forward(const std::vector<float>& input) {
        std::transform(input.begin(), input.end(), weights.begin(), output.begin(), std::plus<float>());
        std::transform(output.begin(), output.end(), biases.begin(), output.begin(), std::plus<float>());
        std::transform(output.begin(), output.end(), output.begin(), [](float x) { return std::max(0.0f, x); });
    }

    void printOutput() const {
        std::cout << "Layer: " << name << "\n";
        std::cout << "Output:\n";
        std::copy(output.begin(), output.end(), std::ostream_iterator<float>(std::cout, " "));
        std::cout << "\n";
    }

private:
    void initializeRandomMatrix(float mean, float stddev) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distribution(mean, stddev);

        weights.resize(output.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i].resize(weights.size());
            for (size_t j = 0; j < weights[i].size(); ++j) {
                weights[i][j] = distribution(gen);
            }
        }
    }

    void initializeRandomVector(float mean, float stddev) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distribution(mean, stddev);

        biases.resize(output.size());
        for (size_t i = 0; i < biases.size(); ++i) {
            biases[i] = distribution(gen);
        }
    }
};
