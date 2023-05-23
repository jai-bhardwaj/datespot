#include <cmath>
#include <random>
#include <string>
#include <chrono>
#include <iostream>

#include <Eigen/Dense>

class FeedForwardNetworkLayer {
private:
    std::string name;
    Eigen::MatrixXf weights;
    Eigen::VectorXf biases;
    Eigen::VectorXf output;
    std::mt19937 randomEngine;

public:
    FeedForwardNetworkLayer(const std::string& name);
    void initialize();
    void forward(const Eigen::VectorXf& input);
    void printOutput() const;

private:
    Eigen::VectorXf linearTransform(const Eigen::VectorXf& input) const;
    Eigen::VectorXf relu(const Eigen::VectorXf& input) const;
    void initializeRandomMatrix(float mean, float stddev);
    void initializeRandomVector(float mean, float stddev);
};

FeedForwardNetworkLayer::FeedForwardNetworkLayer(const std::string& name) : name(name), randomEngine(std::random_device{}()) {
}

void FeedForwardNetworkLayer::initialize() {
    initializeRandomMatrix(0.0f, 0.1f);
    initializeRandomVector(0.0f, 0.1f);
}

void FeedForwardNetworkLayer::forward(const Eigen::VectorXf& input) {
    Eigen::VectorXf transformed = linearTransform(input);
    Eigen::VectorXf activated = relu(transformed);
    output = activated;
}

Eigen::VectorXf FeedForwardNetworkLayer::linearTransform(const Eigen::VectorXf& input) const {
    return weights * input + biases;
}

Eigen::VectorXf FeedForwardNetworkLayer::relu(const Eigen::VectorXf& input) const {
    return input.cwiseMax(0.0f);
}

void FeedForwardNetworkLayer::initializeRandomMatrix(float mean, float stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> distribution(mean, stddev);

    weights = Eigen::MatrixXf::NullaryExpr(weights.rows(), weights.cols(), [&]() { return distribution(gen); });
}

void FeedForwardNetworkLayer::initializeRandomVector(float mean, float stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> distribution(mean, stddev);

    biases = Eigen::VectorXf::NullaryExpr(biases.size(), [&]() { return distribution(gen); });
}

void FeedForwardNetworkLayer::printOutput() const {
    std::cout << "Layer: " << name << "\n";
    std::cout << "Output:\n" << output << "\n";
}