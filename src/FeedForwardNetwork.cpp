#include <random>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include "FeedForwardNetwork.h"

FeedForwardNetwork::FeedForwardNetwork(int num_layers, const std::vector<int>& layer_sizes)
    : num_layers(num_layers), layer_sizes(layer_sizes)
{
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);

    weights.resize(num_layers - 1);
    biases.resize(num_layers - 1);

    for (int i = 0; i < num_layers - 1; ++i) {
        int input_size = layer_sizes[i];
        int output_size = layer_sizes[i + 1];

        weights[i].resize(input_size, std::vector<float>(output_size));
        biases[i].resize(output_size);

        float xavier_init = std::sqrt(2.0f / (input_size + output_size));

        for (auto& weight_row : weights[i]) {
            for (auto& weight : weight_row) {
                weight = dist(gen) * xavier_init;
            }
        }

        for (auto& bias : biases[i]) {
            bias = dist(gen) * xavier_init;
        }
    }
}

void FeedForwardNetwork::forward(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& output) {
    for (int layer = 0; layer < num_layers - 1; ++layer) {
        const auto& layer_weights = weights[layer];
        const auto& layer_biases = biases[layer];

        int input_size = layer_weights.size();
        int output_size = layer_weights[0].size();

        output[layer].resize(output_size);

        for (int j = 0; j < output_size; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < input_size; ++i) {
                sum += input[layer][i] * layer_weights[i][j];
            }
            output[layer][j] = activationFunction(sum + layer_biases[j]);
        }
    }
}

float FeedForwardNetwork::activationFunction(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float FeedForwardNetwork::mseLoss(const std::vector<std::vector<float>>& output, const std::vector<std::vector<float>>& target) {
    float mse = 0.0;
    for (const auto& output_row : output) {
        for (float value : output_row) {
            mse += value * value;
        }
    }
    return mse / (2.0 * output.size() * output[0].size());
}

float FeedForwardNetwork::crossEntropyLoss(const std::vector<std::vector<float>>& output, const std::vector<std::vector<float>>& target) {
    float crossEntropy = 0.0;
    for (int i = 0; i < output.size(); ++i) {
        for (int j = 0; j < output[i].size(); ++j) {
            crossEntropy -= target[i][j] * std::log(output[i][j] + 1e-7f) + (1 - target[i][j]) * std::log(1 - output[i][j] + 1e-7f);
        }
    }
    return crossEntropy / output.size();
}

void FeedForwardNetwork::train(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& target, float learning_rate, const std::string& optimizer, float momentum, float decay_rate, float epsilon, float beta1, float beta2) {
    std::vector<std::vector<float>> output(num_layers - 1);
    output[0] = input;

    forward(input, output);

    std::vector<std::vector<float>> updatedGrads(num_layers - 1);
    backward(input, target, updatedGrads);

    int t = 0;
    for (int layer = 0; layer < num_layers - 1; ++layer) {
        auto& layer_weights = weights[layer];
        auto& layer_biases = biases[layer];
        const auto& layer_grads = updatedGrads[layer];

        int input_size = layer_weights.size();
        int output_size = layer_weights[0].size();

        for (int j = 0; j < output_size; ++j) {
            for (int i = 0; i < input_size; ++i) {
                if (optimizer == "SGD") {
                    layer_weights[i][j] -= learning_rate * layer_grads[i][j];
                }
                else if (optimizer == "NAG") {
                    float prev_weight = layer_weights[i][j];
                    layer_weights[i][j] -= momentum * learning_rate * prev_weight + learning_rate * layer_grads[i][j];
                    layer_weights[i][j] -= momentum * learning_rate * (prev_weight - layer_weights[i][j]);
                }
                else if (optimizer == "RMSProp") {
                    if (rmsprop_cache[layer].empty()) {
                        rmsprop_cache[layer].resize(input_size, std::vector<float>(output_size, 0.0f));
                    }

                    rmsprop_cache[layer][i][j] = decay_rate * rmsprop_cache[layer][i][j] + (1 - decay_rate) * layer_grads[i][j] * layer_grads[i][j];
                    layer_weights[i][j] -= learning_rate * (layer_grads[i][j] / (std::sqrt(rmsprop_cache[layer][i][j]) + epsilon));
                }
                else if (optimizer == "Adagrad") {
                    if (adagrad_cache[layer].empty()) {
                        adagrad_cache[layer].resize(input_size, std::vector<float>(output_size, 0.0f));
                    }

                    adagrad_cache[layer][i][j] += layer_grads[i][j] * layer_grads[i][j];
                    layer_weights[i][j] -= learning_rate * (layer_grads[i][j] / (std::sqrt(adagrad_cache[layer][i][j]) + epsilon));
                }
                else if (optimizer == "Adam") {
                    if (adam_m[layer].empty()) {
                        adam_m[layer].resize(input_size, std::vector<float>(output_size, 0.0f));
                        adam_v[layer].resize(input_size, std::vector<float>(output_size, 0.0f));
                    }

                    adam_m[layer][i][j] = beta1 * adam_m[layer][i][j] + (1 - beta1) * layer_grads[i][j];
                    adam_v[layer][i][j] = beta2 * adam_v[layer][i][j] + (1 - beta2) * layer_grads[i][j] * layer_grads[i][j];

                    float m_hat = adam_m[layer][i][j] / (1 - std::pow(beta1, t));
                    float v_hat = adam_v[layer][i][j] / (1 - std::pow(beta2, t));

                    layer_weights[i][j] -= learning_rate * (m_hat / (std::sqrt(v_hat) + epsilon));
                }
                else if (optimizer == "AdaDelta") {
                    if (rmsprop_cache[layer].empty()) {
                        rmsprop_cache[layer].resize(input_size, std::vector<float>(output_size, 0.0f));
                        adadelta_cache[layer].resize(input_size, std::vector<float>(output_size, 0.0f));
                    }

                    rmsprop_cache[layer][i][j] = decay_rate * rmsprop_cache[layer][i][j] + (1 - decay_rate) * layer_grads[i][j] * layer_grads[i][j];
                    float delta_weights = -std::sqrt(adadelta_cache[layer][i][j] + epsilon) / std::sqrt(rmsprop_cache[layer][i][j] + epsilon) * layer_grads[i][j];
                    layer_weights[i][j] += delta_weights;
                    adadelta_cache[layer][i][j] = decay_rate * adadelta_cache[layer][i][j] + (1 - decay_rate) * delta_weights * delta_weights;
                }
            }
            layer_biases[j] -= learning_rate * layer_grads[j][0];
        }
    }
}

void FeedForwardNetwork::backward(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& target, std::vector<std::vector<float>>& updatedGrads) {
    for (int layer = 0; layer < num_layers - 1; ++layer) {
        const auto& layer_weights = weights[layer];
        int input_size = layer_weights.size();
        int output_size = layer_weights[0].size();
        updatedGrads[layer].resize(input_size, std::vector<float>(output_size, 0.0f));
    }

    int last_layer = num_layers - 2;
    for (int j = 0; j < layer_sizes[last_layer + 1]; ++j) {
        float output = output[last_layer][j];
        float error = (output - target[last_layer][j]) * output * (1.0f - output);
        for (int i = 0; i < layer_sizes[last_layer]; ++i) {
            updatedGrads[last_layer][i][j] = error * output[last_layer - 1][i];
        }
        updatedGrads[last_layer][layer_sizes[last_layer]][j] = error;
    }

    for (int layer = num_layers - 3; layer >= 0; --layer) {
        const auto& layer_weights = weights[layer];
        int input_size = layer_weights.size();
        int output_size = layer_weights[0].size();

        for (int i = 0; i < input_size; ++i) {
            float output = output[layer][i];
            float sum = 0.0f;
            for (int j = 0; j < output_size; ++j) {
                sum += layer_weights[i][j] * updatedGrads[layer + 1][i][j];
            }
            float error = sum * output * (1.0f - output);
            for (int j = 0; j < output_size; ++j) {
                updatedGrads[layer][i][j] = error * output[layer - 1][i];
            }
            updatedGrads[layer][input_size][i] = error;
        }
    }
}
