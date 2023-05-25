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
    gamma.resize(num_layers - 1);
    beta.resize(num_layers - 1);
    batch_mean.resize(num_layers - 1);
    batch_variance.resize(num_layers - 1);

    for (int i = 0; i < num_layers - 1; ++i) {
        int input_size = layer_sizes[i];
        int output_size = layer_sizes[i + 1];

        weights[i].resize(input_size, std::vector<float>(output_size));
        biases[i].resize(output_size);
        gamma[i].resize(output_size, 1.0f);
        beta[i].resize(output_size, 0.0f);

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
    const float epsilon = 1e-7f;

    for (int layer = 0; layer < num_layers - 1; ++layer) {
        const auto& layer_weights = weights[layer];
        const auto& layer_biases = biases[layer];
        const auto& layer_gamma = gamma[layer];
        const auto& layer_beta = beta[layer];

        int input_size = layer_weights.size();
        int output_size = layer_weights[0].size();

        output.push_back(std::vector<float>(output_size));

        std::vector<float> batch_normalized_output(output_size);
        batch_mean[layer] = std::vector<float>(output_size, 0.0f);
        batch_variance[layer] = std::vector<float>(output_size, 0.0f);

        for (int j = 0; j < output_size; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < input_size; ++i) {
                sum += input[layer][i] * layer_weights[i][j];
            }
            float mean = sum / input_size;
            float variance = 0.0f;
            for (int i = 0; i < input_size; ++i) {
                float diff = input[layer][i] * layer_weights[i][j] - mean;
                variance += diff * diff;
            }
            variance /= input_size;

            batch_mean[layer][j] = mean;
            batch_variance[layer][j] = variance;

            float normalized_input = (input[layer][j] * layer_weights[j][j] - mean) / std::sqrt(variance + epsilon);

            float scaled_input = normalized_input * layer_gamma[j] + layer_beta[j];

            batch_normalized_output[j] = scaled_input;
        }

        for (int j = 0; j < output_size; ++j) {
            output[layer][j] = activationFunction(batch_normalized_output[j]);
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
    for (size_t i = 0; i < output.size(); ++i) {
        for (size_t j = 0; j < output[i].size(); ++j) {
            crossEntropy -= target[i][j] * std::log(output[i][j] + epsilon) + (1 - target[i][j]) * std::log(1 - output[i][j] + epsilon);
        }
    }
    return crossEntropy / output.size();
}

void FeedForwardNetwork::train(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& target, float learning_rate, const std::string& optimizer, float momentum, float decay_rate, float epsilon, float beta1, float beta2, int max_epochs, int patience, float weight_decay)
{
    std::vector<std::vector<float>> output;
    output.push_back(input[0]); // Initialize output with input data

    std::vector<float> learning_rates = { learning_rate, 0.01f, 0.001f };
    std::vector<int> layer_sizes = { 128, 256, 512 };
    std::vector<std::string> optimizers = { "SGD", "RMSProp", "Adam" };

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<int> learning_rate_dist(0, learning_rates.size() - 1);
    std::uniform_int_distribution<int> layer_size_dist(0, layer_sizes.size() - 1);
    std::uniform_int_distribution<int> optimizer_dist(0, optimizers.size() - 1);

    int best_learning_rate_index = learning_rate_dist(gen);
    int best_layer_size_index = layer_size_dist(gen);
    int best_optimizer_index = optimizer_dist(gen);

    float best_validation_loss = std::numeric_limits<float>::max();
    int num_epochs_without_improvement = 0;

    std::vector<std::vector<std::vector<float>>> updatedGrads(num_layers - 1);

    for (int epoch = 0; epoch < max_epochs; ++epoch)
    {
        float sampled_learning_rate = learning_rates[best_learning_rate_index];
        int sampled_layer_size = layer_sizes[best_layer_size_index];
        std::string sampled_optimizer = optimizers[best_optimizer_index];

        // Forward pass
        forward(input, output);

        // Calculate loss
        float loss;
        if (sampled_optimizer == "SGD")
            loss = mseLoss(output, target);
        else
            loss = crossEntropyLoss(output, target);

        // Backward pass
        backward(output, target, updatedGrads);

        // Update weights and biases
        for (int layer = 0; layer < num_layers - 1; ++layer) {
            const auto& layer_weights = weights[layer];
            const auto& layer_biases = biases[layer];

            for (int i = 0; i < layer_weights.size(); ++i) {
                for (int j = 0; j < layer_weights[i].size(); ++j) {
                    if (sampled_optimizer == "SGD") {
                        // Stochastic Gradient Descent
                        layer_weights[i][j] -= sampled_learning_rate * updatedGrads[layer][i][j];
                    } else if (sampled_optimizer == "RMSProp") {
                        // RMSProp
                        float& rmsprop_cache_value = rmsprop_cache[layer][i][j];
                        rmsprop_cache_value = decay_rate * rmsprop_cache_value + (1 - decay_rate) * (updatedGrads[layer][i][j] * updatedGrads[layer][i][j]);
                        layer_weights[i][j] -= sampled_learning_rate * updatedGrads[layer][i][j] / (std::sqrt(rmsprop_cache_value) + epsilon);
                    } else if (sampled_optimizer == "Adam") {
                        // Adam
                        float& adam_m_value = adam_m[layer][i][j];
                        float& adam_v_value = adam_v[layer][i][j];
                        adam_m_value = beta1 * adam_m_value + (1 - beta1) * updatedGrads[layer][i][j];
                        adam_v_value = beta2 * adam_v_value + (1 - beta2) * (updatedGrads[layer][i][j] * updatedGrads[layer][i][j]);
                        float m_hat = adam_m_value / (1 - std::pow(beta1, epoch + 1));
                        float v_hat = adam_v_value / (1 - std::pow(beta2, epoch + 1));
                        layer_weights[i][j] -= sampled_learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
                    }
                }
            }

            for (int j = 0; j < layer_biases.size(); ++j) {
                if (sampled_optimizer == "SGD") {
                    layer_biases[j] -= sampled_learning_rate * updatedGrads[layer][layer_weights.size()][j];
                } else if (sampled_optimizer == "RMSProp") {
                    float& rmsprop_cache_value = rmsprop_cache[layer][layer_weights.size()][j];
                    rmsprop_cache_value = decay_rate * rmsprop_cache_value + (1 - decay_rate) * (updatedGrads[layer][layer_weights.size()][j] * updatedGrads[layer][layer_weights.size()][j]);
                    layer_biases[j] -= sampled_learning_rate * updatedGrads[layer][layer_weights.size()][j] / (std::sqrt(rmsprop_cache_value) + epsilon);
                } else if (sampled_optimizer == "Adam") {
                    float& adam_m_value = adam_m[layer][layer_weights.size()][j];
                    float& adam_v_value = adam_v[layer][layer_weights.size()][j];
                    adam_m_value = beta1 * adam_m_value + (1 - beta1) * updatedGrads[layer][layer_weights.size()][j];
                    adam_v_value = beta2 * adam_v_value + (1 - beta2) * (updatedGrads[layer][layer_weights.size()][j] * updatedGrads[layer][layer_weights.size()][j]);
                    float m_hat = adam_m_value / (1 - std::pow(beta1, epoch + 1));
                    float v_hat = adam_v_value / (1 - std::pow(beta2, epoch + 1));
                    layer_biases[j] -= sampled_learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
                }
            }
        }

        // Validate the network
        std::vector<std::vector<float>> validation_output;
        forward(input, validation_output);
        float validation_loss;
        if (sampled_optimizer == "SGD")
            validation_loss = mseLoss(validation_output, target);
        else
            validation_loss = crossEntropyLoss(validation_output, target);

        if (validation_loss < best_validation_loss) {
            best_validation_loss = validation_loss;
            num_epochs_without_improvement = 0;
        } else {
            ++num_epochs_without_improvement;
            if (num_epochs_without_improvement >= patience) {
                std::cout << "Early stopping triggered. No improvement in validation loss for " << patience << " epochs." << std::endl;
                break;
            }
        }

        best_learning_rate_index = learning_rate_dist(gen);
        best_layer_size_index = layer_size_dist(gen);
        best_optimizer_index = optimizer_dist(gen);
    }

    std::cout << "Training completed. Best validation loss: " << best_validation_loss << std::endl;
}

void FeedForwardNetwork::backward(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& target, std::vector<std::vector<std::vector<float>>>& updatedGrads) {
    updatedGrads.resize(num_layers - 1);

    int last_layer = num_layers - 2;
    for (int j = 0; j < layer_sizes[last_layer + 1]; ++j) {
        float output = input[last_layer][j];
        float error = (output - target[last_layer][j]) * output * (1.0f - output);
        for (int i = 0; i < layer_sizes[last_layer]; ++i) {
            updatedGrads[last_layer][i][j] = error * input[last_layer - 1][i];
        }
        updatedGrads[last_layer][layer_sizes[last_layer]][j] = error;
    }

    for (int layer = num_layers - 3; layer >= 0; --layer) {
        const auto& layer_weights = weights[layer];
        int input_size = layer_weights.size();
        int output_size = layer_weights[0].size();

        for (int i = 0; i < input_size; ++i) {
            float output = input[layer][i];
            float sum = 0.0f;
            for (int j = 0; j < output_size; ++j) {
                sum += layer_weights[i][j] * updatedGrads[layer + 1][i][j];
            }
            float error = sum * output * (1.0f - output);
            for (int j = 0; j < output_size; ++j) {
                updatedGrads[layer][i][j] = error * input[layer - 1][i];
            }
            updatedGrads[layer][input_size][i] = error;
        }
    }
}
