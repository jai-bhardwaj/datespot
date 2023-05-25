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
    std::mt19937_64 gen(std::random_device{});
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

        weights[i] = std::vector(input_size, std::vector<float>(output_size));
        biases[i] = std::vector(output_size);
        gamma[i] = std::vector(output_size, 1.0f);
        beta[i] = std::vector(output_size, 0.0f);

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
        const auto& layer_gamma = gamma[layer];
        const auto& layer_beta = beta[layer];

        int input_size = layer_weights.size();
        int output_size = layer_weights[0].size();

        output[layer].resize(output_size);

        std::vector<float> batch_normalized_output(output_size);
        batch_mean[layer] = std::vector(output_size, 0.0f);
        batch_variance[layer] = std::vector(output_size, 0.0f);

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
    for (const auto& output_row : output) {
        for (const auto& value : output_row) {
            crossEntropy -= target[i][j] * std::log(value + 1e-7f) + (1 - target[i][j]) * std::log(1 - value + 1e-7f);
        }
    }
    return crossEntropy / output.size();
}

void FeedForwardNetwork::train(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& target, float learning_rate, const std::string& optimizer, float momentum, float decay_rate, float epsilon, float beta1, float beta2, int max_epochs, int patience, float weight_decay)
{
    std::vector<std::vector<float>> output(num_layers - 1);
    output[0] = input;

    std::vector<float> learning_rates = { learning_rate, 0.01f, 0.001f };
    std::vector<int> layer_sizes = { 128, 256, 512 };
    std::vector<std::string> optimizers = { "SGD", "RMSProp", "Adam" };

    std::random_device rd;
    std::mt19937 gen(std::random_device{});

    std::uniform_int_distribution<int> learning_rate_dist(0, learning_rates.size() - 1);
    std::uniform_int_distribution<int> layer_size_dist(0, layer_sizes.size() - 1);
    std::uniform_int_distribution<int> optimizer_dist(0, optimizers.size() - 1);

    int best_learning_rate_index = learning_rate_dist(gen);
    int best_layer_size_index = layer_size_dist(gen);
    int best_optimizer_index = optimizer_dist(gen);

    float best_validation_loss = std::numeric_limits<float>::max();
    int num_epochs_without_improvement = 0;

    std::vector<std::vector<float>> rmsprop_cache(num_layers - 1);
    std::vector<std::vector<float>> adagrad_cache(num_layers - 1);
    std::vector<std::vector<float>> adam_m(num_layers - 1);
    std::vector<std::vector<float>> adam_v(num_layers - 1);

    for (int epoch = 0; epoch < max_epochs; ++epoch)
    {
        float sampled_learning_rate = learning_rates[best_learning_rate_index];
        int sampled_layer_size = layer_sizes[best_layer_size_index];
        std::string sampled_optimizer = optimizers[best_optimizer_index];

        int input_size = layer_weights.size();
        int output_size = layer_weights[0].size();

        for (int j = 0; j < output_size; ++j)
        {
            for (int i = 0; i < input_size; ++i)
            {
                float weight_decay_term = weight_decay * layer_weights[i][j];
                float weight_grad = layer_grads[i][j] + weight_decay_term;

        std::vector<std::vector<float>> validation_output(num_layers - 1);
        forward(validation_input, validation_output);
        float validation_loss = crossEntropyLoss(validation_output, validation_target);

        if (validation_loss < best_validation_loss)
        {
            best_validation_loss = validation_loss;
            num_epochs_without_improvement = 0;
        }
        else
        {
            ++num_epochs_without_improvement;
            if (num_epochs_without_improvement >= patience)
            {
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

void FeedForwardNetwork::backward(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& target, std::vector<std::vector<float>>& updatedGrads) {
    for (int layer = 0; layer < num_layers - 1; ++layer) {
        const auto& layer_weights = weights[layer];
        int input_size = layer_weights.size();
        int output_size = layer_weights[0].size();
        updatedGrads[layer] = std::vector(input_size, std::vector<float>(output_size, 0.0f));
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
