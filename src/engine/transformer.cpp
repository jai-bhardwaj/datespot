#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <stdexcept>
#include "Layer.h"
#include "Neuron.h"
#include <limits>
#include <string>

constexpr int kNumLayers = 6;
constexpr int kNumHeads = 8;
constexpr int kHiddenSize = 512;
constexpr int kFeedForwardSize = 2048;

class Layer {
protected:
    std::vector<float> input_;
    std::vector<float> output_;

public:
    virtual ~Layer() {}

    void setInput(const std::vector<float>& input) {
        input_ = input;
    }

    const std::vector<float>& getOutput() const {
        return output_;
    }

    virtual void initialize() = 0;
    virtual void forward() = 0;
};

class Neuron : public Layer {
protected:
    std::vector<float> weights_;
    float bias_;

public:
    Neuron(const std::string& name) {}

    void initialize() override {
    }
};

class LinearNeuron : public Neuron {
public:
    LinearNeuron(const std::string& name) : Neuron(name) {}

    void forward() override {
        const int inputSize = input_.size();
        output_.resize(inputSize);

        for (int i = 0; i < inputSize; ++i) {
            output_[i] = input_[i] * weights_[i] + bias_;
        }
    }
};

class ActivationNeuron : public Neuron {
public:
    ActivationNeuron(const std::string& name) : Neuron(name) {}

    void forward() override {
        const int inputSize = input_.size();
        output_.resize(inputSize);

        for (int i = 0; i < inputSize; ++i) {
            output_[i] = activate(input_[i]);
        }
    }

private:
    float activate(float x) {
        return std::max(0.0f, x);
    }
};

class LayerNormalization : public Layer {
private:
    std::vector<float> scale_;
    std::vector<float> bias_;

public:
    LayerNormalization(const std::string& name) {}

    void initialize() override {
    }

    void forward() override {
        const int inputSize = input_.size();
        output_.resize(inputSize);

        for (int i = 0; i < inputSize; ++i) {
            output_[i] = (input_[i] - mean()) / stdDev() * scale_[i] + bias_[i];
        }
    }

private:
    float mean() {
        // Calculate the mean of input_
        float sum = 0.0f;
        for (float val : input_) {
            sum += val;
        }
        return sum / input_.size();
    }

    float stdDev() {
        // Calculate the standard deviation of input_
        float meanVal = mean();
        float sumSquares = 0.0f;
        for (float val : input_) {
            float diff = val - meanVal;
            sumSquares += diff * diff;
        }
        return std::sqrt(sumSquares / input_.size());
    }
};

class MultiHeadAttentionLayer : public Layer {
private:
    std::vector<std::vector<float>> weightQueries_;
    std::vector<std::vector<float>> weightKeys_;
    std::vector<std::vector<float>> weightValues_;
    std::vector<float> biasQueries_;
    std::vector<float> biasKeys_;
    std::vector<float> biasValues_;

public:
    MultiHeadAttentionLayer(const std::string& name) {}

    void initialize() override {
    }

    void forward() override {
        const std::vector<float>& input = input_;

        const int inputSize = input.size();
        const int numHeads = kNumHeads;
        const int headSize = kHiddenSize / numHeads;

        std::vector<float> attentionOutput(inputSize);
        std::vector<float> output;

        for (int head = 0; head < numHeads; ++head) {
            std::vector<float> queries = linearTransform(input, weightQueries_[head], biasQueries_);
            std::vector<float> keys = linearTransform(input, weightKeys_[head], biasKeys_);
            std::vector<float> values = linearTransform(input, weightValues_[head], biasValues_);

            std::vector<float> attentionScores = computeAttentionScores(queries, keys, headSize);
            std::vector<float> attentionWeights = maskedSoftmax(attentionScores, getAttentionMask(inputSize));
            std::vector<float> weightedSum = weightedSumValues(attentionWeights, values, headSize);

            attentionOutput = combineAttentionOutput(attentionOutput, weightedSum, head, numHeads);
        }

        output = linearTransform(attentionOutput, transpose(weightQueries_), transpose(biasQueries_));

        output_ = output;
    }

private:
    std::vector<float> linearTransform(const std::vector<float>& input, const std::vector<std::vector<float>>& weights, const std::vector<float>& biases) {
        const int inputSize = input.size();
        const int numHeads = weights.size();

        std::vector<float> output(inputSize);
        for (int i = 0; i < inputSize; ++i) {
            for (int head = 0; head < numHeads; ++head) {
                output[i] += input[i] * weights[head][i] + biases[head];
            }
        }
        return output;
    }

    std::vector<float> transpose(const std::vector<std::vector<float>>& matrix) {
        const int numRows = matrix.size();
        const int numCols = matrix[0].size();

        std::vector<float> transposed(numCols * numRows);
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
                transposed[j * numRows + i] = matrix[i][j];
            }
        }
        return transposed;
    }

    std::vector<float> computeAttentionScores(const std::vector<float>& queries, const std::vector<float>& keys, int headSize) {
        const int inputSize = queries.size();
        std::vector<float> attentionScores(inputSize, 0.0f);

        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                float score = 0.0f;
                for (int k = 0; k < headSize; ++k) {
                    score += queries[i * headSize + k] * keys[j * headSize + k];
                }
                attentionScores[i] += score / std::sqrt(headSize);
            }
        }

        return attentionScores;
    }

    std::vector<float> maskedSoftmax(const std::vector<float>& input, const std::vector<std::vector<float>>& mask) {
        std::vector<float> output = input;
        const int size = input.size();

        float maxVal = *std::max_element(input.begin(), input.end());
        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            if (mask[i][i] == 0.0f) {
                output[i] = -std::numeric_limits<float>::infinity();
            } else {
                output[i] = std::exp(output[i] - maxVal);
                sum += output[i];
            }
        }

        for (int i = 0; i < size; ++i) {
            if (mask[i][i] == 0.0f) {
                output[i] = 0.0f;
            } else {
                output[i] /= sum;
            }
        }
        return output;
    }

    std::vector<float> weightedSumValues(const std::vector<float>& attentionWeights, const std::vector<float>& values, int headSize) {
        const int inputSize = values.size();
        std::vector<float> output(inputSize, 0.0f);

        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                for (int k = 0; k < headSize; ++k) {
                    output[i] += attentionWeights[j] * values[j * headSize + k];
                }
            }
        }
        return output;
    }

    std::vector<float> combineAttentionOutput(const std::vector<float>& attentionOutput, const std::vector<float>& weightedSum, int head, int numHeads) {
        const int inputSize = attentionOutput.size();
        const int headSize = kHiddenSize / numHeads;
        std::vector<float> output(inputSize, 0.0f);

        for (int i = 0; i < inputSize; ++i) {
            output[i] = attentionOutput[i] + weightedSum[i] / std::sqrt(headSize);
        }
        return output;
    }

    std::vector<std::vector<float>> getAttentionMask(int inputSize) const {
        std::vector<std::vector<float>> attentionMask(inputSize, std::vector<float>(inputSize, 1.0f));

        return attentionMask;
    }
};

class PositionalEncoding {
private:
    std::vector<std::vector<float>> encoding_;

public:
    PositionalEncoding(int maxSeqLength, int hiddenSize) {
        encoding_ = generatePositionalEncoding(maxSeqLength, hiddenSize);
    }

    std::vector<float> operator()(int position) const {
        return encoding_[position];
    }

private:
    std::vector<std::vector<float>> generatePositionalEncoding(int maxSeqLength, int hiddenSize) {
        std::vector<std::vector<float>> encoding(maxSeqLength, std::vector<float>(hiddenSize));
        for (int pos = 0; pos < maxSeqLength; ++pos) {
            for (int i = 0; i < hiddenSize; ++i) {
                float angle = pos / std::pow(10000.0f, (2.0f * i) / hiddenSize);
                encoding[pos][i] = (pos % 2 == 0) ? std::sin(angle) : std::cos(angle);
            }
        }
        return encoding;
    }
};

class FeedForwardNetworkLayer : public Layer {
private:
    std::vector<std::vector<float>> weight1_;
    std::vector<float> bias1_;
    std::vector<std::vector<float>> weight2_;
    std::vector<float> bias2_;

public:
    FeedForwardNetworkLayer(const std::string& name) {}

    void initialize() override {
    }

    void forward() override {
        const std::vector<float>& input = input_;

        std::vector<float> output = linearTransform(input, weight1_, bias1_);
        output = relu(output);
        output = linearTransform(output, weight2_, bias2_);

        output_ = output;
    }

private:
    std::vector<float> linearTransform(const std::vector<float>& input, const std::vector<std::vector<float>>& weights, const std::vector<float>& biases) {
        const int inputSize = input.size();
        const int outputSize = weights[0].size();

        std::vector<float> output(outputSize);
        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                output[i] += input[j] * weights[j][i] + biases[i];
            }
        }
        return output;
    }

    std::vector<float> relu(const std::vector<float>& input) {
        std::vector<float> output = input;
        const int size = input.size();
        for (auto& val : output) {
            val = std::max(0.0f, val);
        }
        return output;
    }
};

class TransformerEncoderLayer : public Layer {
private:
    MultiHeadAttentionLayer multiHeadAttentionLayer_;
    FeedForwardNetworkLayer feedForwardNetworkLayer_;

public:
    TransformerEncoderLayer(const std::string& name) : multiHeadAttentionLayer_(name), feedForwardNetworkLayer_(name) {}

    void initialize() override {
        multiHeadAttentionLayer_.initialize();
        feedForwardNetworkLayer_.initialize();
    }

    void forward() override {
        const std::vector<float>& input = input_;

        std::vector<float> attentionOutput = multiHeadAttentionLayer_.getOutput();
        std::vector<float> attentionOutputNorm = layerNormalization_(attentionOutput);

        feedForwardNetworkLayer_.setInput(attentionOutputNorm);
        feedForwardNetworkLayer_.forward();
        std::vector<float> feedForwardOutput = feedForwardNetworkLayer_.getOutput();
        std::vector<float> feedForwardOutputNorm = layerNormalization_(feedForwardOutput);

        std::vector<float> output = ResidualConnection::add(attentionOutputNorm, feedForwardOutputNorm);

        output_ = output;
    }
};

class TransformerModel : public FeedForwardNetwork {
private:
    std::vector<TransformerEncoderLayer> transformerEncoderLayers_;
    PositionalEncoding positionalEncoding_;

public:
    TransformerModel(int numLayers, int numHeads, int hiddenSize, int feedForwardSize)
        : transformerEncoderLayers_(numLayers), positionalEncoding_(1000, hiddenSize) {}

    void initialize() override {
        for (auto& layer : transformerEncoderLayers_) {
            layer.initialize();
        }
    }

    void forward() override {
        const std::vector<float>& input = input_;

        std::vector<float> embeddedInput = embeddingLayer_(input);
        std::vector<float> positionalEncodedInput = applyPositionalEncoding(embeddedInput);

        std::vector<std::vector<float>> paddingMask = Masking::createPaddingMask(input);

        for (auto& layer : transformerEncoderLayers_) {
            layer.setInput(positionalEncodedInput);
            layer.setAttentionMask(paddingMask);
            layer.forward();
            positionalEncodedInput = layer.getOutput();
        }

        std::vector<float> output = layerNormalization_(positionalEncodedInput);

        output_ = output;
    }

private:
    std::vector<float> embeddingLayer_(const std::vector<float>& input) {
        // Perform embedding layer operations
        std::vector<float> embeddedInput;
        // ...
        return embeddedInput;
    }

    std::vector<float> applyPositionalEncoding(const std::vector<float>& input) {
        const int inputSize = input.size();
        std::vector<float> output(inputSize);

        for (int i = 0; i < inputSize; ++i) {
            std::vector<float> positionalEncoding = positionalEncoding_(i);
            output[i] = input[i] + positionalEncoding[i];
        }

        return output;
    }

    std::vector<float> layerNormalization_(const std::vector<float>& input) {
        // Perform layer normalization operations
        std::vector<float> output;
        // ...
        return output;
    }
};

class Masking {
public:
    static std::vector<std::vector<float>> createPaddingMask(const std::vector<float>& input) {
        std::vector<std::vector<float>> mask(input.size(), std::vector<float>(input.size(), 1.0f));

        for (int i = 0; i < input.size(); ++i) {
            if (input[i] == 0.0f) {
                for (int j = 0; j < input.size(); ++j) {
                    mask[i][j] = 0.0f;
                    mask[j][i] = 0.0f;
                }
            }
        }

        return mask;
    }
};
