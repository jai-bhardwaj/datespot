#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <array>
#include <stdexcept>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

constexpr int kNumLayers = 6;
constexpr int kNumHeads = 8;
constexpr int kHiddenSize = 512;
constexpr int kFeedForwardSize = 2048;

class MultiHeadAttention {
private:
    std::vector<std::vector<float>> weightQueries_;
    std::vector<std::vector<float>> weightKeys_;
    std::vector<std::vector<float>> weightValues_;
    std::vector<float> biasQueries_;
    std::vector<float> biasKeys_;
    std::vector<float> biasValues_;

public:
    MultiHeadAttention() {
        weightQueries_ = initializeWeights(kHiddenSize, kNumHeads, true);
        weightKeys_ = initializeWeights(kHiddenSize, kNumHeads, true);
        weightValues_ = initializeWeights(kHiddenSize, kNumHeads, true);
        biasQueries_ = initializeBias(kNumHeads, true);
        biasKeys_ = initializeBias(kNumHeads, true);
        biasValues_ = initializeBias(kNumHeads, true);
    }

    std::vector<float> operator()(const std::vector<float>& input, const std::vector<std::vector<float>>& attentionMask) {
        const int inputSize = input.size();
        const int numHeads = kNumHeads;
        const int headSize = kHiddenSize / numHeads;

        std::vector<float> attentionOutput(inputSize);
        std::vector<float> output;

        for (int head = 0; head < numHeads; ++head) {
            std::vector<float> queries = linearTransform(input, weightQueries_[head], biasQueries_[head]);
            std::vector<float> keys = linearTransform(input, weightKeys_[head], biasKeys_[head]);
            std::vector<float> values = linearTransform(input, weightValues_[head], biasValues_[head]);

            std::vector<float> attentionScores = computeAttentionScores(queries, keys, headSize, attentionMask);
            std::vector<float> attentionWeights = softmax(attentionScores);
            std::vector<float> weightedSum = weightedSumValues(attentionWeights, values, headSize);

            attentionOutput = combineAttentionOutput(attentionOutput, weightedSum, head, numHeads);
        }

        output = linearTransform(attentionOutput, transpose(weightQueries_), transpose(biasQueries_));
        return output;
    }

private:
    std::vector<std::vector<float>> initializeWeights(int inputSize, int numHeads, bool random) {
        std::vector<std::vector<float>> weights(numHeads, std::vector<float>(inputSize));
        if (random) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> distribution(0.0f, std::sqrt(2.0f / inputSize));

            for (auto& headWeights : weights) {
                for (auto& weight : headWeights) {
                    weight = distribution(gen);
                }
            }
        }
        return weights;
    }

    std::vector<float> initializeBias(int numHeads, bool random) {
        std::vector<float> bias(numHeads);
        if (random) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> distribution(0.0f, std::sqrt(2.0f / kHiddenSize));

            for (auto& b : bias) {
                b = distribution(gen);
            }
        }
        return bias;
    }

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

    std::vector<float> computeAttentionScores(const std::vector<float>& queries, const std::vector<float>& keys, int headSize, const std::vector<std::vector<float>>& attentionMask) {
        const int inputSize = queries.size();
        std::vector<float> attentionScores(inputSize, 0.0f);

        float* queriesDev;
        float* keysDev;
        float* attentionScoresDev;

        cudaMalloc((void**)&queriesDev, inputSize * headSize * sizeof(float));
        cudaMalloc((void**)&keysDev, inputSize * headSize * sizeof(float));
        cudaMalloc((void**)&attentionScoresDev, inputSize * sizeof(float));

        cudaMemcpy(queriesDev, queries.data(), inputSize * headSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(keysDev, keys.data(), inputSize * headSize * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockDim(16, 16);
        dim3 gridDim((inputSize + blockDim.x - 1) / blockDim.x, (inputSize + blockDim.y - 1) / blockDim.y);

        computeAttentionScoresKernel<<<gridDim, blockDim>>>(queriesDev, keysDev, attentionScoresDev, inputSize, headSize);

        cudaMemcpy(attentionScores.data(), attentionScoresDev, inputSize * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(queriesDev);
        cudaFree(keysDev);
        cudaFree(attentionScoresDev);

        return attentionScores;
    }

    __global__ void computeAttentionScoresKernel(const float* queries, const float* keys, float* attentionScores, int inputSize, int headSize) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;

        if (i < inputSize && j < inputSize) {
            float dotProduct = 0.0f;
            for (int k = 0; k < headSize; ++k) {
                dotProduct += queries[i * headSize + k] * keys[j * headSize + k];
            }
            attentionScores[i] += dotProduct * attentionMask[i][j];
        }
    }

    std::vector<float> softmax(const std::vector<float>& input) {
        std::vector<float> output = input;
        const int size = input.size();

        float maxVal = *std::max_element(input.begin(), input.end());
        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            output[i] = std::exp(output[i] - maxVal);
            sum += output[i];
        }

        for (int i = 0; i < size; ++i) {
            output[i] /= sum;
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

class FeedForwardNetwork {
private:
    std::vector<std::vector<float>> weight1_;
    std::vector<float> bias1_;
    std::vector<std::vector<float>> weight2_;
    std::vector<float> bias2_;

public:
    FeedForwardNetwork() {
        weight1_ = initializeWeights(kHiddenSize, kFeedForwardSize, true);
        bias1_ = initializeBias(kFeedForwardSize, true);
        weight2_ = initializeWeights(kFeedForwardSize, kHiddenSize, true);
        bias2_ = initializeBias(kHiddenSize, true);
    }

    std::vector<float> operator()(const std::vector<float>& input) {
        std::vector<float> output = linearTransform(input, weight1_, bias1_);
        output = relu(output);
        output = linearTransform(output, weight2_, bias2_);
        return output;
    }

private:
    std::vector<std::vector<float>> initializeWeights(int inputSize, int outputSize, bool random) {
        std::vector<std::vector<float>> weights(inputSize, std::vector<float>(outputSize));
        if (random) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> distribution(0.0f, std::sqrt(2.0f / inputSize));

            for (auto& inputWeights : weights) {
                for (auto& weight : inputWeights) {
                    weight = distribution(gen);
                }
            }
        }
        return weights;
    }

    std::vector<float> initializeBias(int size, bool random) {
        std::vector<float> bias(size);
        if (random) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> distribution(0.0f, std::sqrt(2.0f / size));

            for (auto& b : bias) {
                b = distribution(gen);
            }
        }
        return bias;
    }

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

class TransformerEncoderLayer {
private:
    MultiHeadAttention multiHeadAttention_;
    FeedForwardNetwork feedForwardNetwork_;

public:
    std::vector<float> operator()(const std::vector<float>& input, const std::vector<std::vector<float>>& attentionMask) {
        std::vector<float> attentionOutput = multiHeadAttention_(input, attentionMask);
        std::vector<float> output = feedForwardNetwork_(attentionOutput);
        return output;
    }
};

class TransformerModel {
private:
    std::vector<TransformerEncoderLayer> transformerEncoderLayers_;
    PositionalEncoding positionalEncoding_;

public:
    TransformerModel() : transformerEncoderLayers_(kNumLayers), positionalEncoding_(1000, kHiddenSize) {}

    std::vector<float> operator()(const std::vector<float>& input) {
        const int inputSize = input.size();
        std::vector<std::vector<float>> attentionMask(inputSize, std::vector<float>(inputSize, 1.0f));
        std::vector<float> encoderOutput = applyPositionalEncoding(input);

        for (auto& layer : transformerEncoderLayers_) {
            encoderOutput = layer(encoderOutput, attentionMask);
        }

        return encoderOutput;
    }

    void printOutput(const std::vector<float>& output) {
        for (const auto& val : output) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    std::vector<float> addVectors(const std::vector<float>& vec1, const std::vector<float>& vec2) {
        if (vec1.size() != vec2.size()) {
            throw std::runtime_error("Error: Vector sizes do not match!");
        }

        std::vector<float> result(vec1.size());
        std::transform(vec1.begin(), vec1.end(), vec2.begin(), result.begin(), std::plus<float>());
        return result;
    }

private:
    std::vector<float> applyPositionalEncoding(const std::vector<float>& input) {
        const int inputSize = input.size();
        std::vector<float> output(inputSize);

        for (int i = 0; i < inputSize; ++i) {
            std::vector<float> positionalEncoding = positionalEncoding_(i);
            output[i] = input[i] + positionalEncoding[i];
        }

        return output;
    }
};

int main() {
    std::vector<float> input = {1.0, 2.0, 3.0, 4.0};
    TransformerModel transformerModel;

    std::vector<float> output = transformerModel(input);
    transformerModel.printOutput(output);

    std::vector<float> input2 = {5.0, 6.0, 7.0, 8.0};
    std::vector<float> output2 = transformerModel(input2);
    transformerModel.printOutput(output2);

    std::vector<float> sum = transformerModel.addVectors(output, output2);
    transformerModel.printOutput(sum);

    return 0;
}

