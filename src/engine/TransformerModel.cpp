#include <vector>
#include <cmath>
#include <numeric>

class TransformerModel {
public:
    TransformerModel(int numLayers, int numHeads, int hiddenSize, int feedForwardSize)
        : transformerEncoderLayers_(numLayers), positionalEncoding_(1000, hiddenSize),
          input_(), output_() {}

    void initialize() {
        for (auto& layer : transformerEncoderLayers_) {
            layer.initialize();
        }
    }

    void forward() {
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

        output_ = layerNormalization_(std::move(positionalEncodedInput));
    }

    void setInput(const std::vector<float>& input) {
        input_ = input;
    }

    const std::vector<float>& getOutput() const {
        return output_;
    }

private:
    std::vector<float> embeddingLayer_(const std::vector<float>& input) {
        const int inputSize = input.size();
        const int embeddingSize = 512;

        std::vector<float> embeddedInput(inputSize * embeddingSize, 0.0f);

        int index = 0;
        for (float inputValue : input) {
            for (int j = 0; j < embeddingSize; ++j) {
                float embeddingValue = inputValue * j;
                embeddedInput[index++] = embeddingValue;
            }
        }

        return embeddedInput;
    }

    std::vector<float> applyPositionalEncoding(const std::vector<float>& input) {
        const int inputSize = input.size();
        std::vector<float> output(inputSize);

        int index = 0;
        for (float inputValue : input) {
            std::vector<float> positionalEncoding = positionalEncoding_(index++);
            output[index] = inputValue + positionalEncoding[index];
        }

        return output;
    }

    std::vector<float> layerNormalization_(std::vector<float>&& input) {
        const int inputSize = input.size();
        const float epsilon = 1e-6;

        std::vector<float> output(inputSize);

        float mean = std::accumulate(input.begin(), input.end(), 0.0f) / inputSize;
        float variance = 0.0f;
        for (float value : input) {
            float diff = value - mean;
            variance += diff * diff;
        }
        variance /= inputSize;

        float stdDev = std::sqrt(variance);

        for (int i = 0; i < inputSize; ++i) {
            float normalizedValue = (input[i] - mean) / (stdDev + epsilon);
            output[i] = normalizedValue;
        }

        return output;
    }

private:
    std::vector<TransformerEncoderLayer> transformerEncoderLayers_;
    PositionalEncoding positionalEncoding_;
    std::vector<float> input_;
    std::vector<float> output_;
};

