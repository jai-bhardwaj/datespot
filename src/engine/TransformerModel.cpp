#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <ranges>
#include <Masking.h>
#include <TransformerEncoderLayer.h>
#include <PositionalEncoding.h>

class TransformerModel {
public:
    TransformerModel(int numLayers, int numHeads, int hiddenSize, int feedForwardSize)
        : transformerEncoderLayers_(numLayers),
          positionalEncoding_(1000, hiddenSize),
          input_(),
          output_()
    {
    }

    void initialize() {
        for (auto& layer : transformerEncoderLayers_) {
            layer.initialize();
        }
    }

    void forward() {
        std::vector<float> embeddedInput = embeddingLayer(input_);
        std::vector<float> positionalEncodedInput = applyPositionalEncoding(embeddedInput);

        std::vector<std::vector<float>> paddingMask = Masking::createPaddingMask(input_);

        for (auto& layer : transformerEncoderLayers_) {
            layer.setInput(std::move(positionalEncodedInput));
            layer.setAttentionMask(paddingMask);
            layer.forward();
            positionalEncodedInput = layer.getOutput();
        }

        output_ = layerNormalization(std::move(positionalEncodedInput));
    }

    void setInput(const std::vector<float>& input) {
        input_ = input;
    }

    const std::vector<float>& getOutput() const {
        return output_;
    }

private:
    std::vector<float> embeddingLayer(const std::vector<float>& input) {
        const int embeddingSize = 512;

        std::vector<float> embeddedInput;
        embeddedInput.reserve(input.size() * embeddingSize);

        for (float inputValue : input) {
            std::ranges::transform(std::views::iota(0), std::back_inserter(embeddedInput),
                [inputValue](int j) {
                    return inputValue * j;
                });
        }

        return embeddedInput;
    }

    std::vector<float> applyPositionalEncoding(const std::vector<float>& input) {
        std::vector<float> output(input.size());
        const std::vector<float>& positionalEncoding = positionalEncoding_();

        std::ranges::transform(input, positionalEncoding, output.begin(),
            [](float inputVal, float positionalEncodingVal) {
                return inputVal + positionalEncodingVal;
            });

        return output;
    }

    std::vector<float> layerNormalization(std::vector<float>&& input) {
        const float epsilon = 1e-6;

        std::vector<float> output(input.size());

        float mean = std::reduce(std::execution::par, input.begin(), input.end()) / input.size();
        float variance = std::transform_reduce(std::execution::par, input.begin(), input.end(), 0.0f,
            [mean](float sum, float value) {
                float diff = value - mean;
                return sum + diff * diff;
            });

        variance /= input.size();
        float stdDev = std::sqrt(variance);

        std::ranges::transform(input, output.begin(),
            [mean, stdDev, epsilon](float value) {
                float normalizedValue = (value - mean) / (stdDev + epsilon);
                return normalizedValue;
            });

        return output;
    }

private:
    std::vector<TransformerEncoderLayer> transformerEncoderLayers_;
    PositionalEncoding positionalEncoding_;
    std::vector<float> input_;
    std::vector<float> output_;
};
