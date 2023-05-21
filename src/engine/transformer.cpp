#include <iostream>
#include <vector>
#include <cmath>

constexpr int kNumLayers = 6;
constexpr int kNumHeads = 8;
constexpr int kHiddenSize = 512;
constexpr int kFeedForwardSize = 2048;

class TransformerEncoderLayer {
public:
    std::vector<float> operator()(const std::vector<float>& input) {
        std::vector<float> self_attention_output = selfAttention(input);
        std::vector<float> add_norm_output = addAndNormalize(input, self_attention_output);
        std::vector<float> feed_forward_output = feedForward(add_norm_output);
        std::vector<float> encoder_output = addAndNormalize(add_norm_output, feed_forward_output);
        return encoder_output;
    }

private:
    std::vector<float> selfAttention(const std::vector<float>& input) {
        std::vector<float> attention_output;
        return attention_output;
    }

    std::vector<float> addAndNormalize(const std::vector<float>& input, const std::vector<float>& output) {
        std::vector<float> normalized_output;
        return normalized_output;
    }

    std::vector<float> feedForward(const std::vector<float>& input) {
        std::vector<float> ff_output;
        return ff_output;
    }
};

class TransformerModel {
public:
    TransformerModel() {
        for (int i = 0; i < kNumLayers; i++) {
            transformerEncoderLayers_.push_back(TransformerEncoderLayer());
        }
    }

    std::vector<float> operator()(const std::vector<float>& input) {
        std::vector<float> encoder_output = input;

        for (int i = 0; i < kNumLayers; i++) {
            encoder_output = transformerEncoderLayers_[i](encoder_output);
        }

        return encoder_output;
    }

    void printOutput(const std::vector<float>& output) {
        for (float val : output) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    std::vector<float> addVectors(const std::vector<float>& vec1, const std::vector<float>& vec2) {
        std::vector<float> result;

        if (vec1.size() == vec2.size()) {
            result.reserve(vec1.size());
            for (std::size_t i = 0; i < vec1.size(); ++i) {
                result.push_back(vec1[i] + vec2[i]);
            }
        } else {
            std::cout << "Error: Vector sizes do not match!" << std::endl;
        }

        return result;
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

