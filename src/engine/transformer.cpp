#include <iostream>
#include <vector>
#include <cmath>

constexpr int kNumLayers = 6;
constexpr int kNumHeads = 8;
constexpr int kHiddenSize = 512;
constexpr int kFeedForwardSize = 2048;

class TransformerEncoderLayer {
public:
    TransformerEncoderLayer() {
    }

    std::vector<float> operator()(const std::vector<float>& input) {
        std::vector<float> self_attention_output = self_attention(input);
        
        std::vector<float> add_norm_output = add_and_normalize(input, self_attention_output);

        std::vector<float> feed_forward_output = feed_forward(add_norm_output);

        std::vector<float> encoder_output = add_and_normalize(add_norm_output, feed_forward_output);

        return encoder_output;
    }

private:
    std::vector<float> self_attention(const std::vector<float>& input) {
    }

    std::vector<float> add_and_normalize(const std::vector<float>& input, const std::vector<float>& output) {
    }

    std::vector<float> feed_forward(const std::vector<float>& input) {
    }
};

class TransformerModel {
public:
    TransformerModel() {
        for (int i = 0; i < kNumLayers; i++) {
            transformer_encoder_layers_.push_back(TransformerEncoderLayer());
        }
    }

    std::vector<float> operator()(const std::vector<float>& input) {
        std::vector<float> encoder_output = input;

        for (int i = 0; i < kNumLayers; i++) {
            encoder_output = transformer_encoder_layers_[i](encoder_output);
        }

        return encoder_output;
    }

private:
    std::vector<TransformerEncoderLayer> transformer_encoder_layers_;
};

int main() {
    std::vector<float> input = {1.0, 2.0, 3.0, 4.0};
    TransformerModel transformer_model;

    std::vector<float> output = transformer_model(input);

    for (float val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
