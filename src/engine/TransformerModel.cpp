#include <iostream>
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

    void forward(const std::vector<float>& input) {
        setInput(input);

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

    int getNumLayers() const {
        return transformerEncoderLayers_.size();
    }

    void setLayerParameters(int layerIndex, int numHeads, int hiddenSize, int feedForwardSize) {
        if (layerIndex >= 0 && layerIndex < transformerEncoderLayers_.size()) {
            transformerEncoderLayers_[layerIndex].setParameters(numHeads, hiddenSize, feedForwardSize);
        }
    }

    void setAllLayerParameters(int numHeads, int hiddenSize, int feedForwardSize) {
        for (auto& layer : transformerEncoderLayers_) {
            layer.setParameters(numHeads, hiddenSize, feedForwardSize);
        }
    }

    void reset() {
        input_.clear();
        output_.clear();

        for (auto& layer : transformerEncoderLayers_) {
            layer.reset();
        }
    }

    void printOutput() const {
        std::cout << "Output: ";
        for (float value : output_) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    void printLayerOutputs() const {
        for (int i = 0; i < transformerEncoderLayers_.size(); ++i) {
            std::cout << "Layer " << i << " Output: ";
            const std::vector<float>& layerOutput = transformerEncoderLayers_[i].getOutput();
            for (float value : layerOutput) {
                std::cout << value << " ";
            }
            std::cout << std::endl;
        }
    }

    void printModelInfo() const {
        std::cout << "Transformer Model Info:" << std::endl;
        std::cout << "Number of Layers: " << transformerEncoderLayers_.size() << std::endl;
        std::cout << "Input Size: " << input_.size() << std::endl;
        std::cout << "Output Size: " << output_.size() << std::endl;
        std::cout << "Transformer Encoder Layer Parameters: " << std::endl;
        for (int i = 0; i < transformerEncoderLayers_.size(); ++i) {
            std::cout << "Layer " << i << ": ";
            transformerEncoderLayers_[i].printLayerInfo();
        }
    }

    void addLayer(const TransformerEncoderLayer& layer) {
        transformerEncoderLayers_.emplace_back(layer);
    }

    void removeLayer(int layerIndex) {
        if (layerIndex >= 0 && layerIndex < transformerEncoderLayers_.size()) {
            transformerEncoderLayers_.erase(transformerEncoderLayers_.begin() + layerIndex);
        }
    }

    void clearLayers() {
        transformerEncoderLayers_.clear();
    }

    void setPosEncodingMethod(const PositionalEncodingMethod& method) {
        positionalEncoding_.setMethod(method);
    }

    std::vector<std::vector<float>> getAttentionWeights(int layerIndex) const {
        std::vector<std::vector<float>> attentionWeights;
        if (layerIndex >= 0 && layerIndex < transformerEncoderLayers_.size()) {
            attentionWeights = transformerEncoderLayers_[layerIndex].getAttentionWeights();
        }
        return attentionWeights;
    }

    void printAttentionWeights(int layerIndex) const {
        std::cout << "Attention Weights for Layer " << layerIndex << ":" << std::endl;
        const std::vector<std::vector<float>>& attentionWeights = getAttentionWeights(layerIndex);
        for (const auto& weights : attentionWeights) {
            for (float weight : weights) {
                std::cout << weight << " ";
            }
            std::cout << std::endl;
        }
    }

    void enableLayerNormalization(bool enable) {
        for (auto& layer : transformerEncoderLayers_) {
            layer.enableLayerNormalization(enable);
        }
    }

    void enableDropout(bool enable, float dropoutRate = 0.1) {
        for (auto& layer : transformerEncoderLayers_) {
            layer.enableDropout(enable, dropoutRate);
        }
    }

    void setAttentionMask(const std::vector<std::vector<float>>& attentionMask) {
        for (auto& layer : transformerEncoderLayers_) {
            layer.setAttentionMask(attentionMask);
        }
    }

    void setMaskedAttention(bool maskedAttention) {
        for (auto& layer : transformerEncoderLayers_) {
            layer.setMaskedAttention(maskedAttention);
        }
    }

    void setOutputProjectionSize(int outputSize) {
        for (auto& layer : transformerEncoderLayers_) {
            layer.setOutputProjectionSize(outputSize);
        }
    }

    void setFeedForwardActivation(const std::function<float(float)>& activationFunction) {
        for (auto& layer : transformerEncoderLayers_) {
            layer.setFeedForwardActivation(activationFunction);
        }
    }

private:
    std::vector<float> embeddingLayer(const std::vector<float>& input) {
        const int embeddingSize = 512;

        std::vector<float> embeddedInput;
        embeddedInput.reserve(input.size() * embeddingSize);

        for (float inputValue : input) {
            for (int j = 0; j < embeddingSize; ++j) {
                embeddedInput.push_back(inputValue * j);
            }
        }

        return embeddedInput;
    }

    std::vector<float> applyPositionalEncoding(const std::vector<float>& input) {
        std::vector<float> output(input.size());
        const std::vector<float>& positionalEncoding = positionalEncoding_();

        std::transform(input.begin(), input.end(), positionalEncoding.begin(), output.begin(),
            [](float inputVal, float positionalEncodingVal) {
                return inputVal + positionalEncodingVal;
            });

        return output;
    }

    std::vector<float> layerNormalization(std::vector<float>&& input) {
        const float epsilon = 1e-6;

        float mean = std::reduce(std::execution::par, input.begin(), input.end()) / input.size();
        float variance = std::transform_reduce(std::execution::par, input.begin(), input.end(), 0.0f,
            [mean](float sum, float value) {
                float diff = value - mean;
                return sum + diff * diff;
            });

        variance /= input.size();
        float stdDev = std::sqrt(variance);

        std::transform(input.begin(), input.end(), input.begin(),
            [mean, stdDev, epsilon](float value) {
                float normalizedValue = (value - mean) / (stdDev + epsilon);
                return normalizedValue;
            });

        return std::move(input);
    }

private:
    std::vector<TransformerEncoderLayer> transformerEncoderLayers_;
    PositionalEncoding positionalEncoding_;
    std::vector<float> input_;
    std::vector<float> output_;
};



