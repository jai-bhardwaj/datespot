#ifndef TRANSFORMER_MODEL_H
#define TRANSFORMER_MODEL_H

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "Masking.h"
#include "TransformerEncoderLayer.h"
#include "PositionalEncoding.h"
#include "FeedForwardNetworkLayer.h"

class TransformerEncoderLayer {
    // Add the declaration for the TransformerEncoderLayer class here
};

enum class PositionalEncodingMethod {
    METHOD_A,
    METHOD_B
};

class PositionalEncoding {
    // Add the declaration for the PositionalEncoding class here
};

class Masking {
public:
    static std::vector<std::vector<float>> createPaddingMask(const std::vector<float>& input) {
        // Implementation of createPaddingMask function
    }
};

class TransformerModel {
public:
    TransformerModel(int numLayers, int numHeads, int hiddenSize, int feedForwardSize);

    void initialize();
    void forward(const std::vector<float>& input);
    void setInput(const std::vector<float>& input);
    const std::vector<float>& getOutput() const;
    int getNumLayers() const;
    void setLayerParameters(int layerIndex, int numHeads, int hiddenSize, int feedForwardSize);
    void setAllLayerParameters(int numHeads, int hiddenSize, int feedForwardSize);
    void reset();
    void printOutput() const;
    void printLayerOutputs() const;
    void printModelInfo() const;
    void addLayer(const TransformerEncoderLayer& layer);
    void removeLayer(int layerIndex);
    void clearLayers();
    void setPosEncodingMethod(const PositionalEncodingMethod& method);
    std::vector<std::vector<float>> getAttentionWeights(int layerIndex) const;
    void printAttentionWeights(int layerIndex) const;
    void enableLayerNormalization(bool enable);
    void enableDropout(bool enable, float dropoutRate = 0.1);
    void setAttentionMask(const std::vector<std::vector<float>>& attentionMask);
    void setMaskedAttention(bool maskedAttention);
    void setOutputProjectionSize(int outputSize);
    void setFeedForwardActivation(const std::function<float(float)>& activationFunction);
    void setLayerNormalizationParameters(float epsilon, bool trackGlobalStats);
    void enableMultiHeadAttention(bool enable);
    void setMultiHeadAttentionParameters(int numHeads, int headSize, float attentionDropoutRate);
    void enablePositionwiseFeedForward(bool enable);
    void setPositionwiseFeedForwardParameters(int feedForwardSize, float dropoutRate);
    void setResidualConnection(bool enable);
    void setResidualConnectionDropoutRate(float dropoutRate);
    std::vector<float> generateSequence(int maxLength);
    void saveModel(const std::string& filename);
    void loadModel(const std::string& filename);

private:
    std::vector<float> embeddingLayer(const std::vector<float>& input);
    std::vector<float> applyPositionalEncoding(const std::vector<float>& input);
    std::vector<float> layerNormalization(std::vector<float>&& input);

private:
    std::vector<TransformerEncoderLayer> transformerEncoderLayers_;
    PositionalEncoding positionalEncoding_;
    std::vector<float> input_;
    std::vector<float> output_;
};

#endif  // TRANSFORMER_MODEL_H

