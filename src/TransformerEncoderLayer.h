#ifndef TRANSFORMER_ENCODER_LAYER_H
#define TRANSFORMER_ENCODER_LAYER_H

#include "MultiHeadAttentionLayer.h"
#include "FeedForwardNetworkLayer.h"
#include "PositionalEncoding.h"
#include "Attention.h"
#include "LayerNorm.h"
#include <string>
#include <vector>
#include <functional>

class TransformerEncoderLayer {
public:
    TransformerEncoderLayer();

    void initialize();
    void setInput(std::vector<float>&& input);
    void setAttentionMask(const std::vector<std::vector<float>>& attentionMask);
    void forward();
    const std::vector<float>& getOutput() const;
    void setParameters(int numHeads, int hiddenSize, int feedForwardSize);
    void reset();
    void printLayerInfo() const;
    void enableLayerNormalization(bool enable);
    void enableDropout(bool enable, float dropoutRate = 0.1);
    void setOutputProjectionSize(int outputSize);
    void setFeedForwardActivation(const std::function<float(float)>& activationFunction);
    void setLayerNormalizationParameters(float epsilon, bool trackGlobalStats);
    void enableMultiHeadAttention(bool enable);
    void setMultiHeadAttentionParameters(int numHeads, int headSize, float attentionDropoutRate);
    void enablePositionwiseFeedForward(bool enable);
    void setPositionwiseFeedForwardParameters(int feedForwardSize, float dropoutRate);
    void setResidualConnection(bool enable);
    void setResidualConnectionDropoutRate(float dropoutRate);
    void forward(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& mask);
    void backward(const std::vector<std::vector<float>>& inputGrad, const std::vector<std::vector<float>>& outputGrad);

    std::vector<std::vector<float>> getAttentionWeights() const;
    std::vector<std::vector<float>> getOutputMask() const;
    std::vector<std::vector<float>> getMultiHeadAttentionParameters() const;
    std::vector<std::vector<float>> getFeedForwardNetworkParameters() const;

    void setMultiHeadAttentionParameters(const std::vector<std::vector<float>>& parameters);
    void setFeedForwardNetworkParameters(const std::vector<std::vector<float>>& parameters);

private:
    std::vector<float> selfAttention(std::vector<float>&& input);
    std::vector<float> feedForwardNetwork(std::vector<float>&& input);
    std::vector<float> layerNormalization(std::vector<float>&& input);
    std::vector<float> dropout(std::vector<float>&& input);
    std::vector<std::vector<float>> residualConnection(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& output);
    std::vector<std::vector<float>> input_;
    std::vector<std::vector<float>> output_;
    std::vector<std::vector<float>> attentionOutputNorm;
    std::vector<std::vector<float>> attentionOutputResidual;
    std::vector<std::vector<float>> feedForwardOutputResidual;

    Attention multiHeadAttentionLayer_;
    LayerNorm layerNorm1_;
    LayerNorm layerNorm2_;
    FeedForwardNetwork feedForwardNetworkLayer_;
    PositionalEncoding positionalEncodingLayer_;

private:
    int numHeads_;
    int hiddenSize_;
    int feedForwardSize_;
    bool enableLayerNormalization_;
    bool enableDropout_;
    float dropoutRate_;
    bool enableMultiHeadAttention_;
    int headSize_;
    float attentionDropoutRate_;
    bool enablePositionwiseFeedForward_;
    int feedForwardSize_;
    float positionwiseFeedForwardDropoutRate_;
    bool enableResidualConnection_;
    float residualConnectionDropoutRate_;
    int outputProjectionSize_;
    std::function<float(float)> feedForwardActivation_;
    std::vector<float> input_;
    std::vector<float> output_;
    std::vector<std::vector<float>> attentionWeights_;
    std::vector<std::vector<float>> attentionMask_;
};

#endif  // TRANSFORMER_ENCODER_LAYER_H
