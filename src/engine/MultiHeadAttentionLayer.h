#ifndef MULTIHEADATTENTIONLAYER_H
#define MULTIHEADATTENTIONLAYER_H

#include <string>
#include <vector>
#include <cmath>

class MultiHeadAttentionLayer {
public:
    MultiHeadAttentionLayer(const std::string& name, const std::vector<std::vector<float>>& queryWeights, const std::vector<float>& queryBiases, const std::vector<std::vector<float>>& keyWeights, const std::vector<float>& keyBiases, const std::vector<std::vector<float>>& valueWeights, const std::vector<float>& valueBiases, int headSize, int numHeads);

    void initialize();
    void forward();
    void backward();

    void setQueryWeights(const std::vector<std::vector<float>>& queryWeights);
    void setQueryBiases(const std::vector<float>& queryBiases);
    void setKeyWeights(const std::vector<std::vector<float>>& keyWeights);
    void setKeyBiases(const std::vector<float>& keyBiases);
    void setValueWeights(const std::vector<std::vector<float>>& valueWeights);
    void setValueBiases(const std::vector<float>& valueBiases);

    void setHeadSize(int headSize);
    void setNumHeads(int numHeads);

    std::vector<float> getOutput() const;
    std::vector<float> getTransformedQueries() const;
    std::vector<float> getTransformedKeys() const;
    std::vector<float> getTransformedValues() const;
    std::vector<float> getInputQueries() const;
    std::vector<float> getInputKeys() const;
    std::vector<float> getInputValues() const;
    int getHeadSize() const;
    int getNumHeads() const;

private:
    std::string name;
    std::vector<std::vector<float>> queryWeights;
    std::vector<float> queryBiases;
    std::vector<std::vector<float>> keyWeights;
    std::vector<float> keyBiases;
    std::vector<std::vector<float>> valueWeights;
    std::vector<float> valueBiases;
    int headSize;
    int numHeads;

    std::vector<float> inputQueries;
    std::vector<float> inputKeys;
    std::vector<float> inputValues;
    std::vector<float> output;
    std::vector<float> transformedQueries;
    std::vector<float> transformedKeys;
    std::vector<float> transformedValues;

    std::vector<float> linearTransform(const std::vector<float>& input, const std::vector<std::vector<float>>& weights, const std::vector<float>& biases);
    std::vector<float> transpose(const std::vector<std::vector<float>>& matrix);
    std::vector<float> computeAttentionScores(const std::vector<float>& queries, const std::vector<float>& keys, int headSize);
    std::vector<float> maskedSoftmax(const std::vector<float>& input, const std::vector<std::vector<float>>& mask);
    std::vector<float> weightedSumValues(const std::vector<float>& attentionWeights, const std::vector<float>& values, int headSize);
    std::vector<float> combineAttentionOutput(const std::vector<float>& attentionOutput, const std::vector<float>& weightedSum, int numHeads);
    std::vector<std::vector<float>> getAttentionMask(int inputSize) const;
};

#endif // MULTIHEADATTENTIONLAYER_H
