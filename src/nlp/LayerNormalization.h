#ifndef LAYER_NORMALIZATION_H
#define LAYER_NORMALIZATION_H

#include <iostream>
#include <cmath>
#include <stdexcept>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>
#include <execution>
#include <fstream>
#include <memory>

class LayerNormalization {
private:
    float epsilon_;
    std::vector<float> mean_;
    std::vector<float> variance_;
    std::vector<float> scale_;
    std::vector<float> shift_;
    bool trainingMode_;
    bool populationStatsMode_;

public:
    LayerNormalization(const std::string& name, float epsilon = 1e-8);

    void initialize();

    void setScaleShift(const std::vector<float>& scaleParams, const std::vector<float>& shiftParams);

    void initializeRandomScaleShift(int size);

    void forward(float* input, int size);

    void calculateBatchStatistics(const float** inputs, int batchSize, int size);

    void calculatePopulationStatistics(const float** inputs, int dataSize, int size);

    void normalize(float* input, int size) const;

    void normalize(const float* input, int size, float* output) const;

    void batchNormalize(const float** inputs, int batchSize, int size);

    void batchNormalize(const float** inputs, int batchSize, int size, float** outputs);

    void inverseNormalize(const float* normalizedInput, int size, float* output) const;

    void setTrainingMode(bool mode);

    void setPopulationStatsMode(bool mode);

    void computeGradients(const float** inputs, int batchSize, int size, float** gradients) const;

    void resizeParameters(int size);

    void setEpsilon(float eps);

    void accumulateGradients(const float** gradients, int batchSize, int size);

    void setMixedPrecision(bool enableMixedPrecision);

    void enableExponentialMovingAverage(float decayRate);

    void setCustomEpsilonPerInstance(const std::vector<float>& epsilonValues);

    void serializeModel(const std::string& filePath) const;

    void deserializeModel(const std::string& filePath);

    void enableMultiGPUTraining(int numGPUs);

    void applyRegularization(float lambda);

    void enableAutomaticDifferentiation(bool enableAutoDiff);

    // Getters and Setters
    float getEpsilon() const;

    const std::vector<float>& getMean() const;

    const std::vector<float>& getVariance() const;

    const std::vector<float>& getScale() const;

    const std::vector<float>& getShift() const;

    bool getTrainingMode() const;

    bool getPopulationStatsMode() const;

    ~LayerNormalization();

    LayerNormalization(const LayerNormalization& other);

    LayerNormalization& operator=(const LayerNormalization& other);

    LayerNormalization(LayerNormalization&& other);

    LayerNormalization& operator=(LayerNormalization&& other);
};

#endif  // LAYER_NORMALIZATION_H

