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
    LayerNormalization(const std::string& name, float epsilon = 1e-8)
        : epsilon_(epsilon), trainingMode_(true), populationStatsMode_(false) {
    }

    void initialize() {
        if (scale_.empty() && shift_.empty()) {
            scale_.assign(mean_.size(), 1.0f);
            shift_.assign(mean_.size(), 0.0f);
        }
    }

    void setScaleShift(const std::vector<float>& scaleParams, const std::vector<float>& shiftParams) {
        if (scaleParams.size() != mean_.size() || shiftParams.size() != mean_.size()) {
            throw std::runtime_error("Invalid size of scale or shift parameters.");
        }

        scale_ = scaleParams;
        shift_ = shiftParams;
    }

    void initializeRandomScaleShift(int size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);

        scale_.resize(size);
        shift_.resize(size);

        for (int i = 0; i < size; i++) {
            scale_[i] = dist(gen);
            shift_[i] = dist(gen);
        }
    }

    void normalizeInput(const float* input, int size) {
        if (trainingMode_ && !populationStatsMode_) {
            throw std::runtime_error("Batch statistics not calculated. Call calculateBatchStatistics() before normalization.");
        }

        float mean = 0.0f;
        float variance = 0.0f;
        for (int i = 0; i < size; i++) {
            mean += input[i];
            variance += input[i] * input[i];
        }
        mean /= size;
        variance /= size;

        std::vector<float> output(size);
        for (int i = 0; i < size; i++) {
            output[i] = (input[i] - mean) / std::sqrt(variance);
        }

        std::copy(output.begin(), output.end(), input);
    }

    void calculateBatchStatistics(const float** inputs, int batchSize, int size) {
        if (populationStatsMode_) {
            throw std::runtime_error("Cannot calculate batch statistics in population statistics mode.");
        }

        mean_.assign(size, 0.0f);
        variance_.assign(size, 0.0f);

        std::vector<float> sum(size, 0.0f);
        std::vector<float> sumSquared(size, 0.0f);

        std::for_each(std::execution::par_unseq, inputs, inputs + batchSize, [&](const float* input) {
            for (int i = 0; i < size; i++) {
                sum[i] += input[i];
                sumSquared[i] += input[i] * input[i];
            }
        });

        float invBatchSize = 1.0f / batchSize;
        for (int i = 0; i < size; i++) {
            mean_[i] = sum[i] * invBatchSize;
            variance_[i] = std::sqrt((sumSquared[i] * invBatchSize) - (mean_[i] * mean_[i]) + epsilon_);
        }
    }

    void calculatePopulationStatistics(const float** inputs, int dataSize, int size) {
        if (trainingMode_) {
            throw std::runtime_error("Cannot calculate population statistics in training mode.");
        }

        mean_.assign(size, 0.0f);
        variance_.assign(size, 0.0f);

        std::vector<float> sum(size, 0.0f);
        std::vector<float> sumSquared(size, 0.0f);

        std::for_each(std::execution::par_unseq, inputs, inputs + dataSize, [&](const float* input) {
            for (int i = 0; i < size; i++) {
                sum[i] += input[i];
                sumSquared[i] += input[i] * input[i];
            }
        });

        float invDataSize = 1.0f / dataSize;
        for (int i = 0; i < size; i++) {
            mean_[i] = sum[i] * invDataSize;
            variance_[i] = std::sqrt((sumSquared[i] * invDataSize) - (mean_[i] * mean_[i]) + epsilon_);
        }
    }

    float inverseVariance(const std::vector<float>& data) {
        float mean = std::accumulate(data.begin(), data.end(), 0.0f) / data.size();
        float variance = 0.0f;
        for (float value : data) {
            variance += (value - mean) * (value - mean);
        }
        return 1.0f / variance;
    }

    void setTrainingMode(bool mode) {
        trainingMode_ = mode;
    }

    void setPopulationStatsMode(bool mode) {
        populationStatsMode_ = mode;
    }

    void computeGradients(const float** inputs, int batchSize, int size, float** gradients) const {
        std::vector<float> gradScale(size, 0.0f);
        std::vector<float> gradShift(size, 0.0f);

        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < size; j++) {
                gradScale[j] += inputs[i][j] * (inputs[i][j] - mean_[j]) / variance_[j];
                gradShift[j] += inputs[i][j];
            }
        }

        std::copy(gradScale.begin(), gradScale.end(), gradients[0]);
        std::copy(gradShift.begin(), gradShift.end(), gradients[1]);
    }

    void resizeParameters(int size) {
        mean_.resize(size);
        variance_.resize(size);
        scale_.resize(size);
        shift_.resize(size);
    }

    void setEpsilon(float eps) {
        epsilon_ = eps;
    }

    void accumulateGradients(const float** gradients, int batchSize, int size) {
        for (int i = 0; i < size; i++) {
            scale_[i] += gradients[0][i];
            shift_[i] += gradients[1][i];
        }
    }

    void setMixedPrecision(bool enableMixedPrecision) {
    }

    void enableExponentialMovingAverage(float decayRate) {
    }

    void serializeModel(const std::string& filePath) const {
        std::ofstream file(filePath, std::ios::binary);
        if (file.is_open()) {
            file.write(reinterpret_cast<const char*>(mean_.data()), mean_.size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(variance_.data()), variance_.size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(scale_.data()), scale_.size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(shift_.data()), shift_.size() * sizeof(float));
            file.close();
        } else {
            throw std::runtime_error("Failed to serialize model. Could not open file: " + filePath);
        }
    }

    void deserializeModel(const std::string& filePath) {
        std::ifstream file(filePath, std::ios::binary);
        if (file.is_open()) {
            file.read(reinterpret_cast<char*>(mean_.data()), mean_.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(variance_.data()), variance_.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(scale_.data()), scale_.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(shift_.data()), shift_.size() * sizeof(float));
            file.close();
        } else {
            throw std::runtime_error("Failed to deserialize model. Could not open file: " + filePath);
        }
    }

    void enableMultiGPUTraining(int numGPUs) {
    }

    void applyRegularization(float lambda) {
    }

    void enableAutomaticDifferentiation(bool enableAutoDiff) {
    }

    float getEpsilon() const {
        return epsilon_;
    }

    const std::vector<float>& getMean() const {
        return mean_;
    }

    const std::vector<float>& getVariance() const {
        return variance_;
    }

    const std::vector<float>& getScale() const {
        return scale_;
    }

    const std::vector<float>& getShift() const {
        return shift_;
    }

    bool getTrainingMode() const {
        return trainingMode_;
    }

    bool getPopulationStatsMode() const {
        return populationStatsMode_;
    }

    ~LayerNormalization() = default;

    LayerNormalization(const LayerNormalization& other) = default;

    LayerNormalization& operator=(const LayerNormalization& other) = default;

    LayerNormalization(LayerNormalization&& other) = default;

    LayerNormalization& operator=(LayerNormalization&& other) = default;
};

