#include "gtest/gtest.h"
#include <string>
#include <random>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <format>

#include "Utils.h"
#include "GpuTypes.h"
#include "Types.h"
#include "TestUtils.h"

namespace fs = std::filesystem;

/**
 * @brief Test fixture for the cost functions tests.
 * 
 * This fixture sets up the necessary environment for the cost functions tests.
 */
class TestCostFunctions : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        getGpu().SetRandomSeed(rd());
        getGpu().CopyConstants();
    }
};

/**
 * @brief Test case for the cost functions.
 * 
 * This test case validates the cost functions by running various scenarios.
 */
TEST_F(TestCostFunctions, testCostFunctions) {
    const size_t batch = 2;
    const fs::path modelPath = fs::path(TEST_DATA_PATH) / "validate_DataScaledMarginalCrossEntropy_02.json";
    
    DataParameters dataParameters = {
        .numberOfSamples = 1024,
        .inpFeatureDimensionality = 2,
        .outFeatureDimensionality = 2
    };
    
    // Validate the Neural Network with Classification Analog
    bool result = validateNeuralNetwork(batch, modelPath.string(), ClassificationAnalog, dataParameters, std::cout);
    ASSERT_TRUE(result) << "failed on DataScaledMarginalCrossEntropy";

    const size_t batch2 = 4;
    const fs::path modelPath2 = fs::path(TEST_DATA_PATH) / "validate_ScaledMarginalCrossEntropy_02.json";
    
    DataParameters dataParameters2 = {
        .numberOfSamples = 1024,
        .inpFeatureDimensionality = 2,
        .outFeatureDimensionality = 2
    };
    
    // Validate the Neural Network with Classification
    bool result2 = validateNeuralNetwork(batch2, modelPath2.string(), Classification, dataParameters2, std::cout);
    ASSERT_TRUE(result2) << "failed on DataScaledMarginalCrossEntropy";

    const size_t batch3 = 4;
    const fs::path modelPath3 = fs::path(TEST_DATA_PATH) / "validate_L2_02.json";
    
    DataParameters dataParameters3 = {
        .numberOfSamples = 1024,
        .inpFeatureDimensionality = 1,
        .outFeatureDimensionality = 1,
        .W0 = -2.f,
        .B0 = 3.f
    };
    
    // Validate the Neural Network with Regression
    bool result3 = validateNeuralNetwork(batch3, modelPath3.string(), Regression, dataParameters3, std::cout);
    ASSERT_TRUE(result3) << "failed on DataScaledMarginalCrossEntropy";
}
