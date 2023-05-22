#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>

#include "Utils.h"
#include "GpuTypes.h"
#include "Types.h"
#include "TestUtils.h"

// TODO: add test paths

/**
 * @brief Test fixture for activation functions testing.
 */
class TestActivationFunctions : public testing::Test {
public:
    /**
     * @brief Tests the activation functions.
     */
    void testActivationFunctions() {
        const std::vector<std::string> modelPaths = {
            std::string(TEST_DATA_PATH) + "",
            std::string(TEST_DATA_PATH) + ""
        };
        const std::vector<size_t> batches = {2, 4};

        for (size_t i = 0; i < modelPaths.size(); i++) {
            DataParameters dataParameters;
            dataParameters.numberOfSamples = 1024;
            dataParameters.inpFeatureDimensionality = 2;
            dataParameters.outFeatureDimensionality = 2;
            bool result = validateNeuralNetwork(batches[i], modelPaths[i], Classification, dataParameters, std::cout);
            std::cout << "batches " << batches[i] << ", model " << modelPaths[i] << '\n';
            ASSERT_TRUE(result) << "failed on testActivationFunctions";
        }
    }
};

/**
 * @brief Test case for activation functions testing.
 */
TEST_F(TestActivationFunctions, testActivationFunctions) {
    const std::vector<std::string> modelPaths = {
        std::string(TEST_DATA_PATH) + "",
        std::string(TEST_DATA_PATH) + ""
    };
    const std::vector<size_t> batches = {2, 4};

    for (size_t i = 0; i < modelPaths.size(); i++) {
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 1024;
        dataParameters.inpFeatureDimensionality = 2;
        dataParameters.outFeatureDimensionality = 2;
        bool result = validateNeuralNetwork(batches[i], modelPaths[i], Classification, dataParameters, std::cout);
        std::cout << "batches " << batches[i] << ", model " << modelPaths[i] << '\n';
        ASSERT_TRUE(result) << "failed on testActivationFunctions";
    }
}

/**
 * @brief Main entry point of the program.
 *
 * @param argc The number of command-line arguments.
 * @param argv The command-line arguments.
 * @return int The exit code.
 */
int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
