#include "gtest/gtest.h"
#include <string>
#include <vector>

#include "TestSort.h"
#include "TestActivationFunctions.h"
#include "TestCostFunctions.h"

/**
 * @brief Main function to run the test suite.
 *
 * @param argc The number of command-line arguments.
 * @param argv The command-line arguments as a vector of strings.
 * @return int The test result code.
 */
int main(int argc, const std::vector<std::string>& argv) {
    testing::InitGoogleTest(&argc, argv);

    getGpu().Startup(0, nullptr);
    getGpu().SetRandomSeed(12345);
    getGpu().CopyConstants();

    int testResult = RUN_ALL_TESTS();

    getGpu().Shutdown();

    return testResult;
}
