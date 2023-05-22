#include <gtest/gtest.h>
#include <string>
#include <memory>
#include <array>
#include "filterKernels.h"

/**
 * @brief Test fixture for GPU-related tests.
 */
class TestGpu : public testing::Test {
protected:
    /**
     * @brief Set up the test suite.
     */
    static void SetUpTestSuite() {
        getGpu().Startup(0, nullptr);
    }

    /**
     * @brief Tear down the test suite.
     */
    static void TearDownTestSuite() {
        getGpu().Shutdown();
    }
};

/**
 * @brief Test case for applying node filter on GPU.
 */
TEST(TestGpu, TestApplyNodeFilter) {
    constexpr int outputKeySize = 6;
    constexpr int filterSize = 3;
    
    std::array<Float, outputKeySize> localOutputKey{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::array<Float, outputKeySize> expectedOutputKey{7.0, 16.0, 27.0, 28.0, 40.0, 54.0};
    std::array<Float, filterSize> localFilter{7.0, 8.0, 9.0};
    std::array<Float, filterSize> expectedFilter{7.0, 8.0, 9.0};
    
    std::unique_ptr<GpuBuffer<Float>> deviceOutputKey(new GpuBuffer<Float>(outputKeySize));
    std::unique_ptr<GpuBuffer<Float>> deviceFilter(new GpuBuffer<Float>(filterSize));
    
    deviceOutputKey->Upload(localOutputKey.data());
    deviceFilter->Upload(localFilter.data());
    
    kApplyNodeFilter(deviceOutputKey->_pDevData, deviceFilter->_pDevData, filterSize, 2);
    
    deviceOutputKey->Download(localOutputKey.data());
    deviceFilter->Download(localFilter.data());
    
    for (int i = 0; i < outputKeySize; ++i) {
        EXPECT_EQ(expectedOutputKey[i], localOutputKey[i]) << "OutputKey is different";
    }
    
    for (int i = 0; i < filterSize; ++i) {
        EXPECT_EQ(expectedFilter[i], localFilter[i]) << "Filter is different";
    }
}
