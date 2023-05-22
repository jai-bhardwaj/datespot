#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <chrono>
#include <random>
#include <string>
#include <algorithm>
#include <iostream>

#include "GpuTypes.h"
#include "Types.h"
#include "kernels.h"
#include "Utils.h"

/**
 * Generates random data for testing.
 * 
 * @param pTarget Pointer to the target data array.
 * @param pOut Pointer to the output data array.
 * @param batch Number of batches.
 * @param nFeatures Number of features.
 * @param stride Stride size.
 */
void randData(Float* pTarget, Float* pOut, const size_t batch, const size_t nFeatures, const size_t stride) {
    std::fill_n(pTarget, stride * batch, 0);
    std::fill_n(pOut, stride * batch, 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dis(0, nFeatures - 1);
    std::uniform_real_distribution<Float> disFloat(0.f, 1.f);

    for (size_t i = 0; i < batch; i++) {
        for (size_t k = 0; k < nFeatures; k++) {
            pTarget[k] = dis(gen);
        }
        for (size_t o = 0; o < nFeatures; o++) {
            pOut[o] = disFloat(gen);
        }
        pTarget += stride;
        pOut += stride;
    }
}

/**
 * Tests the topK function.
 * 
 * @param batch Number of batches.
 * @param topK Top K value.
 * @param nFeatures Number of features.
 * @return True if the test passes, false otherwise.
 */
bool testTopK(const size_t batch = 128, const size_t topK = 128, const size_t nFeatures = 1024) {

    std::cout << "TEST kCalculateOutput with parameters: " << "batch=" << batch << " topK=" << topK << " nFeatures=" << nFeatures << std::endl;
    bool ret = true;

    const float eps = 1.e-6;
    const size_t stride = ((nFeatures + 127) >> 7) << 7;

    std::unique_ptr<Float[]> pbKey(new Float[batch * topK]);
    std::unique_ptr<Float[]> pbFValue(new Float[batch * topK]);
    std::unique_ptr<unsigned int[]> pbUIValue(new unsigned int[batch * topK]);

    std::unique_ptr<Float[]> pbTarget(new Float[batch * stride]);
    std::unique_ptr<Float[]> pbOutput(new Float[batch * stride]);

    std::cout << "1 TEST kCalculateOutput with 3 args" << std::endl;

    {
        Float* pTarget = pbTarget.get();
        Float* pOut = pbOutput.get();

        randData(pTarget, pOut, batch, nFeatures, stride);

        std::fill_n(pbUIValue.get(), batch * topK, 0);
    }
    {
        auto const start = std::chrono::steady_clock::now();
        kCalculateOutput(pbOutput.get(), pbKey.get(), pbUIValue.get(), batch, stride, topK);
        auto const end = std::chrono::steady_clock::now();
        std::cout << "GPU sort: " << std::chrono::duration<double>(end - start).count() << std::endl;
    }

    {
        std::vector<float> keys(nFeatures);
        std::vector<unsigned int> topKvals(topK);
        std::vector<float> topKkeys(topK);

        Float* pOutput = pbOutput.get();
        Float* pKey = pbKey.get();
        unsigned int* pUIValue = pbUIValue.get();

        int countValueError = 0;
        float sumKeyError = 0.f;
        float cpuSort = 0.f;

        for (size_t i = 0; i < batch; i++) {
            auto const start = std::chrono::steady_clock::now();
            topKsort(pOutput, nullptr, nFeatures, topKkeys.data(), topKvals.data(), topK);
            auto const end = std::chrono::steady_clock::now();
            cpuSort += std::chrono::duration<double>(end - start).count();

            for (size_t k = 0; k < topK; k++) {
                unsigned int GPUvalue = pUIValue[k];
                float GPUkey = pKey[k];

                float CPUvalue = topKvals[k];
                float CPUkey = topKkeys[k];

                if (fabs(GPUvalue - CPUvalue) > eps) {
                    countValueError++;
                }
                sumKeyError += fabs(GPUkey - CPUkey);
            }
            pKey += topK;
            pUIValue += topK;
            pOutput += stride;
        }
        std::cout << "CPU sort: " << cpuSort << std::endl;

        if (countValueError && sumKeyError) {
            std::cout << "1 ERROR kCalculateOutput with 3 args; ";
            ret = false;
        } else {
            std::cout << "1 PASS kCalculateOutput with 3 args; ";
        }
        std::cout << "countValueError " << countValueError << " sumKeyError " << sumKeyError << std::endl;
    }

    std::cout << "2 TEST kCalculateOutput with 4 args" << std::endl;

    {
        Float* pTarget = pbTarget.get();
        Float* pOut = pbOutput.get();

        randData(pTarget, pOut, batch, nFeatures, stride);
    }

    kCalculateOutput(pbOutput.get(), pbTarget.get(), pbKey.get(), pbFValue.get(), batch, stride, topK);

    {
        std::vector<float> vals(nFeatures);
        std::vector<float> keys(nFeatures);
        std::vector<float> topKvals(topK);
        std::vector<float> topKkeys(topK);

        Float* pOutput = pbOutput.get();
        Float* pTarget = pbTarget.get();
        Float* pKey = pbKey.get();
        Float* pValue = pbFValue.get();

        int countValueError = 0;
        float sumKeyError = 0;

        for (size_t i = 0; i < batch; i++) {

            topKsort(pOutput, pTarget, nFeatures, topKkeys.data(), topKvals.data(), topK);

            for (size_t k = 0; k < topK; k++) {
                unsigned int GPUvalue = static_cast<unsigned int>(pValue[k]);
                float GPUkey = pKey[k];

                float CPUvalue = topKvals[k];
                float CPUkey = topKkeys[k];

                if (fabs(GPUvalue - CPUvalue) > eps) {
                    countValueError++;
                }
                sumKeyError += fabs(GPUkey - CPUkey);
            }
            pKey += topK;
            pValue += topK;
            pOutput += stride;
            pTarget += stride;
        }

        if (countValueError && sumKeyError) {
            std::cout << "2 ERROR kCalculateOutput with 4 args; ";
            ret = false;
        } else {
            std::cout << "2 PASS kCalculateOutput with 4 args; ";
        }
        std::cout << "countValueError " << countValueError << " sumKeyError " << sumKeyError << std::endl;
    }

    int totalGPUMemory;
    int totalCPUMemory;
    getGpu().GetMemoryUsage(&totalGPUMemory, &totalCPUMemory);
    std::cout << "GPU Memory Usage: " << totalGPUMemory << " KB" << std::endl;
    std::cout << "CPU Memory Usage: " << totalCPUMemory << " KB" << std::endl;

    return ret;
}

/**
 * Test fixture for the TestSort class.
 */
class TestSort : public ::testing::Test {
protected:
    void SetUp() override {
        getGpu().SetRandomSeed(12345);
        getGpu().CopyConstants();
    }
};

/**
 * Test case for testing CPU and GPU sorting.
 */
TEST_F(TestSort, TestCPU_GPUSort) {
    const size_t numberTests = 5;
    const size_t batches[numberTests] =        {128,  128,    128,  128, 128};
    const size_t topK[numberTests] =           {128,  128,    64,   32,  1};
    const size_t numberFeatures[numberTests] = {1024, 100000, 1024, 64,  64};

    for (size_t i = 0; i < numberTests; i++) {
        bool result = testTopK(batches[i], topK[i], numberFeatures[i]);
        std::cout << "batches " << batches[i] <<  ", topK " << topK[i] << ", numberFeatures " << numberFeatures[i] << std::endl;
        EXPECT_TRUE(result);
    }
}
