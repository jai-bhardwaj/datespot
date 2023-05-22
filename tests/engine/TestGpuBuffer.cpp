#include <gtest/gtest.h>
#include "engine/GpuTypes.h"

/**
 * @brief Test fixture for GpuBuffer class.
 */
class TestGpuBuffer : public testing::Test
{
protected:
    /**
     * @brief Set up the test fixture.
     * 
     * This function is called before each test case.
     */
    void SetUp() override
    {
        length = 1024;
        buff.Resize(length, false, true);

        for (std::size_t i = 0; i < length; ++i)
        {
            buff[i] = static_cast<uint32_t>(i);
        }
    }

    /**
     * @brief Tear down the test fixture.
     * 
     * This function is called after each test case.
     */
    void TearDown() override
    {
        // Clean up resources if needed
    }

    std::size_t length;             /**< Length of the GpuBuffer */
    GpuBuffer<uint32_t> buff;       /**< GpuBuffer instance */
};

/**
 * @brief Test case for GpuBuffer Resize function.
 */
TEST_F(TestGpuBuffer, Resize)
{
    for (std::size_t i = 0; i < length; ++i)
    {
        EXPECT_EQ(i, buff[i]);
    }

    buff.Resize(length);
    for (std::size_t i = 0; i < length; ++i)
    {
        EXPECT_EQ(i, buff[i]);
    }

    buff.Resize(length - 1);
    for (std::size_t i = 0; i < length; ++i)
    {
        EXPECT_EQ(i, buff[i]);
    }

    buff.Resize(length + 1);
    bool isSame = true;
    for (std::size_t i = 0; i < length; ++i)
    {
        isSame &= (buff[i] == static_cast<uint32_t>(i));
    }
    EXPECT_FALSE(isSame);
}
