#include <iostream>
#include <gtest/gtest.h>
#include "Utils.h"

/**
 * @brief Test fixture for Utils.
 */
class TestUtils : public ::testing::Test
{
protected:
    /**
     * @brief Test case setup.
     */
    void SetUp() override
    {
        // Any necessary setup for the test cases
    }
};

/**
 * @brief Test case for isNetCDFfile function.
 */
TEST_F(TestUtils, TestIsNetCDFfile)
{
    /**
     * @test Check if isNetCDFfile returns true for valid NetCDF file.
     */
    bool result = isNetCDFfile("network.nc");
    ASSERT_TRUE(result);

    /**
     * @test Check if isNetCDFfile returns false for invalid NetCDF file.
     */
    result = isNetCDFfile("network.nic");
    ASSERT_FALSE(result);
}
