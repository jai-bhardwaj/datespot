#include <gtest/gtest.h>
#include <ranges>

#include "engine/GpuTypes.h"
#include "engine/Types.h"
#include "engine/Layer.h"

/**
 * @brief Test fixture for DataSetDimensions class.
 */
class TestDataSetDimensions : public testing::Test
{
};

/**
 * @brief Test case to verify the number of dimensions in DataSetDimensions objects.
 */
TEST_F(TestDataSetDimensions, testNumDimensions)
{
    DataSetDimensions zero_d(1);       ///< DataSetDimensions object with zero dimensions.
    DataSetDimensions one_d(2);        ///< DataSetDimensions object with one dimension.
    DataSetDimensions two_d(2, 2);     ///< DataSetDimensions object with two dimensions.
    DataSetDimensions three_d(2, 2, 2); ///< DataSetDimensions object with three dimensions.

    EXPECT_EQ(std::ranges::size(zero_d), 0U);     ///< Expecting zero dimensions in zero_d.
    EXPECT_EQ(std::ranges::size(one_d), 1U);      ///< Expecting one dimension in one_d.
    EXPECT_EQ(std::ranges::size(two_d), 2U);      ///< Expecting two dimensions in two_d.
    EXPECT_EQ(std::ranges::size(three_d), 3U);    ///< Expecting three dimensions in three_d.
}
