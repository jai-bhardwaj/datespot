#include <iostream>
#include <cassert>

#include "Utils.h"

/**
 * @brief Test class for Utils.
 */
[[test_suite("TestUtils")]]
class TestUtils
{
public:
    /**
     * @brief Test case for isNetCDFfile function.
     */
    [[test_case("TestIsNetCDFfile")]]
    void TestIsNetCDFfile()
    {
        /**
         * @test Check if isNetCDFfile returns true for valid NetCDF file.
         */
        bool result = isNetCDFfile("network.nc");
        assert(result);

        /**
         * @test Check if isNetCDFfile returns false for invalid NetCDF file.
         */
        result = isNetCDFfile("network.nic");
        assert(!result);
    }
};

