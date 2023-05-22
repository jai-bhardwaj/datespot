#include <cstdlib>
#include <iostream>
#include <gtest/gtest.h>
#include "NetCDFhelper.cpp"
#include "TestUtils.cpp"

int main(int argc, char* argv[])
{
    // Initialize the Google Test framework
    ::testing::InitGoogleTest(&argc, argv);

    // Run the tests
    int exitCode = 0;
    try
    {
        exitCode = RUN_ALL_TESTS();
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Exception occurred during test execution: " << ex.what() << std::endl;
        exitCode = EXIT_FAILURE;
    }
    catch (...)
    {
        std::cerr << "Unknown exception occurred during test execution." << std::endl;
        exitCode = EXIT_FAILURE;
    }

    // Return exit status based on the test results
    return exitCode == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
