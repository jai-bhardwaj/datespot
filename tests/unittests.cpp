#include <gtest/gtest.h>
#include <iostream>
#include <format>

/**
 * @brief Listener for printing test progress information.
 */
class PrintProgressListener
{
public:
  /**
   * @brief Called when a test starts.
   * @tparam TestType The type of the test info.
   * @param test_info The test info.
   */
  template<typename TestType>
  void OnTestStart(const TestType& test_info) const
  {
    std::cout << std::format("Running [{}]\n", test_info.name());
  }

  /**
   * @brief Called when a test ends.
   * @tparam TestType The type of the test info.
   * @param test_info The test info.
   */
  template<typename TestType>
  void OnTestEnd(const TestType& test_info) const
  {
    std::cout << std::format("Finished [{}]\n", test_info.name());
  }
};

/**
 * @brief Main function to run the tests.
 * @param argc The number of command-line arguments.
 * @param argv The command-line arguments.
 * @return Returns 0 if all tests pass, a non-zero value otherwise.
 */
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  PrintProgressListener progressListener;
  testing::UnitTest::GetInstance()->listeners().Append(&progressListener);

  bool wasSuccessful = RUN_ALL_TESTS();
  return !wasSuccessful;
}
