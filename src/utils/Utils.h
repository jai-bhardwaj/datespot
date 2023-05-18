#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <chrono>
#include <string>
#include <random>
#include <filesystem>

namespace fs = std::filesystem;

constexpr std::string_view INPUT_DATASET_SUFFIX = "_input";
constexpr std::string_view OUTPUT_DATASET_SUFFIX = "_output";
constexpr std::string_view NETCDF_FILE_EXTENSION = ".nc";
constexpr unsigned long long FIXED_SEED = 12134ull;

/**
 * @brief Class for updating metrics.
 */
class CWMetric
{
public:
    /**
     * @brief Updates the metrics with the given metric and value.
     *
     * @param metric The metric to update.
     * @param value The value of the metric.
     */
    static void updateMetrics(const std::string& metric, const std::string& value);

    /**
     * @brief Updates the metrics with the given metric and value.
     *
     * @tparam Value The type of the value.
     * @param metric The metric to update.
     * @param value The value of the metric.
     */
    template <typename Value, typename = decltype(std::to_string(std::declval<Value>()))>
    static void updateMetrics(std::string metric, Value&& value)
    {
        updateMetrics(std::move(metric), std::to_string(std::forward<Value>(value)));
    }
};

/**
 * @brief Retrieves the value of a command-line option.
 *
 * @param begin The beginning of the command-line arguments.
 * @param end The end of the command-line arguments.
 * @param option The option to search for.
 * @return char* The value of the option if found, nullptr otherwise.
 */
char* getCmdOption(char**, char**, const std::string&);

/**
 * @brief Checks if a command-line option exists.
 *
 * @param begin The beginning of the command-line arguments.
 * @param end The end of the command-line arguments.
 * @param option The option to search for.
 * @return true if the option exists, false otherwise.
 */
bool cmdOptionExists(char**, char**, const std::string&);

/**
 * @brief Retrieves the value of a required command-line argument.
 *
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line arguments.
 * @param flag The flag for the required argument.
 * @param message The error message to display if the argument is missing.
 * @param usage The function pointer to the usage function.
 * @return std::string The value of the required argument.
 */
std::string getRequiredArgValue(int argc, char** argv, const std::string& flag, const std::string& message, void (*usage)());

/**
 * @brief Retrieves the value of an optional command-line argument.
 *
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line arguments.
 * @param flag The flag for the optional argument.
 * @param defaultValue The default value if the argument is missing.
 * @return std::string The value of the optional argument.
 */
std::string getOptionalArgValue(int argc, char** argv, const std::string& flag, const std::string& defaultValue);

/**
 * @brief Checks if a command-line argument is set.
 *
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line arguments.
 * @param flag The flag to check.
 * @return true if the argument is set, false otherwise.
 */
bool isArgSet(int argc, char** argv, const std::string& flag);

/**
 * @brief Checks if a file exists.
 *
 * @param filename The name of the file.
 * @return true if the file exists, false otherwise.
 */
bool fileExists(const std::string& filename);

/**
 * @brief Checks if a file has the NetCDF extension.
 *
 * @param filename The name of the file.
 * @return true if the file has the NetCDF extension, false otherwise.
 */
bool isNetCDFFile(const std::string& filename);

/**
 * @brief Splits a string into substrings using a delimiter.
 *
 * @param s The string to split.
 * @param delim The delimiter character.
 * @param elems The vector to store the substrings.
 * @return std::vector<std::string>& Reference to the vector of substrings.
 */
std::vector<std::string>& split(const std::string& s, char delim, std::vector<std::string>& elems);

/**
 * @brief Splits a string into substrings using a delimiter.
 *
 * @param s The string to split.
 * @param delim The delimiter character.
 * @return std::vector<std::string> The vector of substrings.
 */
std::vector<std::string> split(const std::string& s, char delim);

/**
 * @brief Computes the elapsed time in seconds between two time points.
 *
 * @tparam Clock The clock type.
 * @tparam Duration1 The duration type of the start time point.
 * @tparam Duration2 The duration type of the end time point.
 * @param start The start time point.
 * @param end The end time point.
 * @return double The elapsed time in seconds.
 */
template <typename Clock, typename Duration1, typename Duration2>
double elapsedSeconds(const std::chrono::time_point<Clock, Duration1>& start,
                      const std::chrono::time_point<Clock, Duration2>& end)
{
    using FloatingPointSeconds = std::chrono::duration<double>;
    return std::chrono::duration_cast<FloatingPointSeconds>(end - start).count();
}

/**
 * @brief Checks if a directory exists.
 *
 * @param dirname The name of the directory.
 * @return true if the directory exists, false otherwise.
 */
bool isDirectory(const std::string& dirname);

/**
 * @brief Checks if a file exists.
 *
 * @param filename The name of the file.
 * @return true if the file exists, false otherwise.
 */
bool isFile(const std::string& filename);

/**
 * @brief Lists files in a directory.
 *
 * @param dirname The name of the directory.
 * @param recursive Flag indicating whether to list files recursively.
 * @param files The vector to store the file paths.
 * @return int 0 if successful, 1 otherwise.
 */
int listFiles(const std::string& dirname, const bool recursive, std::vector<std::string>& files);

/**
 * @brief Sorts key-value pairs by key or value.
 *
 * @tparam Tkey The type of the key.
 * @tparam Tval The type of the value.
 * @param keys The array of keys.
 * @param vals The array of values.
 * @param size The size of the input arrays.
 * @param Outputkeys The array to store the sorted keys.
 * @param Outputvals The array to store the sorted values.
 * @param Output The size of the output arrays.
 * @param sortByKey Flag indicating whether to sort by key (true) or value (false).
 */
template<typename Tkey, typename Tval>
void OutputSort(Tkey* keys, Tval* vals, const int size, Tkey* Outputkeys, Tval* Outputvals, const int Output, const bool sortByKey = true)
{
    // TODO: Implement the OutputSort function
}

/**
 * @brief Generates a random integer between the given min and max values.
 *
 * @param min The minimum value.
 * @param max The maximum value.
 * @return int The generated random integer.
 */
inline int rand(int min, int max)
{
    static std::mt19937 engine(std::random_device{}());
    std::uniform_int_distribution<int> distribution(min, max);
    return distribution(engine);
}

/**
 * @brief Generates a random float between the given min and max values.
 *
 * @param min The minimum value.
 * @param max The maximum value.
 * @return float The generated random float.
 */
inline float rand(float min, float max)
{
    static std::mt19937 engine(std::random_device{}());
    std::uniform_real_distribution<float> distribution(min, max);
    return distribution(engine);
}
