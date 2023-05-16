#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <chrono>
#include <ratio>
#include <string>
#include <utility>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <filesystem>

using std::string;
using std::vector;

/**
 * @brief Suffix for input dataset names.
 */
constexpr std::string_view INPUT_DATASET_SUFFIX = "_input";

/**
 * @brief Suffix for output dataset names.
 */
constexpr std::string_view OUTPUT_DATASET_SUFFIX = "_output";

/**
 * @brief File extension for NetCDF files.
 */
constexpr std::string_view NETCDF_FILE_EXTENTION = ".nc";

/**
 * @brief Fixed seed value for random number generation.
 */
constexpr unsigned long FIXED_SEED = 12134ull;

/**
 * @brief Class for updating metrics.
 */
class CWMetric
{
public:
    /**
     * @brief Updates a metric with a string value.
     *
     * @param metric The name of the metric.
     * @param value The value of the metric as a string.
     */
    static void updateMetrics(const std::string& metric, const std::string& value);

    /**
     * @brief Updates a metric with a generic value.
     *
     * @tparam Value The type of the value.
     * @param metric The name of the metric.
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
 * @return A pointer to the value of the option if found, nullptr otherwise.
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
 * @param argv The array of command-line argument strings.
 * @param flag The flag corresponding to the argument.
 * @param message The error message to display if the argument is missing.
 * @param usage The function pointer to the usage function to call in case of an error.
 * @return The value of the required argument.
 */
std::string getRequiredArgValue(int argc, char** argv, const std::string& flag, const std::string& message, void (*usage)());

/**
 * @brief Retrieves the value of an optional command-line argument.
 *
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line argument strings.
 * @param flag The flag corresponding to the argument.
 * @param defaultValue The default value to return if the argument is missing.
 * @return The value of the optional argument if present, or the default value otherwise.
 */
std::string getOptionalArgValue(int argc, char** argv, const std::string& flag, const std::string& defaultValue);

/**
 * @brief Checks if a command-line argument is set.
 *
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line argument strings.
 * @param flag The flag corresponding to the argument.
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
 * @brief Checks if a file has a NetCDF extension.
 *
 * @param filename The name of the file.
 * @return true if the file has a NetCDF extension, false otherwise.
 */
bool isNetCDFfile(const std::string& filename);

/**
 * @brief Splits a string into a vector of substrings based on a delimiter.
 *
 * @param s The input string to be split.
 * @param delim The delimiter character.
 * @param elems The vector to store the resulting substrings.
 * @return A reference to the vector of substrings.
 */
std::vector<std::string>& split(const std::string& s, char delim, std::vector<std::string>& elems);

/**
 * @brief Splits a string into a vector of substrings based on a delimiter.
 *
 * @param s The input string to be split.
 * @param delim The delimiter character.
 * @return A vector of substrings.
 */
std::vector<std::string> split(const std::string& s, char delim);

/**
 * @brief Calculates the elapsed time in seconds between two time points.
 *
 * @tparam Clock The clock type used for the time points.
 * @tparam Duration1 The duration type of the first time point.
 * @tparam Duration2 The duration type of the second time point.
 * @param start The starting time point.
 * @param end The ending time point.
 * @return The elapsed time in seconds as a floating-point value.
 */
template <typename Clock, typename Duration1, typename Duration2>
double elapsed_seconds(const std::chrono::time_point<Clock, Duration1>& start,
                       const std::chrono::time_point<Clock, Duration2>& end)
{
    using FloatingPointSeconds = std::chrono::duration<double, std::ratio<1>>;
    return std::chrono::duration_cast<FloatingPointSeconds>(end - start).count();
}

/**
 * @brief Checks if a directory exists.
 *
 * @param dirname The directory name.
 * @return true if the directory exists, false otherwise.
 */
bool isDirectory(const std::string& dirname)
{
    return std::filesystem::is_directory(dirname);
}

/**
 * @brief Checks if a file exists.
 *
 * @param filename The file name.
 * @return true if the file exists, false otherwise.
 */
bool isFile(const std::string& filename)
{
    return std::filesystem::is_regular_file(filename);
}

/**
 * @brief Lists files in a directory.
 *
 * @param dirname The directory name.
 * @param recursive Flag indicating whether to list files recursively in subdirectories.
 * @param files A vector to store the list of files.
 * @return An integer indicating the status of the operation (e.g., success or failure).
 */
int listFiles(const std::string& dirname, const bool recursive, std::vector<std::string>& files)
{
    for (const auto& entry : std::filesystem::recursive_directory_iterator(dirname))
    {
        if (!recursive && entry.is_directory())
            continue;

        files.push_back(entry.path().string());
    }

    std::sort(files.begin(), files.end());
    return 0;
}

/**
 * @brief Sorts keys and values using an output array.
 *
 * @tparam Tkey The type of the keys.
 * @tparam Tval The type of the values.
 * @param keys The input array of keys.
 * @param vals The input array of values.
 * @param size The size of the input arrays.
 * @param Outputkeys The output array to store the sorted keys.
 * @param Outputvals The output array to store the sorted values.
 * @param Output The size of the output arrays.
 * @param sortByKey Flag indicating whether to sort by key (default is true).
 */
template<typename Tkey, typename Tval>
void Outputsort(Tkey* keys, Tval* vals, const int size, Tkey* Outputkeys, Tval* Outputvals, const int Output, const bool sortByKey = true)
{
    // TODO Implementation for Outputsort function
}

/**
 * @brief Generates a random integer within the specified range.
 *
 * @param min The minimum value of the range (inclusive).
 * @param max The maximum value of the range (inclusive).
 * @return A random integer within the specified range.
 */
inline int rand(int min, int max)
{
    static std::mt19937 engine(std::random_device{}());
    std::uniform_int_distribution<int> distribution(min, max);
    return distribution(engine);
}

/**
 * @brief Generates a random floating-point number within the specified range.
 *
 * @param min The minimum value of the range (inclusive).
 * @param max The maximum value of the range (inclusive).
 * @return A random floating-point number within the specified range.
 */
inline float rand(float min, float max)
{
    static std::mt19937 engine(std::random_device{}());
    std::uniform_real_distribution<float> distribution(min, max);
    return distribution(engine);
}
