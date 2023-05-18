#include <iostream>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <string_view>
#include <vector>
#include <filesystem>
#include "Utils.h"

/**
 * @brief Updates the metrics with the given metric and value.
 * 
 * @param metric The metric to update.
 * @param value The value of the metric.
 */
void CWMetric::updateMetrics(std::string_view metric, std::string_view value)
{
}

/**
 * @brief Retrieves the value of a command-line option.
 * 
 * @param begin The beginning of the command-line arguments.
 * @param end The end of the command-line arguments.
 * @param option The option to search for.
 * @return char* The value of the option if found, nullptr otherwise.
 */
char* getCmdOption(char** begin, char** end, std::string_view option)
{
    auto itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return nullptr;
}

/**
 * @brief Checks if a command-line option exists.
 * 
 * @param begin The beginning of the command-line arguments.
 * @param end The end of the command-line arguments.
 * @param option The option to search for.
 * @return true if the option exists, false otherwise.
 */
bool cmdOptionExists(char** begin, char** end, std::string_view option)
{
    return std::find(begin, end, option) != end;
}

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
std::string getRequiredArgValue(int argc, char** argv, std::string_view flag, std::string_view message, void (*usage)())
{
    if (!cmdOptionExists(argv, argv + argc, flag))
    {
        std::cout << "Error: Missing required argument: " << flag << ": " << message << std::endl;
        usage();
        exit(1);
    }
    else
    {
        return std::string(getCmdOption(argv, argv + argc, flag));
    }
}

/**
 * @brief Retrieves the value of an optional command-line argument.
 * 
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line arguments.
 * @param flag The flag for the optional argument.
 * @param defaultValue The default value if the argument is missing.
 * @return std::string The value of the optional argument.
 */
std::string getOptionalArgValue(int argc, char** argv, std::string_view flag, std::string_view defaultValue)
{
    if (!cmdOptionExists(argv, argv + argc, flag))
    {
        return std::string(defaultValue);
    }
    else
    {
        return std::string(getCmdOption(argv, argv + argc, flag));
    }
}

/**
 * @brief Checks if a file has the NetCDF extension.
 * 
 * @param filename The name of the file.
 * @return true if the file has the NetCDF extension, false otherwise.
 */
bool isNetCDFfile(std::string_view filename)
{
    std::filesystem::path filePath(filename);
    std::string ext = filePath.extension().string();
    return (ext.compare(NETCDF_FILE_EXTENTION) == 0);
}

/**
 * @brief Splits a string into a vector of substrings using a delimiter.
 * 
 * @param s The string to split.
 * @param delim The delimiter character.
 * @return std::vector<std::string> The vector of substrings.
 */
std::vector<std::string> split(std::string_view s, char delim)
{
    std::vector<std::string> elems;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim))
    {
        elems.push_back(item);
    }
    return elems;
}

/**
 * @brief Checks if a path is a directory.
 * 
 * @param dirname The name of the directory.
 * @return true if the path is a directory, false otherwise.
 */
bool isDirectory(std::string_view dirname)
{
    return std::filesystem::is_directory(dirname);
}

/**
 * @brief Checks if a path is a regular file.
 * 
 * @param filename The name of the file.
 * @return true if the path is a regular file, false otherwise.
 */
bool isFile(std::string_view filename)
{
    return std::filesystem::is_regular_file(filename);
}

/**
 * @brief Lists files in a directory.
 * 
 * @param dirname The name of the directory.
 * @param recursive Flag indicating whether to list files recursively.
 * @param files The vector to store the file paths.
 * @return int 0 if successful, 1 otherwise.
 */
int listFiles(std::string_view dirname, bool recursive, std::vector<std::string>& files)
{
    std::filesystem::path dirPath(dirname);
    if (std::filesystem::is_regular_file(dirPath))
    {
        files.push_back(std::string(dirname));
    }
    else if (std::filesystem::is_directory(dirPath))
    {
        std::filesystem::directory_iterator it(dirPath), end;
        for (; it != end; ++it)
        {
            const std::string& relativeChildFilePath = it->path().filename().string();
            if (relativeChildFilePath == "." || relativeChildFilePath == "..")
            {
                continue;
            }
            std::string absoluteChildFilePath = (dirPath / relativeChildFilePath).string();

            if (recursive && std::filesystem::is_directory(absoluteChildFilePath))
            {
                listFiles(absoluteChildFilePath, recursive, files);
            }
            else
            {
                files.push_back(absoluteChildFilePath);
            }
        }
    }
    else
    {
        return 1;
    }

    std::sort(files.begin(), files.end());
    return 0;
}

/**
 * @brief Sorts key-value pairs by key or value.
 * 
 * @tparam TKey The type of the key.
 * @tparam TValue The type of the value.
 * @param keys The array of keys.
 * @param values The array of values.
 * @param size The size of the input arrays.
 * @param outputKeys The array to store the sorted keys.
 * @param outputValues The array to store the sorted values.
 * @param outputSize The size of the output arrays.
 * @param sortByKey Flag indicating whether to sort by key (true) or value (false).
 */
template<typename TKey, typename TValue>
void Outputsort(TKey* keys, TValue* values, int size, TKey* outputKeys, TValue* outputValues, int outputSize, bool sortByKey = true)
{
    std::vector<std::pair<TKey, TValue>> data;
    for (int i = 0; i < size; i++)
    {
        data.emplace_back(keys[i], values[i]);
    }

    if (sortByKey)
    {
        std::nth_element(data.begin(), data.begin() + outputSize, data.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        std::sort(data.begin(), data.begin() + outputSize, [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
    }
    else
    {
        std::nth_element(data.begin(), data.begin() + outputSize, data.end(), [](const auto& a, const auto& b) {
            return a.second < b.second;
        });
        std::sort(data.begin(), data.begin() + outputSize, [](const auto& a, const auto& b) {
            return a.second < b.second;
        });
    }

    int i = 0;
    for (const auto& [key, value] : data)
    {
        outputKeys[i] = key;
        outputValues[i] = value;
        i++;
        if (i >= outputSize)
            break;
    }
}

template void Outputsort<float, unsigned int>(float* keys, unsigned int* values, int size, float* outputKeys, unsigned int* outputValues, int outputSize, bool sortByKey);

template void Outputsort<float, float>(float* keys, float* values, int size, float* outputKeys, float* outputValues, int outputSize, bool sortByKey);
