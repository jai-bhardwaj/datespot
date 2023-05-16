#include <cerrno>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>
#include <dirent.h>
#include <cstring>
#include <vector>
#include <string>
#include <filesystem>
#include "Utils.h"


/**
 * @brief Updates a metric with a string value.
 *
 * @param metric The name of the metric.
 * @param value The value of the metric as a string.
 */
void CWMetric::updateMetrics(const std::string& metric, const std::string& value)
{
    // TODO Code implementation for EKS metrics
}

/**
 * @brief Retrieves the value of a command-line option.
 *
 * @param begin The beginning of the command-line arguments.
 * @param end The end of the command-line arguments.
 * @param option The option to search for.
 * @return A pointer to the value of the option if found, nullptr otherwise.
 */
char* getCmdOption(char** begin, char** end, const std::string& option)
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
bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

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
std::string getRequiredArgValue(int argc, char** argv, const std::string& flag, const std::string& message, void (*usage)())
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
 * @param argv The array of command-line argument strings.
 * @param flag The flag corresponding to the argument.
 * @param defaultValue The default value to return if the argument is missing.
 * @return The value of the optional argument if present, or the default value otherwise.
 */
std::string getOptionalArgValue(int argc, char** argv, const std::string& flag, const std::string& defaultValue)
{
    if (!cmdOptionExists(argv, argv + argc, flag))
    {
        return defaultValue;
    }
    else
    {
        return std::string(getCmdOption(argv, argv + argc, flag));
    }
}

/**
 * @brief Checks if a command-line argument is set.
 *
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line argument strings.
 * @param flag The flag corresponding to the argument.
 * @return true if the argument is set, false otherwise.
 */
bool isArgSet(int argc, char** argv, const std::string& flag)
{
    return cmdOptionExists(argv, argv + argc, flag);
}

/**
 * @brief Checks if a file exists.
 *
 * @param fileName The file name.
 * @return true if the file exists, false otherwise.
 */
bool fileExists(const std::string& fileName)
{
    std::ifstream stream(fileName);
    return stream.good();
}

/**
 * @brief Checks if a file has a NetCDF extension.
 *
 * @param filename The name of the file.
 * @return true if the file has a NetCDF extension, false otherwise.
 */
bool isNetCDFfile(const std::string& filename) 
{
    std::filesystem::path filePath(filename);
    std::string ext = filePath.extension().string();
    return (ext.compare(NETCDF_FILE_EXTENTION) == 0);
}

/**
 * @brief Splits a string into substrings based on a delimiter and populates a vector.
 *
 * @param s The input string to be split.
 * @param delim The delimiter character.
 * @param elems The vector to store the resulting substrings.
 * @return A reference to the vector of substrings.
 */
std::vector<std::string>& split(const std::string& s, char delim, std::vector<std::string>& elems)
{
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim))
    {
        elems.push_back(item);
    }
    return elems;
}

/**
 * @brief Splits a string into substrings based on a delimiter and returns a vector.
 *
 * @param s The input string to be split.
 * @param delim The delimiter character.
 * @return A vector of substrings.
 */
std::vector<std::string> split(const std::string& s, char delim)
{
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

/**
 * @brief Checks if a directory exists.
 *
 * @param dirname The directory name.
 * @return true if the directory exists, false otherwise.
 */
bool isDirectory(const std::string& dirname) {
    return std::filesystem::is_directory(dirname);
}

/**
 * @brief Checks if a file exists.
 *
 * @param filename The file name.
 * @return true if the file exists, false otherwise.
 */
bool isFile(const std::string& filename) {
    return std::filesystem::is_regular_file(filename);
}

/**
 * @brief Lists files in a directory.
 *
 * @param dirname The directory name.
 * @param recursive Flag indicating whether to list files recursively in subdirectories.
 * @param files A vector to store the list of files.
 * @return An integer indicating the status of the operation (0 for success, 1 for failure).
 */
int listFiles(const std::string& dirname, const bool recursive, std::vector<std::string>& files) {
    std::filesystem::path dirPath(dirname);
    if (std::filesystem::is_regular_file(dirPath)) {
        files.push_back(dirname);
    } else if (std::filesystem::is_directory(dirPath)) {
        std::filesystem::directory_iterator it(dirPath), end;
        for (; it != end; ++it) {
            const std::string& relativeChildFilePath = it->path().filename().string();
            if (relativeChildFilePath == "." || relativeChildFilePath == "..") {
                continue;
            }
            std::string absoluteChildFilePath = dirPath / relativeChildFilePath;

            if (recursive && std::filesystem::is_directory(absoluteChildFilePath)) {
                listFiles(absoluteChildFilePath, recursive, files);
            } else {
                files.push_back(absoluteChildFilePath);
            }
        }
    } else {
        return 1;
    }
    
    std::sort(files.begin(), files.end());
    return 0;
}

/**
 * @brief Comparator function to compare pairs based on the first element.
 *
 * @tparam Tkey The type of the first element in the pair.
 * @tparam Tval The type of the second element in the pair.
 * @param left The left pair to compare.
 * @param right The right pair to compare.
 * @return true if the first element of the left pair is greater than the first element of the right pair, false otherwise.
 */
template<typename Tkey, typename Tval>
bool cmpFirst(const std::pair<Tkey, Tval>& left, const std::pair<Tkey, Tval>& right) {
    return left.first > right.first;
}

/**
 * @brief Comparator function to compare pairs based on the second element.
 *
 * @tparam Tkey The type of the first element in the pair.
 * @tparam Tval The type of the second element in the pair.
 * @param left The left pair to compare.
 * @param right The right pair to compare.
 * @return true if the second element of the left pair is greater than the second element of the right pair, false otherwise.
 */
template<typename Tkey, typename Tval>
bool cmpSecond(const std::pair<Tkey, Tval>& left, const std::pair<Tkey, Tval>& right) {
    return left.second > right.second;
}

/**
 * @brief Sorts the data based on either the key or value and populates the output arrays.
 *
 * @tparam Tkey The type of the keys.
 * @tparam Tval The type of the values.
 * @param keys The input array of keys.
 * @param vals The input array of values (optional, default is nullptr).
 * @param size The size of the input arrays.
 * @param Outputkeys The output array to store the sorted keys.
 * @param Outputvals The output array to store the sorted values.
 * @param Output The size of the output arrays.
 * @param sortByKey Flag indicating whether to sort by key.
 */
template<typename Tkey, typename Tval>
void Outputsort(Tkey* keys, Tval* vals, const int size, Tkey* Outputkeys, Tval* Outputvals, const int Output, const bool sortByKey) {
  if (!keys || !Outputkeys || !Outputvals) {
    std::cout << "null input array" << std::endl;
    exit(0);
  }

  std::vector<std::pair<Tkey, Tval>> data(size);
  if (vals) {
    int i = 0;
    for (auto& key : keys) {
      data[i].first = key;
      data[i].second = vals[i];
      i++;
    }
  } else {
    int i = 0;
    for (auto& key : keys) {
      data[i].first = key;
      data[i].second = i;
      i++;
    }
  }
}

/**
 * @brief Sorts the data based on either the key or value and populates the output arrays.
 *
 * @tparam TKey The type of the keys.
 * @tparam TValue The type of the values.
 * @param keys The input array of keys.
 * @param values The input array of values.
 * @param size The size of the input arrays.
 * @param outputKeys The output array to store the sorted keys.
 * @param outputValues The output array to store the sorted values.
 * @param outputSize The size of the output arrays.
 * @param sortByKey Flag indicating whether to sort by key (default is true).
 */
template<typename TKey, typename TValue>
void Outputsort(TKey* keys, TValue* values, const int size, TKey* outputKeys, TValue* outputValues, const int outputSize, const bool sortByKey = true)
{
  std::vector<std::pair<TKey, TValue>> data;
  for (int i = 0; i < size; i++) {
    data.emplace_back(keys[i], values[i]);
  }

  if (sortByKey) {
    std::nth_element(data.begin(), data.begin() + outputSize, data.end(), [](const auto& a, const auto& b) {
      return a.first < b.first;
    });
    std::sort(data.begin(), data.begin() + outputSize, [](const auto& a, const auto& b) {
      return a.first < b.first;
    });
  } else {
    std::nth_element(data.begin(), data.begin() + outputSize, data.end(), [](const auto& a, const auto& b) {
      return a.second < b.second;
    });
    std::sort(data.begin(), data.begin() + outputSize, [](const auto& a, const auto& b) {
      return a.second < b.second;
    });
  }

  int i = 0;
  for (const auto& pair : data) {
    outputKeys[i] = pair.first;
    outputValues[i] = pair.second;
    i++;
    if (i >= outputSize)
      break;
  }
}

template
void Outputsort<float, unsigned int>(float*, unsigned int*, const int, float*, unsigned int*, const int, const bool);

template
void Outputsort<float, float>(float*, float*, const int, float*, float*, const int, const bool);
