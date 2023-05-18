#include <fstream>
#include <string>
#include <stdexcept>
#include <sstream>
#include <format>
#include <span>
#include <filesystem>
#include <ranges>


#include "DataReader.h"

namespace {
    constexpr int success = 0;
    constexpr int failure = 1;
}

/**
 * @brief Get the number of rows in the data.
 *
 * @return The number of rows.
 */
uint32_t DataReader::getRows() const {
    return rows;
}

/**
 * @brief Get the number of columns in the data.
 *
 * @return The number of columns.
 */
int DataReader::getColumns() const {
    return columns;
}

/**
 * @brief Construct a new TextFileDataReader object.
 *
 * @param fileName The name of the text file to read.
 * @param keyValueDelimiter The delimiter character between key and value in each line.
 * @param vectorDelimiter The delimiter character between elements in the vector.
 */
TextFileDataReader::TextFileDataReader(const std::string& fileName, char keyValueDelimiter, char vectorDelimiter)
    : fileName(fileName),
      fileStream(fileName),
      keyValueDelimiter(keyValueDelimiter),
      vectorDelimiter(vectorDelimiter) {
    findDataDimensions(rows, columns);
}

/**
 * @brief Find the dimensions of the data in the text file.
 *
 * @param rows [out] The number of rows in the data.
 * @param columns [out] The number of columns in the data.
 *
 * @throw std::runtime_error if the file does not exist.
 * @throw std::runtime_error if failed to open the file.
 * @throw std::invalid_argument if a malformed line is encountered.
 * @throw std::invalid_argument if inconsistent number of columns is detected.
 */
void TextFileDataReader::findDataDimensions(uint32_t& rows, int& columns) {
    if (!std::filesystem::exists(fileName)) {
        throw std::runtime_error("File does not exist: " + fileName);
    }

    std::ifstream fs(fileName);
    if (!fs.is_open()) {
        throw std::runtime_error("Failed to open file: " + fileName);
    }

    rows = 0;
    columns = 0;

    std::string line;
    while (std::getline(fs, line)) {
        if (line.empty()) {
            continue;
        }

        ++rows;

        std::string key;
        std::string vectorStr;

        if (splitKeyVector(line, key, vectorStr, keyValueDelimiter)) {
            throw std::invalid_argument(std::format("Malformed line. key-value delimiter [{}] not found in: {}", keyValueDelimiter, line));
        }

        int columnsInRow = std::ranges::distance(std::ranges::istream_view<std::string>(vectorStr) | std::views::split(vectorDelimiter));

        if (columns == 0) {
            columns = columnsInRow;
        } else {
            if (columns != columnsInRow) {
                throw std::invalid_argument(std::format("Inconsistent num columns detected. Expected: {} Actual: {}", columns, columnsInRow));
            }
        }
    }

    fs.close();
}

/**
 * @brief Read the next row of data from the text file.
 *
 * @param key [out] Pointer to store the key value of the row.
 * @param vector [out] Span to store the vector elements of the row.
 *
 * @return true if a row was successfully read, false if end of file is reached.
 *
 * @throw std::invalid_argument if a malformed vector element is encountered.
 * @throw std::invalid_argument if a vector element cannot be parsed as float.
 */
bool TextFileDataReader::readRow(std::string* key, std::span<float> vector) {
    std::string line;
    if (std::getline(fileStream, line)) {
        std::string vectorStr;
        splitKeyVector(line, *key, vectorStr, keyValueDelimiter);

        std::stringstream vectorStrStream(vectorStr);

        for (int i = 0; std::getline(vectorStrStream, vector[i], vectorDelimiter); ++i) {
            try {
                if (auto [p, ec] = std::from_chars(vector[i].data(), vector[i].data() + vector[i].size(), vector[i]); ec != std::errc()) {
                    throw std::invalid_argument(std::format("Malformed vector element: {}", vector[i]));
                }
            } catch (const std::exception& e) {
                throw std::invalid_argument(std::format("ERROR: {} cannot be parsed as float. Column {} of: {}", vector[i], i, line));
            }
        }

        return true;
    } else {
        return false;
    }
}

/**
 * @brief Destroy the TextFileDataReader object and close the file stream.
 */
TextFileDataReader::~TextFileDataReader() {
    fileStream.close();
}
