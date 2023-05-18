#ifndef LIBKNN_DATAREADER_H_
#define LIBKNN_DATAREADER_H_

#include <cstdint>
#include <fstream>
#include <string_view>
#include <filesystem>

/**
 * @brief Abstract base class for data readers.
 */
class DataReader
{
public:
    /**
     * @brief Reads a row of data.
     *
     * @param key The key associated with the row.
     * @param vector Pointer to the vector data.
     * @return true if the row was read successfully, false otherwise.
     */
    [[nodiscard]] virtual bool readRow(std::string_view key, float* vector) = 0;

    /**
     * @brief Gets the number of rows in the data.
     *
     * @return The number of rows.
     */
    [[nodiscard]] uint32_t getRows() const;

    /**
     * @brief Gets the number of columns in the data.
     *
     * @return The number of columns.
     */
    [[nodiscard]] int getColumns() const;

    /**
     * @brief Destructor.
     */
    virtual ~DataReader() = default;

protected:
    uint32_t rows; ///< Number of rows in the data.
    int columns;   ///< Number of columns in the data.
};

/**
 * @brief Data reader for text files.
 */
class TextFileDataReader : public DataReader
{
public:
    /**
     * @brief Constructs a TextFileDataReader object.
     *
     * @param fileName The path to the text file.
     * @param keyValueDelimiter The delimiter between the key and vector in each row.
     * @param vectorDelimiter The delimiter between elements in the vector.
     */
    TextFileDataReader(const std::filesystem::path& fileName, char keyValueDelimiter = '\t', char vectorDelimiter = ' ');

    /**
     * @brief Reads a row of data from the text file.
     *
     * @param key The key associated with the row.
     * @param vector Pointer to the vector data.
     * @return true if the row was read successfully, false otherwise.
     */
    [[nodiscard]] bool readRow(std::string_view key, float* vector) override;

    /**
     * @brief Determines the dimensions of the data in the text file.
     *
     * @param fileName The path to the text file.
     * @param rows Reference to store the number of rows.
     * @param columns Reference to store the number of columns.
     * @param keyValueDelimiter The delimiter between the key and vector in each row.
     * @param vectorDelimiter The delimiter between elements in the vector.
     */
    static void findDataDimensions(const std::filesystem::path& fileName, uint32_t& rows, int& columns,
                                   char keyValueDelimiter = '\t', char vectorDelimiter = ' ');

    /**
     * @brief Destructor.
     */
    ~TextFileDataReader() = default;

private:
    std::filesystem::path fileName; ///< Path to the text file.
    std::ifstream fileStream;       ///< File stream for reading the text file.
    char keyValueDelimiter;         ///< Delimiter between the key and vector in each row.
    char vectorDelimiter;           ///< Delimiter between elements in the vector.
};

#endif
