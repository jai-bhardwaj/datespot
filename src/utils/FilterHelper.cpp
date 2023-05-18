#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <charconv>
#include <filesystem>

#include <json/json.h>

namespace fs = std::filesystem;

/**
 * @brief Read lines from a file and return them as a vector of strings.
 *
 * @param filePath The path to the file.
 * @return A vector of strings containing the lines from the file.
 * @throws std::runtime_error If the file cannot be opened.
 */
[[nodiscard]] std::vector<std::string> readLinesFromFile(const fs::path& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filePath.string());
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    return lines;
}

/**
 * @brief Main function that serves as the entry point of the program.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of C-strings containing the command-line arguments.
 * @return The exit status of the program.
 */
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <filename>\n";
        return 1;
    }

    fs::path filePath(argv[1]);

    try {
        auto lines = readLinesFromFile(filePath);
        for (const auto& line : lines) {
            // Process each line
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
