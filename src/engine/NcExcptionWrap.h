#pragma once

#include <string>
#include <source_location>

/**
 * @class NcException
 * @brief Represents an exception in the program.
 */
class NcException {
public:
    /**
     * @brief Constructs a new NcException object.
     * @param message The error message.
     * @param filename The name of the source file where the exception occurred.
     * @param line The line number where the exception occurred.
     */
    NcException(const char* message, const std::string& filename, int line) {
        // constructor implementation
    }
};

/**
 * @brief Macro for creating an NcException object with the error message, source file, and line number.
 * @param errorStr The error string.
 * @param msg The error message.
 * @return NcException An NcException object initialized with the provided parameters.
 */
#define NC_EXCEPTION(errorStr, msg) NcException(std::string(msg).c_str(), std::source_location::current().file_name(), std::source_location::current().line())