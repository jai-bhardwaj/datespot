#ifndef NC_EXCEPTION_H
#define NC_EXCEPTION_H

#include <string>
#include <string_view>

/**
 * @brief The NC_EXCEPTION macro creates a lambda function that throws an instance of NcException.
 *
 * The macro takes the error string, error message, filename, and line number as arguments
 * and returns a lambda function that throws an instance of NcException.
 *
 * @param errorStr The error string as a std::string_view.
 * @param msg The error message as a std::string.
 * @param filename The filename where the exception occurred as a std::string.
 * @param line The line number where the exception occurred as an integer.
 * @return A lambda function that throws an instance of NcException.
 */
auto NC_EXCEPTION = [](const std::string_view& errorStr, const std::string& msg, const std::string& filename, int line) {
    return NcException(msg.c_str(), filename, line);
};

#endif // NC_EXCEPTION_H
