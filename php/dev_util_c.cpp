#include <string_view>
#include <unordered_map>

#include "dev_util_c.h"

/**
 * Retrieves the error message associated with the given error code.
 *
 * @param code The error code for which the error message is requested.
 * @return A string_view containing the error message.
 *
 * @note If the error code is not found, an empty string_view is returned.
 */
std::string_view getErrorMsg(int code) {
    static std::unordered_map<int, std::string_view> errorCodeMap = {
        { 1000, "CUDA error. Code %d (File:%s Line: %d)\n \n" },
        { 1001, "There are no available device(s) that support CUDA \n" },
        { 2000, "the width of martrixA( %d, %d ) can not match with the height of tensorhubB( %d, %d )\n" },
        { 2001, "tensorhubArrA must be a two-dimensional array\n" },
        { 2002, "tensorhubArrB must be a two-dimensional array\n" },
        { 2003, "tensorhubArrC must be a two-dimensional array\n" },
        { 2004, "the height of martrixA( %d, %d ) can not match with the height of tensorhubC( %d, %d )\n" },
        { 2005, "the width of martrixB( %d, %d ) can not match with the width of widthC( %d, %d )\n" },
        { 2006, "the width of martrixB( %d, %d ) can not match with the width of widthC( %d, %d )\n" },
        { 2007, "elementNumX must be greater or equal to ( elementNumX < 1 + ( heightA - 1 ) * std::abs( (int)strideX ) )\n" },
        { 2008, "%s must be a two-dimensional array\n" }
        // Add more error codes and messages as needed
    };

    auto it = errorCodeMap.find(code);
    if (it != errorCodeMap.end()) {
        return it->second;
    }

    return "Unknown error code\n";
}
