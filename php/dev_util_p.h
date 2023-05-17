#include <exception>
#include <concepts>
#include <type_traits>
#include <string_view>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <ranges>
#include <vector>

#include <zend_exceptions.h>
#include "dev_util_c.h"

/**
 * @brief Exception for CUDA errors.
 * 
 * The `CudaException` class represents an exception that is thrown in case of a CUDA error.
 * It stores information about the error code, the file where the error occurred, and the line number.
 */
class CudaException : public std::exception {
public:
    /**
     * @brief Constructs a `CudaException` object.
     * 
     * @param error The error code.
     * @param file The file where the error occurred.
     * @param line The line number where the error occurred.
     */
    CudaException(int error, const char* file, int line)
        : error_(error), file_(file), line_(line)
    {}

    /**
     * @brief Returns the error code.
     * 
     * @return The error code.
     */
    int getError() const noexcept { return error_; }

    /**
     * @brief Returns a message describing the error.
     * 
     * @return A message describing the error.
     */
    const char* what() const noexcept override {
        return duc_getErrorMsg(1000);
    }

    /**
     * @brief Returns the file where the error occurred.
     * 
     * @return The file where the error occurred.
     */
    const char* getFile() const noexcept { return file_; }

    /**
     * @brief Returns the line number where the error occurred.
     * 
     * @return The line number where the error occurred.
     */
    int getLine() const noexcept { return line_; }

private:
    int error_;
    const char* file_;
    int line_;
};

/**
 * @brief Concept for floating point types.
 * 
 * The `Float` concept checks if the given type is a floating point type.
 * 
 * @tparam T The type to check.
 */
template <typename T>
concept Float = std::is_floating_point_v<T>;

/**
 * @brief Concept for `double` type.
 * 
 * The `Double` concept checks if the given type is `double`.
 * 
 * @tparam T The type to check.
 */
template <typename T>
concept Double = std::is_same_v<T, double>;

/**
 * @brief Concept for CUDA floating point types.
 * 
 * The `CudaFloatType` concept checks if the given type is a floating point type or `double`.
 * 
 * @tparam T The type to check.
 */
template <typename T>
concept CudaFloatType = Float<T> || Double<T>;

/**
 * @brief Checks a CUDA result for an error.
 * 
 * This function takes a CUDA result and file/line information and throws a `CudaException`
 * if the result is not equal to 0.
 * 
 * @tparam CudaFloatType The type of the CUDA result (should be a floating point type or `double`).
 * @param result The CUDA result to check.
 * @param file The file where the result was generated.
 * @param line The line number where the result was generated.
 */
template <CudaFloatType T>
void checkCudaResult(T result, const char* file, int line) {
    if (result != 0) {
        throw CudaException(result, file, line);
    }
}

/**
 * @brief Concept for CUDA array types.
 * 
 * The `CudaArrayType` concept checks if the given type is either a `float*` or a `double*`.
 * 
 * @tparam T The type to check.
 */
template <typename T>
concept CudaArrayType = std::is_same_v<T, float*> || std::is_same_v<T, double*>;

/**
 * @brief Converts a hash table to a 1D array with floating point values.
 * 
 * This function takes a pointer to a `HashTable` and a `std::span` of a floating point type,
 * and fills the `std::span` with the values from the hash table.
 * 
 * @tparam FloatingPoint The type of floating point values to use (e.g., `float` or `double`).
 * @param hashTableP Pointer to the hash table to convert.
 * @param arrP The `std::span` to fill with the values from the hash table.
 */
template<FloatingPoint T>
void dup_HashTableTo1DArr(HashTable* hashTableP, std::span<T> arrP);

/**
 * @brief Converts a hash table to a 1D array with floating point values.
 * 
 * This function takes a pointer to a `HashTable` and a pointer to a floating point value,
 * and fills the array with the values from the hash table.
 * 
 * @tparam FloatingPoint The type of floating point values to use (e.g., `float` or `double`).
 * @param hashTableP Pointer to the hash table to convert.
 * @param arrP Pointer to the array to fill with the values from the hash table.
 */
template<FloatingPoint T>
void dup_HashTableTo1DArrOne(HashTable* hashTableP, T* arrP);

/**
 * @brief Converts a hash table to a 1D `zval` array with integer values.
 * 
 * This function takes a pointer to a `HashTable`, a `zval` array, and a `std::span` of integers,
 * and fills the `zval` array with the values from the hash table. The `std::span`
 * of integers holds information about the shape of the resulting array.
 * 
 * @param hashTableP Pointer to the hash table to convert.
 * @param oneDimensionzval The `zval` array to fill with the values from the hash table.
 * @param shapeInfo The `std::span` of integers holding information about the shape of the resulting array.
 */
void dup_hashTableTo1DZval(HashTable* hashTableP, zval oneDimensionzval, std::span<int> shapeInfo);

/**
 * @brief Reshapes a 1D pointer array to a `zval` array.
 * 
 * This function takes a pointer to a 1D pointer array with floating point values, a `zval` array,
 * a `std::span` of integers, and an integer, and fills the `zval` array with the values from the
 * 1D pointer array, reshaped according to the information in the `std::span` of integers.
 * The integer parameter `previousCount` holds the size of the 1D pointer array.
 * 
 * @tparam FloatingPoint The type of floating point values to use (e.g., `float` or `double`).
 * @param arrP Pointer to the 1D pointer array to reshape.
 * @param reshapedZval The `zval` array to fill with the reshaped values.
 * @param shapeInfo The `std::span` of integers holding information about the shape of the resulting array.
 * @param previousCount The size of the 1D pointer array.
 */
template<FloatingPoint T>
void dup_oneDimesnPointerArrReshapeToZval(T* arrP, zval reshapedZval, std::span<int> shapeInfo, int previousCount);

/**
 * @brief Converts a 1D `zval` array to a pointer array with floating point values.
 * 
 * This function takes a `std::span` of `zval` arrays and a pointer to a floating point value,
 * and fills the array with the values from the `zval` arrays.
 * 
 * @tparam FloatingPoint The type of floating point values to use (e.g., `float` or `double`).
 * @param oneDimensionZavl The `std::span` of `zval` arrays to convert.
 * @param arrP Pointer to the array to fill with the values from the `zval` arrays.
 */
template<FloatingPoint T>
void dup_oneDimensionZavlToPointerArr(std::span<zval> oneDimensionZavl, T* arrP);
