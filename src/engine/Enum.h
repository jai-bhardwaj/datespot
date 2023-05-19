#ifndef ENUM_H
#define ENUM_H

#include <stdexcept>

/**
 * @brief Enumerations related to data attributes.
 */
enum class DataAttribute
{
    Sparse = 1,                /**< Sparse data attribute */
    Boolean = 2,               /**< Boolean data attribute */
    Compressed = 4,            /**< Compressed data attribute */
    Recurrent = 8,             /**< Recurrent data attribute */
    Mutable = 16,              /**< Mutable data attribute */
    SparseIgnoreZero = 32,     /**< Sparse data attribute ignoring zero values */
    Indexed = 64,              /**< Indexed data attribute */
    Weighted = 128             /**< Weighted data attribute */
};

/**
 * @brief Enumerations related to data kinds.
 */
enum class DataKind
{
    Numeric = 0,   /**< Numeric data kind */
    Image = 1,     /**< Image data kind */
    Audio = 2      /**< Audio data kind */
};

/**
 * @brief Enumerations related to sharding.
 */
enum class Sharding
{
    None = 0,   /**< No sharding */
    Model = 1,  /**< Sharding based on the model */
    Data = 2    /**< Sharding based on the data */
};

/**
 * @brief Enumerations for different data types.
 */
enum class DataType
{
    UInt = 0,     /**< Unsigned integer data type */
    Int = 1,      /**< Integer data type */
    LLInt = 2,    /**< Long long integer data type */
    ULLInt = 3,   /**< Unsigned long long integer data type */
    Float = 4,    /**< Float data type */
    Double = 5,   /**< Double data type */
    RGB8 = 6,     /**< RGB 8-bit data type */
    RGB16 = 7,    /**< RGB 16-bit data type */
    UChar = 8,    /**< Unsigned char data type */
    Char = 9      /**< Char data type */
};

/**
 * @brief Template function to retrieve the data type based on a given type.
 *
 * @tparam T The type for which to retrieve the data type.
 * @return The data type associated with the given type.
 *
 * @throws std::runtime_error If the data type is not defined.
 */
template<typename T>
inline DataType getDataType()
{
    static_assert(false, "Default data type not defined");
}

template<>
inline DataType getDataType<uint32_t>()
{
    return DataType::UInt;
}

template<>
inline DataType getDataType<int32_t>()
{
    return DataType::Int;
}

template<>
inline DataType getDataType<int64_t>()
{
    return DataType::LLInt;
}

template<>
inline DataType getDataType<uint64_t>()
{
    return DataType::ULLInt;
}

template<>
inline DataType getDataType<float>()
{
    return DataType::Float;
}

template<>
inline DataType getDataType<double>()
{
    return DataType::Double;
}

template<>
inline DataType getDataType<char>()
{
    return DataType::Char;
}

template<>
inline DataType getDataType<unsigned char>()
{
    return DataType::UChar;
}

#endif // ENUM_H
