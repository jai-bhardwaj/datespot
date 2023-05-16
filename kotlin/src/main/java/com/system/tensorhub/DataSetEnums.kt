package com.system.tensorhub

/**
 * Contains the enumerations and utility functions related to data sets.
 */
class DataSetEnums {
    /**
     * Represents the attribute types of a data set.
     */
    object Attribute {
        /**
         * No special attribute.
         */
        const val None = 0x0

        /**
         * Sparse attribute.
         */
        const val Sparse = 0x1

        /**
         * Boolean attribute.
         */
        const val Boolean = 0x2

        /**
         * Compressed attribute.
         */
        const val Compressed = 0x4

        /**
         * Recurrent attribute.
         */
        const val Recurrent = 0x8

        /**
         * Mutable attribute.
         */
        const val Mutable = 0xF

        /**
         * Sparse attribute with zero values ignored.
         */
        const val SparseIgnoreZero = 0x20

        /**
         * Indexed attribute.
         */
        const val Indexed = 0x40

        /**
         * Weighted attribute.
         */
        const val Weighted = 0x80
    }

    /**
     * Represents the kinds of data sets.
     */
    enum class Kind {
        /**
         * Numeric data set.
         */
        Numeric,

        /**
         * Image data set.
         */
        Image,

        /**
         * Audio data set.
         */
        Audio
    }

    /**
     * Represents the sharding options for data sets.
     */
    enum class Sharding {
        /**
         * No sharding.
         */
        None,

        /**
         * Model sharding.
         */
        Model,

        /**
         * Data sharding.
         */
        Data
    }

    /**
     * Represents the data types used in the data sets.
     */
    enum class DataType {
        /**
         * Unsigned integer data type.
         */
        UInt,

        /**
         * Integer data type.
         */
        Int,

        /**
         * Long long integer data type.
         */
        LLInt,

        /**
         * Unsigned long long integer data type.
         */
        ULLInt,

        /**
         * Floating-point data type.
         */
        Float,

        /**
         * Double precision floating-point data type.
         */
        Double,

        /**
         * RGB 8-bit data type.
         */
        RGB8,

        /**
         * RGB 16-bit data type.
         */
        RGB16,

        /**
         * Unsigned character data type.
         */
        UChar,

        /**
         * Character data type.
         */
        Char;

        companion object {
            /**
             * Returns the size in bytes of the specified data type.
             *
             * @param dataType The data type.
             * @return The size in bytes.
             * @throws IllegalArgumentException if the data type is not supported.
             */
            fun sizeof(dataType: DataType): Int {
                return when (dataType) {
                    DataType.Int -> Integer.SIZE / 8
                    DataType.LLInt -> Long.SIZE / 8
                    DataType.Float -> java.lang.Float.SIZE / 8
                    DataType.Double -> java.lang.Double.SIZE / 8
                    DataType.Char -> Character.SIZE / 8
                    else -> throw IllegalArgumentException("$dataType not supported in Kotlin binding")
                }
            }
        }
    }
}
