package com.system.tensorhub

/**
 * Provides enums and utility functions for data sets.
 */
object DataSetEnums {
    /**
     * Defines attributes for data sets.
     */
    object Attribute {
        const val None = 0x0
        const val Sparse = 0x1
        const val Boolean = 0x2
        const val Compressed = 0x4
        const val Recurrent = 0x8
        const val Mutable = 0xF
        const val SparseIgnoreZero = 0x20
        const val Indexed = 0x40
        const val Weighted = 0x80
    }

    /**
     * Defines the kinds of data sets.
     */
    enum class Kind {
        Numeric,
        Image,
        Audio
    }

    /**
     * Defines sharding options for data sets.
     */
    enum class Sharding {
        None,
        Model,
        Data
    }

    /**
     * Defines data types for data sets.
     */
    enum class DataType {
        UInt,
        Int,
        LLInt,
        ULLInt,
        Float,
        Double,
        RGB8,
        RGB16,
        UChar,
        Char;

        companion object {
            /**
             * Returns the size (in bytes) of the specified data type.
             *
             * @param dataType The data type.
             * @return The size in bytes.
             */
            fun sizeof(dataType: DataType): Int {
                return when (dataType) {
                    DataType.Int -> Integer.SIZE / 8
                    DataType.LLInt -> Long.SIZE / 8
                    DataType.Float -> java.lang.Float.SIZE / 8
                    DataType.Double -> java.lang.Double.SIZE / 8
                    DataType.Char -> Character.SIZE / 8
                    else -> throw IllegalArgumentException("$dataType not supported in java binding")
                }
            }
        }
    }
}
