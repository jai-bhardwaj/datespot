package com.system.tensorhub

/**
 * Contains the enumerations and utility functions related to data sets.
 */
object DataSetEnums {
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
    enum class DataType(val sizeInBytes: Int) {
        /**
         * Unsigned integer data type.
         */
        UInt(Integer.SIZE_BYTES),

        /**
         * Integer data type.
         */
        Int(Integer.SIZE_BYTES),

        /**
         * Long long integer data type.
         */
        LLInt(Long.SIZE_BYTES),

        /**
         * Unsigned long long integer data type.
         */
        ULLInt(Long.SIZE_BYTES),

        /**
         * Floating-point data type.
         */
        Float(java.lang.Float.SIZE_BYTES),

        /**
         * Double precision floating-point data type.
         */
        Double(java.lang.Double.SIZE_BYTES),

        /**
         * RGB 8-bit data type.
         */
        RGB8(Byte.SIZE_BYTES * 3),

        /**
         * RGB 16-bit data type.
         */
        RGB16(Short.SIZE_BYTES * 3),

        /**
         * Unsigned character data type.
         */
        UChar(Byte.SIZE_BYTES),

        /**
         * Character data type.
         */
        Char(Byte.SIZE_BYTES);

        companion object {
            /**
             * Returns the size in bytes of the specified data type.
             *
             * @param dataType The data type.
             * @return The size in bytes.
             * @throws IllegalArgumentException if the data type is not supported.
             */
            fun sizeof(dataType: DataType): Int = dataType.sizeInBytes
        }
    }
}
