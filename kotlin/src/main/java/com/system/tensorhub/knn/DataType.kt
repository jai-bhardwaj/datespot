package com.system.tensorhub.knn

/**
 * Enum representing different data types.
 */
enum class DataType {
    /**
     * Single-precision floating-point data type.
     */
    FP32,

    /**
     * Half-precision floating-point data type.
     */
    FP16;

    companion object {
        /**
         * Converts a string representation of a data type to the corresponding DataType enum value.
         *
         * @param dt The string representation of the data type.
         * @return The DataType enum value corresponding to the input string.
         * @throws IllegalArgumentException if the input string does not match any DataType enum value.
         */
        fun fromString(dt: String): DataType {
            return valueOf(dt.toUpperCase())
        }
    }
}
