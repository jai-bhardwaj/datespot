package com.system.tensorhub.knn

/**
 * Represents the data type for a tensor.
 */
enum class DataType {
    /**
     * 32-bit floating point data type.
     */
    FP32,

    /**
     * 16-bit floating point data type.
     */
    FP16;

    companion object {
        /**
         * Converts a string representation of the data type to the corresponding enum value.
         *
         * @param dt The string representation of the data type.
         * @return The corresponding [DataType] enum value.
         */
        fun fromString(dt: String): DataType = valueOf(dt.toUpperCase())
    }
}
