package com.system.tensorhub.knn

import java.util.Objects

/**
 * Represents a vector.
 */
data class Vector(
    val index: Int,
    val coordinates: List<Float>
) {
    init {
        require(index >= 0) { "Index must be non-negative" }
    }

    /**
     * Computes the hash code of the vector.
     *
     * @return The computed hash code.
     */
    override fun hashCode(): Int {
        return Objects.hash(index, coordinates)
    }

    /**
     * Checks if this Vector is equal to the specified object.
     *
     * @param other The object to compare.
     * @return True if the objects are equal, false otherwise.
     */
    override fun equals(other: Any?): Boolean {
        if (this === other) {
            return true
        }
        if (other !is Vector) {
            return false
        }

        return index == other.index && coordinates == other.coordinates
    }

    /**
     * Returns a string representation of the Vector.
     *
     * @return The string representation of the Vector.
     */
    override fun toString(): String {
        return "Vector(index=$index, coordinates=$coordinates)"
    }

    /**
     * Builder class for constructing Vector instances.
     */
    companion object {
        fun builder(): Builder {
            return Builder()
        }
    }

    /**
     * Builder class for constructing Vector instances.
     */
    class Builder {
        private var index: Int = 0
        private var coordinates: List<Float> = emptyList()

        /**
         * Sets the index of the vector.
         *
         * @param index The index value.
         * @return The Builder instance.
         */
        fun withIndex(index: Int): Builder {
            this.index = index
            return this
        }

        /**
         * Sets the coordinates of the vector.
         *
         * @param coordinates The list of coordinates.
         * @return The Builder instance.
         */
        fun withCoordinates(coordinates: List<Float>): Builder {
            this.coordinates = coordinates.toList()
            return this
        }

        /**
         * Builds a new Vector instance.
         *
         * @return The newly created Vector instance.
         */
        fun build(): Vector {
            return Vector(index, coordinates)
        }
    }
}
