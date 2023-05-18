package com.system.tensorhub.knn

import java.util.Objects

/**
 * Represents a Nearest Neighbors object.
 */
class NearestNeighbors private constructor() {
    var index: Int = 0
        private set
    var neighbors: List<Neighbor>? = null
        private set

    /**
     * Builder class for constructing NearestNeighbors objects.
     */
    class Builder {
        private var index: Int = 0
        private var neighbors: List<Neighbor>? = null

        /**
         * Sets the index for the NearestNeighbors object being built.
         *
         * @param index The index value.
         * @return The Builder object.
         */
        fun withIndex(index: Int): Builder {
            this.index = index
            return this
        }

        /**
         * Sets the list of neighbors for the NearestNeighbors object being built.
         *
         * @param neighbors The list of neighbors.
         * @return The Builder object.
         */
        fun withNeighbors(neighbors: List<Neighbor>): Builder {
            this.neighbors = neighbors
            return this
        }

        /**
         * Builds and returns a NearestNeighbors object.
         *
         * @return The constructed NearestNeighbors object.
         */
        fun build(): NearestNeighbors {
            val instance = NearestNeighbors()
            instance.index = this.index
            instance.neighbors = this.neighbors
            return instance
        }
    }

    /**
     * Calculates the hash code for the NearestNeighbors object.
     *
     * @return The calculated hash code.
     */
    override fun hashCode(): Int {
        return Objects.hash(index, neighbors)
    }

    /**
     * Checks if the NearestNeighbors object is equal to another object.
     *
     * @param other The object to compare for equality.
     * @return True if the NearestNeighbors is equal to the other object, false otherwise.
     */
    override fun equals(other: Any?): Boolean {
        if (this === other) {
            return true
        }
        if (other !is NearestNeighbors) {
            return false
        }
        val that = other as NearestNeighbors
        return index == that.index &&
                neighbors == that.neighbors
    }

    /**
     * Returns a string representation of the NearestNeighbors object.
     *
     * @return The string representation.
     */
    override fun toString(): String {
        return "NearestNeighbors(index=$index, neighbors=$neighbors)"
    }

    companion object {
        /**
         * Returns a new instance of the Builder class for constructing NearestNeighbors objects.
         *
         * @return The Builder instance.
         */
        fun builder(): Builder {
            return Builder()
        }
    }
}
