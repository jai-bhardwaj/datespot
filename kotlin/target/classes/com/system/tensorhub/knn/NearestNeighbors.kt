package com.system.tensorhub.knn

import java.util.Arrays
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

        internal fun populate(instance: NearestNeighbors) {
            instance.index = this.index
            instance.neighbors = this.neighbors
        }

        /**
         * Builds and returns a NearestNeighbors object.
         *
         * @return The constructed NearestNeighbors object.
         */
        fun build(): NearestNeighbors {
            val instance = NearestNeighbors()
            populate(instance)
            return instance
        }
    }

    /**
     * Calculates the hash code for the NearestNeighbors object.
     *
     * @return The calculated hash code.
     */
    override fun hashCode(): Int {
        return internalHashCodeCompute(
            classNameHashCode,
            index,
            neighbors
        )
    }

    private fun internalHashCodeCompute(vararg objects: Any?): Int {
        return Arrays.hashCode(objects)
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

        return Objects.equals(index, that.index) &&
                Objects.equals(neighbors, that.neighbors)
    }

    /**
     * Returns a string representation of the NearestNeighbors object.
     *
     * @return The string representation.
     */
    override fun toString(): String {
        val ret = StringBuilder()
        ret.append("NearestNeighbors(")

        ret.append("index=")
        ret.append(index.toString())
        ret.append(", ")

        ret.append("neighbors=")
        ret.append(neighbors.toString())
        ret.append(")")

        return ret.toString()
    }

    companion object {
        private val classNameHashCode =
            internalHashCodeCompute("com.system.tensorhub.knn.NearestNeighbors")

        /**
         * Returns a new instance of the Builder class for constructing NearestNeighbors objects.
         *
         * @return The Builder instance.
         */
        fun builder(): Builder {
            return Builder()
        }

        private fun internalHashCodeCompute(vararg objects: Any?): Int {
            return Arrays.hashCode(objects)
        }
    }
}
