package com.system.tensorhub.knn

import java.util.Arrays
import java.util.Objects

/**
 * Represents a Nearest Neighbors instance.
 */
class NearestNeighbors private constructor() {
    var index: Int = 0
        private set
    var neighbors: List<Neighbor>? = null
        private set

    /**
     * Builder class for creating NearestNeighbors instances.
     */
    class Builder {
        private var index: Int = 0
        private var neighbors: List<Neighbor>? = null

        /**
         * Sets the index value for the NearestNeighbors instance being built.
         * @param index The index value.
         * @return The Builder instance.
         */
        fun withIndex(index: Int): Builder {
            this.index = index
            return this
        }

        /**
         * Sets the neighbors for the NearestNeighbors instance being built.
         * @param neighbors The neighbors list.
         * @return The Builder instance.
         */
        fun withNeighbors(neighbors: List<Neighbor>): Builder {
            this.neighbors = neighbors
            return this
        }

        internal fun populate(instance: NearestNeighbors) {
            instance.setIndex(this.index)
            instance.setNeighbors(this.neighbors)
        }

        /**
         * Builds the NearestNeighbors instance.
         * @return The built NearestNeighbors instance.
         */
        fun build(): NearestNeighbors {
            val instance = NearestNeighbors()
            populate(instance)
            return instance
        }
    }

    /**
     * Gets the index value.
     * @return The index value.
     */
    fun getIndex(): Int {
        return index
    }

    private fun setIndex(index: Int) {
        this.index = index
    }

    /**
     * Gets the neighbors list.
     * @return The neighbors list.
     */
    fun getNeighbors(): List<Neighbor>? {
        return neighbors
    }

    private fun setNeighbors(neighbors: List<Neighbor>?) {
        this.neighbors = neighbors
    }

    override fun hashCode(): Int {
        return internalHashCodeCompute(
            classNameHashCode,
            getIndex(),
            getNeighbors()
        )
    }

    private fun internalHashCodeCompute(vararg objects: Any): Int {
        return Arrays.hashCode(objects)
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) {
            return true
        }
        if (other !is NearestNeighbors) {
            return false
        }
        val that = other as NearestNeighbors?
        return (getIndex() == that!!.getIndex()
                && Objects.equals(getNeighbors(), that.getNeighbors()))
    }

    override fun toString(): String {
        val ret = StringBuilder()
        ret.append("NearestNeighbors(")
        ret.append("index=")
        ret.append(index)
        ret.append(", ")
        ret.append("neighbors=")
        ret.append(neighbors)
        ret.append(")")
        return ret.toString()
    }

    companion object {
        private val classNameHashCode: Int =
            internalHashCodeCompute("com.system.tensorhub.knn.NearestNeighbors")

        /**
         * Returns a new Builder instance to build NearestNeighbors objects.
         * @return A new Builder instance.
         */
        fun builder(): Builder {
            return Builder()
        }
    }
}
