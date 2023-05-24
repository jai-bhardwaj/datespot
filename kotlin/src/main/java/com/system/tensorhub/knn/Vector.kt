package com.system.tensorhub.knn

import java.util.*

/**
 * Represents a vector.
 */
class Vector private constructor() {
    private var index: Int = 0
    private var coordinates: List<Float>? = null

    /**
     * Gets the index of the vector.
     *
     * @return The index.
     */
    fun getIndex(): Int {
        return index
    }

    /**
     * Sets the index of the vector.
     *
     * @param index The index to set.
     */
    fun setIndex(index: Int) {
        this.index = index
    }

    /**
     * Gets the coordinates of the vector.
     *
     * @return The coordinates.
     */
    fun getCoordinates(): List<Float>? {
        return coordinates
    }

    /**
     * Sets the coordinates of the vector.
     *
     * @param coordinates The coordinates to set.
     */
    fun setCoordinates(coordinates: List<Float>) {
        this.coordinates = coordinates
    }

    /**
     * Computes the hash code of the vector.
     *
     * @return The hash code.
     */
    override fun hashCode(): Int {
        return internalHashCodeCompute(classNameHashCode, getIndex(), getCoordinates())
    }

    /**
     * Checks if this vector is equal to another object.
     *
     * @param other The object to compare.
     * @return `true` if the objects are equal, `false` otherwise.
     */
    override fun equals(other: Any?): Boolean {
        if (this === other) {
            return true
        }
        if (other !is Vector) {
            return false
        }
        val that = other as Vector?
        return getIndex() == that!!.getIndex() && Objects.equals(getCoordinates(), that.getCoordinates())
    }

    /**
     * Returns a string representation of the vector.
     *
     * @return The string representation.
     */
    override fun toString(): String {
        val ret = StringBuilder()
        ret.append("Vector(")
        ret.append("index=")
        ret.append(index)
        ret.append(", ")
        ret.append("coordinates=")
        ret.append(coordinates)
        ret.append(")")
        return ret.toString()
    }

    companion object {
        /**
         * Creates a new instance of the Vector.Builder.
         *
         * @return The Vector.Builder instance.
         */
        fun builder(): Builder {
            return Builder()
        }

        private const val classNameHashCode = internalHashCodeCompute("com.system.tensorhub.knn.Vector")

        private fun internalHashCodeCompute(vararg objects: Any): Int {
            return Arrays.hashCode(objects)
        }
    }

    /**
     * Builder for constructing Vector instances.
     */
    class Builder {
        private var index: Int = 0

        /**
         * Sets the index of the vector being built.
         *
         * @param index The index to set.
         * @return This builder.
         */
        fun withIndex(index: Int): Builder {
            this.index = index
            return this
        }

        private var coordinates: List<Float>? = null

        /**
         * Sets the coordinates of the vector being built.
         *
         * @param coordinates The coordinates to set.
         * @return This builder.
         */
        fun withCoordinates(coordinates: List<Float>): Builder {
            this.coordinates = coordinates
            return this
        }

        /**
         * Populates the fields of the Vector instance being built.
         *
         * @param instance The Vector instance to populate.
         */
        internal fun populate(instance: Vector) {
            instance.setIndex(index)
            instance.setCoordinates(coordinates!!)
        }

        /**
         * Builds a new Vector instance based on the builder's configuration.
         *
         * @return The constructed Vector instance.
         */
        fun build(): Vector {
            val instance = Vector()
            populate(instance)
            return instance
        }
    }
}
