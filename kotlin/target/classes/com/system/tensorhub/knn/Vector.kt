package com.system.tensorhub.knn;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Represents a vector.
 */
class Vector private constructor() {
    var index: Int = 0
        private set
    var coordinates: List<Float>? = null
        private set

    /**
     * Creates a new instance of the Vector.Builder.
     *
     * @return The Vector.Builder instance.
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
        private var coordinates: List<Float>? = null

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
            this.coordinates = coordinates
            return this
        }

        private fun populate(instance: Vector) {
            instance.index = this.index
            instance.coordinates = this.coordinates
        }

        /**
         * Builds a new Vector instance.
         *
         * @return The newly created Vector instance.
         */
        fun build(): Vector {
            val instance = Vector()

            populate(instance)

            return instance
        }
    }

    ;

    private var index: Int = 0
    private var coordinates: List<Float>? = null
    
    /**
     * Gets the index of the vector.
     *
     * @return The index value.
     */
    fun getIndex(): Int {
        return index
    }
    
    /**
     * Sets the index of the vector.
     *
     * @param index The index value.
     */
    fun setIndex(index: Int) {
        this.index = index
    }
    
    /**
     * Gets the coordinates of the vector.
     *
     * @return The list of coordinates.
     */
    val coordinates: List<Float>
        get() = this.coordinates
    
    /**
     * Sets the coordinates of the vector.
     *
     * @param coordinates The list of coordinates.
     */
    fun setCoordinates(coordinates: List<Float>) {
        this.coordinates = coordinates
    }
    
    private val classNameHashCode = internalHashCodeCompute("com.system.tensorhub.knn.Vector")
    
    /**
     * Computes the hash code of the vector.
     *
     * @return The computed hash code.
     */
    override fun hashCode(): Int {
        return internalHashCodeCompute(
            classNameHashCode,
            getIndex(),
            getCoordinates()
        )
    }

    /**
     * Computes the hash code based on the provided objects.
     *
     * @param objects The objects used to compute the hash code.
     * @return The computed hash code.
     */
    private fun internalHashCodeCompute(vararg objects: Any): Int {
        return objects.contentHashCode()
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

        val that = other as Vector

        return getIndex() == that.getIndex() && getCoordinates() == that.getCoordinates()
    }

    /**
     * Returns a string representation of the Vector.
     *
     * @return The string representation of the Vector.
     */
    override fun toString(): String {
        val ret = StringBuilder()
        ret.append("Vector(")

        ret.append("index=")
        ret.append(index.toString())
        ret.append(", ")

        ret.append("coordinates=")
        ret.append(coordinates.toString())
        ret.append(")")

        return ret.toString()
    }    
}
