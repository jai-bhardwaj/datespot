package com.system.tensorhub.knn

import java.util.Arrays
import java.util.Objects

class Neighbor private constructor() {
    /**
     * The ID of the neighbor.
     */
    var id: String? = null
        private set
    
    /**
     * The score of the neighbor.
     */
    var score: Float = 0f
        private set

    companion object {
        /**
         * Creates a new instance of the Neighbor.Builder.
         *
         * @return the Neighbor.Builder instance
         */
        fun builder(): Builder {
            return Builder()
        }
    }

    /**
     * The builder class for Neighbor.
     */
    class Builder {
        private var id: String? = null
        private var score: Float = 0f

        /**
         * Sets the ID of the neighbor in the builder.
         *
         * @param id the ID to set
         * @return the Builder instance
         */
        fun withId(id: String): Builder {
            this.id = id
            return this
        }

        /**
         * Sets the score of the neighbor in the builder.
         *
         * @param score the score to set
         * @return the Builder instance
         */
        fun withScore(score: Float): Builder {
            this.score = score
            return this
        }

        /**
         * Populates the Neighbor instance with the values from the builder.
         *
         * @param instance the Neighbor instance to populate
         */
        protected fun populate(instance: Neighbor) {
            instance.id = this.id
            instance.score = this.score
        }

        /**
         * Builds a Neighbor instance with the set values.
         *
         * @return the built Neighbor instance
         */
        fun build(): Neighbor {
            val instance = Neighbor()

            populate(instance)

            return instance
        }
    }

    /**
     * Calculates the hash code for the Neighbor object.
     *
     * @return The hash code calculated for the Neighbor.
     */
    override fun hashCode(): Int {
        return internalHashCodeCompute(classNameHashCode, id, score)
    }

    /**
     * Calculates the hash code for an array of objects.
     *
     * @param objects The objects to calculate the hash code for.
     * @return The hash code calculated for the objects array.
     */
    private fun internalHashCodeCompute(vararg objects: Any?): Int {
        return Arrays.hashCode(objects)
    }

    /**
     * Checks if the Neighbor object is equal to another object.
     *
     * @param other The object to compare for equality.
     * @return True if the Neighbor is equal to the other object, false otherwise.
     */
    override fun equals(other: Any?): Boolean {
        if (this === other) {
            return true
        }

        if (other !is Neighbor) {
            return false
        }

        val that = other as Neighbor

        return Objects.equals(id, that.id) &&
                Objects.equals(score, that.score)
    }

    /**
     * Returns a string representation of the Neighbor object.
     *
     * @return A string representation of the Neighbor.
     */
    override fun toString(): String {
        val ret = StringBuilder()
        ret.append("Neighbor(")

        ret.append("id=")
        ret.append(id)
        ret.append(", ")

        ret.append("score=")
        ret.append(score)
        ret.append(")")

        return ret.toString()
    }
}
