package com.system.tensorhub.knn

import java.util.Arrays
import java.util.Objects

/**
 * Represents a Neighbor object in a k-nearest neighbors algorithm.
 */
class Neighbor private constructor() {
    var id: String? = null
        private set
    var score = 0f
        private set

    /**
     * Builder class for creating Neighbor instances.
     */
    class Builder {
        private var id: String? = null
        private var score = 0f

        /**
         * Sets the id of the Neighbor.
         *
         * @param id The id to set.
         * @return The Builder instance.
         */
        fun withId(id: String?): Builder {
            this.id = id
            return this
        }

        /**
         * Sets the score of the Neighbor.
         *
         * @param score The score to set.
         * @return The Builder instance.
         */
        fun withScore(score: Float): Builder {
            this.score = score
            return this
        }

        /**
         * Populates the Neighbor instance with the set values.
         *
         * @param instance The Neighbor instance to populate.
         */
        fun populate(instance: Neighbor) {
            instance.setId(id)
            instance.setScore(score)
        }

        /**
         * Builds a Neighbor instance with the set values.
         *
         * @return The built Neighbor instance.
         */
        fun build(): Neighbor {
            val instance = Neighbor()
            populate(instance)
            return instance
        }
    }

    /**
     * Sets the id of the Neighbor.
     *
     * @param id The id to set.
     */
    fun setId(id: String?) {
        this.id = id
    }

    /**
     * Sets the score of the Neighbor.
     *
     * @param score The score to set.
     */
    fun setScore(score: Float) {
        this.score = score
    }

    override fun hashCode(): Int {
        return internalHashCodeCompute(classNameHashCode, id, score)
    }

    private fun internalHashCodeCompute(vararg objects: Any?): Int {
        return Arrays.hashCode(objects)
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) {
            return true
        }
        if (other !is Neighbor) {
            return false
        }
        val that = other
        return id == that.id && score == that.score
    }

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

    companion object {
        private const val classNameHashCode = internalHashCodeCompute("com.system.tensorhub.knn.Neighbor")

        /**
         * Creates a new instance of Neighbor.Builder.
         *
         * @return The new Builder instance.
         */
        fun builder(): Builder {
            return Builder()
        }
    }
}
