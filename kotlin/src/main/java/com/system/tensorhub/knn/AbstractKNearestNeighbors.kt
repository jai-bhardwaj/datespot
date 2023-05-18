package com.system.tensorhub.knn

import java.util.List

/**
 * Represents an abstract base class for K-Nearest Neighbors implementations.
 *
 * @param label The label of the KNN implementation.
 * @param maxK The maximum value for K.
 * @param featureSize The size of the feature vector.
 * @param batchSize The size of the input batch.
 */
abstract class AbstractKNearestNeighbors(
    private val label: String,
    private val maxK: Int,
    private val featureSize: Int,
    private val batchSize: Int
) : KNearestNeighbors {

    /**
     * Validates the feature size of a single input vector.
     *
     * @param inputVector The input vector to validate.
     * @throws IllegalArgumentException if the feature size does not match.
     */
    protected fun validateFeatureSize(inputVector: Vector) {
        val size = inputVector.coordinates.size
        require(size == featureSize) { "Feature size: $size should equal: $featureSize" }
    }

    /**
     * Validates the feature size of a batch of input vectors.
     *
     * @param inputBatch The input batch to validate.
     * @throws IllegalArgumentException if any input vector's feature size does not match.
     */
    protected fun validateFeatureSize(inputBatch: List<Vector>) {
        inputBatch.forEach { validateFeatureSize(it) }
    }

    /**
     * Validates the value of K.
     *
     * @param k The value of K to validate.
     * @throws IllegalArgumentException if the value of K is not within the valid range.
     */
    protected fun validateK(k: Int) {
        require(k in 1..maxK) { "K should be > 0 and <= $maxK. Given: $k" }
    }

    /**
     * Gets the maximum value for K.
     *
     * @return The maximum value for K.
     */
    override fun getMaxK(): Int = maxK

    /**
     * Gets the size of the feature vector.
     *
     * @return The size of the feature vector.
     */
    override fun getFeatureSize(): Int = featureSize
}
