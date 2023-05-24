package com.system.tensorhub.knn

import lombok.AccessLevel
import lombok.Getter
import lombok.RequiredArgsConstructor

/**
 * Abstract class implementing the KNearestNeighbors interface.
 * Provides common functionality and validation for KNN algorithms.
 *
 * @param label The label associated with the KNN algorithm.
 * @param maxK The maximum value of K.
 * @param featureSize The size of the input feature vector.
 * @param batchSize The size of the input batch.
 */
@RequiredArgsConstructor
@Getter(AccessLevel.PROTECTED)
abstract class AbstractKNearestNeighbors(private val label: String, private val maxK: Int, private val featureSize: Int, private val batchSize: Int) : KNearestNeighbors {

    /**
     * Validates the size of the input feature vector.
     *
     * @param inputVector The input feature vector to validate.
     * @throws IllegalArgumentException if the size of the input vector is not equal to the expected feature size.
     */
    protected fun validateFeatureSize(inputVector: Vector) {
        val size = inputVector.coordinates.size
        if (size != featureSize) {
            throw IllegalArgumentException("feature size: $size should equal: $featureSize")
        }
    }

    /**
     * Validates the size of a batch of input feature vectors.
     *
     * @param inputBatch The batch of input feature vectors to validate.
     */
    protected fun validateFeatureSize(inputBatch: List<Vector>) {
        inputBatch.forEach(this::validateFeatureSize)
    }

    /**
     * Validates the value of K.
     *
     * @param k The value of K to validate.
     * @throws IllegalArgumentException if K is less than or equal to 0 or greater than the maximum value of K.
     */
    protected fun validateK(k: Int) {
        if (k <= 0 || k > maxK) {
            throw IllegalArgumentException("k should be > 0 and < $maxK. Given: $k")
        }
    }

    /**
     * Gets the maximum value of K.
     *
     * @return The maximum value of K.
     */
    override fun getMaxK(): Int {
        return maxK
    }

    /**
     * Gets the size of the input feature vector.
     *
     * @return The size of the input feature vector.
     */
    override fun getFeatureSize(): Int {
        return featureSize
    }
}
