package com.system.tensorhub.knn

/**
 * Represents the K-Nearest Neighbors algorithm.
 */
interface KNearestNeighbors {
    /**
     * Finds the K nearest neighbors for a single input vector.
     *
     * @param k The number of nearest neighbors to find.
     * @param inputVector The input vector for which to find the nearest neighbors.
     * @return The nearest neighbors.
     */
    fun findKnn(k: Int, inputVector: Vector): NearestNeighbors

    /**
     * Finds the K nearest neighbors for a batch of input vectors.
     *
     * @param k The number of nearest neighbors to find.
     * @param inputVectors The list of input vectors for which to find the nearest neighbors.
     * @return The list of nearest neighbors for each input vector.
     */
    fun findKnnBatch(k: Int, inputVectors: List<Vector>): List<NearestNeighbors>

    /**
     * Gets the maximum value for K.
     */
    val maxK: Int

    /**
     * Gets the size of the feature vector.
     */
    val featureSize: Int
}
