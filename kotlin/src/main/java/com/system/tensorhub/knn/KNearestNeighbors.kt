package com.system.tensorhub.knn

/**
 * Interface for k-nearest neighbors algorithm.
 */
interface KNearestNeighbors {
    /**
     * Finds the k nearest neighbors for a single input vector.
     *
     * @param k The number of nearest neighbors to find.
     * @param inputVector The input vector for which to find nearest neighbors.
     * @return The nearest neighbors.
     */
    fun findKnn(k: Int, inputVector: Vector): NearestNeighbors

    /**
     * Finds the k nearest neighbors for a batch of input vectors.
     *
     * @param k The number of nearest neighbors to find.
     * @param inputVectors The list of input vectors for which to find nearest neighbors.
     * @return The list of nearest neighbors for each input vector.
     */
    fun findKnnBatch(k: Int, inputVectors: List<Vector>): List<NearestNeighbors>

    /**
     * Gets the maximum value of k allowed for finding nearest neighbors.
     *
     * @return The maximum value of k.
     */
    fun getMaxK(): Int

    /**
     * Gets the size of the feature space for the input vectors.
     *
     * @return The size of the feature space.
     */
    fun getFeatureSize(): Int
}
