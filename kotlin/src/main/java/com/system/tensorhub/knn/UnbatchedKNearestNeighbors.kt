package com.system.tensorhub.knn

import java.util.ArrayList

/**
 * Represents an unbatched k-nearest neighbors implementation.
 * Extends the AbstractKNearestNeighbors class.
 *
 * @property knnCuda The KNearestNeighborsCuda instance.
 * @property inputs The array to store input values.
 * @property scores The array to store scores.
 * @property ids The array to store IDs.
 */
open class UnbatchedKNearestNeighbors(private val knnCuda: KNearestNeighborsCuda) : AbstractKNearestNeighbors() {

    /**
     * The array to store input values.
     */
    private val inputs: FloatArray

    /**
     * The array to store scores.
     */
    private val scores: FloatArray

    /**
     * The array to store IDs.
     */
    private val ids: Array<String?>

    /**
     * Constructor for the UnbatchedKNearestNeighbors class.
     *
     * @param label The label for the instance.
     * @param knnCuda The KNearestNeighborsCuda instance.
     */
    constructor(label: String, knnCuda: KNearestNeighborsCuda) : this(knnCuda) {
        setLabel(label)
        setMaxK(knnCuda.maxK)
        setFeatureSize(knnCuda.featureSize)
        setBatchSize(knnCuda.batchSize)
    }

    init {
        val batchSize = getBatchSize()
        val featureSize = getFeatureSize()
        val maxK = getMaxK()

        inputs = FloatArray(batchSize * featureSize)
        scores = FloatArray(batchSize * maxK)
        ids = arrayOfNulls(batchSize * maxK)
    }

    /**
     * Overrides the findKnn method to find k nearest neighbors for a single input vector.
     *
     * @param k The number of nearest neighbors to find.
     * @param inputVector The input vector for which to find the nearest neighbors.
     * @return The NearestNeighbors object containing the nearest neighbors.
     */
    override fun findKnn(k: Int, inputVector: Vector): NearestNeighbors {
        validateK(k)
        validateFeatureSize(inputVector)

        val coordinates = inputVector.coordinates

        inputs.indices.forEach { i ->
            inputs[i] = coordinates[i]
        }

        knnCuda.findKnn(inputs, scores, ids)

        val nearestNeighbors = ArrayList<Neighbor>(k)
        for (i in 0 until k) {
            val score = scores[i]
            val id = ids[i]
            val neighbor = Neighbor.builder().withId(id).withScore(score).build()
            nearestNeighbors.add(neighbor)
        }

        return NearestNeighbors.builder().withIndex(inputVector.index).withNeighbors(nearestNeighbors).build()
    }

    /**
     * Overrides the findKnnBatch method to find k nearest neighbors for a batch of input vectors.
     *
     * @param k The number of nearest neighbors to find.
     * @param inputBatch The list of input vectors for which to find the nearest neighbors.
     * @return The list of NearestNeighbors objects containing the nearest neighbors for each input vector.
     */
    override fun findKnnBatch(k: Int, inputBatch: List<Vector>): List<NearestNeighbors> {
        return inputBatch.map { vector -> findKnn(k, vector) }
    }
}
