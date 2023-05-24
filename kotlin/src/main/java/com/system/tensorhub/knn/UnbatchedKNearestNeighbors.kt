package com.system.tensorhub.knn

import java.util.ArrayList

/**
 * Represents an unbatched implementation of the K-Nearest Neighbors algorithm.
 *
 * @property knnCuda The underlying KNearestNeighborsCuda object.
 */
class UnbatchedKNearestNeighbors(private val knnCuda: KNearestNeighborsCuda) : AbstractKNearestNeighbors {
    private val inputs: FloatArray
    private val scores: FloatArray
    private val ids: Array<String>

    /**
     * Constructs an UnbatchedKNearestNeighbors instance.
     *
     * @param knnCuda The underlying KNearestNeighborsCuda object.
     */
    constructor(knnCuda: KNearestNeighborsCuda) : this(null, knnCuda)

    /**
     * Constructs an UnbatchedKNearestNeighbors instance with a label.
     *
     * @param label The label for the KNN instance.
     * @param knnCuda The underlying KNearestNeighborsCuda object.
     */
    constructor(label: String?, knnCuda: KNearestNeighborsCuda) : super(label, knnCuda.getMaxK(), knnCuda.getFeatureSize(), knnCuda.getBatchSize()) {
        val batchSize = getBatchSize()
        val featureSize = getFeatureSize()
        val maxK = getMaxK()

        this.knnCuda = knnCuda
        this.inputs = FloatArray(batchSize * featureSize)
        this.scores = FloatArray(batchSize * maxK)
        this.ids = arrayOfNulls(batchSize * maxK)
    }

    /**
     * Finds the K nearest neighbors for a single input vector.
     *
     * @param k The number of nearest neighbors to find.
     * @param inputVector The input vector.
     * @return The NearestNeighbors object containing the index and nearest neighbors.
     */
    override fun findKnn(k: Int, inputVector: Vector): NearestNeighbors {
        validateK(k)
        validateFeatureSize(inputVector)

        val coordinates = inputVector.coordinates
        for (i in coordinates.indices) {
            inputs[i] = coordinates[i]
        }

        knnCuda.findKnn(inputs, scores, ids)

        val nearestNeighbors = ArrayList<Neighbor>(k)
        for (i in 0 until k) {
            val score = scores[i]
            val id = ids[i]
            val neighbor = Neighbor.Builder().withId(id).withScore(score).build()
            nearestNeighbors.add(neighbor)
        }

        return NearestNeighbors.Builder()
            .withIndex(inputVector.index)
            .withNeighbors(nearestNeighbors)
            .build()
    }

    /**
     * Finds the K nearest neighbors for a batch of input vectors.
     *
     * @param k The number of nearest neighbors to find.
     * @param inputBatch The list of input vectors.
     * @return The list of NearestNeighbors objects containing the index and nearest neighbors for each input vector.
     */
    override fun findKnnBatch(k: Int, inputBatch: List<Vector>): List<NearestNeighbors> {
        val results = ArrayList<NearestNeighbors>(inputBatch.size)
        for (vector in inputBatch) {
            results.add(findKnn(k, vector))
        }
        return results
    }
}
