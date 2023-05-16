package com.system.tensorhub.knn

import java.util.ArrayList
import java.util.Arrays
import java.util.List
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutionException
import com.system.tensorhub.knn.TimedBatchExecutor.Work
import lombok.Value

/**
 * Represents a batched implementation of K-Nearest Neighbors using CUDA for acceleration.
 * Extends the AbstractKNearestNeighbors class.
 *
 * @param knnCuda The instance of KNearestNeighborsCuda for performing CUDA-based nearest neighbor computations.
 * @param batchExecutor The TimedBatchExecutor for executing batched nearest neighbor computations asynchronously.
 */
class BatchedKNearestNeighbors(
    private val knnCuda: KNearestNeighborsCuda,
    private val batchExecutor: TimedBatchExecutor<KnnInput, NearestNeighbors>
) : AbstractKNearestNeighbors {

    private val batchInputs: FloatArray
    private val batchScores: FloatArray
    private val batchIds: Array<String>

    /**
     * Represents the input for finding the k nearest neighbors.
     *
     * @param k The number of nearest neighbors to find.
     * @param vector The input vector for which to find the nearest neighbors.
     */
    @Value
    data class KnnInput(val k: Int, val vector: Vector)

    /**
     * Constructs a BatchedKNearestNeighbors instance with the specified KNearestNeighborsCuda and timeout values.
     *
     * @param knnCuda The instance of KNearestNeighborsCuda for performing CUDA-based nearest neighbor computations.
     * @param timeout The timeout value in milliseconds for the batched execution.
     */
    constructor(knnCuda: KNearestNeighborsCuda, timeout: Long) : this(knnCuda, null, timeout)

    /**
     * Constructs a BatchedKNearestNeighbors instance with the specified label, KNearestNeighborsCuda, and timeout values.
     *
     * @param label The label for the BatchedKNearestNeighbors instance.
     * @param knnCuda The instance of KNearestNeighborsCuda for performing CUDA-based nearest neighbor computations.
     * @param timeout The timeout value in milliseconds for the batched execution.
     */
    constructor(label: String?, knnCuda: KNearestNeighborsCuda, timeout: Long) :
            super(label, knnCuda.getMaxK(), knnCuda.getFeatureSize(), knnCuda.getBatchSize()) {
        this.knnCuda = knnCuda

        val batchSize = getBatchSize()
        val featureSize = getFeatureSize()
        val maxK = getMaxK()
        this.batchInputs = FloatArray(batchSize * featureSize)
        this.batchScores = FloatArray(batchSize * maxK)
        this.batchIds = Array(batchSize * maxK) { "" }

        this.batchExecutor = TimedBatchExecutor(label, batchSize, timeout, FindKnnWork())
    }

    /**
     * Represents a work item for finding the k nearest neighbors.
     * Implements the Work interface with KnnInput as the input type and NearestNeighbors as the output type.
     */
    class FindKnnWork : Work<KnnInput, NearestNeighbors> {

        /**
         * Executes the work item to find the k nearest neighbors.
         *
         * @param inputs The list of KnnInput objects containing the input vectors and k values.
         * @param outputs The list of NearestNeighbors objects to store the nearest neighbors found.
         */
        override fun invoke(inputs: List<KnnInput>, outputs: List<NearestNeighbors>) {
            val featureSize = getFeatureSize()
            val maxK = getMaxK()
            val activeBatchSize = inputs.size

            // Prepare batch input data
            for (offset in 0 until inputs.size) {
                val input = inputs[offset]
                val coordinates = input.vector.coordinates
                for (i in coordinates.indices) {
                    batchInputs[offset * featureSize + i] = coordinates[i]
                }
            }

            // Fill remaining batch input slots with zeros if necessary
            if (inputs.size < getBatchSize()) {
                Arrays.fill(batchInputs, inputs.size * featureSize, batchInputs.size, 0f)
            }

            // Invoke the CUDA method to find nearest neighbors
            knnCuda.findKnn(batchInputs, activeBatchSize, batchScores, batchIds)

            // Process the results and populate the outputs
            for (offset in 0 until inputs.size) {
                val k = inputs[offset].k
                val inputIdx = inputs[offset].vector.index

                val nearestNeighbors = ArrayList<Neighbor>(k)
                for (i in 0 until k) {
                    val score = batchScores[maxK * offset + i]
                    val id = batchIds[maxK * offset + i]
                    val neighbor = Neighbor.builder().withId(id).withScore(score).build()
                    nearestNeighbors.add(neighbor)
                }
                outputs.add(NearestNeighbors.builder().withIndex(inputIdx).withNeighbors(nearestNeighbors).build())
            }
        }
    }  

    /**
     * Finds the k nearest neighbors for a given vector.
     *
     * @param k The number of nearest neighbors to find.
     * @param vector The input vector for which to find the nearest neighbors.
     * @return The nearest neighbors found.
     */
    override fun findKnn(k: Int, vector: Vector): NearestNeighbors {
        validateK(k)
        validateFeatureSize(vector)
        return batchExecutor.add(KnnInput(k, vector))
    }

    /**
     * Finds the k nearest neighbors for a batch of input vectors.
     *
     * @param k The number of nearest neighbors to find.
     * @param inputBatch The list of input vectors for which to find the nearest neighbors.
     * @return The list of nearest neighbors found for each input vector.
     */
    override fun findKnnBatch(k: Int, inputBatch: List<Vector>): List<NearestNeighbors> {
        validateK(k)
        validateFeatureSize(inputBatch)

        val futures = ArrayList<CompletableFuture<NearestNeighbors>>(inputBatch.size)
        for (vector in inputBatch) {
            futures.add(batchExecutor.addAsync(KnnInput(k, vector)))
        }

        val results = ArrayList<NearestNeighbors>(inputBatch.size)
        for (future in futures) {
            try {
                results.add(future.get())
            } catch (e: InterruptedException) {
                throw RuntimeException("Error waiting for knn result", e)
            } catch (e: ExecutionException) {
                throw RuntimeException("Error waiting for knn result", e)
            }
        }
        return results
    }
}
