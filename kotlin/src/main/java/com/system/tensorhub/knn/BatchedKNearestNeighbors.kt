package com.system.tensorhub.knn

import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutionException
import com.system.tensorhub.knn.TimedBatchExecutor.Work

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
        val batchSize = getBatchSize()
        val featureSize = getFeatureSize()
        val maxK = getMaxK()
        batchInputs = FloatArray(batchSize * featureSize)
        batchScores = FloatArray(batchSize * maxK)
        batchIds = Array(batchSize * maxK) { "" }

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
            for ((offset, input) in inputs.withIndex()) {
                val coordinates = input.vector.coordinates
                for (i in coordinates.indices) {
                    batchInputs[offset * featureSize + i] = coordinates[i]
                }
            }

            // Fill remaining batch input slots with zeros if necessary
            if (inputs.size < getBatchSize()) {
                batchInputs.fill(0f, inputs.size * featureSize, batchInputs.size)
            }

            // Invoke the CUDA method to find nearest neighbors
            knnCuda.findKnn(batchInputs, activeBatchSize, batchScores, batchIds)

            // Process the results and populate the outputs
            outputs.addAll(inputs.mapIndexed { offset, knnInput ->
                val k = knnInput.k
                val inputIdx = knnInput.vector.index

                val nearestNeighbors = (0 until k).map { i ->
                    val score = batchScores[maxK * offset + i]
                    val id = batchIds[maxK * offset + i]
                    Neighbor.builder().withId(id).withScore(score).build()
                }
                NearestNeighbors.builder().withIndex(inputIdx).withNeighbors(nearestNeighbors).build()
            })
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

        val futures = inputBatch.map { vector ->
            batchExecutor.addAsync(KnnInput(k, vector))
        }

        return futures.map { future ->
            try {
                future.get()
            } catch (e: InterruptedException) {
                throw RuntimeException("Error waiting for knn result", e)
            } catch (e: ExecutionException) {
                throw RuntimeException("Error waiting for knn result", e)
            }
        }
    }
}
