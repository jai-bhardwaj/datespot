package com.system.tensorhub.knn

import java.util.ArrayList
import java.util.Arrays
import java.util.List
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutionException
import com.system.tensorhub.knn.TimedBatchExecutor.Work
import lombok.Value

/**
 * Represents a batched implementation of the k-nearest neighbors algorithm.
 *
 * @property knnCuda The CUDA implementation of k-nearest neighbors.
 * @property batchExecutor The timed batch executor for processing knn requests in batches.
 * @property batchInputs The input batch data for the CUDA implementation.
 * @property batchScores The output batch scores from the CUDA implementation.
 * @property batchIds The output batch IDs from the CUDA implementation.
 */
class BatchedKNearestNeighbors(private val knnCuda: KNearestNeighborsCuda, timeout: Long) :
    AbstractKNearestNeighbors() {

    private val batchExecutor: TimedBatchExecutor<KnnInput, NearestNeighbors>
    private val batchInputs: FloatArray
    private val batchScores: FloatArray
    private val batchIds: Array<String?>

    /**
     * Data class representing an input for the k-nearest neighbors algorithm.
     *
     * @property k The number of nearest neighbors to find.
     * @property vector The input vector for finding nearest neighbors.
     */
    @Value
    data class KnnInput(val k: Int, val vector: Vector)

    init {
        super.setLabel(label)
        super.setMaxK(knnCuda.getMaxK())
        super.setFeatureSize(knnCuda.getFeatureSize())
        super.setBatchSize(knnCuda.getBatchSize())
        this.knnCuda = knnCuda

        val batchSize = super.getBatchSize()
        val featureSize = super.getFeatureSize()
        val maxK = super.getMaxK()
        this.batchInputs = FloatArray(batchSize * featureSize)
        this.batchScores = FloatArray(batchSize * maxK)
        this.batchIds = arrayOfNulls(batchSize * maxK)

        this.batchExecutor = TimedBatchExecutor(label, batchSize, timeout, FindKnnWork())
    }

    internal inner class FindKnnWork : Work<KnnInput, NearestNeighbors> {
        /**
         * Invokes the work to find the nearest neighbors for a batch of input data.
         *
         * @param inputs The list of input data for finding nearest neighbors.
         * @param outputs The list to store the computed nearest neighbors.
         */
        override fun invoke(inputs: List<KnnInput>, outputs: List<NearestNeighbors>) {
            val featureSize = super.getFeatureSize()
            val maxK = super.getMaxK()
            val activeBatchSize = inputs.size

            for (offset in inputs.indices) {
                val input = inputs[offset]
                val coordinates = input.vector.coordinates
                for (i in coordinates.indices) {
                    batchInputs[offset * featureSize + i] = coordinates[i]
                }
            }

            if (inputs.size < super.getBatchSize()) {
                Arrays.fill(
                    batchInputs,
                    inputs.size * featureSize,
                    batchInputs.size,
                    0f
                )
            }

            knnCuda.findKnn(batchInputs, activeBatchSize, batchScores, batchIds)

            for (offset in inputs.indices) {
                val k = inputs[offset].k
                val inputIdx = inputs[offset].vector.index

                val nearestNeighbors: MutableList<Neighbor> = ArrayList(k)
                for (i in 0 until k) {
                    val score = batchScores[maxK * offset + i]
                    val id = batchIds[maxK * offset + i]
                    val neighbor = Neighbor.Builder().withId(id).withScore(score).build()
                    nearestNeighbors.add(neighbor)
                }
                outputs.add(NearestNeighbors.Builder().withIndex(inputIdx).withNeighbors(nearestNeighbors).build())
            }
        }
    }

    /**
     * Finds the k-nearest neighbors for a single vector.
     *
     * @param k The number of nearest neighbors to find.
     * @param vector The input vector for finding nearest neighbors.
     * @return The computed nearest neighbors.
     */
    override fun findKnn(k: Int, vector: Vector): NearestNeighbors {
        validateK(k)
        validateFeatureSize(vector)
        return batchExecutor.add(KnnInput(k, vector))
    }

    /**
     * Finds the k-nearest neighbors for a batch of input vectors.
     *
     * @param k The number of nearest neighbors to find.
     * @param inputBatch The batch of input vectors for finding nearest neighbors.
     * @return The list of computed nearest neighbors for each input vector.
     */
    override fun findKnnBatch(k: Int, inputBatch: List<Vector>): List<NearestNeighbors> {
        validateK(k)
        validateFeatureSize(inputBatch)

        val futures: MutableList<CompletableFuture<NearestNeighbors>> = ArrayList(inputBatch.size)
        for (vector in inputBatch) {
            futures.add(batchExecutor.addAsync(KnnInput(k, vector)))
        }

        val results: MutableList<NearestNeighbors> = ArrayList(inputBatch.size)
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
