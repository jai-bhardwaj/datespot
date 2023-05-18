package com.system.tensorhub.knn

import java.util.ArrayList
import java.util.List
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.BlockingQueue
import java.util.concurrent.CompletableFuture
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * TimedBatchExecutor is a class that allows executing batch processing of inputs with a specified timeout.
 *
 * @param I the type of input elements
 * @param O the type of output elements
 * @property batchSize the maximum size of a batch
 * @property timeout the timeout value in milliseconds
 * @property batchQueue the blocking queue to store batch elements
 * @property daemon the scheduled executor service for executing the batch processor
 * @property batchProcessor the batch processor responsible for processing batches
 * @constructor Creates a TimedBatchExecutor with the specified label, batchSize, timeout, and work.
 * @throws IllegalArgumentException if batchSize is less than or equal to 0, or timeout is less than or equal to 0
 */
class TimedBatchExecutor<I, O>(
    label: String,
    batchSize: Int,
    timeout: Long,
    work: Work<I, O>
) {
    /**
     * Work is an interface that represents the batch processing work.
     */
    interface Work<I, O> {
        /**
         * Invokes the batch processing work with the provided inputs and outputs.
         *
         * @param inputs the list of input elements
         * @param outputs the list of output elements
         */
        fun invoke(inputs: List<I>, outputs: List<O>)
    }

    private val batchSize: Int
    private val timeout: Long
    private val batchQueue: BlockingQueue<BatchElement<I, O>>
    private val batchProcessor: BatchProcessor
    private val queueLock = ReentrantLock()
    private val processLock = ReentrantLock()
    private var timeoutFlusher: ScheduledFuture<*>? = null

    data class BatchElement<I, O>(val input: I, val output: CompletableFuture<O>)

    init {
        require(batchSize > 0) { "batchSize: $batchSize should be > 0" }
        require(timeout > 0) { "timeout: $timeout should be > 0 ms" }

        this.batchSize = batchSize
        this.timeout = timeout
        this.batchQueue = ArrayBlockingQueue(batchSize, true)
        this.batchProcessor = BatchProcessor(work)
    }

    /**
     * Adds an input element to the batch and waits for the corresponding output.
     *
     * @param input the input element to add
     * @return the output element corresponding to the input
     * @throws RuntimeException if there is an error waiting for the output
     */
    fun add(input: I): O {
        return addAsync(input).get()
    }

    /**
     * Asynchronously adds an input element to the batch and returns a CompletableFuture representing the output.
     *
     * @param input the input element to add
     * @return a CompletableFuture representing the output element corresponding to the input
     * @throws RuntimeException if there is an error queueing the input to the current batch
     */
    fun addAsync(input: I): CompletableFuture<O> {
        val output = CompletableFuture<O>()

        queueLock.withLock {
            batchQueue.put(BatchElement(input, output))

            val firstTask = batchQueue.size == 1
            if (firstTask) {
                timeoutFlusher = daemon.schedule(batchProcessor, timeout, TimeUnit.MILLISECONDS)
            } else {
                val batchFull = batchQueue.remainingCapacity() == 0
                if (batchFull) {
                    timeoutFlusher?.cancel(false)
                    timeoutFlusher = null
                    daemon.schedule(batchProcessor, 0, TimeUnit.MILLISECONDS)
                }
            }
        }

        return output
    }

    /**
     * BatchProcessor is a class that processes a batch of elements.
     *
     * @property work the work to be invoked for processing the batch
     * @constructor Creates a BatchProcessor with the specified work.
     */
    class BatchProcessor(private val work: Work<I, O>) : Runnable {
        /**
         * Runs the batch processing logic.
         */
        override fun run() {
            processLock.withLock {
                val batch = ArrayList<BatchElement<I, O>>(batchSize)
                val numInputs = batchQueue.drainTo(batch, batchSize)
                if (numInputs == 0) {
                    return
                }

                val inputs = ArrayList<I>(batchSize)
                val outputs = ArrayList<O>(batchSize)

                for (i in 0 until batch.size) {
                    val batchElement = batch[i]
                    inputs.add(batchElement.input)
                }

                invokeWork(inputs, outputs)

                if (inputs.size != outputs.size) {
                    throw RuntimeException("Num inputs: ${inputs.size} does not match num outputs: ${outputs.size}")
                }

                for (i in 0 until outputs.size) {
                    batch[i].output.complete(outputs[i])
                }
            }
        }

        /**
         * Invokes the work with the provided inputs and outputs.
         *
         * @param inputs the list of input elements
         * @param outputs the list of output elements
         */
        private fun invokeWork(inputs: List<I>, outputs: List<O>) {
            work.invoke(inputs, outputs)
        }
    }

    companion object {
        private val daemon: ScheduledExecutorService = Executors.newSingleThreadScheduledExecutor()
    }
}
