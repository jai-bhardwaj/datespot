package com.system.tensorhub.knn

import java.util.ArrayList
import java.util.List
import java.util.concurrent.*

/**
 * A timed batch executor that processes inputs in batches with a timeout.
 *
 * @param I The type of input elements.
 * @param O The type of output elements.
 * @property label The label for the batch executor.
 * @property batchSize The maximum batch size.
 * @property timeout The timeout in milliseconds for processing a batch.
 * @property work The work to be performed on the batch.
 */
class TimedBatchExecutor<I, O>(private val label: String, private val batchSize: Int, private val timeout: Long, private val work: Work<I, O>) {
    private val batchQueue: BlockingQueue<BatchElement<I, O>> = ArrayBlockingQueue(batchSize, true)
    private val daemon: ScheduledExecutorService = Executors.newSingleThreadScheduledExecutor()
    private val batchProcessor: BatchProcessor = BatchProcessor(work)
    private val queueLock: Lock = ReentrantLock()
    private val processLock: Lock = ReentrantLock()

    private var timeoutFlusher: ScheduledFuture<*>? = null

    /**
     * Represents an element in the batch, containing the input and output CompletableFuture.
     *
     * @property input The input element.
     * @property output The CompletableFuture for the output element.
     */
    data class BatchElement<I, O>(val input: I, val output: CompletableFuture<O>)

    init {
        require(batchSize > 0) { "batchSize: $batchSize should be > 0" }
        require(timeout > 0) { "timeout: $timeout should be > 0 ms" }
    }

    /**
     * Adds an input element to the batch and waits for the output element.
     *
     * @param input The input element to add.
     * @return The output element.
     * @throws RuntimeException if there was an error waiting for the output.
     */
    fun add(input: I): O {
        return try {
            addAsync(input).get()
        } catch (e: InterruptedException) {
            throw RuntimeException("Error waiting for output for input: $input", e)
        } catch (e: ExecutionException) {
            throw RuntimeException("Error waiting for output for input: $input", e.cause)
        }
    }

    /**
     * Adds an input element to the batch asynchronously and returns a CompletableFuture for the output element.
     *
     * @param input The input element to add.
     * @return The CompletableFuture for the output element.
     * @throws RuntimeException if there was an error queueing the input to the batch.
     */
    fun addAsync(input: I): CompletableFuture<O> {
        val output = CompletableFuture<O>()
        try {
            queueLock.lock()
            val firstTask = batchQueue.isEmpty()
            batchQueue.add(BatchElement(input, output))

            if (firstTask) {
                timeoutFlusher = daemon.schedule(batchProcessor, timeout, TimeUnit.MILLISECONDS)
            } else if (batchQueue.remainingCapacity() == 0) {
                timeoutFlusher?.cancel(false)
                timeoutFlusher = null
                daemon.schedule(batchProcessor, 0, TimeUnit.MILLISECONDS)
            }
        } catch (e: InterruptedException) {
            throw RuntimeException("Error queueing input: $input to the current batch", e)
        } finally {
            queueLock.unlock()
        }

        return output
    }

    /**
     * The batch processor that runs the work on the batch.
     *
     * @property work The work to be performed on the batch.
     */
    inner class BatchProcessor(private val work: Work<I, O>) : Runnable {
        override fun run() {
            processLock.lock()
            try {
                val batch = ArrayList<BatchElement<I, O>>(batchSize)
                val numInputs = batchQueue.drainTo(batch, batchSize)
                if (numInputs == 0) {
                    return
                }

                val inputs = ArrayList<I>(batchSize)
                val outputs = ArrayList<O>(batchSize)

                for (batchElement in batch) {
                    inputs.add(batchElement.input)
                }

                invokeWork(inputs, outputs)

                if (inputs.size != outputs.size) {
                    throw RuntimeException("Num inputs: ${inputs.size} does not match num outputs: ${outputs.size}")
                }

                for (i in outputs.indices) {
                    batch[i].output.complete(outputs[i])
                }
            } catch (e: Throwable) {
                e.printStackTrace()
            } finally {
                processLock.unlock()
            }
        }

        /**
         * Invokes the work on the batch inputs and populates the outputs list.
         *
         * @param inputs The list of input elements in the batch.
         * @param outputs The list to populate with output elements.
         */
        private fun invokeWork(inputs: List<I>, outputs: MutableList<O>) {
            work.invoke(inputs, outputs)
        }
    }

    /**
     * Represents the work to be performed on the batch.
     *
     * @param I The type of input elements.
     * @param O The type of output elements.
     */
    interface Work<I, O> {
        /**
         * Performs the work on the batch inputs and populates the outputs list.
         *
         * @param inputs The list of input elements in the batch.
         * @param outputs The list to populate with output elements.
         */
        fun invoke(inputs: List<I>, outputs: MutableList<O>)
    }
}
