package com.system.tensorhub.knn

import java.io.Closeable
import java.io.File
import java.io.IOException

import lombok.extern.slf4j.Slf4j

/**
 * Represents a K-nearest neighbors implementation using CUDA.
 */
@Slf4j
class KNearestNeighborsCuda(
    private val maxK: Int,
    private val batchSize: Int,
    private val featureSize: Int,
    private val dataType: DataType = DataType.FP32,
    private val fileDeviceMapping: Map<File, Int>,
    private val keyValueDelim: Char = '\t',
    private val vectorDelim: Char = ' '
) : Closeable {

    private val NULLPTR = 0L
    private val LIBNAME = "tensorhub_knn_java"

    private var ptr: Long = NULLPTR

    init {
        log.info("Loading library: {}", LIBNAME)
        System.loadLibrary(LIBNAME)
        log.info("Loaded library: {}", LIBNAME)

        val uniqueDevices = fileDeviceMapping.values.toSet().size
        log.info("Initializing with maxK: {}, batchSize: {}, devices: {}", maxK, batchSize, uniqueDevices)
        ptr = initialize(maxK, batchSize, uniqueDevices, dataType.ordinal())

        val files = Array(fileDeviceMapping.size) { "" }
        val devices = IntArray(fileDeviceMapping.size)
        var i = 0
        for ((file, value) in fileDeviceMapping) {
            files[i] = file.absolutePath
            devices[i] = value
            i++
        }

        try {
            log.info("Loading data onto devices. {}", fileDeviceMapping)
            load(files, devices, keyValueDelim, vectorDelim, ptr)
        } catch (e: IOException) {
            throw RuntimeException("Error loading files: $fileDeviceMapping", e)
        }
    }

    fun findKnn(k: Int, inputVectors: FloatArray, width: Int, scores: FloatArray, keys: Array<String>) {
        validateArguments(k, inputVectors, batchSize, width)

        val outputLength = batchSize * maxK
        require(scores.size == outputLength) { "Output scores array must be of size: $outputLength (batchSize x maxK)." }
        require(keys.size == outputLength) { "Output keys array must be of size: $outputLength (batchSize x maxK)." }

        findKnn(k, inputVectors, batchSize, width, scores, keys, ptr)
    }

    fun findKnn(k: Int, inputVectors: FloatArray): KnnResult {
        validateArguments(k, inputVectors, batchSize, featureSize)
        return findKnn(k, inputVectors, batchSize, featureSize, ptr)
    }

    override fun close() {
        if (ptr != NULLPTR) {
            shutdown(ptr)
            ptr = NULLPTR
        }
    }

    private external fun initialize(maxK: Int, maxBatchSize: Int, numGPUs: Int, dataType: Int): Long

    private external fun load(filenames: Array<String>, devices: IntArray, keyValDelim: Char, vectorDelim: Char, ptr: Long)

    private external fun shutdown(ptr: Long)

    private external fun findKnn(
        k: Int,
        input: FloatArray,
        size: Int,
        width: Int,
        scores: FloatArray,
        keys: Array<String>,
        ptr: Long
    )

    private external fun findKnn(k: Int, input: FloatArray, size: Int, width: Int, ptr: Long): KnnResult

    private fun validateArguments(k: Int, inputVectors: FloatArray, activeBatchSize: Int, width: Int) {
        require(width >= 1) { "Dimension of the input vector should be at least one." }
        require(k <= maxK) { "k = $k is greater than maxK = $maxK." }
        require(k >= 1) { "k must be at least 1." }
        require(inputVectors.size % width == 0) { "Width: $width does not divide the vectors: ${inputVectors.size}." }
        val actualBatchSize = inputVectors.size / width
        require(actualBatchSize == batchSize) { "$actualBatchSize is not equal to configured batchSize: $batchSize." }
        require(inputVectors.isNotEmpty()) { "Input vector must contain at least one vector." }
        require(activeBatchSize <= batchSize) { "Active batch size must be less than or equal to batchSize: $batchSize." }
        require(activeBatchSize <= actualBatchSize) { "Active batch size must be less than or equal to actual batch size: $actualBatchSize." }
    }
}
