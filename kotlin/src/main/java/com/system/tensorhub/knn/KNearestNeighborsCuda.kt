package com.system.tensorhub.knn

import java.io.Closeable
import java.io.File
import java.io.IOException
import java.util.Collections
import java.util.HashSet
import java.util.LinkedHashMap
import java.util.List
import java.util.Map
import java.util.Map.Entry

import lombok.Getter
import lombok.RequiredArgsConstructor
import lombok.extern.slf4j.Slf4j

/**
 * KNearestNeighborsCuda class for performing k-nearest neighbors search using CUDA.
 */
@Slf4j
@RequiredArgsConstructor
class KNearestNeighborsCuda(
    val maxK: Int,
    val batchSize: Int,
    val featureSize: Int,
    val dataType: DataType,
    private val fileDeviceMapping: Map<File, Int>,
    private val keyValueDelim: Char,
    private val vectorDelim: Char
) : Closeable {
    companion object {
        private const val NULLPTR: Long = 0
        private const val LIBNAME = "tensorhub_knn_java"
        private const val DEFAULT_KEYVAL_DELIM = '\t'
        private const val DEFAULT_VEC_DELIM = ' '
        private val DEFAULT_DATA_TYPE = DataType.FP32

        /**
         * Converts a string to a character.
         * @param inStr The input string.
         * @return The resulting character.
         * @throws IllegalArgumentException if the input string is not of length 1.
         */
        private fun toChar(inStr: String): Char {
            if (inStr.length != 1) {
                throw IllegalArgumentException("String: $inStr is not length 1, cannot convert to char")
            } else {
                return inStr[0]
            }
        }

        /**
         * Converts a list of data files to a map with the files as keys and their index as values.
         * @param dataFiles The list of data files.
         * @return The resulting map.
         */
        private fun toMapInIndexOrder(dataFiles: List<File>): Map<File, Int> {
            val mapping = LinkedHashMap<File, Int>()
            if (dataFiles != null) {
                for (i in dataFiles.indices) {
                    mapping[dataFiles[i]] = i
                }
            }
            return mapping
        }
    }

    private var ptr: Long = NULLPTR

    constructor(maxK: Int, batchSize: Int, featureSize: Int, dataFiles: List<File>) :
            this(maxK, batchSize, featureSize, DEFAULT_DATA_TYPE, dataFiles)

    constructor(maxK: Int, batchSize: Int, featureSize: Int, dataType: DataType, dataFiles: List<File>) :
            this(maxK, batchSize, featureSize, dataType, toMapInIndexOrder(dataFiles), DEFAULT_KEYVAL_DELIM,
                DEFAULT_VEC_DELIM)

    constructor(
        maxK: Int,
        batchSize: Int,
        featureSize: Int,
        dataType: DataType,
        dataFiles: List<File>,
        keyValueDelim: String,
        vectorDelim: String
    ) : this(
        maxK,
        batchSize,
        featureSize,
        dataType,
        toMapInIndexOrder(dataFiles),
        toChar(keyValueDelim),
        toChar(vectorDelim)
    )

    constructor(maxK: Int, batchSize: Int, featureSize: Int, device: Int, dataFile: File) :
            this(maxK, batchSize, featureSize, DEFAULT_DATA_TYPE, Collections.singletonMap(dataFile, device),
                DEFAULT_KEYVAL_DELIM, DEFAULT_VEC_DELIM)

    constructor(
        maxK: Int,
        batchSize: Int,
        featureSize: Int,
        dataType: DataType,
        fileDeviceMapping: Map<File, Int>,
        keyValueDelim: String,
        vectorDelim: String
    ) : this(
        maxK,
        batchSize,
        featureSize,
        dataType,
        fileDeviceMapping,
        toChar(keyValueDelim),
        toChar(vectorDelim)
    )

    constructor(
        maxK: Int,
        batchSize: Int,
        featureSize: Int,
        fileDeviceMapping: Map<File, Int>,
        keyValueDelim: Char,
        vectorDelim: Char
    ) : this(maxK, batchSize, featureSize, DEFAULT_DATA_TYPE, fileDeviceMapping, keyValueDelim, vectorDelim)

    /**
     * Converts a string to a character.
     * @param in The input string.
     * @return The resulting character.
     * @throws IllegalArgumentException if the input string is not of length 1.
     */
    private fun toChar(inStr: String): Char {
        if (inStr.length != 1) {
            throw IllegalArgumentException("String: $inStr is not length 1, cannot convert to char")
        } else {
            return inStr[0]
        }
    }

    /**
     * Converts a list of data files to a map with the files as keys and their index as values.
     * @param dataFiles The list of data files.
     * @return The resulting map.
     */
    private fun toMapInIndexOrder(dataFiles: List<File>): Map<File, Int> {
        val mapping = LinkedHashMap<File, Int>()
        if (dataFiles != null) {
            for (i in dataFiles.indices) {
                mapping[dataFiles[i]] = i
            }
        }
        return mapping
    }

    /**
     * Initializes the KNearestNeighborsCuda instance.
     */
    fun init() {
        log.info("Loading library: {}", LIBNAME)
        System.loadLibrary(LIBNAME)
        log.info("Loaded library: {}", LIBNAME)

        val files = arrayOfNulls<String>(fileDeviceMapping.size)
        val devices = IntArray(fileDeviceMapping.size)
        val uniqueDevices = uniqueDevices()

        log.info("Initializing with maxK: {}, batchSize: {}, devices: {}", maxK, batchSize, uniqueDevices)
        ptr = initialize(maxK, batchSize, uniqueDevices, dataType.ordinal)

        var i = 0
        for ((key, value) in fileDeviceMapping) {
            files[i] = key.absolutePath
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

    /**
     * Performs k-nearest neighbors search on the input vectors.
     * @param inputVectors The input vectors.
     * @param scores The output array to store the computed scores.
     * @param keys The output array to store the keys of the nearest neighbors.
     */
    fun findKnn(inputVectors: FloatArray, scores: FloatArray, keys: Array<String>) {
        findKnn(maxK, inputVectors, featureSize, scores, keys)
    }

    /**
     * Performs k-nearest neighbors search on the input vectors.
     * @param inputVectors The input vectors.
     * @param activeBatchSize The actual batch size of the input vectors.
     * @param scores The output array to store the computed scores.
     * @param keys The output array to store the keys of the nearest neighbors.
     */
    fun findKnn(
        inputVectors: FloatArray,
        activeBatchSize: Int,
        scores: FloatArray,
        keys: Array<String>
    ) {
        findKnn(maxK, inputVectors, activeBatchSize, featureSize, scores, keys)
    }

    /**
     * Performs k-nearest neighbors search on the input vectors.
     * @param k The value of k for the search.
     * @param inputVectors The input vectors.
     * @param width The dimensionality of the input vectors.
     * @param scores The output array to store the computed scores.
     * @param keys The output array to store the keys of the nearest neighbors.
     */
    fun findKnn(
        k: Int,
        inputVectors: FloatArray,
        width: Int,
        scores: FloatArray,
        keys: Array<String>
    ) {
        findKnn(k, inputVectors, batchSize, width, scores, keys)
    }

    /**
     * Performs k-nearest neighbors search on the input vectors.
     * @param k The value of k for the search.
     * @param inputVectors The input vectors.
     * @param activeBatchSize The actual batch size of the input vectors.
     * @param width The dimensionality of the input vectors.
     * @param scores The output array to store the computed scores.
     * @param keys The output array to store the keys of the nearest neighbors.
     */
    fun findKnn(
        k: Int,
        inputVectors: FloatArray,
        activeBatchSize: Int,
        width: Int,
        scores: FloatArray,
        keys: Array<String>
    ) {
        validateArguments(k, inputVectors, activeBatchSize, width)

        val outputLength = batchSize * maxK
        if (scores.size != outputLength) {
            throw IllegalArgumentException(
                "output scores array must be of size: " + outputLength + " (batchSize x maxK)"
            )
        }
        if (keys.size != outputLength) {
            throw IllegalArgumentException(
                "output keys array must be of size: " + outputLength + " (batchSize x maxK)"
            )
        }

        findKnn(k, inputVectors, batchSize, width, scores, keys, ptr)
    }

    /**
     * Performs k-nearest neighbors search on the input vectors.
     * @param inputVectors The input vectors.
     * @return The result of the search containing the computed scores and keys of the nearest neighbors.
     */
    fun findKnn(inputVectors: FloatArray): KnnResult {
        return findKnn(maxK, inputVectors)
    }

    /**
     * Performs k-nearest neighbors search on the input vectors.
     * @param k The value of k for the search.
     * @param inputVectors The input vectors.
     * @return The result of the search containing the computed scores and keys of the nearest neighbors.
     */
    fun findKnn(k: Int, inputVectors: FloatArray): KnnResult {
        validateArguments(k, inputVectors, batchSize, featureSize)
        return findKnn(k, inputVectors, batchSize, featureSize, ptr)
    }

    /**
     * Validates the input arguments for k-nearest neighbors search.
     * @param k The value of k for the search.
     * @param inputVectors The input vectors.
     * @param activeBatchSize The actual batch size of the input vectors.
     * @param width The dimensionality of the input vectors.
     * @throws IllegalArgumentException if the input arguments are invalid.
     */
    private fun validateArguments(k: Int, inputVectors: FloatArray, activeBatchSize: Int, width: Int) {
        if (width < 1) {
            throw IllegalArgumentException("dimension of the input vector should be at least one")
        }
        if (k > maxK) {
            throw IllegalArgumentException("k = $k is greater than maxK = $maxK")
        }
        if (k < 1) {
            throw IllegalArgumentException("k must be at least 1")
        }
        if (inputVectors.size % width != 0) {
            throw IllegalArgumentException(
                "width: $width does not divide the vectors: " + inputVectors.size
            )
        }
        val actualBatchSize = inputVectors.size / width
        if (actualBatchSize != batchSize) {
            throw IllegalArgumentException(
                "$actualBatchSize is not equal to configured batchSize: $batchSize"
            )
        }
        if (inputVectors.isEmpty()) {
            throw IllegalArgumentException("input vector contain at least one vector")
        }
        if (activeBatchSize > batchSize) {
            throw IllegalArgumentException(
                "active batch size must be less than or equal to batchSize: $batchSize"
            )
        }
        if (activeBatchSize > actualBatchSize) {
            throw IllegalArgumentException(
                "active batch size must be less than or equal to actual batch size: $actualBatchSize"
            )
        }
    }

    /**
     * Closes the KNearestNeighborsCuda instance and releases resources.
     * @throws IOException if an I/O error occurs.
     */
    @Throws(IOException::class)
    override fun close() {
        if (ptr != 0L) {
            shutdown(ptr)
            ptr = 0L
        }
    }

    private external fun initialize(maxK: Int, maxBatchSize: Int, numGPUs: Int, dataType: Int): Long

    @Throws(IOException::class)
    private external fun load(
        filenames: Array<String>,
        devices: IntArray,
        keyValDelim: Char,
        vectorDelim: Char,
        ptr: Long
    )

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
}
