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
 * Logger for KNearestNeighborsCuda class.
 */
@Slf4j
/**
 * Represents a K-nearest neighbors implementation using CUDA.
 */
@RequiredArgsConstructor
/**
 * Represents a K-nearest neighbors implementation using CUDA.
 */
class KNearestNeighborsCuda implements Closeable {
    /**
     * Constant value representing null pointer.
     */
    private final long NULLPTR = 0;
    /**
     * Name of the native library.
     */
    private final String LIBNAME = "eduwise_knn_java";
    /**
     * Default key-value delimiter character.
     */
    private final char DEFAULT_KEYVAL_DELIM = '\t';
    /**
     * Default vector delimiter character.
     */
    private final char DEFAULT_VEC_DELIM = ' ';
    /**
     * Default data type.
     */
    private final DataType DEFAULT_DATA_TYPE = DataType.FP32;

    /**
     * Maximum value of K.
     */
    @Getter
    private final int maxK;
    /**
     * Batch size.
     */
    @Getter
    private final int batchSize;
    /**
     * Feature size.
     */
    @Getter
    private final int featureSize;
    /**
     * Data type.
     */
    @Getter
    private final DataType dataType;
    /**
     * Mapping of files to devices.
     */
    private final Map<File, Integer> fileDeviceMapping;
    /**
     * Key-value delimiter character.
     */
    private final char keyValueDelim;
    /**
     * Vector delimiter character.
     */
    private final char vectorDelim;

    /**
     * Volatile pointer value.
     */
    private volatile long ptr = NULLPTR;

    /**
     * Constructs a new instance of the class.
     *
     * @param maxK          the maximum value of K
     * @param batchSize     the batch size
     * @param featureSize   the feature size
     * @param dataFiles     the list of data files
     */
    constructor(maxK: Int, batchSize: Int, featureSize: Int, dataFiles: List<File>) : this(maxK, batchSize, featureSize, DEFAULT_DATA_TYPE, dataFiles)

    /**
     * Constructs a new instance of the class.
     *
     * @param maxK          the maximum value of K
     * @param batchSize     the batch size
     * @param featureSize   the feature size
     * @param dataType      the data type
     * @param dataFiles     the list of data files
     */
    constructor(maxK: Int, batchSize: Int, featureSize: Int, dataType: DataType, dataFiles: List<File>) : this(maxK, batchSize, featureSize, dataType, toMapInIndexOrder(dataFiles), DEFAULT_KEYVAL_DELIM, DEFAULT_VEC_DELIM)

    /**
     * Constructs a new instance of the class.
     *
     * @param maxK              the maximum value of K
     * @param batchSize         the batch size
     * @param featureSize       the feature size
     * @param dataType          the data type
     * @param dataFiles         the list of data files
     * @param keyValueDelim     the key-value delimiter
     * @param vectorDelim       the vector delimiter
     */
    constructor(maxK: Int, batchSize: Int, featureSize: Int, dataType: DataType, dataFiles: List<File>, keyValueDelim: String, vectorDelim: String) : this(maxK, batchSize, featureSize, dataType, toMapInIndexOrder(dataFiles), keyValueDelim[0], vectorDelim[0])


    /**
     * Constructs a new instance of the class.
     *
     * @param maxK              the maximum value of K
     * @param batchSize         the batch size
     * @param featureSize       the feature size
     * @param device            the device identifier
     * @param dataFile          the data file
     */
    constructor(maxK: Int, batchSize: Int, featureSize: Int, device: Int, dataFile: File) : this(maxK, batchSize, featureSize, DEFAULT_DATA_TYPE, Collections.singletonMap(dataFile, device), DEFAULT_KEYVAL_DELIM, DEFAULT_VEC_DELIM)

    /**
     * Constructs a new instance of the class.
     *
     * @param maxK                  the maximum value of K
     * @param batchSize             the batch size
     * @param featureSize           the feature size
     * @param dataType              the data type
     * @param fileDeviceMapping     the mapping of files to devices
     * @param keyValueDelim         the key-value delimiter
     * @param vectorDelim           the vector delimiter
     */
    constructor(maxK: Int, batchSize: Int, featureSize: Int, dataType: DataType, fileDeviceMapping: Map<File, Int>, keyValueDelim: String, vectorDelim: String) : this(maxK, batchSize, featureSize, dataType, fileDeviceMapping, toChar(keyValueDelim), toChar(vectorDelim))

    /**
     * Constructs a new instance of the class.
     *
     * @param maxK              the maximum value of K
     * @param batchSize         the batch size
     * @param featureSize       the feature size
     * @param fileDeviceMapping the mapping of files to devices
     * @param keyValueDelim     the key-value delimiter
     * @param vectorDelim       the vector delimiter
     */
    constructor(maxK: Int, batchSize: Int, featureSize: Int, fileDeviceMapping: Map<File, Int>, keyValueDelim: Char, vectorDelim: Char) : this(maxK, batchSize, featureSize, DEFAULT_DATA_TYPE, fileDeviceMapping, keyValueDelim, vectorDelim)


    /**
     * A utility class for conversions and mappings.
     */
    companion object {
        /**
         * Converts a string to a char.
         *
         * @param input The input string.
         * @return The converted char.
         * @throws IllegalArgumentException if the input string is not of length 1.
         */
        fun toChar(input: String): Char {
            require(input.length == 1) { "String: $input is not length 1, cannot convert to char" }
            return input[0]
        }

        /**
         * Creates a map with file-index pairs in the order of the input list.
         *
         * @param dataFiles The list of files.
         * @return The map with file-index pairs.
         */
        fun toMapInIndexOrder(dataFiles: List<File>?): Map<File, Int> {
            val mapping: MutableMap<File, Int> = LinkedHashMap()
            if (dataFiles != null) {
                for (i in dataFiles.indices) {
                    mapping[dataFiles[i]] = i
                }
            }
            return mapping
        }
    }

    /**
     * Initializes the component.
     */
    fun init() {
        log.info("Loading library: {}", LIBNAME)
        System.loadLibrary(LIBNAME)
        log.info("Loaded library: {}", LIBNAME)

        val files = Array(fileDeviceMapping.size) { "" }
        val devices = IntArray(fileDeviceMapping.size)
        val uniqueDevices = uniqueDevices()

        log.info("Initializing with maxK: {}, batchSize: {}, devices: {}", maxK, batchSize, uniqueDevices)
        ptr = initialize(maxK, batchSize, uniqueDevices, dataType.ordinal())

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

    /**
     * Retrieves the number of unique devices.
     * @return the number of unique devices.
     */
    private fun uniqueDevices(): Int {
        return fileDeviceMapping.values.toSet().size
    }

    /**
     * Finds the k nearest neighbors using the maximum value of k.
     * @param inputVectors the input vectors.
     * @param scores the output scores.
     * @param keys the output keys.
     */
    fun findKnn(inputVectors: FloatArray, scores: FloatArray, keys: Array<String>) {
        findKnn(maxK, inputVectors, featureSize, scores, keys)
    }

    /**
     * Finds the k nearest neighbors using the maximum value of k.
     * @param inputVectors the input vectors.
     * @param activeBatchSize the active batch size.
     * @param scores the output scores.
     * @param keys the output keys.
     */
    fun findKnn(inputVectors: FloatArray, activeBatchSize: Int, scores: FloatArray, keys: Array<String>) {
        findKnn(maxK, inputVectors, activeBatchSize, featureSize, scores, keys)
    }

    /**
     * Finds the k nearest neighbors.
     * @param k the number of nearest neighbors to find.
     * @param inputVectors the input vectors.
     * @param width the width of the input vectors.
     * @param scores the output scores.
     * @param keys the output keys.
     */
    fun findKnn(k: Int, inputVectors: FloatArray, width: Int, scores: FloatArray, keys: Array<String>) {
        findKnn(k, inputVectors, batchSize, width, scores, keys)
    }

    /**
     * Finds the k nearest neighbors.
     * @param k the number of nearest neighbors to find.
     * @param inputVectors the input vectors.
     * @param activeBatchSize the active batch size.
     * @param width the width of the input vectors.
     * @param scores the output scores.
     * @param keys the output keys.
     */
    fun findKnn(k: Int, inputVectors: FloatArray, activeBatchSize: Int, width: Int, scores: FloatArray, keys: Array<String>) {
        validateArguments(k, inputVectors, activeBatchSize, width)

        val outputLength = batchSize * maxK
        if (scores.size != outputLength) {
            throw IllegalArgumentException("Output scores array must be of size: $outputLength (batchSize x maxK).")
        }
        if (keys.size != outputLength) {
            throw IllegalArgumentException("Output keys array must be of size: $outputLength (batchSize x maxK).")
        }

        findKnn(k, inputVectors, batchSize, width, scores, keys, ptr)
    }

    /**
     * Finds the k nearest neighbors using the maximum value of k.
     * @param inputVectors the input vectors.
     * @return the result of the nearest neighbor search.
     */
    fun findKnn(inputVectors: FloatArray): KnnResult {
        return findKnn(maxK, inputVectors)
    }

    /**
     * Finds the k nearest neighbors.
     * @param k the number of nearest neighbors to find.
     * @param inputVectors the input vectors.
     * @return the result of the nearest neighbor search.
     */
    fun findKnn(k: Int, inputVectors: FloatArray): KnnResult {
        validateArguments(k, inputVectors, batchSize, featureSize)
        return findKnn(k, inputVectors, batchSize, featureSize, ptr)
    }

        /**
         * Validates the arguments for nearest neighbor search.
         * @param k the number of nearest neighbors to find.
         * @param inputVectors the input vectors.
         * @param activeBatchSize the active batch size.
         * @param width the width of the input vectors.
         * @throws IllegalArgumentException if the arguments are invalid.
         */
        private fun validateArguments(k: Int, inputVectors: FloatArray, activeBatchSize: Int, width: Int) {
            if (width < 1) {
                throw IllegalArgumentException("Dimension of the input vector should be at least one.")
            }
            if (k > maxK) {
                throw IllegalArgumentException("k = $k is greater than maxK = $maxK.")
            }
            if (k < 1) {
                throw IllegalArgumentException("k must be at least 1.")
            }
            if (inputVectors.size % width != 0) {
                throw IllegalArgumentException("Width: $width does not divide the vectors: ${inputVectors.size}.")
            }
            val actualBatchSize = inputVectors.size / width
            if (actualBatchSize != batchSize) {
                throw IllegalArgumentException("$actualBatchSize is not equal to configured batchSize: $batchSize.")
            }
            if (inputVectors.isEmpty()) {
                throw IllegalArgumentException("Input vector must contain at least one vector.")
            }
            if (activeBatchSize > batchSize) {
                throw IllegalArgumentException("Active batch size must be less than or equal to batchSize: $batchSize.")
            }
            if (activeBatchSize > actualBatchSize) {
                throw IllegalArgumentException("Active batch size must be less than or equal to actual batch size: $actualBatchSize.")
            }
        }
    }
    /**
     * Closes the resource.
     * @throws IOException if an I/O error occurs.
     */
    @Throws(IOException::class)
    override fun close() {
        if (ptr != 0L) {
            shutdown(ptr)
            ptr = 0L
        }
    }

    /**
     * Initializes the native component.
     * @param maxK the maximum value of k.
     * @param maxBatchSize the maximum batch size.
     * @param numGPUs the number of GPUs.
     * @param dataType the data type.
     * @return the initialized component pointer.
     */
    private external fun initialize(maxK: Int, maxBatchSize: Int, numGPUs: Int, dataType: Int): Long

    /**
     * Loads data into the native component.
     * @param filenames the filenames to load from.
     * @param devices the devices to load onto.
     * @param keyValDelim the key-value delimiter.
     * @param vectorDelim the vector delimiter.
     * @param ptr the component pointer.
     * @throws IOException if an I/O error occurs.
     */
    private external fun load(filenames: Array<String>, devices: IntArray, keyValDelim: Char, vectorDelim: Char, ptr: Long)

    /**
     * Shuts down the native component.
     * @param ptr the component pointer.
     */
    private external fun shutdown(ptr: Long)

    /**
     * Finds k nearest neighbors.
     * @param k the number of nearest neighbors to find.
     * @param input the input data.
     * @param size the size of the input data.
     * @param width the width of the input data.
     * @param scores the output scores.
     * @param keys the output keys.
     * @param ptr the component pointer.
     */
    private external fun findKnn(k: Int, input: FloatArray, size: Int, width: Int, scores: FloatArray, keys: Array<String>, ptr: Long)

    /**
     * Finds k nearest neighbors and returns the result.
     * @param k the number of nearest neighbors to find.
     * @param input the input data.
     * @param size the size of the input data.
     * @param width the width of the input data.
     * @param ptr the component pointer.
     * @return the result of the nearest neighbor search.
     */
    private external fun findKnn(k: Int, input: FloatArray, size: Int, width: Int, ptr: Long): KnnResult
}
