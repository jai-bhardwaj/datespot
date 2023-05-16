package com.system.tensorhub

import org.junit.Assert.fail
import java.io.BufferedReader
import java.io.Closeable
import java.io.File
import java.io.FileReader
import java.io.IOException
import java.util.ArrayList
import java.util.Arrays
import java.util.HashMap
import java.util.List
import java.util.Map
import org.junit.After
import org.junit.Before
import org.junit.Ignore
import org.junit.Test
import com.system.tensorhub.DataSetEnums.DataType
import com.system.tensorhub.data.DenseDataSet
import com.system.tensorhub.data.SparseDataSet

/**
 * Test class for Network functionality.
 * This class contains tests for various aspects of the Network class.
 */
@Ignore
class NetworkTest {
    private val DIR_SUFFIX = "src/test/java/com/system/eduwise/test-data/"

    private val networkFile = DIR_SUFFIX + ""

    private val inputFile = DIR_SUFFIX + ""

    private val indexFile = DIR_SUFFIX + ""

    private val expectedOutputFile = DIR_SUFFIX + ""

    private val k = 16

    private val batchSize = 256

    private val sparseDensity = 0.1

    private lateinit var config: NetworkConfig

    private lateinit var network: Network

    private lateinit var index: MutableMap<String, Long>

    private lateinit var rIndex: MutableMap<Long, String>

    private lateinit var expectedKeys: MutableList<Array<String>>

    private lateinit var expectedScores: MutableList<FloatArray>

    private val testDelta = 0.01f

    /**
     * Setup method executed before each test case.
     * This method initializes the network configuration, loads the network, parses the index file,
     * and reads the expected output for comparison.
     *
     * @throws IOException if there is an error reading the index or expected output files.
     */
    @Before
    fun setup() {
        config = NetworkConfig.with().networkFilePath(networkFile).batchSize(batchSize).k(k).build()

        network = Tensorhub.load(config)

        println("Loaded network: \n$network")

        index = HashMap()
        rIndex = HashMap()

        parseIndex(indexFile, index, rIndex)

        expectedKeys = ArrayList()
        expectedScores = ArrayList()

        readExpectedOutput(expectedKeys, expectedScores)
    }

    /**
     * Teardown method executed after each test case.
     * This method closes the network and clears the index and reverse index maps.
     */
    @After
    fun teardown() {
        network.close()

        index.clear()

        rIndex.clear()
    }

    /**
     * Test method for predicting with sparse input datasets.
     * This method sets up sparse input datasets, loads them into the network, and asserts predictions.
     */
    @Test
    fun testPredictSparse() {
        val inputDatasets = Array<DataSet?>(network.inputLayers.size) { null }

        for (i in network.inputLayers.indices) {
            val inputLayer = network.inputLayers[i]
            inputDatasets[i] = SparseDataSet(Dim(inputLayer.dim, batchSize), DataType.Int, sparseDensity)
            inputDatasets[i]?.name = inputLayer.datasetName
            inputDatasets[i]?.layerName = inputLayer.name
        }

        network.load(inputDatasets.requireNoNulls())

        println("Loaded ${inputDatasets.size} input datasets to network")

        val dp = SparseDataProvider(inputFile, index)

        assertPredictions(inputDatasets, dp)
    }

    /**
     * Assertion method for predictions.
     * This method compares the predicted scores and indexes with the expected values and performs assertions.
     *
     * @param inputDatasets Array of input datasets used for predictions.
     * @param dp DataProvider object for retrieving batches of data.
     */
    private fun assertPredictions(inputDatasets: Array<DataSet>, dp: DataProvider) {
        var batch = 0
        while (dp.getBatch(inputDatasets[0])) {
            val start = System.currentTimeMillis()
            val outputDatasets = network.predict(inputDatasets)
            val end = System.currentTimeMillis()
            println("==== Batch # ${batch + 1} Predict. Took ${end - start} ms ====")
            val output = outputDatasets[0]
            val scores = output.scores
            val indexes = output.indexes

            for (i in 0 until batchSize) {
                val pos = batch * batchSize + i
                val expectedKey = expectedKeys[pos]
                val expectedScr = expectedScores[pos]

                var failMessage: String? = null
                for (j in 0 until k) {
                    val actualKey = rIndex[indexes[i * k + j]]
                    val actualScore = scores[i * k + j]
                    if (expectedKey[j] != actualKey || Math.abs(expectedScr[j] - actualScore) > testDelta) {
                        failMessage = String.format(
                            "Unequal key or score at input %d, k=%d. Expected: %s,%5.3f Actual: %s,%5.3f",
                            pos + 1, j + 1, expectedKey[j], expectedScr[j], actualKey, actualScore
                        )
                    }
                }

                if (failMessage != null) {
                    println("== Actual ==")
                    print("${pos + 1}\t")
                    for (j in 0 until k) {
                        val actualKey = rIndex[indexes[i * k + j]]
                        val actualScore = scores[i * k + j]
                        print(String.format("%s,%1.3f:", actualKey, actualScore))
                    }
                    println()
                    println("== Expected ==")
                    print("${pos + 1}\t")
                    for (j in 0 until k) {
                        print(String.format("%s,%1.3f:", expectedKey[j], expectedScr[j]))
                    }
                    println()
                    fail(failMessage)
                }
            }
            batch++
        }
    }

    /**
     * Test method for predicting with dense input datasets.
     * This method sets up dense input datasets, loads them into the network, and asserts predictions.
     */
    @Ignore
    @Test
    fun testPredictDense() {
        val inputDatasets = arrayOfNulls<DataSet>(network.inputLayers.size)
        for (i in network.inputLayers.indices) {
            val inputLayer = network.inputLayers[i]
            inputDatasets[i] = DenseDataSet(Dim(inputLayer.dim, batchSize), DataType.Int)
            inputDatasets[i]!!.name = inputLayer.datasetName
            inputDatasets[i]!!.layerName = inputLayer.name
        }

        network.load(inputDatasets.requireNoNulls())
        println("Loaded ${inputDatasets.size} input datasets to network")

        val dp = DenseDataProvider(inputDatasets[0]!!.stride, inputFile, index)

        assertPredictions(inputDatasets, dp)
    }

    /**
     * Reads the expected output from the file and populates the indexes and scores lists.
     *
     * @param allIndexes List to store the indexes.
     * @param allScores List to store the scores.
     * @throws IOException if there is an error reading the expected output file.
     */
    private fun readExpectedOutput(allIndexes: MutableList<Array<String>>, allScores: MutableList<FloatArray>) {
        BufferedReader(FileReader(expectedOutputFile)).use { reader ->
            var line: String?
            while (reader.readLine().also { line = it } != null) {
                val kv = line!!.split("\t")
                val pos = kv[0].toInt()
                val vs = kv[1].split(":")
                val idx = Array(vs.size) { "" }
                val scores = FloatArray(vs.size)
                for (i in vs.indices) {
                    val v = vs[i]
                    val split = v.split(",")
                    idx[i] = split[0]
                    scores[i] = split[1].toFloat()
                }
                allScores.add(pos - 1, scores)
                allIndexes.add(pos - 1, idx)
            }
        }
    }

    /**
     * Parses the index file and populates the index and reverse index maps.
     *
     * @param indexFile The path to the index file.
     * @param index Mutable map to store the index.
     * @param rIndex Mutable map to store the reverse index.
     * @throws IOException if there is an error reading the index file.
     * @throws IllegalArgumentException if duplicate keys or indexes are found.
     */
    private fun parseIndex(indexFile: String, index: MutableMap<String, Long>, rIndex: MutableMap<Long, String>) {
        BufferedReader(FileReader(indexFile)).use { reader ->
            var line: String?
            while (reader.readLine().also { line = it } != null) {
                val split = line!!.split("\t")
                val key = split[0]
                val idx = split[1].toLong()
                if (index.put(key, idx) != null) {
                    throw IllegalArgumentException("Duplicate key found: $key")
                }
                if (rIndex.put(idx, key) != null) {
                    throw IllegalArgumentException("Duplicate index found: $idx")
                }
            }
        }
    }
    /**
     * Abstract class for providing data to a DataSet.
     * This class serves as a base class for specific data providers.
     *
     * @param inputFile The path to the input file.
     * @param indexMap A map containing index information.
     */
    abstract class DataProvider(private val inputFile: String, private val indexMap: Map<String, Long>) : Closeable {
        private val input: BufferedReader = BufferedReader(FileReader(inputFile))

        /**
         * Constructor for the DataProvider class.
         * Initializes the input file and reader.
         *
         * @param inputFile The path to the input file.
         * @param indexMap A map containing index information.
         * @throws IOException if there is an error reading the input file.
         */
        init {
            this.inputFile = File(inputFile)
            this.input = BufferedReader(FileReader(inputFile))
        }

        /**
         * Retrieves a batch of data and populates the provided DataSet.
         *
         * @param dataset The DataSet to populate with data.
         * @return true if there is more data available, false otherwise.
         * @throws IOException if there is an error reading the input file.
         */
        @Throws(IOException::class)
        fun getBatch(dataset: DataSet): Boolean {
            val numExamples = dataset.examples

            var eof = false
            for (i in 0 until numExamples) {
                val line = input.readLine()

                if (line == null && i == 0) {
                    eof = true
                    break
                } else if (line != null) {
                    val split = line.split("\t")[1].split(":")
                    val data = IntArray(split.size)
                    val index = LongArray(split.size)

                    for (j in split.indices) {
                        val key = split[j].split(",")[0]
                        val idx = indexMap[key]

                        if (idx == null) {
                            throw RuntimeException("No index found for key: $key")
                        }

                        data[j] = 1
                        index[j] = idx
                    }
                    addExample(dataset, i, index, data)
                }
            }
            return !eof
        }

        /**
         * Adds an example to the DataSet.
         * This method should be implemented in the derived classes.
         *
         * @param dataset The DataSet to add the example to.
         * @param idx The index of the example.
         * @param index The index array.
         * @param data The data array.
         * @throws IOException if there is an error adding the example to the DataSet.
         */
        @Throws(IOException::class)
        protected abstract fun addExample(dataset: DataSet, idx: Int, index: LongArray, data: IntArray)

        /**
         * Closes the input file.
         *
         * @throws IOException if there is an error closing the input file.
         */
        @Throws(IOException::class)
        override fun close() {
            input.close()
        }
    }

/**
 * DataProvider subclass for dense data.
 * This class provides dense data examples to a DataSet.
 *
 * @param stride The stride of the data buffer.
 * @param inputFile The path to the input file.
 * @param indexMap A map containing index information.
 */
class DenseDataProvider(stride: Int, inputFile: String, indexMap: Map<String, Long>) :
    DataProvider(inputFile, indexMap) {
    private val dataBuffer: IntArray = IntArray(stride)

    /**
     * Initializes the DenseDataProvider class.
     * Calls the superclass constructor and initializes the data buffer.
     *
     * @param stride The stride of the data buffer.
     * @param inputFile The path to the input file.
     * @param indexMap A map containing index information.
     * @throws IOException if there is an error initializing the data provider.
     */
    init {
        super(inputFile, indexMap)
    }

        /**
         * Adds a dense example to the DataSet.
         * Fills the data buffer with the example data and adds it to the DataSet.
         *
         * @param dataset The DataSet to add the example to.
         * @param idx The index of the example.
         * @param index The index array.
         * @param data The data array.
         * @throws IOException if there is an error adding the example to the DataSet.
         */
        override fun addExample(dataset: DataSet, idx: Int, index: LongArray, data: IntArray) {
            dataBuffer.fill(0)
            for (i in index.indices) {
                dataBuffer[index[i].toInt()] = data[i]
            }
            dataset.add(idx, dataBuffer)
        }
    }

    /**
     * DataProvider subclass for sparse data.
     * This class provides sparse data examples to a DataSet.
     *
     * @param inputFile The path to the input file.
     * @param indexMap A map containing index information.
     */
    class SparseDataProvider(inputFile: String, indexMap: Map<String, Long>) : DataProvider(inputFile, indexMap) {
        /**
         * Initializes the SparseDataProvider class.
         * Calls the superclass constructor.
         *
         * @param inputFile The path to the input file.
         * @param indexMap A map containing index information.
         * @throws IOException if there is an error initializing the data provider.
         */
        init {
            super(inputFile, indexMap)
        }

        /**
         * Adds a sparse example to the DataSet.
         * Adds the sparse example to the DataSet using the addSparse method.
         *
         * @param dataset The DataSet to add the example to.
         * @param idx The index of the example.
         * @param index The index array.
         * @param data The data array.
         * @throws IOException if there is an error adding the example to the DataSet.
         */
        override fun addExample(dataset: DataSet, idx: Int, index: LongArray, data: IntArray) {
            dataset.addSparse(idx, index, data)
        }
    }
}
