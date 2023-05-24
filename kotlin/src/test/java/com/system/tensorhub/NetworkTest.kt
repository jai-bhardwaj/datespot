package com.system.tensorhub

import org.junit.After
import org.junit.Before
import org.junit.Ignore
import org.junit.Test
import java.io.BufferedReader
import java.io.Closeable
import java.io.File
import java.io.FileReader
import java.io.IOException

/**
 * Test class for the Network class.
 */
class NetworkTest {
    private val DIR_SUFFIX = "src/test/java/com/system/tensorhub/test-data/"
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
     */
    @Before
    fun setup() {
        config = NetworkConfig.with()
            .networkFilePath(networkFile)
            .batchSize(batchSize)
            .k(k)
            .build()
        network = tensorhub.load(config)
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
     */
    @After
    fun teardown() {
        network.close()
        index.clear()
        rIndex.clear()
    }

    /**
     * Test case for predicting using sparse input data.
     */
    @Test
    fun testPredictSparse() {
        val inputDatasets = arrayOfNulls<DataSet>(network.inputLayers.size)
        for (i in network.inputLayers.indices) {
            val inputLayer = network.inputLayers[i]
            inputDatasets[i] = SparseDataSet(
                Dim(inputLayer.dim, batchSize),
                DataType.Int,
                sparseDensity
            )
            inputDatasets[i]?.name = inputLayer.datasetName
            inputDatasets[i]?.layerName = inputLayer.name
        }

        network.load(inputDatasets)
        println("Loaded ${inputDatasets.size} input datasets to network")

        val dp = SparseDataProvider(inputFile, index)
        assertPredictions(inputDatasets, dp)
    }

    /**
     * Helper method to assert the predictions of the network.
     *
     * @param inputDatasets Array of input datasets.
     * @param dp DataProvider instance.
     */
    private fun assertPredictions(inputDatasets: Array<DataSet?>, dp: DataProvider) {
        var batch = 0
        while (dp.getBatch(inputDatasets[0])) {
            val start = System.currentTimeMillis()
            val outputDatasets = network.predict(inputDatasets.requireNoNulls())
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
                        failMessage =
                            String.format(
                                "Unequal key or score at input %d, k=%d. Expected: %s,%5.3f Actual: %s,%5.3f",
                                pos + 1,
                                j + 1,
                                expectedKey[j],
                                expectedScr[j],
                                actualKey,
                                actualScore
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
     * Ignored test case for predicting using dense input data.
     */
    @Ignore
    @Test
    fun testPredictDense() {
        val inputDatasets = arrayOfNulls<DataSet>(network.inputLayers.size)
        for (i in network.inputLayers.indices) {
            val inputLayer = network.inputLayers[i]
            inputDatasets[i] = DenseDataSet(
                Dim(inputLayer.dim, batchSize),
                DataType.Int
            )
            inputDatasets[i]?.name = inputLayer.datasetName
            inputDatasets[i]?.layerName = inputLayer.name
        }

        network.load(inputDatasets)
        println("Loaded ${inputDatasets.size} input datasets to network")

        val dp = DenseDataProvider(inputDatasets[0]?.stride!!, inputFile, index)

        assertPredictions(inputDatasets, dp)
    }

    /**
     * Helper method to read the expected output from a file.
     *
     * @param allIndexes List to store the expected indexes.
     * @param allScores List to store the expected scores.
     */
    private fun readExpectedOutput(allIndexes: MutableList<Array<String>>, allScores: MutableList<FloatArray>) {
        BufferedReader(FileReader(expectedOutputFile)).use { reader ->
            var line: String?
            while (reader.readLine().also { line = it } != null) {
                val kv = line!!.split("\t")
                val pos = kv[0].toInt()
                val vs = kv[1].split(":")
                val idx = arrayOfNulls<String>(vs.size)
                val scores = FloatArray(vs.size)
                for (i in vs.indices) {
                    val v = vs[i]
                    idx[i] = v.split(",")[0]
                    scores[i] = v.split(",")[1].toFloat()
                }
                allScores.add(pos - 1, scores)
                allIndexes.add(pos - 1, idx.requireNoNulls())
            }
        }
    }

    /**
     * Helper method to parse the index file and populate the index and reverse index maps.
     *
     * @param indexFile Path to the index file.
     * @param index Mutable map to store the index.
     * @param rIndex Mutable map to store the reverse index.
     * @throws IOException If an I/O error occurs while reading the index file.
     */
    @Throws(IOException::class)
    private fun parseIndex(indexFile: String, index: MutableMap<String, Long>, rIndex: MutableMap<Long, String>) {
        BufferedReader(FileReader(indexFile)).use { reader ->
            var line: String?
            while (reader.readLine().also { line = it } != null) {
                val split = line!!.split("\t")
                val key = split[0]
                val idx = split[1].toLong()
                require(index.put(key, idx) == null) { "Duplicate key found: $key" }
                require(rIndex.put(idx, key) == null) { "Duplicate index found: $idx" }
            }
        }
    }

    /**
     * Abstract class representing a data provider.
     *
     * @param inputFile Path to the input file.
     * @param indexMap Map containing the indexes.
     */
    internal abstract class DataProvider(private val inputFile: String, private val indexMap: Map<String, Long>) :
        Closeable {
        private val input: BufferedReader = File(inputFile).bufferedReader()

        /**
         * Get a batch of data for the specified dataset.
         *
         * @param dataset The dataset to populate with the batch data.
         * @return True if a batch was successfully read, false if reached the end of the file.
         */
        abstract fun getBatch(dataset: DataSet): Boolean

        /**
         * Add an example to the dataset.
         *
         * @param dataset The dataset to add the example to.
         * @param idx The index of the example.
         * @param index The indexes for the example.
         * @param data The data values for the example.
         */
        protected abstract fun addExample(dataset: DataSet, idx: Int, index: LongArray, data: IntArray)

        /**
         * Close the data provider.
         *
         * @throws IOException If an I/O error occurs while closing the data provider.
         */
        @Throws(IOException::class)
        override fun close() {
            input.close()
        }

        init {
            input.useLines { lines ->
                for (line in lines) {
                    val split = line.split("\t")[1].split(":")
                    val data = IntArray(split.size)
                    val index = LongArray(split.size)

                    for (j in split.indices) {
                        val key = split[j].split(",")[0]
                        val idx = indexMap[key] ?: throw RuntimeException("No index found for key: $key")

                        data[j] = 1
                        index[j] = idx
                    }
                    addExample(dataSet, i, index, data)
                }
            }
            return !eof
        }
    }

    /**
     * Data provider implementation for dense input data.
     *
     * @param stride The stride of the dataset.
     * @param inputFile Path to the input file.
     * @param indexMap Map containing the indexes.
     */
    internal class DenseDataProvider(stride: Int, inputFile: String, indexMap: Map<String, Long>) :
        DataProvider(inputFile, indexMap) {
        private val dataBuffer = IntArray(stride)

        override fun addExample(dataset: DataSet, idx: Int, index: LongArray, data: IntArray) {
            dataBuffer.fill(0)
            for (i in index.indices) {
                dataBuffer[index[i].toInt()] = data[i]
            }
            dataset.add(idx, dataBuffer)
        }
    }

    /**
     * Data provider implementation for sparse input data.
     *
     * @param inputFile Path to the input file.
     * @param indexMap Map containing the indexes.
     */
    internal class SparseDataProvider(inputFile: String, indexMap: Map<String, Long>) :
        DataProvider(inputFile, indexMap) {
        override fun addExample(dataset: DataSet, idx: Int, index: LongArray, data: IntArray) {
            dataset.addSparse(idx, index, data)
        }
    }
}
