package com.system.tensorhub.knn

import java.io.BufferedReader
import java.io.BufferedWriter
import java.io.Closeable
import java.io.File
import java.io.FileInputStream
import java.io.FileNotFoundException
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader
import java.io.OutputStream
import java.io.OutputStreamWriter
import java.io.PrintWriter
import java.io.SequenceInputStream
import java.nio.charset.Charset
import java.util.ArrayList
import java.util.Collections
import java.util.List
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

import com.system.tensorhub.knn.DataUtil.Row

import com.beust.jcommander.JCommander
import com.beust.jcommander.Parameter
import com.google.common.base.Stopwatch
import lombok.extern.slf4j.Slf4j

/**
 * KnnSearch class for performing k-nearest neighbor search.
 */
@Slf4j
class KnnSearch(private val args: Array<String>) : Closeable {

    companion object {
        private const val SEC_IN_MS = 1000
    }

    @Parameter(names = ["--help"], help = true)
    private var help = false

    @Parameter(names = ["--data-files"], variableArity = true, description = "data vector file(s)", required = true)
    private var dataFileNames: MutableList<String> = ArrayList()

    @Parameter(names = ["--input-files"], variableArity = true, description = "input vector file(s)", required = true)
    private var inputFileNames: MutableList<String> = ArrayList()

    @Parameter(names = ["--output-file"], description = "file to write knn results to. stdout if not specified")
    private var outputFileName: String? = null

    @Parameter(names = ["--k"], description = "k")
    private var k = 100

    @Parameter(names = ["--batch-size"], description = "batch size")
    private var batchSize = 128

    @Parameter(names = ["--data-type"], description = "loads the data as fp16 or fp32")
    private var dataType = "fp32"

    @Parameter(names = ["--key-val-delim"], description = "delimiter for separating key-vector on each row")
    private var keyValDelim = "\t"

    @Parameter(names = ["--vec-delim"], description = "delimiter for separating vector elements on each row")
    private var vectorDelim = " "

    @Parameter(names = ["--id-score-sep"], description = "separator for id:score in the output")
    private var idScoreSep = ":"

    @Parameter(names = ["--score-precision"], description = "number of decimal points for output score's precision")
    private var scorePrecision = 3

    @Parameter(names = ["--output-keys-only"], description = "do not output scores")
    private var outputKeysOnly = false

    @Parameter(names = ["--report-interval"], description = "print performance metrics after n batches")
    private var reportInterval = 1000

    private var featureSize: Int = 0
    private var scoreFormat: String = ""
    private var writerThread: ExecutorService? = null
    private var searchThread: ExecutorService? = null

    private var reader: BufferedReader? = null
    private var writer: PrintWriter? = null

    private var knnCuda: KNearestNeighborsCuda? = null

    /**
     * Initializes and configures the KnnSearch object.
     */
    init {
        val jc = JCommander(this)
        jc.setProgramName("knn-search")

        try {
            jc.parse(*args)

            if (help) {
                jc.usage()
                return
            }
        } catch (e: Exception) {
            log.error("Error running command", e)
            jc.usage()
            return
        }

        val inputFile = inputFileNames[0]
        featureSize = DataUtil.findFeatureSize(File(inputFile), keyValDelim, vectorDelim)
        log.info("Auto determined feature size = {} from file {}", featureSize, inputFile)

        scoreFormat = "%." + scorePrecision + "f"
        writerThread = Executors.newSingleThreadScheduledExecutor()
        searchThread = Executors.newSingleThreadScheduledExecutor()
        reader = createReader()
        writer = createWriter()

        knnCuda = KNearestNeighborsCuda(k, batchSize, featureSize, DataType.fromString(dataType),
                toFile(dataFileNames), keyValDelim, vectorDelim)
        knnCuda!!.init()
    }

    /**
     * Runs the k-nearest neighbor search process.
     */
    private fun runKnnSearch() {
        log.info("Starting search. Reporting metrics every {} batches", reportInterval)

        val timer = Stopwatch.createStarted()
        var totalBatchTime: Long = 0

        var batchNum: Long = 0
        var line = reader!!.readLine()
        while (line != null) {
            val batchTimer = Stopwatch.createStarted()

            val inputRowIds = arrayOfNulls<String>(batchSize)
            val inputVectors = FloatArray(batchSize * featureSize)

            var i = 0
            do {
                val row = DataUtil.parseLine(line, keyValDelim, vectorDelim)
                inputRowIds[i] = row.key
                val vector = row.vector
                System.arraycopy(vector, 0, inputVectors, i * featureSize, featureSize)
                i++
                line = reader!!.readLine()
            } while (i < batchSize && line != null)

            val activeBatchSize = i

            searchThread!!.submit {
                val result = knnCuda!!.findKnn(inputVectors)

                writerThread!!.submit {
                    for (j in 0 until activeBatchSize) {
                        val inputRowId = inputRowIds[j]
                        writer!!.print(inputRowId)
                        writer!!.print(keyValDelim)

                        var score = result.getScoreAt(j, 0)
                        val key = result.getKeyAt(j, 0)
                        writer!!.print(key)
                        if (!outputKeysOnly) {
                            writer!!.print(idScoreSep)
                            writer!!.format(scoreFormat, score)
                        }

                        for (m in 1 until k) {
                            score = result.getScoreAt(j, m)
                            val key = result.getKeyAt(j, m)

                            writer!!.print(vectorDelim)
                            writer!!.print(key)
                            if (!outputKeysOnly) {
                                writer!!.print(idScoreSep)
                                writer!!.format(scoreFormat, score)
                            }
                        }
                        writer!!.println()
                    }
                }
            }

            val elapsedBatch = batchTimer.elapsed(TimeUnit.MILLISECONDS)
            val elapsedTotal = timer.elapsed(TimeUnit.SECONDS)
            totalBatchTime += elapsedBatch

            ++batchNum

            if (batchNum % reportInterval == 0L) {
                log.info(String.format("Processed %7d batches in %4ds. Elapsed %7ds. TPS %7d", batchNum,
                        totalBatchTime / SEC_IN_MS, elapsedTotal,
                        (batchNum * batchSize) / timer.elapsed(TimeUnit.SECONDS)))
                totalBatchTime = 0
            }
        }

        try {
            searchThread!!.shutdown()
            searchThread!!.awaitTermination(Long.MAX_VALUE, TimeUnit.MILLISECONDS)
            writerThread!!.shutdown()
            writerThread!!.awaitTermination(Long.MAX_VALUE, TimeUnit.MILLISECONDS)
        } catch (e: InterruptedException) {
            throw RuntimeException(e)
        }

        val totalTime = timer.elapsed(TimeUnit.SECONDS)

        log.info("Done processing {} batches in {} s", batchNum, totalTime)
    }

    override fun close() {
        reader?.close()
        writer?.close()
        knnCuda?.close()
    }

    private fun createReader(): BufferedReader {
        val fis: MutableList<InputStream> = ArrayList()
        for (fileName in inputFileNames) {
            fis.add(FileInputStream(fileName))
        }
        val `is`: InputStream = SequenceInputStream(Collections.enumeration(fis))
        return BufferedReader(InputStreamReader(`is`, Charset.forName("UTF-8")))
    }

    private fun createWriter(): PrintWriter {
        val os: OutputStream
        os = if (outputFileName == null) {
            System.out
        } else {
            val outputFile = File(outputFileName!!)
            outputFile.parentFile.mkdirs()
            outputFile.createNewFile()
            FileOutputStream(outputFileName!!)
        }
        return PrintWriter(BufferedWriter(OutputStreamWriter(os, Charset.forName("UTF-8"))))
    }

    private fun toFile(fileNames: List<String>): List<File> {
        val files: MutableList<File> = ArrayList()
        for (fileName in fileNames) {
            files.add(File(fileName))
        }
        return files
    }

    /**
     * Parses the command-line arguments and handles the help option.
     */
    private fun parseCommandLine() {
        try {
            JCommander(this).parse(*args)
            if (help) {
                JCommander(this).usage()
                return
            }
        } catch (e: Exception) {
            log.error("Error running command", e)
            JCommander(this).usage()
            return
        }
    }

    /**
     * The main entry point of the KnnSearch application.
     */
    companion object {
        /**
         * Runs the KnnSearch application.
         * @param args The command-line arguments.
         */
        @JvmStatic
        fun main(args: Array<String>) {
            val knnSearch = KnnSearch(args)
            knnSearch.parseCommandLine()
            knnSearch.runKnnSearch()
            knnSearch.close()
        }
    }
}
