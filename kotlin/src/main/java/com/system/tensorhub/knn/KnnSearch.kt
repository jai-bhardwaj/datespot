package com.system.tensorhub.knn

import com.beust.jcommander.JCommander
import com.beust.jcommander.Parameter
import com.google.common.base.Stopwatch
import com.system.tensorhub.knn.DataUtil.Row
import lombok.extern.slf4j.Slf4j
import java.io.*

/**
 * The main method that performs the search.
 *
 * @param args the command-line arguments
 */
fun main(args: Array<String>) {
    val params = parseCommandLine(args)
    val featureSize = DataUtil.findFeatureSize(File(params.inputFiles.first()), params.keyValDelim, params.vectorDelim)
    log.info("Auto determined feature size = {} from file {}", featureSize, params.inputFiles.first())

    val scoreFormat = "%.${params.scorePrecision}f"
    val writerThread = Executors.newSingleThreadScheduledExecutor()
    val searchThread = Executors.newSingleThreadScheduledExecutor()

    BufferedReader(FileReader(params.inputFiles.joinToString(File.pathSeparator))).use { reader ->
        PrintWriter(createOutputStreamWriter(params.outputFile)).use { writer ->
            val knnCuda = KNearestNeighborsCuda(params.k, params.batchSize, featureSize,
                    DataType.fromString(params.dataType), params.dataFiles.map(::File),
                    params.keyValDelim, params.vectorDelim)
            knnCuda.init()

            log.info("Starting search. Reporting metrics every {} batches", params.reportInterval)
            val timer = Stopwatch.createStarted()
            var totalBatchTime = 0L
            var batchNum = 0L
            var line = reader.readLine()

            while (line != null) {
                val batchTimer = Stopwatch.createStarted()
                val inputRowIds = mutableListOf<String>()
                val inputVectors = mutableListOf<FloatArray>()

                do {
                    val row = DataUtil.parseLine(line, params.keyValDelim, params.vectorDelim)
                    inputRowIds.add(row.key)
                    inputVectors.add(row.vector)
                    line = reader.readLine()
                } while (inputVectors.size < params.batchSize && line != null)

                val activeBatchSize = inputVectors.size

                searchThread.submit {
                    val result = knnCuda.findKnn(inputVectors.toTypedArray())

                    writerThread.submit {
                        for (j in 0 until activeBatchSize) {
                            val inputRowId = inputRowIds[j]
                            writer.print(inputRowId)
                            writer.print(params.keyValDelim)

                            val score = result.getScoreAt(j, 0)
                            val key = result.getKeyAt(j, 0)
                            writer.print(key)
                            if (!params.outputKeysOnly) {
                                writer.print(params.idScoreSep)
                                writer.format(scoreFormat, score)
                            }

                            for (m in 1 until params.k) {
                                val score = result.getScoreAt(j, m)
                                val key = result.getKeyAt(j, m)

                                writer.print(params.vectorDelim)
                                writer.print(key)
                                if (!params.outputKeysOnly) {
                                    writer.print(params.idScoreSep)
                                    writer.format(scoreFormat, score)
                                }
                            }
                            writer.println()
                        }
                    }
                }

                val elapsedBatch = batchTimer.elapsed().toMillis()
                val elapsedTotal = timer.elapsed().toSeconds()
                totalBatchTime += elapsedBatch
                batchNum++

                if (batchNum % params.reportInterval == 0L) {
                    log.info("Processed %7d batches in %4ds. Elapsed %7ds. TPS %7d".format(
                            batchNum, totalBatchTime / SEC_IN_MS, elapsedTotal,
                            (batchNum * params.batchSize) / timer.elapsed().toSeconds()))
                    totalBatchTime = 0L
                }
            }

            searchThread.shutdown()
            searchThread.awaitTermination(Long.MAX_VALUE, TimeUnit.MILLISECONDS)
            writerThread.shutdown()
            writerThread.awaitTermination(Long.MAX_VALUE, TimeUnit.MILLISECONDS)

            val totalTime = timer.elapsed().toSeconds()
            log.info("Done processing {} batches in {} s".format(batchNum, totalTime))

            knnCuda.close()
        }
    }
}

/**
 * Parses the command line arguments using JCommander.
 *
 * @param args the command-line arguments
 * @return the parsed command-line parameters
 */
private fun parseCommandLine(args: Array<String>): Parameters {
    val params = Parameters()
    val jc = JCommander(params)
    jc.setProgramName("knn-search")

    try {
        jc.parse(*args)

        if (params.help) {
            jc.usage()
            System.exit(0)
        }
    } catch (e: Exception) {
        log.error("Error running command", e)
        jc.usage()
        System.exit(1)
    }

    return params
}

/**
 * Creates a writer for the output file or standard output.
 *
 * @param fileName the name of the output file, or null to use standard output
 * @return the created writer
 * @throws IOException if an I/O error occurs
 */
private fun createOutputStreamWriter(fileName: String?): Writer {
    val os: OutputStream = fileName?.let {
        File(it).apply {
            parentFile.mkdirs()
            createNewFile()
        }.outputStream()
    } ?: System.out
    return OutputStreamWriter(os, Charset.forName("UTF-8"))
}

/**
 * Command-line parameters for the KnnSearch application.
 */
data class Parameters(
    @Parameter(names = "--help", help = true)
    val help: Boolean = false,

    @Parameter(names = "--data-files", variableArity = true, description = "data vector file(s)", required = true)
    val dataFiles: List<String> = emptyList(),

    @Parameter(names = "--input-files", variableArity = true, description = "input vector file(s)", required = true)
    val inputFiles: List<String> = emptyList(),

    @Parameter(names = "--output-file", description = "file to write knn results to. stdout if not specified")
    val outputFile: String? = null,

    @Parameter(names = "--k", description = "k")
    val k: Int = 100,

    @Parameter(names = "--batch-size", description = "batch size")
    val batchSize: Int = 128,

    @Parameter(names = "--data-type", description = "loads the data as fp16 or fp32")
    val dataType: String = "fp32",

    @Parameter(names = "--key-val-delim", description = "delimiter for separating key-vector on each row")
    val keyValDelim: String = "\t",

    @Parameter(names = "--vec-delim", description = "delimiter for separating vector elements on each row")
    val vectorDelim: String = " ",

    @Parameter(names = "--id-score-sep", description = "separator for id:score in the output")
    val idScoreSep: String = ":",

    @Parameter(names = "--score-precision", description = "number of decimal points for output score's precision")
    val scorePrecision: Int = 3,

    @Parameter(names = "--output-keys-only", description = "do not output scores")
    val outputKeysOnly: Boolean = false,

    @Parameter(names = "--report-interval", description = "print performance metrics after n batches")
    val reportInterval: Int = 1000
) {
    companion object {
        private const val SEC_IN_MS = 1000
    }
}
