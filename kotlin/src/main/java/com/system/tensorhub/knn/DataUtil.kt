package com.system.tensorhub.knn

import java.io.BufferedReader
import java.io.File
import java.io.FileInputStream
import java.io.InputStreamReader
import java.nio.charset.Charset

/**
 * Utility class for working with data.
 */
object DataUtil {

    /**
     * Finds the size of the feature vector based on multiple data files.
     *
     * @param dataFiles The collection of data files.
     * @param keyValDelim The delimiter used to separate key-value pairs in each line.
     * @param vectorDelim The delimiter used to separate elements within a vector.
     * @return The size of the feature vector.
     * @throws IllegalArgumentException if the collection of data files is empty or if feature sizes are different.
     */
    fun findFeatureSize(dataFiles: Collection<File>, keyValDelim: String, vectorDelim: String): Int {
        require(dataFiles.isNotEmpty()) { "Data files collection is empty, must pass at least one file" }

        val dataToFeatureSize = LinkedHashMap<File, Int>()

        for (dataFile in dataFiles) {
            val featureSize = findFeatureSize(dataFile, keyValDelim, vectorDelim)
            dataToFeatureSize[dataFile] = featureSize
        }

        val distinctFeatureSizes = dataToFeatureSize.values.distinct()
        require(distinctFeatureSizes.size == 1) { "Feature sizes are different in data files: $dataToFeatureSize" }

        return distinctFeatureSizes.first()
    }

    /**
     * Finds the size of the feature vector based on the data file.
     *
     * @param dataFile The file containing the data.
     * @param keyValDelim The delimiter used to separate key-value pairs in each line.
     * @param vectorDelim The delimiter used to separate elements within a vector.
     * @return The size of the feature vector.
     * @throws IllegalArgumentException if the data file is empty.
     */
    fun findFeatureSize(dataFile: File, keyValDelim: String, vectorDelim: String): Int {
        BufferedReader(InputStreamReader(FileInputStream(dataFile), Charset.forName("UTF-8"))).use { reader ->
            val line = reader.readLine()
            require(line != null) { "File: $dataFile is empty" }
            return findFeatureSize(line, keyValDelim, vectorDelim)
        }
    }

    /**
     * Finds the size of the feature vector based on a line of data.
     *
     * @param line The line of data.
     * @param keyValDelim The delimiter used to separate key-value pairs in the line.
     * @param vectorDelim The delimiter used to separate elements within a vector.
     * @return The size of the feature vector.
     * @throws IllegalArgumentException if the line contains a malformed key-value pair or vector.
     */
    fun findFeatureSize(line: String, keyValDelim: String, vectorDelim: String): Int {
        val row = parseLine(line, keyValDelim, vectorDelim)
        return row.vector.size
    }

    /**
     * Parses a line of data to extract the key and vector information.
     *
     * @param line The line of data.
     * @param keyValDelim The delimiter used to separate key-value pairs in the line.
     * @param vectorDelim The delimiter used to separate elements within a vector.
     * @return A Row object containing the key and vector.
     * @throws IllegalArgumentException if the line contains a malformed key-value pair or vector.
     */
    fun parseLine(line: String, keyValDelim: String, vectorDelim: String): Row {
        val keyValue = line.split(keyValDelim)
        require(keyValue.size == 2) { "Malformed key-value pair in line: $line" }

        val vectorLiteral = keyValue[1].split(vectorDelim)
        require(vectorLiteral.isNotEmpty()) { "Malformed vector in line: $line" }

        val vector = vectorLiteral.map { it.toFloat() }.toFloatArray()

        return Row(keyValue[0], vector)
    }

    /**
     * Represents a row of data containing a key and a vector.
     *
     * @property key The key.
     * @property vector The vector.
     */
    data class Row(val key: String, val vector: FloatArray)
}
