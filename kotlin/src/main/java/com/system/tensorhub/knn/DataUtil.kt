package com.system.tensorhub.knn

import java.io.BufferedReader
import java.io.File
import java.io.FileInputStream
import java.io.InputStreamReader
import java.nio.charset.Charset

object DataUtil {

    fun findFeatureSize(dataFiles: Collection<File>, keyValDelim: String, vectorDelim: String): Int {
        require(dataFiles.isNotEmpty()) { "data files is empty, must pass at least one file" }

        val dataToFeatureSize = LinkedHashMap<File, Int>()

        for (dataFile in dataFiles) {
            val featureSize = findFeatureSize(dataFile, keyValDelim, vectorDelim)
            dataToFeatureSize[dataFile] = featureSize
        }

        var featureSize = 0
        for (fs in dataToFeatureSize.values) {
            if (featureSize == 0) {
                featureSize = fs
            } else {
                require(featureSize == fs) { "Feature sizes are different in data files: $dataToFeatureSize" }
            }
        }
        return featureSize
    }

    fun findFeatureSize(dataFile: File, keyValDelim: String, vectorDelim: String): Int {
        BufferedReader(InputStreamReader(FileInputStream(dataFile), Charset.forName("UTF-8"))).use { reader ->
            val line = reader.readLine()
            require(line != null) { "file: $dataFile is empty" }
            return findFeatureSize(line, keyValDelim, vectorDelim)
        }
    }

    fun findFeatureSize(line: String, keyValDelim: String, vectorDelim: String): Int {
        val row = parseLine(line, keyValDelim, vectorDelim)
        return row.vector.size
    }

    fun parseLine(line: String, keyValDelim: String, vectorDelim: String): Row {
        val keyValue = line.split(keyValDelim)
        require(keyValue.size == 2) { "Malformed key-value pair in line: $line" }

        val vectorLiteral = keyValue[1].split(vectorDelim)
        require(vectorLiteral.isNotEmpty()) { "Malformed vector in line: $line" }

        val vector = FloatArray(vectorLiteral.size)
        for (i in vector.indices) {
            vector[i] = vectorLiteral[i].toFloat()
        }

        return Row(keyValue[0], vector)
    }

    data class Row(val key: String, val vector: FloatArray)
}
