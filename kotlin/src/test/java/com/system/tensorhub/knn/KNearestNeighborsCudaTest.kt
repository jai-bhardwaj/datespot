package com.system.tensorhub.knn

import org.junit.Assert.assertEquals
import org.junit.Ignore
import org.junit.Test
import java.io.File
import java.io.IOException

/**
 * Test class for KNearestNeighborsCuda.
 */
class KNearestNeighborsCudaTest {

    /**
     * Companion object with utility functions.
     */
    companion object {

        /**
         * Prints a matrix of floats.
         *
         * @param data The matrix data.
         * @param rows The number of rows in the matrix.
         * @param cols The number of columns in the matrix.
         */
        fun printMatrix(data: FloatArray, rows: Int, cols: Int) {
            repeat(cols) {
                print("----------\t")
            }
            println()
            repeat(cols) {
                System.out.format("%9d|\t", it)
            }
            println()

            repeat(cols) {
                print("----------\t")
            }
            println()

            for (i in 0 until rows) {
                for (j in 0 until cols) {
                    System.out.format("%10.3f\t", data[i * cols + j])
                }
                println()
            }
            repeat(cols) {
                print("----------\t")
            }
            println()
        }

        /**
         * Prints a matrix of strings.
         *
         * @param data The matrix data.
         * @param rows The number of rows in the matrix.
         * @param cols The number of columns in the matrix.
         */
        fun printMatrix(data: Array<String>, rows: Int, cols: Int) {
            repeat(cols) {
                print("----------\t")
            }
            println()
            repeat(cols) {
                System.out.format("%9d|\t", it)
            }
            println()

            repeat(cols) {
                print("----------\t")
            }
            println()

            for (i in 0 until rows) {
                for (j in 0 until cols) {
                    System.out.format("%10s\t", data[i * cols + j])
                }
                println()
            }
            repeat(cols) {
                print("----------\t")
            }
            println()
        }
    }

    /**
     * Test method for delimiter.
     *
     * @throws IOException If an I/O error occurs.
     */
    @Ignore
    @Test
    @Throws(IOException::class)
    fun testDelimiter() {
        val maxK = 8
        val batchSize = 16
        val featureSize = 256
        val fileToDevice: MutableMap<File, Int> = HashMap()
        fileToDevice[File("tests/system/cuda/algorithms/data/data-delim.txt")] = 0

        val keyValueDelim = ':'
        val vectorDelim = ','

        val knnCuda = KNearestNeighborsCuda(maxK, batchSize, featureSize, DataType.FP32, fileToDevice, keyValueDelim, vectorDelim)
        knnCuda.init()

        val batchInputs = FloatArray(batchSize * featureSize)
        val batchScores = FloatArray(batchSize * maxK)
        val batchIds = arrayOfNulls<String>(batchSize * maxK)

        knnCuda.findKnn(batchInputs, batchScores, batchIds)

        printMatrix(batchScores, batchSize, maxK)
        printMatrix(batchIds, batchSize, maxK)

        knnCuda.close()
    }

    /**
     * Test method for converting string to char.
     */
    @Test
    fun testToChar() {
        assertEquals('\t', KNearestNeighborsCuda.toChar("\t"))
    }

    /**
     * Test method for zero-length string conversion to char.
     *
     * @throws IllegalArgumentException If the string length is zero.
     */
    @Test(expected = IllegalArgumentException::class)
    fun testToChar_zero_length() {
        KNearestNeighborsCuda.toChar("")
    }

    /**
     * Test method for string conversion to char with length greater than one.
     *
     * @throws IllegalArgumentException If the string length is greater than one.
     */
    @Test(expected = IllegalArgumentException::class)
    fun testToChar_length_greater_than_one() {
        KNearestNeighborsCuda.toChar("\t ")
    }

    /**
     * Test method for DataType.fromString with a non-existing enum value.
     *
     * @throws IllegalArgumentException If the string does not match any enum value.
     */
    @Test(expected = IllegalArgumentException::class)
    fun testDataTypeFromString_NoSuchEnum() {
        DataType.fromString("fp64")
    }
}
