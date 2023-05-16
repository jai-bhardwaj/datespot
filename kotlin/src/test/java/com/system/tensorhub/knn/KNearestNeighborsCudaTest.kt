package com.system.tensorhub.knn

import org.junit.Assert.assertEquals
import java.io.File
import java.io.IOException

/**
 * Test class for the KNearestNeighborsCuda class.
 */
class KNearestNeighborsCudaTest {

    /**
     * Companion object that contains utility methods for printing matrices.
     */
    companion object {

        /**
         * Prints the given float array as a matrix with the specified number of rows and columns.
         *
         * @param data The float array representing the matrix data.
         * @param rows The number of rows in the matrix.
         * @param cols The number of columns in the matrix.
         */
        fun printMatrix(data: FloatArray, rows: Int, cols: Int) {
            // Printing column separators
            for (j in 0 until cols) {
                print("----------\t")
            }
            println()

            // Printing column indices
            for (j in 0 until cols) {
                System.out.format("%9d|\t", j)
            }
            println()

            // Printing separator lines
            for (j in 0 until cols) {
                print("----------\t")
            }
            println()

            // Printing matrix data
            for (i in 0 until rows) {
                for (j in 0 until cols) {
                    System.out.format("%10.3f\t", data[i * cols + j])
                }
                println()
            }

            // Printing separator lines
            for (j in 0 until cols) {
                print("----------\t")
            }
            println()
        }

        /**
         * Prints the given array of strings as a matrix with the specified number of rows and columns.
         *
         * @param data The array of strings representing the matrix data.
         * @param rows The number of rows in the matrix.
         * @param cols The number of columns in the matrix.
         */
        fun printMatrix(data: Array<String>, rows: Int, cols: Int) {
            // Printing column separators
            for (j in 0 until cols) {
                print("----------\t")
            }
            println()

            // Printing column indices
            for (j in 0 until cols) {
                System.out.format("%9d|\t", j)
            }
            println()

            // Printing separator lines
            for (j in 0 until cols) {
                print("----------\t")
            }
            println()

            // Printing matrix data
            for (i in 0 until rows) {
                for (j in 0 until cols) {
                    System.out.format("%10s\t", data[i * cols + j])
                }
                println()
            }

            // Printing separator lines
            for (j in 0 until cols) {
                print("----------\t")
            }
            println()
        }
    }

    /**
     * Test case for the 'testDelimiter' method in the KNearestNeighborsCuda class.
     * This test is currently ignored and will not be executed.
     * It throws an IOException.
     *
     * This test case validates the functionality of the 'testDelimiter' method,
     * which performs operations related to delimiter handling in the KNearestNeighborsCuda class.
     */
    @org.junit.Ignore
    @org.junit.Test
    @Throws(IOException::class)
    fun testDelimiter() {
        val maxK = 8
        val batchSize = 16
        val featureSize = 256

        // Mapping of files to device IDs
        val fileToDevice = HashMap<File, Int>()
        fileToDevice[File("tst/system/cuda/algorithms/data/data-delim.txt")] = 0

        val keyValueDelim = ':' // Delimiter for key-value pairs
        val vectorDelim = ',' // Delimiter for vectors

        // Create an instance of KNearestNeighborsCuda
        val knnCuda = KNearestNeighborsCuda(
            maxK,
            batchSize,
            featureSize,
            DataType.FP32,
            fileToDevice,
            keyValueDelim,
            vectorDelim
        )
        knnCuda.init()

        val batchInputs = FloatArray(batchSize * featureSize)
        val batchScores = FloatArray(batchSize * maxK)
        val batchIds = arrayOfNulls<String>(batchSize * maxK)

        // Find k-nearest neighbors using CUDA implementation
        knnCuda.findKnn(batchInputs, batchScores, batchIds)

        // Print the score matrix
        printMatrix(batchScores, batchSize, maxK)

        // Print the ID matrix
        printMatrix(batchIds, batchSize, maxK)

        knnCuda.close()
    }

    /**
     * Test case for the 'toChar' method in the KNearestNeighborsCuda class.
     * It asserts that the method returns the expected character ('\t') when given the input "\t".
     */
    @org.junit.Test
    fun testToChar() {
        assertEquals('\t', KNearestNeighborsCuda.toChar("\t"))
    }

    /**
     * Test case for the 'toChar' method in the KNearestNeighborsCuda class with zero-length input.
     * It expects an IllegalArgumentException to be thrown when an empty string is provided as input.
     */
    @org.junit.Test(expected = IllegalArgumentException::class)
    fun testToChar_zero_length() {
        KNearestNeighborsCuda.toChar("")
    }

    /**
     * Test case for the 'toChar' method in the KNearestNeighborsCuda class with input length greater than one.
     * It expects an IllegalArgumentException to be thrown when the input is "\t ".
     */
    @org.junit.Test(expected = IllegalArgumentException::class)
    fun testToChar_length_greater_than_one() {
        KNearestNeighborsCuda.toChar("\t ")
    }

    /**
     * Test case for the 'dataTypeFromString' method in the KNearestNeighborsCuda class with a non-existing enum value.
     * It expects an IllegalArgumentException to be thrown when the input is "fp64".
     */
    @org.junit.Test(expected = IllegalArgumentException::class)
    fun testDataTypeFromString_NoSuchEnum() {
        DataType.fromString("fp64")
    }
}

