package com.system.tensorhub.knn

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Unit tests for the [KnnResult] class.
 */
class KnnResultTest {

    /**
     * Test case for handling invalid results.
     */
    @Test(expected = IllegalArgumentException::class)
    fun testInvalidResults() {
        KnnResult(arrayOf<String>(1), FloatArray(2), 1)
    }

    /**
     * Test case for handling invalid K value.
     */
    @Test(expected = IllegalArgumentException::class)
    fun testInvalidK() {
        KnnResult(arrayOf<String>(3), FloatArray(3), 2)
    }

    /**
     * Test case for getting the K value of a [KnnResult] object.
     */
    @Test
    fun testGetK() {
        val result = KnnResult(arrayOf<String>(6), FloatArray(6), 3)

        assertEquals(3, result.k)
    }

    /**
     * Test case for getting the batch size of a [KnnResult] object.
     */
    @Test
    fun testGetBatchSize() {
        val result = KnnResult(arrayOf<String>(32), FloatArray(32), 8)

        assertEquals(32 / 8, result.batchSize)
    }
}
