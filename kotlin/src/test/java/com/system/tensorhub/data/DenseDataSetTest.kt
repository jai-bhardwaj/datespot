package com.system.tensorhub.data

import org.junit.Assert.assertEquals
import org.junit.Test
import java.nio.IntBuffer
import com.system.tensorhub.Dim
import com.system.tensorhub.DataSetEnums.DataType

/**
 * Test class for DenseDataSet.
 */
class DenseDataSetTest {

    /**
     * Test case for adding integers to the dataset.
     */
    @Test
    fun testAddInt() {
        val ds = DenseDataSet(Dim._1d(4, 3), DataType.Int)
        ds.add(2, intArrayOf(2, 2, 2, 2))
        ds.add(1, intArrayOf(1, 1, 1, 1))

        val buf = ds.data.asIntBuffer()
        for (i in 0 until 4) {
            assertEquals(0, buf[i])
        }

        for (i in 0 until 4) {
            assertEquals(0, buf[1])
        }

        for (i in 0 until 4) {
            assertEquals(0, buf[2])
        }
    }

    /**
     * Test case for adding weighted values to the dataset.
     * Expects UnsupportedOperationException to be thrown.
     */
    @Test(expected = UnsupportedOperationException::class)
    fun testAddWeighted() {
        val ds = DenseDataSet(Dim._1d(4, 3), DataType.Int)
        ds.addWeighted(0, intArrayOf(), floatArrayOf())
    }

    /**
     * Test case for adding sparse values to the dataset.
     * Expects UnsupportedOperationException to be thrown.
     */
    @Test(expected = UnsupportedOperationException::class)
    fun testAddSparse() {
        val ds = DenseDataSet(Dim._1d(4, 3), DataType.Int)
        ds.addSparse(0, longArrayOf(), intArrayOf())
    }

    /**
     * Test case for adding weighted sparse values to the dataset.
     * Expects UnsupportedOperationException to be thrown.
     */
    @Test(expected = UnsupportedOperationException::class)
    fun testAddSparseWeighted() {
        val ds = DenseDataSet(Dim._1d(4, 3), DataType.Int)
        ds.addSparseWeighted(0, longArrayOf(), floatArrayOf(), intArrayOf())
    }
}
