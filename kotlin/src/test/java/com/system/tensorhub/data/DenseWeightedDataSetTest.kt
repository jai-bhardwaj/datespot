package com.system.tensorhub.data

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import java.nio.FloatBuffer
import com.system.tensorhub.Dim
import com.system.tensorhub.DataSetEnums.DataType

/**
 * Unit tests for the DenseWeightedDataSet class.
 */
class DenseWeightedDataSetTest {

    /**
     * Test case for adding float data to DenseWeightedDataSet.
     */
    @Test
    fun testAddFloat() {
        // Create a DenseWeightedDataSet object
        val ds = DenseWeightedDataSet(Dim._1d(4, 2), DataType.Float)

        // Add float data
        ds.add(0, floatArrayOf(1.2f, 2.3f, 3.4f, 4.5f))
        ds.addWeighted(1, floatArrayOf(1.1f, 2.2f, 3.3f, 4.4f), floatArrayOf(4.4f, 3.3f, 2.2f, 1.1f))

        // Get data as FloatBuffer
        val buf: FloatBuffer = ds.data.asFloatBuffer()

        // Verify the first set of data
        val temp = FloatArray(ds.stride)
        buf.get(temp)
        assertArrayEquals(floatArrayOf(1.2f, 2.3f, 3.4f, 4.5f), temp, 0f)

        // Verify the second set of data
        buf.get(temp)
        assertArrayEquals(floatArrayOf(1.1f, 2.2f, 3.3f, 4.4f), temp, 0f)

        // Verify the weights
        for (i in 0 until ds.stride) {
            assertEquals(1f, ds.weights[i], 0f)
        }

        // Verify individual weighted values
        assertEquals(4.4f, ds.weights[ds.stride + 0], 0f)
        assertEquals(3.3f, ds.weights[ds.stride + 1], 0f)
        assertEquals(2.2f, ds.weights[ds.stride + 2], 0f)
        assertEquals(1.1f, ds.weights[ds.stride + 3], 0f)
    }
}
