package com.system.tensorhub.data

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import java.nio.FloatBuffer

/**
 * Unit tests for [DenseWeightedDataSet].
 */
class DenseWeightedDataSetTest {

    /**
     * Tests the [DenseWeightedDataSet.add] and [DenseWeightedDataSet.addWeighted] methods.
     */
    @Test
    fun testAddFloat() {
        val ds = DenseWeightedDataSet(Dim._1d(4, 2), DataType.Float)
        ds.add(0, floatArrayOf(1.2f, 2.3f, 3.4f, 4.5f))
        ds.addWeighted(1, floatArrayOf(1.1f, 2.2f, 3.3f, 4.4f), floatArrayOf(4.4f, 3.3f, 2.2f, 1.1f))

        val buf = ds.data.asFloatBuffer()
        val temp = FloatArray(ds.stride)
        buf.get(temp)
        assertArrayEquals(floatArrayOf(1.2f, 2.3f, 3.4f, 4.5f), temp, 0f)

        buf.get(temp)
        assertArrayEquals(floatArrayOf(1.1f, 2.2f, 3.3f, 4.4f), temp, 0f)

        for (i in 0 until ds.stride) {
            assertEquals(1f, ds.weights[i], 0f)
        }

        assertEquals(4.4f, ds.weights[ds.stride + 0], 0f)
        assertEquals(3.3f, ds.weights[ds.stride + 1], 0f)
        assertEquals(2.2f, ds.weights[ds.stride + 2], 0f)
        assertEquals(1.1f, ds.weights[ds.stride + 3], 0f)
    }
}
