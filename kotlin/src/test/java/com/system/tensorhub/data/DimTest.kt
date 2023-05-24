package com.system.tensorhub.data

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Tests for the [Dim] class.
 */
class DimTest {

    /**
     * Tests the creation of different dimensional [Dim] objects.
     */
    @Test
    fun testDim() {
        val oneD = Dim._1d(128, 32)
        val twoD = Dim._2d(128, 64, 32)
        val threeD = Dim._3d(128, 64, 32, 16)

        // Assert the dimensions of the objects
        assertEquals(1, oneD.dimensions)
        assertEquals(2, twoD.dimensions)
        assertEquals(3, threeD.dimensions)

        // Assert the stride values of the objects
        assertEquals(128, oneD.stride)
        assertEquals(128 * 64, twoD.stride)
        assertEquals(128 * 64 * 32, threeD.stride)
    }
}
