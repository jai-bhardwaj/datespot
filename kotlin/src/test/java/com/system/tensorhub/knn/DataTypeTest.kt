package com.system.tensorhub.knn

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Unit tests for the [DataType] class.
 */
class DataTypeTest {

    /**
     * Test case for [DataType.fromString] method.
     */
    @Test
    fun testDataTypeFromString() {
        assertEquals(DataType.FP16, DataType.fromString("fp16"))
        assertEquals(DataType.FP32, DataType.fromString("fp32"))
    }

    /**
     * Test case for [DataType.ordinal] method.
     */
    @Test
    fun testDataTypeFromString_ordinal() {
        assertEquals(0, DataType.FP32.ordinal())
        assertEquals(1, DataType.FP16.ordinal())
    }
}
