package com.system.tensorhub.knn

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Tests for the DataType class.
 */
class DataTypeTest {

    /**
     * Tests the DataType.fromString() method.
     */
    @Test
    fun testDataTypeFromString() {
        assertEquals(DataType.FP16, DataType.fromString("fp16"))
        assertEquals(DataType.FP32, DataType.fromString("fp32"))
    }

    /**
     * Tests the ordinal values of DataType enum constants.
     */
    @Test
    fun testDataTypeFromString_ordinal() {
        assertEquals(0, DataType.FP32.ordinal)
        assertEquals(1, DataType.FP16.ordinal)
    }
}
