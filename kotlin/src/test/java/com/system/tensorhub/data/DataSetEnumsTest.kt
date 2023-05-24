package com.system.tensorhub.data

import org.junit.Test
import kotlin.test.assertEquals

/**
 * Tests for DataSetEnums class.
 */
class DataSetEnumsTest {

    /**
     * Tests the ordinal values of Kind enum.
     */
    @Test
    fun testKind() {
        assertEquals(0, Kind.Numeric.ordinal)
        assertEquals(1, Kind.Image.ordinal)
        assertEquals(2, Kind.Audio.ordinal)
    }

    /**
     * Tests the ordinal values of Sharding enum.
     */
    @Test
    fun testSharding() {
        assertEquals(0, Sharding.None.ordinal)
        assertEquals(1, Sharding.Model.ordinal)
        assertEquals(2, Sharding.Data.ordinal)
    }

    /**
     * Tests the ordinal values of DataType enum.
     */
    @Test
    fun testDataType() {
        assertEquals(0, DataType.UInt.ordinal)
        assertEquals(1, DataType.Int.ordinal)
        assertEquals(2, DataType.LLInt.ordinal)
        assertEquals(3, DataType.ULLInt.ordinal)
        assertEquals(4, DataType.Float.ordinal)
        assertEquals(5, DataType.Double.ordinal)
        assertEquals(6, DataType.RGB8.ordinal)
        assertEquals(7, DataType.RGB16.ordinal)
        assertEquals(8, DataType.UChar.ordinal)
        assertEquals(9, DataType.Char.ordinal)
    }
}
