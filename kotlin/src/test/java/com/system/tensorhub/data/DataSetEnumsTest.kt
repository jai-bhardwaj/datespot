package com.system.tensorhub.data

import org.junit.Assert.assertEquals
import org.junit.Test
import com.system.tensorhub.DataSetEnums.DataType
import com.system.tensorhub.DataSetEnums.Kind
import com.system.tensorhub.DataSetEnums.Sharding

/**
 * Unit tests for DataSetEnums.
 */
class DataSetEnumsTest {

    /**
     * Test for Kind enumeration.
     */
    @Test
    fun testKind() {
        assertEquals(0, Kind.Numeric.ordinal) // Numeric should have ordinal 0
        assertEquals(1, Kind.Image.ordinal) // Image should have ordinal 1
        assertEquals(2, Kind.Audio.ordinal) // Audio should have ordinal 2
    }

    /**
     * Test for Sharding enumeration.
     */
    @Test
    fun testSharding() {
        assertEquals(0, Sharding.None.ordinal) // None should have ordinal 0
        assertEquals(1, Sharding.Model.ordinal) // Model should have ordinal 1
        assertEquals(2, Sharding.Data.ordinal) // Data should have ordinal 2
    }

    /**
     * Test for DataType enumeration.
     */
    @Test
    fun testDataType() {
        assertEquals(0, DataType.UInt.ordinal) // UInt should have ordinal 0
        assertEquals(1, DataType.Int.ordinal) // Int should have ordinal 1
        assertEquals(2, DataType.LLInt.ordinal) // LLInt should have ordinal 2
        assertEquals(3, DataType.ULLInt.ordinal) // ULLInt should have ordinal 3
        assertEquals(4, DataType.Float.ordinal) // Float should have ordinal 4
        assertEquals(5, DataType.Double.ordinal) // Double should have ordinal 5
        assertEquals(6, DataType.RGB8.ordinal) // RGB8 should have ordinal 6
        assertEquals(7, DataType.RGB16.ordinal) // RGB16 should have ordinal 7
        assertEquals(8, DataType.UChar.ordinal) // UChar should have ordinal 8
        assertEquals(9, DataType.Char.ordinal) // Char should have ordinal 9
    }
}
