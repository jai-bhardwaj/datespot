package com.system.tensorhub

import org.junit.Assert.assertEquals
import org.junit.Test
import com.system.tensorhub.Layer.Kind

/**
 * Tests for the [Layer] class.
 */
class LayerTest {
    /**
     * Tests the ordinal values of [Kind].
     */
    @Test
    fun testKindOrdinal() {
        assertEquals(0, Kind.Input.ordinal())

        assertEquals(1, Kind.Hidden.ordinal())

        assertEquals(2, Kind.Output.ordinal())

        assertEquals(3, Kind.Target.ordinal())
    }

    /**
     * Tests the attribute mask values of [Layer.Attribute].
     */
    @Test
    fun testAttributeMaskValues() {
        assertEquals(0x0, Layer.Attribute.None)

        assertEquals(0x1, Layer.Attribute.Sparse)

        assertEquals(0x2, Layer.Attribute.Denoising)

        assertEquals(0x4, Layer.Attribute.BatchNormalization)
    }

    /**
     * Tests the creation of [Layer] instances.
     */
    @Test
    fun testCreateLayer() {
        val inputLayer = Layer("input-layer", "input-layer-data", 0, 1, 1, 128, 1, 1)

        val hiddenLayer = Layer("hidden-layer", "", 1, 1, 1, 128, 1, 1)

        val outputLayer = Layer("output-layer", "output-layer-data", 2, 1, 1, 128, 1, 1)

        val targetLayer = Layer("target-layer", "", 3, 1, 1, 128, 1, 1)

        assertEquals(Kind.Input, inputLayer.kind)

        assertEquals(Kind.Hidden, hiddenLayer.kind)

        assertEquals(Kind.Output, outputLayer.kind)

        assertEquals(Kind.Target, targetLayer.kind)
    }
}
