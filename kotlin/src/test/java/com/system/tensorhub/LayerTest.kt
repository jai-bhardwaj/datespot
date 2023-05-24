package com.system.tensorhub

import org.junit.Test
import org.junit.Assert.assertEquals

/**
 * Class to test the Layer class.
 */
class LayerTest {

    /**
     * Test case for checking the ordinal values of the Layer.Kind enum.
     */
    @Test
    fun testKindOrdinal() {
        assertEquals(0, Layer.Kind.Input.ordinal)
        assertEquals(1, Layer.Kind.Hidden.ordinal)
        assertEquals(2, Layer.Kind.Output.ordinal)
        assertEquals(3, Layer.Kind.Target.ordinal)
    }

    /**
     * Test case for checking the attribute mask values of the Layer.Attribute enum.
     */
    @Test
    fun testAttributeMaskValues() {
        assertEquals(0x0, Layer.Attribute.None)
        assertEquals(0x1, Layer.Attribute.Sparse)
        assertEquals(0x2, Layer.Attribute.Denoising)
        assertEquals(0x4, Layer.Attribute.BatchNormalization)
    }

    /**
     * Test case for creating different types of layers and checking their kind.
     */
    @Test
    fun testCreateLayer() {
        val inputLayer = Layer("input-layer", "input-layer-data", 0, 1, 1, 128, 1, 1)
        val hiddenLayer = Layer("hidden-layer", "", 1, 1, 1, 128, 1, 1)
        val outputLayer = Layer("output-layer", "output-layer-data", 2, 1, 1, 128, 1, 1)
        val targetLayer = Layer("target-layer", "", 3, 1, 1, 128, 1, 1)

        assertEquals(Layer.Kind.Input, inputLayer.kind)
        assertEquals(Layer.Kind.Hidden, hiddenLayer.kind)
        assertEquals(Layer.Kind.Output, outputLayer.kind)
        assertEquals(Layer.Kind.Target, targetLayer.kind)
    }
}
