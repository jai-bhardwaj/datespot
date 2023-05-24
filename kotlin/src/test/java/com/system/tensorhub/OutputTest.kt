package com.system.tensorhub

import org.junit.Assert
import org.junit.Test

/**
 * Tests for the Output class.
 */
class OutputTest {
    private val layerName = "test-layer"
    private val datasetName = "test-dataset"
    private val x = 128
    private val y = 1
    private val z = 1
    private val batchSize = 128
    private val layer = Layer(layerName, datasetName, Kind.Input.ordinal(), Attribute.None, 1, x, y, z)

    /**
     * Test to verify that names are set correctly in the output.
     */
    @Test
    fun testNamesAreSet() {
        val config = NetworkConfig.with().batchSize(batchSize).build()
        val output = Output.create(config, layer)
        Assert.assertEquals(datasetName, output.getName())
        Assert.assertEquals(layerName, output.getLayerName())
    }

    /**
     * Test to verify the output when all units are buffered.
     */
    @Test
    fun testOutputAllUnitBuffer() {
        val config = NetworkConfig.with().batchSize(batchSize).build()
        val output = Output.create(config, layer)
        Assert.assertEquals(x, output.getDim().x)
        Assert.assertEquals(y, output.getDim().y)
        Assert.assertEquals(z, output.getDim().z)
        Assert.assertEquals(1, output.getDim().dimensions)
        Assert.assertEquals(batchSize, output.getDim().examples)
    }

    /**
     * Test to verify the output with a specific value of k.
     */
    @Test
    fun testOutput() {
        val k = 100
        val config = NetworkConfig.with().batchSize(batchSize).k(100).build()
        val output = Output.create(config, layer)
        Assert.assertEquals(k, output.getDim().x)
        Assert.assertEquals(y, output.getDim().y)
        Assert.assertEquals(z, output.getDim().z)
        Assert.assertEquals(1, output.getDim().dimensions)
        Assert.assertEquals(batchSize, output.getDim().examples)
    }
}
