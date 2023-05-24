package com.system.tensorhub

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Tests for the [NetworkConfig] class.
 */
class NetworkConfigTest {

    /**
     * Test for getting the network name from the [NetworkConfig] instance.
     */
    @Test
    fun testGetNetworkName() {
        val config = NetworkConfig.with()
            .networkFilePath("/tmp/test-model.nc")
            .networkName("model")
            .build()
        assertEquals("model", config.getNetworkName())
    }

    /**
     * Test for getting the default network name from the [NetworkConfig] instance.
     */
    @Test
    fun testDefaultGetNetworkName() {
        val config = NetworkConfig.with()
            .networkFilePath("/tmp/test-model.nc")
            .build()
        assertEquals("test-model", config.getNetworkName())
    }

}
