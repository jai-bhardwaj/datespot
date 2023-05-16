package com.system.tensorhub

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * This is a test class for NetworkConfig.
 */
class NetworkConfigTest {
    /**
     * Test for the getNetworkName method.
     */
    @Test
    fun testGetNetworkName() {
        val config = NetworkConfig.with().networkFilePath("").networkName("my-model").build()

        assertEquals("my-model", config.networkName)
    }

    /**
     * Test for the default getNetworkName method.
     */
    @Test
    fun testDefaultGetNetworkName() {
        val config = NetworkConfig.with().networkFilePath("").build()

        assertEquals("test-model", config.networkName)
    }
}
