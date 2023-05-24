package com.system.tensorhub

import java.nio.file.Paths

/**
 * Represents the configuration for a network.
 *
 * @property networkFilePath The file path of the network.
 * @property networkName The name of the network.
 * @property batchSize The batch size for the network (default is 32).
 * @property k The value of k (default is ALL).
 * @property inputDataSets The input data sets for the network.
 * @property outputDataSets The output data sets for the network.
 */
data class NetworkConfig(
    private val networkFilePath: String,
    private val networkName: String,
    private val batchSize: Int = 32,
    private val k: Int = ALL,
    @get:Singular private val inputDataSets: Map<String, DataSet>,
    @get:Singular private val outputDataSets: Map<String, DataSet>
) {
    companion object {
        private const val EXTENSION_SEPARATOR: Char = '.'
        const val ALL: Int = -1
    }

    /**
     * Retrieves the network name.
     *
     * If the network name is not specified, it is derived from the network file path.
     *
     * @return The network name.
     */
    fun getNetworkName(): String {
        return if (networkName.isNullOrEmpty()) {
            val fileName = Paths.get(networkFilePath).fileName.toString()
            val index = fileName.lastIndexOf(EXTENSION_SEPARATOR)
            fileName.substring(0, index)
        } else {
            networkName
        }
    }
}
