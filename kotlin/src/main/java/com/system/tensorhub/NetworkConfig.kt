package com.system.tensorhub

import java.nio.file.Paths

/**
 * Represents the configuration for a network.
 *
 * @property networkFilePath The file path of the network.
 * @property networkName The name of the network.
 * @property batchSize The batch size for the network. Default value is 32.
 * @property k The value of K. Default value is [ALL].
 * @property inputDataSets The input data sets for the network.
 * @property outputDataSets The output data sets for the network.
 */
data class NetworkConfig(
    private val networkFilePath: String,
    private val networkName: String,
    @Builder.Default
    private val batchSize: Int = 32,
    @Builder.Default
    private val k: Int = ALL,
    @Singular
    private val inputDataSets: Map<String, DataSet>,
    @Singular
    private val outputDataSets: Map<String, DataSet>
) {
    companion object {
        private const val EXTENSION_SEPARATOR: Char = '.'
        const val ALL = -1
    }

    /**
     * Retrieves the name of the network.
     *
     * If the network name is not provided, it extracts the name from the network file path.
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
