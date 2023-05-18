package com.system.tensorhub

import java.util.*
import com.system.tensorhub.Layer.Kind

/**
 * The main class for Tensorhub functionality.
 */
object Tensorhub {
    /**
     * Null pointer constant.
     */
    private const val NULLPTR: Long = 0x0

    /**
     * Loads the network from the given configuration.
     *
     * @param config The network configuration.
     * @return The loaded network.
     * @throws NetworkLoadException if failed to load the network.
     */
    fun load(config: NetworkConfig): Network {
        val ptr = loadNetwork(config.networkFilePath, config.batchSize, config.k)
        if (ptr == NULLPTR) {
            throw NetworkLoadException("Failed to load network from config: $config")
        }

        val inputLayers = getLayers(ptr, Layer.Kind.Input.ordinal())
        val outputLayers = getLayers(ptr, Layer.Kind.Output.ordinal())

        require(inputLayers.isNotEmpty()) { "No input layers found in: $config" }
        require(outputLayers.isNotEmpty()) { "No output layers found in: $config" }

        return Network(config, ptr, inputLayers, outputLayers)
    }

    /**
     * Loads datasets into the network.
     *
     * @param ptr The network pointer.
     * @param datasets The datasets to load.
     */
    external fun loadDatasets(ptr: Long, datasets: Array<DataSet>)

    /**
     * Shuts down the network.
     *
     * @param ptr The network pointer.
     */
    external fun shutdown(ptr: Long)

    /**
     * Retrieves layers of the specified kind from the network.
     *
     * @param ptr The network pointer.
     * @param kind The kind of layers to retrieve.
     * @return The list of layers.
     */
    private external fun getLayers(ptr: Long, kind: Int): List<Layer>

    /**
     * Performs prediction using the network.
     *
     * @param ptr The network pointer.
     * @param k The value of k.
     * @param inputs The input datasets.
     * @param outputs The output arrays.
     */
    external fun predict(ptr: Long, k: Int, inputs: Array<DataSet>, outputs: Array<Output>)

    /**
     * Loads the network from the specified file path.
     *
     * @param networkFilePath The path to the network file.
     * @param batchSize The batch size.
     * @param maxK The maximum value of k.
     * @return The network pointer.
     */
    private external fun loadNetwork(networkFilePath: String, batchSize: Int, maxK: Int): Long
}
