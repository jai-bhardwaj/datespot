package com.system.tensorhub

import java.util.*

/**
 * The Tensorhub class provides functionality for loading and using networks.
 */
class Tensorhub {
    companion object {
        /** Null pointer constant. */
        const val NULLPTR: Long = 0x0

        init {
            System.loadLibrary("tensorhub_java")
        }

        /**
         * Loads a network from the given configuration.
         *
         * @param config The network configuration.
         * @return The loaded network.
         * @throws RuntimeException if the network fails to load.
         */
        fun load(config: NetworkConfig): Network {
            val ptr = load(config.networkFilePath, config.batchSize, config.k)
            if (ptr == NULLPTR) {
                throw RuntimeException("Failed to load network from config: $config")
            }

            val inputLayers = getLayers(ptr, Layer.Kind.Input.ordinal())
            val outputLayers = getLayers(ptr, Layer.Kind.Output.ordinal())

            if (inputLayers.isEmpty()) {
                throw RuntimeException("No input layers found in: $config")
            }
            if (outputLayers.isEmpty()) {
                throw RuntimeException("No output layers found in: $config")
            }

            return Network(config, ptr, inputLayers, outputLayers)
        }

        /**
         * Loads a network from the specified file path, batch size, and maximum K value.
         *
         * @param networkFilePath The file path of the network.
         * @param batchSize The batch size.
         * @param maxK The maximum K value.
         * @return The pointer to the loaded network.
         */
        private external fun load(networkFilePath: String, batchSize: Int, maxK: Int): Long

        /**
         * Loads the datasets into the specified network.
         *
         * @param ptr The pointer to the network.
         * @param datasets The datasets to load.
         */
        external fun loadDatasets(ptr: Long, datasets: Array<DataSet>)

        /**
         * Shuts down the specified network.
         *
         * @param ptr The pointer to the network.
         */
        private external fun shutdown(ptr: Long)

        /**
         * Retrieves layers of the specified kind from the network.
         *
         * @param ptr The pointer to the network.
         * @param kind The kind of layers to retrieve.
         * @return The list of layers.
         */
        private external fun getLayers(ptr: Long, kind: Int): List<Layer>

        /**
         * Predicts the output using the specified network, K value, inputs, and outputs.
         *
         * @param ptr The pointer to the network.
         * @param k The K value.
         * @param inputs The input datasets.
         * @param outputs The output datasets.
         */
        external fun predict(ptr: Long, k: Int, inputs: Array<DataSet>, outputs: Array<Output>)
    }
}
