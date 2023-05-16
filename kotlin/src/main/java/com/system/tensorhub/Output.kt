package com.system.tensorhub

import lombok.Getter
import lombok.Setter

/**
 * Represents the output of a neural network.
 *
 * @property dim The dimensions of the output.
 */
@Getter
class Output(private val dim: Dim) {
    /**
     * The name of the output.
     */
    @Setter
    var name = ""

    /**
     * The name of the layer associated with the output.
     */
    @Setter
    var layerName = ""

    private val scores: FloatArray
    private val indexes: LongArray

    init {
        scores = FloatArray(dim.x * dim.y * dim.z * dim.examples)
        indexes = LongArray(dim.x * dim.y * dim.z * dim.examples)
    }

    companion object {
        /**
         * Creates an instance of the Output class based on the provided configuration and output layer.
         *
         * @param config The network configuration.
         * @param outputLayer The output layer.
         * @return The created Output instance.
         * @throws IllegalArgumentException if top k outputs are only supported on 1-D outputs.
         */
        fun create(config: NetworkConfig, outputLayer: Layer): Output {
            val k = config.k
            val batchSize = config.batchSize
            val outputLayerDim = outputLayer.dim

            val outputDataset: Output
            if (config.k == NetworkConfig.ALL) {
                outputDataset = Output(Dim(outputLayerDim, batchSize))
            } else {
                if (outputLayerDim.dimensions > 1) {
                    throw IllegalArgumentException("Top k outputs only supported on 1-D outputs")
                }
                outputDataset = Output(Dim._1d(k, batchSize))
            }
            outputDataset.name = outputLayer.datasetName
            outputDataset.layerName = outputLayer.name
            return outputDataset
        }
    }
}
