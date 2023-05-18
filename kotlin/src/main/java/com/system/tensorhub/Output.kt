package com.system.tensorhub

/**
 * Represents the output of a neural network.
 *
 * @property dim The dimensions of the output.
 * @property name The name of the output.
 * @property layerName The name of the layer associated with the output.
 */
class Output(val dim: Dim) {
    var name = ""
    var layerName = ""
    private val scores: FloatArray = FloatArray(dim.totalSize)
    private val indexes: LongArray = LongArray(dim.totalSize)

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

            if (k != NetworkConfig.ALL && outputLayerDim.dimensions > 1) {
                throw IllegalArgumentException("Top k outputs only supported on 1-D outputs")
            }

            val outputDataset = Output(
                if (k == NetworkConfig.ALL) Dim(outputLayerDim, batchSize) else Dim._1d(k, batchSize)
            )

            outputDataset.name = outputLayer.datasetName
            outputDataset.layerName = outputLayer.name
            return outputDataset
        }
    }
}
