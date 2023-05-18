package com.system.tensorhub

import java.io.Closeable

/**
 * Represents a neural network with configurable parameters and layers.
 *
 * @param config The configuration of the network.
 * @param inputLayers Array of input layers.
 * @param outputLayers Array of output layers.
 */
class Network(
    val config: NetworkConfig,
    val inputLayers: Array<Layer>,
    val outputLayers: Array<Layer>
) : Closeable {
    @Volatile
    private var ptr: Long = 0

    /**
     * Initializes the Network object.
     * Throws a RuntimeException if the number of output layers is more than 1.
     */
    init {
        require(outputLayers.size == 1) { "Only one output layer is supported at the moment. Got ${outputLayers.size}" }
    }

    /**
     * Loads the specified datasets into the network.
     *
     * @param datasets Array of datasets to load.
     */
    fun load(datasets: Array<DataSet>) {
        Tensorhub.load_datasets(ptr, datasets)
    }

    /**
     * Predicts the output based on a single input dataset.
     * Throws an UnsupportedOperationException if the network has more than one input or output layer.
     *
     * @param input The input dataset for prediction.
     * @param output The output object to store the predicted result.
     */
    fun predict(input: DataSet, output: Output) {
        require(inputLayers.size == 1 && outputLayers.size == 1) { "Method can only be used with networks with a single input/output layer" }
        predict(arrayOf(input), arrayOf(output))
    }

    /**
     * Predicts the output based on an array of input datasets.
     *
     * @param inputs Array of input datasets for prediction.
     * @return Array of output objects containing the predicted results.
     */
    fun predict(inputs: Array<DataSet>): Array<Output> {
        predict(inputs, outputs)
        return outputs
    }

    /**
     * Predicts the output based on the provided input datasets and stores the results in the output objects.
     *
     * @param inputs Array of input datasets for prediction.
     * @param outputs Array of output objects to store the predicted results.
     */
    fun predict(inputs: Array<DataSet>, outputs: Array<Output>) {
        checkArguments(inputs, outputs)
        Tensorhub.predict(ptr, config.k, inputs, outputs)
    }

    /**
     * Checks the arguments of the network for consistency between input data, input layers,
     * output data, and output layers.
     *
     * @param inputs Array of input data sets.
     * @param outputs Array of output data sets.
     * @throws IllegalArgumentException if the number of input data and input layers do not match,
     * if there is a dimension mismatch between input layers and data,
     * if there is a dimension mismatch between input layers and input data,
     * if the number of output data and output layers do not match,
     * if there is a dimension mismatch between output layers and data,
     * if there is a dimension mismatch between output layers and output data,
     * or if the examples in input/output data do not match the batch size of the network.
     */
    private fun checkArguments(inputs: Array<DataSet>, outputs: Array<Output>) {
        require(inputs.size == inputLayers.size) { "Number of input data and input layers do not match" }
        require(outputs.size == outputLayers.size) { "Number of output data and output layers do not match" }

        for (i in inputs.indices) {
            val datasetName = inputs[i].name
            val dataDim = inputs[i].dim
            val inputLayer = inputLayers[i]

            when {
                dataDim.dimensions != inputLayer.dimensions ->
                    throw IllegalArgumentException("Num dimension mismatch between layer ${inputLayer.name} and data $datasetName")
                dataDim.x != inputLayer.dimX || dataDim.y != inputLayer.dimY || dataDim.z != inputLayer.dimZ ->
                    throw IllegalArgumentException("Dimension mismatch between input layer ${inputLayer.name} and input data $datasetName")
                dataDim.examples != config.batchSize ->
                    throw IllegalArgumentException("Examples in input data $i do not match the batch size of the network")
            }
        }

        for (i in outputs.indices) {
            val outputLayer = outputLayers[i]
            val datasetName = outputs[i].name
            val dataDim = outputs[i].dim

            if (config.k == NetworkConfig.ALL) {
                require(dataDim.x == outputLayer.dimX && dataDim.y == outputLayer.dimY && dataDim.z == outputLayer.dimZ) {
                    "Dimension mismatch between output layer ${outputLayer.name} and output data $datasetName"
                }
            } else {
                require(dataDim.x == config.k && dataDim.y == 1 && dataDim.z == 1) {
                    "Data dimX != k or dimY != dimZ != 1 for dataset $datasetName"
                }
            }

            require(dataDim.examples == config.batchSize) { "Examples in output data $i do not match the batch size of the network" }
        }
    }

    /**
     * Closes the Network and releases any associated resources.
     * This method shuts down the Tensorhub using the provided pointer.
     */
    override fun close() {
        Tensorhub.shutdown(ptr)
    }
}
