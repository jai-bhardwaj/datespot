package com.system.tensorhub

import java.io.Closeable

import lombok.Getter
import lombok.ToString

/**
 * Represents a neural network.
 *
 * @property config The network configuration.
 * @property ptr The pointer to the network.
 * @property inputLayers The input layers of the network.
 * @property outputLayers The output layers of the network.
 */
@ToString
class Network(private val config: NetworkConfig, private val ptr: Long, inputLayers: List<Layer>, outputLayers: List<Layer>) :
    Closeable {
    @Getter
    private val inputLayers: Array<Layer> = inputLayers.toTypedArray()
    @Getter
    private val outputLayers: Array<Layer> = outputLayers.toTypedArray()

    init {
        if (outputLayers.size > 1) {
            throw RuntimeException("Only one output layer is supported at the moment. Got ${outputLayers.size}")
        }
    }

    /**
     * Loads the given datasets into the network.
     *
     * @param datasets The datasets to load.
     */
    fun load(datasets: Array<DataSet>) {
        tensorhub.load_datasets(ptr, datasets)
    }

    /**
     * Predicts the output for a single input dataset.
     *
     * @param input The input dataset.
     * @param output The output to store the prediction.
     */
    fun predict(input: DataSet, output: Output) {
        if (inputLayers.size != 1 || outputLayers.size != 1) {
            throw UnsupportedOperationException("Method can only be used with networks with single input/output layer")
        }

        predict(arrayOf(input), arrayOf(output))
    }

    /**
     * Predicts the outputs for the given input datasets.
     *
     * @param inputs The input datasets.
     * @return The array of predicted outputs.
     */
    fun predict(inputs: Array<DataSet>): Array<Output> {
        val outputs = Array(outputLayers.size) { Output.create(config, outputLayers[it]) }
        predict(inputs, outputs)
        return outputs
    }

    /**
     * Predicts the outputs for the given input datasets and stores them in the provided output arrays.
     *
     * @param inputs The input datasets.
     * @param outputs The output arrays to store the predictions.
     */
    fun predict(inputs: Array<DataSet>, outputs: Array<Output>) {
        checkArguments(inputs, outputs)
        tensorhub.predict(ptr, config.k, inputs, outputs)
    }

    /**
     * Checks the compatibility between input datasets and output arrays.
     *
     * @param inputs The input datasets.
     * @param outputs The output arrays.
     */
    private fun checkArguments(inputs: Array<DataSet>, outputs: Array<Output>) {
        if (inputs.size != inputLayers.size) {
            throw IllegalArgumentException("Number of input data and input layers do not match")
        }

        for (i in inputs.indices) {
            val datasetName = inputs[i].name
            val dataDim = inputs[i].dim
            val inputLayer = inputLayers[i]

            if (dataDim.dimensions != inputLayer.dimensions) {
                throw IllegalArgumentException(
                    "Num dimension mismatch between layer ${inputLayer.name} and data $datasetName"
                )
            }

            if (dataDim.x != inputLayer.dimX
                || dataDim.y != inputLayer.dimY
                || dataDim.z != inputLayer.dimZ
            ) {
                throw IllegalArgumentException(
                    "Dimension mismatch between input layer ${inputLayer.name} and input data $datasetName"
                )
            }

            if (dataDim.examples != config.batchSize) {
                throw IllegalArgumentException(
                    "Examples in input data $i does not match the batch size of the network"
                )
            }
        }

        if (outputs.size != outputLayers.size) {
            throw IllegalArgumentException("Number of output data and output layers do not match")
        }

        for (i in outputs.indices) {
            val outputLayer = outputLayers[i]
            val datasetName = outputs[i].name
            val dataDim = outputs[i].dim

            if (config.k == NetworkConfig.ALL) {
                if (dataDim.x != outputLayer.dimX
                    || dataDim.y != outputLayer.dimY
                    || dataDim.z != outputLayer.dimZ
                ) {
                    throw IllegalArgumentException(
                        "Dimension mismatch between output layer ${outputLayer.name} and output data $datasetName"
                    )
                }
            } else {
                if (dataDim.x != config.k
                    || dataDim.y != 1
                    || dataDim.z != 1
                ) {
                    throw IllegalArgumentException(
                        "Data dimX != k or dimY != dimZ != 1 for dataset $datasetName"
                    )
                }
            }

            if (dataDim.examples != config.batchSize) {
                throw IllegalArgumentException(
                    "Examples in output data $i does not match the batch size of the network"
                )
            }
        }
    }

    override fun close() {
        tensorhub.shutdown(ptr)
    }
}
