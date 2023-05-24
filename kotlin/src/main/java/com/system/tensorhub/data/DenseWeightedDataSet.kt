package com.system.tensorhub.data

import com.system.tensorhub.Dim
import com.system.tensorhub.DataSetEnums.DataType
import java.util.*

/**
 * Represents a dense weighted dataset.
 *
 * @param dim The dimensions of the dataset.
 * @param dataType The data type of the dataset.
 */
class DenseWeightedDataSet(dim: Dim, dataType: DataType) : DenseDataSet(dim, dataType) {
    private val weights: FloatArray = FloatArray(dim.stride * dim.examples)

    init {
        Arrays.fill(weights, 1f)
    }

    /**
     * Sets the weights for a specific index in the dataset.
     *
     * @param index The index to set the weights for.
     * @param weights The weights to set.
     */
    private fun putWeight(index: Int, weights: FloatArray) {
        System.arraycopy(weights, 0, this.weights, index * getStride(), getStride())
    }

    /**
     * Retrieves the weights associated with the dataset.
     *
     * @return The weights array.
     */
    override fun getWeights(): FloatArray {
        return weights
    }

    /**
     * Adds data to the dataset at the specified index and sets the weights to 1.0.
     *
     * @param index The index at which to add the data.
     * @param data The character array data to add.
     */
    override fun add(index: Int, data: CharArray) {
        super.add(index, data)
        putWeight(index, floatArrayOf(1f))
    }

    /**
     * Adds data to the dataset at the specified index and sets the weights to 1.0.
     *
     * @param index The index at which to add the data.
     * @param data The integer array data to add.
     */
    override fun add(index: Int, data: IntArray) {
        super.add(index, data)
        putWeight(index, floatArrayOf(1f))
    }

    /**
     * Adds data to the dataset at the specified index and sets the weights to 1.0.
     *
     * @param index The index at which to add the data.
     * @param data The float array data to add.
     */
    override fun add(index: Int, data: FloatArray) {
        super.add(index, data)
        putWeight(index, floatArrayOf(1f))
    }

    /**
     * Adds data to the dataset at the specified index and sets the weights to 1.0.
     *
     * @param index The index at which to add the data.
     * @param data The double array data to add.
     */
    override fun add(index: Int, data: DoubleArray) {
        super.add(index, data)
        putWeight(index, floatArrayOf(1f))
    }

    /**
     * Adds weighted data to the dataset at the specified index.
     *
     * @param index The index at which to add the data.
     * @param data The character array data to add.
     * @param weights The weights associated with the data.
     */
    override fun addWeighted(index: Int, data: CharArray, weights: FloatArray) {
        super.add(index, data)
        putWeight(index, weights)
    }

    /**
     * Adds weighted data to the dataset at the specified index.
     *
     * @param index The index at which to add the data.
     * @param data The integer array data to add.
     * @param weights The weights associated with the data.
     */
    override fun addWeighted(index: Int, data: IntArray, weights: FloatArray) {
        super.add(index, data)
        putWeight(index, weights)
    }

    /**
     * Adds weighted data to the dataset at the specified index.
     *
     * @param index The index at which to add the data.
     * @param data The float array data to add.
     * @param weights The weights associated with the data.
     */
    override fun addWeighted(index: Int, data: FloatArray, weights: FloatArray) {
        super.add(index, data)
        putWeight(index, weights)
    }

    /**
     * Adds weighted data to the dataset at the specified index.
     *
     * @param index The index at which to add the data.
     * @param data The double array data to add.
     * @param weights The weights associated with the data.
     */
    override fun addWeighted(index: Int, data: DoubleArray, weights: FloatArray) {
        super.add(index, data)
        putWeight(index, weights)
    }
}
