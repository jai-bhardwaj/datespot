package com.system.tensorhub.data

import java.util.Arrays
import com.system.tensorhub.Dim
import com.system.tensorhub.DataSetEnums.DataType

/**
 * Represents a DenseWeightedDataSet, which is a subclass of DenseDataSet.
 * It stores data along with corresponding weights.
 *
 * @param dim the dimensions of the dataset
 * @param dataType the type of data stored in the dataset
 */
class DenseWeightedDataSet(dim: Dim, dataType: DataType) : DenseDataSet(dim, dataType) {
    private val weights: FloatArray = FloatArray(dim.stride * dim.examples)

    /**
     * Fills the weights array at the specified index with 1.0f.
     *
     * @param index the index at which to fill the weights array
     */
    private fun putWeightOne(index: Int) {
        Arrays.fill(weights, index * stride, index + stride, 1f)
    }

    /**
     * Copies the provided weights array to the weights array of this dataset
     * starting from the specified index.
     *
     * @param index the starting index at which to copy the weights
     * @param weights the weights array to copy
     */
    private fun putWeight(index: Int, weights: FloatArray) {
        System.arraycopy(weights, 0, this.weights, index * stride, stride)
    }

    /**
     * Returns the weights array of this dataset.
     *
     * @return the weights array
     */
    override fun getWeights(): FloatArray {
        return weights
    }

    /**
     * Adds the specified CharArray data at the given index.
     * Overrides the add method in the superclass.
     *
     * @param index the index at which to add the data
     * @param data the CharArray data to add
     */
    override fun add(index: Int, data: CharArray) {
        super.add(index, data)
        putWeightOne(index)
    }

    /**
     * Adds the specified IntArray data at the given index.
     * Overrides the add method in the superclass.
     *
     * @param index the index at which to add the data
     * @param data the IntArray data to add
     */
    override fun add(index: Int, data: IntArray) {
        super.add(index, data)
        putWeightOne(index)
    }

    /**
     * Adds the specified FloatArray data at the given index.
     * Overrides the add method in the superclass.
     *
     * @param index the index at which to add the data
     * @param data the FloatArray data to add
     */
    override fun add(index: Int, data: FloatArray) {
        super.add(index, data)
        putWeightOne(index)
    }
    
    /**
     * Adds the specified DoubleArray data at the given index.
     * Overrides the add method in the superclass.
     *
     * @param index the index at which to add the data
     * @param data the DoubleArray data to add
     */
    override fun add(index: Int, data: DoubleArray) {
        super.add(index, data)
        putWeightOne(index)
    }

    /**
     * Adds the specified CharArray data with corresponding weights at the given index.
     * Overrides the addWeighted method in the superclass.
     *
     * @param index the index at which to add the data
     * @param data the CharArray data to add
     * @param weights the weights corresponding to the data
     */
    override fun addWeighted(index: Int, data: CharArray, weights: FloatArray) {
        super.add(index, data)
        putWeight(index, weights)
    }

    /**
     * Adds the specified IntArray data with corresponding weights at the given index.
     * Overrides the addWeighted method in the superclass.
     *
     * @param index the index at which to add the data
     * @param data the IntArray data to add
     * @param weights the weights corresponding to the data
     */
    override fun addWeighted(index: Int, data: IntArray, weights: FloatArray) {
        super.add(index, data)
        putWeight(index, weights)
    }

    /**
     * Adds the specified FloatArray data with corresponding weights at the given index.
     * Overrides the addWeighted method in the superclass.
     *
     * @param index the index at which to add the data
     * @param data the FloatArray data to add
     * @param weights the weights corresponding to the data
     */
    override fun addWeighted(index: Int, data: FloatArray, weights: FloatArray) {
        super.add(index, data)
        putWeight(index, weights)
    }

    /**
     * Adds the specified DoubleArray data with corresponding weights at the given index.
     * Overrides the addWeighted method in the superclass.
     *
     * @param index the index at which to add the data
     * @param data the DoubleArray data to add
     * @param weights the weights corresponding to the data
     */
    override fun addWeighted(index: Int, data: DoubleArray, weights: FloatArray) {
        super.add(index, data)
        putWeight(index, weights)
    }
}
