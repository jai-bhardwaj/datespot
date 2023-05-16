package com.system.tensorhub.data

import java.util.Arrays
import com.system.tensorhub.Dim
import com.system.tensorhub.DataSetEnums.DataType

/**
 * Represents a sparse weighted data set.
 *
 * @param dim The dimensions of the data set.
 * @param dataType The data type of the data set.
 * @param sparseDensity The density of the sparse data set.
 */
class SparseWeightedDataSet(dim: Dim, dataType: DataType, sparseDensity: Double) : SparseDataSet(dim, dataType, sparseDensity) {
    private val weights: FloatArray = FloatArray(getStride() * dim.examples)

    /**
     * Fills the weights array with the value 1.0 for the specified index.
     *
     * @param index The index to set the weights for.
     */
    private fun putWeightOne(index: Int) {
        Arrays.fill(weights, index * getStride(), index + getStride(), 1f)
    }

    /**
     * Copies the weights array to the specified index from the given weights array.
     *
     * @param index The index to copy the weights to.
     * @param weights The weights to be copied.
     */
    private fun putWeight(index: Int, weights: FloatArray) {
        System.arraycopy(weights, 0, this.weights, index * getStride(), getStride())
    }

    /**
     * Gets the weights array.
     *
     * @return The weights array.
     */
    override fun getWeights(): FloatArray {
        return weights
    }

    /**
     * Adds sparse data with character array and updates the weights array.
     *
     * @param index The index of the data.
     * @param sparseIndex The sparse indices.
     * @param data The data as a character array.
     */
    override fun addSparse(index: Int, sparseIndex: LongArray, data: CharArray) {
        super.addSparse(index, sparseIndex, data)
        putWeightOne(index)
    }

    /**
     * Adds sparse data with integer array and updates the weights array.
     *
     * @param index The index of the data.
     * @param sparseIndex The sparse indices.
     * @param data The data as an integer array.
     */
    override fun addSparse(index: Int, sparseIndex: LongArray, data: IntArray) {
        super.addSparse(index, sparseIndex, data)
        putWeightOne(index)
    }

    /**
     * Adds sparse data with float array and updates the weights array.
     *
     * @param index The index of the data.
     * @param sparseIndex The sparse indices.
     * @param data The data as a float array.
     */
    override fun addSparse(index: Int, sparseIndex: LongArray, data: FloatArray) {
        super.addSparse(index, sparseIndex, data)
        putWeightOne(index)
    }

    /**
     * Adds sparse data with double array and updates the weights array.
     *
     * @param index The index of the data.
     * @param sparseIndex The sparse indices.
     * @param data The data as a double array.
     */
    override fun addSparse(index: Int, sparseIndex: LongArray, data: DoubleArray) {
        super.addSparse(index, sparseIndex, data)
        putWeightOne(index)
    }
    
    /**
     * Adds sparse weighted data with character array and updates the weights array.
     *
     * @param index The index of the data.
     * @param sparseIndex The sparse indices.
     * @param weights The weights array.
     * @param data The data as a character array.
     */
    override fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: CharArray) {
        super.addSparse(index, sparseIndex, data)
        putWeight(index, weights)
    }

    /**
     * Adds sparse weighted data with integer array and updates the weights array.
     *
     * @param index The index of the data.
     * @param sparseIndex The sparse indices.
     * @param weights The weights array.
     * @param data The data as an integer array.
     */
    override fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: IntArray) {
        super.addSparse(index, sparseIndex, data)
        putWeight(index, weights)
    }

    /**
     * Adds sparse weighted data with float array and updates the weights array.
     *
     * @param index The index of the data.
     * @param sparseIndex The sparse indices.
     * @param weights The weights array.
     * @param data The data as a float array.
     */
    override fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: FloatArray) {
        super.addSparse(index, sparseIndex, data)
        putWeight(index, weights)
    }

    /**
     * Adds sparse weighted data with double array and updates the weights array.
     *
     * @param index The index of the data.
     * @param sparseIndex The sparse indices.
     * @param weights The weights array.
     * @param data The data as a double array.
     */
    override fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: DoubleArray) {
        super.addSparse(index, sparseIndex, data)
        putWeight(index, weights)
    } 
}
