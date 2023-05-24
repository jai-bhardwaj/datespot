package com.system.tensorhub.data

import java.util.Arrays
import com.system.tensorhub.Dim
import com.system.tensorhub.DataSetEnums.DataType

/**
 * Represents a sparse weighted data set.
 */
class SparseWeightedDataSet(dim: Dim, dataType: DataType, sparseDensity: Double) : SparseDataSet(dim, dataType, sparseDensity) {

    /**
     * Array to store the weights associated with the data set.
     */
    private val weights: FloatArray = FloatArray(getStride() * dim.examples)

    /**
     * Fills the weight array with a value of 1.0 at the specified index.
     *
     * @param index The index at which to set the weight.
     */
    private fun putWeightOne(index: Int) {
        Arrays.fill(weights, index * getStride(), index + getStride(), 1f)
    }

    /**
     * Copies the weights array to the specified index.
     *
     * @param index The index at which to copy the weights.
     * @param weights The weights to be copied.
     */
    private fun putWeight(index: Int, weights: FloatArray) {
        System.arraycopy(weights, 0, this.weights, index * getStride(), getStride())
    }

    /**
     * Returns the weights associated with the data set.
     *
     * @return The weights array.
     */
    override fun getWeights(): FloatArray {
        return weights
    }

    /**
     * Adds a sparse data entry to the data set with a weight of 1.0.
     *
     * @param index The index of the data entry.
     * @param sparseIndex The sparse indices of the data entry.
     * @param data The data to be added.
     */
    override fun addSparse(index: Int, sparseIndex: LongArray, data: CharArray) {
        super.addSparse(index, sparseIndex, data)
        putWeightOne(index)
    }

    /**
     * Adds a sparse data entry to the data set with a weight of 1.0.
     *
     * @param index The index of the data entry.
     * @param sparseIndex The sparse indices of the data entry.
     * @param data The data to be added.
     */
    override fun addSparse(index: Int, sparseIndex: LongArray, data: IntArray) {
        super.addSparse(index, sparseIndex, data)
        putWeightOne(index)
    }

    /**
     * Adds a sparse data entry to the data set with a weight of 1.0.
     *
     * @param index The index of the data entry.
     * @param sparseIndex The sparse indices of the data entry.
     * @param data The data to be added.
     */
    override fun addSparse(index: Int, sparseIndex: LongArray, data: FloatArray) {
        super.addSparse(index, sparseIndex, data)
        putWeightOne(index)
    }

    /**
     * Adds a sparse data entry to the data set with a weight of 1.0.
     *
     * @param index The index of the data entry.
     * @param sparseIndex The sparse indices of the data entry.
     * @param data The data to be added.
     */
    override fun addSparse(index: Int, sparseIndex: LongArray, data: DoubleArray) {
        super.addSparse(index, sparseIndex, data)
        putWeightOne(index)
    }

    /**
     * Adds a sparse data entry to the data set with specified weights.
     *
     * @param index The index of the data entry.
     * @param sparseIndex The sparse indices of the data entry.
     * @param weights The weights associated with the data entry.
     * @param data The data to be added.
     */
    override fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: CharArray) {
        super.addSparse(index, sparseIndex, data)
        putWeight(index, weights)
    }

    /**
     * Adds a sparse data entry to the data set with specified weights.
     *
     * @param index The index of the data entry.
     * @param sparseIndex The sparse indices of the data entry.
     * @param weights The weights associated with the data entry.
     * @param data The data to be added.
     */
    override fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: IntArray) {
        super.addSparse(index, sparseIndex, data)
        putWeight(index, weights)
    }

    /**
     * Adds a sparse data entry to the data set with specified weights.
     *
     * @param index The index of the data entry.
     * @param sparseIndex The sparse indices of the data entry.
     * @param weights The weights associated with the data entry.
     * @param data The data to be added.
     */
    override fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: FloatArray) {
        super.addSparse(index, sparseIndex, data)
        putWeight(index, weights)
    }

    /**
     * Adds a sparse data entry to the data set with specified weights.
     *
     * @param index The index of the data entry.
     * @param sparseIndex The sparse indices of the data entry.
     * @param weights The weights associated with the data entry.
     * @param data The data to be added.
     */
    override fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: DoubleArray) {
        super.addSparse(index, sparseIndex, data)
        putWeight(index, weights)
    }
}
