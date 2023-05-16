package com.system.tensorhub.data

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.DoubleBuffer
import java.nio.FloatBuffer
import java.nio.IntBuffer
import com.system.tensorhub.Dim
import com.system.tensorhub.DataSet
import com.system.tensorhub.DataSetEnums.DataType

/**
 * Represents a dense dataset.
 *
 * @param dim The dimensions of the dataset.
 * @param dataType The data type of the dataset.
 */
class DenseDataSet(dim: Dim, dataType: DataType) : DataSet(dim, dataType) {
    private var data: ByteBuffer

    init {
        data = ByteBuffer.allocateDirect(stride * dim.examples * DataType.sizeof(dataType))
        data.order(ByteOrder.nativeOrder())
    }

    /**
     * Returns the stride of the dataset.
     *
     * @return The stride of the dataset.
     */
    override fun getStride(): Int {
        return dim.stride
    }

    /**
     * Returns the sparse start array, which is empty for dense datasets.
     *
     * @return An empty array.
     */
    override fun getSparseStart(): LongArray {
        return EMPTY_LONG_ARRAY
    }

    /**
     * Returns the sparse end array, which is empty for dense datasets.
     *
     * @return An empty array.
     */
    override fun getSparseEnd(): LongArray {
        return EMPTY_LONG_ARRAY
    }

    /**
     * Returns the sparse index array, which is empty for dense datasets.
     *
     * @return An empty array.
     */
    override fun getSparseIndex(): LongArray {
        return EMPTY_LONG_ARRAY
    }

    /**
     * Returns the weights array, which is empty for dense datasets.
     *
     * @return An empty array.
     */
    override fun getWeights(): FloatArray {
        return EMPTY_FLOAT_ARRAY
    }

    /**
     * Adds a data entry of type CharArray to the dense dataset.
     *
     * @param index The index at which to add the data.
     * @param data The data to add.
     */
    override fun add(index: Int, data: CharArray) {
        this.data.position(index * stride)
        this.data.asCharBuffer().put(data, 0, stride)
    }

    /**
     * Adds a data entry of type IntArray to the dense dataset.
     *
     * @param index The index at which to add the data.
     * @param data The data to add.
     */
    override fun add(index: Int, data: IntArray) {
        val buffView: IntBuffer = this.data.asIntBuffer()
        setPosition(buffView, index)
        buffView.put(data, 0, stride)
    }

    /**
     * Adds a data entry of type FloatArray to the dense dataset.
     *
     * @param index The index at which to add the data.
     * @param data The data to add.
     */
    override fun add(index: Int, data: FloatArray) {
        val buffView = data.asFloatBuffer()
        setPosition(buffView, index)
        buffView.put(data, 0, stride)
    }

    /**
     * Adds a data entry of type DoubleArray to the dense dataset.
     *
     * @param index The index at which to add the data.
     * @param data The data to add.
     */
    override fun add(index: Int, data: DoubleArray) {
        val buffView = data.asDoubleBuffer()
        setPosition(buffView, index)
        buffView.put(data, 0, stride)
    }

    /**
     * Adds a weighted data entry of type CharArray to the dense dataset.
     * This operation is not supported for dense datasets.
     * Use the `add` method instead.
     *
     * @param index The index at which to add the weighted data.
     * @param data The data to add.
     * @param weights The weights associated with the data.
     * @throws UnsupportedOperationException This operation is not supported for dense datasets.
     */
    override fun addWeighted(index: Int, data: CharArray, weights: FloatArray) {
        throw UnsupportedOperationException("addWeighted not supported for dense dataset, use add")
    }

    /**
     * Adds a weighted data entry of type IntArray to the dense dataset.
     * This operation is not supported for dense datasets.
     * Use the `add` method instead.
     *
     * @param index The index at which to add the weighted data.
     * @param data The data to add.
     * @param weights The weights associated with the data.
     * @throws UnsupportedOperationException This operation is not supported for dense datasets.
     */
    override fun addWeighted(index: Int, data: IntArray, weights: FloatArray) {
        throw UnsupportedOperationException("addWeighted not supported for dense dataset, use add")
    }
    
    /**
     * Adds a weighted data entry to the dense dataset.
     * This operation is not supported for dense datasets.
     * Use the `add` method instead.
     *
     * @param index The index at which to add the weighted data.
     * @param data The data to add.
     * @param weights The weights associated with the data.
     * @throws UnsupportedOperationException This operation is not supported for dense datasets.
     */
    override fun addWeighted(index: Int, data: FloatArray, weights: FloatArray) {
        throw UnsupportedOperationException("addWeighted not supported for dense dataset, use add")
    }

    /**
     * Adds a weighted data entry to the dense dataset.
     * This operation is not supported for dense datasets.
     * Use the `add` method instead.
     *
     * @param index The index at which to add the weighted data.
     * @param data The data to add.
     * @param weights The weights associated with the data.
     * @throws UnsupportedOperationException This operation is not supported for dense datasets.
     */
    override fun addWeighted(index: Int, data: DoubleArray, weights: FloatArray) {
        throw UnsupportedOperationException("addWeighted not supported for dense dataset, use add")
    }

    /**
     * Adds a sparse data entry to the dense dataset.
     * This operation is not supported for dense datasets.
     * Use the `add` method instead.
     *
     * @param index The index at which to add the sparse data.
     * @param sparseIndex The sparse indices.
     * @param data The sparse data.
     * @throws UnsupportedOperationException This operation is not supported for dense datasets.
     */
    override fun addSparse(index: Int, sparseIndex: LongArray, data: CharArray) {
        throw UnsupportedOperationException("addSparse not supported for dense dataset, use add")
    }

    /**
     * Adds a sparse data entry to the dense dataset.
     * This operation is not supported for dense datasets.
     * Use the `add` method instead.
     *
     * @param index The index at which to add the sparse data.
     * @param sparseIndex The sparse indices.
     * @param data The sparse data.
     * @throws UnsupportedOperationException This operation is not supported for dense datasets.
     */
    override fun addSparse(index: Int, sparseIndex: LongArray, data: IntArray) {
        throw UnsupportedOperationException("addSparse not supported for dense dataset, use add")
    }

    /**
     * Adds a sparse data entry to the dense dataset.
     * This operation is not supported for dense datasets.
     * Use the `add` method instead.
     *
     * @param index The index at which to add the sparse data.
     * @param sparseIndex The sparse indices.
     * @param data The sparse data.
     * @throws UnsupportedOperationException This operation is not supported for dense datasets.
     */
    override fun addSparse(index: Int, sparseIndex: LongArray, data: FloatArray) {
        throw UnsupportedOperationException("addSparse not supported for dense dataset, use add")
    }
    
    /**
     * Adds a sparse data entry to the dense dataset.
     * This operation is not supported for dense datasets.
     * Use the `add` method instead.
     *
     * @param index The index at which to add the sparse data.
     * @param sparseIndex The sparse indices.
     * @param data The sparse data.
     * @throws UnsupportedOperationException This operation is not supported for dense datasets.
     */
    override fun addSparse(index: Int, sparseIndex: LongArray, data: DoubleArray) {
        throw UnsupportedOperationException("addSparse not supported for dense dataset, use add")
    }

    /**
     * Adds a sparse weighted data entry to the dense dataset.
     * This operation is not supported for dense datasets.
     * Use the `add` method instead.
     *
     * @param index The index at which to add the sparse weighted data.
     * @param sparseIndex The sparse indices.
     * @param weights The weights associated with the sparse data.
     * @param data The sparse data.
     * @throws UnsupportedOperationException This operation is not supported for dense datasets.
     */
    override fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: CharArray) {
        throw UnsupportedOperationException("addSparseWeighted not supported for dense dataset, use add")
    }

    /**
     * Adds a sparse weighted data entry to the dense dataset.
     * This operation is not supported for dense datasets.
     * Use the `add` method instead.
     *
     * @param index The index at which to add the sparse weighted data.
     * @param sparseIndex The sparse indices.
     * @param weights The weights associated with the sparse data.
     * @param data The sparse data.
     * @throws UnsupportedOperationException This operation is not supported for dense datasets.
     */
    override fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: IntArray) {
        throw UnsupportedOperationException("addSparseWeighted not supported for dense dataset, use add")
    }

    /**
     * Adds a sparse weighted data entry to the dense dataset.
     * This operation is not supported for dense datasets.
     * Use the `add` method instead.
     *
     * @param index The index at which to add the sparse weighted data.
     * @param sparseIndex The sparse indices.
     * @param weights The weights associated with the sparse data.
     * @param data The sparse data.
     * @throws UnsupportedOperationException This operation is not supported for dense datasets.
     */
    override fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: FloatArray) {
        throw UnsupportedOperationException("addSparseWeighted not supported for dense dataset, use add")
    }

    /**
     * Adds a sparse weighted data entry to the dense dataset.
     * This operation is not supported for dense datasets.
     * Use the `add` method instead.
     *
     * @param index The index at which to add the sparse weighted data.
     * @param sparseIndex The sparse indices.
     * @param weights The weights associated with the sparse data.
     * @param data The sparse data.
     * @throws UnsupportedOperationException This operation is not supported for dense datasets.
     */
    override fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: DoubleArray) {
        throw UnsupportedOperationException("addSparseWeighted not supported for dense dataset, use add")
    }
}
