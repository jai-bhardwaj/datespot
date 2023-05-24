package com.system.tensorhub.data

import com.system.tensorhub.Dim
import com.system.tensorhub.DataSet
import com.system.tensorhub.DataSetEnums.Attribute
import com.system.tensorhub.DataSetEnums.DataType
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.CharBuffer
import java.nio.DoubleBuffer
import java.nio.FloatBuffer
import java.nio.IntBuffer

/**
 * Represents a sparse data set.
 *
 * @property sparseDensity The density of sparsity for the data set.
 * @property stride The stride value for the data set.
 * @property sparseStart The start positions of sparse data.
 * @property sparseEnd The end positions of sparse data.
 * @property sparseIndex The sparse index array.
 */
class SparseDataSet(dim: Dim, dataType: DataType, val sparseDensity: Double) : DataSet(dim, dataType) {

    private val stride: Int
    private val sparseStart: LongArray
    private val sparseEnd: LongArray
    private val sparseIndex: LongArray

    init {
        if (sparseDensity < 0.0 || sparseDensity > 1.0) {
            throw IllegalArgumentException("sparseDensity should be between 0.0 and 1.0")
        }
        this.sparseDensity = sparseDensity
        this.stride = (dim.stride * sparseDensity).toInt()

        this.sparseStart = LongArray(dim.examples)
        this.sparseEnd = LongArray(dim.examples)

        this.sparseIndex = LongArray(stride * dim.examples)
        this.data = ByteBuffer.allocateDirect(stride * dim.examples * dataType.sizeof())
        this.data.order(ByteOrder.nativeOrder())
    }

    /**
     * Gets the stride value for the data set.
     *
     * @return The stride value.
     */
    override fun getStride(): Int {
        return stride
    }

    /**
     * Gets the attribute of the data set.
     *
     * @return The attribute value.
     */
    override fun getAttribute(): Int {
        return Attribute.Sparse
    }

    /**
     * Gets the weights of the data set.
     *
     * @return The weights array.
     */
    override fun getWeights(): FloatArray {
        return EMPTY_FLOAT_ARRAY
    }

    /**
     * Gets the start positions of sparse data.
     *
     * @return The sparse start positions array.
     */
    override fun getSparseStart(): LongArray {
        return sparseStart
    }

    /**
     * Gets the end positions of sparse data.
     *
     * @return The sparse end positions array.
     */
    override fun getSparseEnd(): LongArray {
        return sparseEnd
    }

    /**
     * Gets the sparse index array.
     *
     * @return The sparse index array.
     */
    override fun getSparseIndex(): LongArray {
        return sparseIndex
    }

    /**
     * Adds data to the data set.
     *
     * @param index The index of the data.
     * @param data The data to add.
     * @throws UnsupportedOperationException This operation is not supported for sparse datasets. Use addSparse instead.
     */
    override fun add(index: Int, data: CharArray) {
        throw UnsupportedOperationException("add not supported for sparse dataset, use addSparse")
    }

    /**
     * Adds data to the data set.
     *
     * @param index The index of the data.
     * @param data The data to add.
     * @throws UnsupportedOperationException This operation is not supported for sparse datasets. Use addSparse instead.
     */
    override fun add(index: Int, data: IntArray) {
        throw UnsupportedOperationException("add not supported for sparse dataset, useaddSparse")
    }

    /**
     * Adds data to the data set.
     *
     * @param index The index of the data.
     * @param data The data to add.
     * @throws UnsupportedOperationException This operation is not supported for sparse datasets. Use addSparse instead.
     */
    override fun add(index: Int, data: FloatArray) {
        throw UnsupportedOperationException("add not supported for sparse dataset, use addSparse")
    }

    /**
     * Adds data to the data set.
     *
     * @param index The index of the data.
     * @param data The data to add.
     * @throws UnsupportedOperationException This operation is not supported for sparse datasets. Use addSparse instead.
     */
    override fun add(index: Int, data: DoubleArray) {
        throw UnsupportedOperationException("add not supported for sparse dataset, use addSparse")
    }

    /**
     * Adds weighted data to the data set.
     *
     * @param index The index of the data.
     * @param data The data to add.
     * @param weights The weights of the data.
     * @throws UnsupportedOperationException This operation is not supported for sparse datasets. Use addSparse instead.
     */
    override fun addWeighted(index: Int, data: CharArray, weights: FloatArray) {
        throw UnsupportedOperationException("addWeighted not supported for sparse dataset, use addSparse")
    }

    /**
     * Adds weighted data to the data set.
     *
     * @param index The index of the data.
     * @param data The data to add.
     * @param weights The weights of the data.
     * @throws UnsupportedOperationException This operation is not supported for sparse datasets. Use addSparse instead.
     */
    override fun addWeighted(index: Int, data: IntArray, weights: FloatArray) {
        throw UnsupportedOperationException("addWeighted not supported for sparse dataset, use addSparse")
    }

    /**
     * Adds weighted data to the data set.
     *
     * @param index The index of the data.
     * @param data The data to add.
     * @param weights The weights of the data.
     * @throws UnsupportedOperationException This operation is not supported for sparse datasets. Use addSparse instead.
     */
    override fun addWeighted(index: Int, data: FloatArray, weights: FloatArray) {
        throw UnsupportedOperationException("addWeighted not supported for sparse dataset, use addSparse")
    }

    /**
     * Adds weighted data to the data set.
     *
     * @param index The index of the data.
     * @param data The data to add.
     * @param weights The weights of the data.
     * @throws UnsupportedOperationException This operation is not supported for sparse datasets. Use addSparse instead.
     */
    override fun addWeighted(index: Int, data: DoubleArray, weights: FloatArray) {
        throw UnsupportedOperationException("addWeighted not supported for sparse dataset, use addSparse")
    }

    /**
     * Adds sparse data to the data set.
     *
     * @param index The index of the data.
     * @param sparseIndex The sparse index array.
     * @param data The data to add.
     */
    override fun addSparse(index: Int, sparseIndex: LongArray, data: CharArray) {
        checkLength(sparseIndex.size, data.size)
        val buffView: CharBuffer = this.data.asCharBuffer()
        setPosition(buffView, index)
        buffView.put(data)
        putSparseInfo(index, sparseIndex)
    }

    /**
     * Checks the length of the sparse index and data arrays.
     *
     * @param sparseIndexLength The length of the sparse index array.
     * @param sparseDataLength The length of the sparse data array.
     * @throws IllegalArgumentException If the lengths do not match.
     * @throws IllegalArgumentException If the sparse index length is greater than the stride.
     */
    private fun checkLength(sparseIndexLength: Int, sparseDataLength: Int) {
        if (sparseIndexLength != sparseDataLength) {
            throw IllegalArgumentException(
                "sparseIndex length ($sparseIndexLength) != sparseDataLength ($sparseDataLength)"
            )
        }
        if (sparseIndexLength > getStride()) {
            throw IllegalArgumentException(
                "Cannot add example larger than stride. Data length: $sparseIndexLength Stride: ${getStride()}"
            )
        }
    }

    /**
     * Puts the sparse index information into the data set.
     *
     * @param index The index of the data.
     * @param sparseIndex The sparse index array.
     */
    protected fun putSparseInfo(index: Int, sparseIndex: LongArray) {
        val offset = index * getStride()
        System.arraycopy(sparseIndex, 0, this.sparseIndex, offset, sparseIndex.size)
        this.sparseStart[index] = offset.toLong()
        this.sparseEnd[index] = offset + sparseIndex.size
    }

    /**
     * Adds sparse data to the data set.
     *
     * @param index The index of the data.
     * @param sparseIndex The sparse index array.
     * @param data The data to add.
     */
    override fun addSparse(index: Int, sparseIndex: LongArray, data: IntArray) {
        checkLength(sparseIndex.size, data.size)
        val buffView: IntBuffer = this.data.asIntBuffer()
        setPosition(buffView, index)
        buffView.put(data)
        putSparseInfo(index, sparseIndex)
    }

    /**
     * Adds sparse data to the data set.
     *
     * @param index The index of the data.
     * @param sparseIndex The sparse index array.
     * @param data The data to add.
     */
    override fun addSparse(index: Int, sparseIndex: LongArray, data: FloatArray) {
        checkLength(sparseIndex.size, data.size)
        val buffView: FloatBuffer = this.data.asFloatBuffer()
        setPosition(buffView, index)
        buffView.put(data)
        putSparseInfo(index, sparseIndex)
    }

    /**
     * Adds sparse data to the data set.
     *
     * @param index The index of the data.
     * @param sparseIndex The sparse index array.
     * @param data The data to add.
     */
    override fun addSparse(index: Int, sparseIndex: LongArray, data: DoubleArray) {
        checkLength(sparseIndex.size, data.size)
        val buffView: DoubleBuffer = this.data.asDoubleBuffer()
        buffView.put(data)
        setPosition(buffView, index)
        putSparseInfo(index, sparseIndex)
    }

    /**
     * Adds sparse weighted data to the data set.
     *
     * @param index The index of the data.
     * @param sparseIndex The sparse index array.
     * @param weights The weights of the data.
     * @param data The data to add.
     * @throws UnsupportedOperationException This operation is not supported for sparse datasets. Use addSparse instead.
     */
    override fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: CharArray) {
        throw UnsupportedOperationException("addSparseWeighted not supported for sparse dataset, use addSparse")
    }

    /**
     * Adds sparse weighted data to the data set.
     *
     * @param index The index of the data.
     * @param sparseIndex The sparse index array.
     * @param weights The weights of the data.
     * @param data The data to add.
     * @throws UnsupportedOperationException This operation is not supported for sparse datasets. Use addSparse instead.
     */
    override fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: IntArray) {
        throw UnsupportedOperationException("addSparseWeighted not supported for sparse dataset, use addSparse")
    }

    /**
     * Adds sparse weighted data to the data set.
     *
     * @param index The index of the data.
     * @param sparseIndex The sparse index array.
     * @param weights The weights of the data.
     * @param data The data to add.
     * @throws UnsupportedOperationException This operation is not supported for sparse datasets. Use addSparse instead.
     */
    override fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: FloatArray) {
        throw UnsupportedOperationException("addSparseWeighted not supported for sparse dataset, use addSparse")
    }

    /**
     * Adds sparse weighted data to the data set.
     *
     * @param index The index of the data.
     * @param sparseIndex The sparse index array.
     * @param weights The weights of the data.
     * @param data The data to add.
     * @throws UnsupportedOperationException This operation is not supported for sparse datasets. Use addSparse instead.
     */
    override fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: DoubleArray) {
        throw UnsupportedOperationException("addSparseWeighted not supported for sparse dataset, use addSparse")
    }

    companion object {
        private val EMPTY_FLOAT_ARRAY = FloatArray(0)
    }
}
