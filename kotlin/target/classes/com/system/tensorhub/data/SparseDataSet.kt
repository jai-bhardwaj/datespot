package com.system.tensorhub.data

import com.system.tensorhub.DataSet
import com.system.tensorhub.DataSetEnums.Attribute
import com.system.tensorhub.DataSetEnums.DataType
import com.system.tensorhub.Dim
import java.nio.ByteBuffer
import java.nio.ByteOrder

import java.nio.CharBuffer
import java.nio.DoubleBuffer
import java.nio.FloatBuffer
import java.nio.IntBuffer

import com.system.tensorhub.DataSetEnums.DataType.sizeof

import lombok.Getter

/**
 * Represents a sparse data set, extending the DataSet class.
 *
 * @param dim The dimensions of the sparse data set.
 * @param dataType The data type of the sparse data set.
 * @param sparseDensity The density of sparsity for the data set.
 */
@Getter
class SparseDataSet(
    dim: Dim, 
    dataType: DataType, 
    private val sparseDensity: Double
) : DataSet(dim, dataType) {

    /**
     * The stride of the sparse data set.
     */
    private val stride: Int

    /**
     * The start indices of sparse data.
     */
    private val sparseStart: LongArray

    /**
     * The end indices of sparse data.
     */
    private val sparseEnd: LongArray

    /**
     * The indices of sparse data.
     */
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
        this.data = ByteBuffer.allocateDirect(stride * dim.examples * DataType.sizeof(dataType))
        this.data.order(ByteOrder.nativeOrder())
    }

    /**
     * @return The stride of the sparse data set.
     */
    override fun getStride(): Int {
        return stride
    }

    /**
     * @return The attribute value for the sparse data set.
     */
    override fun getAttribute(): Int {
        return Attribute.Sparse
    }

    /**
     * @return The weights of the sparse data set.
     */
    override fun getWeights(): FloatArray {
        return EMPTY_FLOAT_ARRAY
    }
    
    /**
     * @return The start indices of sparse data.
     */
    override fun getSparseStart(): LongArray {
        return sparseStart
    }
    
    /**
     * @return The end indices of sparse data.
     */
    override fun getSparseEnd(): LongArray {
        return sparseEnd
    }
    
    /**
     * @return The indices of sparse data.
     */
    override fun getSparseIndex(): LongArray {
        return sparseIndex
    }
    
    /**
     * Adds data to the sparse data set at the specified index.
     * This method is not supported for the sparse data set. Use addSparse instead.
     *
     * @param index The index at which the data should be added.
     * @param data The data to be added.
     * @throws UnsupportedOperationException This exception is always thrown, as adding data directly is not supported for the sparse data set.
     */
    override fun add(index: Int, data: CharArray) {
        throw UnsupportedOperationException("add not supported for sparse dataset, use addSparse")
    }

    /**
     * Adds data to the dataset at the specified index.
     * This method is not supported for sparse datasets. Use the addSparse method instead.
     *
     * @param index The index at which to add the data.
     * @param data The integer array representing the data to be added.
     * @throws UnsupportedOperationException if add is called for a sparse dataset.
     */
    override fun add(index: Int, data: IntArray) {
        throw UnsupportedOperationException("add not supported for sparse dataset, use addSparse")
    }

    /**
     * Adds data to the dataset at the specified index.
     * This method is not supported for sparse datasets. Use the addSparse method instead.
     *
     * @param index The index at which to add the data.
     * @param data The float array representing the data to be added.
     * @throws UnsupportedOperationException if add is called for a sparse dataset.
     */
    override fun add(index: Int, data: FloatArray) {
        throw UnsupportedOperationException("add not supported for sparse dataset, use addSparse")
    }

    /**
     * Adds data to the dataset at the specified index.
     * This method is not supported for sparse datasets. Use the addSparse method instead.
     *
     * @param index The index at which to add the data.
     * @param data The double array representing the data to be added.
     * @throws UnsupportedOperationException if add is called for a sparse dataset.
     */
    override fun add(index: Int, data: DoubleArray) {
        throw UnsupportedOperationException("add not supported for sparse dataset, use addSparse")
    }
    
    /**
     * Adds weighted data to the dataset at the specified index.
     * This method is not supported for sparse datasets. Use the addSparse method instead.
     *
     * @param index The index at which to add the data.
     * @param data The character array representing the data to be added.
     * @param weights The array of weights corresponding to the data.
     * @throws UnsupportedOperationException if addWeighted is called for a sparse dataset.
     */
    override fun addWeighted(index: Int, data: CharArray, weights: FloatArray) {
        throw UnsupportedOperationException("addWeighted not supported for sparse dataset, use addSparse")
    }

    /**
     * Adds weighted data to the dataset at the specified index.
     * This method is not supported for sparse datasets. Use the addSparse method instead.
     *
     * @param index The index at which to add the data.
     * @param data The integer array representing the data to be added.
     * @param weights The array of weights corresponding to the data.
     * @throws UnsupportedOperationException if addWeighted is called for a sparse dataset.
     */
    override fun addWeighted(index: Int, data: IntArray, weights: FloatArray) {
        throw UnsupportedOperationException("addWeighted not supported for sparse dataset, use addSparse")
    }

    /**
     * Adds weighted data to the dataset at the specified index.
     * This method is not supported for sparse datasets. Use the addSparse method instead.
     *
     * @param index The index at which to add the data.
     * @param data The float array representing the data to be added.
     * @param weights The array of weights corresponding to the data.
     * @throws UnsupportedOperationException if addWeighted is called for a sparse dataset.
     */
    override fun addWeighted(index: Int, data: FloatArray, weights: FloatArray) {
        throw UnsupportedOperationException("addWeighted not supported for sparse dataset, use addSparse")
    }

    /**
     * Adds weighted data to the dataset at the specified index.
     * This method is not supported for sparse datasets. Use the addSparse method instead.
     *
     * @param index The index at which to add the data.
     * @param data The double array representing the data to be added.
     * @param weights The array of weights corresponding to the data.
     * @throws UnsupportedOperationException if addWeighted is called for a sparse dataset.
     */
    override fun addWeighted(index: Int, data: DoubleArray, weights: FloatArray) {
        throw UnsupportedOperationException("addWeighted not supported for sparse dataset, use addSparse")
    }
    
    /**
     * Adds sparse data to the dataset at the specified index.
     *
     * @param index The index at which to add the data.
     * @param sparseIndex The array of long values representing the sparse indices.
     * @param data The character array representing the sparse data.
     */
    override fun addSparse(index: Int, sparseIndex: LongArray, data: CharArray) {
        checkLength(sparseIndex.size, data.size)
        val buffView = this.data.asCharBuffer()
        setPosition(buffView, index)
        buffView.put(data)
        putSparseInfo(index, sparseIndex)
    }

    /**
     * Checks the length of the sparse index and data arrays and throws an exception if they are not equal.
     * Additionally, throws an exception if the sparse index length exceeds the stride value.
     *
     * @param sparseIndexLength The length of the sparse index array.
     * @param sparseDataLength The length of the sparse data array.
     * @throws IllegalArgumentException if the lengths are not equal or if the sparse index length exceeds the stride value.
     */
    private fun checkLength(sparseIndexLength: Int, sparseDataLength: Int) {
        if (sparseIndexLength != sparseDataLength) {
            throw IllegalArgumentException(
                "sparseIndex length ($sparseIndexLength) != sparseDataLength ($sparseDataLength)")
        }
        if (sparseIndexLength > stride) {
            throw IllegalArgumentException(
                "Cannot add example larger than stride. Data length: $sparseIndexLength, Stride: $stride")
        }
    }

    /**
     * Copies the sparse index array to the dataset's sparse index array at the specified offset.
     * Updates the sparse start and end arrays accordingly.
     *
     * @param index The index at which to add the sparse index.
     * @param sparseIndex The array of long values representing the sparse indices.
     */
    protected fun putSparseInfo(index: Int, sparseIndex: LongArray) {
        val offset = index * stride
        System.arraycopy(sparseIndex, 0, this.sparseIndex, offset, sparseIndex.size)
        sparseStart[index] = offset
        sparseEnd[index] = offset + sparseIndex.size
    }
    
    /**
     * Adds sparse data to the dataset at the specified index.
     *
     * @param index The index at which to add the data.
     * @param sparseIndex The array of long values representing the sparse indices.
     * @param data The integer array representing the sparse data.
     */
    override fun addSparse(index: Int, sparseIndex: LongArray, data: IntArray) {
        checkLength(sparseIndex.size, data.size)
        val buffView = this.data.asIntBuffer()
        setPosition(buffView, index)
        buffView.put(data)
        putSparseInfo(index, sparseIndex)
    }

    /**
     * Adds sparse data to the dataset at the specified index.
     *
     * @param index The index at which to add the data.
     * @param sparseIndex The array of long values representing the sparse indices.
     * @param data The float array representing the sparse data.
     */
    override fun addSparse(index: Int, sparseIndex: LongArray, data: FloatArray) {
        checkLength(sparseIndex.size, data.size)
        val buffView = this.data.asFloatBuffer()
        setPosition(buffView, index)
        buffView.put(data)
        putSparseInfo(index, sparseIndex)
    }

    /**
     * Adds sparse data to the dataset at the specified index.
     *
     * @param index The index at which to add the data.
     * @param sparseIndex The array of long values representing the sparse indices.
     * @param data The double array representing the sparse data.
     */
    override fun addSparse(index: Int, sparseIndex: LongArray, data: DoubleArray) {
        checkLength(sparseIndex.size, data.size)
        val buffView = this.data.asDoubleBuffer()
        buffView.put(data)
        setPosition(buffView, index)
        putSparseInfo(index, sparseIndex)
    }
    
    /**
     * Adds sparse weighted data to the dataset at the specified index.
     * This method is not supported for sparse datasets. Use the addSparse method instead.
     *
     * @param index The index at which to add the data.
     * @param sparseIndex The array of long values representing the sparse indices.
     * @param weights The array of weights corresponding to the data.
     * @param data The character array representing the sparse data.
     * @throws UnsupportedOperationException if addSparseWeighted is called for a sparse dataset.
     */
    override fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: CharArray) {
        throw UnsupportedOperationException("addSparseWeighted not supported for sparse dataset, use addSparse")
    }

    /**
     * Adds sparse weighted data to the dataset at the specified index.
     * This method is not supported for sparse datasets. Use the addSparse method instead.
     *
     * @param index The index at which to add the data.
     * @param sparseIndex The array of long values representing the sparse indices.
     * @param weights The array of weights corresponding to the data.
     * @param data The integer array representing the sparse data.
     * @throws UnsupportedOperationException if addSparseWeighted is called for a sparse dataset.
     */
    override fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: IntArray) {
        throw UnsupportedOperationException("addSparseWeighted not supported for sparse dataset, use addSparse")
    }

    /**
     * Adds sparse weighted data to the dataset at the specified index.
     * This method is not supported for sparse datasets. Use the addSparse method instead.
     *
     * @param index The index at which to add the data.
     * @param sparseIndex The array of long values representing the sparse indices.
     * @param weights The array of weights corresponding to the data.
     * @param data The float array representing the sparse data.
     * @throws UnsupportedOperationException if addSparseWeighted is called for a sparse dataset.
     */
    override fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: FloatArray) {
        throw UnsupportedOperationException("addSparseWeighted not supported for sparse dataset, use addSparse")
    }

    /**
     * Adds sparse weighted data to the dataset at the specified index.
     * This method is not supported for sparse datasets. Use the addSparse method instead.
     *
     * @param index The index at which to add the data.
     * @param sparseIndex The array of long values representing the sparse indices.
     * @param weights The array of weights corresponding to the data.
     * @param data The double array representing the sparse data.
     * @throws UnsupportedOperationException if addSparseWeighted is called for a sparse dataset.
     */
    override fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: DoubleArray) {
        throw UnsupportedOperationException("addSparseWeighted not supported for sparse dataset, use addSparse")
    }
}
