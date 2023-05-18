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

class SparseDataSet(
    dim: Dim,
    dataType: DataType,
    private val sparseDensity: Double
) : DataSet(dim, dataType) {
    private val stride: Int
    private val sparseStart: LongArray
    private val sparseEnd: LongArray
    private val sparseIndex: LongArray

    init {
        require(sparseDensity in 0.0..1.0) { "sparseDensity should be between 0.0 and 1.0" }
        stride = (dim.stride * sparseDensity).toInt()
        sparseStart = LongArray(dim.examples)
        sparseEnd = LongArray(dim.examples)
        sparseIndex = LongArray(stride * dim.examples)
        data = ByteBuffer.allocateDirect(stride * dim.examples * DataType.sizeof(dataType))
        data.order(ByteOrder.nativeOrder())
    }

    override fun getStride(): Int = stride

    override fun getAttribute(): Int = Attribute.Sparse

    override fun getWeights(): FloatArray = FloatArray(0)

    override fun getSparseStart(): LongArray = sparseStart

    override fun getSparseEnd(): LongArray = sparseEnd

    override fun getSparseIndex(): LongArray = sparseIndex

    override fun add(index: Int, data: CharArray) {
        throw UnsupportedOperationException("add not supported for sparse dataset, use addSparse")
    }

    override fun add(index: Int, data: IntArray) {
        throw UnsupportedOperationException("add not supported for sparse dataset, use addSparse")
    }

    override fun add(index: Int, data: FloatArray) {
        throw UnsupportedOperationException("add not supported for sparse dataset, use addSparse")
    }

    override fun add(index: Int, data: DoubleArray) {
        throw UnsupportedOperationException("add not supported for sparse dataset, use addSparse")
    }

    override fun addWeighted(index: Int, data: CharArray, weights: FloatArray) {
        throw UnsupportedOperationException("addWeighted not supported for sparse dataset, use addSparse")
    }

    override fun addWeighted(index: Int, data: IntArray, weights: FloatArray) {
        throw UnsupportedOperationException("addWeighted not supported for sparse dataset, use addSparse")
    }

    override fun addWeighted(index: Int, data: FloatArray, weights: FloatArray) {
        throw UnsupportedOperationException("addWeighted not supported for sparse dataset, use addSparse")
    }

    override fun addWeighted(index: Int, data: DoubleArray, weights: FloatArray) {
        throw UnsupportedOperationException("addWeighted not supported for sparse dataset, use addSparse")
    }

    override fun addSparse(index: Int, sparseIndex: LongArray, data: CharArray) {
        checkLength(sparseIndex.size, data.size)
        val buffView = dataView.asCharBuffer()
        setPosition(buffView, index)
        buffView.put(data)
        putSparseInfo(index, sparseIndex)
    }

    override fun addSparse(index: Int, sparseIndex: LongArray, data: IntArray) {
        checkLength(sparseIndex.size, data.size)
        val buffView = dataView.asIntBuffer()
        setPosition(buffView, index)
        buffView.put(data)
        putSparseInfo(index, sparseIndex)
    }

    override fun addSparse(index: Int, sparseIndex: LongArray, data: FloatArray) {
        checkLength(sparseIndex.size, data.size)
        val buffView = dataView.asFloatBuffer()
        setPosition(buffView, index)
        buffView.put(data)
        putSparseInfo(index, sparseIndex)
    }

    override fun addSparse(index: Int, sparseIndex: LongArray, data: DoubleArray) {
        checkLength(sparseIndex.size, data.size)
        val buffView = dataView.asDoubleBuffer()
        setPosition(buffView, index)
        buffView.put(data)
        putSparseInfo(index, sparseIndex)
    }

    override fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: CharArray) {
        throw UnsupportedOperationException("addSparseWeighted not supported for sparse dataset, use addSparse")
    }

    override fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: IntArray) {
        throw UnsupportedOperationException("addSparseWeighted not supported for sparse dataset, use addSparse")
    }

    override fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: FloatArray) {
        throw UnsupportedOperationException("addSparseWeighted not supported for sparse dataset, use addSparse")
    }

    override fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: DoubleArray) {
        throw UnsupportedOperationException("addSparseWeighted not supported for sparse dataset, use addSparse")
    }

    private fun checkLength(sparseIndexLength: Int, sparseDataLength: Int) {
        require(sparseIndexLength == sparseDataLength) {
            "sparseIndex length ($sparseIndexLength) != sparseDataLength ($sparseDataLength)"
        }
        require(sparseIndexLength <= stride) {
            "Cannot add example larger than stride. Data length: $sparseIndexLength, Stride: $stride"
        }
    }

    private fun putSparseInfo(index: Int, sparseIndex: LongArray) {
        val offset = index * stride
        System.arraycopy(sparseIndex, 0, this.sparseIndex, offset, sparseIndex.size)
        sparseStart[index] = offset.toLong()
        sparseEnd[index] = offset + sparseIndex.size.toLong()
    }
}
