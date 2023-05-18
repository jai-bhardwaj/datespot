package com.system.tensorhub.data

import com.system.tensorhub.Dim
import com.system.tensorhub.DataSet
import com.system.tensorhub.DataSetEnums.DataType
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.DoubleBuffer
import java.nio.FloatBuffer
import java.nio.IntBuffer

class DenseDataSet(dim: Dim, dataType: DataType) : DataSet(dim, dataType) {
    private val data: ByteBuffer = ByteBuffer.allocateDirect(stride * dim.examples * DataType.sizeof(dataType)).order(ByteOrder.nativeOrder())

    override fun getStride(): Int = dim.stride

    override fun getSparseStart(): LongArray = EMPTY_LONG_ARRAY

    override fun getSparseEnd(): LongArray = EMPTY_LONG_ARRAY

    override fun getSparseIndex(): LongArray = EMPTY_LONG_ARRAY

    override fun getWeights(): FloatArray = EMPTY_FLOAT_ARRAY

    override fun add(index: Int, data: CharArray) {
        this.data.position(index * stride)
        this.data.asCharBuffer().put(data, 0, stride)
    }

    override fun add(index: Int, data: IntArray) {
        val buffView: IntBuffer = this.data.asIntBuffer()
        setPosition(buffView, index)
        buffView.put(data, 0, stride)
    }

    override fun add(index: Int, data: FloatArray) {
        val buffView: FloatBuffer = this.data.asFloatBuffer()
        setPosition(buffView, index)
        buffView.put(data, 0, stride)
    }

    override fun add(index: Int, data: DoubleArray) {
        val buffView: DoubleBuffer = this.data.asDoubleBuffer()
        setPosition(buffView, index)
        buffView.put(data, 0, stride)
    }

    override fun addWeighted(index: Int, data: CharArray, weights: FloatArray) {
        throw UnsupportedOperationException("addWeighted not supported for dense dataset, use add")
    }

    override fun addWeighted(index: Int, data: IntArray, weights: FloatArray) {
        throw UnsupportedOperationException("addWeighted not supported for dense dataset, use add")
    }

    override fun addWeighted(index: Int, data: FloatArray, weights: FloatArray) {
        throw UnsupportedOperationException("addWeighted not supported for dense dataset, use add")
    }

    override fun addWeighted(index: Int, data: DoubleArray, weights: FloatArray) {
        throw UnsupportedOperationException("addWeighted not supported for dense dataset, use add")
    }

    override fun addSparse(index: Int, sparseIndex: LongArray, data: CharArray) {
        throw UnsupportedOperationException("addSparse not supported for dense dataset, use add")
    }

    override fun addSparse(index: Int, sparseIndex: LongArray, data: IntArray) {
        throw UnsupportedOperationException("addSparse not supported for dense dataset, use add")
    }

    override fun addSparse(index: Int, sparseIndex: LongArray, data: FloatArray) {
        throw UnsupportedOperationException("addSparse not supported for dense dataset, use add")
    }

    override fun addSparse(index: Int, sparseIndex: LongArray, data: DoubleArray) {
        throw UnsupportedOperationException("addSparse not supported for dense dataset, use add")
    }

    override fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: CharArray) {
        throw UnsupportedOperationException("addSparseWeighted not supported for dense dataset, use add")
    }

    override fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: IntArray) {
        throw UnsupportedOperationException("addSparseWeighted not supported for dense dataset, use add")
    }

    override fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: FloatArray) {
        throw UnsupportedOperationException("addSparseWeighted not supported for dense dataset, use add")
    }

    override fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: DoubleArray) {
        throw UnsupportedOperationException("addSparseWeighted not supported for dense dataset, use add")
    }
}
