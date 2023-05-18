package com.system.tensorhub.data

import com.system.tensorhub.Dim
import com.system.tensorhub.DataSetEnums.DataType
import java.util.Arrays

class DenseWeightedDataSet(dim: Dim, dataType: DataType) : DenseDataSet(dim, dataType) {
    private val weights: FloatArray = FloatArray(dim.stride * dim.examples)

    override fun getWeights(): FloatArray = weights

    private fun putWeightOne(index: Int) {
        Arrays.fill(weights, index * stride, index * stride + stride, 1f)
    }

    private fun putWeight(index: Int, weights: FloatArray) {
        System.arraycopy(weights, 0, this.weights, index * stride, stride)
    }

    override fun add(index: Int, data: CharArray) {
        super.add(index, data)
        putWeightOne(index)
    }

    override fun add(index: Int, data: IntArray) {
        super.add(index, data)
        putWeightOne(index)
    }

    override fun add(index: Int, data: FloatArray) {
        super.add(index, data)
        putWeightOne(index)
    }

    override fun add(index: Int, data: DoubleArray) {
        super.add(index, data)
        putWeightOne(index)
    }

    override fun addWeighted(index: Int, data: CharArray, weights: FloatArray) {
        super.add(index, data)
        putWeight(index, weights)
    }

    override fun addWeighted(index: Int, data: IntArray, weights: FloatArray) {
        super.add(index, data)
        putWeight(index, weights)
    }

    override fun addWeighted(index: Int, data: FloatArray, weights: FloatArray) {
        super.add(index, data)
        putWeight(index, weights)
    }

    override fun addWeighted(index: Int, data: DoubleArray, weights: FloatArray) {
        super.add(index, data)
        putWeight(index, weights)
    }
}
