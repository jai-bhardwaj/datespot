package com.system.tensorhub

import java.nio.Buffer
import java.nio.ByteBuffer

import com.system.tensorhub.DataSetEnums.Attribute
import com.system.tensorhub.DataSetEnums.DataType

import lombok.Getter
import lombok.RequiredArgsConstructor
import lombok.Setter

/**
 * Abstract class representing a DataSet.
 */
@Getter
@RequiredArgsConstructor
abstract class DataSet {
    /**
     * Empty long array constant.
     */
    protected val EMPTY_LONG_ARRAY: LongArray = LongArray(0)

    /**
     * Empty float array constant.
     */
    protected val EMPTY_FLOAT_ARRAY: FloatArray = FloatArray(0)

    /**
     * Name of the DataSet.
     */
    @Setter
    var name: String = ""

    /**
     * Layer name associated with the DataSet.
     */
    @Setter
    var layerName: String = ""

    /**
     * Dimension of the DataSet.
     */
    private val dim: Dim,
    /**
     * Data type of the DataSet.
     */
    private val dataType: DataType,

    /**
     * Data buffer.
     */
    protected var data: ByteBuffer? = null

    /**
     * Sharding type of the DataSet.
     */
    @Setter
    var sharding: Sharding = Sharding.None

    /**
     * Returns the stride of the DataSet.
     *
     * @return The stride as an integer value.
     */
    abstract fun getStride(): Int

    /**
     * Returns the number of dimensions of the DataSet.
     */
    val dimensions: Int
        get() = dim.dimensions

    /**
     * Returns the x-dimension of the DataSet.
     */
    val dimX: Int
        get() = dim.x

    /**
     * Returns the y-dimension of the DataSet.
     */
    val dimY: Int
        get() = dim.y

    /**
     * Returns the z-dimension of the DataSet.
     */
    val dimZ: Int
        get() = dim.z

    /**
     * Returns the number of examples in the DataSet.
     */
    val examples: Int
        get() = dim.examples

    /**
     * Returns the attribute of the DataSet.
     */
    val attribute: Int
        get() = Attribute.None

    /**
     * Returns the data buffer of the DataSet.
     */
    fun getData(): ByteBuffer? {
        setPosition(data, 0)
        return data
    }

    /**
     * Gets the sparse start values.
     */
    abstract fun getSparseStart(): LongArray

    /**
     * Gets the sparse end values.
     */
    abstract fun getSparseEnd(): LongArray

    /**
     * Gets the sparse index values.
     */
    abstract fun getSparseIndex(): LongArray

    /**
     * Gets the weights.
     */
    abstract fun getWeights(): FloatArray

    /**
     * Gets the ordinal value of the data type.
     */
    val dataTypeOrdinal: Int
        get() = dataType.ordinal

    /**
     * Sets the position of the buffer.
     */
    protected fun setPosition(buff: Buffer, index: Int) {
        buff.position(index * getStride())
    }

    /**
     * Checks if the data set is sparse.
     */
    fun isSparse(): Boolean {
        return (attribute and Attribute.Sparse) != 0
    }

    /**
     * Checks if the data set is boolean.
     */
    fun isBoolean(): Boolean {
        return (attribute and Attribute.Boolean) != 0
    }

    /**
     * Checks if the data set is weighted.
     */
    fun isWeighted(): Boolean {
        return (attribute and Attribute.Weighted) != 0
    }

    /**
     * Checks if the data set is indexed.
     */
    fun isIndexed(): Boolean {
        return (attribute and Attribute.Indexed) != 0
    }

    /**
     * Adds data with a character array.
     */
    abstract fun add(index: Int, data: CharArray)

    /**
     * Adds data with an integer array.
     */
    abstract fun add(index: Int, data: IntArray)

    /**
     * Adds data with a float array.
     */
    abstract fun add(index: Int, data: FloatArray)

    /**
     * Adds data with a double array.
     */
    abstract fun add(index: Int, data: DoubleArray)

    /**
     * Adds weighted data with a character array.
     */
    abstract fun addWeighted(index: Int, data: CharArray, weights: FloatArray)

    /**
     * Adds weighted data with an integer array.
     */
    abstract fun addWeighted(index: Int, data: IntArray, weights: FloatArray)

    /**
     * Adds weighted data with a float array.
     */
    abstract fun addWeighted(index: Int, data: FloatArray, weights: FloatArray)

    /**
     * Adds weighted data with a double array.
     */
    abstract fun addWeighted(index: Int, data: DoubleArray, weights: FloatArray)

    /**
     * Adds sparse data with a character array.
     */
    abstract fun addSparse(index: Int, sparseIndex: LongArray, data: CharArray)

    /**
     * Adds sparse data with an integer array.
     */
    abstract fun addSparse(index: Int, sparseIndex: LongArray, data: IntArray)

    /**
     * Adds sparse data with a float array.
     */
    abstract fun addSparse(index: Int, sparseIndex: LongArray, data: FloatArray)

    /**
     * Adds sparse data with a double array.
     */
    abstract fun addSparse(index: Int, sparseIndex: LongArray, data: DoubleArray)

    /**
     * Adds sparse weighted data with a character array.
     */
    abstract fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: CharArray)

    /**
     * Adds sparse weighted data with an integer array.
     */
    abstract fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: IntArray)

    /**
     * Adds sparse weighted data with a float array.
     */
    abstract fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: FloatArray)

    /**
     * Adds sparse weighted data with a double array.
     */
    abstract fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: DoubleArray)
}
