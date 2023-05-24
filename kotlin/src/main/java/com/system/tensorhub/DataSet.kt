package com.system.tensorhub

import java.nio.Buffer
import java.nio.ByteBuffer

import com.system.tensorhub.DataSetEnums.Attribute
import com.system.tensorhub.DataSetEnums.DataType
import com.system.tensorhub.DataSetEnums.Sharding

import lombok.Getter
import lombok.RequiredArgsConstructor
import lombok.Setter

/**
 * Represents a data set.
 *
 * @property dim The dimension of the data set.
 * @property dataType The data type of the data set.
 */
@Getter
@RequiredArgsConstructor
abstract class DataSet(val dim: Dim, val dataType: DataType) {

    companion object {
        protected val EMPTY_LONG_ARRAY = LongArray(0)
        protected val EMPTY_FLOAT_ARRAY = FloatArray(0)
    }

    @Setter
    var name = ""

    @Setter
    var layerName = ""

    protected var data: ByteBuffer? = null

    @Setter
    var sharding = Sharding.None

    /**
     * Gets the stride of the data set.
     */
    abstract val stride: Int

    /**
     * Gets the number of dimensions in the data set.
     */
    val dimensions: Int
        get() = dim.dimensions

    /**
     * Gets the size of the data set in the X dimension.
     */
    val dimX: Int
        get() = dim.x

    /**
     * Gets the size of the data set in the Y dimension.
     */
    val dimY: Int
        get() = dim.y

    /**
     * Gets the size of the data set in the Z dimension.
     */
    val dimZ: Int
        get() = dim.z

    /**
     * Gets the number of examples in the data set.
     */
    val examples: Int
        get() = dim.examples

    /**
     * Gets the attribute of the data set.
     */
    open val attribute: Int
        get() = Attribute.None

    /**
     * Gets the data buffer of the data set.
     */
    val dataBuffer: ByteBuffer?
        get() {
            setPosition(data, 0)
            return data
        }

    /**
     * Gets the start indices of sparse data.
     *
     * @return The start indices.
     */
    abstract fun getSparseStart(): LongArray

    /**
     * Gets the end indices of sparse data.
     *
     * @return The end indices.
     */
    abstract fun getSparseEnd(): LongArray

    /**
     * Gets the indices of sparse data.
     *
     * @return The indices.
     */
    abstract fun getSparseIndex(): LongArray

    /**
     * Gets the weights of the data set.
     *
     * @return The weights.
     */
    abstract fun getWeights(): FloatArray

    /**
     * Gets the ordinal value of the data type.
     *
     * @return The ordinal value.
     */
    val dataTypeOrdinal: Int
        get() = dataType.ordinal

    /**
     * Sets the position of the buffer.
     *
     * @param buff The buffer.
     * @param index The index.
     */
    protected fun setPosition(buff: Buffer, index: Int) {
        buff.position(index * stride)
    }

    /**
     * Checks if the data set is sparse.
     *
     * @return `true` if the data set is sparse, `false` otherwise.
     */
    val isSparse: Boolean
        get() = (attribute and Attribute.Sparse) != 0

    /**
     * Checks if the data set is boolean.
     *
     * @return `true` if the data set is boolean, `false` otherwise.
     */
    val isBoolean: Boolean
        get() = (attribute and Attribute.Boolean) != 0

    /**
     * Checks if the data set is weighted.
     *
     * @return `true` if the data set is weighted, `false` otherwise.
     */
    val isWeighted: Boolean
        get() = (attribute and Attribute.Weighted) != 0

    /**
     * Checks if the data set is indexed.
     *
     * @return `true` if the data set is indexed, `false` otherwise.
     */
    val isIndexed: Boolean
        get() = (attribute and Attribute.Indexed) != 0

    /**
     * Adds data to the data set at the specified index.
     *
     * @param index The index.
     * @param data The data.
     */
    abstract fun add(index: Int, data: CharArray)

    /**
     * Adds data to the data set at the specified index.
     *
     * @param index The index.
     * @param data The data.
     */
    abstract fun add(index: Int, data: IntArray)

    /**
     * Adds data to the data set at the specified index.
     *
     * @param index The index.
     * @param data The data.
     */
    abstract fun add(index: Int, data: FloatArray)

    /**
     * Adds data to the data set at the specified index.
     *
     * @param index The index.
     * @param data The data.
     */
    abstract fun add(index: Int, data: DoubleArray)

    /**
     * Adds weighted data to the data set at the specified index.
     *
     * @param index The index.
     * @param data The data.
     * @param weights The weights.
     */
    abstract fun addWeighted(index: Int, data: CharArray, weights: FloatArray)

    /**
     * Adds weighted data to the data set at the specified index.
     *
     * @param index The index.
     * @param data The data.
     * @param weights The weights.
     */
    abstract fun addWeighted(index: Int, data: IntArray, weights: FloatArray)

    /**
     * Adds weighted data to the data set at the specified index.
     *
     * @param index The index.
     * @param data The data.
     * @param weights The weights.
     */
    abstract fun addWeighted(index: Int, data: FloatArray, weights: FloatArray)

    /**
     * Adds weighted data to the data set at the specified index.
     *
     * @param index The index.
     * @param data The data.
     * @param weights The weights.
     */
    abstract fun addWeighted(index: Int, data: DoubleArray, weights: FloatArray)

    /**
     * Adds sparse data to the data set at the specified index.
     *
     * @param index The index.
     * @param sparseIndex The sparse indices.
     * @param data The data.
     */
    abstract fun addSparse(index: Int, sparseIndex: LongArray, data: CharArray)

    /**
     * Adds sparse data to the data set at the specified index.
     *
     * @param index The index.
     * @param sparseIndex The sparse indices.
     * @param data The data.
     */
    abstract fun addSparse(index: Int, sparseIndex: LongArray, data: IntArray)

    /**
     * Adds sparse data to the data set at the specified index.
     *
     * @param index The index.
     * @param sparseIndex The sparse indices.
     * @param data The data.
     */
    abstract fun addSparse(index: Int, sparseIndex: LongArray, data: FloatArray)

    /**
     * Adds sparse data to the data set at the specified index.
     *
     * @param index The index.
     * @param sparseIndex The sparse indices.
     * @param data The data.
     */
    abstract fun addSparse(index: Int, sparseIndex: LongArray, data: DoubleArray)

    /**
     * Adds weighted sparse data to the data set at the specified index.
     *
     * @param index The index.
     * @param sparseIndex The sparse indices.
     * @param weights The weights.
     * @param data The data.
     */
    abstract fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: CharArray)

    /**
     * Adds weighted sparse data to the data set at the specified index.
     *
     * @param index The index.
     * @param sparseIndex The sparse indices.
     * @param weights The weights.
     * @param data The data.
     */
    abstract fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: IntArray)

    /**
     * Adds weighted sparse data to the data set at the specified index.
     *
     * @param index The index.
     * @param sparseIndex The sparse indices.
     * @param weights The weights.
     * @param data The data.
     */
    abstract fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: FloatArray)

    /**
     * Adds weighted sparse data to the data set at the specified index.
     *
     * @param index The index.
     * @param sparseIndex The sparse indices.
     * @param weights The weights.
     * @param data The data.
     */
    abstract fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: DoubleArray)
}
