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
    private val dim: Dim

    /**
     * Data type of the DataSet.
     */
    private val dataType: DataType

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
     *
     * @return The number of dimensions as an integer value.
     */
    fun getDimensions(): Int {
        return dim.dimensions
    }

    /**
     * Returns the x-dimension of the DataSet.
     *
     * @return The x-dimension as an integer value.
     */
    fun getDimX(): Int {
        return dim.x
    }

    /**
     * Returns the y-dimension of the DataSet.
     *
     * @return The y-dimension as an integer value.
     */
    fun getDimY(): Int {
        return dim.y
    }

    /**
     * Returns the z-dimension of the DataSet.
     *
     * @return The z-dimension as an integer value.
     */
    fun getDimZ(): Int {
        return dim.z
    }

    /**
     * Returns the number of examples in the DataSet.
     *
     * @return The number of examples as an integer value.
     */
    fun getExamples(): Int {
        return dim.examples
    }

    /**
     * Returns the attribute of the DataSet.
     *
     * @return The attribute as an integer value.
     */
    fun getAttribute(): Int {
        return Attribute.None
    }

    /**
     * Returns the data buffer of the DataSet.
     *
     * @return The data buffer as a ByteBuffer.
     */
    fun getData(): ByteBuffer? {
        setPosition(data, 0)
        return data
    }

    /**
     * Gets the sparse start values.
     *
     * @return The sparse start values as a LongArray.
     */
    abstract fun getSparseStart(): LongArray

    /**
     * Gets the sparse end values.
     *
     * @return The sparse end values as a LongArray.
     */
    abstract fun getSparseEnd(): LongArray

    /**
     * Gets the sparse index values.
     *
     * @return The sparse index values as a LongArray.
     */
    abstract fun getSparseIndex(): LongArray

    /**
     * Gets the weights.
     *
     * @return The weights as a FloatArray.
     */
    abstract fun getWeights(): FloatArray

    /**
     * Gets the ordinal value of the data type.
     *
     * @return The ordinal value as an integer.
     */
    fun getDataTypeOrdinal(): Int {
        return getDataType().ordinal
    }

    /**
     * Sets the position of the buffer.
     *
     * @param buff The buffer to set the position.
     * @param index The index to calculate the position.
     */
    protected fun setPosition(buff: Buffer, index: Int) {
        buff.position(index * getStride())
    }

    /**
     * Checks if the data set is sparse.
     *
     * @return true if the data set is sparse, false otherwise.
     */
    fun isSparse(): Boolean {
        return (getAttribute() and Attribute.Sparse) != 0
    }

    /**
     * Checks if the data set is boolean.
     *
     * @return true if the data set is boolean, false otherwise.
     */
    fun isBoolean(): Boolean {
        return (getAttribute() and Attribute.Boolean) != 0
    }

    /**
     * Checks if the data set is weighted.
     *
     * @return true if the data set is weighted, false otherwise.
     */
    fun isWeighted(): Boolean {
        return (getAttribute() and Attribute.Weighted) != 0
    }

    /**
     * Checks if the data set is indexed.
     *
     * @return true if the data set is indexed, false otherwise.
     */
    fun isIndexed(): Boolean {
        return (getAttribute() and Attribute.Indexed) != 0
    }


    /**
     * Adds data with a character array.
     *
     * @param index The index to add the data.
     * @param data The character array data.
     */
    abstract fun add(index: Int, data: CharArray)

    /**
     * Adds data with an integer array.
     *
     * @param index The index to add the data.
     * @param data The integer array data.
     */
    abstract fun add(index: Int, data: IntArray)

    /**
     * Adds data with a float array.
     *
     * @param index The index to add the data.
     * @param data The float array data.
     */
    abstract fun add(index: Int, data: FloatArray)

    /**
     * Adds data with a double array.
     *
     * @param index The index to add the data.
     * @param data The double array data.
     */
    abstract fun add(index: Int, data: DoubleArray)

    /**
     * Adds weighted data with a character array.
     *
     * @param index The index to add the data.
     * @param data The character array data.
     * @param weights The weight array.
     */
    abstract fun addWeighted(index: Int, data: CharArray, weights: FloatArray)

    /**
     * Adds weighted data with an integer array.
     *
     * @param index The index to add the data.
     * @param data The integer array data.
     * @param weights The weight array.
     */
    abstract fun addWeighted(index: Int, data: IntArray, weights: FloatArray)

    /**
     * Adds weighted data with a float array.
     *
     * @param index The index to add the data.
     * @param data The float array data.
     * @param weights The weight array.
     */
    abstract fun addWeighted(index: Int, data: FloatArray, weights: FloatArray)

    /**
     * Adds weighted data with a double array.
     *
     * @param index The index to add the data.
     * @param data The double array data.
     * @param weights The weight array.
     */
    abstract fun addWeighted(index: Int, data: DoubleArray, weights: FloatArray)

    /**
     * Adds sparse data with a character array.
     *
     * @param index The index to add the data.
     * @param sparseIndex The sparse index.
     * @param data The character array data.
     */
    abstract fun addSparse(index: Int, sparseIndex: LongArray, data: CharArray)

    /**
     * Adds sparse data with an integer array.
     *
     * @param index The index to add the data.
     * @param sparseIndex The sparse index.
     * @param data The integer array data.
     */
    abstract fun addSparse(index: Int, sparseIndex: LongArray, data: IntArray)

    /**
     * Adds sparse data with a float array.
     *
     * @param index The index to add the data.
     * @param sparseIndex The sparse index.
     * @param data The float array data.
     */
    abstract fun addSparse(index: Int, sparseIndex: LongArray, data: FloatArray)

    /**
     * Adds sparse data with a double array.
     *
     * @param index The index to add the data.
     * @param sparseIndex The sparse index.
     * @param data The double array data.
     */
    abstract fun addSparse(index: Int, sparseIndex: LongArray, data: DoubleArray)

    /**
     * Adds sparse weighted data with a character array.
     *
     * @param index The index to add the data.
     * @param sparseIndex The sparse index.
     * @param weights The weight array.
     * @param data The character array data.
     */
    abstract fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: CharArray)

    /**
     * Adds sparse weighted data with an integer array.
     *
     * @param index The index to add the data.
     * @param sparseIndex The sparse index.
     * @param weights The weight array.
     * @param data The integer array data.
     */
    abstract fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: IntArray)

    /**
     * Adds sparse weighted data with a float array.
     *
     * @param index The index to add the data.
     * @param sparseIndex The sparse index.
     * @param weights The weight array.
     * @param data The float array data.
     */
    abstract fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: FloatArray)

    /**
     * Adds sparse weighted data with a double array.
     *
     * @param index The index to add the data.
     * @param sparseIndex The sparse index.
     * @param weights The weight array.
     * @param data The double array data.
     */
    abstract fun addSparseWeighted(index: Int, sparseIndex: LongArray, weights: FloatArray, data: DoubleArray)
        

}
