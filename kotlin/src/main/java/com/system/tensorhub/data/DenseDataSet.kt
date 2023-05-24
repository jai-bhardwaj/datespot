package com.system.tensorhub.data;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import com.system.tensorhub.Dim;
import com.system.tensorhub.DataSet;
import com.system.tensorhub.DataSetEnums.DataType;

/**
 * Represents a dense dataset.
 */
class DenseDataSet extends DataSet {

    /**
     * Constructs a DenseDataSet with the specified dimensions and data type.
     *
     * @param dim      The dimensions of the dataset.
     * @param dataType The data type of the dataset.
     */
    DenseDataSet(Dim dim, DataType dataType) {
        super(dim, dataType);
    }

    /**
     * Initializes the dataset by allocating memory for the data buffer.
     * The buffer size is calculated based on the stride, number of examples, and data type.
     */
    @Override
    void init() {
        data = ByteBuffer.allocateDirect(getStride() * dim.examples * DataType.sizeof(dataType))
                .order(ByteOrder.nativeOrder());
    }

    /**
     * Retrieves the stride of the dataset.
     *
     * @return The stride of the dataset.
     */
    @Override
    int getStride() {
        return dim.stride;
    }

    /**
     * Retrieves the sparse start values of the dataset.
     *
     * @return An empty long array, as sparse start is not applicable for dense dataset.
     */
    @Override
    long[] getSparseStart() {
        return EMPTY_LONG_ARRAY;
    }

    /**
     * Retrieves the sparse end values of the dataset.
     *
     * @return An empty long array, as sparse end is not applicable for dense dataset.
     */
    @Override
    long[] getSparseEnd() {
        return EMPTY_LONG_ARRAY;
    }

    /**
     * Retrieves the sparse indices of the dataset.
     *
     * @return An empty long array, as sparse indices are not applicable for dense dataset.
     */
    @Override
    long[] getSparseIndex() {
        return EMPTY_LONG_ARRAY;
    }

    /**
     * Retrieves the weights of the dataset.
     *
     * @return An empty float array, as weights are not applicable for dense dataset.
     */
    @Override
    float[] getWeights() {
        return EMPTY_FLOAT_ARRAY;
    }

    /**
     * Adds data to the dataset at the specified index.
     * This method is used for adding character array data.
     *
     * @param index The index at which to add the data.
     * @param data  The character array data to add.
     */
    @Override
    void add(int index, char[] data) {
        this.data.position(index * getStride());
        this.data.asCharBuffer().put(data, 0, getStride());
    }

    /**
     * Adds data to the dataset at the specified index.
     * This method is used for adding integer array data.
     *
     * @param index The index at which to add the data.
     * @param data  The integer array data to add.
     */
    @Override
    void add(int index, int[] data) {
        IntBuffer buffView = this.data.asIntBuffer();
        setPosition(buffView, index);
        buffView.put(data, 0, getStride());
    }

    /**
     * Adds data to the dataset at the specified index.
     * This method is used for adding float array data.
     *
     * @param index The index at which to add the data.
     * @param data  The float array data to add.
     */
    @Override
    void add(int index, float[] data) {
        FloatBuffer buffView = this.data.asFloatBuffer();
        setPosition(buffView, index);
        buffView.put(data, 0, getStride());
    }

    /**
     * Adds data to the dataset at the specified index.
     * This method is used for adding double array data.
     *
     * @param index The index at which to add the data.
     * @param data  The double array data to add.
     */
    @Override
    void add(int index, double[] data) {
        DoubleBuffer buffView = this.data.asDoubleBuffer();
        setPosition(buffView, index);
        buffView.put(data, 0, getStride());
    }

    /**
     * Adds weighted data to the dataset at the specified index.
     * This method is not supported for dense dataset.
     *
     * @param index   The index at which to add the data.
     * @param data    The character array data to add.
     * @param weights The weights associated with the data.
     * @throws UnsupportedOperationException Always thrown as this method is not supported for dense dataset.
     */
    @Override
    void addWeighted(int index, char[] data, float[] weights) {
        throw new UnsupportedOperationException("addWeighted not supported for dense dataset, use add");
    }

    /**
     * Adds weighted data to the dataset at the specified index.
     * This method is not supported for dense dataset.
     *
     * @param index   The index at which to add the data.
     * @param data    The integer array data to add.
     * @param weights The weights associated with the data.
     * @throws UnsupportedOperationException Always thrown as this method is not supported for dense dataset.
     */
    @Override
    void addWeighted(int index, int[] data, float[] weights) {
        throw new UnsupportedOperationException("addWeighted not supported for dense dataset, use add");
    }

    /**
     * Adds weighted data to the dataset at the specified index.
     * This method is not supported for dense dataset.
     *
     * @param index   The index at which to add the data.
     * @param data    The float array data to add.
     * @param weights The weights associated with the data.
     * @throws UnsupportedOperationException Always thrown as this method is not supported for dense dataset.
     */
    @Override
    void addWeighted(int index, float[] data, float[] weights) {
        throw new UnsupportedOperationException("addWeighted not supported for dense dataset, use add");
    }

    /**
     * Adds weighted data to the dataset at the specified index.
     * This method is not supported for dense dataset.
     *
     * @param index   The index at which to add the data.
     * @param data    The double array data to add.
     * @param weights The weights associated with the data.
     * @throws UnsupportedOperationException Always thrown as this method is not supported for dense dataset.
     */
    @Override
    void addWeighted(int index, double[] data, float[] weights) {
        throw new UnsupportedOperationException("addWeighted not supported for dense dataset, use add");
    }

    /**
     * Adds sparse data to the dataset at the specified index.
     * This method is not supported for dense dataset.
     *
     * @param index       The index at which to add the data.
     * @param sparseIndex The sparse indices associated with the data.
     * @param data        The character array data to add.
     * @throws UnsupportedOperationException Always thrown as this method is not supported for dense dataset.
     */
    @Override
    void addSparse(int index, long[] sparseIndex, char[] data) {
        throw new UnsupportedOperationException("addSparse not supported for dense dataset, use add");
    }

    /**
     * Adds sparse data to the dataset at the specified index.
     * This method is not supported for dense dataset.
     *
     * @param index       The index at which to add the data.
     * @param sparseIndex The sparse indices associated with the data.
     * @param data        The integer array data to add.
     * @throws UnsupportedOperationException Always thrown as this method is not supported for dense dataset.
     */
    @Override
    void addSparse(int index, long[] sparseIndex, int[] data) {
        throw new UnsupportedOperationException("addSparse not supported for dense dataset, use add");
    }

    /**
     * Adds sparse data to the dataset at the specified index.
     * This method is not supported for dense dataset.
     *
     * @param index       The index at which to add the data.
     * @param sparseIndex The sparse indices associated with the data.
     * @param data        The float array data to add.
     * @throws UnsupportedOperationException Always thrown as this method is not supported for dense dataset.
     */
    @Override
    void addSparse(int index, long[] sparseIndex, float[] data) {
        throw new UnsupportedOperationException("addSparse not supported for dense dataset, use add");
    }

    /**
     * Adds sparse data to the dataset at the specified index.
     * This method is not supported for dense dataset.
     *
     * @param index       The index at which to add the data.
     * @param sparseIndex The sparse indices associated with the data.
     * @param data        The double array data to add.
     * @throws UnsupportedOperationException Always thrown as this method is not supported for dense dataset.
     */
    @Override
    void addSparse(int index, long[] sparseIndex, double[] data) {
        throw new UnsupportedOperationException("addSparse not supported for dense dataset, use add");
    }

    /**
     * Adds weighted sparse data to the dataset at the specified index.
     * This method is not supported for dense dataset.
     *
     * @param index       The index at which to add the data.
     * @param sparseIndex The sparse indices associated with the data.
     * @param weights     The weights associated with the data.
     * @param data        The character array data to add.
     * @throws UnsupportedOperationException Always thrown as this method is not supported for dense dataset.
     */
    @Override
    void addSparseWeighted(int index, long[] sparseIndex, float[] weights, char[] data) {
        throw new UnsupportedOperationException("addSparseWeighted not supported for dense dataset, use add");
    }

    /**
     * Adds weighted sparse data to the dataset at the specified index.
     * This method is not supported for dense dataset.
     *
     * @param index       The index at which to add the data.
     * @param sparseIndex The sparse indices associated with the data.
     * @param weights     The weights associated with the data.
     * @param data        The integer array data to add.
     * @throws UnsupportedOperationException Always thrown as this method is not supported for dense dataset.
     */
    @Override
    void addSparseWeighted(int index, long[] sparseIndex, float[] weights, int[] data) {
        throw new UnsupportedOperationException("addSparseWeighted not supported for dense dataset, use add");
    }

    /**
     * Adds weighted sparse data to the dataset at the specified index.
     * This method is not supported for dense dataset.
     *
     * @param index       The index at which to add the data.
     * @param sparseIndex The sparse indices associated with the data.
     * @param weights     The weights associated with the data.
     * @param data        The float array data to add.
     * @throws UnsupportedOperationException Always thrown as this method is not supported for dense dataset.
     */
    @Override
    void addSparseWeighted(int index, long[] sparseIndex, float[] weights, float[] data) {
        throw new UnsupportedOperationException("addSparseWeighted not supported for dense dataset, use add");
    }

    /**
     * Adds weighted sparse data to the dataset at the specified index.
     * This method is not supported for dense dataset.
     *
     * @param index       The index at which to add the data.
     * @param sparseIndex The sparse indices associated with the data.
     * @param weights     The weights associated with the data.
     * @param data        The double array data to add.
     * @throws UnsupportedOperationException Always thrown as this method is not supported for dense dataset.
     */
    @Override
    void addSparseWeighted(int index, long[] sparseIndex, float[] weights, double[] data) {
        throw new UnsupportedOperationException("addSparseWeighted not supported for dense dataset, use add");
    }

    /**
     * The empty long array used for sparse operations in dense dataset.
     */
    private static final long[] EMPTY_LONG_ARRAY = new long[0];

    /**
     * The empty float array used for weighted operations in dense dataset.
     */
    private static final float[] EMPTY_FLOAT_ARRAY = new float[0];
}
