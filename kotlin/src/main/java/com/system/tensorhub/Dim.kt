package com.system.tensorhub

/**
 * Represents the dimensions of a tensor.
 * @property dimensions The number of dimensions.
 * @property x The size of the first dimension.
 * @property y The size of the second dimension.
 * @property z The size of the third dimension.
 * @property examples The number of examples.
 * @property stride The stride calculated based on the dimensions.
 */
data class Dim(
    val dimensions: Int,
    val x: Int,
    val y: Int,
    val z: Int,
    val examples: Int
) {
    /**
     * The stride calculated based on the dimensions.
     */
    val stride: Int = x * y * z

    /**
     * Constructs a `Dim` object with the specified dimensions, sizes, and number of examples.
     * @param dimensions The number of dimensions.
     * @param x The size of the first dimension.
     * @param y The size of the second dimension.
     * @param z The size of the third dimension.
     * @param examples The number of examples.
     */
    constructor(dimensions: Int, x: Int, y: Int, z: Int, examples: Int) : this(
        dimensions,
        x,
        y,
        z,
        examples
    )

    /**
     * Constructs a `Dim` object based on an existing `Dim` object and the number of examples.
     * @param dim The existing `Dim` object.
     * @param examples The number of examples.
     */
    constructor(dim: Dim, examples: Int) : this(
        dim.dimensions,
        dim.x,
        dim.y,
        dim.z,
        examples
    )

    companion object {
        /**
         * Creates a 1-dimensional `Dim` object with the specified size and number of examples.
         * @param x The size of the first dimension.
         * @param examples The number of examples.
         * @return The created 1-dimensional `Dim` object.
         */
        fun _1d(x: Int, examples: Int): Dim = Dim(1, x, 1, 1, examples)

        /**
         * Creates a 2-dimensional `Dim` object with the specified sizes and number of examples.
         * @param x The size of the first dimension.
         * @param y The size of the second dimension.
         * @param examples The number of examples.
         * @return The created 2-dimensional `Dim` object.
         */
        fun _2d(x: Int, y: Int, examples: Int): Dim = Dim(2, x, y, 1, examples)

        /**
         * Creates a 3-dimensional `Dim` object with the specified sizes and number of examples.
         * @param x The size of the first dimension.
         * @param y The size of the second dimension.
         * @param z The size of the third dimension.
         * @param examples The number of examples.
         * @return The created 3-dimensional `Dim` object.
         */
        fun _3d(x: Int, y: Int, z: Int, examples: Int): Dim = Dim(3, x, y, z, examples)
    }
}
