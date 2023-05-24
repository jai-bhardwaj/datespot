package com.system.tensorhub

/**
 * Represents the dimensions of a tensor.
 *
 * @property dimensions The number of dimensions.
 * @property x The size of the tensor in the X dimension.
 * @property y The size of the tensor in the Y dimension.
 * @property z The size of the tensor in the Z dimension.
 * @property examples The number of examples in the tensor.
 * @property stride The stride of the tensor.
 */
data class Dim(
    val dimensions: Int,
    val x: Int,
    val y: Int,
    val z: Int,
    val examples: Int,
    val stride: Int = x * y * z
) {
    companion object {
        /**
         * Creates a 1-dimensional tensor dimension.
         *
         * @param x The size of the tensor in the X dimension.
         * @param examples The number of examples in the tensor.
         * @return The 1-dimensional tensor dimension.
         */
        fun _1d(x: Int, examples: Int): Dim {
            return Dim(1, x, 1, 1, examples)
        }

        /**
         * Creates a 2-dimensional tensor dimension.
         *
         * @param x The size of the tensor in the X dimension.
         * @param y The size of the tensor in the Y dimension.
         * @param examples The number of examples in the tensor.
         * @return The 2-dimensional tensor dimension.
         */
        fun _2d(x: Int, y: Int, examples: Int): Dim {
            return Dim(2, x, y, 1, examples)
        }

        /**
         * Creates a 3-dimensional tensor dimension.
         *
         * @param x The size of the tensor in the X dimension.
         * @param y The size of the tensor in the Y dimension.
         * @param z The size of the tensor in the Z dimension.
         * @param examples The number of examples in the tensor.
         * @return The 3-dimensional tensor dimension.
         */
        fun _3d(x: Int, y: Int, z: Int, examples: Int): Dim {
            return Dim(3, x, y, z, examples)
        }
    }
}
