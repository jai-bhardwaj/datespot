package com.system.tensorhub

import lombok.Value

/**
 * Represents a layer in a neural network.
 *
 * @property name The name of the layer.
 * @property datasetName The name of the associated dataset.
 * @property kind The kind of layer.
 * @property attributes The attributes of the layer.
 * @property dimensions The number of dimensions in the layer.
 * @property dimX The size of the layer in the X dimension.
 * @property dimY The size of the layer in the Y dimension.
 * @property dimZ The size of the layer in the Z dimension.
 */
@Value
class Layer(
    val name: String,
    val datasetName: String,
    val kind: Kind,
    val attributes: Int,
    val dimensions: Int,
    val dimX: Int,
    val dimY: Int,
    val dimZ: Int
) {
    /**
     * The kind of layer.
     */
    enum class Kind {
        Input, Hidden, Output, Target;
    }

    companion object {
        const val AttributeNone = 0x0
        const val AttributeSparse = 0x1
        const val AttributeDenoising = 0x2
        const val AttributeBatchNormalization = 0x4
    }

    /**
     * Gets the dimension of the layer.
     *
     * @return The dimension of the layer.
     */
    fun getDim(): Dim {
        return Dim(dimensions, dimX, dimY, dimZ, 0)
    }
}
