package com.system.tensorhub;

/**
 * Represents a layer in the TensorHub system.
 */
data class Layer(
    /**
     * The name of the layer.
     */
    val name: String,
    /**
     * The name of the dataset associated with the layer.
     */
    val datasetName: String,
    /**
     * The kind of layer.
     */
    val kind: Kind,
    /**
     * The attributes of the layer.
     */
    val attributes: Int,
    /**
     * The total number of dimensions of the layer.
     */
    val dimensions: Int,
    /**
     * The size of the X dimension.
     */
    val dimX: Int,
    /**
     * The size of the Y dimension.
     */
    val dimY: Int,
    /**
     * The size of the Z dimension.
     */
    val dimZ: Int
) {
    /**
     * Enumeration representing the kind of layer.
     */
    enum class Kind {
        Input, Hidden, Output, Target
    }

    companion object Attribute {
        /**
         * No attributes.
         */
        const val None = 0x0
        /**
         * Sparse attribute.
         */
        const val Sparse = 0x1
        /**
         * Denoising attribute.
         */
        const val Denoising = 0x2
        /**
         * Batch normalization attribute.
         */
        const val BatchNormalization = 0x4
    }

    /**
     * Secondary constructor for creating a Layer instance with integer values for kind and attributes.
     *
     * @param name The name of the layer.
     * @param datasetName The name of the dataset associated with the layer.
     * @param kind The kind of layer as an integer value.
     * @param attributes The attributes of the layer as an integer value.
     * @param dimensions The total number of dimensions of the layer.
     * @param dimX The size of the X dimension.
     * @param dimY The size of the Y dimension.
     * @param dimZ The size of the Z dimension.
     */
    constructor(
        name: String,
        datasetName: String,
        kind: Int,
        attributes: Int,
        dimensions: Int,
        dimX: Int,
        dimY: Int,
        dimZ: Int
    ) : this(
        name,
        datasetName,
        Kind.values()[kind],
        attributes,
        dimensions,
        dimX,
        dimY,
        dimZ
    )

    /**
     * Retrieves the dimensions of the layer.
     *
     * @return The dimensions of the layer.
     */
    fun getDim(): Dim {
        return Dim(dimensions, dimX, dimY, dimZ, 0)
    }
}

/**
 * Represents the dimensions of a layer.
 */
data class Dim(
    /**
     * The total number of dimensions.
     */
    val dimensions: Int,
    /**
     * The size of the X dimension.
     */
    val dimX: Int,
    /**
     * The size of the Y dimension.
     */
    val dimY: Int,
    /**
     * The size of the Z dimension.
     */
    val dimZ: Int,
    /**
     * An unknown value.
     */
    val unknown: Int
)
