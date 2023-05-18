package com.system.tensorhub

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
    val attributes: Set<Attribute>,
    /**
     * The dimensions of the layer.
     */
    val dimensions: Dim
) {
    /**
     * Enumeration representing the kind of layer.
     */
    enum class Kind {
        Input, Hidden, Output, Target
    }

    /**
     * Enumeration representing the attributes of a layer.
     */
    enum class Attribute {
        Sparse, Denoising, BatchNormalization
    }

    /**
     * Secondary constructor for creating a Layer instance with integer values for kind and attributes.
     *
     * @param name The name of the layer.
     * @param datasetName The name of the dataset associated with the layer.
     * @param kind The kind of layer as an integer value.
     * @param attributes The attributes of the layer as an integer value.
     * @param dimensions The dimensions of the layer.
     */
    constructor(
        name: String,
        datasetName: String,
        kind: Int,
        attributes: Int,
        dimensions: Dim
    ) : this(
        name,
        datasetName,
        Kind.values()[kind],
        parseAttributes(attributes),
        dimensions
    )

    companion object {
        /**
         * Parses the integer representation of attributes and returns a set of Attribute values.
         *
         * @param attributes The attributes as an integer value.
         * @return The set of Attribute values.
         */
        private fun parseAttributes(attributes: Int): Set<Attribute> {
            val attributeSet = mutableSetOf<Attribute>()
            if (attributes and Attribute.Sparse.ordinal != 0) {
                attributeSet.add(Attribute.Sparse)
            }
            if (attributes and Attribute.Denoising.ordinal != 0) {
                attributeSet.add(Attribute.Denoising)
            }
            if (attributes and Attribute.BatchNormalization.ordinal != 0) {
                attributeSet.add(Attribute.BatchNormalization)
            }
            return attributeSet
        }
    }

    /**
     * Retrieves the dimensions of the layer.
     *
     * @return The dimensions of the layer.
     */
    fun getDim(): Dim = dimensions
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
    val unknown: Int = 0
)
