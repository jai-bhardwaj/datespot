package com.system.tensorhub.knn

/**
 * Represents the result of a k-nearest neighbors (KNN) algorithm.
 *
 * @property keys The array of keys corresponding to the nearest neighbors.
 * @property scores The array of scores representing the distances or similarities of the nearest neighbors.
 * @property k The value of k used in the KNN algorithm.
 */
class KnnResult(private val keys: Array<String>, private val scores: FloatArray, private val k: Int) {

    init {
        require(scores.size == keys.size) { "Scores and keys have different lengths (scores: ${scores.size}, keys: ${keys.size})" }
        require(scores.size % k == 0) { "k: $k must divide the length of data: ${scores.size}" }
    }

    /**
     * Returns the batch size of the KNN result.
     */
    val batchSize: Int
        get() = scores.size / k

    /**
     * Returns the key at the specified row and index.
     *
     * @param rowIndex The row index.
     * @param index The index.
     * @return The key.
     */
    fun getKeyAt(rowIndex: Int, index: Int): String {
        return keys[rowIndex * k + index]
    }

    /**
     * Returns the score at the specified row and index.
     *
     * @param rowIndex The row index.
     * @param index The index.
     * @return The score.
     */
    fun getScoreAt(rowIndex: Int, index: Int): Float {
        return scores[rowIndex * k + index]
    }
}
