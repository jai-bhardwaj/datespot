package com.system.tensorhub.knn

/**
 * Represents the result of a k-nearest neighbors (KNN) algorithm.
 *
 * @property keys The array of keys associated with the KNN result.
 * @property scores The array of scores corresponding to the keys.
 * @property k The value of k used in the KNN algorithm.
 */
class KnnResult(
    private val keys: Array<String>,
    private val scores: FloatArray,
    private val k: Int
) {
    init {
        require(keys.size == scores.size) {
            "Keys and scores have different lengths (keys: ${keys.size}, scores: ${scores.size})"
        }
        require(scores.size % k == 0) {
            "k: $k must divide the length of scores: ${scores.size}"
        }
    }

    /**
     * Returns the batch size of the KNN result.
     *
     * @return The batch size.
     */
    fun getBatchSize(): Int {
        return scores.size / k
    }

    /**
     * Returns the key at the specified row and index.
     *
     * @param row The row index.
     * @param i The index within the row.
     * @return The key at the specified position.
     */
    fun getKeyAt(row: Int, i: Int): String {
        return keys[row * k + i]
    }

    /**
     * Returns the score at the specified row and index.
     *
     * @param row The row index.
     * @param i The index within the row.
     * @return The score at the specified position.
     */
    fun getScoreAt(row: Int, i: Int): Float {
        return scores[row * k + i]
    }
}
