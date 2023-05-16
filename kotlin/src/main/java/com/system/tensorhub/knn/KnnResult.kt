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
        val slen = scores.size
        val ilen = keys.size
        if (slen != ilen) {
            throw IllegalArgumentException("scores and indexes have different lengths (scores: $slen, indexes: $ilen)")
        }
        if (slen % k != 0) {
            throw IllegalArgumentException("k: $k must divide the length of data: $slen")
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
     * @param i The index.
     * @return The key.
     */
    fun getKeyAt(row: Int, i: Int): String {
        return keys[row * k + i]
    }

    /**
     * Returns the score at the specified row and index.
     *
     * @param row The row index.
     * @param i The index.
     * @return The score.
     */
    fun getScoreAt(row: Int, i: Int): Float {
        return scores[row * k + i]
    }
}
