#include <jni.h>

#ifndef _Included_com_system_tensorhub_knn_KNearestNeighborsCuda
#define _Included_com_system_tensorhub_knn_KNearestNeighborsCuda
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Constant representing the value of a null pointer.
 */
#undef com_system_tensorhub_knn_KNearestNeighborsCuda_NULLPTR
#define com_system_tensorhub_knn_KNearestNeighborsCuda_NULLPTR 0LL

/**
 * @brief Constant representing the default delimiter for key-value pairs in the input data.
 */
#undef com_system_tensorhub_knn_KNearestNeighborsCuda_DEFAULT_KEYVAL_DELIM
#define com_system_tensorhub_knn_KNearestNeighborsCuda_DEFAULT_KEYVAL_DELIM 9L

/**
 * @brief Constant representing the default delimiter for vector elements in the input data.
 */
#undef com_system_tensorhub_knn_KNearestNeighborsCuda_DEFAULT_VEC_DELIM
#define com_system_tensorhub_knn_KNearestNeighborsCuda_DEFAULT_VEC_DELIM 32L

/**
 * @brief JNI function to initialize the KNearestNeighborsCuda instance.
 *
 * @param env Pointer to the JNI environment.
 * @param clazz Reference to the calling class.
 * @param rows Number of rows in the input data.
 * @param cols Number of columns in the input data.
 * @param k Number of nearest neighbors to be found.
 * @param num_threads Number of threads to be used in the computation.
 * @return Handle to the KNearestNeighborsCuda instance.
 */
JNIEXPORT jlong JNICALL Java_com_system_tensorhub_knn_KNearestNeighborsCuda_initialize
  (JNIEnv *env, jclass clazz, jint rows, jint cols, jint k, jint num_threads);

/**
 * @brief JNI function to load the input data into the KNearestNeighborsCuda instance.
 *
 * @param env Pointer to the JNI environment.
 * @param clazz Reference to the calling class.
 * @param data Object array holding the input data.
 * @param labels Integer array holding the labels for the input data.
 * @param keyval_delim Delimiter for key-value pairs in the input data.
 * @param vec_delim Delimiter for vector elements in the input data.
 * @param handle Handle to the KNearestNeighborsCuda instance.
 */
JNIEXPORT void JNICALL Java_com_system_tensorhub_knn_KNearestNeighborsCuda_load
  (JNIEnv *env, jclass clazz, jobjectArray data, jintArray labels, jchar keyval_delim, jchar vec_delim, jlong handle);

/**
 * @brief JNI function to shut down the KNearestNeighborsCuda instance.
 *
 * @param env Pointer to the JNI environment.
 * @param clazz Reference to the calling class.
 * @param handle Handle to the KNearestNeighborsCuda instance.
 */
JNIEXPORT void JNICALL Java_com_system_tensorhub_knn_KNearestNeighborsCuda_shutdown
  (JNIEnv *env, jclass clazz, jlong handle);

/**
 * @brief JNI function to find the k nearest neighbors for a query vector.
 *
 * @param env Pointer to the JNI environment.
 * @param clazz Reference to the calling class.
 * @param k Number of nearest neighbors to be found.
 * @param query_vec Query vector.
 * @param query_len Length of the query vector.
 * @param num_results Number of results to be returned.
 * @param distances Array to store the distances of the nearest neighbors.
 * @param labels Array to store the labels of the nearest neighbors.
 * @param handle Handle to the KNearestNeighborsCuda instance.
 */
JNIEXPORT void JNICALL Java_com_system_tensorhub_knn_KNearestNeighborsCuda_findKnn__I_3FII_3F_3Ljava_lang_String_2J
  (JNIEnv *env, jclass clazz, jint k, jfloatArray query_vec, jint query_len, jint num_results, jfloatArray distances, jobjectArray labels, jlong handle);

/**
 * @brief JNI function to find the k nearest neighbors for a query vector and return the results as a KnnResult object.
 *
 * @param env Pointer to the JNI environment.
 * @param clazz Reference to the calling class.
 * @param k Number of nearest neighbors to be found.
 * @param query_vec Query vector.
 * @param query_len Length of the query vector.
 * @param num_results Number of results to be returned.
 * @param handle Handle to the KNearestNeighborsCuda instance.
 * @return KnnResult object holding the results of the nearest neighbor search.
 */
JNIEXPORT jobject JNICALL Java_com_system_tensorhub_knn_KNearestNeighborsCuda_findKnn__I_3FIIJ
  (JNIEnv *env, jclass clazz, jint k, jfloatArray query_vec, jint query_len, jint num_results, jlong handle);

#ifdef __cplusplus
}
#endif
#endif
