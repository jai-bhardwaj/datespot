#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

using jlong = long long;
using jint = int;
using jchar = char;
using jfloatArray = float*;
using jintArray = int*;
using jobjectArray = jobject*;
using jobject = void*;

/**
 * @brief Initializes the KNearestNeighborsCuda instance.
 *
 * @param env The JNI environment.
 * @param clazz The Java class.
 * @param rows The number of rows in the data.
 * @param cols The number of columns in the data.
 * @param k The number of nearest neighbors to find.
 * @param num_threads The number of threads to use.
 * @return The handle to the initialized instance.
 */
inline jlong Java_com_system_tensorhub_knn_KNearestNeighborsCuda_initialize(JNIEnv* env, jclass clazz, jint rows, jint cols, jint k, jint num_threads);

/**
 * @brief Loads the data into the KNearestNeighborsCuda instance.
 *
 * @param env The JNI environment.
 * @param clazz The Java class.
 * @param data The data array.
 * @param labels The labels array.
 * @param keyval_delim The delimiter for key-value pairs.
 * @param vec_delim The delimiter for vector elements.
 * @param handle The handle to the KNearestNeighborsCuda instance.
 */
inline void Java_com_system_tensorhub_knn_KNearestNeighborsCuda_load(JNIEnv* env, jclass clazz, jobjectArray data, jintArray labels, jchar keyval_delim, jchar vec_delim, jlong handle);

/**
 * @brief Shuts down the KNearestNeighborsCuda instance.
 *
 * @param env The JNI environment.
 * @param clazz The Java class.
 * @param handle The handle to the KNearestNeighborsCuda instance.
 */
inline void Java_com_system_tensorhub_knn_KNearestNeighborsCuda_shutdown(JNIEnv* env, jclass clazz, jlong handle);

/**
 * @brief Finds the K-nearest neighbors for a given query vector and returns the distances and labels.
 *
 * @param env The JNI environment.
 * @param clazz The Java class.
 * @param k The number of nearest neighbors to find.
 * @param query_vec The query vector.
 * @param query_len The length of the query vector.
 * @param num_results The number of results to return.
 * @param distances The array to store the distances.
 * @param labels The array to store the labels.
 * @param handle The handle to the KNearestNeighborsCuda instance.
 */
inline void Java_com_system_tensorhub_knn_KNearestNeighborsCuda_findKnn__I_3FII_3F_3Ljava_lang_String_2J(JNIEnv* env, jclass clazz, jint k, jfloatArray query_vec, jint query_len, jint num_results, jfloatArray distances, jobjectArray labels, jlong handle);

/**
 * @brief Finds the K-nearest neighbors for a given query vector and returns a Java object containing the results.
 *
 * @param env The JNI environment.
 * @param clazz The Java class.
 * @param k The number of nearest neighbors to find.
 * @param query_vec The query vector.
 * @param query_len The length of the query vector.
 * @param num_results The number of results to return.
 * @param handle The handle to the KNearestNeighborsCuda instance.
 * @return The Java object containing the results.
 */
inline jobject Java_com_system_tensorhub_knn_KNearestNeighborsCuda_findKnn__I_3FIIJ(JNIEnv* env, jclass clazz, jint k, jfloatArray query_vec, jint query_len, jint num_results, jlong handle);

#ifdef __cplusplus
}
#endif
