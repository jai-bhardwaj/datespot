#ifndef INCLUDED_COM_SYSTEM_TENSORHUB_KNN_KNEARESTNEIGHBORSCUDA
#define INCLUDED_COM_SYSTEM_TENSORHUB_KNN_KNEARESTNEIGHBORSCUDA

#include <jni.h>
#include <array>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @typedef KNearestNeighborsCudaPtr
 * @brief Alias for the pointer to KNearestNeighborsCuda.
 */
using KNearestNeighborsCudaPtr = jlong;

/**
 * @typedef FloatArray
 * @brief Alias for the jfloatArray type.
 */
using FloatArray = jfloatArray;

/**
 * @typedef IntArray
 * @brief Alias for the jintArray type.
 */
using IntArray = jintArray;

/**
 * @typedef Char
 * @brief Alias for the jchar type.
 */
using Char = jchar;

/**
 * @typedef StringArray
 * @brief Alias for the jobjectArray type.
 */
using StringArray = jobjectArray;

/**
 * @brief Initializes the KNearestNeighborsCuda instance.
 * @param env The JNIEnv pointer.
 * @param clazz The Java class associated with the native method.
 * @param arg1 The first integer argument.
 * @param arg2 The second integer argument.
 * @param arg3 The third integer argument.
 * @param arg4 The fourth integer argument.
 * @return A pointer to the initialized KNearestNeighborsCuda instance.
 */
JNIEXPORT KNearestNeighborsCudaPtr JNICALL Java_com_system_tensorhub_knn_KNearestNeighborsCuda_initialize
  (JNIEnv* env, jclass clazz, jint arg1, jint arg2, jint arg3, jint arg4);

/**
 * @brief Loads data into the KNearestNeighborsCuda instance.
 * @param env The JNIEnv pointer.
 * @param clazz The Java class associated with the native method.
 * @param arg1 The array of string objects.
 * @param arg2 The integer array.
 * @param arg3 The first character argument.
 * @param arg4 The second character argument.
 * @param ptr A pointer to the KNearestNeighborsCuda instance.
 */
JNIEXPORT void JNICALL Java_com_system_tensorhub_knn_KNearestNeighborsCuda_load
  (JNIEnv* env, jclass clazz, StringArray arg1, IntArray arg2, Char arg3, Char arg4, KNearestNeighborsCudaPtr ptr);

/**
 * @brief Shuts down the KNearestNeighborsCuda instance.
 * @param env The JNIEnv pointer.
 * @param clazz The Java class associated with the native method.
 * @param ptr A pointer to the KNearestNeighborsCuda instance.
 */
JNIEXPORT void JNICALL Java_com_system_tensorhub_knn_KNearestNeighborsCuda_shutdown
  (JNIEnv* env, jclass clazz, KNearestNeighborsCudaPtr ptr);

/**
 * @brief Finds k-nearest neighbors using the KNearestNeighborsCuda instance.
 * @param env The JNIEnv pointer.
 * @param clazz The Java class associated with the native method.
 * @param arg1 The first integer argument.
 * @param arg2 The float array.
 * @param arg3 The second integer argument.
 * @param arg4 The third integer argument.
 * @param arg5 The second float array.
 * @param arg6 The array of string objects.
 * @param ptr A pointer to the KNearestNeighborsCuda instance.
 */
JNIEXPORT void JNICALL Java_com_system_tensorhub_knn_KNearestNeighborsCuda_findKnn__I_3FII_3F_3Ljava_lang_String_2J
  (JNIEnv* env, jclass clazz, jint arg1, FloatArray arg2, jint arg3, jint arg4, FloatArray arg5, StringArray arg6, KNearestNeighborsCudaPtr ptr);

/**
 * @brief Finds k-nearest neighbors using the KNearestNeighborsCuda instance.
 * @param env The JNIEnv pointer.
 * @param clazz The Java class associated with the native method.
 * @param arg1 The first integer argument.
 * @param arg2 The float array.
 * @param arg3 The second integer argument.
 * @param arg4 The third integer argument.
 * @param ptr A pointer to the KNearestNeighborsCuda instance.
 * @return The result as a Java object.
 */
JNIEXPORT jobject JNICALL Java_com_system_tensorhub_knn_KNearestNeighborsCuda_findKnn__I_3FIIJ
  (JNIEnv* env, jclass clazz, jint arg1, FloatArray arg2, jint arg3, jint arg4, KNearestNeighborsCudaPtr ptr);

#ifdef __cplusplus
}
#endif

#endif
