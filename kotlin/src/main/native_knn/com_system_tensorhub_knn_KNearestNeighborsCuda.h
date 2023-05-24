#include <jni.h>

#ifndef INCLUDED_COM_SYSTEM_TENSORHUB_KNN_KNEARESTNEIGHBORSCUDA
#define INCLUDED_COM_SYSTEM_TENSORHUB_KNN_KNEARESTNEIGHBORSCUDA

extern "C" {

/**
 * \defgroup KNearestNeighborsCuda KNearestNeighborsCuda
 * @{
 */

/// Constant representing a null pointer.
constexpr jlong com_system_tensorhub_knn_KNearestNeighborsCuda_NULLPTR = 0LL;

/// Default key-value delimiter constant.
constexpr jlong com_system_tensorhub_knn_KNearestNeighborsCuda_DEFAULT_KEYVAL_DELIM = 9L;

/// Default vector delimiter constant.
constexpr jlong com_system_tensorhub_knn_KNearestNeighborsCuda_DEFAULT_VEC_DELIM = 32L;

/**
 * \brief Initializes the KNearestNeighborsCuda module.
 *
 * \param env The JNI environment.
 * \param clazz The Java class object.
 * \param param1 The first parameter.
 * \param param2 The second parameter.
 * \param param3 The third parameter.
 * \param param4 The fourth parameter.
 * \return The initialized module as a jlong.
 */
JNIEXPORT jlong JNICALL Java_com_system_tensorhub_knn_KNearestNeighborsCuda_initialize(
    JNIEnv* env,
    jclass clazz,
    jint param1,
    jint param2,
    jint param3,
    jint param4
);

/**
 * \brief Loads data into the KNearestNeighborsCuda module.
 *
 * \param env The JNI environment.
 * \param clazz The Java class object.
 * \param data The data array.
 * \param dataSizes The data sizes array.
 * \param param1 The first parameter.
 * \param param2 The second parameter.
 * \param param3 The third parameter.
 * \return void
 */
JNIEXPORT void JNICALL Java_com_system_tensorhub_knn_KNearestNeighborsCuda_load(
    JNIEnv* env,
    jclass clazz,
    jobjectArray data,
    jintArray dataSizes,
    jchar param1,
    jchar param2,
    jlong param3
);

/**
 * \brief Shuts down the KNearestNeighborsCuda module.
 *
 * \param env The JNI environment.
 * \param clazz The Java class object.
 * \param module The module as a jlong.
 * \return void
 */
JNIEXPORT void JNICALL Java_com_system_tensorhub_knn_KNearestNeighborsCuda_shutdown(
    JNIEnv* env,
    jclass clazz,
    jlong module
);

/**
 * \brief Finds the k-nearest neighbors using the KNearestNeighborsCuda module.
 *
 * \param env The JNI environment.
 * \param clazz The Java class object.
 * \param param1 The first parameter.
 * \param param2 The second parameter.
 * \param param3 The third parameter.
 * \param param4 The fourth parameter.
 * \param param5 The fifth parameter.
 * \param param6 The sixth parameter.
 * \param module The module as a jlong.
 * \return The k-nearest neighbors as a jobject.
 */
JNIEXPORT void JNICALL Java_com_system_tensorhub_knn_KNearestNeighborsCuda_findKnn__I_3FII_3F_3Ljava_lang_String_2J(
    JNIEnv* env,
    jclass clazz,
    jint param1,
    jfloatArray param2,
    jint param3,
    jint param4,
    jfloatArray param5,
    jobjectArray param6,
    jlong module
);

/**
 * \brief Finds the k-nearest neighbors using the KNearestNeighborsCuda module.
 *
 * \param env The JNI environment.
 * \param clazz The Java class object.
 * \param param1 The first parameter.
 * \param param2 The second parameter.
 * \param param3 The third parameter.
 * \param param4 The fourth parameter.
 * \param module The module as a jlong.
 * \return The k-nearest neighbors as a jobject.
 */
JNIEXPORT jobject JNICALL Java_com_system_tensorhub_knn_KNearestNeighborsCuda_findKnn__I_3FIIJ(
    JNIEnv* env,
    jclass clazz,
    jint param1,
    jfloatArray param2,
    jint param3,
    jint param4,
    jlong module
);

/**
 * @}
 */

} // extern "C"

#endif // INCLUDED_COM_SYSTEM_TENSORHUB_KNN_KNEARESTNEIGHBORSCUDA
