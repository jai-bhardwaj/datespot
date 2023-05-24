#include <jni.h>

#ifndef INCLUDED_COM_SYSTEM_TENSORHUB_TENSORHUB
#define INCLUDED_COM_SYSTEM_TENSORHUB_TENSORHUB

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief A constant representing a null pointer for the Tensorhub class.
 */
constexpr jlong com_system_tensorhub_tensorhub_NULLPTR = 0LL;

/**
 * @brief Loads the Tensorhub.
 * 
 * This function loads the Tensorhub with the specified parameters.
 * 
 * @param env The JNI environment.
 * @param clazz The Java class object.
 * @param str The string parameter.
 * @param param1 The first integer parameter.
 * @param param2 The second integer parameter.
 * @return The loaded Tensorhub as a jlong.
 */
JNIEXPORT auto JNICALL Java_com_system_tensorhub_tensorhub_load(JNIEnv* env, jclass clazz, jstring str, jint param1, jint param2) -> jlong;

/**
 * @brief Loads the datasets for the Tensorhub.
 * 
 * This function loads the datasets for the Tensorhub object.
 * 
 * @param env The JNI environment.
 * @param clazz The Java class object.
 * @param tensorhub The jlong representing the Tensorhub object.
 * @param datasets The array of dataset objects.
 */
JNIEXPORT void JNICALL Java_com_system_tensorhub_tensorhub_load_1datasets(JNIEnv* env, jclass clazz, jlong tensorhub, jobjectArray datasets);

/**
 * @brief Shuts down the Tensorhub.
 * 
 * This function shuts down the Tensorhub object.
 * 
 * @param env The JNI environment.
 * @param clazz The Java class object.
 * @param tensorhub The jlong representing the Tensorhub object.
 */
JNIEXPORT void JNICALL Java_com_system_tensorhub_tensorhub_shutdown(JNIEnv* env, jclass clazz, jlong tensorhub);

/**
 * @brief Retrieves the layers of the Tensorhub.
 * 
 * This function retrieves the layers of the Tensorhub object.
 * 
 * @param env The JNI environment.
 * @param clazz The Java class object.
 * @param tensorhub The jlong representing the Tensorhub object.
 * @param param The integer parameter.
 * @return The layers of the Tensorhub as a jobject.
 */
JNIEXPORT auto JNICALL Java_com_system_tensorhub_tensorhub_get_1layers(JNIEnv* env, jclass clazz, jlong tensorhub, jint param) -> jobject;

/**
 * @brief Performs prediction using the Tensorhub.
 * 
 * This function performs prediction using the Tensorhub object.
 * 
 * @param env The JNI environment.
 * @param clazz The Java class object.
 * @param tensorhub The jlong representing the Tensorhub object.
 * @param param1 The first integer parameter.
 * @param param2 The second integer parameter.
 * @param inputArray The input array of objects.
 * @param outputArray The output array of objects.
 */
JNIEXPORT void JNICALL Java_com_system_tensorhub_tensorhub_predict(JNIEnv* env, jclass clazz, jlong tensorhub, jint param1, jobjectArray inputArray, jobjectArray outputArray);

#ifdef __cplusplus
}
#endif

#endif
