#include <jni.h>
#include <span>

#ifndef _Included_com_system_tensorhub_tensorhub
#define _Included_com_system_tensorhub_tensorhub
#ifdef __cplusplus
extern "C" {
#endif
#undef com_system_tensorhub_tensorhub_NULLPTR
#define com_system_tensorhub_tensorhub_NULLPTR 0LL

/**
 * @brief Loads the tensorhub model from the specified file.
 *
 * @param env JNI environment pointer.
 * @param clazz Java class that the method belongs to.
 * @param file_path The file path of the tensorhub model.
 * @param height The height of the inputs to the model.
 * @param width The width of the inputs to the model.
 *
 * @return The pointer to the loaded tensorhub model.
 */
JNIEXPORT jlong JNICALL Java_com_system_tensorhub_tensorhub_load
  (JNIEnv *env, jclass, std::string_view file_path, jint, jint);

/**
 * @brief Loads the datasets into the tensorhub model.
 *
 * @param env JNI environment pointer.
 * @param clazz Java class that the method belongs to.
 * @param ptr The pointer to the tensorhub model.
 * @param datasets The datasets to be loaded into the model.
 */
JNIEXPORT void JNICALL Java_com_system_tensorhub_tensorhub_load_1datasets
  (JNIEnv *env, jclass, jlong, std::span<jobject> datasets);

/**
 * @brief Shuts down the tensorhub model.
 *
 * @param env JNI environment pointer.
 * @param clazz Java class that the method belongs to.
 * @param ptr The pointer to the tensorhub model.
 */
JNIEXPORT void JNICALL Java_com_system_tensorhub_tensorhub_shutdown
  (JNIEnv *, jclass, jlong);

/**
 * @brief Gets the layers of the tensorhub model.
 *
 * @param env JNI environment pointer.
 * @param clazz Java class that the method belongs to.
 * @param ptr The pointer to the tensorhub model.
 * @param layer_index The index of the layer to retrieve.
 *
 * @return The layer of the tensorhub model.
 */
JNIEXPORT jobject JNICALL Java_com_system_tensorhub_tensorhub_get_1layers
  (JNIEnv *, jclass, jlong, jint);

/**
 * @param inputs The inputs to the tensorhub model.
 * @param outputs The buffer to store the outputs of the model.
 */
JNIEXPORT void JNICALL Java_com_system_tensorhub_tensorhub_predict
  (JNIEnv *env, jclass, jlong, jint, std::span<jobject> inputs, std::span<jobject> outputs);

#ifdef __cplusplus
}
#endif
#endif
