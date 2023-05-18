#include <jni.h>
#include <span>

#ifndef _Included_com_system_tensorhub_tensorhub
#define _Included_com_system_tensorhub_tensorhub
#ifdef __cplusplus
extern "C" {
#endif
#undef com_system_tensorhub_tensorhub_NULLPTR
#define com_system_tensorhub_tensorhub_NULLPTR 0LL

JNIEXPORT jlong JNICALL Java_com_system_tensorhub_tensorhub_load
  (JNIEnv* env, jclass, std::string_view file_path, jint, jint);

JNIEXPORT void JNICALL Java_com_system_tensorhub_tensorhub_load_1datasets
  (JNIEnv* env, jclass, jlong, std::span<jobject> datasets);

JNIEXPORT void JNICALL Java_com_system_tensorhub_tensorhub_shutdown
  (JNIEnv*, jclass, jlong);

JNIEXPORT jobject JNICALL Java_com_system_tensorhub_tensorhub_get_1layers
  (JNIEnv*, jclass, jlong, jint);

JNIEXPORT void JNICALL Java_com_system_tensorhub_tensorhub_predict
  (JNIEnv* env, jclass, jlong, jint, std::span<jobject> inputs, std::span<jobject> outputs);

#ifdef __cplusplus
}
#endif
#endif
