#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <memory>
#include <vector>
#include <map>

#include <jni.h>

#include "engine/knn/KnnData.h"
#include "engine/knn/DataReader.h"
#include "engine/knn/KnnExactGpu.h"
#include "com_system_tensorhub_knn_KNearestNeighborsCuda.h"

using namespace std;

// Global variables for JNI class and method references
static jclass JCLASS_KNN_RESULT = nullptr;
static jmethodID JMETHODID_KNN_RESULT_CONSTRUCTOR = nullptr;

static jclass JCLASS_STRING = nullptr;

constexpr const char* KNN_RESULT = "com/system/tensorhub/knn/KnnResult";
constexpr const char* STRING = "java/lang/String";

// Exception types
constexpr const char* ILLEGAL_ARGUMENT_EXCEPTION = "java/lang/IllegalArgumentException";
constexpr const char* CLASS_NOT_FOUND_EXCEPTION = "java/lang/ClassNotFoundException";
constexpr const char* NO_SUCH_METHOD_EXCEPTION = "java/lang/NoSuchMethodException";
constexpr const char* FILE_NOT_FOUND_EXCEPTION = "java/io/FileNotFoundException";
constexpr const char* RUNTIME_EXCEPTION = "java/lang/RuntimeException";
constexpr const char* NULL_POINTER_EXCEPTION = "java/lang/NullPointerException";

/**
 * Throws a Java exception with the given exception type and message.
 *
 * @param env The JNI environment.
 * @param exceptionType The fully qualified name of the exception class.
 * @param msg The exception message.
 */
void throw_java_exception(JNIEnv* env, const char* exceptionType, const char* msg)
{
  jclass exc = env->FindClass(exceptionType);
  if (exc == nullptr)
  {
    exc = env->FindClass(RUNTIME_EXCEPTION);
  }
  env->ThrowNew(exc, msg);
}

/**
 * Throws a Java exception with the given exception type and message.
 *
 * @param env The JNI environment.
 * @param exceptionType The fully qualified name of the exception class.
 * @param msg The exception message.
 */
void throw_java_exception(JNIEnv* env, const char* exceptionType, const string& msg)
{
  throw_java_exception(env, exceptionType, msg.c_str());
}

/**
 * Finds the JNI class with the given name.
 *
 * @param env The JNI environment.
 * @param className The fully qualified name of the class.
 * @return The JNI class reference.
 */
jclass find_class(JNIEnv* env, const char* className)
{
  jclass clazz = env->FindClass(className);
  if (clazz == nullptr)
  {
    throw_java_exception(env, CLASS_NOT_FOUND_EXCEPTION, className);
  }
  return clazz;
}

/**
 * Finds the JNI method ID for the method with the given name and descriptor in the given class.
 *
 * @param env The JNI environment.
 * @param className The fully qualified name of the class.
 * @param methodName The name of the method.
 * @param methodDescriptor The descriptor of the method.
 * @return The JNI method ID.
 */
jmethodID find_method_id(JNIEnv* env, const char* className, const char* methodName, const char* methodDescriptor)
{
  jclass clazz = find_class(env, className);
  jmethodID methodId = env->GetMethodID(clazz, methodName, methodDescriptor);
  if (methodId == nullptr)
  {
    stringstream msg;
    msg << className << "#" << methodName << methodDescriptor;
    throw_java_exception(env, NO_SUCH_METHOD_EXCEPTION, msg.str());
  }
  return methodId;
}

/**
 * JNI_OnLoad function that is called when the native library is loaded.
 *
 * @param vm The Java virtual machine.
 * @param reserved Reserved argument.
 * @return The JNI version.
 */
jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
  JNIEnv* env;
  if (vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK)
  {
    return JNI_ERR;
  }
  else
  {
    // Find and cache the JNI class and method references
    jclass localKnnResultClass = find_class(env, KNN_RESULT);
    jclass localStringClass = find_class(env, STRING);

    JCLASS_KNN_RESULT = static_cast<jclass>(env->NewGlobalRef(localKnnResultClass));
    JCLASS_STRING = static_cast<jclass>(env->NewGlobalRef(localStringClass));

    JMETHODID_KNN_RESULT_CONSTRUCTOR = find_method_id(env, KNN_RESULT, "<init>", "([Ljava/lang/String;[FI)V");

    env->DeleteLocalRef(localKnnResultClass);
    env->DeleteLocalRef(localStringClass);

    return JNI_VERSION_10;
  }
}

/**
 * JNI_OnUnload function that is called when the native library is unloaded.
 *
 * @param vm The Java virtual machine.
 * @param reserved Reserved argument.
 */
void JNI_OnUnload(JavaVM* vm, void* reserved)
{
  JNIEnv* env;
  if (vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK)
  {
    return;
  }
  else
  {
    // Release the global references
    if (JCLASS_KNN_RESULT != nullptr)
    {
      env->DeleteGlobalRef(JCLASS_KNN_RESULT);
      env->DeleteGlobalRef(JCLASS_STRING);

      JCLASS_KNN_RESULT = nullptr;
      JCLASS_STRING = nullptr;
      JMETHODID_KNN_RESULT_CONSTRUCTOR = nullptr;
    }
  }
}

namespace knn = astdl::knn;

/**
 * Helper function to check if the given KnnData pointer is valid.
 *
 * @param env The JNI environment.
 * @param ptr The pointer to the KnnData object.
 * @return The valid KnnData object.
 */
knn::KnnData* checkKnnDataPointer(JNIEnv* env, jlong ptr)
{
  knn::KnnData* knnData = reinterpret_cast<knn::KnnData*>(ptr);

  if (knnData == nullptr)
  {
    throw_java_exception(env, NULL_POINTER_EXCEPTION, "null pointer passed as KnnData, call initialize first!");
  }
  return knnData;
}

/**
 * Initializes the KNearestNeighborsCuda.
 *
 * @param env The JNI environment.
 * @param clazz The JNI class.
 * @param maxK The maximum number of nearest neighbors to find.
 * @param batchSize The batch size.
 * @param numGpus The number of GPUs.
 * @param jDataType The data type.
 * @return The pointer to the KnnData object.
 */
JNIEXPORT jlong JNICALL Java_com_system_tensorhub_knn_KNearestNeighborsCuda_initialize(JNIEnv* env, jclass clazz,
                                                                                    jint maxK, jint batchSize, jint numGpus, jint jDataType)
{
  knn::DataType dataType;
  switch (jDataType)
  {
    case 0:
      dataType = knn::DataType::FP32;
      break;
    case 1:
      dataType = knn::DataType::FP16;
      break;
    default:
      throw_java_exception(env, ILLEGAL_ARGUMENT_EXCEPTION, "Unknown data type");
  }

  auto knnData = make_unique<knn::KnnData>(numGpus, batchSize, maxK, dataType);

  return reinterpret_cast<jlong>(knnData.release());
}

/**
 * Loads the data for KNearestNeighborsCuda.
 *
 * @param env The JNI environment.
 * @param clazz The JNI class.
 * @param jFnames The array of file names.
 * @param jDevices The array of device IDs.
 * @param keyValDelim The key-value delimiter character.
 * @param vecElemDelim The vector element delimiter character.
 * @param ptr The pointer to the KnnData object.
 */
JNIEXPORT void JNICALL Java_com_system_tensorhub_knn_KNearestNeighborsCuda_load(JNIEnv* env, jclass clazz,
                                                                               jobjectArray jFnames, jintArray jDevices, jchar keyValDelim, jchar vecElemDelim, jlong ptr)
{
  auto knnData = checkKnnDataPointer(env, ptr);

  jsize fnamesLen = env->GetArrayLength(jFnames);
  jsize devicesLen = env->GetArrayLength(jDevices);
  if (fnamesLen != devicesLen)
  {
    stringstream msg;
    msg << "filenames.length (" << fnamesLen << ") != devices.length (" << devicesLen << ")";
    throw_java_exception(env, ILLEGAL_ARGUMENT_EXCEPTION, msg.str());
  }

  map<int, unique_ptr<DataReader>> dataReaders;

  auto devices = env->GetIntArrayElements(jDevices, nullptr);
  for (int i = 0; i < fnamesLen; ++i)
  {
    int device = devices[i];

    jstring jFname = static_cast<jstring>(env->GetObjectArrayElement(jFnames, i));
    const char* fnamePtr = env->GetStringUTFChars(jFname, nullptr);
    string fname(fnamePtr);
    env->ReleaseStringUTFChars(jFname, fnamePtr);

    auto reader = make_unique<TextFileDataReader>(fname, keyValDelim, vecElemDelim);

    dataReaders.insert({device, move(reader)});
  }

  env->ReleaseIntArrayElements(jDevices, devices, JNI_ABORT);

  knnData->load(dataReaders);

  for (auto& entry : dataReaders)
  {
    entry.second.release();
  }
}

/**
 * Shuts down the KNearestNeighborsCuda.
 *
 * @param env The JNI environment.
 * @param clazz The JNI class.
 * @param ptr The pointer to the KnnData object.
 */
JNIEXPORT void JNICALL Java_com_system_tensorhub_knn_KNearestNeighborsCuda_shutdown(JNIEnv* env, jclass clazz, jlong ptr)
{
  auto knnData = checkKnnDataPointer(env, ptr);
  delete knnData;
}

/**
 * Finds the k nearest neighbors for the input vectors and returns the scores and keys.
 *
 * @param env The JNI environment.
 * @param clazz The JNI class.
 * @param k The number of nearest neighbors to find.
 * @param jInputVectors The array of input vectors.
 * @param size The active batch size.
 * @param width The feature size of the input vectors.
 * @param jScores The array to store the scores.
 * @param jKeys The array to store the keys.
 * @param ptr The pointer to the KnnData object.
 */
JNIEXPORT void JNICALL Java_com_system_tensorhub_knn_KNearestNeighborsCuda_findKnn__I_3FII_3F_3Ljava_lang_String_2J(JNIEnv* env, jclass clazz,
                                                                                                                 jint k, jfloatArray jInputVectors, jint size, jint width, jfloatArray jScores, jobjectArray jKeys, jlong ptr)
{
  auto knnData = checkKnnDataPointer(env, ptr);

  jsize length = env->GetArrayLength(jInputVectors);
  int batchSize = length / width;

  if (length % width != 0)
  {
    stringstream msg;
    msg << "feature size (" << width << ") does not divide data length (" << length << ")";
    throw_java_exception(env, ILLEGAL_ARGUMENT_EXCEPTION, msg.str());
  }

  if (batchSize != knnData->batchSize)
  {
    stringstream msg;
    msg << "length of input vectors (" << length << ") / feature size (" << width << ") != batch size (" << knnData->batchSize << ")";
    throw_java_exception(env, ILLEGAL_ARGUMENT_EXCEPTION, msg.str());
  }

  if (size > batchSize)
  {
    stringstream msg;
    msg << "active batch size (" << size << ") must be less than or equal to batch size (" << batchSize << ")";
    throw_java_exception(env, ILLEGAL_ARGUMENT_EXCEPTION, msg.str());
  }
  else
  {
    batchSize = size;
  }

  vector<string> keys(k * batchSize);
  auto inputVectors = env->GetPrimitiveArrayCritical(jInputVectors, nullptr);
  auto scores = env->GetPrimitiveArrayCritical(jScores, nullptr);

  knn::KnnExactGpu knnCuda(*knnData);

  knnCuda.search(k, reinterpret_cast<jfloat*>(inputVectors), batchSize, keys.data(), reinterpret_cast<jfloat*>(scores));

  env->ReleasePrimitiveArrayCritical(jInputVectors, inputVectors, JNI_ABORT);
  env->ReleasePrimitiveArrayCritical(jScores, scores, 0);

  for (int i = 0; i < batchSize; ++i)
  {
    for (int j = 0; j < k; ++j)
    {
      jstring key = env->NewStringUTF(keys[i * k + j].c_str());
      env->SetObjectArrayElement(jKeys, i * k + j, key);
    }
  }
}

/**
 * Finds the k nearest neighbors for the input vectors and returns the KnnResult object.
 *
 * @param env The JNI environment.
 * @param clazz The JNI class.
 * @param k The number of nearest neighbors to find.
 * @param jInputVectors The array of input vectors.
 * @param size The active batch size.
 * @param width The feature size of the input vectors.
 * @param ptr The pointer to the KnnData object.
 * @return The KnnResult object.
 */
JNIEXPORT jobject JNICALL Java_com_system_tensorhub_knn_KNearestNeighborsCuda_findKnn__I_3FIIJ(JNIEnv* env, jclass clazz,
                                                                                            jint k, jfloatArray jInputVectors, jint size, jint width, jlong ptr)
{
  auto knnData = checkKnnDataPointer(env, ptr);
  int batchSize = knnData->batchSize;

  jfloatArray jScores = env->NewFloatArray(k * size);
  jobjectArray jKeys = env->NewObjectArray(k * size, JCLASS_STRING, nullptr);

  Java_com_system_tensorhub_knn_KNearestNeighborsCuda_findKnn__I_3FII_3F_3Ljava_lang_String_2J(env, clazz, k,
                                                                                               jInputVectors, size, width, jScores, jKeys, ptr);

  jobject knnResult = env->NewObject(JCLASS_KNN_RESULT, JMETHODID_KNN_RESULT_CONSTRUCTOR, jKeys, jScores, k);

  if (knnResult == nullptr)
  {
    stringstream msg;
    msg << "Unable to create new object " << KNN_RESULT;
    throw_java_exception(env, RUNTIME_EXCEPTION, msg.str().c_str());
  }

  return knnResult;
}
