#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <sstream>
#include <map>
#include <memory>

#include "src/knn/KnnData.h"
#include "src/knn/DataReader.h"
#include "src/knn/KnnExactGpu.h"
#include "com_system_tensorhub_knn_KNearestNeighborsCuda.h"

// Global constants
constexpr char KNN_RESULT[] = "com/system/tensorhub/knn/KnnResult";
constexpr char STRING[] = "java/lang/String";
constexpr char ILLEGAL_ARGUMENT_EXCEPTION[] = "java/lang/IllegalArgumentException";
constexpr char CLASS_NOT_FOUND_EXCEPTION[] = "java/lang/ClassNotFoundException";
constexpr char NO_SUCH_METHOD_EXCEPTION[] = "java/lang/NoSuchMethodException";
constexpr char FILE_NOT_FOUND_EXCEPTION[] = "java/io/FileNotFoundException";
constexpr char RUNTIME_EXCEPTION[] = "java/lang/RuntimeException";
constexpr char nullptr_POINTER_EXCEPTION[] = "java/lang/NullPointerException";

// Global variables
static jclass JCLASS_KNN_RESULT;
static jmethodID JMETHODID_KNN_RESULT_CONSTRUCTOR;
static jclass JCLASS_STRING;

// Function to throw a Java exception
void throw_java_exception(JNIEnv *env, const char *exceptionType, const char *msg) {
  jclass exc = env->FindClass(exceptionType);
  if (exc == nullptr) {
    exc = env->FindClass(RUNTIME_EXCEPTION);
  }
  env->ThrowNew(exc, msg);
}

// Function to throw a Java exception with a string message
void throw_java_exception(JNIEnv *env, const char *exceptionType, const std::string &msg) {
  throw_java_exception(env, exceptionType, msg.c_str());
}

// Function to find a Java class by name
jclass find_class(JNIEnv *env, const char *className) {
  jclass clazz = env->FindClass(className);
  if (clazz == nullptr) {
    throw_java_exception(env, CLASS_NOT_FOUND_EXCEPTION, className);
  }
  return clazz;
}

// Function to find a Java method by name and descriptor
jmethodID find_method_id(JNIEnv *env, const char* className, const char *methodName, const char *methodDescriptor) {
  jclass clazz = find_class(env, className);
  jmethodID methodId = env->GetMethodID(clazz, methodName, methodDescriptor);
  if (methodId == nullptr) {
    std::stringstream msg;
    msg << className << "#" << methodName << methodDescriptor;
    throw_java_exception(env, NO_SUCH_METHOD_EXCEPTION, msg.str().c_str());
  }
  return methodId;
}

// JNI Initialization function
jint JNI_OnLoad(JavaVM *vm, void *reserved) {
  JNIEnv* env;
  if (vm->GetEnv((void **)&env, JNI_VERSION_1_6) != JNI_OK) {
    return JNI_ERR;
  } else {
    // Load necessary classes and method IDs
    jclass localKnnResultClass = find_class(env, KNN_RESULT);
    jclass localStringClass = find_class(env, STRING);

    JCLASS_KNN_RESULT = static_cast<jclass>(env->NewGlobalRef(localKnnResultClass));
    JCLASS_STRING = static_cast<jclass>(env->NewGlobalRef(localStringClass));

    JMETHODID_KNN_RESULT_CONSTRUCTOR = find_method_id(env, KNN_RESULT, "<init>", "([Ljava/lang/String;[FI)V");

    // Clean up local references
    env->DeleteLocalRef(localKnnResultClass);
    env->DeleteLocalRef(localStringClass);

    return JNI_VERSION_1_8;
  }
}

// JNI Unload function
void JNI_OnUnload(JavaVM *vm, void *reserved) {
  JNIEnv* env;
  if (vm->GetEnv((void **)&env, JNI_VERSION_1_6) != JNI_OK) {
    return;
  } else {
    // Delete global references and reset variables
    if (JCLASS_KNN_RESULT != nullptr) {
      env->DeleteGlobalRef(JCLASS_KNN_RESULT);
      env->DeleteGlobalRef(JCLASS_STRING);

      JCLASS_KNN_RESULT = nullptr;
      JCLASS_STRING = nullptr;
      JMETHODID_KNN_RESULT_CONSTRUCTOR = nullptr;
    }
  }
}

// Function to check if the given KnnData pointer is valid
astdl::knn::KnnData* checkKnnDataPointer(JNIEnv *env, jlong ptr) {
  astdl::knn::KnnData *knnData = reinterpret_cast<astdl::knn::KnnData*>(ptr);

  if (knnData == nullptr) {
    throw_java_exception(env, nullptr_POINTER_EXCEPTION, "nullptr pointer passed as KnnData, call initialize first!");
  }
  return knnData;
}

// JNI function to initialize KnnData
JNIEXPORT jlong JNICALL Java_com_system_tensorhub_knn_KNearestNeighborsCuda_initialize(JNIEnv *env, jclass clazz,
  jint maxK, jint batchSize, jint numGpus, jint jDataType) {
  astdl::knn::DataType dataType;
  switch (jDataType) {
    case 0:
      dataType = astdl::knn::DataType::FP32;
      break;
    case 1:
      dataType = astdl::knn::DataType::FP16;
      break;
    default:
      std::stringstream msg("Unknown data type [");
      msg << jDataType << "]";
      throw_java_exception(env, ILLEGAL_ARGUMENT_EXCEPTION, msg.str());
  }

  std::unique_ptr<astdl::knn::KnnData> knnData = std::make_unique<astdl::knn::KnnData>(numGpus, batchSize, maxK, dataType);

  return reinterpret_cast<jlong>(knnData.release());
}

// JNI function to load data
JNIEXPORT void JNICALL Java_com_system_tensorhub_knn_KNearestNeighborsCuda_load
(JNIEnv *env, jclass clazz, jobjectArray jFnames, jintArray jDevices, jchar keyValDelim, jchar vecElemDelim, jlong ptr) {
  astdl::knn::KnnData *knnData = checkKnnDataPointer(env, ptr);

  jsize fnamesLen = env->GetArrayLength(jFnames);
  jsize devicesLen = env->GetArrayLength(jDevices);
  if (fnamesLen != devicesLen) {
    std::stringstream msg;
    msg << "filenames.length (" << fnamesLen << ") != devices.length (" << devicesLen << ")";
    throw_java_exception(env, ILLEGAL_ARGUMENT_EXCEPTION, msg.str());
  }

  std::map<int, std::unique_ptr<DataReader>> dataReaders;

  jint *devices = env->GetIntArrayElements(jDevices, nullptr);
  for (int i = 0; i < fnamesLen; ++i) {
    jint device = devices[i];

    jstring jFname = static_cast<jstring>(env->GetObjectArrayElement(jFnames, i));
    const char* fnamePtr = env->GetStringUTFChars(jFname, nullptr);
    std::string fname(fnamePtr);
    env->ReleaseStringUTFChars(jFname, fnamePtr);

    std::unique_ptr<DataReader> reader = std::make_unique<TextFileDataReader>(fname, keyValDelim, vecElemDelim);

    dataReaders.insert({ device, std::move(reader) });
  }

  env->ReleaseIntArrayElements(jDevices, devices, JNI_ABORT);

  knnData->load(dataReaders);

  for (auto const& entry: dataReaders) {
    entry.second.release();
  }
}

// JNI function to shutdown KnnData
JNIEXPORT void JNICALL Java_com_system_tensorhub_knn_KNearestNeighborsCuda_shutdown(JNIEnv *env, jclass clazz, jlong ptr) {
  astdl::knn::KnnData *knnData = checkKnnDataPointer(env, ptr);
  delete knnData;
}

// JNI function to find K nearest neighbors
JNIEXPORT void JNICALL Java_com_system_tensorhub_knn_KNearestNeighborsCuda_findKnn__I_3FII_3F_3Ljava_lang_String_2J
(JNIEnv *env, jclass clazz, jint k, jfloatArray jInputVectors, jint size, jint width, jfloatArray jScores, jobjectArray jKeys, jlong ptr) {
  astdl::knn::KnnData *knnData = checkKnnDataPointer(env, ptr);

  jsize length = env->GetArrayLength(jInputVectors);
  int batchSize = length / width;

  if (length % width != 0) {
    std::stringstream msg;
    msg << "feature size (" << width << ") does not divide data length (" << length << ")";
    throw_java_exception(env, ILLEGAL_ARGUMENT_EXCEPTION, msg.str());
  }

  if (batchSize != knnData->batchSize) {
    std::stringstream msg;
    msg << "length of input vectors (" << length << ") / feature size (" << width << ") != batch size (" << knnData->batchSize << ")";
    throw_java_exception(env, ILLEGAL_ARGUMENT_EXCEPTION, msg.str());
  }

  if (size > batchSize) {
    std::stringstream msg;
    msg << "active batch size (" << size << ") must be less than or equal to batch size (" << batchSize <<")";
    throw_java_exception(env, ILLEGAL_ARGUMENT_EXCEPTION, msg.str());
  } else {
    batchSize = size;
  }

  std::unique_ptr<std::string[]> keys = std::make_unique<std::string[]>(k * batchSize);
  jfloat *inputVectors = env->GetPrimitiveArrayCritical(jInputVectors, nullptr);
  jfloat *scores = env->GetPrimitiveArrayCritical(jScores, nullptr);

  astdl::knn::KnnExactGpu knnCuda(knnData);

  knnCuda.search(k, inputVectors, batchSize, keys.get(), scores);

  env->ReleasePrimitiveArrayCritical(jInputVectors, inputVectors, JNI_ABORT);
  env->ReleasePrimitiveArrayCritical(jScores, scores, 0);

  for (int i = 0; i < batchSize; ++i) {
    for (int j = 0; j < k; ++j) {
      jstring key = env->NewStringUTF(keys[i * k + j].c_str());
      env->SetObjectArrayElement(jKeys, i * k + j, key);
    }
  }
}

// JNI function to find K nearest neighbors and return a KnnResult object
JNIEXPORT jobject JNICALL Java_com_system_tensorhub_knn_KNearestNeighborsCuda_findKnn__I_3FIIJ(JNIEnv *env, jclass clazz,
  jint k, jfloatArray jInputVectors, jint size, jint width, jlong ptr) {
  astdl::knn::KnnData *knnData = checkKnnDataPointer(env, ptr);
  int batchSize = knnData->batchSize;

  jfloatArray jScores = env->NewFloatArray(k * size);
  jobjectArray jKeys = env->NewObjectArray(k * size, JCLASS_STRING, nullptr);

  Java_com_system_tensorhub_knn_KNearestNeighborsCuda_findKnn__I_3FII_3F_3Ljava_lang_String_2J(env, clazz, k,
    jInputVectors, size, width, jScores, jKeys, ptr);

  jobject knnResult = env->NewObject(JCLASS_KNN_RESULT, JMETHODID_KNN_RESULT_CONSTRUCTOR, jKeys, jScores, k);

  if (knnResult == nullptr) {
    std::stringstream msg;
    msg << "Unable to create new object " << KNN_RESULT;
    throw_java_exception(env, RUNTIME_EXCEPTION, msg.str().c_str());
  }

  return knnResult;
}
