#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <sstream>

#include "src/knn/KnnData.h"
#include "src/knn/DataReader.h"
#include "src/knn/KnnExactGpu.h"
#include "com_system_tensorhub_knn_KNearestNeighborsCuda.h"

/**
 * @brief Constant string representing the fully-qualified class name of the KnnResult class.
 */
constexpr char KNN_RESULT[] = "com/system/eduwise/knn/KnnResult";

/**
 * @brief Constant string representing the fully-qualified class name of the String class.
 */
constexpr char STRING[] = "java/lang/String";

/**
 * @brief Constant string representing the fully-qualified class name of the IllegalArgumentException class.
 */
constexpr char ILLEGAL_ARGUMENT_EXCEPTION[] = "java/lang/IllegalArgumentException";

/**
 * @brief Constant string representing the fully-qualified class name of the ClassNotFoundException class.
 */
constexpr char CLASS_NOT_FOUND_EXCEPTION[] = "java/lang/ClassNotFoundException";

/**
 * @brief Constant string representing the fully-qualified class name of the NoSuchMethodException class.
 */
constexpr char NO_SUCH_METHOD_EXCEPTION[] = "java/lang/NoSuchMethodException";

/**
 * @brief Constant string representing the fully-qualified class name of the FileNotFoundException class.
 */
constexpr char FILE_NOT_FOUND_EXCEPTION[] = "java/io/FileNotFoundException";

/**
 * @brief Constant string representing the fully-qualified class name of the RuntimeException class.
 */
constexpr char RUNTIME_EXCEPTION[] = "java/lang/RuntimeException";

/**
 * @brief Constant string representing the fully-qualified class name of the NullPointerException class.
 */
constexpr char nullptr_POINTER_EXCEPTION[] = "java/lang/NullPointerException";

/**
 * @brief Static variable holding the JNI reference to the KnnResult class.
 */
static jclass JCLASS_KNN_RESULT;

/**
 * @brief Static variable holding the JNI reference to the constructor of the KnnResult class.
 */
static jmethodID JMETHODID_KNN_RESULT_CONSTRUCTOR;

/**
 * @brief Static variable holding the JNI reference to the String class.
 */
static jclass JCLASS_STRING;

/**
 * @brief Function to throw a Java exception.
 *
 * @param env Pointer to the JNI environment.
 * @param exceptionType Fully-qualified class name of the exception to be thrown.
 * @param msg Message to be passed with the exception.
 */
void throw_java_exception(JNIEnv *env, const char *exceptionType, const char *msg)
{
  jclass exc = env->FindClass(exceptionType);
  if (exc == nullptr)
  {
    exc = env->FindClass(RUNTIME_EXCEPTION);
  }
  env->ThrowNew(exc, msg);
}

/**
 * @brief Function to throw a Java exception
 *
 * @param env         Pointer to JNI environment
 * @param exceptionType   Type of the exception to be thrown
 * @param msg         Error message string
 */
void throw_java_exception(JNIEnv *env, const char *exceptionType, const std::string &msg)
{
  throw_java_exception(env, exceptionType, msg.c_str());
}

/**
 * @brief Function to find a class
 *
 * @param env         Pointer to JNI environment
 * @param className   Name of the class to be found
 *
 * @return Reference to the found class
 * @throws ClassNotFoundException if the class is not found
 */
jclass find_class(JNIEnv *env, const char *className)
{
  jclass clazz = env->FindClass(className);
  if (clazz == nullptr)
  {
    throw_java_exception(env, CLASS_NOT_FOUND_EXCEPTION, className);
  }
  return clazz;
}

/**
 * @brief Function to find a method ID
 *
 * @param env         Pointer to JNI environment
 * @param className   Name of the class in which the method is defined
 * @param methodName  Name of the method to be found
 * @param methodDescriptor Descriptor of the method to be found
 *
 * @return Method ID of the found method
 * @throws NoSuchMethodException if the method is not found
 */
jmethodID find_method_id(JNIEnv *env, const char* className, const char *methodName, const char *methodDescriptor)
{
  jclass clazz = find_class(env, className);
  jmethodID methodId = env->GetMethodID(clazz, methodName, methodDescriptor);
  if (methodId == nullptr)
  {
    std::stringstream msg;
    msg << className << "#" << methodName << methodDescriptor;
    throw_java_exception(env, NO_SUCH_METHOD_EXCEPTION, msg.str().c_str());
  }
  return methodId;
}

/**
 * @brief Function to load JNI resources
 *
 * @param vm      Java virtual machine
 * @param reserved Unused
 *
 * @return JNI_VERSION_1_8 if the function was successful
 * @return JNI_ERR if there was an error
 */
jint JNI_OnLoad(JavaVM *vm, void *reserved)
{
  JNIEnv* env;
  if (vm->GetEnv((void **) &env, JNI_VERSION_1_6) != JNI_OK)
  {
    return JNI_ERR;
  } else
  {
    jclass localKnnResultClass = find_class(env, KNN_RESULT);
    jclass localStringClass = find_class(env, STRING);

    JCLASS_KNN_RESULT = (jclass) env->NewGlobalRef(localKnnResultClass);
    JCLASS_STRING = (jclass) env->NewGlobalRef(localStringClass);

    JMETHODID_KNN_RESULT_CONSTRUCTOR = find_method_id(env, KNN_RESULT, "<init>", "([Ljava/lang/String;[FI)V");

    env->DeleteLocalRef(localKnnResultClass);
    env->DeleteLocalRef(localStringClass);

    return JNI_VERSION_1_8;
  }
}

/**
 * @brief Cleanup function to unload JNI resources
 *
 * @param vm      Java virtual machine
 * @param reserved Unused
 *
 */
void JNI_OnUnload(JavaVM *vm, void *reserved)
{
  JNIEnv* env;
  if (vm->GetEnv((void **) &env, JNI_VERSION_1_6) != JNI_OK)
  {
    return;
  } else
  {
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

/**
 * @brief Check if the pointer to the KNN data structure is valid
 *
 * @param env JNI environment pointer
 * @param ptr Pointer to the KNN data structure
 *
 * @return Pointer to the KNN data structure if it is valid
 *
 * @throws NullPointerException if the pointer is nullptr
 */
astdl::knn::KnnData* checkKnnDataPointer(JNIEnv *env, jlong ptr)
{
  astdl::knn::KnnData *knnData = (astdl::knn::KnnData*) ptr;

  if (knnData == nullptrptr)
  {
    throw_java_exception(env, nullptr_POINTER_EXCEPTION, "nullptr pointer passed as KnnData, call initialize first!");
  }
  return knnData;
}

/**
 * @brief Initialize the KNN data structure
 *
 * @param env       JNI environment pointer
 * @param clazz     Class calling this function
 * @param maxK      Maximum number of nearest neighbors to find
 * @param batchSize Batch size for processing data
 * @param numGpus   Number of GPUs to use for processing
 * @param jDataType Data type of the input vectors
 *
 * @return A long integer pointer to the newly created KNN data structure
 *
 * @throws IllegalArgumentException if the data type is unknown.
 */
JNIEXPORT jlong JNICALL Java_com_system_eduwise_knn_KNearestNeighborsCuda_initialize(JNIEnv *env, jclass clazz,
  jint maxK, jint batchSize, jint numGpus, jint jDataType)
{
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

  astdl::knn::KnnData *knnData = new astdl::knn::KnnData(numGpus, batchSize, maxK, dataType);

  return (jlong) knnData;
}

/**
 * @brief Load KNN data from text files
 *
 * @param env         JNI environment pointer
 * @param clazz       Class calling this function
 * @param jFnames     JNI object array containing filenames of the data files
 * @param jDevices    JNI integer array containing device IDs for each data file
 * @param keyValDelim Delimiter character for separating key and value in the data files
 * @param vecElemDelim Delimiter character for separating elements of a vector in the data files
 * @param ptr         Pointer to the data structure storing KNN data
 *
 * @throws IllegalArgumentException if lengths of filenames and devices arrays are different.
 */
JNIEXPORT void JNICALL Java_com_system_eduwise_knn_KNearestNeighborsCuda_load
(JNIEnv *env, jclass clazz, jobjectArray jFnames, jintArray jDevices, jchar keyValDelim, jchar vecElemDelim, jlong ptr)
{
  astdl::knn::KnnData *knnData = checkKnnDataPointer(env, ptr);

  jsize fnamesLen = env->GetArrayLength(jFnames);
  jsize devicesLen = env->GetArrayLength(jDevices);
  if (fnamesLen != devicesLen)
  {
    std::stringstream msg;
    msg << "filenames.length (" << fnamesLen << ") != devices.length (" << devicesLen << ")";
    throw_java_exception(env, ILLEGAL_ARGUMENT_EXCEPTION, msg.str());
  }

  std::map<int, DataReader*> dataReaders;

  jint *devices = env->GetIntArrayElements(jDevices, nullptr);
  for (int i = 0; i < fnamesLen; ++i)
  {
    jint device = devices[i];

    jstring jFname = (jstring) env->GetObjectArrayElement(jFnames, i);
    const char* fnamePtr = env->GetStringUTFChars(jFname, nullptr);
    std::string fname(fnamePtr);
    env->ReleaseStringUTFChars(jFname, fnamePtr);

    DataReader *reader = new TextFileDataReader(fname, keyValDelim, vecElemDelim);

    dataReaders.insert(
      { device, reader});
  }

  env->ReleaseIntArrayElements(jDevices, devices, JNI_ABORT);

  knnData->load(dataReaders);

  for(auto const& entry: dataReaders)
  {
    delete entry.second;
  }
}

/**
 * @brief Deallocate memory for the KNN data structure
 *
 * @param env   JNI environment pointer
 * @param clazz Class calling this function
 * @param ptr   Pointer to the data structure storing KNN data
 */
JNIEXPORT void JNICALL Java_com_system_eduwise_knn_KNearestNeighborsCuda_shutdown(JNIEnv *env, jclass clazz, jlong ptr)
{
  astdl::knn::KnnData *knnData = checkKnnDataPointer(env, ptr);
  delete knnData;
}

/**
 * @brief Find k nearest neighbors for each input vector in a batch
 *
 * @param env         JNI environment pointer
 * @param clazz       Class calling this function
 * @param k           Number of nearest neighbors to find
 * @param jInputVectors  JNI float array containing input vectors
 * @param size        Number of active input vectors in the batch
 * @param width       Width (number of features) of each input vector
 * @param jScores     JNI float array to store distances to nearest neighbors
 * @param jKeys       JNI object array to store keys of nearest neighbors
 * @param ptr         Pointer to the data structure storing KNN data
 *
 * @throws IllegalArgumentException if feature size does not divide data length,
 *         if length of input vectors divided by feature size is not equal to batch size,
 *         or if active batch size is greater than batch size.
 */
JNIEXPORT void JNICALL Java_com_system_eduwise_knn_KNearestNeighborsCuda_findKnn__I_3FII_3F_3Ljava_lang_String_2J
(JNIEnv *env, jclass clazz, jint k, jfloatArray jInputVectors, jint size, jint width, jfloatArray jScores, jobjectArray jKeys, jlong ptr)
{
  astdl::knn::KnnData *knnData = checkKnnDataPointer(env, ptr);

  jsize length = env->GetArrayLength(jInputVectors);
  int batchSize = length / width;

  if (length % width != 0)
  {
    std::stringstream msg;
    msg << "feature size (" << width << ") does not divide data length (" << length << ")";
    throw_java_exception(env, ILLEGAL_ARGUMENT_EXCEPTION, msg.str());
  }

  if(batchSize != knnData->batchSize)
  {
    std::stringstream msg;
    msg << "length of input vectors (" << length << ") / feature size (" << width << ") != batch size (" << knnData->batchSize << ")";
    throw_java_exception(env, ILLEGAL_ARGUMENT_EXCEPTION, msg.str());
  }

  if(size > batchSize)
  {
    std::stringstream msg;
    msg << "active batch size (" << size << ") must be less than or equal to batch size (" << batchSize <<")";
    throw_java_exception(env, ILLEGAL_ARGUMENT_EXCEPTION, msg.str());
  } else
  {
    batchSize = size;
  }

  std::string *keys = new std::string[k * batchSize];
  jfloat *inputVectors = (jfloat*) env->GetPrimitiveArrayCritical(jInputVectors, nullptr);
  jfloat *scores = (jfloat*) env->GetPrimitiveArrayCritical(jScores, nullptr);

  astdl::knn::KnnExactGpu knnCuda(knnData);

  knnCuda.search(k, inputVectors, batchSize, keys, scores);

  env->ReleasePrimitiveArrayCritical(jInputVectors, (void*) inputVectors, JNI_ABORT);
  env->ReleasePrimitiveArrayCritical(jScores, scores, 0);

  for (int i = 0; i < batchSize; ++i)
  {
    for (int j = 0; j < k; ++j)
    {
      jstring key = env->NewStringUTF(keys[i * k + j].c_str());
      env->SetObjectArrayElement(jKeys, i * k + j, key);
    }
  }

  delete[] (keys);
}

/**
 * Finds the k nearest neighbors for the input vectors and returns the result.
 *
 * @param env The JNI environment.
 * @param clazz The class object.
 * @param k The number of nearest neighbors to find.
 * @param jInputVectors The input vectors.
 * @param size The number of input vectors.
 * @param width The width of the input vectors.
 * @param ptr A pointer to the KNN data.
 * @return A KNNResult object, containing the keys and scores of the nearest neighbors.
 */
JNIEXPORT jobject JNICALL Java_com_system_eduwise_knn_KNearestNeighborsCuda_findKnn__I_3FIIJ(JNIEnv *env, jclass clazz,
  jint k, jfloatArray jInputVectors, jint size, jint width, jlong ptr)
{
  astdl::knn::KnnData *knnData = checkKnnDataPointer(env, ptr);
  int batchSize = knnData->batchSize;

  jfloatArray jScores = env->NewFloatArray(k * size);
  jobjectArray jKeys = env->NewObjectArray(k * size, JCLASS_STRING, nullptr);

  Java_com_system_eduwise_knn_KNearestNeighborsCuda_findKnn__I_3FII_3F_3Ljava_lang_String_2J(env, clazz, k,
    jInputVectors, size, width, jScores, jKeys, ptr);

  jobject knnResult = env->NewObject(JCLASS_KNN_RESULT, JMETHODID_KNN_RESULT_CONSTRUCTOR, jKeys, jScores, k);

  if (knnResult == nullptr)
  {
    std::stringstream msg;
    msg << "Unable to create new object " << KNN_RESULT;
    throw_java_exception(env, RUNTIME_EXCEPTION, msg.str().c_str());
  }

  return knnResult;
}
