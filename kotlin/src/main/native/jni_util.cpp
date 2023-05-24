#include <iostream>
#include <unordered_map>
#include <string>
#include <sstream>
#include <format>
#include <cstdarg>
#include <jni.h>

#include "jni_util.h"

namespace {
constexpr std::string_view CONSTRUCTOR_METHOD_NAME = "<init>";
}

namespace tensorhub {
namespace jni {

/**
 * @brief Delete the global references stored in `refs`.
 *
 * @param env The JNI environment.
 * @param refs The References object containing the global references.
 */
void deleteReferences(JNIEnv* env, References& refs) {
  for (auto& entry : refs.classGlobalRefs) {
    env->DeleteGlobalRef(entry.second);
  }
  refs.classGlobalRefs.clear();
}

/**
 * @brief Get the global reference to a Java class by its name.
 *
 * @param className The fully qualified name of the Java class.
 * @return The global reference to the Java class.
 */
jclass References::getClassGlobalRef(const std::string& className) const {
  return classGlobalRefs.at(className);
}

/**
 * @brief Check if a global reference to a Java class is stored in `refs`.
 *
 * @param className The fully qualified name of the Java class.
 * @return true if the global reference is present, false otherwise.
 */
bool References::containsClassGlobalRef(const std::string& className) const {
  return classGlobalRefs.contains(className);
}

/**
 * @brief Store a global reference to a Java class in `refs`.
 *
 * @param className The fully qualified name of the Java class.
 * @param classRef The global reference to the Java class.
 */
void References::putClassGlobalRef(const std::string& className, jclass classRef) {
  classGlobalRefs[className] = classRef;
}

/**
 * @brief Throw a Java exception with a formatted error message.
 *
 * @param env The JNI environment.
 * @param exceptionType The fully qualified name of the Java exception class.
 * @param fmt The format string for the error message.
 * @param ... The variadic arguments for formatting the error message.
 */
void throwJavaException(JNIEnv* env, const std::string& exceptionType, const char* fmt, ...) {
  jclass exc = env->FindClass(exceptionType.c_str());

  std::va_list args;
  std::va_start(args, fmt);

  static constexpr size_t MAX_MSG_LEN = 1024;
  char buffer[MAX_MSG_LEN];
  if (std::vsnprintf(buffer, MAX_MSG_LEN, fmt, args) >= 0) {
    env->ThrowNew(exc, buffer);
  } else {
    env->ThrowNew(exc, "");
  }

  std::va_end(args);
}

/**
 * @brief Find the global reference to a Java class by its name.
 *        If the global reference is not found, it will be created and stored in `refs`.
 *
 * @param env The JNI environment.
 * @param refs The References object containing the global references.
 * @param className The fully qualified name of the Java class.
 * @return The global reference to the Java class.
 */
jclass findClassGlobalRef(JNIEnv* env, References& refs, const std::string& className) {
  if (refs.containsClassGlobalRef(className)) {
    return refs.getClassGlobalRef(className);
  }

  jclass classLocalRef = env->FindClass(className.c_str());

  jthrowable exc = env->ExceptionOccurred();
  if (exc) {
    env->ExceptionDescribe();
    throwJavaException(env, jni::ClassNotFoundException, "{}", className);
  }

  if (classLocalRef == NULL) {
    throwJavaException(env, jni::ClassNotFoundException, "{}", className);
  }

  jclass classGlobalRef = (jclass)env->NewGlobalRef(classLocalRef);
  refs.putClassGlobalRef(className, classGlobalRef);
  env->DeleteLocalRef(classLocalRef);
  return classGlobalRef;
}

/**
 * @brief Find the method ID of a Java method.
 *
 * @param env The JNI environment.
 * @param refs The References object containing the global references.
 * @param className The fully qualified name of the Java class.
 * @param methodName The name of the Java method.
 * @param methodDescriptor The descriptor of the Java method.
 * @return The method ID of the Java method.
 */
jmethodID findMethodId(JNIEnv* env, References& refs, const std::string& className, const std::string& methodName,
                       const std::string& methodDescriptor) {
  jclass clazz = findClassGlobalRef(env, refs, className);
  jmethodID methodId = env->GetMethodID(clazz, methodName.c_str(), methodDescriptor.c_str());

  jthrowable exc = env->ExceptionOccurred();
  if (exc) {
    env->ExceptionDescribe();
    throwJavaException(env, jni::NoSuchMethodException, "{}#{}{}", className, methodName, methodDescriptor);
  }

  if (methodId == NULL) {
    throwJavaException(env, jni::NoSuchMethodException, "{}#{}{}", className, methodName, methodDescriptor);
  }

  return methodId;
}

/**
 * @brief Find the constructor ID of a Java class.
 *
 * @param env The JNI environment.
 * @param refs The References object containing the global references.
 * @param className The fully qualified name of the Java class.
 * @param methodDescriptor The descriptor of the Java constructor.
 * @return The constructor ID of the Java class.
 */
jmethodID findConstructorId(JNIEnv* env, References& refs, const std::string& className,
                            const std::string& methodDescriptor) {
  return findMethodId(env, refs, className, CONSTRUCTOR_METHOD_NAME, methodDescriptor);
}

/**
 * @brief Create a new Java object using a constructor.
 *
 * @param env The JNI environment.
 * @param refs The References object containing the global references.
 * @param className The fully qualified name of the Java class.
 * @param jConstructor The constructor ID of the Java class.
 * @param ... The variadic arguments for the constructor.
 * @return The new Java object.
 */
jobject newObject(JNIEnv* env, const References& refs, const std::string& className, jmethodID jConstructor, ...) {
  jclass clazz = refs.getClassGlobalRef(className);

  std::va_list args;
  std::va_start(args, jConstructor);
  jobject obj = env->NewObjectV(clazz, jConstructor, args);
  std::va_end(args);

  jthrowable exc = env->ExceptionOccurred();
  if (exc) {
    env->ExceptionDescribe();
    throwJavaException(env, jni::RuntimeException, "Unable to create new object: {}#{}", className,
                       CONSTRUCTOR_METHOD_NAME);
  }

  if (obj == NULL) {
    throwJavaException(env, jni::RuntimeException, "Unable to create new object: {}#{}", className,
                       CONSTRUCTOR_METHOD_NAME);
  }
  return obj;
}

}  // namespace jni
}  // namespace tensorhub
