#include <cstdio>
#include <map>
#include <iostream>
#include <string_view>
#include <sstream>

#include "jni_util.h"

/**
 * @brief Constant string that represents the constructor method name in Java.
 */
namespace {
const std::string_view CONSTRUCTOR_METHOD_NAME = "<init>";
}

/**
 * @brief Namespace for JNI related functions and constants.
 */
namespace tensorhub {
namespace jni {

/**
 * @brief Constant string that represents the java.lang.RuntimeException class.
 */
const std::string_view RuntimeException = "java/lang/RuntimeException";

/**
 * @brief Constant string that represents the java.lang.NullPointerException class.
 */
const std::string_view NullPointerException = "java/lang/NullPointerException";

/**
 * @brief Constant string that represents the java.lang.IllegalStateException class.
 */
const std::string_view IllegalStateException = "java/lang/IllegalStateException";

/**
 * @brief Constant string that represents the java.lang.IllegalArgumentException class.
 */
const std::string_view IllegalArgumentException = "java/lang/IllegalArgumentException";

/**
 * @brief Constant string that represents the java.lang.ClassNotFoundException class.
 */
const std::string_view ClassNotFoundException = "java/lang/ClassNotFoundException";

/**
 * @brief Constant string that represents the java.lang.NoSuchMethodException class.
 */
const std::string_view NoSuchMethodException = "java/lang/NoSuchMethodException";

/**
 * @brief Constant string that represents the java.io.FileNotFoundException class.
 */
const std::string_view FileNotFoundException = "java/io/FileNotFoundException";

/**
 * @brief Constant string that represents the java.lang.UnsupportedOperationException class.
 */
const std::string_view UnsupportedOperationException = "java/lang/UnsupportedOperationException";

/**
 * @brief Constant string that represents the java.util.ArrayList class.
 */
const std::string_view ArrayList = "java/util/ArrayList";

/**
 * @brief Constant string that represents the java.lang.String class.
 */
const std::string_view String = "java/lang/String";

/**
 * @brief Constant string that represents a constructor with no arguments in Java.
 */
const std::string_view NO_ARGS_CONSTRUCTOR = "()V";

    /**
     * @brief Deletes all global references in a References object.
     *
     * @param env JNI environment pointer.
     * @param refs Object containing references to Java classes.
     */
    void deleteReferences(JNIEnv *env, References &refs) {
      // Iterate over all class references
      for (auto &entry : refs.classGlobalRefs) {
        // Delete the global reference
        env->DeleteGlobalRef(entry.second);
      }
      // Clear the class references map
      refs.classGlobalRefs.clear();
    }

    /**
     * @brief Gets the global reference to a Java class stored in the References object.
     *
     * @param className The name of the Java class.
     *
     * @return The global reference to the Java class.
     */
    jclass References::getClassGlobalRef(const std::string_view &className) const {
      // Return the global reference to the specified Java class
      return classGlobalRefs.at(std::string{className});
    }

    /**
     * @brief Checks if the References object contains a global reference to a specified Java class.
     *
     * @param className The name of the Java class.
     *
     * @return True if the References object contains the global reference, False otherwise.
     */
    bool References::containsClassGlobalRef(const std::string_view &className) const {
      // Check if the global reference to the specified Java class exists
      return classGlobalRefs.find(std::string{className}) != classGlobalRefs.end();
    }

    /**
     * @brief Stores a global reference to a Java class in the References object.
     *
     * @param className The name of the Java class.
     * @param classRef The global reference to the Java class.
     */
    void References::putClassGlobalRef(const std::string_view &className, jclass classRef) {
      // Store the global reference to the Java class
      classGlobalRefs[std::string{className}] = classRef;
    }

    /**
     * @brief Throws a Java exception of the specified type with a formatted message.
     *
     * @param env JNI environment pointer.
     * @param exceptionType The name of the Java exception class.
     * @param fmt A format string for the exception message.
     * @param ... Variable argument list for the format string.
     */
    void throwJavaException(JNIEnv* env, const std::string_view &exceptionType, const char *fmt, ...) {
      // Find the Java exception class
      jclass exc = env->FindClass(std::string{exceptionType}.c_str());

      // Start processing the variable argument list
      va_list args;
      va_start(args, fmt);

      // Define a buffer for the exception message
      static const size_t MAX_MSG_LEN = 1024;
      char buffer[MAX_MSG_LEN];
      // Format the exception message
      if (vsnprintf(buffer, MAX_MSG_LEN, fmt, args) >= 0) {
        // Throw the exception with the formatted message
        env->ThrowNew(exc, buffer);
      } else {
        // Throw the exception with an empty message
        env->ThrowNew(exc, "");
      }

      // End processing the variable argument list
      va_end(args);
    }

    /**
     * @brief Finds the global reference to a specified Java class.
     *
     * @param env JNI environment pointer.
     * @param refs Object containing references to Java classes.
     * @param className The name of the Java class to find.
     *
     * @return The global reference to the specified Java class.
     *
     * @throws jni::ClassNotFoundException if the class is not found.
     */
    jclass findClassGlobalRef(JNIEnv *env, References &refs, const std::string_view &className) {
      // Check if the reference to the Java class already exists
      if (refs.containsClassGlobalRef(className)) {
        // Return the existing reference
        return refs.getClassGlobalRef(className);
      }

      // Find the local reference to the Java class
      jclass classLocalRef = env->FindClass(std::string{className}.c_str());

      // Check if an exception occurred while finding the class
      jthrowable exc = env->ExceptionOccurred();
      if (exc) {
        // Describe the exception
        env->ExceptionDescribe();
        // Exit the program
        exit(1);
      }

      // Check if the local reference is NULL
      if (classLocalRef == NULL) {
        // Throw a ClassNotFoundException with the error message
        throwJavaException(env, jni::ClassNotFoundException, "%s", std::string{className});
      }

      // Create a global reference to the class
      jclass classGlobalRef = (jclass) env->NewGlobalRef(classLocalRef);
      // Store the global reference in the References object
      refs.putClassGlobalRef(std::string{className}, classGlobalRef);
      // Delete the local reference
      env->DeleteLocalRef(classLocalRef);
      // Return the global reference
      return classGlobalRef;
    }

    /**
     * @brief Finds the method ID for a specified method in a Java class.
     *
     * @param env JNI environment pointer.
     * @param refs Object containing references to Java classes.
     * @param className The name of the Java class to find the method in.
     * @param methodName The name of the method.
     * @param methodDescriptor The method descriptor.
     *
     * @return The method ID for the specified method.
     *
     * @throws jni::NoSuchMethodException if the method does not exist.
     */
    jmethodID findMethodId(JNIEnv *env, References &refs, const std::string_view &className, const std::string_view &methodName,
                          const std::string_view &methodDescriptor) {
      // Get the global reference to the Java class
      jclass clazz = findClassGlobalRef(env, refs, className);
      // Get the method ID for the specified method and method descriptor
      jmethodID methodId = env->GetMethodID(clazz, std::string{methodName}.c_str(), std::string{methodDescriptor}.c_str());

      // Check if an exception occurred while getting the method ID
      jthrowable exc = env->ExceptionOccurred();
      if (exc) {
        // Print an error message
        std::cerr << "Error finding method " << std::string{className} << "#" << std::string{methodName} << std::string{methodDescriptor} << std::endl;
        // Describe the exception
        env->ExceptionDescribe();
        // Exit the program
        exit(1);
      }

      // Check if the method ID is NULL
      if (methodId == NULL) {
        // Throw a NoSuchMethodException with the error message
        throwJavaException(env, jni::NoSuchMethodException, "%s#%s%s", std::string{className}, std::string{methodName}, std::string{methodDescriptor});
      }

      // Return the method ID
      return methodId;
    }

    /**
     * @brief Finds the constructor ID for a specified constructor in a Java class.
     *
     * @param env JNI environment pointer.
     * @param refs Object containing references to Java classes.
     * @param className The name of the Java class to find the constructor in.
     * @param methodDescriptor The constructor descriptor.
     *
     * @return The constructor ID for the specified constructor.
     *
     * @throws jni::NoSuchMethodException if the constructor does not exist.
     */
    jmethodID findConstructorId(JNIEnv *env, References &refs, const std::string_view &className,
                                const std::string_view &methodDescriptor) {
      // Call findMethodId with the CONSTRUCTOR_METHOD_NAME constant as the method name
      return findMethodId(env, refs, className, CONSTRUCTOR_METHOD_NAME, methodDescriptor);
    }

    /**
     * @brief Creates a new Java object of the specified class.
     *
     * @param env JNI environment pointer.
     * @param refs Object containing references to Java classes.
     * @param className The name of the Java class to create the object from.
     * @param jConstructor The constructor method ID.
     * @param ... Variable argument list for the constructor parameters.
     *
     * @return The newly created Java object.
     *
     * @throws jni::RuntimeException if the object creation fails.
     */
    jobject newObject(JNIEnv *env, const References &refs, const std::string_view &className, jmethodID jConstructor, ...) {

      // Get the global reference to the Java class
      jclass clazz = refs.getClassGlobalRef(std::string{className});

      // Start processing the variable argument list
      va_list args;
      va_start(args, jConstructor);
      // Call the constructor using the variable argument list
      jobject obj = env->NewObjectV(clazz, jConstructor, args);
      // End processing the variable argument list
      va_end(args);

      // Check if an exception occurred during object creation
      jthrowable exc = env->ExceptionOccurred();
      if (exc) {
        // Describe the exception
        env->ExceptionDescribe();
        // Exit the program
        exit(1);
      }

      // Check if the object creation failed
      if (obj == NULL) {
        // Throw a runtime exception with the error message
        throwJavaException(env, jni::RuntimeException, "Unable to create new object: %s#%s", std::string{className},
                          CONSTRUCTOR_METHOD_NAME);
      }
      // Return the newly created object
      return obj;
    }
  }
}
