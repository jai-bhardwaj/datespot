#ifndef JNI_UTIL_H_
#define JNI_UTIL_H_

#include <jni.h>
#include <map>
#include <string_view>
#include <tuple>

namespace tensorhub {
namespace jni {

/**
 * @brief Exception type string for `java.lang.RuntimeException`.
 */
extern const std::string_view RuntimeException;

/**
 * @brief Exception type string for `java.lang.NullPointerException`.
 */
extern const std::string_view NullPointerException;

/**
 * @brief Exception type string for `java.lang.IllegalStateException`.
 */
extern const std::string_view IllegalStateException;

/**
 * @brief Exception type string for `java.lang.IllegalArgumentException`.
 */
extern const std::string_view IllegalArgumentException;

/**
 * @brief Exception type string for `java.lang.ClassNotFoundException`.
 */
extern const std::string_view ClassNotFoundException;

/**
 * @brief Exception type string for `java.lang.NoSuchMethodException`.
 */
extern const std::string_view NoSuchMethodException;

/**
 * @brief Exception type string for `java.io.FileNotFoundException`.
 */
extern const std::string_view FileNotFoundException;

/**
 * @brief Exception type string for `java.lang.UnsupportedOperationException`.
 */
extern const std::string_view UnsupportedOperationException;

/**
 * @brief Class name string for `java.util.ArrayList`.
 */
extern const std::string_view ArrayList;

/**
 * @brief Class name string for `java.lang.String`.
 */
extern const std::string_view String;

/**
 * @brief Descriptor for no-args constructor.
 */
extern const std::string_view NO_ARGS_CONSTRUCTOR;

/**
 * @brief Structure to store and manage global references to Java classes.
 */
struct References;

/**
 * @brief Deletes all global references stored in the given reference structure.
 *
 * @param env Pointer to the JNI environment.
 * @param refs Reference structure to be deleted.
 */
void deleteReferences(JNIEnv *env, References &refs);

/**
 * @brief Structure to store and manage global references to Java classes.
 */
struct References {
 private:
  std::map<std::string, jclass> classGlobalRefs;
 public:
  friend void deleteReferences(JNIEnv *env, References &refs);

  /**
   * @brief Retrieves the global reference to the specified class.
   *
   * @param className Name of the class.
   * @return Global reference to the specified class.
   */
  jclass getClassGlobalRef(const std::string_view &className) const;

  /**
   * @brief Checks if a global reference to the specified class is stored in the reference structure.
   *
   * @param className Name of the class.
   * @return True if a global reference to the specified class is stored, false otherwise.
   */
  bool containsClassGlobalRef(const std::string_view &className) const;

  /**
   * @brief Stores a global reference to the specified class in the reference structure.
   *
   * @param className Name of the class.
   * @param classRef Global reference to the class.
   */
  void putClassGlobalRef(const std::string_view &className, jclass classRef);
};

/**
 * @brief Throws a Java exception with a formatted message.
 *
 * @param env Pointer to the JNI environment.
 * @param exceptionType Type of the exception to be thrown.
 * @param fmt Format string for the exception message.
 * @param ... Additional arguments for the format string.
 */
void throwJavaException(JNIEnv* env, const std::string_view &exceptionType, const char *fmt, ...);

/**
 * @brief Finds the specified class and returns a global reference to it.
 *
 * @param env Pointer to the JNI environment.
 * @param refs Reference structure to store the references.
 * @param className Name of the class to be found.
 * @return Global reference to the found class.
 */
jclass findClassGlobalRef(JNIEnv *env, References &refs, const std::string_view &className);

/**
 * @brief Finds the specified method and returns its method ID.
 *
 * @param env Pointer to the JNI environment.
 * @param refs Reference structure to store the references.
 * @param className Name of the class where the method is defined.
 * @param methodName Name of the method to be found.
 * @param methodDescriptor Descriptor of the method to be found.
 * @return Method ID of the found method.
 */
jmethodID findMethodId(JNIEnv *env, References &refs, const std::string_view &className, const std::string_view &methodName,
                       const std::string_view &methodDescriptor);

/**
 * @brief Finds the constructor of the specified class and returns its method ID.
 *
 * @param env Pointer to the JNI environment.
 * @param refs Reference structure to store the references.
 * @param className Name of the class where the constructor is defined.
 * @param methodDescriptor Descriptor of the constructor to be found.
 * @return Method ID of the found constructor.
 */
jmethodID findConstructorId(JNIEnv *env, References &refs, const std::string_view &className,
                            const std::string_view &methodDescriptor);

/**
 * @brief Creates a new object of the specified class using the specified constructor.
 *
 * @param env Pointer to the JNI environment.
 * @param refs Reference structure holding the class and constructor references.
 * @param className Name of the class to be instantiated.
 * @param jConstructor Method ID of the constructor to be used.
 * @param ... Arguments for the constructor.
 * @return New instance of the specified class.
 */
jobject newObject(JNIEnv *env, const References &refs, const std::string_view &className, jmethodID jConstructor, ...);

}
}
#endif
