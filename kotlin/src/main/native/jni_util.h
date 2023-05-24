#ifndef JNI_UTIL_H_
#define JNI_UTIL_H_

#include <jni.h>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <format>

namespace tensorhub {
namespace jni {

/**
 * Exception types
 */
const std::string RuntimeException = "RuntimeException";
const std::string NullPointerException = "NullPointerException";
const std::string IllegalStateException = "IllegalStateException";
const std::string IllegalArgumentException = "IllegalArgumentException";
const std::string ClassNotFoundException = "ClassNotFoundException";
const std::string NoSuchMethodException = "NoSuchMethodException";
const std::string FileNotFoundException = "FileNotFoundException";
const std::string UnsupportedOperationException = "UnsupportedOperationException";

/**
 * Common class names
 */
const std::string ArrayList = "ArrayList";
const std::string String = "String";

const std::string NO_ARGS_CONSTRUCTOR = "NO_ARGS_CONSTRUCTOR";

/**
 * Stores references to global class references.
 */
struct References {
    std::unordered_map<std::string, jclass> classGlobalRefs;

    /**
     * Get the global reference to the specified class.
     * @param className The name of the class.
     * @return The global reference to the class or nullptr if not found.
     */
    jclass getClassGlobalRef(const std::string& className) const {
        auto it = classGlobalRefs.find(className);
        return (it != classGlobalRefs.end()) ? it->second : nullptr;
    }

    /**
     * Check if a global reference to the specified class exists.
     * @param className The name of the class.
     * @return True if a global reference exists, false otherwise.
     */
    bool containsClassGlobalRef(const std::string& className) const {
        return classGlobalRefs.find(className) != classGlobalRefs.end();
    }

    /**
     * Store the global reference to the specified class.
     * @param className The name of the class.
     * @param classRef The global reference to the class.
     */
    void putClassGlobalRef(const std::string& className, jclass classRef) {
        classGlobalRefs.emplace(className, classRef);
    }
};

/**
 * Delete the global references stored in the References struct.
 * @param env The JNI environment.
 * @param refs The References struct containing the global references.
 */
[[maybe_unused]] void deleteReferences(JNIEnv* env, References& refs);

/**
 * Throw a Java exception with the specified type and formatted error message.
 * @param env The JNI environment.
 * @param exceptionType The type of the Java exception to throw.
 * @param fmt The format string for the error message.
 * @param ... Additional arguments to format the error message.
 */
void throwJavaException(JNIEnv* env, const std::string& exceptionType, const std::string& fmt, ...);

/**
 * Find and return the global reference to the specified class.
 * If the reference does not exist, it will be created and stored.
 * @param env The JNI environment.
 * @param refs The References struct to store the global references.
 * @param className The name of the class.
 * @return The global reference to the class.
 */
jclass findClassGlobalRef(JNIEnv* env, References& refs, const std::string& className);

/**
 * Find and return the method ID of the specified method in the given class.
 * @param env The JNI environment.
 * @param refs The References struct containing the global references.
 * @param className The name of the class.
 * @param methodName The name of the method.
 * @param methodDescriptor The descriptor of the method.
 * @return The method ID or nullptr if not found.
 */
jmethodID findMethodId(JNIEnv* env, References& refs, const std::string& className, const std::string& methodName,
    const std::string& methodDescriptor);

/**
 * Find and return the constructor ID of the specified class and method descriptor.
 * @param env The JNI environment.
 * @param refs The References struct containing the global references.
 * @param className The name of the class.
 * @param methodDescriptor The descriptor of the constructor.
 * @return The constructor ID or nullptr if not found.
 */
jmethodID findConstructorId(JNIEnv* env, References& refs, const std::string& className,
    const std::string& methodDescriptor);

/**
 * Create a new Java object of the specified class using the provided constructor and arguments.
 * @tparam Args The types of the constructor arguments.
 * @param env The JNI environment.
 * @param refs The References struct containing the global references.
 * @param className The name of the class.
 * @param jConstructor The constructor ID.
 * @param args The arguments to pass to the constructor.
 * @return The created Java object or nullptr if an error occurred.
 */
template <typename... Args>
jobject newObject(JNIEnv* env, const References& refs, const std::string& className, jmethodID jConstructor,
    Args&&... args) {
    jclass cls = refs.getClassGlobalRef(className);
    if (cls == nullptr) {
        throwJavaException(env, ClassNotFoundException, "Class '%s' not found", className.c_str());
        return nullptr;
    }

    jobject obj = env->NewObject(cls, jConstructor, std::forward<Args>(args)...);
    if (env->ExceptionCheck()) {
        throwJavaException(env, RuntimeException, "Failed to create object of class '%s'", className.c_str());
        return nullptr;
    }

    return obj;
}

}  // namespace jni
}  // namespace tensorhub

#endif  // JNI_UTIL_H_`
