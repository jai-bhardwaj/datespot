#ifndef JNI_UTIL_H_
#define JNI_UTIL_H_

#include <jni.h>
#include <map>
#include <string_view>
#include <tuple>
#include <utility>
#include <string>

namespace tensorhub::jni {

constexpr inline std::string_view RuntimeException = "RuntimeException";
constexpr inline std::string_view NullPointerException = "NullPointerException";
constexpr inline std::string_view IllegalStateException = "IllegalStateException";
constexpr inline std::string_view IllegalArgumentException = "IllegalArgumentException";
constexpr inline std::string_view ClassNotFoundException = "ClassNotFoundException";
constexpr inline std::string_view NoSuchMethodException = "NoSuchMethodException";
constexpr inline std::string_view FileNotFoundException = "FileNotFoundException";
constexpr inline std::string_view UnsupportedOperationException = "UnsupportedOperationException";
constexpr inline std::string_view ArrayList = "ArrayList";
constexpr inline std::string_view String = "String";
constexpr inline std::string_view NO_ARGS_CONSTRUCTOR = "NO_ARGS_CONSTRUCTOR";

struct References {
 private:
  std::map<std::string, jclass> classGlobalRefs;

 public:
  friend void deleteReferences(JNIEnv *env, References &refs);

  jclass getClassGlobalRef(const std::string_view &className) const {
    auto it = classGlobalRefs.find(std::string(className));
    return it != classGlobalRefs.end() ? it->second : nullptr;
  }

  bool containsClassGlobalRef(const std::string_view &className) const {
    return classGlobalRefs.find(std::string(className)) != classGlobalRefs.end();
  }

  void putClassGlobalRef(const std::string_view &className, jclass classRef) {
    classGlobalRefs[std::string(className)] = classRef;
  }
};

void deleteReferences(JNIEnv *env, References &refs);

void throwJavaException(JNIEnv* env, const std::string_view &exceptionType, const char *fmt, ...);

jclass findClassGlobalRef(JNIEnv *env, References &refs, const std::string_view &className);

jmethodID findMethodId(JNIEnv *env, References &refs, const std::string_view &className,
                       const std::string_view &methodName, const std::string_view &methodDescriptor);

jmethodID findConstructorId(JNIEnv *env, References &refs, const std::string_view &className,
                            const std::string_view &methodDescriptor);

template<typename... Args>
jobject newObject(JNIEnv *env, const References &refs, const std::string_view &className,
                  jmethodID jConstructor, Args&&... args) {
  jclass cls = refs.getClassGlobalRef(className);
  if (!cls) {
    throwJavaException(env, ClassNotFoundException, "Class not found: %.*s", static_cast<int>(className.size()), className.data());
    return nullptr;
  }

  jobject newObj = env->NewObject(cls, jConstructor, std::forward<Args>(args)...);
  if (!newObj) {
    throwJavaException(env, RuntimeException, "Failed to create new object of class: %.*s", static_cast<int>(className.size()), className.data());
  }

  return newObj;
}

}  // namespace jni
}  // namespace tensorhub

#endif  // JNI_UTIL_H_

