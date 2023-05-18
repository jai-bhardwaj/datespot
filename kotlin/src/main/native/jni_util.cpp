#include <cstdio>
#include <map>
#include <iostream>
#include <string_view>
#include <sstream>
#include <format>

#include "jni_util.h"

namespace tensorhub {
namespace jni {

const std::string_view RuntimeException = "java/lang/RuntimeException";
const std::string_view NullPointerException = "java/lang/NullPointerException";
const std::string_view IllegalStateException = "java/lang/IllegalStateException";
const std::string_view IllegalArgumentException = "java/lang/IllegalArgumentException";
const std::string_view ClassNotFoundException = "java/lang/ClassNotFoundException";
const std::string_view NoSuchMethodException = "java/lang/NoSuchMethodException";
const std::string_view FileNotFoundException = "java/io/FileNotFoundException";
const std::string_view UnsupportedOperationException = "java/lang/UnsupportedOperationException";
const std::string_view ArrayList = "java/util/ArrayList";
const std::string_view String = "java/lang/String";
const std::string_view NO_ARGS_CONSTRUCTOR = "()V";

namespace {
    constexpr std::string_view CONSTRUCTOR_METHOD_NAME = "<init>";
}

void deleteReferences(JNIEnv* env, References& refs) {
    for (auto& [_, entry] : refs.classGlobalRefs) {
        env->DeleteGlobalRef(entry);
    }
    refs.classGlobalRefs.clear();
}

jclass References::getClassGlobalRef(const std::string_view& className) const {
    return classGlobalRefs.at(className);
}

bool References::containsClassGlobalRef(const std::string_view& className) const {
    return classGlobalRefs.contains(className);
}

void References::putClassGlobalRef(const std::string_view& className, jclass classRef) {
    classGlobalRefs[className] = classRef;
}

void throwJavaException(JNIEnv* env, const std::string_view& exceptionType, const char* fmt, ...) {
    jclass exc = env->FindClass(exceptionType.data());

    va_list args;
    va_start(args, fmt);

    constexpr size_t MAX_MSG_LEN = 1024;
    std::string buffer(MAX_MSG_LEN, '\0');
    if (std::vsnprintf(buffer.data(), MAX_MSG_LEN, fmt, args) >= 0) {
        env->ThrowNew(exc, buffer.c_str());
    } else {
        env->ThrowNew(exc, "");
    }

    va_end(args);
}

jclass findClassGlobalRef(JNIEnv* env, References& refs, const std::string_view& className) {
    if (refs.containsClassGlobalRef(className)) {
        return refs.getClassGlobalRef(className);
    }

    jclass classLocalRef = env->FindClass(className.data());

    jthrowable exc = env->ExceptionOccurred();
    if (exc) {
        env->ExceptionDescribe();
        exit(1);
    }

    if (classLocalRef == nullptr) {
        throwJavaException(env, jni::ClassNotFoundException, "{}", className);
    }

    jclass classGlobalRef = static_cast<jclass>(env->NewGlobalRef(classLocalRef));
    refs.putClassGlobalRef(className, classGlobalRef);
    env->DeleteLocalRef(classLocalRef);
    return classGlobalRef;
}

jmethodID findMethodId(JNIEnv* env, References& refs, const std::string_view& className,
                       const std::string_view& methodName, const std::string_view& methodDescriptor) {
    jclass clazz = findClassGlobalRef(env, refs, className);
    jmethodID methodId = env->GetMethodID(clazz, methodName.data(), methodDescriptor.data());

    jthrowable exc = env->ExceptionOccurred();
    if (exc) {
        std::cerr << "Error finding method " << className << "#" << methodName << methodDescriptor << std::endl;
        env->ExceptionDescribe();
        exit(1);
    }

    if (methodId == nullptr) {
        throwJavaException(env, jni::NoSuchMethodException, "{}#{}{}", className, methodName, methodDescriptor);
    }

    return methodId;
}

jmethodID findConstructorId(JNIEnv* env, References& refs, const std::string_view& className,
                            const std::string_view& methodDescriptor) {
    return findMethodId(env, refs, className, CONSTRUCTOR_METHOD_NAME, methodDescriptor);
}

jobject newObject(JNIEnv* env, const References& refs, const std::string_view& className, jmethodID jConstructor, ...) {
    jclass clazz = refs.getClassGlobalRef(className);

    va_list args;
    va_start(args, jConstructor);
    jobject obj = env->NewObjectV(clazz, jConstructor, args);
    va_end(args);

    jthrowable exc = env->ExceptionOccurred();
    if (exc) {
        env->ExceptionDescribe();
        exit(1);
    }

    if (obj == nullptr) {
        throwJavaException(env, jni::RuntimeException, "Unable to create new object: {}#{}", className, CONSTRUCTOR_METHOD_NAME);
    }
    return obj;
}

} // namespace jni
} // namespace tensorhub
