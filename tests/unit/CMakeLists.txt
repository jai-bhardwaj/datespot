cmake_minimum_required(VERSION 3.22.0)

project(tensorhub)

include(CheckCXXCompilerFlag)

CHECK_CXX_COMPILER_FLAG("-std=c++20" COMPILER_SUPPORTS_CXX20)

if(COMPILER_SUPPORTS_CXX20)
    set(CMAKE_CXX_STANDARD 20)
else()
    message(FATAL_ERROR "Your compiler ${CMAKE_CXX_COMPILER} has no C++20 support. Please use a different C++ compiler.")
endif()

find_package(PkgConfig)
find_package(GTest REQUIRED)

set(ENGINE_DIR ../../src/engine)
set(UTILS_DIR ../../src/utils)

include_directories(
    ${ENGINE_DIR}
    ${UTILS_DIR}
    ${GTEST_INCLUDE_DIRS}
)

set(UTILS_SOURCES
    ${UTILS_DIR}/CDFhelper.cpp
    ${UTILS_DIR}/Utils.cpp
)

set(TEST_SOURCES
    main.cpp
)

add_executable(unittests
    ${TEST_SOURCES}
    ${UTILS_SOURCES}
)

target_link_libraries(unittests
    ${GTEST_LIBRARIES}
)
