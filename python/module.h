#ifndef __MODULE_H__
#define __MODULE_H__

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_12_API_VERSION
#include <numpy/arrayobject.h>

#include <stdint.h>
#include <stdlib.h>
#include <string>

#include "GpuTypes.h"
#include "Types.h"
#include "cdl.h"
#include <span>

using std::map;
using std::string;

static std::unordered_map<std::string_view, Mode> stringToIntModeMap = {
    {"Prediction",  Mode::Prediction}, // Mapping from "Prediction" to Mode::Prediction.
    {"Training",    Mode::Training},   // Mapping from "Training" to Mode::Training.
    {"Validation",  Mode::Validation}, // Mapping from "Validation" to Mode::Validation.
    {"Unspecified", Mode::Unspecified} // Mapping from "Unspecified" to Mode::Unspecified.
};

/**
 * @brief Mapping from string representation to corresponding Mode enumeration.
 */
using StringToIntModeMap = std::unordered_map<std::string_view, Mode>;

static std::unordered_map<Mode, std::string_view> intToStringModeMap = {
    {Mode::Prediction,  "Prediction"}, // Mapping from Mode::Prediction to "Prediction".
    {Mode::Training,    "Training"},   // Mapping from Mode::Training to "Training".
    {Mode::Validation,  "Validation"}, // Mapping from Mode::Validation to "Validation".
    {Mode::Unspecified, "Unspecified"} // Mapping from Mode::Unspecified to "Unspecified".
};

/**
 * @brief Mapping from Mode enumeration to corresponding string representation.
 */
using IntToStringModeMap = std::unordered_map<Mode, std::string_view>;

static std::unordered_map<std::string_view, TrainingMode> stringToIntTrainingModeMap = {
    {"SGD",      TrainingMode::SGD},      // Mapping from "SGD" to TrainingMode::SGD.
    {"Momentum", TrainingMode::Momentum}, // Mapping from "Momentum" to TrainingMode::Momentum.
    {"AdaGrad",  TrainingMode::AdaGrad},  // Mapping from "AdaGrad" to TrainingMode::AdaGrad.
    {"Nesterov", TrainingMode::Nesterov}, // Mapping from "Nesterov" to TrainingMode::Nesterov.
    {"RMSProp",  TrainingMode::RMSProp},  // Mapping from "RMSProp" to TrainingMode::RMSProp.
    {"AdaDelta", TrainingMode::AdaDelta}, // Mapping from "AdaDelta" to TrainingMode::AdaDelta.
    {"Adam",     TrainingMode::Adam}      // Mapping from "Adam" to TrainingMode::Adam.
};

/**
 * @brief Mapping from string representation to corresponding TrainingMode enumeration.
 */
using StringToIntTrainingModeMap = std::unordered_map<std::string_view, TrainingMode>;

static std::unordered_map<TrainingMode, std::string_view> intToStringTrainingModeMap = {
    {TrainingMode::SGD,      "SGD"},      // Mapping from TrainingMode::SGD to "SGD".
    {TrainingMode::Momentum, "Momentum"}, // Mapping from TrainingMode::Momentum to "Momentum".
    {TrainingMode::AdaGrad,  "AdaGrad"},  // Mapping from TrainingMode::AdaGrad to "AdaGrad".
    {TrainingMode::Nesterov, "Nesterov"}, // Mapping from TrainingMode::Nesterov to "Nesterov".
    {TrainingMode::RMSProp,  "RMSProp"},  // Mapping from TrainingMode::RMSProp to "RMSProp".
    {TrainingMode::AdaDelta, "AdaDelta"}, // Mapping from TrainingMode::AdaDelta to "AdaDelta".
    {TrainingMode::Adam,     "Adam"}      // Mapping from TrainingMode::Adam to "Adam".
};

/**
 * @brief Mapping from TrainingMode enumeration to corresponding string representation.
 */
using IntToStringTrainingModeMap = std::unordered_map<TrainingMode, std::string_view>;

template <typename T>
/**
 * @brief Parses a Python tuple and extracts a pointer from the capsule.
 *
 * @param args The Python tuple containing the arguments.
 * @param key The key associated with the capsule.
 * @return T const The extracted pointer.
 * @note If the parsing fails or the capsule is invalid, nullptr is returned.
 * @note The caller is responsible for managing the lifetime and validity of the returned pointer.
 */
static T const parsePtr(PyObject* args, char const* key) {
    PyObject* capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O", &capsule))
        return nullptr;

    if (PyCapsule_IsValid(capsule, key) != 0) {
        T const ptr = reinterpret_cast<T const>(static_cast<void*>(PyCapsule_GetPointer(capsule, key)));
        if (ptr == nullptr)
            return nullptr;
        return ptr;
    } else {
        std::string_view capsuleName(PyCapsule_GetName(capsule));
        std::string message = "parsePtr invalid capsule: name = " + std::string(capsuleName) + "  key = " + std::string(key);
        PyErr_Format(PyExc_RuntimeError, "%s", message.c_str());
        return nullptr;
    }
}

template <typename T, typename V>
/**
 * @brief Parses a Python tuple and extracts a pointer and one value.
 *
 * @param args The Python tuple containing the arguments.
 * @param value The reference to the value to be extracted.
 * @param key The key associated with the capsule.
 * @param format The format string for parsing the tuple.
 * @return T const The extracted pointer.
 * @note If the parsing fails or the capsule is invalid, nullptr is returned.
 * @note The caller is responsible for managing the lifetime and validity of the returned pointer.
 */
static T const parsePtrAndOneValue(PyObject* args, V& value, char const* key, char const* format) {
    PyObject* capsule = nullptr;
    if (!PyArg_ParseTuple(args, format, &capsule, &value))
        return nullptr;

    if (PyCapsule_IsValid(capsule, key) != 0) {
        T const ptr = reinterpret_cast<T const>(static_cast<void*>(PyCapsule_GetPointer(capsule, key)));
        if (ptr == nullptr)
            return nullptr;
        return ptr;
    } else {
        std::string_view capsuleName(PyCapsule_GetName(capsule));
        std::string message = "parsePtrAndOneValue invalid capsule: name = " + std::string(capsuleName) + "  key = " + std::string(key);
        PyErr_Format(PyExc_RuntimeError, "%s", message.c_str());
        return nullptr;
    }
}

template <typename T, typename V, typename W>
/**
 * @brief Parses a Python tuple and extracts a pointer and two values.
 *
 * @param args The Python tuple containing the arguments.
 * @param value1 The reference to the first value to be extracted.
 * @param value2 The reference to the second value to be extracted.
 * @param key The key associated with the capsule.
 * @param format The format string for parsing the tuple.
 * @return T const The extracted pointer.
 * @note If the parsing fails or the capsule is invalid, nullptr is returned.
 * @note The caller is responsible for managing the lifetime and validity of the returned pointer.
 */
static T const parsePtrAndTwoValues(PyObject* args, V& value1, W& value2, char const* key, char const* format) {
    PyObject* capsule = nullptr;
    if (!PyArg_ParseTuple(args, format, &capsule, &value1, &value2))
        return nullptr;

    if (PyCapsule_IsValid(capsule, key) != 0) {
        T const ptr = reinterpret_cast<T const>(static_cast<void*>(PyCapsule_GetPointer(capsule, key)));
        if (ptr == nullptr)
            return nullptr;
        return ptr;
    } else {
        std::string_view capsuleName(PyCapsule_GetName(capsule));
        std::string message = "parsePtrAndTwoValues invalid capsule: name = " + std::string(capsuleName) + "  key = " + std::string(key);
        PyErr_Format(PyExc_RuntimeError, "%s", message.c_str());
        return nullptr;
    }
}

template <typename T, typename V, typename W, typename X>
/**
 * @brief Parses a Python tuple and extracts a pointer and three values.
 *
 * @param args The Python tuple containing the arguments.
 * @param value1 The reference to the first value to be extracted.
 * @param value2 The reference to the second value to be extracted.
 * @param value3 The reference to the third value to be extracted.
 * @param key The key associated with the capsule.
 * @param format The format string for parsing the tuple.
 * @return T const The extracted pointer.
 * @note If the parsing fails or the capsule is invalid, nullptr is returned.
 * @note The caller is responsible for managing the lifetime and validity of the returned pointer.
 */
static T const parsePtrAndThreeValues(PyObject* args, V& value1, W& value2, X& value3, char const* key, char const* format) {
    PyObject* capsule = nullptr;
    if (!PyArg_ParseTuple(args, format, &capsule, &value1, &value2, &value3))
        return nullptr;

    if (PyCapsule_IsValid(capsule, key) != 0) {
        T const ptr = reinterpret_cast<T const>(static_cast<void*>(PyCapsule_GetPointer(capsule, key)));
        if (ptr == nullptr)
            return nullptr;
        return ptr;
    } else {
        std::string_view capsuleName(PyCapsule_GetName(capsule));
        std::string message = "parsePtrAndThreeValues invalid capsule: name = " + std::string(capsuleName) + "  key = " + std::string(key);
        PyErr_Format(PyExc_RuntimeError, "%s", message.c_str());
        return nullptr;
    }
}

template <typename T, typename V, typename W, typename X, typename Y>
/**
 * @brief Parses a Python tuple and extracts a pointer and four values.
 *
 * @param args The Python tuple containing the arguments.
 * @param value1 The reference to the first value to be extracted.
 * @param value2 The reference to the second value to be extracted.
 * @param value3 The reference to the third value to be extracted.
 * @param value4 The reference to the fourth value to be extracted.
 * @param key The key associated with the capsule.
 * @param format The format string for parsing the tuple.
 * @return T const The extracted pointer.
 * @note If the parsing fails or the capsule is invalid, nullptr is returned.
 * @note The caller is responsible for managing the lifetime and validity of the returned pointer.
 */
static T const parsePtrAndFourValues(PyObject* args, V& value1, W& value2, X& value3, Y& value4, char const* key, char const* format) {
    PyObject* capsule = nullptr;
    if (!PyArg_ParseTuple(args, format, &capsule, &value1, &value2, &value3, &value4))
        return nullptr;

    if (PyCapsule_IsValid(capsule, key) != 0) {
        T const ptr = reinterpret_cast<T const>(static_cast<void*>(PyCapsule_GetPointer(capsule, key)));
        if (ptr == nullptr)
            return nullptr;
        return ptr;
    } else {
        std::string_view capsuleName(PyCapsule_GetName(capsule));
        std::string message = "parsePtrAndFourValues invalid capsule: name = " + std::string(capsuleName) + "  key = " + std::string(key);
        PyErr_Format(PyExc_RuntimeError, "%s", message.c_str());
        return nullptr;
    }
}

template <typename T, typename V, typename W, typename X, typename Y, typename Z>
/**
 * @brief Parses a Python tuple and extracts a pointer and five values.
 *
 * @param args The Python tuple containing the arguments.
 * @param value1 The reference to the first value to be extracted.
 * @param value2 The reference to the second value to be extracted.
 * @param value3 The reference to the third value to be extracted.
 * @param value4 The reference to the fourth value to be extracted.
 * @param value5 The reference to the fifth value to be extracted.
 * @param key The key associated with the capsule.
 * @param format The format string for parsing the tuple.
 * @return T const The extracted pointer.
 * @note If the parsing fails or the capsule is invalid, nullptr is returned.
 * @note The caller is responsible for managing the lifetime and validity of the returned pointer.
 */
static T const parsePtrAndFiveValues(PyObject* args, V& value1, W& value2, X& value3, Y& value4, Z& value5, char const* key, char const* format) {
    PyObject* capsule = nullptr;
    if (!PyArg_ParseTuple(args, format, &capsule, &value1, &value2, &value3, &value4, &value5))
        return nullptr;

    if (PyCapsule_IsValid(capsule, key) != 0) {
        T const ptr = reinterpret_cast<T const>(static_cast<void*>(PyCapsule_GetPointer(capsule, key)));
        if (ptr == nullptr)
            return nullptr;
        return ptr;
    } else {
        std::string_view capsuleName(PyCapsule_GetName(capsule));
        std::string message = "parsePtrAndFiveValues invalid capsule: name = " + std::string(capsuleName) + "  key = " + std::string(key);
        PyErr_Format(PyExc_RuntimeError, "%s", message.c_str());
        return nullptr;
    }
}

template <typename T, typename U, typename V, typename W, typename X, typename Y, typename Z>
/**
 * @brief Parses a Python tuple and extracts a pointer and six values.
 *
 * @param args The Python tuple containing the arguments.
 * @param value1 The reference to the first value to be extracted.
 * @param value2 The reference to the second value to be extracted.
 * @param value3 The reference to the third value to be extracted.
 * @param value4 The reference to the fourth value to be extracted.
 * @param value5 The reference to the fifth value to be extracted.
 * @param value6 The reference to the sixth value to be extracted.
 * @param key The key associated with the capsule.
 * @param format The format string for parsing the tuple.
 * @return T const The extracted pointer.
 * @note If the parsing fails or the capsule is invalid, nullptr is returned.
 * @note The caller is responsible for managing the lifetime and validity of the returned pointer.
 */
static T const parsePtrAndSixValues(PyObject* args, U& value1, V& value2, W& value3, X& value4, Y& value5, Z& value6,
                                    char const* key, char const* format) {
    PyObject* capsule = nullptr;
    if (!PyArg_ParseTuple(args, format, &capsule, &value1, &value2, &value3, &value4, &value5, &value6))
        return nullptr;

    if (PyCapsule_IsValid(capsule, key) != 0) {
        T const ptr = reinterpret_cast<T const>(static_cast<void*>(PyCapsule_GetPointer(capsule, key)));
        if (ptr == nullptr)
            return nullptr;
        return ptr;
    } else {
        std::string_view capsuleName(PyCapsule_GetName(capsule));
        std::string message = "parsePtrAndFiveValues invalid capsule: name = " + std::string(capsuleName) + "  key = " + std::string(key);
        PyErr_Format(PyExc_RuntimeError, "%s", message.c_str());
        return nullptr;
    }
}

/**
 * @brief Checks if the given NumPy array is of the expected type (float32).
 *
 * @param numpyArray The NumPy array to be checked.
 * @return PyArrayObject* The input NumPy array if it meets the expected type, otherwise nullptr.
 * @note If the input array does not match the expected type, a runtime error is set.
 * @note The caller is responsible for managing the memory and lifetime of the NumPy array.
 */
static PyArrayObject* CheckNumPyArray(PyArrayObject* numpyArray) {
    if (!PyArray_ISFLOAT(numpyArray) || !std::is_same_v<float, decltype(*PyArray_DATA(numpyArray))>) {
        PyErr_SetString(PyExc_RuntimeError, "CheckNumPyArray received incorrect NumPy array type; expected float32");
        return nullptr;
    }
    return numpyArray;
}

/**
 * @brief Converts a NumPy array to a vector of floats.
 *
 * @param numpyArray The NumPy array to be converted.
 * @return vector<float> The vector of floats containing the data from the NumPy array.
 * @note The returned vector may be empty if the conversion fails or the input array is empty.
 * @note The caller is responsible for managing the memory and lifetime of the NumPy array.
 */
static vector<float> NumPyArrayToVector(PyArrayObject* numpyArray) {
    npy_intp n = std::size(numpyArray);
    std::vector<float> v(n);
    void* data = PyArray_DATA(numpyArray);
    if (data == nullptr)
        return v;
    std::span<float> dataSpan(static_cast<float*>(data), n);
    std::copy(dataSpan.begin(), dataSpan.end(), v.begin());
    return v;
}

/**
 * @brief Converts a Python list to a vector of DataSetBase pointers.
 *
 * @param list The Python list containing the DataSetBase pointers.
 * @return vector<DataSetBase*> The vector of DataSetBase pointers.
 * @note The returned vector may contain nullptr elements if conversion fails for any item.
 * @note The caller is responsible for managing the memory and lifetime of the DataSetBase pointers.
 */
static vector<DataSetBase*> PythonListToDataSetBaseVector(PyObject* list) {
    Py_ssize_t size = PyList_Size(list);
    std::vector<DataSetBase*> vect(size, nullptr);
    try {
        for (Py_ssize_t i = 0; i < size; i++) {
            PyObject* capsule = PyList_GetItem(list, i);
            if (capsule == nullptr)
                throw std::runtime_error("Failed to retrieve item from Python list.");
            DataSetBase* pDataSetBase = reinterpret_cast<DataSetBase*>(PyCapsule_GetPointer(capsule, "data set"));
            if (pDataSetBase == nullptr)
                throw std::runtime_error("Failed to retrieve pointer from capsule.");
            vect.at(i) = pDataSetBase;
        }
    } catch (...) {
        vect.clear();
    }
    return vect;
}

/**
 * @brief Converts a vector of DataSetBase pointers to a Python list.
 *
 * @param vDataSetBase The vector of DataSetBase pointers.
 * @return PyObject* The Python list containing the converted DataSetBase pointers.
 * @note The caller is responsible for managing the reference count of the returned PyObject*.
 */
static PyObject* DataSetBaseVectorToPythonList(vector<DataSetBase*>& vDataSetBase) {
    Py_ssize_t size = std::size(vDataSetBase);
    PyObject* list = PyList_New(size);
    if (list == nullptr) {
        std::string message = "DataSetVectorToPythonArray failed in PyList_New(" + std::to_string(size) + ")";
        PyErr_SetString(PyExc_RuntimeError, message.c_str());
        return nullptr;
    }
    try {
        for (Py_ssize_t i = 0; i < size; i++) {
            PyObject* pDataSetBase = PyCapsule_New(reinterpret_cast<void*>(vDataSetBase.at(i)), "data set", nullptr);
            if (PyList_SetItem(list, i, pDataSetBase) < 0) {
                std::string message = "DataSetVectorToPythonArray failed in PyList_SetItem for index = " + std::to_string(i);
                PyErr_SetString(PyExc_RuntimeError, message.c_str());
                throw std::runtime_error("Failed to set item in Python list.");
            }
        }
        return list;
    } catch (...) {
        Py_DECREF(list);
        return nullptr;
    }
}

#endif
