#ifndef __MODULE_H__
#define __MODULE_H__

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_12_API_VERSION
#include <numpy/arrayobject.h>

#include <stdint.h>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "GpuTypes.h"
#include "Types.h"
#include "cdl.h"
#include <span>

/**
 * @brief Mapping from string representation to corresponding Mode enumeration.
 */
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

/**
 * @brief Mapping from Mode enumeration to corresponding string representation.
 */
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

/**
 * @brief Mapping from string representation to corresponding TrainingMode enumeration.
 */
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

/**
 * @brief Mapping from TrainingMode enumeration to corresponding string representation.
 */
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

/**
 * @brief Parses a Python tuple and extracts a pointer from the capsule.
 *
 * @tparam T The type of the pointer.
 * @param args The Python tuple containing the arguments.
 * @param key The key associated with the capsule.
 * @return T const The extracted pointer.
 * @note If the parsing fails or the capsule is invalid, nullptr is returned.
 * @note The caller is responsible for managing the lifetime and validity of the returned pointer.
 */
template <typename T>
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

/**
 * @brief Parses a Python tuple and extracts a pointer and one value.
 *
 * @tparam T The type of the pointer.
 * @tparam V The type of the value.
 * @param args The Python tuple containing the arguments.
 * @param value The reference to the value to be extracted.
 * @param key The key associated with the capsule.
 * @param format The format string for parsing the tuple.
 * @return T const The extracted pointer.
 * @note If the parsing fails or the capsule is invalid, nullptr is returned.
 * @note The caller is responsible for managing the lifetime and validity of the returned pointer.
 */
template <typename T, typename V>
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

/**
 * @brief Parses a Python tuple and extracts a pointer and two values.
 *
 * @tparam T The type of the pointer.
 * @tparam V The type of the first value.
 * @tparam W The type of the second value.
 * @param args The Python tuple containing the arguments.
 * @param value1 The reference to the first value to be extracted.
 * @param value2 The reference to the second value to be extracted.
 * @param key The key associated with the capsule.
 * @param format The format string for parsing the tuple.
 * @return T const The extracted pointer.
 * @note If the parsing fails or the capsule is invalid, nullptr is returned.
 * @note The caller is responsible for managing the lifetime and validity of the returned pointer.
 */
template <typename T, typename V, typename W>
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

/**
 * @brief Parses a Python tuple and extracts a pointer and three values.
 *
 * @tparam T The type of the pointer.
 * @tparam V The type of the first value.
 * @tparam W The type of the second value.
 * @tparam X The type of the third value.
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
template <typename T, typename V, typename W, typename X>
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

/**
 * @brief Parses a Python tuple and extracts a pointer and four values.
 *
 * @tparam T The type of the pointer.
 * @tparam V The type of the first value.
 * @tparam W The type of the second value.
 * @tparam X The type of the third value.
 * @tparam Y The type of the fourth value.
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
template <typename T, typename V, typename W, typename X, typename Y>
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

/**
 * @brief Parses a Python tuple and extracts a pointer and five values.
 *
 * @tparam T The type of the pointer.
 * @tparam V The type of the first value.
 * @tparam W The type of the second value.
 * @tparam X The type of the third value.
 * @tparam Y The type of the fourth value.
 * @tparam Z The type of the fifth value.
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
template <typename T, typename V, typename W, typename X, typename Y, typename Z>
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

/**
 * @brief Module class.
 */
class Module {
public:
    /**
     * @brief Default constructor.
     */
    Module();

    /**
     * @brief Destructor.
     */
    ~Module();

    /**
     * @brief Initializes the module.
     *
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the initialization.
     */
    static PyObject* initialize(PyObject* self, PyObject* args);

    /**
     * @brief Sets the mode of the module.
     *
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of setting the mode.
     */
    static PyObject* setMode(PyObject* self, PyObject* args);

    /**
     * @brief Sets the training mode of the module.
     *
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of setting the training mode.
     */
    static PyObject* setTrainingMode(PyObject* self, PyObject* args);

    /**
     * @brief Registers a module with the module manager.
     *
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of registering the module.
     */
    static PyObject* registerModule(PyObject* self, PyObject* args);

    /**
     * @brief Registers a layer with the module manager.
     *
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of registering the layer.
     */
    static PyObject* registerLayer(PyObject* self, PyObject* args);

    /**
     * @brief Computes the forward pass of the module.
     *
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the forward pass.
     */
    static PyObject* forward(PyObject* self, PyObject* args);

    /**
     * @brief Computes the backward pass of the module.
     *
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the backward pass.
     */
    static PyObject* backward(PyObject* self, PyObject* args);

    /**
     * @brief Updates the module parameters using the specified optimizer.
     *
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the update.
     */
    static PyObject* update(PyObject* self, PyObject* args);

private:
    Mode mode_; ///< The current mode of the module.
    TrainingMode trainingMode_; ///< The current training mode of the module.
    std::unordered_map<std::string, ModuleBase*> modules_; ///< The registered modules.
    std::unordered_map<std::string, LayerBase*> layers_; ///< The registered layers.
};

#endif  // __MODULE_H__


