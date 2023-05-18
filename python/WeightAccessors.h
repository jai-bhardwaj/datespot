#ifndef __WEIGHTACCESSORS_H__
#define __WEIGHTACCESSORS_H__

#include <optional>
#include <tuple>
#include <vector>
#include <cstdint>
#include <stdexcept>

/**
 * @brief Class for accessing weight data.
 */
class WeightAccessors {
public:
    /**
     * @brief Copy the weights from another weight object.
     *
     * @param self The calling Python object.
     * @param args The arguments passed to the function.
     * @return PyObject* Returns a Python object indicating the success of the operation.
     */
    static inline PyObject* CopyWeights(PyObject* self, PyObject* args);

    /**
     * @brief Set the weights of a weight object using a NumPy array.
     *
     * @param self The calling Python object.
     * @param args The arguments passed to the function.
     * @return PyObject* Returns a Python object indicating the success of the operation.
     */
    static inline PyObject* SetWeights(PyObject* self, PyObject* args);

    /**
     * @brief Set the biases of a weight object using a NumPy array.
     *
     * @param self The calling Python object.
     * @param args The arguments passed to the function.
     * @return PyObject* Returns a Python object indicating the success of the operation.
     */
    static inline PyObject* SetBiases(PyObject* self, PyObject* args);

    /**
     * @brief Get the weights of a weight object as a NumPy array.
     *
     * @param self The calling Python object.
     * @param args The arguments passed to the function.
     * @return PyObject* Returns a NumPy array containing the weights.
     */
    static inline PyObject* GetWeights(PyObject* self, PyObject* args);

    /**
     * @brief Get the biases of a weight object as a NumPy array.
     *
     * @param self The calling Python object.
     * @param args The arguments passed to the function.
     * @return PyObject* Returns a NumPy array containing the biases.
     */
    static inline PyObject* GetBiases(PyObject* self, PyObject* args);

    /**
     * @brief Set the normalization factor of a weight object.
     *
     * @param self The calling Python object.
     * @param args The arguments passed to the function.
     * @return PyObject* Returns a Python object indicating the success of the operation.
     */
    static inline PyObject* SetNorm(PyObject* self, PyObject* args);
};

/**
 * @brief Copies weights from one Weight object to another.
 *
 * This function receives two arguments: the destination weight object and the source weight object.
 * It extracts the Weight pointers and capsules from the arguments and performs the copy operation.
 *
 * @param self A pointer to the WeightAccessors object.
 * @param args The arguments passed to the function.
 * @return A new Python object representing the result of the copy operation, or nullptr on failure.
 */
PyObject* WeightAccessors::CopyWeights(PyObject* self, PyObject* args) {
    std::optional<PyObject*> capsule;
    std::optional<Weight*> pWeight;
    std::tie(pWeight, capsule) = parsePtrAndOneValue<Weight*, PyObject*>(args, "weight", "OO");
    if (!pWeight.has_value()) {
        return nullptr;
    }
    Weight* pSrcWeight = reinterpret_cast<Weight*>(PyCapsule_GetPointer(capsule.value(), "weight"));
    if (pSrcWeight == nullptr) {
        return nullptr;
    }
    return Py_BuildValue("i", pWeight.value()->CopyWeights(pSrcWeight));
}

/**
 * @brief Sets the weights of a Weight object using a NumPy array.
 *
 * This function receives two arguments: the weight object and the NumPy array containing the weights.
 * It extracts the Weight pointer and the NumPy array pointer from the arguments and performs the weight setting operation.
 *
 * @param self A pointer to the WeightAccessors object.
 * @param args The arguments passed to the function.
 * @return A new Python object representing the result of the weight setting operation, or nullptr on failure.
 */
PyObject* WeightAccessors::SetWeights(PyObject* self, PyObject* args) {
    std::optional<PyArrayObject*> numpyArray;
    std::optional<Weight*> pWeight;
    std::tie(pWeight, numpyArray) = parsePtrAndOneValue<Weight*, PyArrayObject*>(args, "weight", "OO");
    if (!pWeight.has_value()) {
        return nullptr;
    }
    if (CheckNumPyArray(numpyArray.value()) == nullptr) {
        return nullptr;
    }
    std::vector<float> weights = NumPyArrayToVector(numpyArray.value());
    if (weights.empty()) {
        PyErr_SetString(PyExc_RuntimeError, "WeightAccessors::SetWeights received empty weights vector");
        return nullptr;
    }
    return Py_BuildValue("i", pWeight.value()->SetWeights(weights));
}

/**
 * @brief Sets the biases of a Weight object using a NumPy array.
 *
 * This function receives two arguments: the weight object and the NumPy array containing the biases.
 * It extracts the Weight pointer and the NumPy array pointer from the arguments and performs the bias setting operation.
 *
 * @param self A pointer to the WeightAccessors object.
 * @param args The arguments passed to the function.
 * @return A new Python object representing the result of the bias setting operation, or nullptr on failure.
 */
PyObject* WeightAccessors::SetBiases(PyObject* self, PyObject* args) {
    std::optional<PyArrayObject*> numpyArray;
    std::optional<Weight*> pWeight;
    std::tie(pWeight, numpyArray) = parsePtrAndOneValue<Weight*, PyArrayObject*>(args, "weight", "OO");
    if (!pWeight.has_value()) {
        return nullptr;
    }
    if (CheckNumPyArray(numpyArray.value()) == nullptr) {
        return nullptr;
    }
    std::vector<float> biases = NumPyArrayToVector(numpyArray.value());
    if (biases.empty()) {
        PyErr_SetString(PyExc_RuntimeError, "WeightAccessors::SetBiases received empty biases vector");
        return nullptr;
    }
    return Py_BuildValue("i", pWeight.value()->SetBiases(biases));
}

/**
 * @brief Retrieves the weights of a Weight object as a NumPy array.
 *
 * This function receives one argument: the weight object.
 * It extracts the Weight pointer from the argument and retrieves the weights.
 * The weights are then copied into a NumPy array, which is returned as a Python object.
 *
 * @param self A pointer to the WeightAccessors object.
 * @param args The arguments passed to the function.
 * @return A NumPy array representing the weights of the Weight object, or nullptr on failure.
 */
PyObject* WeightAccessors::GetWeights(PyObject* self, PyObject* args) {
    std::vector<float> weights;
    std::optional<Weight*> pWeight;
    pWeight = parsePtr<Weight*>(args, "weight");
    if (!pWeight.has_value()) {
        return nullptr;
    }
    pWeight.value()->GetWeights(weights);
    std::vector<uint64_t> dimensions;
    if (!pWeight.value()->GetDimensions(dimensions)) {
        PyErr_SetString(PyExc_RuntimeError, "GetWeights failed in Weight::GetDimensions");
        return nullptr;
    }
    std::size_t nd = dimensions.size();
    std::vector<npy_intp> dims(nd);
    for (std::size_t i = 0; i < nd; i++) {
        dims[i] = static_cast<npy_intp>(dimensions.at(i));
    }
    PyArrayObject* numpyArray = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(nd, dims.data(), NPY_FLOAT32));
    if (numpyArray == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "WeightAccessors::GetWeights failed in PyArray_SimpleNew");
        Py_XDECREF(numpyArray);
        return nullptr;
    }
    memcpy(PyArray_DATA(numpyArray), weights.data(), weights.size() * sizeof(float));
    return reinterpret_cast<PyObject*>(numpyArray);
}

/**
 * @brief Retrieves the biases of a Weight object as a NumPy array.
 *
 * This function receives one argument: the weight object.
 * It extracts the Weight pointer from the argument and retrieves the biases.
 * The biases are then copied into a NumPy array, which is returned as a Python object.
 *
 * @param self A pointer to the WeightAccessors object.
 * @param args The arguments passed to the function.
 * @return A NumPy array representing the biases of the Weight object, or nullptr on failure.
 */
PyObject* WeightAccessors::GetBiases(PyObject* self, PyObject* args) {
    std::vector<float> biases;
    std::optional<Weight*> pWeight;
    pWeight = parsePtr<Weight*>(args, "weight");
    if (!pWeight.has_value()) {
        return nullptr;
    }
    pWeight.value()->GetBiases(biases);
    std::vector<uint64_t> dimensions;
    if (!pWeight.value()->GetDimensions(dimensions)) {
        PyErr_SetString(PyExc_RuntimeError, "GetBiases failed in Weight::GetDimensions");
        return nullptr;
    }
    std::size_t nd = dimensions.size();
    std::vector<npy_intp> dims(nd);
    for (std::size_t i = 0; i < nd; i++) {
        dims[i] = static_cast<npy_intp>(dimensions.at(i));
    }
    PyArrayObject* numpyArray = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(nd, dims.data(), NPY_FLOAT32));
    if (numpyArray == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "WeightAccessors::GetBiases failed in PyArray_SimpleNew");
        Py_XDECREF(numpyArray);
        return nullptr;
    }
    memcpy(PyArray_DATA(numpyArray), biases.data(), biases.size() * sizeof(float));
    return reinterpret_cast<PyObject*>(numpyArray);
}

/**
 * @brief Sets the norm of a Weight object.
 *
 * This function receives two arguments: the weight object and the desired norm value.
 * It extracts the Weight pointer and the norm value from the arguments and performs the norm setting operation.
 *
 * @param self A pointer to the WeightAccessors object.
 * @param args The arguments passed to the function.
 * @return A new Python object representing the result of the norm setting operation, or nullptr on failure.
 */
PyObject* WeightAccessors::SetNorm(PyObject* self, PyObject* args) {
    float norm = 0.0;
    std::optional<Weight*> pWeight;
    std::tie(pWeight, norm) = parsePtrAndOneValue<Weight*, float>(args, "weight", "Of");
    if (!pWeight.has_value()) {
        return nullptr;
    }
    return Py_BuildValue("i", pWeight.value()->SetNorm(norm));
}

#endif
