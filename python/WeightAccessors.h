#ifndef __WEIGHTACCESSORS_H__
#define __WEIGHTACCESSORS_H__

#include <optional>
#include <tuple>
#include <vector>
#include <cstdint>
#include <stdexcept>

class WeightAccessors {
public:
    /**
     * @brief Copy the weights from one Weight object to another Weight object.
     *
     * This function copies the weights from a source Weight object to a target Weight object.
     *
     * @param self The reference to the WeightAccessors object.
     * @param args The arguments passed to the function.
     * @return A Py_BuildValue object containing the result of the CopyWeights method of the target Weight object.
     *         Returns nullptr if either Weight object is not found, the capsule is not valid, or an error occurs.
     */
    static inline PyObject* CopyWeights(PyObject* self, PyObject* args);

    /**
     * @brief Set the weights of a Weight object.
     *
     * This function sets the weights of a Weight object based on the provided NumPy array.
     *
     * @param self The reference to the WeightAccessors object.
     * @param args The arguments passed to the function.
     * @return A Py_BuildValue object containing the result of the SetWeights method of the Weight object.
     *         Returns nullptr if the Weight object is not found, the NumPy array is not valid, or an error occurs.
     */
    static inline PyObject* SetWeights(PyObject* self, PyObject* args);

    /**
     * @brief Set the biases of a Weight object.
     *
     * This function sets the biases of a Weight object based on the provided NumPy array.
     *
     * @param self The reference to the WeightAccessors object.
     * @param args The arguments passed to the function.
     * @return A Py_BuildValue object containing the result of the SetBiases method of the Weight object.
     *         Returns nullptr if the Weight object is not found, the NumPy array is not valid, or an error occurs.
     */
    static inline PyObject* SetBiases(PyObject* self, PyObject* args);

    /**
     * @brief Get the weights of a Weight object and return them as a NumPy array.
     *
     * This function retrieves the weights of a Weight object, converts them into a NumPy array,
     * and returns the NumPy array as a PyObject.
     *
     * @param self The reference to the WeightAccessors object.
     * @param args The arguments passed to the function.
     * @return A PyObject representing the NumPy array containing the weights of the Weight object.
     *         Returns nullptr if the Weight object is not found or an error occurs.
     */
    static inline PyObject* GetWeights(PyObject* self, PyObject* args);
    /**
     * @brief Get the biases of a Weight object and return them as a NumPy array.
     *
     * This function retrieves the biases of a Weight object, converts them into a NumPy array,
     * and returns the NumPy array as a PyObject.
     *
     * @param self The reference to the WeightAccessors object.
     * @param args The arguments passed to the function.
     * @return A PyObject representing the NumPy array containing the biases of the Weight object.
     *         Returns nullptr if the Weight object is not found or an error occurs.
     */
    static inline PyObject* GetBiases(PyObject* self, PyObject* args);

    /**
     * @brief Set the norm of a Weight object.
     *
     * This function sets the norm of a Weight object based on the provided value.
     *
     * @param self The reference to the WeightAccessors object.
     * @param args The arguments passed to the function.
     * @return A Py_BuildValue object containing the result of the SetNorm method of the Weight object.
     *         Returns nullptr if the Weight object is not found or an error occurs.
     */
    static inline PyObject* SetNorm(PyObject* self, PyObject* args);
};

/**
 * @brief Copy the weights from one Weight object to another Weight object.
 *
 * This function copies the weights from a source Weight object to a target Weight object.
 *
 * @param self The reference to the WeightAccessors object.
 * @param args The arguments passed to the function.
 * @return A Py_BuildValue object containing the result of the CopyWeights method of the target Weight object.
 *         Returns nullptr if either Weight object is not found, the capsule is not valid, or an error occurs.
 */
PyObject* WeightAccessors::CopyWeights(PyObject* self, PyObject* args) {
    std::optional<PyObject*> capsule;
    Weight* pWeight = parsePtrAndOneValue<Weight*, PyObject*>(args, capsule, "weight", "OO");
    if (pWeight == nullptr) {
        return nullptr;
    }
    Weight* pSrcWeight = reinterpret_cast<Weight*>(PyCapsule_GetPointer(capsule.value(), "weight"));
    if (pSrcWeight == nullptr) {
        return nullptr;
    }
    return Py_BuildValue("i", pWeight->CopyWeights(pSrcWeight));
}

/**
 * @brief Set the weights of a Weight object.
 *
 * This function sets the weights of a Weight object based on the provided NumPy array.
 *
 * @param self The reference to the WeightAccessors object.
 * @param args The arguments passed to the function.
 * @return A Py_BuildValue object containing the result of the SetWeights method of the Weight object.
 *         Returns nullptr if the Weight object is not found, the NumPy array is not valid, or an error occurs.
 */
PyObject* WeightAccessors::SetWeights(PyObject* self, PyObject* args) {
    std::optional<PyArrayObject*> numpyArray;
    Weight* pWeight = parsePtrAndOneValue<Weight*, PyArrayObject*>(args, numpyArray, "weight", "OO");
    if (pWeight == nullptr) {
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
    return Py_BuildValue("i", pWeight->SetWeights(weights));
}

/**
 * @brief Set the biases of a Weight object.
 *
 * This function sets the biases of a Weight object based on the provided NumPy array.
 *
 * @param self The reference to the WeightAccessors object.
 * @param args The arguments passed to the function.
 * @return A Py_BuildValue object containing the result of the SetBiases method of the Weight object.
 *         Returns nullptr if the Weight object is not found, the NumPy array is not valid, or an error occurs.
 */
PyObject* WeightAccessors::SetBiases(PyObject* self, PyObject* args) {
    std::optional<PyArrayObject*> numpyArray;
    Weight* pWeight = parsePtrAndOneValue<Weight*, PyArrayObject*>(args, numpyArray, "weight", "OO");
    if (pWeight == nullptr) {
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
    return Py_BuildValue("i", pWeight->SetBiases(biases));
}

/**
 * @brief Get the weights of a Weight object and return them as a NumPy array.
 *
 * This function retrieves the weights of a Weight object, converts them into a NumPy array,
 * and returns the NumPy array as a PyObject.
 *
 * @param self The reference to the WeightAccessors object.
 * @param args The arguments passed to the function.
 * @return A PyObject representing the NumPy array containing the weights of the Weight object.
 *         Returns nullptr if the Weight object is not found or an error occurs.
 */
PyObject* WeightAccessors::GetWeights(PyObject* self, PyObject* args) {
    std::vector<float> weights;
    Weight* pWeight = parsePtr<Weight*>(args, "weight");
    if (pWeight == nullptr) {
        return nullptr;
    }
    pWeight->GetWeights(weights);
    std::vector<uint64_t> dimensions;
    if (!pWeight->GetDimensions(dimensions)) {
        PyErr_SetString(PyExc_RuntimeError, "GetWeights failed in Weight::GetDimensions");
        return nullptr;
    }
    int nd = dimensions.size();
    std::vector<npy_intp> dims(nd);
    for (int i = 0; i < nd; i++) {
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
 * @brief Get the biases of a Weight object and return them as a NumPy array.
 *
 * This function retrieves the biases of a Weight object, converts them into a NumPy array,
 * and returns the NumPy array as a PyObject.
 *
 * @param self The reference to the WeightAccessors object.
 * @param args The arguments passed to the function.
 * @return A PyObject representing the NumPy array containing the biases of the Weight object.
 *         Returns nullptr if the Weight object is not found or an error occurs.
 */
PyObject* WeightAccessors::GetBiases(PyObject* self, PyObject* args) {
    std::vector<float> biases;
    Weight* pWeight = parsePtr<Weight*>(args, "weight");
    if (pWeight == nullptr) {
        return nullptr;
    }
    pWeight->GetBiases(biases);
    std::vector<uint64_t> dimensions;
    if (!pWeight->GetDimensions(dimensions)) {
        PyErr_SetString(PyExc_RuntimeError, "GetBiases failed in Weight::GetDimensions");
        return nullptr;
    }
    int nd = dimensions.size();
    std::vector<npy_intp> dims(nd);
    for (int i = 0; i < nd; i++) {
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
 * @brief Set the norm of a Weight object.
 *
 * This function sets the norm of a Weight object based on the provided value.
 *
 * @param self The reference to the WeightAccessors object.
 * @param args The arguments passed to the function.
 * @return A Py_BuildValue object containing the result of the SetNorm method of the Weight object.
 *         Returns nullptr if the Weight object is not found or an error occurs.
 */
PyObject* WeightAccessors::SetNorm(PyObject* self, PyObject* args) {
    float norm = 0.0;
    std::optional<Weight*> pWeight;
    std::tie(pWeight, norm) = parsePtrAndOneValue<Weight*, float>(args, norm, "weight", "Of");
    if (!pWeight.has_value()) {
        return nullptr;
    }
    return Py_BuildValue("i", pWeight.value()->SetNorm(norm));
}

#endif
