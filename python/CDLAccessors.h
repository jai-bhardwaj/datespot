#ifndef __CDLACCESSORS_H__
#define __CDLACCESSORS_H__

#include <string_view>
#include <optional>
#include <memory>
#include <stdexcept>

using PtrCDL = std::unique_ptr<CDL>;

class CDLAccessors {
    public:
        /**
         * Get the random seed from a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* GetRandomSeed(PyObject* self, PyObject* args);

        /**
         * Set the random seed for a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* SetRandomSeed(PyObject* self, PyObject* args);

        /**
         * Get the number of epochs from a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* GetEpochs(PyObject* self, PyObject* args);

        /**
         * Set the number of epochs for a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* SetEpochs(PyObject* self, PyObject* args);

        /**
         * Get the batch size from a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* GetBatch(PyObject* self, PyObject* args);

        /**
         * Set the batch size for a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* SetBatch(PyObject* self, PyObject* args);

        /**
         * Get the checkpoint interval from a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* GetCheckpointInterval(PyObject* self, PyObject* args);

        /**
         * Set the checkpoint interval for a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* SetCheckpointInterval(PyObject* self, PyObject* args);

        /**
         * Get the alpha interval from a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* GetAlphaInterval(PyObject* self, PyObject* args);

        /**
         * Set the alpha interval for a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* SetAlphaInterval(PyObject* self, PyObject* args);

        /**
         * Get the shuffle indexes from a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* GetShuffleIndexes(PyObject* self, PyObject* args);

        /**
         * Set the shuffle indexes for a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* SetShuffleIndexes(PyObject* self, PyObject* args);

        /**
         * Get the alpha value from a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* GetAlpha(PyObject* self, PyObject* args);

        /**
         * Set the alpha value for a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* SetAlpha(PyObject* self, PyObject* args);

        /**
         * Get the alpha multiplier from a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* GetAlphaMultiplier(PyObject* self, PyObject* args);

        /**
         * Set the alpha multiplier for a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* SetAlphaMultiplier(PyObject* self, PyObject* args);

        /**
         * Get the lambda value from a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* GetLambda(PyObject* self, PyObject* args);

        /**
         * Set the lambda value for a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* SetLambda(PyObject* self, PyObject* args);

        /**
         * Get the mu value from a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* GetMu(PyObject* self, PyObject* args);

        /**
         * Set the mu value for a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* SetMu(PyObject* self, PyObject* args);

        /**
         * Get the mode from a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* GetMode(PyObject* self, PyObject* args);

        /**
         * Set the mode for a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* SetMode(PyObject* self, PyObject* args);

        /**
         * Get the optimizer from a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* GetOptimizer(PyObject* self, PyObject* args);

        /**
         * Set the optimizer for a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* SetOptimizer(PyObject* self, PyObject* args);

        /**
         * Get the network file name from a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* GetNetworkFileName(PyObject* self, PyObject* args);

        /**
         * Set the network file name for a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* SetNetworkFileName(PyObject* self, PyObject* args);

        /**
         * Get the data file name from a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* GetDataFileName(PyObject* self, PyObject* args);

        /**
         * Set the data file name for a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* SetDataFileName(PyObject* self, PyObject* args);

        /**
         * Get the checkpoint file name from a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* GetCheckpointFileName(PyObject* self, PyObject* args);

        /**
         * Set the checkpoint file name for a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* SetCheckpointFileName(PyObject* self, PyObject* args);

        /**
         * Get the results file name from a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* GetResultsFileName(PyObject* self, PyObject* args);

        /**
         * Set the results file name for a CDL object.
         *
         * @param self The PyObject pointer.
         * @param args The PyObject arguments.
         * @return PyObject* The result of the operation.
         */
        static inline PyObject* SetResultsFileName(PyObject* self, PyObject* args);

};

/**
 * Get the random seed from a CDL object.
 *
 * @param self The PyObject pointer.
 * @param args The PyObject arguments.
 * @return PyObject* The result of the operation.
 */
PyObject* CDLAccessors::GetRandomSeed(PyObject* self, PyObject* args) {
    PtrCDL pCDL = parsePtr<PtrCDL>(args, "cdl");
    if (!pCDL) return nullptr;
    return Py_BuildValue("i", pCDL->_randomSeed);
}

/**
 * Set the random seed for a CDL object.
 *
 * @param self The PyObject pointer.
 * @param args The PyObject arguments.
 * @return PyObject* The result of the operation.
 */
PyObject* CDLAccessors::SetRandomSeed(PyObject* self, PyObject* args) {
    std::optional<int> randomSeed;
    PtrCDL pCDL = parsePtrAndOneValue<PtrCDL, int>(args, randomSeed, "cdl", "Oi");
    pCDL->_randomSeed = randomSeed.value_or(0);
    Py_RETURN_NONE;
}

/**
 * Get the number of epochs from a CDL object.
 *
 * @param self The PyObject pointer.
 * @param args The PyObject arguments.
 * @return PyObject* The result of the operation.
 */
PyObject* CDLAccessors::GetEpochs(PyObject* self, PyObject* args) {
    PtrCDL pCDL = parsePtr<PtrCDL>(args, "cdl");
    if (!pCDL) return nullptr;
    return Py_BuildValue("i", pCDL->_epochs);
}

/**
 * Set the number of epochs for a CDL object.
 *
 * @param self The PyObject pointer.
 * @param args The PyObject arguments.
 * @return PyObject* The result of the operation.
 */
PyObject* CDLAccessors::SetEpochs(PyObject* self, PyObject* args) {
    std::optional<int> epochs;
    PtrCDL pCDL = parsePtrAndOneValue<PtrCDL, int>(args, epochs, "cdl", "Oi");
    pCDL->_epochs = epochs.value_or(0);
    Py_RETURN_NONE;
}

/**
 * Get the batch size from a CDL object.
 *
 * @param self The PyObject pointer.
 * @param args The PyObject arguments.
 * @return PyObject* The result of the operation.
 */
PyObject* CDLAccessors::GetBatch(PyObject* self, PyObject* args) {
    PtrCDL pCDL = parsePtr<PtrCDL>(args, "cdl");
    if (!pCDL) return nullptr;
    return Py_BuildValue("i", pCDL->_batch);
}

/**
 * Set the batch size for a CDL object.
 *
 * @param self The PyObject pointer.
 * @param args The PyObject arguments.
 * @return PyObject* The result of the operation.
 */
PyObject* CDLAccessors::SetBatch(PyObject* self, PyObject* args) {
    std::optional<int> batch;
    PtrCDL pCDL = parsePtrAndOneValue<PtrCDL, int>(args, batch, "cdl", "Oi");
    pCDL->_batch = batch.value_or(0);
    Py_RETURN_NONE;
}

/**
 * Get the checkpoint interval from a CDL object.
 *
 * @param self The PyObject pointer.
 * @param args The PyObject arguments.
 * @return PyObject* The result of the operation.
 */
PyObject* CDLAccessors::GetCheckpointInterval(PyObject* self, PyObject* args) {
    PtrCDL pCDL = parsePtr<PtrCDL>(args, "cdl");
    if (!pCDL) return nullptr;
    return Py_BuildValue("i", pCDL->_checkpointInterval);
}

/**
 * Set the checkpoint interval for a CDL object.
 *
 * @param self The PyObject pointer.
 * @param args The PyObject arguments.
 * @return PyObject* The result of the operation.
 */
PyObject* CDLAccessors::SetCheckpointInterval(PyObject* self, PyObject* args) {
    std::optional<int> checkpointInterval;
    PtrCDL pCDL = parsePtrAndOneValue<PtrCDL, int>(args, checkpointInterval, "cdl", "Oi");
    pCDL->_checkpointInterval = checkpointInterval.value_or(0);
    Py_RETURN_NONE;
}

/**
 * Get the alpha interval from a CDL object.
 *
 * @param self The PyObject pointer.
 * @param args The PyObject arguments.
 * @return PyObject* The result of the operation.
 */
PyObject* CDLAccessors::GetAlphaInterval(PyObject* self, PyObject* args) {
    PtrCDL pCDL = parsePtr<PtrCDL>(args, "cdl");
    if (!pCDL) return nullptr;
    return Py_BuildValue("i", pCDL->_alphaInterval);
}

/**
 * Set the alpha interval for a CDL object.
 *
 * @param self The PyObject pointer.
 * @param args The PyObject arguments.
 * @return PyObject* The result of the operation.
 */
PyObject* CDLAccessors::SetAlphaInterval(PyObject* self, PyObject* args) {
    std::optional<int> alphaInterval;
    PtrCDL pCDL = parsePtrAndOneValue<PtrCDL, int>(args, alphaInterval, "cdl", "Oi");
    pCDL->_alphaInterval = alphaInterval.value_or(0);
    Py_RETURN_NONE;
}

/**
 * Get the shuffle indexes from a CDL object.
 *
 * @param self The PyObject pointer.
 * @param args The PyObject arguments.
 * @return PyObject* The result of the operation.
 */
PyObject* CDLAccessors::GetShuffleIndexes(PyObject* self, PyObject* args) {
    PtrCDL pCDL = parsePtr<PtrCDL>(args, "cdl");
    if (!pCDL) return nullptr;
    return Py_BuildValue("i", pCDL->_shuffleIndexes);
}

/**
 * Set the shuffle indexes for a CDL object.
 *
 * @param self The PyObject pointer.
 * @param args The PyObject arguments.
 * @return PyObject* The result of the operation.
 */
PyObject* CDLAccessors::SetShuffleIndexes(PyObject* self, PyObject* args) {
    std::optional<int> shuffleIndices;
    PtrCDL pCDL = parsePtrAndOneValue<PtrCDL, int>(args, shuffleIndices, "cdl", "Oi");
    pCDL->_shuffleIndexes = shuffleIndices.value_or(0);
    Py_RETURN_NONE;
}

/**
 * Get the alpha value from a CDL object.
 *
 * @param self The PyObject pointer.
 * @param args The PyObject arguments.
 * @return PyObject* The result of the operation.
 */
PyObject* CDLAccessors::GetAlpha(PyObject* self, PyObject* args) {
    PtrCDL pCDL = parsePtr<PtrCDL>(args, "cdl");
    if (!pCDL) return nullptr;
    return Py_BuildValue("f", pCDL->_alpha);
}

/**
 * Set the alpha value for a CDL object.
 *
 * @param self The PyObject pointer.
 * @param args The PyObject arguments.
 * @return PyObject* The result of the operation.
 */
PyObject* CDLAccessors::SetAlpha(PyObject* self, PyObject* args) {
    std::optional<float> alpha;
    PtrCDL pCDL = parsePtrAndOneValue<PtrCDL, float>(args, alpha, "cdl", "Of");
    pCDL->_alpha = alpha.value_or(0);
    Py_RETURN_NONE;
}

/**
 * Get the alpha multiplier from a CDL object.
 *
 * @param self The PyObject pointer.
 * @param args The PyObject arguments.
 * @return PyObject* The result of the operation.
 */
PyObject* CDLAccessors::GetAlphaMultiplier(PyObject* self, PyObject* args) {
    PtrCDL pCDL = parsePtr<PtrCDL>(args, "cdl");
    if (!pCDL) return nullptr;
    return Py_BuildValue("f", pCDL->_alphaMultiplier);
}

/**
 * Set the alpha multiplier for a CDL object.
 *
 * @param self The PyObject pointer.
 * @param args The PyObject arguments.
 * @return PyObject* The result of the operation.
 */
PyObject* CDLAccessors::SetAlphaMultiplier(PyObject* self, PyObject* args) {
    std::optional<float> alphaMultiplier;
    PtrCDL pCDL = parsePtrAndOneValue<PtrCDL, float>(args, alphaMultiplier, "cdl", "Of");
    pCDL->_alphaMultiplier = alphaMultiplier.value_or(0);
    Py_RETURN_NONE;
}

/**
 * Get the lambda value from a CDL object.
 *
 * @param self The PyObject pointer.
 * @param args The PyObject arguments.
 * @return PyObject* The result of the operation.
 */
PyObject* CDLAccessors::GetLambda(PyObject* self, PyObject* args) {
    PtrCDL pCDL = parsePtr<PtrCDL>(args, "cdl");
    if (!pCDL) return nullptr;
    return Py_BuildValue("f", pCDL->_lambda);
}

/**
 * Set the lambda value for a CDL object.
 *
 * @param self The PyObject pointer.
 * @param args The PyObject arguments.
 * @return PyObject* The result of the operation.
 */
PyObject* CDLAccessors::SetLambda(PyObject* self, PyObject* args) {
    std::optional<float> lambda;
    PtrCDL pCDL = parsePtrAndOneValue<PtrCDL, float>(args, lambda, "cdl", "Of");
    pCDL->_lambda = lambda.value_or(0);
    Py_RETURN_NONE;
}

/**
 * Get the mu value from a CDL object.
 *
 * @param self The PyObject pointer.
 * @param args The PyObject arguments.
 * @return PyObject* The result of the operation.
 */
PyObject* CDLAccessors::GetMu(PyObject* self, PyObject* args) {
    PtrCDL pCDL = parsePtr<PtrCDL>(args, "cdl");
    if (!pCDL) return nullptr;
    return Py_BuildValue("f", pCDL->_mu);
}

/**
 * Set the mu value for a CDL object.
 *
 * @param self The PyObject pointer.
 * @param args The PyObject arguments.
 * @return PyObject* The result of the operation.
 */
PyObject* CDLAccessors::SetMu(PyObject* self, PyObject* args) {
    std::optional<float> mu;
    PtrCDL pCDL = parsePtrAndOneValue<PtrCDL, float>(args, mu, "cdl", "Of");
    pCDL->_mu = mu.value_or(0);
    Py_RETURN_NONE;
}

/**
 * Get the mode for a CDL object.
 *
 * @param self The PyObject pointer.
 * @param args The PyObject arguments.
 * @return PyObject* The result of the operation.
 */
PyObject* CDLAccessors::GetMode(PyObject* self, PyObject* args) {
    PtrCDL pCDL = parsePtr<PtrCDL>(args, "cdl");
    if (!pCDL) return nullptr;
    auto it = intToStringModeMap.find(pCDL->_mode);
    if (it == intToStringModeMap.end()) {
        PyErr_SetString(PyExc_RuntimeError, "CDLAccessors::GetMode received unsupported mode enumerator");
        return nullptr;
    }
    return Py_BuildValue("s", it->second.c_str());
}

/**
 * Set the mode for a CDL object.
 *
 * @param self The PyObject pointer.
 * @param args The PyObject arguments.
 * @return PyObject* The result of the operation.
 */
PyObject* CDLAccessors::SetMode(PyObject* self, PyObject* args) {
    std::optional<char*> mode;
    PtrCDL pCDL = parsePtrAndOneValue<PtrCDL, char*>(args, mode, "cdl", "Os");
    auto it = stringToIntModeMap.find(std::string_view(mode.value_or("")));
    if (it == stringToIntModeMap.end()) {
        PyErr_SetString(PyExc_RuntimeError, "CDLAccessors::SetMode received unsupported mode enumerator string");
        return nullptr;
    }
    pCDL->_mode = it->second;
    Py_RETURN_NONE;
}

/**
 * Get the optimizer for a CDL object.
 *
 * @param self The PyObject pointer.
 * @param args The PyObject arguments.
 * @return PyObject* The result of the operation.
 */
PyObject* CDLAccessors::GetOptimizer(PyObject* self, PyObject* args) {
    PtrCDL pCDL = parsePtr<PtrCDL>(args, "cdl");
    if (!pCDL) return nullptr;
    auto it = intToStringTrainingModeMap.find(pCDL->_optimizer);
    if (it == intToStringTrainingModeMap.end()) {
        PyErr_SetString(PyExc_RuntimeError, "CDLAccessors::GetOptimizer received unsupported training mode enumerator");
        return nullptr;
    }
    return Py_BuildValue("s", it->second.c_str());
}

/**
 * Set the optimizer for a CDL object.
 *
 * @param self The PyObject pointer.
 * @param args The PyObject arguments.
 * @return PyObject* The result of the operation.
 */
PyObject* CDLAccessors::SetOptimizer(PyObject* self, PyObject* args) {
    std::optional<char*> trainingMode;
    PtrCDL pCDL = parsePtrAndOneValue<PtrCDL, char*>(args, trainingMode, "cdl", "Os");
    auto it = stringToIntTrainingModeMap.find(std::string_view(trainingMode.value_or("")));
    if (it == stringToIntTrainingModeMap.end()) {
        PyErr_SetString(PyExc_RuntimeError, "CDLAccessors::SetOptimizer received unsupported training mode enumerator string");
        return nullptr;
    }
    pCDL->_optimizer = it->second;
    Py_RETURN_NONE;
}

/**
 * Get the network file name from a CDL object.
 *
 * @param self The PyObject pointer.
 * @param args The PyObject arguments.
 * @return PyObject* The result of the operation.
 */
PyObject* CDLAccessors::GetNetworkFileName(PyObject* self, PyObject* args) {
    PtrCDL pCDL = parsePtr<PtrCDL>(args, "cdl");
    if (!pCDL) return nullptr;
    return Py_BuildValue("s", pCDL->_networkFileName.c_str());
}

/**
 * @brief Sets the network filename for a CDL object.
 *
 * This function sets the network filename for a CDL object based on the provided argument.
 *
 * @param self A pointer to the CDLAccessors object.
 * @param args The arguments passed to the function.
 * @return None.
 */
PyObject* CDLAccessors::SetNetworkFileName(PyObject* self, PyObject* args) {
    std::string_view networkFilename;
    CDL* const pCDL = parsePtrAndOneValue<CDL*, std::string_view>(args, networkFilename, "cdl", "Os");
    pCDL->_networkFileName = std::string(networkFilename);
    Py_RETURN_NONE;
}

/**
 * @brief Retrieves the data file name from a CDL object.
 *
 * This function takes a CDL object pointer and returns its data file name as a Python string.
 *
 * @param self The pointer to the Python object (self).
 * @param args The arguments passed to the function.
 *
 * @return A Python string representing the data file name of the CDL object.
 *         Returns nullptr if the CDL object pointer is null.
 */
PyObject* CDLAccessors::GetDataFileName(PyObject* self, PyObject* args) {
    if (CDL* pCDL = parsePtr<CDL*>(args, "cdl"); pCDL != nullptr) {
        return Py_BuildValue("s", pCDL->_dataFileName.c_str());
    }
    Py_RETURN_NONE;
}

/**
 * @brief Sets the data file name for a CDL object.
 *
 * This function takes a file name as a parameter and assigns it as the data file name for
 * the CDL object. The CDL object pointer is obtained through the `parsePtrAndOneValue` function.
 *
 * @param self The pointer to the Python object (self).
 * @param args The arguments passed to the function.
 *
 * @return None.
 */
PyObject* CDLAccessors::SetDataFileName(PyObject* self, PyObject* args) {
    const char* dataFilename = nullptr;
    if (CDL* const pCDL = parsePtrAndOneValue<CDL*, const char*>(args, dataFilename, "cdl", "Os"); pCDL != nullptr) {
        pCDL->_dataFileName = std::string(dataFilename);
    }
    Py_RETURN_NONE;
}

/**
 * @brief Retrieves the checkpoint file name from a CDL object.
 *
 * This function takes a CDL object pointer and returns its checkpoint file name as a Python string.
 *
 * @param self The pointer to the Python object (self).
 * @param args The arguments passed to the function.
 *
 * @return A Python string representing the checkpoint file name of the CDL object.
 *         Returns nullptr if the CDL object pointer is null.
 */
PyObject* CDLAccessors::GetCheckpointFileName(PyObject* self, PyObject* args) {
    if (CDL* pCDL = parsePtr<CDL*>(args, "cdl"); pCDL != nullptr) {
        return Py_BuildValue("s", pCDL->_checkpointFileName.c_str());
    }
    Py_RETURN_NONE;
}

/**
 * @brief Sets the checkpoint file name for a CDL object.
 *
 * This function takes a file name as a parameter and assigns it as the checkpoint file name for
 * the CDL object. The CDL object pointer is obtained through the `parsePtrAndOneValue` function.
 *
 * @param self The pointer to the Python object (self).
 * @param args The arguments passed to the function.
 *
 * @return None.
 */
PyObject* CDLAccessors::SetCheckpointFileName(PyObject* self, PyObject* args) {
    const char* checkpointFilename = nullptr;
    if (CDL* const pCDL = parsePtrAndOneValue<CDL*, const char*>(args, checkpointFilename, "cdl", "Os"); pCDL != nullptr) {
        pCDL->_checkpointFileName = std::string(checkpointFilename);
    }
    Py_RETURN_NONE;
}

/**
 * @brief Retrieves the results file name from a CDL object.
 *
 * This function takes a CDL object pointer and returns its results file name as a Python string.
 *
 * @param self The pointer to the Python object (self).
 * @param args The arguments passed to the function.
 *
 * @return A Python string representing the results file name of the CDL object.
 *         Returns nullptr if the CDL object pointer is null.
 */
PyObject* CDLAccessors::GetResultsFileName(PyObject* self, PyObject* args) {
    if (CDL* pCDL = parsePtr<CDL*>(args, "cdl"); pCDL != nullptr) {
        return Py_BuildValue("s", pCDL->_resultsFileName.c_str());
    }
    Py_RETURN_NONE;
}

/**
 * @brief Sets the results file name for a CDL object.
 *
 * This function takes a file name as a parameter and assigns it as the results file name for
 * the CDL object. The CDL object pointer is obtained through the `parsePtrAndOneValue` function.
 *
 * @param self The pointer to the Python object (self).
 * @param args The arguments passed to the function.
 *
 * @return None.
 */
PyObject* CDLAccessors::SetResultsFileName(PyObject* self, PyObject* args) {
    const char* resultsFilename = nullptr;
    if (CDL* const pCDL = parsePtrAndOneValue<CDL*, const char*>(args, resultsFilename, "cdl", "Os"); pCDL != nullptr) {
        pCDL->_resultsFileName = std::string(resultsFilename);
    }
    Py_RETURN_NONE;
}

#endif
