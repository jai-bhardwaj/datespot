#ifndef __NETWORKFUNCTIONS_H__
#define __NETWORKFUNCTIONS_H__

#include <filesystem>
#include <optional>
#include <tuple>
#include <vector>
#include <cstdint>
#include <stdexcept>

class NetworkFunctions {
    public:
        /**
         * Clear datasets from the neural network.
         *
         * This function clears the datasets from the neural network.
         *
         * @param self The Python object calling this function (not used).
         * @param args The arguments passed to the function.
         *
         * @return PyObject* None.
         */
        static inline PyObject* ClearDataSets(PyObject* self, PyObject* args);

        /**
         * Load datasets into the neural network.
         *
         * This function loads the specified datasets into the neural network for training and validation.
         *
         * @param self The Python object calling this function (not used).
         * @param args The arguments passed to the function.
         *
         * @return PyObject* None.
         */
        static inline PyObject* LoadDataSets(PyObject* self, PyObject* args);

        /**
         * Randomize the neural network.
         *
         * This function randomizes the neural network by assigning random weights to its layers.
         *
         * @param self The Python object calling this function (not used).
         * @param args The arguments passed to the function.
         *
         * @return PyObject* None.
         */
        static inline PyObject* Randomize(PyObject* self, PyObject* args);

        /**
         * Validate the neural network.
         *
         * This function validates the neural network by running the validation process.
         *
         * @param self The Python object calling this function (not used).
         * @param args The arguments passed to the function.
         *
         * @return PyObject* The result of the validation as a Python integer.
         *                   Returns NULL in case of error.
         */
        static inline PyObject* Validate(PyObject* self, PyObject* args);

        /**
         * Train the neural network.
         *
         * This function trains the neural network using the specified number of epochs,
         * learning rate alpha, regularization parameters lambda, lambda1, mu, and mu1.
         *
         * @param self The Python object calling this function (not used).
         * @param args The arguments passed to the function.
         *
         * @return PyObject* The final training loss as a Python float.
         *                   Returns NULL in case of error.
         */
        static inline PyObject* Train(PyObject* self, PyObject* args);

        /**
         * Predict the batch using the neural network.
         *
         * This function predicts the output batch using the neural network and the specified number of layers.
         *
         * @param self The Python object calling this function (not used).
         * @param args The arguments passed to the function.
         *
         * @return PyObject* None.
         */
        static inline PyObject* PredictBatch(PyObject* self, PyObject* args);

        /**
         * Calculate the output using the neural network.
         *
         * This function calculates the output using the neural network, layer, k value,
         * and GPU buffers for key and value.
         *
         * @param self The Python object calling this function (not used).
         * @param args The arguments passed to the function.
         *
         * @return PyObject* None.
         */
        static inline PyObject* CalculateOutput(PyObject* self, PyObject* args);

        /**
         * Predict the output using the neural network.
         *
         * This function predicts the output using the neural network, datasets, and the specified CDL object.
         *
         * @param self The Python object calling this function (not used).
         * @param args The arguments passed to the function.
         *
         * @return PyObject* The predicted output as a Python list.
         *                   Returns NULL in case of error.
         */
        static inline PyObject* PredictOutput(PyObject* self, PyObject* args);

        /**
         * Calculate the Mean Reciprocal Rank (MRR) using the neural network.
         *
         * This function calculates the MRR using the neural network, datasets,
         * the specified output layer, output index, and NxOutput value.
         *
         * @param self The Python object calling this function (not used).
         * @param args The arguments passed to the function.
         *
         * @return PyObject* The calculated MRR as a Python float.
         *                   Returns NULL in case of error.
         */
        static inline PyObject* CalculateMRR(PyObject* self, PyObject* args);

        /**
         * Save the batch to a file using the neural network.
         *
         * This function saves the batch to a file using the neural network and the specified filename.
         *
         * @param self The Python object calling this function (not used).
         * @param args The arguments passed to the function.
         *
         * @return PyObject* None.
         */
        static inline PyObject* SaveBatch(PyObject* self, PyObject* args);

        /**
         * Dump the batch using the neural network.
         *
         * This function dumps the batch using the neural network and the specified file pointer.
         *
         * @param self The Python object calling this function (not used).
         * @param args The arguments passed to the function.
         *
         * @return PyObject* None.
         */
        static inline PyObject* DumpBatch(PyObject* self, PyObject* args);

        /**
         * Save the layer using the neural network.
         *
         * This function saves the layer using the neural network and the specified filename and layer name.
         *
         * @param self The Python object calling this function (not used).
         * @param args The arguments passed to the function.
         *
         * @return PyObject* None.
         */
        static inline PyObject* SaveLayer(PyObject* self, PyObject* args);

        /**
         * Dump the layer using the neural network.
         *
         * This function dumps the layer using the neural network and the specified file pointer and layer name.
         *
         * @param self The Python object calling this function (not used).
         * @param args The arguments passed to the function.
         *
         * @return PyObject* None.
         */
        static inline PyObject* DumpLayer(PyObject* self, PyObject* args);

        /**
         * Save the weights using the neural network.
         *
         * This function saves the weights using the neural network and the specified filename, input layer, and output layer.
         *
         * @param self The Python object calling this function (not used).
         * @param args The arguments passed to the function.
         *
         * @return PyObject* None.
         */
        static inline PyObject* SaveWeights(PyObject* self, PyObject* args);

        /**
         * Lock the weights in the neural network.
         *
         * This function locks the weights in the neural network for the specified input and output layers.
         *
         * @param self The Python object calling this function (not used).
         * @param args The arguments passed to the function.
         *
         * @return PyObject* None.
         */
        static inline PyObject* LockWeights(PyObject* self, PyObject* args);

        /**
         * Unlock the weights in the neural network.
         *
         * This function unlocks the weights in the neural network for the specified input and output layers.
         *
         * @param self The Python object calling this function (not used).
         * @param args The arguments passed to the function.
         *
         * @return PyObject* None.
         */
        static inline PyObject* UnlockWeights(PyObject* self, PyObject* args);

        /**
         * Save the neural network to a NetCDF file.
         *
         * This function saves the neural network to a NetCDF file with the specified filename.
         *
         * @param self The Python object calling this function (not used).
         * @param args The arguments passed to the function.
         *
         * @return PyObject* None.
         */
        static inline PyObject* SaveNetCDF(PyObject* self, PyObject* args);

        /**
         * Perform P2P broadcast operation in the neural network.
         *
         * This function performs the P2P broadcast operation in the neural network for the specified input layer and output layer.
         *
         * @param self The Python object calling this function (not used).
         * @param args The arguments passed to the function.
         *
         * @return PyObject* None.
         */
        static inline PyObject* P2P_Bcast(PyObject* self, PyObject* args);

        /**
         * Perform P2P allreduce operation in the neural network.
         *
         * This function performs the P2P allreduce operation in the neural network for the specified input layer and output layer.
         *
         * @param self The Python object calling this function (not used).
         * @param args The arguments passed to the function.
         *
         * @return PyObject* None.
         */
        static inline PyObject* P2P_Allreduce(PyObject* self, PyObject* args);

};

/**
 * Clear datasets from the neural network.
 *
 * This function clears the datasets from the neural network.
 *
 * @param self The Python object calling this function (not used).
 * @param args The arguments passed to the function.
 *
 * @return PyObject* None.
 */
PyObject* NetworkFunctions::ClearDataSets(PyObject* self, PyObject* args) {
    auto pNetwork = parsePtr<Network>(args, "neural network");
    if (!pNetwork) return nullptr;
    pNetwork->ClearDataSets();
    Py_RETURN_NONE;
}
  
/**
 * Load datasets into the neural network.
 *
 * This function loads the specified datasets into the neural network for training and validation.
 *
 * @param self The Python object calling this function (not used).
 * @param args The arguments passed to the function.
 *
 * @return PyObject* None.
 */
PyObject* NetworkFunctions::LoadDataSets(PyObject* self, PyObject* args) {
    PyObject* pDataSetBaseList = nullptr; // List of dataset base objects
    auto pNetwork = parsePtrAndOneValue<Network, PyObject*>(args, pDataSetBaseList, "neural network", "OO");
    if (!pNetwork) return nullptr;

    std::vector<DataSetBase*> vDataSetBase = PythonListToDataSetBaseVector(pDataSetBaseList);
    if (vDataSetBase.empty()) {
        PyErr_SetString(PyExc_RuntimeError, "NetworkFunctions::LoadDataSets received empty vDataSetBase vector");
        return nullptr;
    }

    pNetwork->LoadDataSets(vDataSetBase);
    Py_RETURN_NONE;
}
  
/**
 * Randomize the neural network.
 *
 * This function randomizes the neural network by assigning random weights to its layers.
 *
 * @param self The Python object calling this function (not used).
 * @param args The arguments passed to the function.
 *
 * @return PyObject* None.
 */
PyObject* NetworkFunctions::Randomize(PyObject* self, PyObject* args) {
    auto pNetwork = parsePtr<Network>(args, "neural network");
    if (!pNetwork) return nullptr;
    pNetwork->Randomize();
    Py_RETURN_NONE;
}
  
/**
 * Validate the neural network.
 *
 * This function validates the neural network by running the validation process.
 *
 * @param self The Python object calling this function (not used).
 * @param args The arguments passed to the function.
 *
 * @return PyObject* The result of the validation as a Python integer.
 *                   Returns NULL in case of error.
 */
PyObject* NetworkFunctions::Validate(PyObject* self, PyObject* args) {
    auto pNetwork = parsePtr<Network>(args, "neural network");
    if (!pNetwork) return nullptr;
    return Py_BuildValue("i", pNetwork->Validate());
}
  
/**
 * Train the neural network.
 *
 * This function trains the neural network using the specified number of epochs,
 * learning rate alpha, regularization parameters lambda, lambda1, mu, and mu1.
 *
 * @param self The Python object calling this function (not used).
 * @param args The arguments passed to the function.
 *
 * @return PyObject* The final training loss as a Python float.
 *                   Returns NULL in case of error.
 */
PyObject* NetworkFunctions::Train(PyObject* self, PyObject* args) {
    uint32_t epochs = 0; // Number of training epochs
    NNFloat alpha = 0.0; // Learning rate alpha
    NNFloat lambda = 0.0; // Regularization parameter lambda
    NNFloat lambda1 = 0.0; // Regularization parameter lambda1
    NNFloat mu = 0.0; // Regularization parameter mu
    NNFloat mu1 = 0.0; // Regularization parameter mu1
    auto pNetwork = parsePtrAndSixValues<Network, uint32_t, NNFloat, NNFloat, NNFloat, NNFloat, NNFloat>(
        args, epochs, alpha, lambda, lambda1, mu, mu1, "neural network", "OIfffff");
    if (!pNetwork) return nullptr;
    return Py_BuildValue("f", pNetwork->Train(epochs, alpha, lambda, lambda1, mu, mu1));
}
  
/**
 * Predict the batch using the neural network.
 *
 * This function predicts the batch using the given neural network and the number of layers.
 *
 * @param self The Python object calling this function (not used).
 * @param args The arguments passed to the function.
 *
 * @return PyObject* None.
 */
PyObject* NetworkFunctions::PredictBatch(PyObject* self, PyObject* args) {
    uint32_t layers = 0; // Number of layers to predict
    auto pNetwork = parsePtrAndOneValue<Network, uint32_t>(args, layers, "neural network", "OI");
    if (!pNetwork) return nullptr;
    pNetwork->PredictBatch(layers);
    Py_RETURN_NONE;
}

/**
 * Calculate the output using the neural network.
 *
 * This function calculates the output using the given neural network, layer,
 * k value, and GPU buffers for key and value.
 *
 * @param self The Python object calling this function (not used).
 * @param args The arguments passed to the function.
 *
 * @return PyObject* None.
 */
PyObject* NetworkFunctions::CalculateOutput(PyObject* self, PyObject* args) {
    char const* layer = nullptr; // Name of the layer
    uint32_t k = 0; // Value of k
    PyObject* pbKeyCapsule = nullptr; // Capsule holding the GPU buffer for key
    PyObject* pbValueCapsule = nullptr; // Capsule holding the GPU buffer for value
    auto pNetwork = parsePtrAndFourValues<Network, char const*, uint32_t, PyObject*, PyObject*>(
        args, layer, k, pbKeyCapsule, pbValueCapsule, "neural network", "OsIOO");
    if (!pNetwork) return nullptr;

    GpuBuffer<NNFloat>* pbKey = reinterpret_cast<GpuBuffer<NNFloat>*>(PyCapsule_GetPointer(pbKeyCapsule, "float gpu buffer"));
    GpuBuffer<uint32_t>* pbValue = reinterpret_cast<GpuBuffer<uint32_t>*>(PyCapsule_GetPointer(pbValueCapsule, "unsigned gpu buffer"));
    if (!pbKey || !pbValue) return nullptr;

    pNetwork->CalculateOutput(std::string(layer), k, pbKey, pbValue);
    Py_RETURN_NONE;
}

/**
 * Predict the output using the neural network.
 *
 * This function predicts the output using the given neural network, dataset base list,
 * CDL capsule, and k value.
 *
 * @param self The Python object calling this function (not used).
 * @param args The arguments passed to the function.
 *
 * @return PyObject* The result of the prediction as a Python object.
 *                   Returns NULL in case of error.
 */
PyObject* NetworkFunctions::PredictOutput(PyObject* self, PyObject* args) {
    PyObject* pDataSetBaseList = nullptr; // List of dataset base objects
    PyObject* pCdlCapsule = nullptr; // Capsule holding the CDL object
    uint32_t k = 0; // Value of k
    auto pNetwork = parsePtrAndThreeValues<Network, PyObject*, PyObject*, uint32_t>(args, pDataSetBaseList, pCdlCapsule, k, "neural network", "OOOI");
    if (!pNetwork) return nullptr;

    CDL* pCDL = reinterpret_cast<CDL*>(PyCapsule_GetPointer(pCdlCapsule, "cdl"));
    if (!pCDL) return nullptr;

    std::vector<DataSetBase*> vDataSetBase = PythonListToDataSetBaseVector(pDataSetBaseList);
    if (vDataSetBase.empty()) {
        PyErr_SetString(PyExc_RuntimeError, "NetworkFunctions::PredictOutput received empty vDataSetBase vector");
        return nullptr;
    }

    return tensorhubcalculate_PredictOutput(pNetwork, vDataSetBase, *pCDL, k);
}
  
/**
 * Calculate Mean Reciprocal Rank (MRR) using the neural network.
 *
 * This function calculates the Mean Reciprocal Rank (MRR) using the given neural network,
 * dataset base list, output layer, output index, and NxOutput.
 *
 * @param self The Python object calling this function (not used).
 * @param args The arguments passed to the function.
 *
 * @return PyObject* The result of the calculation as a Python object.
 *                   Returns NULL in case of error.
 */
PyObject* NetworkFunctions::CalculateMRR(PyObject* self, PyObject* args) {
    PyObject* pDataSetBaseList = nullptr; // List of dataset base objects
    PyObject* pOutputLayerCapsule = nullptr; // Capsule holding the output layer
    uint32_t outputIndex = 0; // Output index
    uint32_t NxOutput = 0; // NxOutput value
    auto pNetwork = parsePtrAndFourValues<Network, PyObject*, PyObject*, uint32_t, uint32_t>(
        args, pDataSetBaseList, pOutputLayerCapsule, outputIndex, NxOutput, "neural network", "OOOII");
    if (!pNetwork) return nullptr;

    Layer* pOutputLayer = reinterpret_cast<Layer*>(PyCapsule_GetPointer(pOutputLayerCapsule, "layer"));
    if (!pOutputLayer) return nullptr;

    std::vector<DataSetBase*> vDataSetBase = PythonListToDataSetBaseVector(pDataSetBaseList);
    if (vDataSetBase.empty()) {
        PyErr_SetString(PyExc_RuntimeError, "NetworkFunctions::CalculateMRR received empty vDataSetBase vector");
        return nullptr;
    }

    return tensorhubcalculate_CalculateMRR(pNetwork, vDataSetBase, pOutputLayer, outputIndex, NxOutput);
}
  
/**
 * Save the current batch of the neural network to a file.
 *
 * This function saves the current batch of the neural network to a file
 * specified by the file name.
 *
 * @param self The Python object calling this function (not used).
 * @param args The arguments passed to the function.
 *
 * @return PyObject* None.
 */
PyObject* NetworkFunctions::SaveBatch(PyObject* self, PyObject* args) {
    char const* fname = nullptr; // File name to save the batch
    auto pNetwork = parsePtrAndOneValue<Network, char const*>(args, fname, "neural network", "Os");
    if (!pNetwork) return nullptr;
    pNetwork->SaveBatch(std::string(fname));
    Py_RETURN_NONE;
}

/**
 * Dump the current batch of the neural network to a file.
 *
 * This function dumps the current batch of the neural network to a file
 * indicated by the file pointer capsule.
 *
 * @param self The Python object calling this function (not used).
 * @param args The arguments passed to the function.
 *
 * @return PyObject* None.
 */
PyObject* NetworkFunctions::DumpBatch(PyObject* self, PyObject* args) {
    PyObject* fpCapsule = nullptr; // Capsule holding the file pointer
    auto pNetwork = parsePtrAndOneValue<Network, PyObject*>(args, fpCapsule, "neural network", "OO");
    if (!pNetwork) return nullptr;
    std::FILE* fp = reinterpret_cast<std::FILE*>(PyCapsule_GetPointer(fpCapsule, "file pointer"));
    if (!fp) return nullptr;
    pNetwork->DumpBatch(fp);
    Py_RETURN_NONE;
}

/**
 * Save the specified layer of the neural network to a file.
 *
 * This function saves the specified layer of the neural network to a file
 * indicated by the file name.
 *
 * @param self The Python object calling this function (not used).
 * @param args The arguments passed to the function.
 *
 * @return PyObject* None.
 */
PyObject* NetworkFunctions::SaveLayer(PyObject* self, PyObject* args) {
    char const* fname = nullptr;
    char const* layer = nullptr;
    auto pNetwork = parsePtrAndTwoValues<Network, char const*, char const*>(args, fname, layer, "neural network", "Oss");
    if (!pNetwork) return nullptr;
    pNetwork->SaveLayer(std::string(fname), std::string(layer));
    Py_RETURN_NONE;
}

/**
 * Dump the specified layer of the neural network to a file.
 *
 * This function dumps the specified layer of the neural network to a file
 * indicated by the file pointer capsule.
 *
 * @param self The Python object calling this function (not used).
 * @param args The arguments passed to the function.
 *
 * @return PyObject* None.
 */
PyObject* NetworkFunctions::DumpLayer(PyObject* self, PyObject* args) {
    PyObject* fpCapsule = nullptr;
    char const* layer = nullptr;
    auto pNetwork = parsePtrAndTwoValues<Network, PyObject*, char const*>(args, fpCapsule, layer, "neural network", "OOs");
    if (!pNetwork) return nullptr;
    std::FILE* fp = reinterpret_cast<std::FILE*>(PyCapsule_GetPointer(fpCapsule, "file pointer"));
    if (!fp) return nullptr;
    pNetwork->DumpLayer(fp, std::string(layer));
    Py_RETURN_NONE;
}

/**
 * Save the weights between input and output layers to a file.
 *
 * This function saves the weights between the specified input and output layers
 * in the neural network to a file specified by the file name.
 *
 * @param self The Python object calling this function (not used).
 * @param args The arguments passed to the function.
 *
 * @return PyObject* None.
 */
PyObject* NetworkFunctions::SaveWeights(PyObject* self, PyObject* args) {
    char const* fname = nullptr;
    char const* inputLayer = nullptr;
    char const* outputLayer = nullptr;
    auto pNetwork = parsePtrAndThreeValues<Network, char const*, char const*, char const*>(args, fname, inputLayer, outputLayer, "neural network", "Osss");
    if (!pNetwork) return nullptr;
    
    pNetwork->SaveWeights(std::string(fname), std::string(inputLayer), std::string(outputLayer));
    Py_RETURN_NONE;
}

/**
 * Lock the weights between input and output layers.
 *
 * This function locks the weights between the specified input and output layers
 * in the neural network.
 *
 * @param self The Python object calling this function (not used).
 * @param args The arguments passed to the function.
 *
 * @return PyObject* The result of the LockWeights operation as a Python integer.
 *                   Returns NULL in case of error.
 */
PyObject* NetworkFunctions::LockWeights(PyObject* self, PyObject* args) {
    char const* inputLayer = nullptr;
    char const* outputLayer = nullptr;
    auto pNetwork = parsePtrAndTwoValues<Network, char const*, char const*>(args, inputLayer, outputLayer, "neural network", "Oss");
    if (!pNetwork) return nullptr;
    
    return Py_BuildValue("i", pNetwork->LockWeights(std::string(inputLayer), std::string(outputLayer)));
}

/**
 * Unlock the weights between input and output layers.
 *
 * This function unlocks the weights between the specified input and output layers
 * in the neural network.
 *
 * @param self The Python object calling this function (not used).
 * @param args The arguments passed to the function.
 *
 * @return PyObject* The result of the UnlockWeights operation as a Python integer.
 *                   Returns NULL in case of error.
 */
PyObject* NetworkFunctions::UnlockWeights(PyObject* self, PyObject* args) {
    char const* inputLayer = nullptr;
    char const* outputLayer = nullptr;
    auto pNetwork = parsePtrAndTwoValues<Network, char const*, char const*>(args, inputLayer, outputLayer, "neural network", "Oss");
    if (!pNetwork) return nullptr;
    
    return Py_BuildValue("i", pNetwork->UnlockWeights(std::string(inputLayer), std::string(outputLayer)));
}

/**
 * Save the neural network to a NetCDF file.
 *
 * This function saves the given neural network to a NetCDF file specified by the file name.
 *
 * @param self The Python object calling this function (not used).
 * @param args The arguments passed to the function.
 *
 * @return PyObject* The result of the SaveNetCDF operation as a Python integer.
 *                   Returns NULL in case of error.
 */
PyObject* NetworkFunctions::SaveNetCDF(PyObject* self, PyObject* args) {
    char const* fname = nullptr;
    auto pNetwork = parsePtrAndOneValue<Network, char const*>(args, fname, "neural network", "Os");
    if (!pNetwork) return nullptr;
    
    std::string filePath(fname);
    if (!std::filesystem::path(filePath).has_extension()) {
        filePath += ".nc";
    }

    return Py_BuildValue("i", pNetwork->SaveNetCDF(filePath));
}

/**
 * Perform P2P broadcast operation.
 *
 * This function performs a P2P broadcast operation on the given neural network
 * and buffer.
 *
 * @param self The Python object calling this function (not used).
 * @param args The arguments passed to the function.
 *
 * @return PyObject* The result of the broadcast operation as a Python integer.
 *                   Returns NULL in case of error.
 */
PyObject* NetworkFunctions::P2P_Bcast(PyObject* self, PyObject* args) {
    PyObject* capsule = nullptr;
    size_t size = 0;
    auto pNetwork = parsePtrAndTwoValues<Network, PyObject*, size_t>(args, capsule, size, "neural network", "OOI");
    if (!pNetwork) return nullptr;

    auto [oldName, newName] = [capsule]() {
        char const* name = PyCapsule_GetName(capsule);
        if (PyErr_Occurred() != nullptr) return std::make_tuple(nullptr, nullptr);
        if (PyCapsule_SetName(capsule, nullptr) != 0) return std::make_tuple(nullptr, nullptr);
        void* pBuffer = PyCapsule_GetPointer(capsule, nullptr);
        if (PyCapsule_SetName(capsule, name) != 0) return std::make_tuple(nullptr, nullptr);
        return std::make_tuple(name, pBuffer);
    }();

    if (!oldName || !newName) return nullptr;

    return Py_BuildValue("i", pNetwork->P2P_Bcast(newName, size));
}

/**
 * Perform P2P allreduce operation.
 *
 * This function performs a P2P allreduce operation on the given neural network
 * and buffer.
 *
 * @param self The Python object calling this function (not used).
 * @param args The arguments passed to the function.
 *
 * @return PyObject* The result of the allreduce operation as a Python integer.
 *                   Returns NULL in case of error.
 */
PyObject* NetworkFunctions::P2P_Allreduce(PyObject* self, PyObject* args) {
    PyObject* capsule = nullptr;
    size_t size = 0;
    auto pNetwork = parsePtrAndTwoValues<Network, PyObject*, size_t>(args, capsule, size, "neural network", "OOI");
    if (!pNetwork) return nullptr;

    NNFloat* pBuffer = reinterpret_cast<NNFloat*>(PyCapsule_GetPointer(capsule, "float"));
    if (!pBuffer) return nullptr;

    return Py_BuildValue("i", pNetwork->P2P_Allreduce(pBuffer, size));
}

#endif
