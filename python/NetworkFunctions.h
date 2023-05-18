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
     * @brief Clear the datasets in the neural network.
     *
     * @param self Pointer to the Python object.
     * @param args Arguments passed to the function.
     * @return PyObject* Return value of the function.
     */
    static PyObject* ClearDataSets(PyObject* self, PyObject* args);

    /**
     * @brief Load datasets into the neural network.
     *
     * @param self Pointer to the Python object.
     * @param args Arguments passed to the function.
     * @return PyObject* Return value of the function.
     */
    static PyObject* LoadDataSets(PyObject* self, PyObject* args);

    /**
     * @brief Randomize the neural network.
     *
     * @param self Pointer to the Python object.
     * @param args Arguments passed to the function.
     * @return PyObject* Return value of the function.
     */
    static PyObject* Randomize(PyObject* self, PyObject* args);

    /**
     * @brief Validate the neural network.
     *
     * @param self Pointer to the Python object.
     * @param args Arguments passed to the function.
     * @return PyObject* Return value of the function.
     */
    static PyObject* Validate(PyObject* self, PyObject* args);

    /**
     * @brief Train the neural network.
     *
     * @param self Pointer to the Python object.
     * @param args Arguments passed to the function.
     * @return PyObject* Return value of the function.
     */
    static PyObject* Train(PyObject* self, PyObject* args);

    /**
     * @brief Predict batches using the neural network.
     *
     * @param self Pointer to the Python object.
     * @param args Arguments passed to the function.
     * @return PyObject* Return value of the function.
     */
    static PyObject* PredictBatch(PyObject* self, PyObject* args);

    /**
     * @brief Calculate the output using the neural network.
     *
     * @param self Pointer to the Python object.
     * @param args Arguments passed to the function.
     * @return PyObject* Return value of the function.
     */
    static PyObject* CalculateOutput(PyObject* self, PyObject* args);

    /**
     * @brief Predict the output using the neural network.
     *
     * @param self Pointer to the Python object.
     * @param args Arguments passed to the function.
     * @return PyObject* Return value of the function.
     */
    static PyObject* PredictOutput(PyObject* self, PyObject* args);

    /**
     * @brief Calculate the Mean Reciprocal Rank (MRR) using the neural network.
     *
     * @param self Pointer to the Python object.
     * @param args Arguments passed to the function.
     * @return PyObject* Return value of the function.
     */
    static PyObject* CalculateMRR(PyObject* self, PyObject* args);

    /**
     * @brief Save the batch in the neural network.
     *
     * @param self Pointer to the Python object.
     * @param args Arguments passed to the function.
     * @return PyObject* Return value of the function.
     */
    static PyObject* SaveBatch(PyObject* self, PyObject* args);

    /**
     * @brief Dump the batch in the neural network.
     *
     * @param self Pointer to the Python object.
     * @param args Arguments passed to the function.
     * @return PyObject* Return value of the function.
     */
    static PyObject* DumpBatch(PyObject* self, PyObject* args);

    /**
     * @brief Save a layer in the neural network.
     *
     * @param self Pointer to the Python object.
     * @param args Arguments passed to the function.
     * @return PyObject* Return value of the function.
     */
    static PyObject* SaveLayer(PyObject* self, PyObject* args);

    /**
     * @brief Dump a layer in the neural network.
     *
     * @param self Pointer to the Python object.
     * @param args Arguments passed to the function.
     * @return PyObject* Return value of the function.
     */
    static PyObject* DumpLayer(PyObject* self, PyObject* args);

    /**
     * @brief Save the weights in the neural network.
     *
     * @param self Pointer to the Python object.
     * @param args Arguments passed to the function.
     * @return PyObject* Return value of the function.
     */
    static PyObject* SaveWeights(PyObject* self, PyObject* args);

    /**
     * @brief Lock the weights in the neural network.
     *
     * @param self Pointer to the Python object.
     * @param args Arguments passed to the function.
     * @return PyObject* Return value of the function.
     */
    static PyObject* LockWeights(PyObject* self, PyObject* args);

    /**
     * @brief Unlock the weights in the neural network.
     *
     * @param self Pointer to the Python object.
     * @param args Arguments passed to the function.
     * @return PyObject* Return value of the function.
     */
    static PyObject* UnlockWeights(PyObject* self, PyObject* args);

    /**
     * @brief Save the neural network to a NetCDF file.
     *
     * @param self Pointer to the Python object.
     * @param args Arguments passed to the function.
     * @return PyObject* Return value of the function.
     */
    static PyObject* SaveNetCDF(PyObject* self, PyObject* args);

    /**
     * @brief Perform a peer-to-peer broadcast operation on the neural network.
     *
     * @param self Pointer to the Python object.
     * @param args Arguments passed to the function.
     * @return PyObject* Return value of the function.
     */
    static PyObject* P2P_Bcast(PyObject* self, PyObject* args);

    /**
     * @brief Perform a peer-to-peer allreduce operation on the neural network.
     *
     * @param self Pointer to the Python object.
     * @param args Arguments passed to the function.
     * @return PyObject* Return value of the function.
     */
    static PyObject* P2P_Allreduce(PyObject* self, PyObject* args);
};

/**
 * @brief Clears the data sets of the neural network.
 * 
 * @param self The Python object pointer.
 * @param args The Python arguments.
 * @return PyObject* Returns None on success or nullptr on failure.
 */
inline PyObject* NetworkFunctions::ClearDataSets(PyObject* self, PyObject* args)
{
    auto pNetwork = parsePtr<Network>(args, "neural network");
    if (!pNetwork)
        return nullptr;
    pNetwork->ClearDataSets();
    Py_RETURN_NONE;
}

/**
 * @brief Loads data sets into the neural network.
 * 
 * @param self The Python object pointer.
 * @param args The Python arguments.
 * @return PyObject* Returns None on success or nullptr on failure.
 */
inline PyObject* NetworkFunctions::LoadDataSets(PyObject* self, PyObject* args)
{
    PyObject* pDataSetBaseList = nullptr;
    auto pNetwork = parsePtrAndOneValue<Network, PyObject*>(args, pDataSetBaseList, "neural network", "OO");
    if (!pNetwork)
        return nullptr;

    std::vector<DataSetBase*> vDataSetBase = PythonListToDataSetBaseVector(pDataSetBaseList);
    if (vDataSetBase.empty()) {
        PyErr_SetString(PyExc_RuntimeError, "NetworkFunctions::LoadDataSets received empty vDataSetBase vector");
        return nullptr;
    }

    pNetwork->LoadDataSets(vDataSetBase);
    Py_RETURN_NONE;
}

/**
 * @brief Randomizes the neural network.
 * 
 * @param self The Python object pointer.
 * @param args The Python arguments.
 * @return PyObject* Returns None on success or nullptr on failure.
 */
inline PyObject* NetworkFunctions::Randomize(PyObject* self, PyObject* args)
{
    auto pNetwork = parsePtr<Network>(args, "neural network");
    if (!pNetwork)
        return nullptr;
    pNetwork->Randomize();
    Py_RETURN_NONE;
}

/**
 * @brief Validates the neural network.
 * 
 * @param self The Python object pointer.
 * @param args The Python arguments.
 * @return PyObject* Returns an integer representing the validation result on success or nullptr on failure.
 */
inline PyObject* NetworkFunctions::Validate(PyObject* self, PyObject* args)
{
    auto pNetwork = parsePtr<Network>(args, "neural network");
    if (!pNetwork)
        return nullptr;
    return Py_BuildValue("i", pNetwork->Validate());
}

/**
 * @brief Trains the neural network.
 * 
 * @param self The Python object pointer.
 * @param args The Python arguments.
 * @return PyObject* Returns a float representing the training result on success or nullptr on failure.
 */
inline PyObject* NetworkFunctions::Train(PyObject* self, PyObject* args)
{
    uint32_t epochs = 0;
    NNFloat alpha = 0.0;
    NNFloat lambda = 0.0;
    NNFloat lambda1 = 0.0;
    NNFloat mu = 0.0;
    NNFloat mu1 = 0.0;
    auto pNetwork = parsePtrAndSixValues<Network, uint32_t, NNFloat, NNFloat, NNFloat, NNFloat, NNFloat>(
        args, epochs, alpha, lambda, lambda1, mu, mu1, "neural network", "OIfffff");
    if (!pNetwork)
        return nullptr;
    return Py_BuildValue("f", pNetwork->Train(epochs, alpha, lambda, lambda1, mu, mu1));
}

/**
 * @brief Performs batch prediction using the neural network.
 * 
 * @param self The Python object pointer.
 * @param args The Python arguments.
 * @return PyObject* Returns None on success or nullptr on failure.
 */
inline PyObject* NetworkFunctions::PredictBatch(PyObject* self, PyObject* args)
{
    uint32_t layers = 0;
    auto pNetwork = parsePtrAndOneValue<Network, uint32_t>(args, layers, "neural network", "OI");
    if (!pNetwork)
        return nullptr;
    pNetwork->PredictBatch(layers);
    Py_RETURN_NONE;
}

/**
 * @brief Calculates the output of a specific layer in the neural network.
 * 
 * @param self The Python object pointer.
 * @param args The Python arguments.
 * @return PyObject* Returns None on success or nullptr on failure.
 */
inline PyObject* NetworkFunctions::CalculateOutput(PyObject* self, PyObject* args)
{
    const char* layer = nullptr;
    uint32_t k = 0;
    PyObject* pbKeyCapsule = nullptr;
    PyObject* pbValueCapsule = nullptr;
    auto pNetwork = parsePtrAndFourValues<Network, const char*, uint32_t, PyObject*, PyObject*>(
        args, layer, k, pbKeyCapsule, pbValueCapsule, "neural network", "OsIOO");
    if (!pNetwork)
        return nullptr;

    GpuBuffer<NNFloat>* pbKey = reinterpret_cast<GpuBuffer<NNFloat>*>(PyCapsule_GetPointer(pbKeyCapsule, "float gpu buffer"));
    GpuBuffer<uint32_t>* pbValue = reinterpret_cast<GpuBuffer<uint32_t>*>(PyCapsule_GetPointer(pbValueCapsule, "unsigned gpu buffer"));
    if (!pbKey || !pbValue)
        return nullptr;

    pNetwork->CalculateOutput(std::string(layer), k, pbKey, pbValue);
    Py_RETURN_NONE;
}

/**
 * @brief Performs output prediction using the neural network.
 * 
 * @param self The Python object pointer.
 * @param args The Python arguments.
 * @return PyObject* Returns the prediction result on success or nullptr on failure.
 */
inline PyObject* NetworkFunctions::PredictOutput(PyObject* self, PyObject* args)
{
    PyObject* pDataSetBaseList = nullptr;
    PyObject* pCdlCapsule = nullptr;
    uint32_t k = 0;
    auto pNetwork = parsePtrAndThreeValues<Network, PyObject*, PyObject*, uint32_t>(args, pDataSetBaseList, pCdlCapsule, k, "neural network", "OOOI");
    if (!pNetwork)
        return nullptr;

    CDL* pCDL = reinterpret_cast<CDL*>(PyCapsule_GetPointer(pCdlCapsule, "cdl"));
    if (!pCDL)
        return nullptr;

    std::vector<DataSetBase*> vDataSetBase = PythonListToDataSetBaseVector(pDataSetBaseList);
    if (vDataSetBase.empty()) {
        PyErr_SetString(PyExc_RuntimeError, "NetworkFunctions::PredictOutput received empty vDataSetBase vector");
        return nullptr;
    }

    return tensorhubcalculate_PredictOutput(pNetwork, vDataSetBase, *pCDL, k);
}

/**
 * @brief Calculates the Mean Reciprocal Rank (MRR) using the neural network.
 * 
 * @param self The Python object pointer.
 * @param args The Python arguments.
 * @return PyObject* Returns the MRR result on success or nullptr on failure.
 */
inline PyObject* NetworkFunctions::CalculateMRR(PyObject* self, PyObject* args)
{
    PyObject* pDataSetBaseList = nullptr;
    PyObject* pOutputLayerCapsule = nullptr;
    uint32_t outputIndex = 0;
    uint32_t NxOutput = 0;
    auto pNetwork = parsePtrAndFourValues<Network, PyObject*, PyObject*, uint32_t, uint32_t>(
        args, pDataSetBaseList, pOutputLayerCapsule, outputIndex, NxOutput, "neural network", "OOOII");
    if (!pNetwork)
        return nullptr;

    Layer* pOutputLayer = reinterpret_cast<Layer*>(PyCapsule_GetPointer(pOutputLayerCapsule, "layer"));
    if (!pOutputLayer)
        return nullptr;

    std::vector<DataSetBase*> vDataSetBase = PythonListToDataSetBaseVector(pDataSetBaseList);
    if (vDataSetBase.empty()) {
        PyErr_SetString(PyExc_RuntimeError, "NetworkFunctions::CalculateMRR received empty vDataSetBase vector");
        return nullptr;
    }

    return tensorhubcalculate_CalculateMRR(pNetwork, vDataSetBase, pOutputLayer, outputIndex, NxOutput);
}

/**
 * @brief Saves the neural network to a batch file.
 * 
 * @param self The Python object pointer.
 * @param args The Python arguments.
 * @return PyObject* Returns None on success or nullptr on failure.
 */
inline PyObject* NetworkFunctions::SaveBatch(PyObject* self, PyObject* args)
{
    const char* fname = nullptr;
    auto pNetwork = parsePtrAndOneValue<Network, const char*>(args, fname, "neural network", "Os");
    if (!pNetwork)
        return nullptr;
    pNetwork->SaveBatch(std::string(fname));
    Py_RETURN_NONE;
}

/**
 * @brief Dumps the neural network to a file.
 * 
 * @param self The Python object pointer.
 * @param args The Python arguments.
 * @return PyObject* Returns None on success or nullptr on failure.
 */
inline PyObject* NetworkFunctions::DumpBatch(PyObject* self, PyObject* args)
{
    PyObject* fpCapsule = nullptr;
    auto pNetwork = parsePtrAndOneValue<Network, PyObject*>(args, fpCapsule, "neural network", "OO");
    if (!pNetwork)
        return nullptr;
    std::FILE* fp = reinterpret_cast<std::FILE*>(PyCapsule_GetPointer(fpCapsule, "file pointer"));
    if (!fp)
        return nullptr;
    pNetwork->DumpBatch(fp);
    Py_RETURN_NONE;
}

/**
 * @brief Saves a specific layer of the neural network to a file.
 * 
 * @param self The Python object pointer.
 * @param args The Python arguments.
 * @return PyObject* Returns None on success or nullptr on failure.
 */
inline PyObject* NetworkFunctions::SaveLayer(PyObject* self, PyObject* args)
{
    const char* fname = nullptr;
    const char* layer = nullptr;
    auto pNetwork = parsePtrAndTwoValues<Network, const char*, const char*>(args, fname, layer, "neural network", "Oss");
    if (!pNetwork)
        return nullptr;
    pNetwork->SaveLayer(std::string(fname), std::string(layer));
    Py_RETURN_NONE;
}

/**
 * @brief Dumps a specific layer of the neural network to a file.
 * 
 * @param self The Python object pointer.
 * @param args The Python arguments.
 * @return PyObject* Returns None on success or nullptr on failure.
 */
inline PyObject* NetworkFunctions::DumpLayer(PyObject* self, PyObject* args)
{
    PyObject* fpCapsule = nullptr;
    const char* layer = nullptr;
    auto pNetwork = parsePtrAndTwoValues<Network, PyObject*, const char*>(args, fpCapsule, layer, "neural network", "OOs");
    if (!pNetwork)
        return nullptr;
    std::FILE* fp = reinterpret_cast<std::FILE*>(PyCapsule_GetPointer(fpCapsule, "file pointer"));
    if (!fp)
        return nullptr;
    pNetwork->DumpLayer(fp, std::string(layer));
    Py_RETURN_NONE;
}

/**
 * @brief Saves the weights of the neural network to a file.
 * 
 * @param self The Python object pointer.
 * @param args The Python arguments.
 * @return PyObject* Returns None on success or nullptr on failure.
 */
inline PyObject* NetworkFunctions::SaveWeights(PyObject* self, PyObject* args)
{
    const char* fname = nullptr;
    const char* inputLayer = nullptr;
    const char* outputLayer = nullptr;
    auto pNetwork = parsePtrAndThreeValues<Network, const char*, const char*, const char*>(args, fname, inputLayer, outputLayer, "neural network", "Osss");
    if (!pNetwork)
        return nullptr;

    pNetwork->SaveWeights(std::string(fname), std::string(inputLayer), std::string(outputLayer));
    Py_RETURN_NONE;
}

/**
 * @brief Locks the weights between two layers in the neural network.
 * 
 * @param self The Python object pointer.
 * @param args The Python arguments.
 * @return PyObject* Returns an integer representing the lock status on success or nullptr on failure.
 */
inline PyObject* NetworkFunctions::LockWeights(PyObject* self, PyObject* args)
{
    const char* inputLayer = nullptr;
    const char* outputLayer = nullptr;
    auto pNetwork = parsePtrAndTwoValues<Network, const char*, const char*>(args, inputLayer, outputLayer, "neural network", "Oss");
    if (!pNetwork)
        return nullptr;

    return Py_BuildValue("i", pNetwork->LockWeights(std::string(inputLayer), std::string(outputLayer)));
}

/**
 * @brief Unlocks the weights between two layers in the neural network.
 * 
 * @param self The Python object pointer.
 * @param args The Python arguments.
 * @return PyObject* Returns an integer representing the lock status on success or nullptr on failure.
 */
inline PyObject* NetworkFunctions::UnlockWeights(PyObject* self, PyObject* args)
{
    const char* inputLayer = nullptr;
    const char* outputLayer = nullptr;
    auto pNetwork = parsePtrAndTwoValues<Network, const char*, const char*>(args, inputLayer, outputLayer, "neural network", "Oss");
    if (!pNetwork)
        return nullptr;

    return Py_BuildValue("i", pNetwork->UnlockWeights(std::string(inputLayer), std::string(outputLayer)));
}

/**
 * @brief Saves the neural network to a NetCDF file.
 * 
 * @param self The Python object pointer.
 * @param args The Python arguments.
 * @return PyObject* Returns an integer representing the save status on success or nullptr on failure.
 */
inline PyObject* NetworkFunctions::SaveNetCDF(PyObject* self, PyObject* args)
{
    const char* fname = nullptr;
    auto pNetwork = parsePtrAndOneValue<Network, const char*>(args, fname, "neural network", "Os");
    if (!pNetwork)
        return nullptr;

    std::string filePath(fname);
    if (!std::filesystem::path(filePath).has_extension()) {
        filePath += ".nc";
    }

    return Py_BuildValue("i", pNetwork->SaveNetCDF(filePath));
}

/**
 * @brief Performs a peer-to-peer broadcast operation on the neural network.
 * 
 * @param self The Python object pointer.
 * @param args The Python arguments.
 * @return PyObject* Returns an integer representing the operation status on success or nullptr on failure.
 */
inline PyObject* NetworkFunctions::P2P_Bcast(PyObject* self, PyObject* args)
{
    PyObject* capsule = nullptr;
    size_t size = 0;
    auto pNetwork = parsePtrAndTwoValues<Network, PyObject*, size_t>(args, capsule, size, "neural network", "OOI");
    if (!pNetwork)
        return nullptr;

    auto [oldName, newName] = [capsule]() {
        const char* name = PyCapsule_GetName(capsule);
        if (PyErr_Occurred() != nullptr)
            return std::make_tuple(nullptr, nullptr);
        if (PyCapsule_SetName(capsule, nullptr) != 0)
            return std::make_tuple(nullptr, nullptr);
        void* pBuffer = PyCapsule_GetPointer(capsule, nullptr);
        if (PyCapsule_SetName(capsule, name) != 0)
            return std::make_tuple(nullptr, nullptr);
        return std::make_tuple(name, pBuffer);
    }();

    if (!oldName || !newName)
        return nullptr;

    return Py_BuildValue("i", pNetwork->P2P_Bcast(newName, size));
}

/**
 * @brief Performs a peer-to-peer allreduce operation on the neural network.
 * 
 * @param self The Python object pointer.
 * @param args The Python arguments.
 * @return PyObject* Returns an integer representing the operation status on success or nullptr on failure.
 */
inline PyObject* NetworkFunctions::P2P_Allreduce(PyObject* self, PyObject* args)
{
    PyObject* capsule = nullptr;
    size_t size = 0;
    auto pNetwork = parsePtrAndTwoValues<Network, PyObject*, size_t>(args, capsule, size, "neural network", "OOI");
    if (!pNetwork)
        return nullptr;

    NNFloat* pBuffer = reinterpret_cast<NNFloat*>(PyCapsule_GetPointer(capsule, "float"));
    if (!pBuffer)
        return nullptr;

    return Py_BuildValue("i", pNetwork->P2P_Allreduce(pBuffer, size));
}

#endif
