#ifndef __UTILITIES_H__
#define __UTILITIES_H__

#include <optional>
#include <tuple>
#include <vector>
#include <cstdint>
#include <stdexcept>

class Utilities {
    public:
        /**
         * @brief Startup the GPU.
         *
         * This function starts up the GPU by calling the `Startup` function of the GPU object with the provided command-line arguments.
         *
         * @param self The reference to the Utilities object.
         * @param args The arguments passed to the function.
         * @return Py_None indicating the successful startup of the GPU.
         *         Returns nullptr if an error occurs during parsing or startup.
         */
        static inline PyObject* Startup(PyObject* self, PyObject* args);

        /**
         * @brief Shutdown the GPU.
         *
         * This function shuts down the GPU by calling the `Shutdown` function of the GPU object.
         *
         * @param self The reference to the Utilities object.
         * @param args The arguments passed to the function.
         * @return Py_None indicating the successful shutdown of the GPU.
         */
        static inline PyObject* Shutdown(PyObject* self, PyObject* args);

        /**
         * @brief Create a CDL object from a JSON file.
         *
         * This function creates a CDL object and initializes it by loading the contents from a JSON file specified by `jsonFilename`.
         *
         * @param self The reference to the Utilities object.
         * @param args The arguments passed to the function.
         * @return A PyCapsule object representing the created CDL object.
         *         Returns nullptr if an error occurs during loading or initialization.
         */
        static inline PyObject* CreateCDLFromJSON(PyObject* self, PyObject* args);

        /**
         * @brief Create a CDL object with default values.
         *
         * This function creates a CDL object with default values and returns it as a PyCapsule object.
         *
         * @param self The reference to the Utilities object.
         * @param args The arguments passed to the function.
         * @return A PyCapsule object representing the created CDL object.
         */
        static inline PyObject* CreateCDLFromDefaults(PyObject* self, PyObject* args);

        /**
         * @brief Delete a CDL object.
         *
         * This function deletes the CDL object represented by the given pointer `pCDL`.
         *
         * @param self The reference to the Utilities object.
         * @param args The arguments passed to the function.
         * @return Py_None indicating successful deletion of the CDL object.
         *         Returns nullptr if the CDL pointer is null.
         */
        static inline PyObject* DeleteCDL(PyObject* self, PyObject* args);

        /**
         * @brief Load a data set from a NetCDF file.
         *
         * This function loads a data set from a NetCDF file specified by `dataFilename` and returns it as a list of DataSetBase objects.
         *
         * @param self The reference to the Utilities object.
         * @param args The arguments passed to the function.
         * @return A Python list object containing the loaded data set as DataSetBase objects.
         *         Returns nullptr if an error occurs during loading or if there is insufficient memory.
         */
        static inline PyObject* LoadDataSetFromNetCDF(PyObject* self, PyObject* args);

        /**
         * @brief Delete a data set object.
         *
         * This function deletes the data set object represented by the given pointer `pDataSet`.
         *
         * @param self The reference to the Utilities object.
         * @param args The arguments passed to the function.
         * @return Py_None indicating successful deletion of the data set object.
         *         Returns nullptr if the data set pointer is null.
         */
        static inline PyObject* DeleteDataSet(PyObject* self, PyObject* args);

        /**
         * @brief Load a neural network from a NetCDF file.
         *
         * This function loads a neural network from a NetCDF file specified by `networkFilename` and returns it as a PyCapsule object.
         *
         * @param self The reference to the Utilities object.
         * @param args The arguments passed to the function.
         * @return A PyCapsule object representing the loaded neural network.
         *         Returns nullptr if an error occurs during loading or if the loaded network is nullptr.
         */
        static inline PyObject* LoadNeuralNetworkFromNetCDF(PyObject* self, PyObject* args);

        /**
         * @brief Load a neural network from a JSON file.
         *
         * This function loads a neural network from a JSON file specified by `jsonFilename` and returns it as a PyCapsule object.
         *
         * @param self The reference to the Utilities object.
         * @param args The arguments passed to the function.
         * @return A PyCapsule object representing the loaded neural network.
         *         Returns nullptr if an error occurs during loading or if the loaded network is nullptr.
         */
        static inline PyObject* LoadNeuralNetworkFromJSON(PyObject* self, PyObject* args);

        /**
         * @brief Delete a neural network object.
         *
         * This function deletes the neural network object represented by the given pointer `pNetwork`.
         *
         * @param self The reference to the Utilities object.
         * @param args The arguments passed to the function.
         * @return Py_None indicating successful deletion of the neural network object.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* DeleteNetwork(PyObject* self, PyObject* args);

        /**
         * @brief Open a file.
         *
         * This function opens a file specified by `filename` with the given `mode` and returns a PyCapsule object representing the opened file.
         *
         * @param self The reference to the Utilities object.
         * @param args The arguments passed to the function.
         * @return A PyCapsule object representing the opened file.
         *         Returns nullptr if an error occurs during opening or if the file pointer is null.
         */
        static inline PyObject* OpenFile(PyObject* self, PyObject* args);

        /**
         * @brief Close a file.
         *
         * This function closes the file represented by the given pointer `pFILE`.
         *
         * @param self The reference to the Utilities object.
         * @param args The arguments passed to the function.
         * @return Py_None indicating successful closing of the file.
         *         Returns nullptr if the file pointer is null or if an error occurs during closing.
         */
        static inline PyObject* CloseFile(PyObject* self, PyObject* args);

        /**
         * @brief Set the random seed for the GPU.
         *
         * This function sets the random seed for the GPU to ensure reproducible random number generation.
         *
         * @param self The reference to the Utilities object.
         * @param args The arguments passed to the function.
         * @return Py_None indicating successful setting of the random seed.
         */
        static inline PyObject* SetRandomSeed(PyObject* self, PyObject* args);

        /**
         * @brief Get the memory usage of the GPU and CPU.
         *
         * This function retrieves the memory usage of the GPU and CPU and returns it as a list of two integers: [gpuMemoryUsage, cpuMemoryUsage].
         *
         * @param self The reference to the Utilities object.
         * @param args The arguments passed to the function.
         * @return A Python list object containing the GPU and CPU memory usage as two integers.
         */
        static inline PyObject* GetMemoryUsage(PyObject* self, PyObject* args);

        /**
         * @brief Normalize a numpy array.
         *
         * This function normalizes a numpy array specified by `inputArray` using the given `mean` and `stdDev`.
         *
         * @param self The reference to the Utilities object.
         * @param args The arguments passed to the function.
         * @return A normalized numpy array.
         *         Returns nullptr if an error occurs during normalization or if the input array is invalid.
         */
        static inline PyObject* Normalize(PyObject* self, PyObject* args);

        /**
         * @brief Transpose a numpy array.
         *
         * This function transposes a numpy array specified by `ASINWeightArray` and `pEmbeddingArray`.
         *
         * @param self The reference to the Utilities object.
         * @param args The arguments passed to the function.
         * @return The transposed numpy array.
         *         Returns nullptr if an error occurs during transposition or if the input arrays are invalid.
         */
        static inline PyObject* Transpose(PyObject* self, PyObject* args);

        /**
         * @brief Create a float GPU buffer.
         *
         * This function creates a float GPU buffer of the specified `size` and returns it as a PyCapsule object.
         *
         * @param self The reference to the Utilities object.
         * @param args The arguments passed to the function.
         * @return A PyCapsule object representing the created float GPU buffer.
         *         Returns nullptr if an error occurs during creation or if there is insufficient memory.
         */
        static inline PyObject* CreateFloatGpuBuffer(PyObject* self, PyObject* args);

        /**
         * @brief Delete a float GPU buffer.
         *
         * This function deletes the float GPU buffer represented by the given pointer `pGpuBuffer`.
         *
         * @param self The reference to the Utilities object.
         * @param args The arguments passed to the function.
         * @return Py_None indicating successful deletion of the float GPU buffer.
         *         Returns nullptr if the float GPU buffer pointer is null.
         */
        static inline PyObject* DeleteFloatGpuBuffer(PyObject* self, PyObject* args);

        /**
         * @brief Create an unsigned GPU buffer.
         *
         * This function creates an unsigned GPU buffer of the specified `size` and returns it as a PyCapsule object.
         *
         * @param self The reference to the Utilities object.
         * @param args The arguments passed to the function.
         * @return A PyCapsule object representing the created unsigned GPU buffer.
         *         Returns nullptr if an error occurs during creation or if there is insufficient memory.
         */
        static inline PyObject* CreateUnsignedGpuBuffer(PyObject* self, PyObject* args);

        /**
         * @brief Delete an unsigned GPU buffer.
         *
         * This function deletes the unsigned GPU buffer represented by the given pointer `pGpuBuffer`.
         *
         * @param self The reference to the Utilities object.
         * @param args The arguments passed to the function.
         * @return Py_None indicating successful deletion of the unsigned GPU buffer.
         *         Returns nullptr if the unsigned GPU buffer pointer is null.
         */
        static inline PyObject* DeleteUnsignedGpuBuffer(PyObject* self, PyObject* args);

};

/**
 * @brief Startup the GPU.
 *
 * This function starts up the GPU by calling the `Startup` function of the GPU object with the provided command-line arguments.
 *
 * @param self The reference to the Utilities object.
 * @param args The arguments passed to the function.
 * @return Py_None indicating the successful startup of the GPU.
 *         Returns nullptr if an error occurs during parsing or startup.
 */
PyObject* Utilities::Startup(PyObject* self, PyObject* args) {
    PyObject* arglist = nullptr;
    if (!PyArg_ParseTuple(args, "O", &arglist))
        return nullptr;

    int argc = PyList_Size(arglist);
    std::vector<std::string_view> argv;
    argv.reserve(argc);
    for (PyObject* item : arglist) {
        if (!PyUnicode_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "Argument list must contain string items");
            return nullptr;
        }
        argv.emplace_back(PyUnicode_AsUTF8(item));
    }

    getGpu().Startup(argc, argv.data());

    Py_RETURN_NONE;
}

/**
 * @brief Shutdown the GPU.
 *
 * This function shuts down the GPU by calling the `Shutdown` function of the GPU object.
 *
 * @param self The reference to the Utilities object.
 * @param args The arguments passed to the function.
 * @return Py_None indicating the successful shutdown of the GPU.
 */
PyObject* Utilities::Shutdown(PyObject* self, PyObject* args) {
    getGpu().Shutdown();
    Py_RETURN_NONE;
}

/**
 * @brief Create a CDL object from a JSON file.
 *
 * This function creates a CDL object and initializes it by loading the contents from a JSON file specified by `jsonFilename`.
 *
 * @param self The reference to the Utilities object.
 * @param args The arguments passed to the function.
 * @return A PyCapsule object representing the created CDL object.
 *         Returns nullptr if an error occurs during loading or initialization.
 */
PyObject* Utilities::CreateCDLFromJSON(PyObject* self, PyObject* args) {
    std::string_view jsonFilename;
    if (!PyArg_ParseTuple(args, "s", &jsonFilename))
        return nullptr;

    std::unique_ptr<CDL> pCDL = std::make_unique<CDL>();
    if (pCDL->Load_JSON(std::string(jsonFilename)) != 0) {
        std::string message = "Load_JSON could not parse JSON file: " + std::string(jsonFilename);
        PyErr_SetString(PyExc_RuntimeError, message.c_str());
        return nullptr;
    }

    return PyCapsule_New(pCDL.release(), "cdl", nullptr);
}

/**
 * @brief Create a CDL object with default values.
 *
 * This function creates a CDL object with default values and returns it as a PyCapsule object.
 *
 * @param self The reference to the Utilities object.
 * @param args The arguments passed to the function.
 * @return A PyCapsule object representing the created CDL object.
 */
PyObject* Utilities::CreateCDLFromDefaults(PyObject* self, PyObject* args) {
    std::unique_ptr<CDL> pCDL = std::make_unique<CDL>();
    return PyCapsule_New(pCDL.release(), "cdl", nullptr);
}

/**
 * @brief Delete a CDL object.
 *
 * This function deletes the CDL object represented by the given pointer `pCDL`.
 *
 * @param self The reference to the Utilities object.
 * @param args The arguments passed to the function.
 * @return Py_None indicating successful deletion of the CDL object.
 *         Returns nullptr if the CDL pointer is null.
 */
PyObject* Utilities::DeleteCDL(PyObject* self, PyObject* args) {
    std::unique_ptr<CDL> pCDL = parsePtr<CDL*>(args, "cdl");
    if (pCDL == nullptr)
        return nullptr;

    return Py_None;
}

/**
 * @brief Load a data set from a NetCDF file.
 *
 * This function loads a data set from a NetCDF file specified by `dataFilename` and returns it as a list of DataSetBase objects.
 *
 * @param self The reference to the Utilities object.
 * @param args The arguments passed to the function.
 * @return A Python list object containing the loaded data set as DataSetBase objects.
 *         Returns nullptr if an error occurs during loading or if there is insufficient memory.
 */  
PyObject* Utilities::LoadDataSetFromNetCDF(PyObject* self, PyObject* args) {
    std::string_view dataFilename;
    if (!PyArg_ParseTuple(args, "s", &dataFilename))
        return nullptr;

    std::vector<DataSetBase*> vDataSetBase = LoadNetCDF(std::string(dataFilename));
    if (vDataSetBase.empty()) {
        PyErr_NoMemory();
        return nullptr;
    }

    return DataSetBaseVectorToPythonList(vDataSetBase);
}

/**
 * @brief Delete a data set.
 *
 * This function deletes the data set object represented by the given pointer `pDataSet`.
 *
 * @param self The reference to the Utilities object.
 * @param args The arguments passed to the function.
 * @return Py_None indicating successful deletion of the data set.
 *         Returns nullptr if the data set pointer is null.
 */
PyObject* Utilities::DeleteDataSet(PyObject* self, PyObject* args) {
    std::unique_ptr<DataSetBase> pDataSet = parsePtr<DataSetBase*>(args, "data set");
    if (pDataSet == nullptr)
        return nullptr;

    return Py_None;
}

/**
 * @brief Load a neural network from a NetCDF file.
 *
 * This function loads a neural network from a NetCDF file specified by `networkFilename` and initializes it with the given `batch` size.
 *
 * @param self The reference to the Utilities object.
 * @param args The arguments passed to the function.
 * @return A PyCapsule object representing the loaded neural network as a Network pointer.
 *         Returns nullptr if an error occurs during loading or initialization.
 */
PyObject* Utilities::LoadNeuralNetworkFromNetCDF(PyObject* self, PyObject* args) {
    char const* networkFilename = nullptr;
    uint32_t batch = 0;
    if (!PyArg_ParseTuple(args, "sI", &networkFilename, &batch))
        return nullptr;

    Network* pNetwork = LoadNeuralNetworkNetCDF(std::string(networkFilename), batch);
    if (pNetwork == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "Utilities::LoadNeuralNetworkFromNetCDF received nullptr result from LoadNeuralNetworkNetCDF");
        return nullptr;
    }

    return PyCapsule_New(pNetwork, "neural network", nullptr);
}

/**
 * @brief Load a neural network from JSON.
 *
 * This function loads a neural network from a JSON file specified by `jsonFilename` and initializes it with the given `batch` size and `vDataSetBase` vector.
 * The `vDataSetBase` vector contains a list of DataSetBase objects.
 *
 * @param self The reference to the Utilities object.
 * @param args The arguments passed to the function.
 * @return A PyCapsule object representing the loaded neural network as a Network pointer.
 *         Returns nullptr if an error occurs during loading or initialization.
 */
PyObject* Utilities::LoadNeuralNetworkFromJSON(PyObject* self, PyObject* args) {
    std::string_view jsonFilename;
    uint32_t batch;
    PyObject* pDataSetBaseList = nullptr;
    if (!PyArg_ParseTuple(args, "sIO", &jsonFilename, &batch, &pDataSetBaseList))
        return nullptr;

    std::vector<DataSetBase*> vDataSetBase = PythonListToDataSetBaseVector(pDataSetBaseList);
    if (vDataSetBase.empty()) {
        PyErr_SetString(PyExc_ValueError, "Utilities::LoadNeuralNetworkFromJSON received empty vDataSetBase");
        return nullptr;
    }

    Network* pNetwork = LoadNeuralNetworkJSON(std::string(jsonFilename), batch, vDataSetBase);
    if (pNetwork == nullptr) {
        PyErr_SetString(PyExc_IOError, "Utilities::LoadNeuralNetworkFromJSON received nullptr result from LoadNeuralNetworkNetCDF");
        return nullptr;
    }

    return PyCapsule_New(pNetwork, "neural network", nullptr);
}

/**
 * @brief Delete a neural network.
 *
 * This function deletes the neural network object represented by the given pointer `pNetwork`.
 *
 * @param self The reference to the Utilities object.
 * @param args The arguments passed to the function.
 * @return Py_None indicating successful deletion of the neural network.
 *         Returns nullptr if the neural network pointer is null.
 */
PyObject* Utilities::DeleteNetwork(PyObject* self, PyObject* args) {
    std::unique_ptr<Network> pNetwork = parsePtr<Network*>(args, "neural network");
    if (pNetwork == nullptr)
        return nullptr;

    return Py_None;
}
  
/**
 * @brief Open a file.
 *
 * This function opens a file with the specified filename and mode.
 *
 * @param self The reference to the Utilities object.
 * @param args The arguments passed to the function.
 * @return A PyCapsule object representing the opened file as a FILE pointer.
 *         Returns nullptr if an error occurs during file open.
 */
PyObject* Utilities::OpenFile(PyObject* self, PyObject* args) {
    const char* filename = nullptr;
    const char* mode = nullptr;
    if (!PyArg_ParseTuple(args, "ss", &filename, &mode))
        return nullptr;

    FILE* pFILE = fopen(filename, mode);
    if (pFILE == nullptr) {
        PyErr_SetFromErrno(PyExc_IOError);
        return nullptr;
    }

    return PyCapsule_New(pFILE, "file", nullptr);
}

/**
 * @brief Close a file.
 *
 * This function closes a file represented by the given FILE pointer `pFILE`.
 *
 * @param self The reference to the Utilities object.
 * @param args The arguments passed to the function.
 * @return Py_None indicating successful file close.
 *         Returns nullptr if the FILE pointer is null or an error occurs during file close.
 */
PyObject* Utilities::CloseFile(PyObject* self, PyObject* args) {
    FILE* pFILE = parsePtr<FILE*>(args, "file");
    if (pFILE == nullptr)
        return nullptr;

    if (fclose(pFILE) != 0) {
        PyErr_SetString(PyExc_IOError, "File close error");
        return nullptr;
    }

    return Py_None;
}
  
/**
 * @brief Set the random seed for the GPU.
 *
 * This function sets the random seed for the GPU by extracting the random seed value from the Python argument `randomSeedObj`
 * and calling `getGpu().SetRandomSeed()` with the converted unsigned long value.
 *
 * @param self The reference to the Utilities object.
 * @param args The arguments passed to the function.
 * @return Py_None indicating successful setting of the random seed.
 *         Returns nullptr if an error occurs during argument parsing or conversion.
 */
PyObject* Utilities::SetRandomSeed(PyObject* self, PyObject* args) {
    PyObject* randomSeedObj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &randomSeedObj))
        return nullptr;

    unsigned long randomSeed = PyLong_AsUnsignedLong(randomSeedObj);
    if (randomSeed == static_cast<unsigned long>(-1) && PyErr_Occurred())
        return nullptr;

    getGpu().SetRandomSeed(randomSeed);
    Py_RETURN_NONE;
}

/**
 * @brief Get the memory usage of the GPU and CPU.
 *
 * This function retrieves the memory usage of the GPU and CPU using the `getGpu().GetMemoryUsage()` function.
 * The memory usage values are returned as a Python list containing two long integers representing the GPU and CPU memory usage, respectively.
 *
 * @param self The reference to the Utilities object.
 * @param args The arguments passed to the function.
 * @return A Python list containing two long integers representing the GPU and CPU memory usage, respectively.
 *         Returns nullptr if an error occurs.
 */
PyObject* Utilities::GetMemoryUsage(PyObject* self, PyObject* args) {
    auto [gpuMemoryUsage, cpuMemoryUsage] = getGpu().GetMemoryUsage();
    return Py_BuildValue("[NN]", PyLong_FromLong(gpuMemoryUsage), PyLong_FromLong(cpuMemoryUsage));
}

/**
 * @brief Transpose the given arrays.
 *
 * This function transposes the given arrays, `ASINWeightArray` and `pEmbeddingArray`, using the `tensorhubcalculate_Transpose` function.
 * The transposed result is returned as a PyObject.
 *
 * @param self The reference to the Utilities object.
 * @param args The arguments passed to the function.
 * @return A PyObject representing the transposed result.
 *         Returns nullptr if an error occurs during the transposition.
 */
PyObject* Utilities::Transpose(PyObject* self, PyObject* args) {
    PyArrayObject* ASINWeightArray = nullptr;
    PyArrayObject* pEmbeddingArray = nullptr;

    if (!PyArg_ParseTuple(args, "OO", &ASINWeightArray, &pEmbeddingArray))
        return nullptr;

    try {
        auto result = std::unique_ptr<PyObject, decltype(&Py_DECREF)>(tensorhubcalculate_Transpose(ASINWeightArray, pEmbeddingArray), &Py_DECREF);
        return result.release();
    } catch (...) {
        return nullptr;
    }
}

/**
 * @brief Create a float GPU buffer.
 *
 * This function creates a float GPU buffer of the specified size using std::make_unique
 * and returns a PyCapsule object containing the buffer.
 *
 * @param self The reference to the Utilities object.
 * @param args The arguments passed to the function.
 * @return A PyCapsule object containing the float GPU buffer.
 *         Returns nullptr if an error occurs or memory allocation fails.
 */
PyObject* Utilities::CreateFloatGpuBuffer(PyObject* self, PyObject* args) {
    uint32_t size = 0;
    if (!PyArg_ParseTuple(args, "I", &size))
        return nullptr;

    try {
        auto pGpuBuffer = std::make_unique<GpuBuffer<NNFloat>>(size, true);
        return PyCapsule_New(static_cast<void*>(pGpuBuffer.release()), "float gpu buffer", nullptr);
    } catch (...) {
        PyErr_SetNone(PyExc_MemoryError);
        return nullptr;
    }
}
  
/**
 * @brief Delete a float GPU buffer.
 *
 * This function deletes a float GPU buffer by taking ownership of the buffer using std::unique_ptr
 * and then resetting the pointer to nullptr.
 *
 * @param self The reference to the Utilities object.
 * @param args The arguments passed to the function.
 * @return Py_None indicating successful deletion of the GPU buffer.
 */
PyObject* Utilities::DeleteFloatGpuBuffer(PyObject* self, PyObject* args) {
    std::unique_ptr<GpuBuffer<NNFloat>> pGpuBuffer = parsePtr<GpuBuffer<NNFloat>*>(args, "float gpu buffer");
    pGpuBuffer.reset();
    return Py_None;
}
  
/**
 * @brief Create an unsigned GPU buffer.
 *
 * This function creates an unsigned GPU buffer of the specified size using std::make_unique
 * and returns a PyCapsule object containing the buffer.
 *
 * @param self The reference to the Utilities object.
 * @param args The arguments passed to the function.
 * @return A PyCapsule object containing the unsigned GPU buffer.
 *         Returns nullptr if an error occurs or memory allocation fails.
 */
PyObject* Utilities::CreateUnsignedGpuBuffer(PyObject* self, PyObject* args) {
    uint32_t size = 0;
    if (!PyArg_ParseTuple(args, "I", &size))
        return nullptr;

    try {
        auto pGpuBuffer = std::make_unique<GpuBuffer<uint32_t>>(size, true);
        return PyObjectPtr(PyCapsule_New(pGpuBuffer.release(), "unsigned gpu buffer", nullptr));
    } catch (...) {
        PyErr_SetNone(PyExc_MemoryError);
        return nullptr;
    }
}
  
/**
 * @brief Delete an unsigned GPU buffer.
 *
 * This function deletes an unsigned GPU buffer by taking ownership of the buffer using std::unique_ptr
 * and then resetting the pointer to nullptr.
 *
 * @param self The reference to the Utilities object.
 * @param args The arguments passed to the function.
 * @return Py_None indicating successful deletion of the GPU buffer.
 */
PyObject* Utilities::DeleteUnsignedGpuBuffer(PyObject* self, PyObject* args) {
    std::unique_ptr<GpuBuffer<uint32_t>> pGpuBuffer = parsePtr<GpuBuffer<uint32_t>*>(args, "unsigned gpu buffer");
    pGpuBuffer.reset();
    return Py_None;
}

#endif
