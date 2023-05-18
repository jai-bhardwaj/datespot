#ifndef __UTILITIES_H__
#define __UTILITIES_H__

#include <optional>
#include <tuple>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class Utilities {
public:
    /**
     * Starts up the GPU.
     *
     * @param self The instance of the Utilities class.
     * @param args The arguments (not used in this method).
     * @return None.
     */
    static inline py::object Startup(py::object self, py::object args);

    /**
     * Shuts down the GPU.
     *
     * @param self The instance of the Utilities class.
     * @param args The arguments (not used in this method).
     * @return None.
     */
    static inline py::object Shutdown(py::object self, py::object args);

    /**
     * Creates a CDL (Color Decision List) object from a JSON file.
     *
     * @param self The instance of the Utilities class.
     * @param args The arguments containing the JSON filename.
     * @return A Python capsule object containing the created CDL.
     * @throws std::runtime_error if the JSON file parsing fails.
     */
    static inline py::object CreateCDLFromJSON(py::object self, py::object args);

    /**
     * Creates a CDL (Color Decision List) object with default values.
     *
     * @param self The instance of the Utilities class.
     * @param args The arguments (not used in this method).
     * @return A Python capsule object containing the created CDL.
     */
    static inline py::object CreateCDLFromDefaults(py::object self, py::object args);

    /**
     * Deletes a CDL (Color Decision List) object.
     *
     * @param self The instance of the Utilities class.
     * @param args The arguments containing the CDL to delete.
     * @return None.
     */
    static inline py::object DeleteCDL(py::object self, py::object args);

    /**
     * Loads a dataset from a NetCDF file.
     *
     * @param self The instance of the Utilities class.
     * @param args The arguments containing the NetCDF filename.
     * @return A Python list containing the loaded dataset(s).
     * @throws std::bad_alloc if memory allocation fails or if the dataset vector is empty.
     */
    static inline py::object LoadDataSetFromNetCDF(py::object self, py::object args);

    /**
     * Deletes a dataset.
     *
     * @param self The instance of the Utilities class.
     * @param args The arguments containing the dataset to delete.
     * @return None.
     */
    static inline py::object DeleteDataSet(py::object self, py::object args);

    /**
     * Loads a neural network from a NetCDF file.
     *
     * @param self The instance of the Utilities class.
     * @param args The arguments containing the NetCDF filename and batch size.
     * @return A Python capsule object containing the loaded neural network.
     * @throws std::runtime_error if loading the neural network fails.
     */
    static inline py::object LoadNeuralNetworkFromNetCDF(py::object self, py::object args);

    /**
     * Loads a neural network from a JSON file.
     *
     * @param self The instance of the Utilities class.
     * @param args The arguments containing the JSON filename, batch size, and dataset list.
     * @return A Python capsule object containing the loaded neural network.
     * @throws std::runtime_error if the dataset list is empty or if loading the neural network fails.
     */
    static inline py::object LoadNeuralNetworkFromJSON(py::object self, py::object args);

    /**
     * Deletes a network.
     *
     * @param self The instance of the Utilities class.
     * @param args The arguments containing the network to delete.
     * @return None.
     */
    static inline py::object DeleteNetwork(py::object self, py::object args);

    /**
     * Opens a file.
     *
     * @param self The instance of the Utilities class.
     * @param args The arguments containing the filename and mode to open the file.
     * @return A Python capsule object containing the opened file pointer.
     * @throws std::runtime_error if an error occurs while opening the file.
     */
    static inline py::object OpenFile(py::object self, py::object args);

    /**
     * Closes a file.
     *
     * @param self The instance of the Utilities class.
     * @param args The arguments containing the file pointer to close.
     * @return None.
     * @throws std::runtime_error if an error occurs while closing the file.
     */
    static inline py::object CloseFile(py::object self, py::object args);

    /**
     * Sets the random seed for GPU operations.
     *
     * @param self The instance of the Utilities class.
     * @param args The arguments containing the random seed value.
     * @return None.
     * @throws pybind11::error_already_set if an error occurs during casting or if the random seed is -1 and a Python error has occurred.
     */
    static inline py::object SetRandomSeed(py::object self, py::object args);

    /**
     * Retrieves the memory usage information.
     *
     * @param self The instance of the Utilities class.
     * @param args The arguments (not used in this method).
     * @return A tuple containing the GPU memory usage and CPU memory usage.
     */
    static inline py::object GetMemoryUsage(py::object self, py::object args);

    /**
     * Transposes the given arrays.
     *
     * @param self The instance of the Utilities class.
     * @param args The arguments containing the arrays to transpose.
     * @return The transposed array.
     * @note The ownership of the returned object is transferred to the caller.
     */
    static inline py::object Transpose(py::object self, py::object args);

    /**
     * Creates a float GPU buffer.
     *
     * @param self The instance of the Utilities class.
     * @param args The arguments containing the size of the GPU buffer to create.
     * @return A Python capsule object containing the created float GPU buffer.
     * @throws std::bad_alloc if memory allocation fails.
     */
    static inline py::object CreateFloatGpuBuffer(py::object self, py::object args);

    /**
     * Deletes a float GPU buffer.
     *
     * @param self The instance of the Utilities class.
     * @param args The arguments containing the GPU buffer to delete.
     * @return None.
     */
    static inline py::object DeleteFloatGpuBuffer(py::object self, py::object args);

    /**
     * Creates an unsigned GPU buffer.
     *
     * @param self The instance of the Utilities class.
     * @param args The arguments containing the size of the GPU buffer to create.
     * @return A Python capsule object containing the created unsigned GPU buffer.
     * @throws std::bad_alloc if memory allocation fails.
     */
    static inline py::object CreateUnsignedGpuBuffer(py::object self, py::object args);

    /**
     * Deletes an unsigned GPU buffer.
     *
     * @param self The instance of the Utilities class.
     * @param args The arguments containing the GPU buffer to delete.
     * @return None.
     */
    static inline py::object DeleteUnsignedGpuBuffer(py::object self, py::object args);

};
/**
 * Starts up the GPU.
 *
 * @param self The instance of the Utilities class.
 * @param args The arguments containing the command-line arguments as a list of strings.
 * @return None.
 * @throws pybind11::type_error if any item in the argument list is not a string.
 */
py::object Utilities::Startup(py::object self, py::object args) {
    py::list arglist = args.cast<py::list>();
    int argc = arglist.size();
    std::vector<std::string_view> argv;
    argv.reserve(argc);
    for (py::handle item : arglist) {
        if (!py::isinstance<py::str>(item)) {
            throw py::type_error("Argument list must contain string items");
        }
        argv.emplace_back(py::str(item));
    }

    getGpu().Startup(argc, argv.data());

    Py_RETURN_NONE;
}
/**
 * Shuts down the GPU.
 *
 * @param self The instance of the Utilities class.
 * @param args The arguments (not used in this method).
 * @return None.
 */
py::object Utilities::Shutdown(py::object self, py::object args) {
    getGpu().Shutdown();
    Py_RETURN_NONE;
}
/**
 * Creates a CDL (Color Decision List) object from a JSON file.
 *
 * @param self The instance of the Utilities class.
 * @param args The arguments containing the JSON filename.
 * @return A Python capsule object containing the created CDL.
 * @throws std::runtime_error if the JSON file parsing fails.
 */
py::object Utilities::CreateCDLFromJSON(py::object self, py::object args) {
    std::string_view jsonFilename = args.cast<std::string_view>();

    std::unique_ptr<CDL> pCDL = std::make_unique<CDL>();
    if (pCDL->Load_JSON(std::string(jsonFilename)) != 0) {
        std::string message = "Load_JSON could not parse JSON file: " + std::string(jsonFilename);
        throw std::runtime_error(message);
    }

    return py::capsule(pCDL.release(), "cdl");
}
/**
 * Creates a CDL (Color Decision List) object with default values.
 *
 * @param self The instance of the Utilities class.
 * @param args The arguments (not used in this method).
 * @return A Python capsule object containing the created CDL.
 */
py::object Utilities::CreateCDLFromDefaults(py::object self, py::object args) {
    std::unique_ptr<CDL> pCDL = std::make_unique<CDL>();
    return py::capsule(pCDL.release(), "cdl");
}
/**
 * Deletes a CDL (Color Decision List) object.
 *
 * @param self The instance of the Utilities class.
 * @param args The arguments containing the CDL to delete.
 * @return None.
 */
py::object Utilities::DeleteCDL(py::object self, py::object args) {
    std::unique_ptr<CDL> pCDL = args.cast<std::unique_ptr<CDL>>();
    return Py_None;
}
/**
 * Loads a dataset from a NetCDF file.
 *
 * @param self The instance of the Utilities class.
 * @param args The arguments containing the NetCDF filename.
 * @return A Python list containing the loaded dataset(s).
 * @throws std::bad_alloc if memory allocation fails or if the dataset vector is empty.
 */
py::object Utilities::LoadDataSetFromNetCDF(py::object self, py::object args) {
    std::string_view dataFilename = args.cast<std::string_view>();

    std::vector<DataSetBase*> vDataSetBase = LoadNetCDF(std::string(dataFilename));
    if (vDataSetBase.empty()) {
        throw std::bad_alloc();
    }

    return DataSetBaseVectorToPythonList(vDataSetBase);
}
/**
 * Deletes a dataset.
 *
 * @param self The instance of the Utilities class.
 * @param args The arguments containing the dataset to delete.
 * @return None.
 */
py::object Utilities::DeleteDataSet(py::object self, py::object args) {
    std::unique_ptr<DataSetBase> pDataSet = args.cast<std::unique_ptr<DataSetBase>>();
    return Py_None;
}
/**
 * Loads a neural network from a NetCDF file.
 *
 * @param self The instance of the Utilities class.
 * @param args The arguments containing the NetCDF filename and batch size.
 * @return A Python capsule object containing the loaded neural network.
 * @throws std::runtime_error if loading the neural network fails.
 */
py::object Utilities::LoadNeuralNetworkFromNetCDF(py::object self, py::object args) {
    std::string_view networkFilename;
    uint32_t batch;
    std::tie(networkFilename, batch) = args.cast<std::tuple<std::string_view, uint32_t>>();

    Network* pNetwork = LoadNeuralNetworkNetCDF(std::string(networkFilename), batch);
    if (pNetwork == nullptr) {
        throw std::runtime_error("Utilities::LoadNeuralNetworkFromNetCDF received nullptr result from LoadNeuralNetworkNetCDF");
    }

    return py::capsule(pNetwork, "neural network");
}

/**
 * Loads a neural network from a JSON file.
 *
 * @param self The instance of the Utilities class.
 * @param args The arguments containing the JSON filename, batch size, and dataset list.
 * @return A Python capsule object containing the loaded neural network.
 * @throws std::runtime_error if the dataset list is empty or if loading the neural network fails.
 */
py::object Utilities::LoadNeuralNetworkFromJSON(py::object self, py::object args) {
    std::string_view jsonFilename;
    uint32_t batch;
    py::object pDataSetBaseList;
    std::tie(jsonFilename, batch, pDataSetBaseList) = args.cast<std::tuple<std::string_view, uint32_t, py::object>>();
    std::vector<DataSetBase*> vDataSetBase = PythonListToDataSetBaseVector(pDataSetBaseList);
    if (vDataSetBase.empty()) {
        throw std::runtime_error("Utilities::LoadNeuralNetworkFromJSON received empty vDataSetBase");
    }
    Network* pNetwork = LoadNeuralNetworkJSON(std::string(jsonFilename), batch, vDataSetBase);
    if (pNetwork == nullptr) {
        throw std::runtime_error("Utilities::LoadNeuralNetworkFromJSON received nullptr result from LoadNeuralNetworkNetCDF");
    }
    return py::capsule(pNetwork, "neural network");
}

/**
 * Deletes a network.
 *
 * @param self The instance of the Utilities class.
 * @param args The arguments containing the network to delete.
 * @return None.
 */
py::object Utilities::DeleteNetwork(py::object self, py::object args) {
    std::unique_ptr<Network> pNetwork = args.cast<std::unique_ptr<Network>>();
    return Py_None;
}

/**
 * Opens a file.
 *
 * @param self The instance of the Utilities class.
 * @param args The arguments containing the filename and mode to open the file.
 * @return A Python capsule object containing the opened file pointer.
 * @throws std::runtime_error if an error occurs while opening the file.
 */
py::object Utilities::OpenFile(py::object self, py::object args) {
    const char* filename;
    const char* mode;
    std::tie(filename, mode) = args.cast<std::tuple<const char*, const char*>>();
    FILE* pFILE = fopen(filename, mode);
    if (pFILE == nullptr) {
        throw std::runtime_error("File open error");
    }

    return py::capsule(pFILE, "file");
}

/**
 * Closes a file.
 *
 * @param self The instance of the Utilities class.
 * @param args The arguments containing the file pointer to close.
 * @return None.
 * @throws std::runtime_error if an error occurs while closing the file.
 */
py::object Utilities::CloseFile(py::object self, py::object args) {
    FILE* pFILE = args.cast<FILE*>();
    if (fclose(pFILE) != 0) {
        throw std::runtime_error("File close error");
    }

    return Py_None;
}

/**
 * Sets the random seed for GPU operations.
 *
 * @param self The instance of the Utilities class.
 * @param args The arguments containing the random seed value.
 * @return None.
 * @throws pybind11::error_already_set if an error occurs during casting or if the random seed is -1 and a Python error has occurred.
 */
py::object Utilities::SetRandomSeed(py::object self, py::object args) {
    py::object randomSeedObj = args.cast<py::object>();

    unsigned long randomSeed = randomSeedObj.cast<unsigned long>();
    if (randomSeed == static_cast<unsigned long>(-1) && PyErr_Occurred()) {
        throw py::error_already_set();
    }

    getGpu().SetRandomSeed(randomSeed);
    Py_RETURN_NONE;
}

/**
 * Retrieves the memory usage information.
 *
 * @param self The instance of the Utilities class.
 * @param args The arguments (not used in this method).
 * @return A tuple containing the GPU memory usage and CPU memory usage.
 */
py::object Utilities::GetMemoryUsage(py::object self, py::object args) {
    auto [gpuMemoryUsage, cpuMemoryUsage] = getGpu().GetMemoryUsage();
    return py::make_tuple(gpuMemoryUsage, cpuMemoryUsage);
}

/**
 * Transposes the given arrays.
 *
 * @param self The instance of the Utilities class.
 * @param args The arguments containing the arrays to transpose.
 * @return The transposed array.
 * @note The ownership of the returned object is transferred to the caller.
 */
py::object Utilities::Transpose(py::object self, py::object args) {
    py::array_t<double> ASINWeightArray;
    py::array_t<double> pEmbeddingArray;

    std::tie(ASINWeightArray, pEmbeddingArray) = args.cast<std::tuple<py::array_t<double>, py::array_t<double>>>();

    try {
        auto result = std::unique_ptr<py::object>(tensorhubcalculate_Transpose(ASINWeightArray, pEmbeddingArray));
        return result.release();
    } catch (...) {
        return nullptr;
    }
}

/**
 * Creates a float GPU buffer.
 *
 * @param self The instance of the Utilities class.
 * @param args The arguments containing the size of the GPU buffer to create.
 * @return A Python capsule object containing the created float GPU buffer.
 * @throws std::bad_alloc if memory allocation fails.
 */
py::object Utilities::CreateFloatGpuBuffer(py::object self, py::object args) {
    uint32_t size = args.cast<uint32_t>();

    try {
        auto pGpuBuffer = std::make_unique<GpuBuffer<NNFloat>>(size, true);
        return py::capsule(pGpuBuffer.release(), "float gpu buffer");
    } catch (...) {
        throw std::bad_alloc();
    }
}

/**
 * Deletes a float GPU buffer.
 *
 * @param self The instance of the Utilities class.
 * @param args The arguments containing the GPU buffer to delete.
 * @return None.
 */
py::object Utilities::DeleteFloatGpuBuffer(py::object self, py::object args) {
    std::unique_ptr<GpuBuffer<NNFloat>> pGpuBuffer = args.cast<std::unique_ptr<GpuBuffer<NNFloat>>>();
    pGpuBuffer.reset();
    return Py_None;
}

/**
 * Creates an unsigned GPU buffer.
 *
 * @param self The instance of the Utilities class.
 * @param args The arguments containing the size of the GPU buffer to create.
 * @return A Python capsule object containing the created unsigned GPU buffer.
 * @throws std::bad_alloc if memory allocation fails.
 */
py::object Utilities::CreateUnsignedGpuBuffer(py::object self, py::object args) {
    uint32_t size = args.cast<uint32_t>();

    try {
        auto pGpuBuffer = std::make_unique<GpuBuffer<uint32_t>>(size, true);
        return py::capsule(pGpuBuffer.release(), "unsigned gpu buffer");
    } catch (...) {
        throw std::bad_alloc();
    }
}

/**
 * Deletes an unsigned GPU buffer.
 *
 * @param self The instance of the Utilities class.
 * @param args The arguments containing the GPU buffer to delete.
 * @return None.
 */
py::object Utilities::DeleteUnsignedGpuBuffer(py::object self, py::object args) {
    std::unique_ptr<GpuBuffer<uint32_t>> pGpuBuffer = args.cast<std::unique_ptr<GpuBuffer<uint32_t>>>();
    pGpuBuffer.reset();

    return Py_None;
}

#endif
