#ifndef __DATASETACCESSORS_H__
#define __DATASETACCESSORS_H__

class DataSetAccessors {

    public:
        /**
         * @brief Retrieves the name of a DataSetBase object.
         *
         * This static inline function takes a DataSetBase object pointer and returns its name as a Python string.
         *
         * @param self The pointer to the Python object (self).
         * @param args The arguments passed to the function.
         *
         * @return A Python string representing the name of the DataSetBase object.
         *         Returns nullptr if the DataSetBase object pointer is null.
         */
        static inline PyObject* GetDataSetName(PyObject* self, PyObject* args);

        /**
         * @brief Creates a dense DataSetBase object.
         *
         * This static inline function takes input arguments representing the name and a NumPy array object
         * containing the dense data. It creates the DataSetBase object, copies the dense data into it,
         * and returns a Python capsule object encapsulating the DataSetBase object pointer.
         *
         * @param self The pointer to the Python object (self).
         * @param args The arguments passed to the function.
         *
         * @return A Python capsule object containing the pointer to the created DataSetBase object.
         *         Returns nullptr if there was an error during creation or data copying.
         */
        static inline PyObject* CreateDenseDataSet(PyObject* self, PyObject* args);

        /**
         * @brief Creates a sparse DataSetBase object.
         *
         * This static inline function takes input arguments representing the name, shape, data, indices, and
         * indptr of a sparse DataSetBase object. It creates the DataSetBase object, copies the sparse data into it,
         * and returns a Python capsule object encapsulating the DataSetBase object pointer.
         *
         * @param self The pointer to the Python object (self).
         * @param args The arguments passed to the function.
         *
         * @return A Python capsule object containing the pointer to the created DataSetBase object.
         *         Returns nullptr if there was an error during creation or data copying.
         */
        static inline PyObject* CreateSparseDataSet(PyObject* self, PyObject* args);

        /**
         * @brief Sets the streaming status of a DataSetBase object.
         *
         * This static inline function takes a DataSetBase object pointer and an integer value representing
         * the streaming status. It sets the streaming status of the DataSetBase object and returns the updated
         * streaming status as a Python integer.
         *
         * @param self The pointer to the Python object (self).
         * @param args The arguments passed to the function.
         *
         * @return A Python integer representing the updated streaming status of the DataSetBase object.
         *         Returns nullptr if the DataSetBase object pointer is null.
         */
        static inline PyObject* SetStreaming(PyObject* self, PyObject* args);

        /**
         * @brief Retrieves the streaming status from a DataSetBase object.
         *
         * This static inline function takes a DataSetBase object pointer and returns the streaming status
         * as a Python integer.
         *
         * @param self The pointer to the Python object (self).
         * @param args The arguments passed to the function.
         *
         * @return A Python integer representing the streaming status of the DataSetBase object.
         *         Returns nullptr if the DataSetBase object pointer is null.
         */
        static inline PyObject* GetStreaming(PyObject* self, PyObject* args);

};

/**
 * @brief Retrieves the name of a DataSetBase object.
 *
 * This function takes a DataSetBase object pointer and returns its name as a Python string.
 *
 * @param self The pointer to the Python object (self).
 * @param args The arguments passed to the function.
 *
 * @return A Python string representing the name of the DataSetBase object.
 *         Returns nullptr if the DataSetBase object pointer is null.
 */
PyObject* DataSetAccessors::GetDataSetName(PyObject* self, PyObject* args) {
    if (DataSetBase* pDataSetBase = parsePtr<DataSetBase*>(args, "data set"); pDataSetBase != nullptr) {
        return Py_BuildValue("s", pDataSetBase->_name.c_str());
    }
    return nullptr;
}

/**
 * @brief Retrieves the data type of a NumPy array.
 *
 * This function takes a PyArrayObject pointer representing a NumPy array and determines its
 * data type based on its properties, such as item size and signedness. It returns the
 * corresponding DataSetEnums::DataType value representing the data type.
 *
 * @param numpyArray The pointer to the PyArrayObject representing the NumPy array.
 *
 * @return The DataSetEnums::DataType value representing the data type of the NumPy array.
 *         If the data type is unsupported, it returns DataSetEnums::DataType::RGB8.
 */
static DataSetEnums::DataType getDataType(PyArrayObject* numpyArray) {
    if constexpr (PyArray_ITEMSIZE(numpyArray) == 4) {
        if (PyArray_ISUNSIGNED(numpyArray)) {
            return DataSetEnums::DataType::UInt;
        }
        if (PyArray_ISSIGNED(numpyArray)) {
            return DataSetEnums::DataType::Int;
        }
        if (PyArray_ISFLOAT(numpyArray)) {
            return DataSetEnums::DataType::Float;
        }
    } else if constexpr (PyArray_ITEMSIZE(numpyArray) == 8) {
        if (PyArray_ISFLOAT(numpyArray)) {
            return DataSetEnums::DataType::Double;
        }
    } else if constexpr (PyArray_ITEMSIZE(numpyArray) == 1) {
        if (PyArray_ISUNSIGNED(numpyArray)) {
            return DataSetEnums::DataType::UChar;
        }
        if (PyArray_ISSIGNED(numpyArray)) {
            return DataSetEnums::DataType::Char;
        }
    }

    PyErr_SetString(PyExc_RuntimeError, "Unsupported NumPy data type");
    return DataSetEnums::DataType::RGB8;
}

/**
 * @brief Creates a dense DataSetBase object.
 *
 * This function takes input arguments representing the name and a NumPy array object
 * containing the dense data. It creates the DataSetBase object, copies the dense data
 * into it, and returns a Python capsule object encapsulating the DataSetBase object pointer.
 *
 * @param self The pointer to the Python object (self).
 * @param args The arguments passed to the function.
 *
 * @return A Python capsule object containing the pointer to the created DataSetBase object.
 *         Returns nullptr if there was an error during creation or data copying.
 */
PyObject* DataSetAccessors::CreateDenseDataSet(PyObject* self, PyObject* args) {
    char const* name = nullptr;
    PyArrayObject* numpyArray = nullptr;
    if (!PyArg_ParseTuple(args, "sO", &name, &numpyArray))
        return nullptr;

    DataSetEnums::DataType dataType = getDataType(numpyArray);
    if (dataType == DataSetEnums::DataType::RGB8) {
        PyErr_SetString(PyExc_RuntimeError, "DataSetAccessors::CreateDenseDataSet received unsupported data type");
        return nullptr;
    }

    uint32_t width = 1, height = 1, length = 1, examples = 1;
    int ndim = PyArray_NDIM(numpyArray);
    if (ndim >= 1) {
        examples = PyArray_DIM(numpyArray, 0);
    }
    if (ndim >= 2) {
        width = PyArray_DIM(numpyArray, 0);
        examples = PyArray_DIM(numpyArray, 1);
    }
    if (ndim >= 3) {
        height = PyArray_DIM(numpyArray, 1);
        examples = PyArray_DIM(numpyArray, 2);
    }
    if (ndim >= 4) {
        length = PyArray_DIM(numpyArray, 2);
        examples = PyArray_DIM(numpyArray, 3);
    }

    DataSetDimensions dimensions(width, height, length);
    DataSetDescriptor descriptor;
    descriptor._name = std::string(name);
    descriptor._dataType = dataType;
    descriptor._attributes = 0;
    descriptor._dim = dimensions;
    descriptor._examples = examples;
    descriptor._sparseDensity = Float(1.0);

    auto pDataSet = createDataSet(descriptor);
    pDataSet->CopyDenseData(numpyArray);

    return PyCapsule_New(reinterpret_cast<void*>(pDataSet), "data set", nullptr);
}

/**
 * @brief Creates a sparse DataSetBase object.
 *
 * This function takes input arguments representing the name, shape, data, indices, and
 * indptr of a sparse DataSetBase object. It creates the DataSetBase object, copies the
 * sparse data into it, and returns a Python capsule object encapsulating the DataSetBase
 * object pointer.
 *
 * @param self The pointer to the Python object (self).
 * @param args The arguments passed to the function.
 *
 * @return A Python capsule object containing the pointer to the created DataSetBase object.
 *         Returns nullptr if there was an error during creation or data copying.
 */
PyObject* DataSetAccessors::CreateSparseDataSet(PyObject* self, PyObject* args) {
    char const* name = nullptr;
    PyObject* shape = nullptr;
    PyArrayObject* data = nullptr;
    PyArrayObject* indices = nullptr;
    PyArrayObject* indptr = nullptr;

    if (!PyArg_ParseTuple(args, "sOOOO", &name, &shape, &data, &indices, &indptr))
        return nullptr;

    DataSetEnums::DataType dataType = getDataType(data);
    if (dataType == DataSetEnums::DataType::RGB8) {
        PyErr_SetString(PyExc_RuntimeError, "DataSetAccessors::CreateSparseDataSet received unsupported data type");
        return nullptr;
    }

    int rows = -1, cols = -1;
    if (!PyArg_ParseTuple(shape, "ii", &rows, &cols))
        return nullptr;

    DataSetDimensions dimensions(cols, 1, 1);
    DataSetDescriptor descriptor;
    descriptor._name = std::string(name);
    descriptor._dataType = dataType;
    descriptor._attributes = DataSetEnums::Attributes::Sparse;
    descriptor._dim = dimensions;
    descriptor._examples = rows;
    descriptor._sparseDensity = Float(PyArray_SIZE(data)) / Float(rows * cols);

    int* indicesData = static_cast<int*>(PyArray_DATA(indices));
    npy_intp indicesSize = PyArray_SIZE(indices);
    std::vector<uint32_t> sparseIndex(indicesData, indicesData + indicesSize);

    int* indptrData = static_cast<int*>(PyArray_DATA(indptr));
    npy_intp indptrSize = PyArray_SIZE(indptr);
    std::vector<uint64_t> sparseStart(indptrData, indptrData + indptrSize - 1);
    std::vector<uint64_t> sparseEnd(indptrData + 1, indptrData + indptrSize);

    auto pDataSet = std::make_unique<DataSetBase>(descriptor);
    pDataSet->CopySparseData(sparseStart.data(), sparseEnd.data(), data, sparseIndex.data());

    return PyCapsule_New(reinterpret_cast<void*>(pDataSet.release()), "data set", nullptr);
}

/**
 * @brief Sets the streaming status of a DataSetBase object.
 *
 * This function takes a DataSetBase object pointer and an integer value representing
 * the streaming status. It sets the streaming status of the DataSetBase object and
 * returns the updated streaming status as a Python integer.
 *
 * @param self The pointer to the Python object (self).
 * @param args The arguments passed to the function.
 *
 * @return A Python integer representing the updated streaming status of the DataSetBase object.
 *         Returns nullptr if the DataSetBase object pointer is null.
 */
PyObject* DataSetAccessors::SetStreaming(PyObject* self, PyObject* args) {
    int streaming = 0;
    if (DataSetBase* pDataSet = parsePtrAndOneValue<DataSetBase*, int>(args, streaming, "data set", "Oi"); pDataSet != nullptr) {
        return Py_BuildValue("i", pDataSet->SetStreaming(streaming));
    }
    return nullptr;
}

/**
 * @brief Retrieves the streaming status from a DataSetBase object.
 *
 * This function takes a DataSetBase object pointer and returns the streaming status
 * as a Python integer.
 *
 * @param self The pointer to the Python object (self).
 * @param args The arguments passed to the function.
 *
 * @return A Python integer representing the streaming status of the DataSetBase object.
 *         Returns nullptr if the DataSetBase object pointer is null.
 */
PyObject* DataSetAccessors::GetStreaming(PyObject* self, PyObject* args) {
    if (DataSetBase* pDataSet = parsePtr<DataSetBase*>(args, "data set"); pDataSet != nullptr) {
        return Py_BuildValue("i", pDataSet->GetStreaming());
    }
    return nullptr;
}

#endif
