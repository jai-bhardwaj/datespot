#ifndef __LAYERACCESSORS_H__
#define __LAYERACCESSORS_H__

class LayerAccessors {
    public:
        /**
         * @brief Retrieves the name of the Layer object.
         *
         * @param self The Python object representing the LayerAccessors class.
         * @param args The arguments passed to the function.
         * @return PyObject* The name of the Layer as a Python object, or nullptr on error.
         */
        static inline PyObject* GetName(PyObject* self, PyObject* args);

        /**
         * @brief Retrieves the kind of the Layer object.
         *
         * @param self The Python object representing the LayerAccessors class.
         * @param args The arguments passed to the function.
         * @return PyObject* The kind of the Layer as a Python object, or nullptr on error.
         */
        static inline PyObject* GetKind(PyObject* self, PyObject* args);

        /**
         * @brief Retrieves the type of the Layer object.
         *
         * @param self The Python object representing the LayerAccessors class.
         * @param args The arguments passed to the function.
         * @return PyObject* The type of the Layer as a Python object, or nullptr on error.
         */
        static inline PyObject* GetType(PyObject* self, PyObject* args);

        /**
         * @brief Retrieves the attributes of the Layer object.
         *
         * @param self The Python object representing the LayerAccessors class.
         * @param args The arguments passed to the function.
         * @return PyObject* The attributes of the Layer as a Python object, or nullptr on error.
         */
        static inline PyObject* GetAttributes(PyObject* self, PyObject* args);

        /**
         * @brief Retrieves the data set value.
         *
         * @param self The Python object representing the LayerAccessors class.
         * @param args The arguments passed to the function.
         * @return PyObject* The data set value as a Python object, or nullptr on error.
         */
        static inline PyObject* GetDataSet(PyObject* self, PyObject* args);

        /**
         * @brief Retrieves the number of dimensions.
         *
         * @param self The Python object representing the LayerAccessors class.
         * @param args The arguments passed to the function.
         * @return PyObject* The number of dimensions as a Python object, or nullptr on error.
         */
        static inline PyObject* GetNumDimensions(PyObject* self, PyObject* args);

        /**
         * @brief Retrieves the dimensions value.
         *
         * @param self The Python object representing the LayerAccessors class.
         * @param args The arguments passed to the function.
         * @return PyObject* The dimensions value as a Python object, or nullptr on error.
         */
        static inline PyObject* GetDimensions(PyObject* self, PyObject* args);

        /**
         * @brief Retrieves the local dimensions value.
         *
         * @param self The Python object representing the LayerAccessors class.
         * @param args The arguments passed to the function.
         * @return PyObject* The local dimensions value as a Python object, or nullptr on error.
         */
        static inline PyObject* GetLocalDimensions(PyObject* self, PyObject* args);

        /**
         * @brief Retrieves the kernel dimensions value.
         *
         * @param self The Python object representing the LayerAccessors class.
         * @param args The arguments passed to the function.
         * @return PyObject* The kernel dimensions value as a Python object, or nullptr on error.
         */
        static inline PyObject* GetKernelDimensions(PyObject* self, PyObject* args);

        /**
         * @brief Retrieves the kernel stride value.
         *
         * @param self The Python object representing the LayerAccessors class.
         * @param args The arguments passed to the function.
         * @return PyObject* The kernel stride value as a Python object, or nullptr on error.
         */
        static inline PyObject* GetKernelStride(PyObject* self, PyObject* args);

        /**
         * @brief Retrieves the units value.
         *
         * @param self The Python object representing the LayerAccessors class.
         * @param args The arguments passed to the function.
         * @return PyObject* The units value as a Python object, or nullptr on error.
         */
        static inline PyObject* GetUnits(PyObject* self, PyObject* args);

        /**
         * @brief Sets the units value.
         *
         * @param self The Python object representing the LayerAccessors class.
         * @param args The arguments passed to the function.
         * @return PyObject* None on success, or nullptr on error.
         */
        static inline PyObject* SetUnits(PyObject* self, PyObject* args);

        /**
         * @brief Retrieves the deltas value.
         *
         * @param self The Python object representing the LayerAccessors class.
         * @param args The arguments passed to the function.
         * @return PyObject* The deltas value as a Python object, or nullptr on error.
         */
        static inline PyObject* GetDeltas(PyObject* self, PyObject* args);

        /**
         * @brief Sets the deltas value.
         *
         * @param self The Python object representing the LayerAccessors class.
         * @param args The arguments passed to the function.
         * @return PyObject* None on success, or nullptr on error.
         */
        static inline PyObject* SetDeltas(PyObject* self, PyObject* args);

};

/**
 * @brief Mapping from Layer kind enumeration values to their corresponding string representations.
 */
static const std::map<Layer::Kind, std::string> intToStringKindMap = {
    {Layer::Kind::Input,  "Input"},
    {Layer::Kind::Hidden, "Hidden"},
    {Layer::Kind::Output, "Output"},
    {Layer::Kind::Target, "Target"}
};

/**
 * @brief Mapping from Layer type enumeration values to their corresponding string representations.
 */
static const std::map<Layer::Type, std::string> intToStringTypeMap = {
    {Layer::Type::FullyConnected, "FullyConnected"},
    {Layer::Type::Convolutional,  "Convolutional"},
    {Layer::Type::Pooling,        "Pooling"}
};

/**
 * @brief Retrieves the name of a Layer object.
 * 
 * @param self The Python object representing the LayerAccessors class.
 * @param args The arguments passed to the function.
 * @return PyObject* The name of the Layer as a Python string, or nullptr on error.
 */
PyObject* LayerAccessors::GetName(PyObject* self, PyObject* args) {
    Layer* pLayer = parsePtr<Layer*>(args, "layer");
    if (pLayer == nullptr) {
        return nullptr;
    }
    std::string_view name = pLayer->GetName();

    return Py_BuildValue("s", name.data());
}

/**
 * @brief Get the kind of a layer.
 *
 * This function retrieves the kind of a layer and returns it as a Python object.
 *
 * @param self The Python object representing the layer accessors.
 * @param args The arguments passed to the function.
 * @return PyObject* A Python object representing the kind of the layer.
 */
PyObject* LayerAccessors::GetKind(PyObject* self, PyObject* args) {
    Layer* pLayer = parsePtr<Layer*>(args, "layer"); /**< The layer object. */
    if (pLayer == nullptr) return nullptr;
    auto it = intToStringKindMap.find(pLayer->GetKind()); /**< Iterator for finding the kind. */
    if (it == intToStringKindMap.end()) {
        PyErr_SetString(PyExc_RuntimeError, "LayerAccessors::GetKind received unsupported kind enumerator");
        return nullptr;
    }
    return Py_BuildValue("s", it->second.c_str());
}

/**
 * @brief Get the type of a layer.
 *
 * This function retrieves the type of a layer and returns it as a Python object.
 *
 * @param self The Python object representing the layer accessors.
 * @param args The arguments passed to the function.
 * @return PyObject* A Python object representing the type of the layer.
 */
PyObject* LayerAccessors::GetType(PyObject* self, PyObject* args) {
    Layer* pLayer = parsePtr<Layer*>(args, "layer"); /**< The layer object. */
    if (pLayer == nullptr) return nullptr;
    std::map<Layer::Type, std::string>::iterator it = intToStringTypeMap.find(pLayer->GetType()); /**< Iterator for finding the type. */
    if (it == intToStringTypeMap.end()) {
        PyErr_SetString(PyExc_RuntimeError, "LayerAccessors::GetType received illegal type value");
        return nullptr;
    }
    return Py_BuildValue("s", it->second.c_str());
}

/**
 * @brief Get the attributes of a layer.
 *
 * This function retrieves the attributes of a layer and returns them as a Python object.
 *
 * @param self The Python object representing the layer accessors.
 * @param args The arguments passed to the function.
 * @return PyObject* A Python object representing the attributes.
 */
PyObject* LayerAccessors::GetAttributes(PyObject* self, PyObject* args) {
    Layer* pLayer = parsePtr<Layer*>(args, "layer"); /**< The layer object. */
    if (pLayer == nullptr) return nullptr;
    return Py_BuildValue("I", pLayer->GetAttributes());
}

/**
 * @brief Get the data set associated with a layer.
 *
 * This function retrieves the data set associated with a layer and returns it as a Python capsule object.
 *
 * @param self The Python object representing the layer accessors.
 * @param args The arguments passed to the function.
 * @return PyObject* A Python capsule object representing the data set.
 */
PyObject* LayerAccessors::GetDataSet(PyObject* self, PyObject* args) {
    Layer* pLayer = parsePtr<Layer*>(args, "layer"); /**< The layer object. */
    if (pLayer == nullptr) return nullptr;
    return PyCapsule_New(reinterpret_cast<void*>(const_cast<DataSetBase*>(pLayer->GetDataSet())), "data set", nullptr);
}

/**
 * @brief Get the number of dimensions for a layer.
 *
 * This function retrieves the number of dimensions for a layer and returns it as a Python object.
 *
 * @param self The Python object representing the layer accessors.
 * @param args The arguments passed to the function.
 * @return PyObject* A Python object representing the number of dimensions.
 */
PyObject* LayerAccessors::GetNumDimensions(PyObject* self, PyObject* args) {
    Layer* pLayer = parsePtr<Layer*>(args, "layer"); /**< The layer object. */
    if (pLayer == nullptr) return nullptr;
    return Py_BuildValue("I", pLayer->GetNumDimensions());
}

/**
 * @brief Get the dimensions for a layer.
 *
 * This function retrieves the dimensions for a layer and returns them as a list.
 *
 * @param self The Python object representing the layer accessors.
 * @param args The arguments passed to the function.
 * @return PyObject* A Python list object representing the dimensions.
 */
PyObject* LayerAccessors::GetDimensions(PyObject* self, PyObject* args) {
    Layer* pLayer = parsePtr<Layer*>(args, "layer"); /**< The layer object. */
    if (pLayer == nullptr) return nullptr;
    uint32_t Nx = 0, Ny = 0, Nz = 0, Nw = 0;
    std::tie(Nx, Ny, Nz, Nw) = pLayer->GetDimensions(); /**< Get the dimensions. */
    return Py_BuildValue("[IIII]", Nx, Ny, Nz, Nw);
}

/**
 * @brief Get the local dimensions for a layer.
 *
 * This function retrieves the local dimensions for a layer and returns them as a list.
 *
 * @param self The Python object representing the layer accessors.
 * @param args The arguments passed to the function.
 * @return PyObject* A Python list object representing the local dimensions.
 */
PyObject* LayerAccessors::GetLocalDimensions(PyObject* self, PyObject* args) {
    Layer* pLayer = parsePtr<Layer*>(args, "layer"); /**< The layer object. */
    if (pLayer == nullptr) return nullptr;
    uint32_t Nx = 0, Ny = 0, Nz = 0, Nw = 0;
    std::tie(Nx, Ny, Nz, Nw) = pLayer->GetLocalDimensions(); /**< Get the local dimensions. */
    return Py_BuildValue("[IIII]", Nx, Ny, Nz, Nw);
}

/**
 * @brief Get the kernel dimensions for a layer.
 *
 * This function retrieves the kernel dimensions for a layer and returns them as a list.
 *
 * @param self The Python object representing the layer accessors.
 * @param args The arguments passed to the function.
 * @return PyObject* A Python list object representing the kernel dimensions.
 */
PyObject* LayerAccessors::GetKernelDimensions(PyObject* self, PyObject* args) {
    Layer* pLayer = parsePtr<Layer*>(args, "layer"); /**< The layer object. */
    if (pLayer == nullptr) return nullptr;
    uint32_t kernelX = 0, kernelY = 0, kernelZ = 0;
    std::tie(kernelX, kernelY, kernelZ) = pLayer->GetKernelDimensions(); /**< Get the kernel dimensions. */
    return Py_BuildValue("[III]", kernelX, kernelY, kernelZ);
}

/**
 * @brief Get the kernel stride for a layer.
 *
 * This function retrieves the kernel stride for a layer and returns it as a list.
 *
 * @param self The Python object representing the layer accessors.
 * @param args The arguments passed to the function.
 * @return PyObject* A Python list object representing the kernel stride values.
 */
PyObject* LayerAccessors::GetKernelStride(PyObject* self, PyObject* args) {
    Layer* pLayer = parsePtr<Layer*>(args, "layer"); /**< The layer object. */
    if (pLayer == nullptr) return nullptr;
    uint32_t kernelStrideX = 0, kernelStrideY = 0, kernelStrideZ = 0;
    std::tie(kernelStrideX, kernelStrideY, kernelStrideZ) = pLayer->GetKernelStride(); /**< Get the kernel stride values. */
    return Py_BuildValue("[III]", kernelStrideX, kernelStrideY, kernelStrideZ);
}

/**
 * @brief Get the units for a layer.
 *
 * This function retrieves the units for a layer based on the given input.
 *
 * @param self The Python object representing the layer accessors.
 * @param args The arguments passed to the function.
 * @return PyObject* A Python object representing the result of getting the units.
 */
PyObject* LayerAccessors::GetUnits(PyObject* self, PyObject* args) {
    PyArrayObject* unitsArray = nullptr; /**< The PyArrayObject representing the units array. */
    uint32_t offset = 0; /**< The offset value for accessing units. */
    Layer* pLayer = parsePtrAndTwoValues<Layer*, PyArrayObject*, uint32_t>(args, unitsArray, offset, "layer", "OOI"); /**< The layer object. */
    if (pLayer == nullptr) return nullptr;
    if (CheckNumPyArray(unitsArray) == nullptr) return nullptr;
    float* pUnits = static_cast<float*>(PyArray_DATA(unitsArray));
    if (pUnits == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "LayerAccessors::GetUnits received NULL NumPy array data pointer");
        return nullptr;
    }
    pUnits += offset;
    return Py_BuildValue("i", pLayer->GetUnits(pUnits));
}

/**
 * @brief Set the units for a layer.
 *
 * This function sets the units for a layer based on the given input.
 *
 * @param self The Python object representing the layer accessors.
 * @param args The arguments passed to the function.
 * @return PyObject* A Python object representing the result of setting the units.
 */
PyObject* LayerAccessors::SetUnits(PyObject* self, PyObject* args) {
    PyArrayObject* unitsArray = nullptr; /**< The PyArrayObject representing the units array. */
    Layer* pLayer = parsePtrAndOneValue<Layer*, PyArrayObject*>(args, unitsArray, "layer", "OO"); /**< The layer object. */
    if (pLayer == nullptr) return nullptr;
    if (CheckNumPyArray(unitsArray) == nullptr) return nullptr;
    std::vector<float> units = NumPyArrayToVector(unitsArray);
    if (units.empty()) {
        PyErr_SetString(PyExc_RuntimeError, "LayerAccessors::SetUnits received empty units vector");
        return nullptr;
    }
    return Py_BuildValue("i", pLayer->SetUnits(units));
}

/**
 * @brief Get the deltas for a layer.
 *
 * This function retrieves the deltas for a layer based on the given input.
 *
 * @param self The Python object representing the layer accessors.
 * @param args The arguments passed to the function.
 * @return PyObject* A Python object representing the result of getting the deltas.
 */
PyObject* LayerAccessors::GetDeltas(PyObject* self, PyObject* args) {
    PyArrayObject* deltasArray = nullptr; /**< The PyArrayObject representing the deltas array. */
    uint32_t offset = 0; /**< The offset value for accessing deltas. */
    Layer* pLayer = parsePtrAndTwoValues<Layer*, PyArrayObject*, uint32_t>(args, deltasArray, offset, "layer", "OOI"); /**< The layer object. */
    if (pLayer == nullptr) return nullptr;
    if (CheckNumPyArray(deltasArray) == nullptr) return nullptr;
    float* pDeltas = static_cast<float*>(PyArray_DATA(deltasArray));
    if (pDeltas == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "LayerAccessors::GetDeltas received NULL NumPy array data pointer");
        return nullptr;
    }
    pDeltas += offset;
    return Py_BuildValue("i", pLayer->GetDeltas(pDeltas));
}

/**
 * @brief Set the deltas for a layer.
 *
 * This function sets the deltas for a layer based on the given input.
 *
 * @param self The Python object representing the layer accessors.
 * @param args The arguments passed to the function.
 * @return PyObject* A Python object representing the result of setting the deltas.
 */
PyObject* LayerAccessors::SetDeltas(PyObject* self, PyObject* args) {
    PyArrayObject* deltasArray = nullptr; /**< The PyArrayObject representing the deltas array. */
    Layer* pLayer = parsePtrAndOneValue<Layer*, PyArrayObject*>(args, deltasArray, "layer", "OO"); /**< The layer object. */
    if (pLayer == nullptr) return nullptr;
    if (CheckNumPyArray(deltasArray) == nullptr) return nullptr;
    std::vector<float> deltas = NumPyArrayToVector(deltasArray);
    if (deltas.empty()) {
        PyErr_SetString(PyExc_RuntimeError, "LayerAccessors::SetDeltas received empty deltas vector");
        return nullptr;
    }
    return Py_BuildValue("i", pLayer->SetDeltas(deltas));
}

#endif
