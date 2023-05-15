#ifndef __NETWORKACCESSORS_H__
#define __NETWORKACCESSORS_H__

#include <string_view>
#include <unordered_map>
#include <optional>
#include <memory>
#include <tuple>
#include <string>

class NetworkAccessors {
    public:
        /**
         * @brief Accessor function to retrieve the batch size from a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python object representing the batch size of the neural network.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* GetBatch(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to set the batch size in a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return None.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* SetBatch(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to retrieve the position from a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python object representing the position of the neural network.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* GetPosition(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to set the position in a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return None.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* SetPosition(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to retrieve the shuffle indices flag from a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python object representing the shuffle indices flag of the neural network.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* GetShuffleIndices(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to set the shuffle indices flag in a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return None.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* SetShuffleIndices(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to retrieve the sparseness penalty parameters from a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python list object representing the sparseness penalty parameters [p, beta].
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* GetSparsenessPenalty(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to set the sparseness penalty parameters in a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python object representing the result of setting the sparseness penalty parameters.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* SetSparsenessPenalty(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to retrieve the denoising parameter from a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python object representing the denoising parameter of the neural network.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* GetDenoising(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to set the denoising parameter in a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return None.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* SetDenoising(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to retrieve the delta boost parameters from a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python list object representing the delta boost parameters [one, zero].
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* GetDeltaBoost(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to set the delta boost parameters in a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python object representing the result of setting the delta boost parameters.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* SetDeltaBoost(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to retrieve the debug level from a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python object representing the debug level of the neural network.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* GetDebugLevel(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to set the debug level in a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return None.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* SetDebugLevel(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to retrieve the checkpoint information from a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python list object representing the checkpoint information [filename, interval].
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* GetCheckpoint(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to set the checkpoint information in a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python object representing the result of setting the checkpoint information.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* SetCheckpoint(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to retrieve the LRN (Local Response Normalization) parameters from a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python list object representing the LRN parameters [k, n, alpha, beta].
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* GetLRN(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to set the LRN (Local Response Normalization) parameters in a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python object representing the result of setting the LRN parameters.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* SetLRN(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to retrieve the SMCE (Softmax Cross-Entropy) parameters from a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python list object representing the SMCE parameters [oneTarget, zeroTarget, oneScale, zeroScale].
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* GetSMCE(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to set the SMCE (Softmax Cross-Entropy) parameters in a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python object representing the result of setting the SMCE parameters.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* SetSMCE(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to retrieve the maxout parameter from a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python object representing the maxout parameter of the neural network.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* GetMaxout(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to set the maxout parameter in a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python object representing the result of setting the maxout parameter.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* SetMaxout(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to retrieve the number of examples in a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python object representing the number of examples in the neural network.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* GetExamples(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to retrieve the weight of a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python object representing the weight of the neural network.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* GetWeight(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to retrieve the buffer size of a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python object representing the buffer size of the neural network.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* GetBufferSize(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to retrieve a specific layer from a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python object representing the requested layer.
         *         Returns nullptr if the neural network pointer is null or if the layer index is invalid.
         */
        static inline PyObject* GetLayer(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to retrieve all layers from a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python list object representing all the layers in the neural network.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* GetLayers(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to retrieve the name of a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python object representing the name of the neural network.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* GetName(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to retrieve the unit buffer of a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python object representing the unit buffer of the neural network.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* GetUnitBuffer(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to retrieve the delta buffer of a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python object representing the delta buffer of the neural network.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* GetDeltaBuffer(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to retrieve the weight buffer of a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python object representing the weight buffer of the neural network.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* GetWeightBuffer(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to retrieve the scratch buffer of a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python object representing the scratch buffer of the neural network.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* GetScratchBuffer(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to retrieve the P2P send buffer of a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python object representing the P2P send buffer of the neural network.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* GetP2PSendBuffer(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to retrieve the P2P receive buffer of a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python object representing the P2P receive buffer of the neural network.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* GetP2PReceiveBuffer(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to retrieve the P2P CPU buffer of a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python object representing the P2P CPU buffer of the neural network.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* GetP2PCPUBuffer(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to retrieve the peer buffer of a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python object representing the peer buffer of the neural network.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* GetPeerBuffer(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to retrieve the peer back buffer of a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return A Python object representing the peer back buffer of the neural network.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* GetPeerBackBuffer(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to set the clear velocity flag in a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return None.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* SetClearVelocity(PyObject* self, PyObject* args);

        /**
         * @brief Accessor function to set the training mode in a neural network.
         *
         * @param self A pointer to the current object.
         * @param args Arguments passed to the function.
         * @return None.
         *         Returns nullptr if the neural network pointer is null.
         */
        static inline PyObject* SetTrainingMode(PyObject* self, PyObject* args);

};

/**
 * @brief Accessor function to retrieve the batch size from a neural network.
 *
 * @param self A pointer to the current object.
 * @param args Arguments passed to the function.
 * @return A Python object representing the batch size of the neural network.
 *         Returns nullptr if the neural network pointer is null.
 */
PyObject* NetworkAccessors::GetBatch(PyObject* self, PyObject* args) {
    std::optional<Network*> pNetwork = parsePtr<Network*>(args, "neural network");
    if (!pNetwork) return nullptr;
    return Py_BuildValue("I", pNetwork->GetBatch());
}

/**
 * @brief Accessor function to set the batch size in a neural network.
 *
 * @param self A pointer to the current object.
 * @param args Arguments passed to the function.
 * @return None.
 *         Returns nullptr if the neural network pointer is null.
 */
PyObject* NetworkAccessors::SetBatch(PyObject* self, PyObject* args) {
    uint32_t batch = 0;
    std::optional<Network*> pNetwork = parsePtrAndOneValue<Network*, uint32_t>(args, batch, "neural network", "OI");
    if (!pNetwork) return nullptr;
    pNetwork->SetBatch(batch);
    Py_RETURN_NONE;
}

/**
 * @brief Accessor function to retrieve the position from a neural network.
 *
 * @param self A pointer to the current object.
 * @param args Arguments passed to the function.
 * @return A Python object representing the position of the neural network.
 *         Returns nullptr if the neural network pointer is null.
 */
PyObject* NetworkAccessors::GetPosition(PyObject* self, PyObject* args) {
    std::optional<Network*> pNetwork = parsePtr<Network*>(args, "neural network");
    if (!pNetwork) return nullptr;
    return Py_BuildValue("I", pNetwork->GetPosition());
}

/**
 * @brief Accessor function to set the position in a neural network.
 *
 * @param self A pointer to the current object.
 * @param args Arguments passed to the function.
 * @return None.
 *         Returns nullptr if the neural network pointer is null.
 */
PyObject* NetworkAccessors::SetPosition(PyObject* self, PyObject* args) {
    uint32_t position = 0;
    std::optional<Network*> pNetwork = parsePtrAndOneValue<Network*, uint32_t>(args, position, "neural network", "OI");
    if (!pNetwork) return nullptr;
    pNetwork->SetPosition(position);
    Py_RETURN_NONE;
}

/**
 * @brief Accessor function to retrieve the shuffle indices flag from a neural network.
 *
 * @param self A pointer to the current object.
 * @param args Arguments passed to the function.
 * @return A Python object representing the shuffle indices flag of the neural network.
 *         Returns nullptr if the neural network pointer is null.
 */
PyObject* NetworkAccessors::GetShuffleIndices(PyObject* self, PyObject* args) {
    std::optional<Network*> pNetwork = parsePtr<Network*>(args, "neural network");
    if (!pNetwork) return nullptr;

    int bShuffleIndices;
    std::tie(bShuffleIndices) = pNetwork->GetShuffleIndices();

    return Py_BuildValue("i", bShuffleIndices);
}

/**
 * @brief Accessor function to set the shuffle indices flag in a neural network.
 *
 * @param self A pointer to the current object.
 * @param args Arguments passed to the function.
 * @return None.
 *         Returns nullptr if the neural network pointer is null.
 */
PyObject* NetworkAccessors::SetShuffleIndices(PyObject* self, PyObject* args) {
    int bShuffleIndices = 0;
    std::optional<Network*> pNetwork = parsePtrAndOneValue<Network*, int>(args, bShuffleIndices, "neural network", "Oi");
    if (!pNetwork) return nullptr;
    pNetwork->SetShuffleIndices(bShuffleIndices);
    Py_RETURN_NONE;
}

/**
 * @brief Accessor function to retrieve the sparseness penalty parameters from a neural network.
 *
 * @param self A pointer to the current object.
 * @param args Arguments passed to the function.
 * @return A Python list object representing the sparseness penalty parameters [p, beta].
 *         Returns nullptr if the neural network pointer is null.
 */
PyObject* NetworkAccessors::GetSparsenessPenalty(PyObject* self, PyObject* args) {
    std::optional<Network*> pNetwork = parsePtr<Network*>(args, "neural network");
    if (!pNetwork) return nullptr;

    NNFloat p, beta;
    std::tie(p, beta) = pNetwork->GetSparsenessPenalty();

    return Py_BuildValue("[ff]", p, beta);
}

/**
 * @brief Accessor function to set the sparseness penalty parameters in a neural network.
 *
 * @param self A pointer to the current object.
 * @param args Arguments passed to the function.
 * @return A Python object representing the result of setting the sparseness penalty parameters.
 *         Returns nullptr if the neural network pointer is null.
 */
PyObject* NetworkAccessors::SetSparsenessPenalty(PyObject* self, PyObject* args) {
    NNFloat p = 0.0, beta = 0.0;
    std::optional<Network*> pNetwork = parsePtrAndTwoValues<Network*, NNFloat, NNFloat>(args, p, beta, "neural network", "Off");
    if (!pNetwork) return nullptr;
    return Py_BuildValue("i", pNetwork->SetSparsenessPenalty(p, beta));
}

/**
 * @brief Accessor function to retrieve the denoising probability from a neural network.
 *
 * @param self A pointer to the current object.
 * @param args Arguments passed to the function.
 * @return A Python object representing the denoising probability of the neural network.
 *         Returns nullptr if the neural network pointer is null.
 */
PyObject* NetworkAccessors::GetDenoising(PyObject* self, PyObject* args) {
    std::optional<Network*> pNetwork = parsePtr<Network*>(args, "neural network");
    if (!pNetwork) return nullptr;

    NNFloat denoisingP;
    std::tie(denoisingP) = pNetwork->GetDenoising();

    return Py_BuildValue("f", denoisingP);
}

/**
 * @brief Accessor function to set the denoising probability in a neural network.
 *
 * @param self A pointer to the current object.
 * @param args Arguments passed to the function.
 * @return A Python object representing the result of setting the denoising probability.
 *         Returns nullptr if the neural network pointer is null.
 */
PyObject* NetworkAccessors::SetDenoising(PyObject* self, PyObject* args) {
    NNFloat denoisingP = 0.0;
    std::optional<Network*> pNetwork = parsePtrAndOneValue<Network*, NNFloat>(args, denoisingP, "neural network", "Of");
    if (!pNetwork) return nullptr;
    return Py_BuildValue("i", pNetwork->SetDenoising(denoisingP));
}

/**
 * @brief Accessor function to retrieve the delta boost values from a neural network.
 *
 * @param self A pointer to the current object.
 * @param args Arguments passed to the function.
 * @return A Python list object representing the delta boost values [one, zero].
 *         Returns nullptr if the neural network pointer is null.
 */
PyObject* NetworkAccessors::GetDeltaBoost(PyObject* self, PyObject* args) {
    std::optional<Network*> pNetwork = parsePtr<Network*>(args, "neural network");
    if (!pNetwork) return nullptr;

    NNFloat one, zero;
    std::tie(one, zero) = pNetwork->GetDeltaBoost();

    return Py_BuildValue("[ff]", one, zero);
}

/**
 * @brief Accessor function to set the delta boost values in a neural network.
 *
 * @param self A pointer to the current object.
 * @param args Arguments passed to the function.
 * @return A Python object representing the result of setting the delta boost values.
 *         Returns nullptr if the neural network pointer is null.
 */
PyObject* NetworkAccessors::SetDeltaBoost(PyObject* self, PyObject* args) {
    NNFloat one = 0.0, zero = 0.0;
    std::optional<Network*> pNetwork = parsePtrAndTwoValues<Network*, NNFloat, NNFloat>(args, one, zero, "neural network", "Off");
    if (!pNetwork) return nullptr;
    return Py_BuildValue("i", pNetwork->SetDeltaBoost(one, zero));
}

/**
 * @brief Accessor function to retrieve the debug level from a neural network.
 *
 * @param self A pointer to the current object.
 * @param args Arguments passed to the function.
 * @return A Python object representing the debug level of the neural network.
 *         Returns nullptr if the neural network pointer is null.
 */
PyObject* NetworkAccessors::GetDebugLevel(PyObject* self, PyObject* args) {
    std::optional<Network*> pNetwork = parsePtr<Network*>(args, "neural network");
    if (!pNetwork) return nullptr;
    return Py_BuildValue("i", pNetwork->GetDebugLevel());
}

/**
 * @brief Accessor function to set the debug level in a neural network.
 *
 * @param self A pointer to the current object.
 * @param args Arguments passed to the function.
 * @return None.
 *         Returns nullptr if the neural network pointer is null.
 */
PyObject* NetworkAccessors::SetDebugLevel(PyObject* self, PyObject* args) {
    int bDebugLevel = 0;
    std::optional<Network*> pNetwork = parsePtrAndOneValue<Network*, int>(args, bDebugLevel, "neural network", "Oi");
    if (!pNetwork) return nullptr;
    pNetwork->SetDebugLevel(bDebugLevel);
    Py_RETURN_NONE;
}

/**
 * @brief Accessor function to retrieve the checkpoint parameters from a neural network.
 *
 * @param self A pointer to the current object.
 * @param args Arguments passed to the function.
 * @return A Python list object representing the checkpoint parameters [filename, interval].
 *         Returns nullptr if the neural network pointer is null.
 */
PyObject* NetworkAccessors::GetCheckpoint(PyObject* self, PyObject* args) {
    std::optional<Network*> pNetwork = parsePtr<Network*>(args, "neural network");
    if (!pNetwork) return nullptr;

    std::string filename;
    int32_t interval;
    std::tie(filename, interval) = pNetwork->GetCheckpoint();

    return Py_BuildValue("[si]", filename.c_str(), interval);
}

/**
 * @brief Accessor function to set the checkpoint parameters in a neural network.
 *
 * @param self A pointer to the current object.
 * @param args Arguments passed to the function.
 * @return A Python object representing the result of setting the checkpoint parameters.
 *         Returns nullptr if the neural network pointer is null.
 */
PyObject* NetworkAccessors::SetCheckpoint(PyObject* self, PyObject* args) {
    char const* filename = nullptr;
    int32_t interval = 0;
    std::optional<Network*> pNetwork = parsePtrAndTwoValues<Network*, char const*, int32_t>(
        args, filename, interval, "neural network", "Osi");
    if (!pNetwork) return nullptr;
    return Py_BuildValue("i", pNetwork->SetCheckpoint(std::string(filename), interval));
}

/**
 * @brief Accessor function to retrieve the Local Response Normalization (LRN) parameters from a neural network.
 *
 * @param self A pointer to the current object.
 * @param args Arguments passed to the function.
 * @return A Python list object representing the LRN parameters [k, n, alpha, beta].
 *         Returns nullptr if the neural network pointer is null.
 */
PyObject* NetworkAccessors::GetLRN(PyObject* self, PyObject* args) {
    std::optional<Network*> pNetwork = parsePtr<Network*>(args, "neural network");
    if (!pNetwork) return nullptr;

    NNFloat k, alpha, beta;
    uint32_t n;
    std::tie(k, n, alpha, beta) = pNetwork->GetLRN();

    return Py_BuildValue("[fIff]", k, n, alpha, beta);
}

/**
 * @brief Accessor function to set the Local Response Normalization (LRN) parameters in a neural network.
 *
 * @param self A pointer to the current object.
 * @param args Arguments passed to the function.
 * @return A Python object representing the result of setting the LRN parameters.
 *         Returns nullptr if the neural network pointer is null.
 */
PyObject* NetworkAccessors::SetLRN(PyObject* self, PyObject* args) {
    uint32_t n = 0;
    NNFloat k = 0.0, alpha = 0.0, beta = 0.0;
    std::optional<Network*> pNetwork = parsePtrAndFourValues<Network*, NNFloat, uint32_t, NNFloat, NNFloat>(
        args, k, n, alpha, beta, "neural network", "OfIff");
    if (!pNetwork) return nullptr;
    return Py_BuildValue("i", pNetwork->SetLRN(k, n, alpha, beta));
}

/**
 * @brief Accessor function to retrieve the Smoothed Multiclass Cross-Entropy (SMCE) values from a neural network.
 *
 * @param self A pointer to the current object.
 * @param args Arguments passed to the function.
 * @return A Python list object representing the SMCE values [oneTarget, zeroTarget, oneScale, zeroScale]
 *         Returns nullptr if the neural network pointer is null.
 */
PyObject* NetworkAccessors::GetSMCE(PyObject* self, PyObject* args) {
    std::optional<Network*> pNetwork = parsePtr<Network*>(args, "neural network");
    if (!pNetwork) return nullptr;

    NNFloat oneTarget, zeroTarget, oneScale, zeroScale;
    std::tie(oneTarget, zeroTarget, oneScale, zeroScale) = pNetwork->GetSMCE();

    return Py_BuildValue("[ffff]", oneTarget, zeroTarget, oneScale, zeroScale);
}

/**
 * @brief Accessor function to set the Smoothed Multiclass Cross-Entropy (SMCE) values in a neural network.
 *
 * @param self A pointer to the current object.
 * @param args Arguments passed to the function.
 * @return A Python object representing the result of setting the SMCE values.
 *         Returns nullptr if the neural network pointer is null.
 */
PyObject* NetworkAccessors::SetSMCE(PyObject* self, PyObject* args) {
    NNFloat oneTarget = 0.0, zeroTarget = 0.0, oneScale = 0.0, zeroScale = 0.0;
    std::optional<Network*> pNetwork = parsePtrAndFourValues<Network*, NNFloat, NNFloat, NNFloat, NNFloat>(
        args, oneTarget, zeroTarget, oneScale, zeroScale, "neural network", "Offff");

    if (!pNetwork) return nullptr;

    return Py_BuildValue("i", pNetwork->SetSMCE(oneTarget, zeroTarget, oneScale, zeroScale));
}

/**
 * @brief Accessor function to retrieve the maxout value from a neural network.
 *
 * @param self A pointer to the current object.
 * @param args Arguments passed to the function.
 * @return A Python object representing the maxout value from the neural network.
 *         Returns nullptr if the neural network pointer is null.
 */
PyObject* NetworkAccessors::GetMaxout(PyObject* self, PyObject* args) {
    std::optional<Network*> pNetwork = parsePtr<Network*>(args, "neural network");
    if (!pNetwork) return nullptr;

    uint32_t k;
    std::tie(k) = pNetwork->GetMaxout();
    return Py_BuildValue("I", k);
}

/**
 * @brief Accessor function to set the maxout value in a neural network.
 *
 * @param self A pointer to the current object.
 * @param args Arguments passed to the function.
 * @return A Python object representing the result of setting the maxout value.
 *         Returns nullptr if the neural network pointer is null.
 */
PyObject* NetworkAccessors::SetMaxout(PyObject* self, PyObject* args) {
    uint32_t maxoutK = 0;
    std::optional<Network*> pNetwork = parsePtrAndOneValue<Network*, uint32_t>(args, maxoutK, "neural network", "OI");
    if (!pNetwork) return nullptr;
    return Py_BuildValue("i", pNetwork->SetMaxout(maxoutK));
}

/**
 * @brief Accessor function to retrieve the number of examples in a neural network.
 *
 * @param self A pointer to the current object.
 * @param args Arguments passed to the function.
 * @return A Python object representing the number of examples in the neural network.
 *         Returns nullptr if the neural network pointer is null.
 */
PyObject* NetworkAccessors::GetExamples(PyObject* self, PyObject* args) {
    std::optional<Network*> pNetwork = parsePtr<Network*>(args, "neural network");
    if (!pNetwork) return nullptr;
    return Py_BuildValue("I", pNetwork->GetExamples());
}

/**
 * @brief Retrieves a specific weight from the neural network.
 *
 * @param self The object pointer.
 * @param args The argument list.
 * @return PyObject* PyCapsule containing the weight object, or nullptr on failure.
 */
PyObject* NetworkAccessors::GetWeight(PyObject* self, PyObject* args) {
    char const* inputLayer = nullptr;
    char const* outputLayer = nullptr;
    if (auto [pNetwork, parseResult] = parsePtrAndTwoValues<Network*, char const*, char const*>(args, inputLayer, outputLayer, "neural network", "Oss"); parseResult && pNetwork) {
        return PyCapsule_New(reinterpret_cast<void*>(const_cast<Weight*>(pNetwork->GetWeight(std::string_view(inputLayer), std::string_view(outputLayer)))), "weight", nullptr);
    }
    return nullptr;
}

/**
 * @brief Retrieves the buffer size of a specific layer in the neural network.
 *
 * @param self The object pointer.
 * @param args The argument list.
 * @return PyObject* Python integer object containing the buffer size, or nullptr on failure.
 */
PyObject* NetworkAccessors::GetBufferSize(PyObject* self, PyObject* args) {
    char const* layer = nullptr;
    if (auto [pNetwork, parseResult] = parsePtrAndOneValue<Network*, char const*>(args, layer, "neural network", "Os"); parseResult && pNetwork) {
        return Py_BuildValue("K", pNetwork->GetBufferSize(std::string_view(layer)));
    }
    return nullptr;
}

/**
 * @brief Retrieves a specific layer from the neural network.
 *
 * @param self The object pointer.
 * @param args The argument list.
 * @return PyObject* PyCapsule containing the layer object, or nullptr on failure.
 */
PyObject* NetworkAccessors::GetLayer(PyObject* self, PyObject* args) {
    char const* layer = nullptr;
    if (auto [pNetwork, parseResult] = parsePtrAndOneValue<Network*, char const*>(args, layer, "neural network", "Os"); parseResult && pNetwork) {
        return PyCapsule_New(reinterpret_cast<void*>(const_cast<Layer*>(pNetwork->GetLayer(std::string_view(layer)))), "layer", nullptr);
    }
    return nullptr;
}

/**
 * @brief Retrieves the layers of the neural network.
 *
 * @param self The object pointer.
 * @param args The argument list.
 * @return PyObject* PyList object containing the layers of the neural network, or nullptr on failure.
 */
PyObject* NetworkAccessors::GetLayers(PyObject* self, PyObject* args) {
    if (auto pNetwork = parsePtr<Network*>(args, "neural network"); pNetwork) {
        std::vector<std::string> layers = pNetwork->GetLayers();
        if (layers.empty()) {
            PyErr_SetString(PyExc_RuntimeError, "NetworkAccessors::GetLayers received empty layers vector");
            return nullptr;
        }

        size_t n = layers.size();
        PyObject* list = PyList_New(n);
        if (list == nullptr) {
            std::string message = "NetworkAccessors::GetLayers failed in PyList_New(" + std::to_string(n) + ")";
            PyErr_SetString(PyExc_RuntimeError, message.c_str());
            return nullptr;
        }

        for (size_t i = 0; i < n; i++) {
            PyObject* item = PyUnicode_FromStringAndSize(layers[i].data(), layers[i].size());
            if (item == nullptr) {
                std::string message = "NetworkAccessors::GetLayers failed in PyUnicode_FromStringAndSize for index = " + std::to_string(i);
                PyErr_SetString(PyExc_RuntimeError, message.c_str());
                Py_DECREF(list);
                return nullptr;
            }

            if (PyList_SetItem(list, i, item) < 0) {
                std::string message = "NetworkAccessors::GetLayers failed in PyList_SetItem for index = " + std::to_string(i);
                PyErr_SetString(PyExc_RuntimeError, message.c_str());
                Py_DECREF(list);
                Py_DECREF(item);
                return nullptr;
            }
        }

        return list;
    }

    return nullptr;
}

/**
 * @brief Retrieves the name of the neural network.
 *
 * @param self The object pointer.
 * @param args The argument list.
 * @return PyObject* Python string object containing the name of the neural network, or nullptr on failure.
 */
PyObject* NetworkAccessors::GetName(PyObject* self, PyObject* args) {
    if (auto pNetwork = parsePtr<Network*>(args, "neural network"); pNetwork) {
        std::string_view name = pNetwork->GetName();
        return PyUnicode_FromStringAndSize(name.data(), name.size());
    }
    return nullptr;
}

/**
 * @brief Retrieves the unit buffer from a neural network.
 *
 * @param self The object pointer.
 * @param args The argument list.
 * @return PyObject* PyCapsule containing the unit buffer as a float pointer, or nullptr on failure.
 */
PyObject* NetworkAccessors::GetUnitBuffer(PyObject* self, PyObject* args) {
    char const* layer = nullptr;
    if (auto [pNetwork, parseResult] = parsePtrAndOneValue<Network*, char const*>(args, layer, "neural network", "Os"); parseResult && pNetwork) {
        return PyCapsule_New(reinterpret_cast<void*>(pNetwork->GetUnitBuffer(std::string(layer))), "float", nullptr);
    }
    return nullptr;
}

/**
 * @brief Retrieves the delta buffer from a neural network.
 *
 * @param self The object pointer.
 * @param args The argument list.
 * @return PyObject* PyCapsule containing the delta buffer as a float pointer, or nullptr on failure.
 */
PyObject* NetworkAccessors::GetDeltaBuffer(PyObject* self, PyObject* args) {
    char const* layer = nullptr;
    if (auto [pNetwork, parseResult] = parsePtrAndOneValue<Network*, char const*>(args, layer, "neural network", "Os"); parseResult && pNetwork) {
        return PyCapsule_New(reinterpret_cast<void*>(pNetwork->GetDeltaBuffer(std::string(layer))), "float", nullptr);
    }
    return nullptr;
}

/**
 * @brief Retrieves the weight buffer from a neural network.
 *
 * @param self The object pointer.
 * @param args The argument list.
 * @return PyObject* PyCapsule containing the weight buffer as a float pointer, or nullptr on failure.
 */
PyObject* NetworkAccessors::GetWeightBuffer(PyObject* self, PyObject* args) {
    char const* inputLayer = nullptr;
    char const* outputLayer = nullptr;
    if (auto [pNetwork, parseResult] = parsePtrAndTwoValues<Network*, char const*, char const*>(args, inputLayer, outputLayer, "neural network", "Oss"); parseResult && pNetwork) {
        return PyCapsule_New(reinterpret_cast<void*>(pNetwork->GetWeightBuffer(std::string(inputLayer), std::string(outputLayer))), "float", nullptr);
    }
    return nullptr;
}

/**
 * @brief Retrieves the scratch buffer from a neural network.
 *
 * @param self The object pointer.
 * @param args The argument list.
 * @return PyObject* PyCapsule containing the scratch buffer as a float pointer, or nullptr on failure.
 */
PyObject* NetworkAccessors::GetScratchBuffer(PyObject* self, PyObject* args) {
    size_t size = 0;
    if (auto [pNetwork, parseResult] = parsePtrAndOneValue<Network*, size_t>(args, size, "neural network", "OI"); parseResult && pNetwork) {
        return PyCapsule_New(reinterpret_cast<void*>(pNetwork->GetScratchBuffer(size)), "float", nullptr);
    }
    return nullptr;
}

/**
 * @brief Retrieves the P2P send buffer from a neural network.
 *
 * @param self The object pointer.
 * @param args The argument list.
 * @return PyObject* PyCapsule containing the P2P send buffer as a float pointer, or nullptr on failure.
 */
PyObject* NetworkAccessors::GetP2PSendBuffer(PyObject* self, PyObject* args) {
    // Parse the argument to obtain the neural network pointer
    if (auto pNetwork = parsePtr<Network*>(args, "neural network"); pNetwork) {
        // Retrieve the P2P send buffer from the neural network
        auto buffer = pNetwork->GetP2PSendBuffer();
        if (buffer) {
            // Create a unique_ptr to manage the dynamically allocated float buffer
            auto deleter = [](void* p) { delete static_cast<float*>(p); };
            return PyCapsule_New(std::make_unique<float>(*buffer).release(), "float", deleter);
        }
    }

    return nullptr; // Return nullptr on failure
}

/**
 * @brief Retrieves the P2P receive buffer from a neural network.
 *
 * @param self The object pointer.
 * @param args The argument list.
 * @return PyObject* PyCapsule containing the P2P receive buffer as a float pointer, or nullptr on failure.
 */
PyObject* NetworkAccessors::GetP2PReceiveBuffer(PyObject* self, PyObject* args) {
    // Parse the argument to obtain the neural network pointer
    if (auto pNetwork = parsePtr<Network*>(args, "neural network"); pNetwork) {
        // Retrieve the P2P receive buffer from the neural network
        auto buffer = pNetwork->GetP2PReceiveBuffer();
        if (buffer) {
            // Create a unique_ptr to manage the dynamically allocated float buffer
            auto deleter = [](void* p) { delete static_cast<float*>(p); };
            return PyCapsule_New(std::make_unique<float>(*buffer).release(), "float", deleter);
        }
    }

    return nullptr; // Return nullptr on failure
}

/**
 * @brief Retrieves the P2P CPU buffer from a neural network.
 *
 * @param self The object pointer.
 * @param args The argument list.
 * @return PyObject* PyCapsule containing the P2P CPU buffer as a float pointer, or NULL on failure.
 */
PyObject* NetworkAccessors::GetP2PCPUBuffer(PyObject* self, PyObject* args) {
    // Parse the argument to obtain the neural network pointer
    if (auto pNetwork = parsePtr<Network*>(args, "neural network"); pNetwork) {
        // Retrieve the P2P CPU buffer from the neural network
        auto buffer = pNetwork->GetP2PCPUBuffer();
        if (buffer) {
            // Create a unique_ptr to manage the dynamically allocated float buffer
            auto deleter = [](void* p) { delete static_cast<float*>(p); };
            return PyCapsule_New(std::make_unique<float>(*buffer).release(), "float", deleter);
        }
    }

    return NULL; // Return NULL on failure
}

/**
 * @brief Retrieves the peer buffer from a neural network.
 *
 * @param self The object pointer.
 * @param args The argument list.
 * @return PyObject* PyCapsule containing the peer buffer as a float pointer, or NULL on failure.
 */
PyObject* NetworkAccessors::GetPeerBuffer(PyObject* self, PyObject* args) {
    // Parse the argument to obtain the neural network pointer
    if (auto pNetwork = parsePtr<Network*>(args, "neural network"); pNetwork) {
        // Retrieve the peer buffer from the neural network
        auto buffer = pNetwork->GetPeerBuffer();
        if (buffer) {
            // Create a unique_ptr to manage the dynamically allocated float buffer
            auto deleter = [](void* p) { delete static_cast<float*>(p); };
            return PyCapsule_New(std::make_unique<float>(*buffer).release(), "float", deleter);
        }
    }

    return NULL; // Return NULL on failure
}

/**
 * @brief Retrieves the peer back buffer from a neural network.
 *
 * @param self The object pointer.
 * @param args The argument list.
 * @return PyObject* PyCapsule containing the peer back buffer as a float pointer, or NULL on failure.
 */
PyObject* NetworkAccessors::GetPeerBackBuffer(PyObject* self, PyObject* args) {
    // Parse the argument to obtain the neural network pointer
    if (auto pNetwork = parsePtr<Network*>(args, "neural network"); pNetwork) {
        // Retrieve the peer back buffer from the neural network
        auto buffer = pNetwork->GetPeerBackBuffer();
        if (buffer) {
            // Create a unique_ptr to manage the dynamically allocated float buffer
            auto deleter = [](void* p) { delete static_cast<float*>(p); };
            return PyCapsule_New(std::make_unique<float>(*buffer).release(), "float", deleter);
        }
    }

    return NULL; // Return NULL on failure
}

/**
 * @brief Sets the clear velocity flag for a neural network.
 *
 * @param self The object pointer.
 * @param args The argument list.
 * @return PyObject* None on success, NULL on failure.
 */
PyObject* NetworkAccessors::SetClearVelocity(PyObject* self, PyObject* args) {
    int bClearVelocity = 0;

    // Parse the arguments to obtain the neural network pointer and the clear velocity flag
    if (auto [pNetwork, parseResult] = parsePtrAndOneValue<Network*, int>(args, bClearVelocity, "neural network", "Oi"); parseResult && pNetwork) {
        // Call the SetClearVelocity function on the neural network
        pNetwork->SetClearVelocity(bClearVelocity);
        Py_RETURN_NONE; // Return None on success
    }

    return NULL; // Return NULL on failure
}

/**
 * @brief Sets the training mode for a neural network.
 *
 * @param self The Python object representing the network accessors.
 * @param args The arguments passed to the function.
 * @return PyObject* The result of the function.
 */
PyObject* NetworkAccessors::SetTrainingMode(PyObject* self, PyObject* args) {
    char const* trainingMode = nullptr;
    if (!PyArg_ParseTuple(args, "s", &trainingMode)) {
        return nullptr;
    }
    
    // Find the training mode in the map
    auto it = stringToIntTrainingModeMap.find(trainingMode);
    if (it == stringToIntTrainingModeMap.end()) {
        PyErr_SetString(PyExc_RuntimeError, "SetTrainingMode received unsupported training mode enumerator string");
        return nullptr;
    }
    
    // Parse the neural network pointer and a value from the arguments
    Network* pNetwork = parsePtrAndOneValue<Network*, char const*>(args, trainingMode, "neural network", "Os");
    if (pNetwork == nullptr) {
        return nullptr;
    }
    
    // Set the training mode for the network
    pNetwork->SetTrainingMode(it->second);
    Py_RETURN_NONE;
}

#endif
