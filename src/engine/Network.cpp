#include "GpuTypes.h"
#include "NcExcptionWrap.h"
#include "Types.h"
#include "kernels.h"
#include "Utils.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <queue>
#include <set>
#include <cfloat>
#include <chrono>
#include <iostream>
#include <format>

using namespace netCDF;
using namespace netCDF::exceptions;

class NetworkDescriptor
{
public:
    NetworkDescriptor() = default;

    /**
     * @brief Enum representing the type of network.
     */
    Network::Kind _kind = Network::Kind::FeedForward;

    /**
     * @brief Enum representing the error function used by the network.
     */
    ErrorFunction _errorFunction = ErrorFunction::CrossEntropy;

    /**
     * @brief Flag indicating whether to shuffle indices during training.
     */
    bool _bShuffleIndices = true;

    /**
     * @brief The value of 'k' parameter used in maxout layers.
     */
    uint32_t _maxout_k = 2;

    /**
     * @brief The decay factor used for weight decay regularization.
     */
    Float _decay = 0.0;

    /**
     * @brief The value of 'k' parameter used in local response normalization layers.
     */
    uint32_t _LRN_k = 2;

    /**
     * @brief The value of 'n' parameter used in local response normalization layers.
     */
    uint32_t _LRN_n = 5;

    /**
     * @brief The value of 'alpha' parameter used in local response normalization layers.
     */
    Float _LRN_alpha = 0.0001;

    /**
     * @brief The value of 'beta' parameter used in local response normalization layers.
     */
    Float _LRN_beta = 0.75;

    /**
     * @brief The slope parameter used in rectified linear unit (ReLU) activation.
     */
    Float _RELUSlope = 1.0;

    /**
     * @brief The alpha parameter used in exponential linear unit (ELU) activation.
     */
    Float _ELUAlpha = 1.0;

    /**
     * @brief The lambda parameter used in scaled exponential linear unit (SELU) activation.
     */
    Float _SELULambda = 1.050701;

    /**
     * @brief Flag indicating whether to apply sparseness penalty during training.
     */
    bool _bSparsenessPenalty = false;

    /**
     * @brief The value of 'p' parameter used in sparseness penalty regularization.
     */
    Float _sparsenessPenalty_p = 0.0;

    /**
     * @brief The value of 'beta' parameter used in sparseness penalty regularization.
     */
    Float _sparsenessPenalty_beta = 0.0;

    /**
     * @brief Flag indicating whether to apply denoising during training.
     */
    bool _bDenoising = false;

    /**
     * @brief The value of 'p' parameter used in denoising regularization.
     */
    Float _denoising_p = 0.0;

    /**
     * @brief The boost factor for positive delta values during training.
     */
    Float _deltaBoost_one = 1.0;

    /**
     * @brief The boost factor for zero delta values during training.
     */
    Float _deltaBoost_zero = 1.0;

    /**
     * @brief The target value for positive samples in the softmax cross-entropy loss.
     */
    Float _SMCE_oneTarget = 0.9;

    /**
     * @brief The target value for negative samples in the softmax cross-entropy loss.
     */
    Float _SMCE_zeroTarget = 0.1;

    /**
     * @brief The scale factor for positive samples in the softmax cross-entropy loss.
     */
    Float _SMCE_oneScale = 1.0;

    /**
     * @brief The scale factor for negative samples in the softmax cross-entropy loss.
     */
    Float _SMCE_zeroScale = 1.0;

    /**
     * @brief The name of the network.
     */
    std::string _name = "";

    /**
     * @brief The name of the checkpoint file.
     */
    std::string _checkpoint_name = "checkpoint";

    /**
     * @brief The interval between checkpoint saves during training.
     */
    uint32_t _checkpoint_interval = 0;

    /**
     * @brief The number of epochs between checkpoint saves during training.
     */
    uint32_t _checkpoint_epochs = 0;

    /**
     * @brief Flag indicating whether convolutional layers have been calculated.
     */
    bool _bConvLayersCalculated = false;

    /**
     * @brief Vector of layer descriptors representing the network architecture.
     */
    std::vector<LayerDescriptor> _vLayerDescriptor;

    /**
     * @brief Vector of weight descriptors representing the network's weights.
     */
    std::vector<WeightDescriptor> _vWeightDescriptor;

    friend std::ostream& operator<< (std::ostream& out, NetworkDescriptor& d);
};

std::ostream& operator<< (std::ostream& out, NetworkDescriptor& d)
{
    using namespace std::chrono;
    
    /**
     * @brief Prints the details of the data object to the output stream.
     *
     * @param out The output stream.
     * @param d The data object.
     * @return The updated output stream.
     */
    out << std::format("Name:                    {}\n", d._name);

    /**
     * @brief Prints the details of the data object to the output stream.
     *
     * @param out The output stream.
     * @param d The data object.
     * @return The updated output stream.
     */
    out << std::format("Kind:                    {}\n", d._kind);

    /**
     * @brief Prints the details of the data object to the output stream.
     *
     * @param out The output stream.
     * @param d The data object.
     * @return The updated output stream.
     */
    out << std::format("bShuffleIndices          {}\n", std::boolalpha(d._bShuffleIndices));

    /**
     * @brief Prints the details of the data object to the output stream.
     *
     * @param out The output stream.
     * @param d The data object.
     * @return The updated output stream.
     */
    out << std::format("Error Function:          {}\n", d._errorFunction);

    /**
     * @brief Prints the details of the data object to the output stream.
     *
     * @param out The output stream.
     * @param d The data object.
     * @return The updated output stream.
     */
    out << std::format("MaxOut_k:                {}\n", d._maxout_k);

    /**
     * @brief Prints the details of the data object to the output stream.
     *
     * @param out The output stream.
     * @param d The data object.
     * @return The updated output stream.
     */
    out << std::format("LRN_k:                   {}\n", d._LRN_k);

    /**
     * @brief Prints the details of the data object to the output stream.
     *
     * @param out The output stream.
     * @param d The data object.
     * @return The updated output stream.
     */
    out << std::format("LRN_n:                   {}\n", d._LRN_n);

    /**
     * @brief Prints the details of the data object to the output stream.
     *
     * @param out The output stream.
     * @param d The data object.
     * @return The updated output stream.
     */
    out << std::format("LRN_beta:                {}\n", d._LRN_beta);

    /**
     * @brief Prints the details of the data object to the output stream.
     *
     * @param out The output stream.
     * @param d The data object.
     * @return The updated output stream.
     */
    out << std::format("LRN_alpha:               {}\n", d._LRN_alpha);

    /**
     * @brief Prints the details of the data object to the output stream.
     *
     * @param out The output stream.
     * @param d The data object.
     * @return The updated output stream.
     */
    out << std::format("bSparsenessPenalty:      {}\n", std::boolalpha(d._bSparsenessPenalty));

    /**
     * @brief Prints the details of the data object to the output stream.
     *
     * @param out The output stream.
     * @param d The data object.
     * @return The updated output stream.
     */
    out << std::format("sparsenessPenalty_beta:  {}\n", d._sparsenessPenalty_beta);

    /**
     * @brief Prints the details of the data object to the output stream.
     *
     * @param out The output stream.
     * @param d The data object.
     * @return The updated output stream.
     */
    out << std::format("sparsenessPenalty_p:     {}\n", d._sparsenessPenalty_p);

    /**
     * @brief Prints the details of the data object to the output stream.
     *
     * @param out The output stream.
     * @param d The data object.
     * @return The updated output stream.
     */
    out << std::format("bDenoising:              {}\n", std::boolalpha(d._bDenoising));

    /**
     * @brief Prints the details of the data object to the output stream.
     *
     * @param out The output stream.
     * @param d The data object.
     * @return The updated output stream.
     */
    out << std::format("denoising_p:             {}\n", d._denoising_p);

    /**
     * @brief Prints the details of the data object to the output stream.
     *
     * @param out The output stream.
     * @param d The data object.
     * @return The updated output stream.
     */
    out << std::format("deltaBoost_one:          {}\n", d._deltaBoost_one);

    /**
     * @brief Prints the details of the data object to the output stream.
     *
     * @param out The output stream.
     * @param d The data object.
     * @return The updated output stream.
     */
    out << std::format("deltaBoost_zero:         {}\n", d._deltaBoost_zero);

    /**
     * @brief Prints the details of the data object to the output stream.
     *
     * @param out The output stream.
     * @param d The data object.
     * @return The updated output stream.
     */
    out << std::format("SMCE_oneTarget:          {}\n", d._SMCE_oneTarget);

    /**
     * @brief Prints the details of the data object to the output stream.
     *
     * @param out The output stream.
     * @param d The data object.
     * @return The updated output stream.
     */
    out << std::format("SMCE_zeroTarget:         {}\n", d._SMCE_zeroTarget);

    /**
     * @brief Prints the details of the data object to the output stream.
     *
     * @param out The output stream.
     * @param d The data object.
     * @return The updated output stream.
     */
    out << std::format("SMCE_oneScale:           {}\n", d._SMCE_oneScale);

    /**
     * @brief Prints the details of the data object to the output stream.
     *
     * @param out The output stream.
     * @param d The data object.
     * @return The updated output stream.
     */
    out << std::format("SMCE_zeroScale:          {}\n", d._SMCE_zeroScale);

    /**
     * @brief Prints the details of the data object to the output stream.
     *
     * @param out The output stream.
     * @param d The data object.
     * @return The updated output stream.
     */
    out << std::format("checkpoint_name:         {}\n", d._checkpoint_name);

    /**
     * @brief Prints the details of the data object to the output stream.
     *
     * @param out The output stream.
     * @param d The data object.
     * @return The updated output stream.
     */
    out << std::format("checkpoint_interval:     {}\n", d._checkpoint_interval);

    /**
     * @brief Prints the layers of the data object to the output stream.
     *
     * @param out The output stream.
     * @param d The data object.
     * @return The updated output stream.
     */
    out << "\nLayers:\n";
    for (const auto& layer : d._vLayerDescriptor)
    {
        out << "Layer " << layer << '\n';
    }

    /**
     * @brief Prints the weights of the data object to the output stream.
     *
     * @param out The output stream.
     * @param d The data object.
     * @return The updated output stream.
     */
    out << "\nWeights:\n";
    for (const auto& weight : d._vWeightDescriptor)
    {
        out << "Weight " << weight << '\n';
    }
    return out;
}
/**
 * @brief Validates the network descriptor.
 *
 * @param d The network descriptor to validate.
 * @return True if the network descriptor is valid, false otherwise.
 */
bool ValidateNetworkDescriptor(NetworkDescriptor& d)
{
    return true;
}

/**
 * @brief Gets the LRN (Local Response Normalization) parameters of the network.
 *
 * @return A tuple containing the LRN parameters: (LRN_k, LRN_n, LRN_alpha, LRN_beta).
 */
std::tuple<Float, uint32_t, Float, Float> Network::GetLRN() const
{
    return {_LRN_k, _LRN_n, _LRN_alpha, _LRN_beta};
}

/**
 * @brief Gets the decay parameter of the network.
 *
 * @return A tuple containing the decay parameter.
 */
std::tuple<Float> Network::GetDecay() const
{
    return {_decay};
}

/**
 * @brief Gets the maxout parameter of the network.
 *
 * @return A tuple containing the maxout parameter.
 */
std::tuple<uint32_t> Network::GetMaxout() const
{
    return {_maxout_k};
}

/**
 * @brief Gets the sparseness penalty parameters of the network.
 *
 * @return A tuple containing the sparseness penalty parameters: (sparsenessPenalty_p, sparsenessPenalty_beta).
 */
std::tuple<Float, Float> Network::GetSparsenessPenalty() const
{
    return {_sparsenessPenalty_p, _sparsenessPenalty_beta};
}

/**
 * @brief Gets the denoising parameter of the network.
 *
 * @return A tuple containing the denoising parameter.
 */
std::tuple<Float> Network::GetDenoising() const
{
    return {_denoising_p};
}

/**
 * @brief Gets the delta boost parameters of the network.
 *
 * @return A tuple containing the delta boost parameters: (deltaBoost_one, deltaBoost_zero).
 */
std::tuple<Float, Float> Network::GetDeltaBoost() const
{
    return {_deltaBoost_one, _deltaBoost_zero};
}

/**
 * @brief Gets the SMCE (Sparse Multiclass Cross Entropy) parameters of the network.
 *
 * @return A tuple containing the SMCE parameters: (SMCE_oneTarget, SMCE_zeroTarget, SMCE_oneScale, SMCE_zeroScale).
 */
std::tuple<Float, Float, Float, Float> Network::GetSMCE() const
{
    return {_SMCE_oneTarget, _SMCE_zeroTarget, _SMCE_oneScale, _SMCE_zeroScale};
}

/**
 * @brief Gets the shuffle indices flag of the network.
 *
 * @return A tuple containing the shuffle indices flag.
 */
std::tuple<bool> Network::GetShuffleIndices() const
{
    return {_bShuffleIndices};
}

/**
 * @brief Gets the checkpoint information of the network.
 *
 * @return A tuple containing the checkpoint information: (checkpoint_name, checkpoint_interval).
 */
std::tuple<std::string, int32_t> Network::GetCheckPoint() const
{
    return {_checkpoint_name, _checkpoint_interval};
}

/**
 * @brief Gets the layer with the specified name from the network.
 *
 * @param layer The name of the layer.
 * @return A pointer to the layer if found, nullptr otherwise.
 */
Layer* Network::GetLayer(const std::string& layer) const
{
    auto itr = _mLayer.find(layer);
    if (itr == _mLayer.end())
    {
        if (getGpu()._id == 0)
        {
            std::cout("Network::GetLayerDimensions: Unknown layer %s.\n", layer.c_str());
        }
        return nullptr;
    }
    return itr->second;
}
/**
 * @brief Retrieves the layers of the specified kind from the network and stores them in the provided vector.
 *
 * @param layerKind The kind of layers to retrieve.
 * @param layers The vector to store the retrieved layers.
 * @return An iterator pointing to the end of the stored layers in the vector.
 */
auto Network::GetLayers(Layer::Kind layerKind, std::vector<const Layer*>& layers) const -> std::vector<const Layer*>::iterator
{
    int count = 0;
    for (const auto& layerName : Network::GetLayers())
    {
        const Layer* layer = Network::GetLayer(layerName);
        if (layerKind == layer->_kind)
        {
            layers.push_back(layer);
            ++count;
        }
    }
    return layers.end() - count;
}

/**
 * @brief Retrieves a scratch buffer of the specified size from the network.
 *
 * @param size The size of the scratch buffer.
 * @return A pointer to the scratch buffer.
 */
Float* Network::GetScratchBuffer(size_t size)
{
    if (size > _scratchBufferSize)
    {
        _pbScratchBuffer.reset(new GpuBuffer<Float>(size));
        _scratchBufferSize = size;
    }
    return _pbScratchBuffer->get();
}

/**
 * @brief Sets the size of the CUDA workspace for the network.
 *
 * @param size The size of the CUDA workspace.
 */
void Network::SetCUDWorkspace(size_t size)
{
    if (size > _maxCUDWorkspaceSize)
    {
        _maxCUDWorkspaceSize = size;
    }
}

/**
 * @brief Retrieves the P2P (Peer-to-Peer) send buffer from the network.
 *
 * @return A pointer to the P2P send buffer.
 */
Float* Network::GetP2PSendBuffer()
{
    return _pbP2PBuffer[_sendIndex]->get();
}

/**
 * @brief Retrieves the P2P (Peer-to-Peer) receive buffer from the network.
 *
 * @return A pointer to the P2P receive buffer.
 */
Float* Network::GetP2PReceiveBuffer()
{
    return _pbP2PBuffer[_receiveIndex]->get();
}

/**
 * @brief Retrieves the P2P (Peer-to-Peer) CPU buffer from the network.
 *
 * @return A pointer to the P2P CPU buffer.
 */
Float* Network::GetP2PCPUBuffer()
{
    return _pCPUBuffer.get();
}

/**
 * @brief Retrieves the peer buffer from the network.
 *
 * @return A pointer to the peer buffer.
 */
Float* Network::GetPeerBuffer()
{
    return _pPeerBuffer[_receiveIndex].get();
}

/**
 * @brief Retrieves the peer back buffer from the network.
 *
 * @return A pointer to the peer back buffer.
 */
Float* Network::GetPeerBackBuffer()
{
    return _pPeerBuffer[_sendIndex].get();
}
bool Network::SetLRN(Float k, uint32_t n, Float alpha, Float beta)
{
    _LRN_k = k;
    _LRN_n = n;
    _LRN_alpha = alpha;
    _LRN_beta = beta;
    _bDirty = true;

    if (getGpu()._id == 0)
        std::cout << "Network::SetLRN: k set to " << k << ", n set to " << n << ", alpha set to " << alpha << ", beta set to " << beta << ".\n";

    return true;
}

bool Network::SetDecay(Float decay)
{
    if (decay >= 0.0)
    {
        _decay = decay;
        if (getGpu()._id == 0)
            std::cout << "Network::SetDecay: decay set to " << decay << ".\n";
        return true;
    }
    else
    {
        if (getGpu()._id == 0)
            std::cout << "Network::SetDecay: invalid decay rate (<0.0) " << decay << ".\n";
        return false;
    }
}

bool Network::SetMaxout(uint32_t k)
{
    if (k != _maxout_k)
    {
        _maxout_k = k;
        _bDirty = true;
    }

    if (getGpu()._id == 0)
        std::cout << "Network::SetMaxout: k set to " << k << ".\n";

    return true;
}

bool Network::SetSparsenessPenalty(Float p, Float beta)
{
    if (p < 0.0 || p > 1.0)
    {
        if (getGpu()._id == 0)
            std::cout << "Network::SetSparsenessPenalty: Target sparseness must be >=0 and <=1.\n";
        return false;
    }

    _sparsenessPenalty_p = p;
    _sparsenessPenalty_beta = beta;
    _bSparsenessPenalty = (beta > 0.0);
    _bDirty = true;

    if (getGpu()._id == 0)
        std::cout << "Network::SetSparsenessPenalty: p set to " << p << ", beta set to " << beta << ".\n";

    return true;
}
bool Network::SetDenoising(Float p)
{
    if (p < 0.0 || p >= 1.0)
    {
        if (getGpu()._id == 0)
            std::cout << "Network::SetDenoising: Denoising probability must be >=0 and <1.\n";
        return false;
    }

    if (_denoising_p != p)
    {
        _denoising_p = p;
        _bDenoising = (_denoising_p > 0.0);
        _bDirty = true;
    }

    if (getGpu()._id == 0)
        std::cout << "Network::SetDenoising: p set to " << p << ".\n";

    return true;
}

bool Network::SetDeltaBoost(Float one, Float zero)
{
    if (one < 0.0)
    {
        if (getGpu()._id == 0)
            std::cout << "Network::SetDeltaBoost: Illegal value for one (" << one << ").\n";
        return false;
    }
    else if (zero < 0.0)
    {
        if (getGpu()._id == 0)
            std::cout << "Network::SetDeltaBoost: Illegal value for zero (" << zero << ").\n";
        return false;
    }

    _deltaBoost_one = one;
    _deltaBoost_zero = zero;
    _bDirty = true;

    if (getGpu()._id == 0)
        std::cout << "Network::SetDeltaBoost: one set to " << one << ", zero set to " << zero << ".\n";

    return true;
}

bool Network::SetSMCE(Float oneTarget, Float zeroTarget, Float oneScale, Float zeroScale)
{
    if (oneTarget < 0.0 || oneTarget > 1.0)
    {
        if (getGpu()._id == 0)
            std::cout << "Network::SetSMCE: Illegal value for oneTarget (" << oneTarget << ").\n";
        return false;
    }
    else if (zeroTarget < 0.0 || zeroTarget > 1.0)
    {
        if (getGpu()._id == 0)
            std::cout << "Network::SetSMCE: Illegal value for zeroTarget (" << zeroTarget << ").\n";
        return false;
    }
    else if (oneScale < 0.0)
    {
        if (getGpu()._id == 0)
            std::cout << "Network::SetSMCE: Illegal value for oneScale (" << oneScale << ").\n";
        return false;
    }
    else if (zeroScale < 0.0)
    {
        if (getGpu()._id == 0)
            std::cout << "Network::SetSMCE: Illegal value for zeroScale (" << zeroScale << ").\n";
        return false;
    }

    _SMCE_oneTarget = oneTarget;
    _SMCE_zeroTarget = zeroTarget;
    _SMCE_oneScale = oneScale;
    _SMCE_zeroScale = zeroScale;
    _bDirty = true;

    if (getGpu()._id == 0)
        std::cout << "Network::SetSMCE: oneTarget set to " << oneTarget << ", zeroTarget set to " << zeroTarget << ", oneScale set to " << oneScale << ", zeroScale set to " << zeroScale << ".\n";

    return true;
}
bool Network::SetCheckpoint(std::string name, int32_t interval)
{
    _checkpoint_name = std::move(name);
    _checkpoint_interval = interval;

    if (getGpu()._id == 0) {
        std::cout << "Network::SetCheckPoint: filename set to " << _checkpoint_name << ", interval set to " << interval << " epochs.\n";
    }
    return true;
}

Network::Network(NetworkDescriptor& d, uint32_t batch) :
/**
 * @brief Initializes a new instance of the class.
 *
 * @param d The source instance to copy the values from.
 */
_name(d._name),

/**
 * @brief Initializes a new instance of the class.
 *
 * @param d The source instance to copy the values from.
 */
_kind(d._kind),

/**
 * @brief Initializes a new instance of the class with the specified mode.
 *
 * @param batch The batch size.
 */
_mode(Prediction),

/**
 * @brief Initializes a new instance of the class with the specified mode.
 *
 * @param SGD The training mode.
 */
_trainingMode(SGD),

/**
 * @brief Initializes a new instance of the class with the specified batch.
 *
 * @param batch The batch size.
 */
_batch(batch),

/**
 * @brief Initializes a new instance of the class with the specified batch.
 *
 * @param batch The batch size.
 */
_localBatch(batch),

/**
 * @brief Initializes a new instance of the class with the specified position.
 *
 * @param position The starting position.
 */
_position(0),

/**
 * @brief Initializes a new instance of the class with the specified position.
 *
 * @param position The starting position.
 */
_localPosition(0),

/**
 * @brief Initializes a new instance of the class with the specified shuffle indices.
 *
 * @param bShuffleIndices Flag indicating whether to shuffle indices.
 */
_bShuffleIndices(d._bShuffleIndices),

/**
 * @brief Initializes a new instance of the class with the specified shuffle indices.
 *
 * @param shuffleIndices The shuffle indices.
 */
_shuffleIndices(0),

/**
 * @brief Initializes a new instance of the class with the specified shuffle index pointer.
 *
 * @param pShuffleIndex The shuffle index pointer.
 */
_pShuffleIndex(nullptr),

/**
 * @brief Initializes a new instance of the class with the specified shuffle index sort.
 *
 * @param pShuffleIndexSort The shuffle index sort.
 */
_pShuffleIndexSort(),

/**
 * @brief Initializes a new instance of the class with the specified shuffle index buffer.
 *
 * @param pbShuffleIndex The shuffle index buffer.
 */
_pbShuffleIndex(),

/**
 * @brief Initializes a new instance of the class with the specified examples found flag.
 *
 * @param bExamplesFound Flag indicating whether examples are found.
 */
_bExamplesFound(false),

/**
 * @brief Initializes a new instance of the class with the specified data loaded flag.
 *
 * @param bAllDataLoaded Flag indicating whether all data is loaded.
 */
_bAllDataLoaded(true),

/**
 * @brief Initializes a new instance of the class with the specified examples.
 *
 * @param examples The examples.
 */
_examples(0),

/**
 * @brief Initializes a new instance of the class with the specified error function.
 *
 * @param errorFunction The error function.
 */
_errorFunction(d._errorFunction),

/**
 * @brief Initializes a new instance of the class with the specified decay.
 *
 * @param decay The decay value.
 */
_decay(d._decay),

/**
 * @brief Initializes a new instance of the class with the specified LRN parameters.
 *
 * @param LRN_k The LRN k parameter.
 * @param LRN_n The LRN n parameter.
 * @param LRN_alpha The LRN alpha parameter.
 * @param LRN_beta The LRN beta parameter.
 */
_LRN_k(d._LRN_k),
_LRN_n(d._LRN_n),
_LRN_alpha(d._LRN_alpha),
_LRN_beta(d._LRN_beta),

/**
 * @brief Initializes a new instance of the class with the specified maxout k.
 *
 * @param maxout_k The maxout k value.
 */
_maxout_k(d._maxout_k),

/**
 * @brief Initializes a new instance of the class with the specified sparseness penalty flag.
 *
 * @param bSparsenessPenalty Flag indicating whether to apply sparseness penalty.
 */
_bSparsenessPenalty(d._bSparsenessPenalty),

/**
 * @brief Initializes a new instance of the class with the specified sparseness penalty beta.
 *
 * @param sparsenessPenalty_beta The sparseness penalty beta value.
 */
_sparsenessPenalty_beta(d._sparsenessPenalty_beta),

/**
 * @brief Initializes a new instance of the class with the specified sparseness penalty p.
 *
 * @param sparsenessPenalty_p The sparseness penalty p value.
 */
_sparsenessPenalty_p(d._sparsenessPenalty_p),

/**
 * @brief Initializes a new instance of the class with the specified denoising flag.
 *
 * @param bDenoising Flag indicating whether to apply denoising.
 */
_bDenoising(d._bDenoising),

/**
 * @brief Initializes a new instance of the class with the specified denoising p.
 *
 * @param denoising_p The denoising p value.
 */
_denoising_p(d._denoising_p),

/**
 * @brief Initializes a new instance of the class with the specified delta boost one.
 *
 * @param deltaBoost_one The delta boost one value.
 */
_deltaBoost_one(d._deltaBoost_one),

/**
 * @brief Initializes a new instance of the class with the specified delta boost zero.
 *
 * @param deltaBoost_zero The delta boost zero value.
 */
_deltaBoost_zero(d._deltaBoost_zero),

/**
 * @brief Initializes a new instance of the class with the specified SMCE one target.
 *
 * @param SMCE_oneTarget The SMCE one target value.
 */
_SMCE_oneTarget(d._SMCE_oneTarget),

/**
 * @brief Initializes a new instance of the class with the specified SMCE zero target.
 *
 * @param SMCE_zeroTarget The SMCE zero target value.
 */
_SMCE_zeroTarget(d._SMCE_zeroTarget),

/**
 * @brief Initializes a new instance of the class with the specified SMCE one scale.
 *
 * @param SMCE_oneScale The SMCE one scale value.
 */
_SMCE_oneScale(d._SMCE_oneScale),

/**
 * @brief Initializes a new instance of the class with the specified SMCE zero scale.
 *
 * @param SMCE_zeroScale The SMCE zero scale value.
 */
_SMCE_zeroScale(d._SMCE_zeroScale),

/**
 * @brief Initializes a new instance of the class with the specified checkpoint name.
 *
 * @param checkpoint_name The checkpoint name.
 */
_checkpoint_name(d._checkpoint_name),

/**
 * @brief Initializes a new instance of the class with the specified checkpoint interval.
 *
 * @param checkpoint_interval The checkpoint interval.
 */
_checkpoint_interval(d._checkpoint_interval),

/**
 * @brief Initializes a new instance of the class with the specified checkpoint epochs.
 *
 * @param checkpoint_epochs The checkpoint epochs.
 */
_checkpoint_epochs(0),

/**
 * @brief Initializes a new instance of the class with the specified epochs.
 *
 * @param epochs The number of epochs.
 */
_epochs(0),

/**
 * @brief Initializes a new instance of the class with the specified batches.
 *
 * @param batches The number of batches.
 */
_batches(0),

/**
 * @brief Initializes a new instance of the class with the specified clear velocity flag.
 *
 * @param bClearVelocity Flag indicating whether to clear velocity.
 */
_bClearVelocity(true),

/**
 * @brief Initializes a new instance of the class with the specified dirty flag.
 *
 * @param bDirty Flag indicating whether the instance is dirty.
 */
_bDirty(true),

/**
 * @brief Initializes a new instance of the class with the specified maximum stride.
 *
 * @param maxStride The maximum stride.
 */
_maxStride(0),

/**
 * @brief Initializes a new instance of the class with the specified scratch buffer size.
 *
 * @param scratchBufferSize The size of the scratch buffer.
 */
_scratchBufferSize(0),

/**
 * @brief Initializes a new instance of the class with the specified scratch buffer.
 *
 * @param pbScratchBuffer The scratch buffer.
 */
_pbScratchBuffer(),

/**
 * @brief Initializes a new instance of the class with the specified peer buffer.
 *
 * @param pPeerBuffer The peer buffer.
 */
_pPeerBuffer{nullptr, nullptr},

/**
 * @brief Initializes a new instance of the class with the specified P2P buffer.
 *
 * @param pbP2PBuffer The P2P buffer.
 */
_pbP2PBuffer(),

/**
 * @brief Initializes a new instance of the class with the specified CPU buffer.
 *
 * @param pCPUBuffer The CPU buffer.
 */
_pCPUBuffer(),

/**
 * @brief Initializes a new instance of the class with the specified send and receive indices.
 *
 * @param sendIndex The send index.
 * @param receiveIndex The receive index.
 */
_sendIndex(0),
_receiveIndex(1),

/**
 * @brief Initializes a new instance of the class with the specified CUD workspace size.
 *
 * @param CUDWorkspaceSize The CUD workspace size.
 */
_CUDWorkspaceSize(0),

/**
 * @brief Initializes a new instance of the class with the specified maximum CUD workspace size.
 *
 * @param maxCUDWorkspaceSize The maximum CUD workspace size.
 */
_maxCUDWorkspaceSize(0),

/**
 * @brief Initializes a new instance of the class with the specified CUD workspace buffer.
 *
 * @param pbCUDWorkspace The CUD workspace buffer.
 */
_pbCUDWorkspace(),

/**
 * @brief Initializes a new instance of the class with the specified verbose flag.
 *
 * @param verbose Flag indicating whether to enable verbosity.
 */
_verbose(false)

{

    for (const auto& l : d._vLayerDescriptor)
    {
        _vLayer.push_back(new Layer(l, batch));
        _mLayer[_vLayer.back()->_name] = _vLayer.back();

        if (_vLayer.back()->_kind == Layer::Kind::Input)
        {
            _vInputLayer.push_back(_vLayer.back());
        }
        else if (_vLayer.back()->_kind == Layer::Kind::Output)
        {
            _vOutputLayer.push_back(_vLayer.back());
        }
    }

    if (getGpu()._id == 0)
    {
        std::cout << "Network::Network: " << _vInputLayer.size() << " input layer" << (_vInputLayer.size() > 1 ? "s" : "") << std::endl;
        std::cout << "Network::Network: " << _vOutputLayer.size() << " output layer" << (_vOutputLayer.size() > 1 ? "s" : "") << std::endl;
    }

    for (auto l : _vLayer)
    {
        for (auto s : l->_vSkip)
        {
            Layer* pLayer = _mLayer[s];

            if (pLayer->_stride != l->_stride)
            {
                if (getGpu()._id == 0)
                    std::cout << "Network::Network: Layer dimensions do not match for skip connection between layer " << l->_name << " and " << pLayer->_name << ".\n";
                getGpu().Shutdown();
                exit(-1);
            }

            l->_vIncomingSkip.push_back(pLayer);
            pLayer->_vOutgoingSkip.push_back(l);
        }

        if (l->_type == Layer::Type::Pooling)
        {
            for (auto s : l->_vSource)
            {
                Layer* pLayer = _mLayer[s];
                l->_vIncomingLayer.push_back(pLayer);
                pLayer->_vOutgoingLayer.push_back(l);

                if ((l->_poolingFunction == PoolingFunction::DotProduct) ||
                    (l->_poolingFunction == PoolingFunction::Cosine) ||
                    (l->_poolingFunction == PoolingFunction::Maxout))
                {
                    if (pLayer->_stride != l->_vIncomingLayer[0]->_stride)
                    {
                        if (getGpu()._id == 0)
                        {
                            std::cout << "Network::Network: All source layer dimensions must match for " << l->_poolingFunction << " layer " << l->_name << std::endl;
                        }
                        getGpu().Shutdown();
                        exit(-1);
                    }
                }
            }
        }
    }

    for (auto wd : d._vWeightDescriptor)
    {
        Layer* pInputLayer = _mLayer[wd._inputLayer];
        Layer* pOutputLayer = _mLayer[wd._outputLayer];
        Weight* pWeight = new Weight(*pInputLayer, *pOutputLayer, wd._bShared, wd._bTransposed, wd._bLocked, wd._norm);
        _vWeight.push_back(pWeight);

        if ((wd._vWeight.size() == 0) || (wd._vBias.size() == 0))
        {
            pWeight->Randomize();
        }

        if (!wd._bShared && (wd._vWeight.size() != 0))
        {
            if (getGpu()._numprocs > 1)
            {
                Float* pDst = pWeight->_vWeight.data();
                uint32_t outgoingSize = pOutputLayer->_stride * 3;
                uint32_t incomingSize = pInputLayer->_stride * 2;

                if (outgoingSize > incomingSize)
                {
                    Float* pSrc = wd._vWeight.data() + pOutputLayer->_minX;
                    for (size_t i = 0; i < pInputLayer->_stride; i++)
                    {
                        /**
                         * @brief Copy data from source buffer to destination buffer using memcpy.
                         *
                         * @param pDst Pointer to the destination buffer.
                         * @param pSrc Pointer to the source buffer.
                         * @param size The number of elements to copy, in terms of Float type.
                         */
                        std::memcpy(pDst, pSrc, pOutputLayer->_localStride * sizeof(Float));

                        /**
                         * @brief Advance the source buffer pointer by the stride of the output layer.
                         *
                         * @param pSrc Pointer to the source buffer.
                         */
                        pSrc += pOutputLayer->_stride;

                        /**
                         * @brief Advance the destination buffer pointer by the local stride of the output layer.
                         *
                         * @param pDst Pointer to the destination buffer.
                         */
                        pDst += pOutputLayer->_localStride;
                    }
                }
                else
                {
                    /**
                     * @brief Calculate the source pointer by adding an offset to the weight data pointer.
                     *
                     * @param wd The weight descriptor containing the weight data.
                     * @param pInputLayer Pointer to the input layer.
                     * @param pOutputLayer Pointer to the output layer.
                     */
                    Float* pSrc = wd._vWeight.data() + pInputLayer->_minX * pOutputLayer->_stride;

                    /**
                     * @brief Copy data from the source buffer to the destination buffer using memcpy.
                     *
                     * @param pDst Pointer to the destination buffer.
                     * @param pSrc Pointer to the source buffer.
                     * @param size The number of elements to copy, calculated as the product of input layer's local stride and output layer's stride, in terms of Float type.
                     */
                    std::memcpy(pDst, pSrc, pInputLayer->_localStride * pOutputLayer->_stride * sizeof(Float));
                }
            }
            else
            {
                pWeight->_vWeight = wd._vWeight;
            }
            pWeight->_pbWeight->Upload(pWeight->_vWeight.data());
        }

        if (wd._vBias.size() != 0)
        {
            if (getGpu()._numprocs > 1)
            {
                /**
                 * @brief Calculate the source pointer by adding an offset to the bias data pointer.
                 *
                 * @param wd The weight descriptor containing the bias data.
                 * @param pOutputLayer Pointer to the output layer.
                 */
                Float* pSrc = wd._vBias.data() + pOutputLayer->_minX;

                /**
                 * @brief Calculate the destination pointer as the bias data pointer of the weight.
                 *
                 * @param pDst Pointer to the destination buffer.
                 * @param pWeight Pointer to the weight.
                 */
                Float* pDst = pWeight->_vBias.data();

                /**
                 * @brief Copy data from the source buffer to the destination buffer using memcpy.
                 *
                 * @param pDst Pointer to the destination buffer.
                 * @param pSrc Pointer to the source buffer.
                 * @param size The number of elements to copy, in terms of Float type, which is determined by the output layer's local stride.
                 */
                std::memcpy(pDst, pSrc, pOutputLayer->_localStride * sizeof(Float));
            }
            else
            {
                pWeight->_vBias = wd._vBias;
            }
            pWeight->_pbBias->Upload(pWeight->_vBias.data());
        }
    }

    for (uint32_t i = 0; i < d._vWeightDescriptor.size(); i++)
    {
        WeightDescriptor& wd = d._vWeightDescriptor[i];
        if (wd._bShared)
        {
            Weight* pWeight = _vWeight[i];
            std::string inputLayer = wd._sourceInputLayer;
            std::string outputLayer = wd._sourceOutputLayer;
            bool bFound = false;
            for (int j = 0; j < _vWeight.size(); j++)
            {
                if (!(_vWeight[j]->_bShared) &&
                    (_vWeight[j]->_inputLayer._name == inputLayer) &&
                    (_vWeight[j]->_outputLayer._name == outputLayer))
                {
                    if (wd._bTransposed)
                    {
                        if (wd._length > 1)
                        {
                            if (getGpu()._id == 0)
                                std::cout << "Network::Network: Can't transpose 3D weight matrix for shared weights between layers " <<
                                _vWeight[i]->_inputLayer._name << " and " << _vWeight[i]->_outputLayer._name << std::endl;
                            getGpu().Shutdown();
                            exit(-1);
                        }

                        if ((_vWeight[i]->_width != _vWeight[j]->_height) || (_vWeight[i]->_height != _vWeight[j]->_width))
                        {
                            if (getGpu()._id == 0)
                                /**
                                 * @brief Print an error message when the transposed dimensions for shared weights between layers do not match.
                                 *
                                 * @param _vWeight The vector of weights.
                                 * @param i The index of the weight being checked.
                                 */
                                std::cout << "Network::Network: Transposed dimensions for shared weights between layers " <<
                                _vWeight[i]->_inputLayer._name << " and " << _vWeight[i]->_outputLayer._name << " do not match" << std::endl;
                            }

                            /**
                             * @brief Shut down the GPU.
                             */
                            getGpu().Shutdown();

                            /**
                             * @brief Exit the program with a non-zero status code.
                             */
                            exit(-1);
                        }
                    }
                    else if ((_vWeight[i]->_width != _vWeight[j]->_width) ||
                             (_vWeight[i]->_height != _vWeight[j]->_height) ||
                             (_vWeight[i]->_length != _vWeight[j]->_length))
                    {
                        if (getGpu()._id == 0)
                            std::cout << "Network::Network: Dimensions for shared weights between layers " <<
                            _vWeight[i]->_inputLayer._name << " and " << _vWeight[i]->_outputLayer._name << " do not match" << std::endl;
                        getGpu().Shutdown();
                        exit(-1);
                    }

                    _vWeight[i]->_pSharedWeight = _vWeight[j];
                    if (_vWeight[j]->_sharingCount == 1)
                        _vSharedWeight.push_back(_vWeight[j]);
                    _vWeight[j]->_sharingCount++;
                    bFound = true;
                    break;
                }
            }

            if (!bFound)
            {
                if (getGpu()._id == 0)
                    std::cout << "Network::Network: Unable to locate shared weights for connection between layers " <<
                    _vWeight[i]->_inputLayer._name << " and " << _vWeight[i]->_outputLayer._name << ".\n";
                getGpu().Shutdown();
                exit(-1);
            }
        }
    }

    CalculatePropagationOrder();
}
#include <iostream>

void Network::Randomize()
{
    for (auto pw : _vWeight)
        pw->Randomize();
}

void Network::SetBatch(uint32_t batch)
{
    if (batch % getGpu()._numprocs)
    {
        if (getGpu()._id == 0)
            std::cout << "Network::SetBatch: Batch size must be a multiple of process count.\n";
        return;
    }

    if (batch != _batch)
    {
        _batch = batch;
        for (auto pL : _vLayer)
        {
            pL->SetBatch(batch);
        }

        _bDirty = true;
        if (getGpu()._id == 0)
            std::cout << "Network::SetBatch: Batch size set to " << _batch << ".\n";
    }
}

uint32_t Network::GetBatch() const
{
    return _batch;
}

uint32_t Network::GetExamples() const
{
    return _examples;
}

void Network::SetShuffleIndices(bool bShuffleIndices)
{
    if (_bShuffleIndices != bShuffleIndices)
    {
        _bShuffleIndices = bShuffleIndices;
        _bDirty = true;
    }

    if (getGpu()._id == 0)
        std::cout << "Network::SetShuffleIndices: Index shuffling is now " << (_bShuffleIndices ? "on" : "off") << "\n";
}

uint32_t Network::GetPosition() const
{
    return _position;
}

void Network::SetPosition(uint32_t position)
{
    if (_bExamplesFound)
    {
        if (position < _examples)
            _position = position;
        else if (getGpu()._id == 0)
            std::cout << "Network::SetPosition: Invalid position setting: " << position << ", maximum " << _examples << "\n";
    }
    else if (getGpu()._id == 0)
    {
        std::cout << "Network::SetPosition: Illegal attempt to set position without examples count information.\n";
    }
}
bool Network::LockWeights(const std::string& inputLayer, const std::string& outputLayer)
{
    auto pInputLayer = _mLayer.find(inputLayer);
    auto pOutputLayer = _mLayer.find(outputLayer);

    if (pInputLayer == _mLayer.end())
    {
        if (getGpu()._id == 0)
            std::cout << "Network::LockWeights: Unable to find input layer " << inputLayer << ".\n";
        return false;
    }

    if (pOutputLayer == _mLayer.end())
    {
        if (getGpu()._id == 0)
            std::cout << "Network::LockWeights: Unable to find output layer " << outputLayer << ".\n";
        return false;
    }

    auto weightMatrix = std::ranges::find_if(_vWeight, [&](const auto& weight) {
        return (weight->_inputLayer._name == pInputLayer->second->_name) &&
               (weight->_outputLayer._name == pOutputLayer->second->_name);
    });

    if (weightMatrix != _vWeight.end())
    {
        (*weightMatrix)->Lock();
        return true;
    }

    if (getGpu()._id == 0)
        std::cout << "Network::LockWeights: Unable to find weight matrix between input layer "
                  << inputLayer << " and output layer " << outputLayer << ".\n";
    return false;
}

bool Network::UnlockWeights(const std::string& inputLayer, const std::string& outputLayer)
{
    auto pInputLayer = _mLayer.find(inputLayer);
    auto pOutputLayer = _mLayer.find(outputLayer);

    if (pInputLayer == _mLayer.end())
    {
        if (getGpu()._id == 0)
            std::cout << "Network::UnlockWeights: Unable to find input layer " << inputLayer << ".\n";
        return false;
    }

    if (pOutputLayer == _mLayer.end())
    {
        if (getGpu()._id == 0)
            std::cout << "Network::UnlockWeights: Unable to find output layer " << outputLayer << ".\n";
        return false;
    }

    auto weightMatrix = std::ranges::find_if(_vWeight, [&](const auto& weight) {
        return (weight->_inputLayer._name == pInputLayer->second->_name) &&
               (weight->_outputLayer._name == pOutputLayer->second->_name);
    });

    if (weightMatrix != _vWeight.end())
    {
        (*weightMatrix)->Unlock();
        return true;
    }

    if (getGpu()._id == 0)
        std::cout << "Network::UnlockWeights: Unable to find weight matrix between input layer "
                  << inputLayer << " and output layer " << outputLayer << ".\n";
    return false;
}

void Network::SetTrainingMode(TrainingMode mode)
{
    if (_trainingMode != mode)
    {
        _trainingMode = mode;
        _bDirty = true;
    }

    if (getGpu()._id == 0)
        std::cout << "Network::SetTrainingMode: Optimizer is now " << _trainingMode << '\n';
}

void Network::RefreshShuffleBuffers()
{
    if (_bAllDataLoaded && _bShuffleIndices && (_mode == Training))
    {
        if (_shuffleIndices != _examples)
        {
            if (getGpu()._id == 0)
            {
                _pShuffleIndexSort.reset();
            }
            else
            {
                _pbShuffleIndex.reset();
            }

            _shuffleIndices = _examples;

            if (getGpu()._id == 0)
            {
                _pShuffleIndexSort = std::make_unique<GpuSort<uint32_t, uint32_t>>(_shuffleIndices);
                _pShuffleIndex = _pShuffleIndexSort->GetValuePointer();

                uint32_t stride = ((_shuffleIndices + 511) >> 9) << 9;
                std::vector<uint32_t> vIndex(stride * 2);
                for (uint32_t i = 0; i < _examples; i++)
                {
                    vIndex[i] = i;
                }
                _pShuffleIndexSort->GetValueBuffer()->Upload(vIndex.data());
            }
            else
            {
                _pbShuffleIndex = std::make_unique<GpuBuffer<uint32_t>>(_shuffleIndices);
                _pShuffleIndex = _pbShuffleIndex->_pDevData;
            }
        }
    }
}

void Network::ShuffleIndices()
{
    if (getGpu()._id == 0)
    {
        uint32_t stride = ((_shuffleIndices + 511) >> 9) << 9;
        std::vector<uint32_t> vIndex(stride * 2);
        for (uint32_t i = 0; i < _examples; i++)
        {
            vIndex[i] = i;
        }
        _pShuffleIndexSort->GetValueBuffer()->Upload(vIndex.data());

        curandGenerate(getGpu()._RNG, _pShuffleIndexSort->GetKeyPointer(), _shuffleIndices);

        _pShuffleIndexSort->Sort();
    }

    if (getGpu()._numprocs > 1)
    {
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        P2P_Bcast(_pShuffleIndex, _examples * sizeof(uint32_t));
    }
}

void Network::RefreshState()
{
    if (!_bAllDataLoaded)
    {
        _bAllDataLoaded = true;

        for (auto l : _vInputLayer)
        {
            if (l->_pDataSet == nullptr)
            {
                if (getGpu()._id == 0)
                    std::cout << "Network::RefreshState: Missing data set " << l->_dataSet << " for input layer " << l->_name << '\n';
                _bAllDataLoaded = false;
            }
        }

        if (_mode != Prediction)
        {
            for (auto l : _vOutputLayer)
            {
                if (l->_pDataSet == nullptr)
                {
                    if (getGpu()._id == 0)
                        std::cout << "Network::RefreshState: Missing data set " << l->_dataSet << " for output layer " << l->_name << '\n';
                    _bAllDataLoaded = false;
                }
            }
        }
    }

    if (_bDirty)
    {
        for (auto l : _vLayer)
        {
            if (l->_bDirty)
            {
                l->RefreshState(this, _trainingMode, _mode == Validation);
            }
        }

        for (auto w : _vWeight)
        {
            w->RefreshState(this, _trainingMode);
        }

        RefreshShuffleBuffers();
    }

    if (getGpu()._numprocs > 1)
    {
        DeallocatePeerBuffers();
        AllocatePeerBuffers();
    }

    if (_maxCUDWorkspaceSize > _CUDWorkspaceSize)
    {
        if (getGpu()._id == 0)
            std::cout << "Network::RefreshState: Setting cuDNN workspace size to " << _maxCUDWorkspaceSize << " bytes.\n";
        _CUDWorkspaceSize = _maxCUDWorkspaceSize;
        _pbCUDWorkspace = std::make_unique<GpuBuffer<uint8_t>>(_CUDWorkspaceSize);
    }

    if (_bDirty || (getGpu()._pNetwork != this))
    {
        getGpu().SetNeuralNetwork(this);
    }

    _bDirty = false;
}

void Network::ClearDataSets()
{
    _examples = 0;
    _bExamplesFound = false;
    for (auto l : _vInputLayer)
        l->_pDataSet = nullptr;
    for (auto l : _vOutputLayer)
        l->_pDataSet = nullptr;
}

#include <iostream>
#include <format>

void Network::LoadDataSets(std::vector<DataSetBase*>& vData)
{
    _bAllDataLoaded = false;

    for (auto l : _vInputLayer)
    {
        for (auto d : vData)
        {
            if (l->_dataSet == d->_name)
            {
                if (l->_dimensions != d->_dimensions)
                {
                    if (getGpu()._id == 0)
                    {
                        std::cout << std::format("Network::LoadDataSets: Dimensionality mismatch {}D input layer {} versus {}D data set {}\n",
                                                 l->_dimensions, l->_name, d->_dimensions, d->_name);
                    }
                }

                if ((l->_Nx < d->_width) || (l->_Ny < d->_height) || (l->_Nz < d->_length))
                {
                    if (getGpu()._id == 0)
                    {
                        std::cout << std::format("Network::LoadDataSets: Data element mismatch ({}, {}, {}) input layer {} versus ({}, {}, {}) data set {}\n",
                                                 l->_Nx, l->_Ny, l->_Nz, l->_name, d->_width, d->_height, d->_length, d->_name);
                    }
                    break;
                }

                if (!_bExamplesFound)
                {
                    _examples = d->_examples;
                    _bExamplesFound = true;
                }

                if (d->_examples != _examples)
                {
                    if (getGpu()._id == 0)
                    {
                        std::cout << std::format("Network::LoadDataSets: Mismatched examples count ({} vs {}) in dataset {}\n",
                                                 _examples, d->_examples, d->_name);
                    }
                    break;
                }

                l->_pDataSet = d;
                l->_bSparse = d->_attributes & DataSetEnums::Attributes::Sparse;
                l->_bDirty = true;
                if (getGpu()._id == 0)
                {
                    std::cout << std::format("Network::LoadDataSets: Found data set {} for input layer {}\n",
                                             d->_name, l->_name);
                }
                break;
            }
        }
    }

    for (auto l : _vOutputLayer)
    {
        for (auto d : vData)
        {
            if (l->_dataSet == d->_name)
            {
                if (l->_dimensions != d->_dimensions)
                {
                    if (getGpu()._id == 0)
                    {
                        std::cout << std::format("Network::LoadDataSets: Dimensionality mismatch {}D output layer {} versus {}D data set {}\n",
                                                 l->_dimensions, l->_name, d->_dimensions, d->_name);
                    }
                }

                if ((l->_Nx < d->_width) || (l->_Ny < d->_height) || (l->_Nz < d->_length))
                {
                    if (getGpu()._id == 0)
                    {
                        std::cout << std::format("Network::LoadDataSets: Data element mismatch ({}, {}, {}) output layer {} versus ({}, {}, {}) data set {}\n",
                                                 l->_Nx, l->_Ny, l->_Nz, l->_name, d->_width, d->_height, d->_length, d->_name);
                    }
                    break;
                }

                if (!_bExamplesFound)
                {
                    _examples = d->_examples;
                    _bExamplesFound = true;
                }

                if (d->_examples != _examples)
                {
                    if (getGpu()._id == 0)
                    {
                        std::cout << std::format("Network::LoadDataSets: Mismatched examples count ({} vs {}) in dataset {}\n",
                                                 _examples, d->_examples, d->_name);
                    }
                    break;
                }

                l->_pDataSet = d;
                l->_bDirty = true;
                if (getGpu()._id == 0)
                {
                    std::cout << std::format("Network::LoadDataSets: Found data set {} for output layer {}\n",
                                             d->_name, l->_name);
                }
                break;
            }
        }
    }

    _bDirty = true;
}

#include <iostream>

void Network::LoadBatch()
{
    if (_bDirty)
        RefreshState();

    uint32_t batch = _batch;
    if (_position + batch > _examples)
        batch = _examples - _position;

    for (auto l : _vInputLayer)
    {
        switch (_mode)
        {
            case Prediction:
                l->LoadPredictionBatch(_position, batch);
                break;

            case Training:
                l->LoadTrainingBatch(_position, batch);
                break;

            case Validation:
                l->LoadValidationBatch(_position, batch);
                break;

            default:
                std::cout << "unsupported mode in LoadBatch" << std::endl;
                std::exit(1);
        }
    }
}
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

void Network::SaveWeights(const std::string& fname, const std::string& inputLayer, const std::string& outputLayer)
{
    bool bResult = true;

    if (getGpu()._id == 0)
    {
        Layer* pInputLayer = _mLayer[inputLayer];
        Layer* pOutputLayer = _mLayer[outputLayer];

        if (pInputLayer == nullptr)
        {
            std::cout << "Network::SaveWeights: Unable to find input layer " << inputLayer << ".\n";
            bResult = false;
            goto exit;
        }

        if (pOutputLayer == nullptr)
        {
            std::cout << "Network::SaveWeights: Unable to find output layer " << outputLayer << ".\n";
            bResult = false;
            goto exit;
        }

        for (auto w : _vWeight)
        {
            if ((w->_inputLayer._name == pInputLayer->_name) && (w->_outputLayer._name == pOutputLayer->_name))
            {
                std::ofstream outputFile(fname);
                if (!outputFile)
                {
                    std::cout << "Network::SaveWeights: Failed to open output file " << fname << ".\n";
                    bResult = false;
                    goto exit;
                }

                w->_pbWeight->Download(w->_vWeight.data());
                w->_pbBias->Download(w->_vBias.data());

                outputFile << w->_width << "," << w->_height << '\n';

                for (int j = 0; j < w->_height; j++)
                {
                    for (int k = 0; k < w->_width; k++)
                    {
                        outputFile << std::fixed << std::setprecision(8) << w->_vWeight[j * w->_width + k];
                        if (k != w->_width - 1)
                            outputFile << ",";
                        else
                            outputFile << '\n';
                    }
                }

                for (int k = 0; k < w->_width; k++)
                {
                    outputFile << std::fixed << std::setprecision(8) << w->_vBias[k];
                    if (k != w->_width - 1)
                        outputFile << ",";
                    else
                        outputFile << '\n';
                }

                outputFile.close();
                bResult = true;
                goto exit;
            }
        }

        std::cout << "Network::SaveWeights: Unable to find weight matrix between input layer " << inputLayer
                  << " and output layer " << outputLayer << ".\n";
        bResult = false;
    }

exit:
    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (!bResult)
    {
        getGpu().Shutdown();
        std::exit(-1);
    }
}
#include <iostream>
#include <fstream>

void Network::SaveLayer(const std::string& fname, const std::string& layer)
{
    bool bResult = true;

    if (getGpu()._id == 0)
    {
        Layer* pLayer = _mLayer[layer];

        if (pLayer == nullptr)
        {
            std::cout << "Network::SaveLayer: Attempt to save nonexistent layer " << layer << ".\n";
            bResult = false;
            goto exit;
        }

        std::ofstream outputFile(fname);
        if (!outputFile)
        {
            std::cout << "Network::SaveLayer: Failed to open output file " << fname << ".\n";
            bResult = false;
            goto exit;
        }

        DumpLayer(outputFile, layer);
        outputFile.close();
    }

exit:
    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (!bResult)
    {
        getGpu().Shutdown();
        std::exit(-1);
    }
}
void Network::DumpLayer(std::ofstream& outputFile, const std::string& layer)
{
    bool bResult = true;

    if (getGpu()._id == 0)
    {
        Layer* pLayer = _mLayer[layer];

        if (pLayer == nullptr)
        {
            std::cout << "Network::DumpLayer: Attempt to dump nonexistent layer " << layer << ".\n";
            bResult = false;
            goto exit;
        }

        uint64_t batch = pLayer->_batch;
        if (batch + _position > _examples)
        {
            batch = _examples - _position;
        }
        uint32_t stride = pLayer->_localStride;
        uint64_t size = _batch * stride;
        std::vector<float> vData(size);
        pLayer->_pbUnit->Download(vData.data());

        for (uint32_t j = 0; j < batch; j++)
        {
            for (uint32_t k = 0; k < stride; k++)
            {
                outputFile << vData[j * stride + k];
                if (k < (stride - 1))
                    outputFile << ",";
                else
                    outputFile << "\n";
            }
        }
    }

exit:
    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (!bResult)
    {
        getGpu().Shutdown();
        std::exit(-1);
    }
}
void Network::SaveBatch(const std::string& fname)
{
    bool bResult = true;

    if (getGpu()._id == 0)
    {
        std::ofstream outputFile(fname);
        if (!outputFile)
        {
            std::cout << "Network::SaveBatch: Failed to open output file " << fname << ".\n";
            bResult = false;
            goto exit;
        }

        DumpBatch(outputFile);
        outputFile.close();
    }

exit:
    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (!bResult)
    {
        getGpu().Shutdown();
        std::exit(-1);
    }
}
void Network::DumpBatch(std::ofstream& outputFile)
{
    if (getGpu()._id == 0)
    {
        for (int i = 0; i < _vOutputLayer.size(); i++)
        {
            uint32_t stride = _vOutputLayer[i]->_localStride;
            uint32_t batch = _vOutputLayer[i]->_batch;

            if (batch + _position > _examples)
                batch = _examples - _position;

            uint64_t size = static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride);
            std::vector<Float> vData(size);
            _vOutputLayer[i]->_pbUnit->Download(vData.data());

            for (uint32_t j = 0; j < batch; j++)
            {
                for (uint32_t k = 0; k < stride; k++)
                {
                    outputFile << vData[j * stride + k];

                    if (k < (stride - 1))
                        outputFile << ",";
                    else
                        outputFile << "\n";
                }
            }
        }
    }
}

#include <iostream>

void Network::PredictBatch(uint32_t layers)
{
    uint32_t maxLayers = _vLayer.size();

    if (layers > _vLayer.size())
    {
        if (getGpu()._id == 0)
            std::cout << "Network::PredictBatch: Attempt to predict more layers than present in neural network " << _name << "\n";
        return;
    }

    if (_mode != Prediction)
    {
        _mode = Prediction;
        _bDirty = true;
    }

    if (_bDirty)
    {
        RefreshState();

        if (!_bAllDataLoaded)
        {
            if (getGpu()._id == 0)
            {
                std::cout << "Network::PredictBatch: Attempt to predict with neural network " << _name << " without providing data sets\n";
                std::cout << "for all input and output layers.\n";
            }
            getGpu().Shutdown();
            std::exit(-1);
        }
    }

    uint32_t batch = _batch;
    if (_position + batch > _examples)
        batch = _examples - _position;

    ClearUpdates();
    LoadBatch();

    for (auto l : _vFPOrder)
    {
        l->ForwardPropagate(_position, batch, false);
    }
}
#include <iostream>

void Network::PredictTrainingBatch(uint32_t layers)
{
    uint32_t maxLayers = _vLayer.size();

    if (layers > _vLayer.size())
    {
        if (getGpu()._id == 0)
            std::cout << "Network::PredictTrainingBatch: Attempt to predict more layers than present in neural network " << _name << "\n";
        return;
    }

    if (_bDirty)
    {
        RefreshState();

        if (!_bAllDataLoaded)
        {
            if (getGpu()._id == 0)
            {
                std::cout << "Network::PredictTrainingBatch: Attempt to predict with neural network " << _name << " without providing data sets\n";
                std::cout << "for all input and output layers.\n";
            }
            getGpu().Shutdown();
            std::exit(-1);
        }
    }

    uint32_t batch = _batch;
    if (_position + batch > _examples)
        batch = _examples - _position;

    LoadBatch();

    for (auto l : _vFPOrder)
    {
        l->ForwardPropagate(_position, batch, true);
    }
}
#include <iostream>

void Network::PredictValidationBatch(uint32_t layers)
{
    uint32_t maxLayers = _vLayer.size();

    if (layers > _vLayer.size())
    {
        if (getGpu()._id == 0)
            std::cout << "Network::PredictValidationBatch: Attempt to predict more layers than present in neural network " << _name << "\n";
        return;
    }

    if (_mode != Validation)
    {
        _mode = Validation;
        _bDirty = true;
    }

    if (_bDirty)
    {
        RefreshState();

        if (!_bAllDataLoaded)
        {
            if (getGpu()._id == 0)
            {
                std::cout << "Network::PredictValidationBatch: Attempt to predict with neural network " << _name << " without providing data sets\n";
                std::cout << "for all input and output layers.\n";
            }
            getGpu().Shutdown();
            std::exit(-1);
        }
    }

    uint32_t batch = _batch;
    if (_position + batch > _examples)
        batch = _examples - _position;

    LoadBatch();

    ClearUpdates();
    for (auto l : _vFPOrder)
    {
        l->ForwardPropagate(_position, batch, false);
    }
}
#include <cmath>
#include <chrono>
#include <iostream>
#include <limits>

Float Network::Train(uint32_t epochs, Float alpha, Float lambda, Float lambda1, Float mu, Float mu1)
{
    if (_mode != Training)
    {
        _mode = Training;
        _bDirty = true;
    }

    if (_bDirty)
    {
        RefreshState();

        if (!_bAllDataLoaded)
        {
            if (getGpu()._id == 0)
            {
                std::cout << "Network::Train: Attempt to train neural network " << _name << " without providing data sets\n"
                          << "for all input and output layers." << std::endl;
            }
            getGpu().Shutdown();
            std::exit(-1);
        }
    }

    if (_trainingMode != SGD && _bClearVelocity)
    {
        for (auto& weight : _vWeight)
        {
            weight->ClearVelocity();
        }
        _batches = 0;
    }

    Float total_error_training = 0.0;
    Float total_error_regularization = 0.0;
    Float average_error_training = std::numeric_limits<Float>::max();
    Float average_error_regularization = 0.0;
    Float moving_average = 0.0;
    uint32_t brake_steps = 0;
    uint32_t init_steps = 100;

    for (uint32_t epoch = 0; epoch < epochs; ++epoch)
    {
        const auto start = std::chrono::steady_clock::now();
        total_error_training = 0.0;
        total_error_regularization = 0.0;

        if (_bDenoising)
        {
            for (auto& layer : _vInputLayer)
            {
                if (layer->_bDenoising)
                {
                    layer->GenerateDenoisingData();
                }
            }
        }

        if (_bShuffleIndices)
        {
            ShuffleIndices();
        }

        for (uint32_t pos = 0; pos < GetExamples(); pos += GetBatch())
        {
            SetPosition(pos);
            ClearUpdates();
            PredictTrainingBatch();
            Float error_training, error_regularization, error;
            std::tie(error_training, error_regularization) = CalculateError(lambda, lambda1);
            uint32_t minibatch = GetBatch();
            if (_examples - pos < minibatch)
            {
                minibatch = _examples - pos;
            }
            total_error_training += error_training;
            total_error_regularization += error_regularization * minibatch;
            if (_verbose && getGpu()._id == 0)
            {
                std::std::cout("Network::Train: Minibatch@%u, average error %f, (%f training, %f regularization), alpha %f\n",
                            pos, error_training / minibatch + error_regularization, error_training / minibatch, error_regularization, alpha);
            }

            Float step_alpha = (_decay <= 0.0) ? alpha : alpha * (1.0 / (1.0 + _decay * static_cast<Float>(_batches)));
            moving_average = 0.9 * moving_average + 0.1 * error_training;
            if (init_steps == 0)
            {
                if (error_training > 2.0 * moving_average)
                {
                    brake_steps = 25;
                    if (getGpu()._id == 0)
                    {
                        std::std::cout("Network::Train: Detected network divergence, attempting recovery.\n");
                    }
                }
            }
            else
            {
                --init_steps;
            }

            if (brake_steps > 0)
            {
                step_alpha *= 0.1;
                --brake_steps;
            }

            if (brake_steps < 24)
            {
                BackPropagate();

                ++_batches;

                UpdateWeights(step_alpha, lambda, lambda1, mu, mu1);
            }
        }
        const auto end = std::chrono::steady_clock::now();
        average_error_training = total_error_training / GetExamples();
        average_error_regularization = total_error_regularization / GetExamples();
        if (getGpu()._id == 0)
        {
            std::std::cout("Network::Train: Epoch %d, average error %f, average training error %f, average regularization error %f, elapsed time %fs\n",
                        ++_epochs, average_error_training + average_error_regularization,
                        average_error_training, average_error_regularization,
                        elapsed_seconds(start, end));
        }

        if (_checkpoint_interval > 0)
        {
            _checkpoint_epochs++;
            if (_checkpoint_epochs >= _checkpoint_interval)
            {
                std::string filename = _checkpoint_name + std::to_string(_epochs) + ".nc";
                if (getGpu()._id == 0)
                {
                    std::std::cout("Network::Train: saving checkpoint %s\n", filename.c_str());
                }

                SaveNetCDF(filename);
                _checkpoint_epochs = 0;
            }
        }
    }

    return average_error_training + average_error_regularization;
}
void Network::ClearUpdates()
{
    for (auto& weight : _vWeight)
    {
        weight->_updateCount = 0;
    }

    for (auto& layer : _vLayer)
    {
        layer->ClearUpdates();
    }
}

std::tuple<Float, Float> Network::CalculateError(Float lambda, Float lambda1)
{
    Float error_training = 0.0;
    Float error_regularization = 0.0;

    uint32_t batch = _batch;
    if (_position + batch > _examples)
    {
        batch = _examples - _position;
    }

    for (auto& outputLayer : _vOutputLayer)
    {
        error_training += outputLayer->CalculateError(_position, batch, _errorFunction);
    }

    if ((lambda != 0.0) || (lambda1 != 0.0))
    {
        for (auto& weight : _vWeight)
        {
            error_regularization += weight->CalculateRegularizationError(lambda, lambda1);
        }
    }

    if (getGpu()._numprocs > 1)
    {
        double derror_training = error_training;
        double derror_regularization = error_regularization;
        MPI_Allreduce(MPI_IN_PLACE, &derror_training, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &derror_regularization, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        error_training = derror_training;
        error_regularization = derror_regularization;
    }

    return std::make_tuple(error_training, error_regularization);
}
void Network::BackPropagate()
{
    uint32_t batch = _batch;
    if (_position + batch > _examples)
    {
        batch = _examples - _position;
    }

    for (auto& layer : _vBPOrder)
    {
        switch (layer->_kind)
        {
            case Layer::Kind::Output:
                layer->CalculateOutputDelta(_position, batch, _errorFunction);
                [[fallthrough]];

            case Layer::Kind::Hidden:
                layer->BackPropagate(_position, batch);
                break;
        }
    }
}

void Network::UpdateWeights(Float alpha, Float lambda, Float lambda1, Float mu, Float mu1)
{
    uint32_t batch = _batch;
    if (_position + batch > _examples)
    {
        batch = _examples - _position;
    }

    for (int64_t i = static_cast<int64_t>(_vWeight.size()) - 1; i >= 0; --i)
    {
        _vWeight[i]->UpdateWeights(_trainingMode, batch, alpha, lambda, lambda1, mu, mu1, _batches);
    }

    for (auto& layer : _vLayer)
    {
        if (layer->_bBatchNormalization)
        {
            layer->UpdateWeights(_trainingMode, batch, alpha, lambda, lambda1, mu, mu1, _batches);
        }
    }
}
void Network::CalculateOutput(const std::string& layer, uint32_t k, GpuBuffer<Float>* pbKey, GpuBuffer<unsigned int>* pbValue)
{
    if (Layer* pLayer = _mLayer[layer]; pLayer != nullptr)
    {
        if (k > 128)
        {
            if (getGpu()._id == 0)
                std::std::cout("Network::CalculateOutput: Can only calculate 128 or fewer elements.\n");
            return;
        }
        else if (k > pLayer->_Nx * pLayer->_Ny * pLayer->_Nz)
        {
            if (getGpu()._id == 0)
                std::std::cout("Network::CalculateOutput: Layer has fewer elements than k (%u vs %u).\n", k, pLayer->_Nx * pLayer->_Ny * pLayer->_Nz);
            return;
        }

        uint32_t batch = _batch;
        if (_position + batch > _examples)
            batch = _examples - _position;

        kCalculateOutput(pLayer->_pbUnit->_pDevData, pbKey->_pDevData, pbValue->_pDevData, batch, pLayer->_localStride, k);
    }
    else
    {
        if (getGpu()._id == 0)
            std::std::cout("Network::CalculateOutput: Unknown layer %s.\n", layer.c_str());
    }
}

bool Network::SaveNetCDF(const std::string& fname)
{
    bool bResult = true;

    std::vector<std::vector<Float>> vvWeight;
    std::vector<std::vector<Float>> vvBias;
    for (auto& w : _vWeight)
    {
        std::vector<Float> vWeight;
        std::vector<Float> vBias;

        if (!w->_bShared)
        {
            w->_pbWeight->Download(w->_vWeight.data());

            if (getGpu()._numprocs == 1)
            {
                vWeight = w->_vWeight;
            }
            else
            {
                uint32_t outgoingSize = w->_outputLayer._stride * 3;
                uint32_t incomingSize = w->_inputLayer._stride * 2;
                if (getGpu()._id == 0)
                {
                    vWeight.resize(w->_outputLayer._stride * w->_inputLayer._stride);
                    Float* pWeight = vWeight.data();
                    if (outgoingSize > incomingSize)
                    {
                        cudaMemcpy2D(pWeight, w->_outputLayer._stride * sizeof(Float), w->_vWeight.data(), w->_outputLayer._localStride * sizeof(Float), w->_outputLayer._localStride * sizeof(Float), w->_inputLayer._stride, cudaMemcpyDefault);
                        pWeight += w->_outputLayer._localStride;
                        for (uint32_t i = 1; i < getGpu()._numprocs; i++)
                        {
                            uint64_t size;
                            MPI_Status status;
                            MPI_Recv(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
                            std::vector<Float> vTemp(size);
                            MPI_Recv(vTemp.data(), size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
                            uint64_t lstride = size / w->_inputLayer._stride;
                            Float* pSrcWeight = vTemp.data();
                            Float* pDstWeight = pWeight;
                            for (uint32_t j = 0; j < w->_inputLayer._stride; j++)
                            {
                                memcpy(pDstWeight, pSrcWeight, lstride * sizeof(Float));
                                pSrcWeight += lstride;
                                pDstWeight += w->_outputLayer._stride;
                            }
                            pWeight += lstride;
                        }
                    }
                    else
                    {
                        cudaMemcpy(pWeight, w->_vWeight.data(), w->_outputLayer._stride * w->_inputLayer._localStride * sizeof(Float), cudaMemcpyDefault);
                        pWeight += w->_outputLayer._stride * w->_inputLayer._localStride;
                        for (uint32_t i = 1; i < getGpu()._numprocs; i++)
                        {
                            uint64_t size;
                            MPI_Status status;
                            MPI_Recv(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
                            MPI_Recv(pWeight, size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
                            pWeight += size;
                        }
                    }
                }
                else
                {
                    uint64_t size = w->_vWeight.size();
                    MPI_Send(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD);
                    MPI_Send(w->_vWeight.data(), size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
                }
            }
        }

        w->_pbBias->Download(w->_vBias.data());
        if (getGpu()._id == 0)
        {
            vBias = w->_vBias;
            vBias.resize(w->_outputLayer._stride);
            uint64_t offset = w->_vBias.size();
            for (size_t i = 1; i < getGpu()._numprocs; i++)
            {
                uint64_t size;
                MPI_Status status;
                MPI_Recv(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(vBias.data() + offset, size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
                offset += size;
            }
        }
        else
        {
            uint64_t size = w->_vBias.size();
            MPI_Send(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD);
            MPI_Send(w->_vBias.data(), size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }

        vvWeight.push_back(vWeight);
        vvBias.push_back(vBias);
    }

    if (getGpu()._id == 0)
    {
        try
        {
            NcFile nc(fname, NcFile::replace);

            nc.putAtt("version", ncFloat, _VERSION);
            nc.putAtt("name", _name);
            nc.putAtt("kind", ncUint, _kind);
            nc.putAtt("errorFunction", ncUint, _errorFunction);
            nc.putAtt("maxout_k", ncInt, _maxout_k);
            nc.putAtt("decay", ncFloat, _decay);
            nc.putAtt("LRN_k", ncFloat, _LRN_k);
            nc.putAtt("LRN_n", ncInt, _LRN_n);
            nc.putAtt("LRN_alpha", ncFloat, _LRN_alpha);
            nc.putAtt("LRN_beta", ncFloat, _LRN_beta);
            nc.putAtt("bSparsenessPenalty", ncUint, static_cast<uint32_t>(_bSparsenessPenalty));
            nc.putAtt("sparsenessPenalty_p", ncFloat, _sparsenessPenalty_p);
            nc.putAtt("sparsenessPenalty_beta", ncFloat, _sparsenessPenalty_beta);
            nc.putAtt("bDenoising", ncUint, static_cast<uint32_t>(_bDenoising));
            nc.putAtt("denoising_p", ncFloat, _denoising_p);
            nc.putAtt("deltaBoost_one", ncFloat, _deltaBoost_one);
            nc.putAtt("deltaBoost_zero", ncFloat, _deltaBoost_zero);
            nc.putAtt("SMCE_oneScale", ncFloat, _SMCE_oneScale);
            nc.putAtt("SMCE_zeroScale", ncFloat, _SMCE_zeroScale);
            nc.putAtt("SMCE_oneTarget", ncFloat, _SMCE_oneTarget);
            nc.putAtt("SMCE_zeroTarget", ncFloat, _SMCE_zeroTarget);
            nc.putAtt("ShuffleIndices", ncUint, static_cast<uint32_t>(_bShuffleIndices));
            nc.putAtt("checkpoint_name", _checkpoint_name);
            nc.putAtt("checkpoint_interval", ncInt, _checkpoint_interval);
            nc.putAtt("checkpoint_epochs", ncInt, _checkpoint_epochs);

            nc.putAtt("layers", ncUint, static_cast<uint32_t>(_vLayer.size()));
            for (uint32_t i = 0; i < _vLayer.size(); ++i)
                _vLayer[i]->WriteNetCDF(nc, i);

            nc.putAtt("weights", ncUint, static_cast<uint32_t>(_vWeight.size()));
            for (uint32_t i = 0; i < _vWeight.size(); ++i)
                _vWeight[i]->WriteNetCDF(nc, i, vvWeight[i].data(), vvBias[i].data());
        }
        catch (NcException& e)
        {
            std::std::cout("Network::SaveNetCDF Error opening binary output file %s to save neural network %s.\n", fname.c_str(), _name.c_str());
            bResult = false;
        }
    }

    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!bResult)
    {
        getGpu().Shutdown();
        std::exit(-1);
    }

    return bResult;
}
std::vector<std::string> Network::GetLayers() const
{
    std::vector<std::string> vResult;
    for (const auto& l : _vLayer)
    {
        vResult.push_back(l->_name);
    }

    return vResult;
}

const std::string& Network::GetName() const
{
    return _name;
}

Float* Network::GetUnitBuffer(const std::string& layer)
{
    const auto itr = _mLayer.find(layer);
    if (itr == _mLayer.end())
    {
        if (getGpu()._id == 0)
        {
            std::std::cout("Network::GetUnitBuffer: Unknown layer %s.\n", layer.c_str());
        }

        return nullptr;
    }

    return itr->second->GetUnitBuffer();
}

Float* Network::GetDeltaBuffer(const std::string& layer)
{
    const auto itr = _mLayer.find(layer);
    if (itr == _mLayer.end())
    {
        if (getGpu()._id == 0)
        {
            std::std::cout("Network::GetDeltaBuffer: Unknown layer %s.\n", layer.c_str());
        }

        return nullptr;
    }

    return itr->second->GetDeltaBuffer();
}

uint64_t Network::GetBufferSize(const std::string& layer) const
{
    const auto itr = _mLayer.find(layer);
    if (itr == _mLayer.end())
    {
        if (getGpu()._id == 0)
        {
            std::std::cout("Network::GetDeltaBuffer: Unknown layer %s.\n", layer.c_str());
        }

        return 0;
    }

    return itr->second->GetBufferSize();
}

Weight* Network::GetWeight(const std::string& inputLayer, const std::string& outputLayer) const
{
    const auto inputLayerItr = _mLayer.find(inputLayer);
    if (inputLayerItr == _mLayer.end())
    {
        if (getGpu()._id == 0)
        {
            std::std::cout("Network::GetWeight: Unknown input layer %s.\n", inputLayer.c_str());
        }

        return nullptr;
    }

    const auto outputLayerItr = _mLayer.find(outputLayer);
    if (outputLayerItr == _mLayer.end())
    {
        if (getGpu()._id == 0)
        {
            std::std::cout("Network::GetWeight: Unknown output layer %s.\n", outputLayer.c_str());
        }

        return nullptr;
    }

    const Layer* pInputLayer = inputLayerItr->second;
    const Layer* pOutputLayer = outputLayerItr->second;

    for (const auto& p : _vWeight)
    {
        if ((p->_inputLayer == pInputLayer) && (p->_outputLayer == pOutputLayer))
        {
            return p;
        }
    }

    if (getGpu()._id == 0)
    {
        std::std::cout("Network::GetWeight: No set of weights connecting layer %s to layer %s.\n", inputLayer.c_str(), outputLayer.c_str());
    }

    return nullptr;
}
Float* Network::GetWeightBuffer(const std::string& inputLayer, const std::string& outputLayer)
{
    const auto inputLayerItr = _mLayer.find(inputLayer);
    if (inputLayerItr == _mLayer.end())
    {
        if (getGpu()._id == 0)
        {
            std::std::cout("Network::GetWeight: Unknown input layer %s.\n", inputLayer.c_str());
        }

        return nullptr;
    }

    const auto outputLayerItr = _mLayer.find(outputLayer);
    if (outputLayerItr == _mLayer.end())
    {
        if (getGpu()._id == 0)
        {
            std::std::cout("Network::GetWeightBuffer: Unknown output layer %s.\n", outputLayer.c_str());
        }
        return nullptr;
    }

    const Layer* pInputLayer = inputLayerItr->second;
    const Layer* pOutputLayer = outputLayerItr->second;

    for (const auto& p : _vWeight)
    {
        if ((p->_inputLayer == pInputLayer) && (p->_outputLayer == pOutputLayer))
        {
            return p->_vWeight.data();
        }
    }

    if (getGpu()._id == 0)
    {
        std::std::cout("Network::GetWeightBuffer: No set of weights connecting layer %s to layer %s.\n", inputLayer.c_str(), outputLayer.c_str());
    }

    return nullptr;
}

Network::~Network()
{
    DeallocatePeerBuffers();

    for (auto* weight : _vWeight)
        delete weight;

    for (auto* layer : _vLayer)
        delete layer;
}

uint32_t CalculateConvolutionDimensions(uint32_t width, uint32_t filter, uint32_t stride)
{
    if (width <= filter)
        return 1;
    else if (stride == 1)
        return width;
    else
        return (width - filter) / stride + 1;
}

void CalculateDerivedLayerDimensions(NetworkDescriptor& d)
{
    std::map<LayerDescriptor*, bool> mbDimensionsCalculated;
    std::map<std::string, LayerDescriptor*> mLayer;

    for (auto& layerDescriptor : d._vLayerDescriptor)
    {
        LayerDescriptor* pL = &layerDescriptor;
        bool bFlag = true;
        if ((pL->_kind == Layer::Kind::Hidden) &&
            ((pL->_type == Layer::Type::Pooling) || (pL->_type == Layer::Type::Convolutional)))
            bFlag = false;
        mbDimensionsCalculated[pL] = bFlag;
        mLayer[pL->_name] = pL;
    }

    bool bFinished;
    do {
        bFinished = true;

        for (auto& layerDescriptor : d._vLayerDescriptor)
        {
            LayerDescriptor* pL = &layerDescriptor;
            bool bPooling = pL->_type == Layer::Type::Pooling;
            bool bLRN = bPooling && (pL->_poolingFunction == PoolingFunction::LRN);
            bool bDotProduct = bPooling && ((pL->_poolingFunction == PoolingFunction::DotProduct) || (pL->_poolingFunction == PoolingFunction::Cosine));

            if (!mbDimensionsCalculated[pL])
            {
                bool bAllInputsCalculated = true;
                for (const auto& s : pL->_vSource)
                {
                    LayerDescriptor* pS = mLayer[s];
                    bAllInputsCalculated &= mbDimensionsCalculated[pS];
                }

                if (!bAllInputsCalculated)
                {
                    bFinished = false;
                    continue;
                }

                bool bSized = false;
                LayerDescriptor* pL0 = mLayer[pL->_vSource[0]];
                uint32_t N = pL->_Nx;
                uint32_t oldNx = bDotProduct ? pL0->_Nx : 1;
                uint32_t oldNy = bDotProduct ? pL0->_Ny : 1;
                uint32_t oldNz = bDotProduct ? pL0->_Nz : 1;
                uint32_t nx = bDotProduct ? pL->_vSource.size() - 1 : 1;
                uint32_t ny = 1;
                uint32_t nz = 1;
                uint32_t nw = 1;
                for (const auto& s : pL->_vSource)
                {
                    LayerDescriptor* pS = mLayer[s];

                    if (bDotProduct)
                    {
                        if ((oldNx != pS->_Nx) || (oldNy != pS->_Ny) || (oldNz != pS->_Nz))
                        {
                            if (getGpu()._id == 0)
                                std::std::cout("Network::CalculateDerivedLayerDimensions: Inconsistent incoming data size for dot product layer %s\n", pL->_name.c_str());
                            getGpu().Shutdown();
                            exit(-1);
                        }
                    }
                    else
                    {
                        if (!bLRN)
                        {
                            nx = CalculateConvolutionDimensions(pS->_Nx, pL->_kernelX, pL->_kernelStrideX);
                            ny = CalculateConvolutionDimensions(pS->_Ny, pL->_kernelY, pL->_kernelStrideY);
                            nz = CalculateConvolutionDimensions(pS->_Nz, pL->_kernelZ, pL->_kernelStrideZ);
                            nw = pS->_Nw;
                            if (bPooling)
                                pL->_dimensions = pS->_dimensions;
                        }
                        else
                        {
                            nx = pS->_Nx;
                            ny = pS->_Ny;
                            nz = pS->_Nz;
                            nw = pS->_Nw;
                            pL->_dimensions = pS->_dimensions;
                        }

                        switch (pL->_kernelDimensions)
                        {
                            case 3:
                                if (pS->_Nz < pL->_kernelZ)
                                {
                                    pL->_kernelPaddingZ = (pL->_kernelZ - pS->_Nz + 1) / 2;
                                }
                                else if (pL->_kernelStrideZ == 1)
                                {
                                    pL->_kernelPaddingZ = pL->_kernelZ / 2;
                                }

                            case 2:
                                if (pS->_Ny < pL->_kernelY)
                                {
                                    pL->_kernelPaddingY = (pL->_kernelY - pS->_Ny + 1) / 2;
                                }
                                else if (pL->_kernelStrideY == 1)
                                {
                                    pL->_kernelPaddingY = pL->_kernelY / 2;
                                }

                            case 1:
                                if (pS->_Nx < pL->_kernelX)
                                {
                                    pL->_kernelPaddingX = (pL->_kernelX - pS->_Nx + 1) / 2;
                                }
                                else if (pL->_kernelStrideX == 1)
                                {
                                    pL->_kernelPaddingX = pL->_kernelX / 2;
                                }
                        }

                        if (bSized)
                        {
                            if ((nx != oldNx) || (ny != oldNy) || (nz != oldNz))
                            {
                                if (getGpu()._id == 0)
                                    std::std::cout("Network::CalculateDerivedLayerDimensions: Inconsistent incoming data size for convolution layer %s\n", pL->_name.c_str());
                                getGpu().Shutdown();
                                exit(-1);
                            }
                        }
                        bSized = true;
                        oldNx = nx;
                        oldNy = ny;
                        oldNz = nz;
                        mbDimensionsCalculated[pL] = true;
                    }
                }
                pL->_Nx = nx;
                pL->_Ny = ny;
                pL->_Nz = nz;
                pL->_Nw = nw;
                if (!bPooling)
                {
                    switch (pL->_kernelDimensions)
                    {
                        case 1:
                            pL->_Ny = N;
                            pL->_dimensions = 2;
                            break;

                        case 2:
                            pL->_Nz = N;
                            pL->_dimensions = 3;
                            break;

                        case 3:
                            pL->_Nw = N;
                            pL->_dimensions = 4;
                            break;
                    }
                }
            }
        }
    } while (!bFinished);
}
void Network::CalculatePropagationOrder()
{
    for (auto p : _vLayer)
    {
        p->_priority = (p->_kind == Layer::Kind::Input) ? 0 : -1;
    }

    auto compareLayer = [](Layer* l1, Layer* l2) {
        return l1->_priority < l2->_priority;
    };

    std::priority_queue<Layer*, std::vector<Layer*>, decltype(compareLayer)> pqueue(compareLayer);

    for (auto p : _vInputLayer)
    {
        pqueue.push(p);
    }

    while (!pqueue.empty())
    {
        Layer* pLayer = pqueue.top();
        pqueue.pop();

        int32_t priority = pLayer->_priority + 1;
        for (auto p : pLayer->_vOutgoingLayer)
        {
            if (p->_priority < priority)
            {
                p->_priority = priority;
                pqueue.push(p);
            }
        }

        for (auto p : pLayer->_vOutgoingSkip)
        {
            if (p->_priority < priority)
            {
                p->_priority = priority;
                pqueue.push(p);
            }
        }
    }

    _vFPOrder.resize(_vLayer.size());
    std::ranges::copy(_vLayer, _vFPOrder.begin());
    std::ranges::sort(_vFPOrder, compareLayer);

    for (auto p : _vLayer)
    {
        p->_priority = (p->_kind == Layer::Kind::Output) ? 0 : -1;
    }

    for (auto p : _vOutputLayer)
    {
        pqueue.push(p);
    }

    while (!pqueue.empty())
    {
        Layer* pLayer = pqueue.top();
        pqueue.pop();

        int32_t priority = pLayer->_priority + 1;
        for (auto p : pLayer->_vIncomingLayer)
        {
            if (p->_priority < priority)
            {
                p->_priority = priority;
                pqueue.push(p);
            }
        }

        for (auto p : pLayer->_vIncomingSkip)
        {
            if (p->_priority < priority)
            {
                p->_priority = priority;
                pqueue.push(p);
            }
        }
    }

    _vBPOrder.resize(_vLayer.size());
    std::ranges::copy(_vLayer, _vBPOrder.begin());
    std::ranges::sort(_vBPOrder, compareLayer);
}
bool Network::Validate()
{
    const Float delta = 0.001f;
    const Float alpha = 1.0f;
    const Float lambda = 0.0f;
    const Float lambda1 = 0.0f;
    const Float mu = 0.0f;
    const Float mu1 = 0.0f;

    const Float epsilon = delta * 20.0f;

    if (getGpu()._numprocs > 1)
    {
        std::cout << "Network::Validate: Do not call this method from a multi-process run, just don't, mmkay?" << std::endl;
        return false;
    }

    if (_mode != Validation)
    {
        _mode = Validation;
        _bDirty = true;
    }

    if (_bDirty)
    {
        RefreshState();

        if (!_bAllDataLoaded)
        {
            std::cout << "Network::Validate: Attempt to train neural network " << _name << " without providing data sets" << std::endl;
            std::cout << "for all input and output layers." << std::endl;
            getGpu().Shutdown();
            exit(-1);
        }
    }

    if (_trainingMode != SGD && _bClearVelocity)
    {
        for (auto& weight : _vWeight)
        {
            weight->ClearVelocity();
        }
    }

    if (_bShuffleIndices)
    {
        ShuffleIndices();
    }

    std::cout << "Validating network weights and biases with epsilon error threshold of " << epsilon << std::endl;

    SetPosition(0);
    ClearUpdates();
    PredictValidationBatch();
    Float initialErrorTraining, initialErrorRegularization, initialError;
    std::tie(initialErrorTraining, initialErrorRegularization) = CalculateError(lambda, lambda1);
    initialError = initialErrorTraining + initialErrorRegularization;
    std::cout << "initialErrorTraining " << initialErrorTraining << "; initialErrorRegularization " << initialErrorRegularization << std::endl;

    BackPropagate();

    std::vector<std::vector<Float>> vWeightGradient;
    vWeightGradient.reserve(_vWeight.size());
    for (auto weight : _vWeight)
    {
        weight->_pbWeight->Download(weight->_vWeight.data());
        weight->_pbBias->Download(weight->_vBias.data());
        weight->_pbWeightGradient->Download(std::back_inserter(vWeightGradient.emplace_back(), weight->_vWeight.size()));
    }

    std::vector<std::vector<Float>> vBiasGradient;
    UpdateWeights(alpha, lambda, lambda1, mu, mu1);

    for (size_t id = 0; id < _vWeight.size(); ++id)
    {
        auto& weight = _vWeight[id];
        vBiasGradient.emplace_back(weight->_vBias.size());
        auto& bias = vBiasGradient.back();

        std::cout << "Validating weights between layer " << weight->_inputLayer._name << " and " << weight->_outputLayer._name << std::endl;

        weight->_pbWeight->Upload(weight->_vWeight.data());
        std::vector<Float> bias_g(weight->_pbBias->_length);
        weight->_pbBias->Download(bias_g.data());
        for (size_t b = 0; b < bias_g.size(); ++b)
        {
            bias[b] = bias_g[b] - weight->_vBias[b];
        }
        weight->_pbBias->Upload(weight->_vBias.data());
    }

    for (size_t id = 0; id < _vWeight.size(); ++id)
    {
        auto& weight = _vWeight[id];

        std::cout << "Validating weights between layer " << weight->_inputLayer._name << " and " << weight->_outputLayer._name << std::endl;

        std::cout << "Tweak weights" << std::endl;
        for (size_t i = 0; i < weight->_vWeight.size(); ++i)
        {
            Float oldWeight = weight->_vWeight[i];
            weight->_vWeight[i] += delta / (_batch * weight->_sharingCount);
            weight->_pbWeight->Upload(weight->_vWeight.data());
            PredictValidationBatch();
            weight->_vWeight[i] = oldWeight;
            Float errorTraining, errorRegularization, error;
            std::tie(errorTraining, errorRegularization) = CalculateError(lambda, lambda1);
            error = errorTraining + errorRegularization;
            Float dEdW = (error - initialError) / delta;
            Float weightGradient = vWeightGradient[id][i];
            std::cout << "errorTraining " << errorTraining << "; errorRegularization " << errorRegularization <<
                "; dEdW " << dEdW << "; weightGradient " << weightGradient << std::endl;
            if (std::fabs(dEdW + weightGradient) > epsilon)
            {
                std::cout << error << " " << initialError << std::endl;
                std::cout << "Failed Weight " << i << " exceeds error threshold: " << dEdW << " vs " << weightGradient << std::endl;
                return false;
            }
        }
        weight->_pbWeight->Upload(weight->_vWeight.data());

        std::cout << "Tweak biases" << std::endl;
        for (size_t i = 0; i < weight->_vBias.size(); ++i)
        {
            Float oldBias = weight->_vBias[i];
            weight->_vBias[i] += delta / (_batch);
            weight->_pbBias->Upload(weight->_vBias.data());
            PredictValidationBatch();
            weight->_vBias[i] = oldBias;
            Float errorTraining, errorRegularization, error;
            std::tie(errorTraining, errorRegularization) = CalculateError(lambda, lambda1);
            error = errorTraining + errorRegularization;
            Float dEdb = (error - initialError) / delta;
            Float biasGradient = vBiasGradient[id][i];
            std::cout << "errorTraining " << errorTraining << "; errorRegularization " << errorRegularization <<
                "; dEdb " << dEdb << "; biasGradient " << biasGradient << std::endl;
            if (std::fabs(dEdb + biasGradient) > epsilon)
            {
                std::cout << error << " " << initialError << std::endl;
                std::cout << "Failed Bias " << i << " exceeds error threshold: " << dEdb << " vs " << biasGradient << std::endl;
                return false;
            }
        }
        weight->_pbBias->Upload(weight->_vBias.data());
    }

    return true;
}
void Network::DeallocatePeerBuffers()
{
    if (getGpu()._numprocs > 1)
    {
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        for (auto& buffer : _pPeerBuffer)
        {
            if (buffer != nullptr)
            {
                cudaError_t status = cudaIpcCloseMemHandle(buffer);
                RTERROR(status, "Network::DeallocatePeerBuffers: Error closing IpcMemHandle");
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        _pbP2PBuffer.fill(nullptr);
        _pCPUBuffer.reset();
    }
}

void Network::AllocatePeerBuffers()
{
    if (getGpu()._numprocs > 1)
    {
        _maxStride = 0;
        for (const auto& weight : _vWeight)
        {
            uint32_t stride = std::max(weight->_inputLayer._stride, weight->_outputLayer._stride);
            _maxStride = std::max(_maxStride, stride);
        }
        uint64_t maxMemory = _maxStride * _batch;
        maxMemory = std::max(maxMemory, _examples);

        _pbP2PBuffer.fill(nullptr);
        for (auto& buffer : _pbP2PBuffer)
        {
            buffer = std::make_unique<GpuBuffer<Float>>(maxMemory);
        }

        if (getGpu()._bP2P)
        {
            std::vector<cudaIpcMemHandle_t> memHandles(2 * getGpu()._numprocs);
            size_t pos = getGpu()._id * 2;
            cudaError_t status = cudaIpcGetMemHandle(&(memHandles[pos]), _pbP2PBuffer[0]->_pDevData);
            RTERROR(status, "Network::AllocatePeerBuffers: Error getting first P2P IPCMemHandle");
            status = cudaIpcGetMemHandle(&(memHandles[pos + 1]), _pbP2PBuffer[1]->_pDevData);
            RTERROR(status, "Network::AllocatePeerBuffers: Error getting second P2P IPCMemHandle");
            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_nullptr, memHandles.data(), 2 * sizeof(cudaIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);
            unsigned int peer = 2 * ((getGpu()._id + getGpu()._numprocs - 1) % getGpu()._numprocs);
            status = cudaIpcOpenMemHandle((void**)&(_pPeerBuffer[0]), memHandles[peer], cudaIpcMemLazyEnablePeerAccess);
            RTERROR(status, "Network::AllocatePeerBuffers: Unable to open first peer IPCMemHandle");
            status = cudaIpcOpenMemHandle((void**)&(_pPeerBuffer[1]), memHandles[peer + 1], cudaIpcMemLazyEnablePeerAccess);
            RTERROR(status, "Network::AllocatePeerBuffers: Unable to open second peer IPCMemHandle");
        }
        else
        {
            _pCPUBuffer = std::make_unique<Float[]>(maxMemory);
        }
    }
}
void Network::SwapPeerBuffers()
{
    std::swap(_sendIndex, _receiveIndex);
}

std::unordered_map<Network::Kind, std::string> Network::_sKindMap = {
    { Network::Kind::FeedForward, "FeedForward" },
    { Network::Kind::AutoEncoder, "AutoEncoder" }
};

std::ostream& operator<<(std::ostream& out, Network::Kind k)
{
    out << Network::_sKindMap[k];
    return out;
}

template <typename T>
void MPI_Bcast_value(T& value)
{
    MPI_Bcast(&value, 1, MPI_Type_map<T>(), 0, MPI_COMM_WORLD);
}

void MPI_Bcast_string(std::string& str)
{
    int32_t size = str.size();
    MPI_Bcast_value(size);
    str.resize(size);
    if (size > 0)
    {
        MPI_Bcast(str.data(), size, MPI_CHAR, 0, MPI_COMM_WORLD);
    }
}

template <typename T>
void MPI_Bcast_vector(std::vector<T>& vec)
{
    uint32_t size = vec.size();
    MPI_Bcast_value(size);
    vec.resize(size);
    if (size > 0)
    {
        MPI_Bcast(vec.data(), size, MPI_Type_map<T>(), 0, MPI_COMM_WORLD);
    }
}

uint32_t MPI_Bcast_NetworkDescriptor(NetworkDescriptor& d)
{
    /**
     * @brief Broadcasts the name of the data structure using MPI.
     *
     * @param[in,out] name The name of the data structure to be broadcasted.
     */
    MPI_Bcast_string(d._name);

    /**
     * @brief Broadcasts the kind value of the data structure using MPI.
     *
     * @param[in,out] kind The kind value to be broadcasted.
     */
    MPI_Bcast_value(d._kind);

    /**
     * @brief Broadcasts the error function value of the data structure using MPI.
     *
     * @param[in,out] errorFunction The error function value to be broadcasted.
     */
    MPI_Bcast_value(d._errorFunction);

    /**
     * @brief Broadcasts the maxout_k value of the data structure using MPI.
     *
     * @param[in,out] maxout_k The maxout_k value to be broadcasted.
     */
    MPI_Bcast_value(d._maxout_k);

    /**
     * @brief Broadcasts the LRN_k value of the data structure using MPI.
     *
     * @param[in,out] LRN_k The LRN_k value to be broadcasted.
     */
    MPI_Bcast_value(d._LRN_k);

    /**
     * @brief Broadcasts the LRN_n value of the data structure using MPI.
     *
     * @param[in,out] LRN_n The LRN_n value to be broadcasted.
     */
    MPI_Bcast_value(d._LRN_n);

    /**
     * @brief Broadcasts the LRN_alpha value of the data structure using MPI.
     *
     * @param[in,out] LRN_alpha The LRN_alpha value to be broadcasted.
     */
    MPI_Bcast_value(d._LRN_alpha);

    /**
     * @brief Broadcasts the LRN_beta value of the data structure using MPI.
     *
     * @param[in,out] LRN_beta The LRN_beta value to be broadcasted.
     */
    MPI_Bcast_value(d._LRN_beta);

    /**
     * @brief Broadcasts the bSparsenessPenalty value of the data structure using MPI.
     *
     * @param[in,out] bSparsenessPenalty The bSparsenessPenalty value to be broadcasted.
     */
    MPI_Bcast_value(d._bSparsenessPenalty);

    /**
     * @brief Broadcasts the sparsenessPenalty_beta value of the data structure using MPI.
     *
     * @param[in,out] sparsenessPenalty_beta The sparsenessPenalty_beta value to be broadcasted.
     */
    MPI_Bcast_value(d._sparsenessPenalty_beta);

    /**
     * @brief Broadcasts the sparsenessPenalty_p value of the data structure using MPI.
     *
     * @param[in,out] sparsenessPenalty_p The sparsenessPenalty_p value to be broadcasted.
     */
    MPI_Bcast_value(d._sparsenessPenalty_p);

    /**
     * @brief Broadcasts the bDenoising value of the data structure using MPI.
     *
     * @param[in,out] bDenoising The bDenoising value to be broadcasted.
     */
    MPI_Bcast_value(d._bDenoising);

    /**
     * @brief Broadcasts the denoising_p value of the data structure using MPI.
     *
     * @param[in,out] denoising_p The denoising_p value to be broadcasted.
     */
    MPI_Bcast_value(d._denoising_p);

    /**
     * @brief Broadcasts the deltaBoost_one value of the data structure using MPI.
     *
     * @param[in,out] deltaBoost_one The deltaBoost_one value to be broadcasted.
     */
    MPI_Bcast_value(d._deltaBoost_one);

    /**
     * @brief Broadcasts the deltaBoost_zero value of the data structure using MPI.
     *
     * @param[in,out] deltaBoost_zero The deltaBoost_zero value to be broadcasted.
     */
    MPI_Bcast_value(d._deltaBoost_zero);

    /**
     * @brief Broadcasts the SMCE_oneScale value of the data structure using MPI.
     *
     * @param[in,out] SMCE_oneScale The SMCE_oneScale value to be broadcasted.
     */
    MPI_Bcast_value(d._SMCE_oneScale);

    /**
     * @brief Broadcasts the SMCE_zeroScale value of the data structure using MPI.
     *
     * @param[in,out] SMCE_zeroScale The SMCE_zeroScale value to be broadcasted.
     */
    MPI_Bcast_value(d._SMCE_zeroScale);

    /**
     * @brief Broadcasts the SMCE_oneTarget value of the data structure using MPI.
     *
     * @param[in,out] SMCE_oneTarget The SMCE_oneTarget value to be broadcasted.
     */
    MPI_Bcast_value(d._SMCE_oneTarget);

    /**
     * @brief Broadcasts the SMCE_zeroTarget value of the data structure using MPI.
     *
     * @param[in,out] SMCE_zeroTarget The SMCE_zeroTarget value to be broadcasted.
     */
    MPI_Bcast_value(d._SMCE_zeroTarget);

    /**
     * @brief Broadcasts the checkpoint_interval value of the data structure using MPI.
     *
     * @param[in,out] checkpoint_interval The checkpoint_interval value to be broadcasted.
     */
    MPI_Bcast_value(d._checkpoint_interval);

    /**
     * @brief Broadcasts the checkpoint_epochs value of the data structure using MPI.
     *
     * @param[in,out] checkpoint_epochs The checkpoint_epochs value to be broadcasted.
     */
    MPI_Bcast_value(d._checkpoint_epochs);

    /**
     * @brief Broadcasts the checkpoint_name of the data structure using MPI.
     *
     * @param[in,out] checkpoint_name The checkpoint_name to be broadcasted.
     */
    MPI_Bcast_string(d._checkpoint_name);

    /**
     * @brief Broadcasts the bShuffleIndices value of the data structure using MPI.
     *
     * @param[in,out] bShuffleIndices The bShuffleIndices value to be broadcasted.
     */
    MPI_Bcast_value(d._bShuffleIndices);

    /**
     * @brief Broadcasts the vLayerDescriptor vector of the data structure using MPI.
     *
     * @param[in,out] vLayerDescriptor The vLayerDescriptor vector to be broadcasted.
     */
    MPI_Bcast_vector(d._vLayerDescriptor);

    /**
     * @brief Broadcasts the vWeightDescriptor vector of the data structure using MPI.
     *
     * @param[in,out] vWeightDescriptor The vWeightDescriptor vector to be broadcasted.
     */
    MPI_Bcast_vector(d._vWeightDescriptor);

    return 0;
}
Network* LoadNeuralNetworkJSON(const string& fname, const uint32_t batch, const vector<DataSetBase*>& vDataSet)
{
    Network* pNetwork = nullptr;
    NetworkDescriptor nd;
    Json::Value index;
    Json::Reader reader;
    bool bValid = true;
    bool bWeightsSupplied = false;
    string wfname;

    if (getGpu()._id == 0)
    {
        std::ifstream stream(fname, std::ifstream::binary);
        bool parsedSuccess = reader.parse(stream, index, false);

        if (!parsedSuccess)
        {
            std::cout("LoadNeuralNetworkJSON: Failed to parse JSON file: %s, error: %s\n", fname.c_str(), reader.getFormattedErrorMessages().c_str());
            bValid = false;
        }
        else
        {
            Float version = _VERSION;
            set<string> sLayer;
            for (Json::ValueIterator itr = index.begin(); itr != index.end() ; itr++)
            {
                string name = itr.name();
                std::transform(name.begin(), name.end(), name.begin(), ::tolower);
                Json::Value key = itr.key();
                Json::Value value = *itr;
                string vstring = value.isString() ? value.asString() : "";
                std::transform(vstring.begin(), vstring.end(), vstring.begin(), ::tolower);

                if (name.compare("version") == 0)
                {
                    version = value.asFloat();
                    if (version < 0.6999)
                    {
                        std::cout("LoadNeuralNetworkJSON: version %f (must be at least 0.7)\n", version);
                        bValid = false;
                        goto exit;
                    }
                }

                else if (name.compare("name") == 0)
                {
                    nd._name = value.asString();
                }

                else if (name.compare("kind") == 0)
                {
                    if (vstring.compare("feedforward") == 0)
                        nd._kind = Network::Kind::FeedForward;
                    else if (vstring.compare("autoencoder") == 0)
                        nd._kind = Network::Kind::AutoEncoder;
                    else
                    {
                        std::cout("LoadNeuralNetworkJSON: Invalid network kind: %s\n", value.asString().c_str());
                        bValid = false;
                        goto exit;
                    }
                }

                else if (name.compare("weightsdata") == 0)
                {
                    bWeightsSupplied = true;
                    wfname = value.asString();
                }

                else if ((name.compare("lrn") == 0) || (name.compare("localresponsenormalization") == 0))
                {
                    for (Json::ValueIterator pitr = value.begin(); pitr != value.end() ; pitr++)
                    {
                        string pname = pitr.name();
                        std::transform(pname.begin(), pname.end(), pname.begin(), ::tolower);
                        Json::Value pkey = pitr.key();
                        Json::Value pvalue = *pitr;
                        if (pname.compare("k") == 0)
                            nd._LRN_k = pvalue.asFloat();
                        else if (pname.compare("n") == 0)
                            nd._LRN_n = pvalue.asInt();
                        else if (pname.compare("alpha") == 0)
                            nd._LRN_alpha = pvalue.asFloat();
                        else if (pname.compare("beta") == 0)
                            nd._LRN_beta = pvalue.asFloat();
                        else
                        {
                            name = pitr.name();
                            std::cout("LoadNeuralNetworkJSON: Invalid LocalResponseNormalization parameter: %s\n", name.c_str());
                            bValid = false;
                            goto exit;
                        }
                    }
                }

                else if (name.compare("maxout") == 0)
                {
                    for (Json::ValueIterator pitr = value.begin(); pitr != value.end() ; pitr++)
                    {
                        string pname = pitr.name();
                        std::transform(pname.begin(), pname.end(), pname.begin(), ::tolower);
                        Json::Value pkey = pitr.key();
                        Json::Value pvalue = *pitr;
                        if (pname.compare("k") == 0)
                            nd._maxout_k = pvalue.asFloat();
                        else
                        {
                            name = pitr.name();
                            std::cout("LoadNeuralNetworkJSON: Invalid MaxOut parameter: %s\n", name.c_str());
                            bValid = false;
                            goto exit;
                        }
                    }
                }
                else if (name.compare("sparsenesspenalty") == 0)
                {
                    for (Json::ValueIterator pitr = value.begin(); pitr != value.end() ; pitr++)
                    {
                        string pname = pitr.name();
                        std::transform(pname.begin(), pname.end(), pname.begin(), ::tolower);
                        Json::Value pkey = pitr.key();
                        Json::Value pvalue = *pitr;
                        if (pname.compare("p") == 0)
                            nd._sparsenessPenalty_p = pvalue.asFloat();
                        else if (pname.compare("beta") == 0)
                            nd._sparsenessPenalty_beta  = pvalue.asFloat();
                        else
                        {
                            name = pitr.name();
                            std::cout("LoadNeuralNetworkJSON: Invalid SparsenessPenalty parameter: %s\n", name.c_str());
                            bValid = false;
                            goto exit;
                        }
                    }
                }

                else if (name.compare("denoising") == 0)
                {
                    for (Json::ValueIterator pitr = value.begin(); pitr != value.end() ; pitr++)
                    {
                        string pname = pitr.name();
                        std::transform(pname.begin(), pname.end(), pname.begin(), ::tolower);
                        Json::Value pkey = pitr.key();
                        Json::Value pvalue = *pitr;
                        if (pname.compare("p") == 0)
                        {
                            nd._denoising_p = pvalue.asFloat();
                        }
                        else
                        {
                            name = pitr.name();
                            std::cout("LoadNeuralNetworkJSON: Invalid Denoising parameter: %s\n", name.c_str());
                            bValid = false;
                            goto exit;
                        }
                    }
                }

                else if (name.compare("deltaboost") == 0)
                {
                    for (Json::ValueIterator pitr = value.begin(); pitr != value.end() ; pitr++)
                    {
                        string pname = pitr.name();
                        std::transform(pname.begin(), pname.end(), pname.begin(), ::tolower);
                        Json::Value pkey = pitr.key();
                        Json::Value pvalue = *pitr;
                        if (pname.compare("one") == 0)
                            nd._deltaBoost_one = pvalue.asFloat();
                        else if (pname.compare("zero") == 0)
                            nd._deltaBoost_zero = pvalue.asFloat();
                        else
                        {
                            name = pitr.name();
                            std::cout("LoadNeuralNetworkJSON: Invalid DeltaBoost parameter: %s\n", name.c_str());
                            bValid = false;
                            goto exit;
                        }
                    }
                }

                else if ((name.compare("scaledmarginalcrossentropy") == 0) ||
                         (name.compare("datascaledmarginalcrossentropy") == 0))
                {
                    for (Json::ValueIterator pitr = value.begin(); pitr != value.end() ; pitr++)
                    {
                        string pname = pitr.name();
                        std::transform(pname.begin(), pname.end(), pname.begin(), ::tolower);
                        Json::Value pkey = pitr.key();
                        Json::Value pvalue = *pitr;
                        if (pname.compare("onescale") == 0)
                            nd._SMCE_oneScale = pvalue.asFloat();
                        else if (pname.compare("zeroscale") == 0)
                            nd._SMCE_zeroScale = pvalue.asFloat();
                        else if (pname.compare("onetarget") == 0)
                            nd._SMCE_oneTarget = pvalue.asFloat();
                        else if (pname.compare("zerotarget") == 0)
                            nd._SMCE_zeroTarget = pvalue.asFloat();
                        else
                        {
                            name = pitr.name();
                            std::cout("LoadNeuralNetworkJSON: Invalid ScaledMarginalCrossentropy parameter: %s\n", name.c_str());
                            bValid = false;
                            goto exit;
                        }
                    }
                }

                else if (name.compare("shuffleindices") == 0)
                {
                    /**
                     * @brief Set whether to shuffle indices.
                     */
                    nd._bShuffleIndices = value.asBool();
                }

                else if ((name.compare("reluslope") == 0) || (name.compare("slope") == 0))
                {
                    /**
                     * @brief Set the slope value for the ReLU activation function.
                     */
                    nd._RELUSlope = value.asFloat();
                }
                else if (name.compare("elualpha") == 0)
                {
                    /**
                     * @brief Set the alpha value for the ELU activation function.
                     */
                    nd._ELUAlpha = value.asFloat();
                }
                else if (name.compare("selulambda") == 0)
                {
                    /**
                     * @brief Set the lambda value for the SELU activation function.
                     */
                    nd._SELULambda = value.asFloat();
                }
                else if (name.compare("decay") == 0)
                {
                    /**
                     * @brief Set the decay value.
                     */
                    nd._decay = value.asFloat();
                }

                else if (name.compare("errorfunction") == 0)
                {
                    if (vstring.compare("l1") == 0)
                        nd._errorFunction = ErrorFunction::L1;
                    else if (vstring.compare("l2") == 0)
                        nd._errorFunction = ErrorFunction::L2;
                    else if (vstring.compare("l2hinge") == 0)
                        nd._errorFunction = ErrorFunction::L2Hinge;
                    else if (vstring.compare("hinge") == 0)
                        nd._errorFunction = ErrorFunction::Hinge;
                    else if ((vstring.compare("crossentropy") == 0) || (vstring.compare("cross entropy") == 0))
                        nd._errorFunction = ErrorFunction::CrossEntropy;
                    else if (vstring.compare("scaledmarginalcrossentropy") == 0)
                        nd._errorFunction = ErrorFunction::ScaledMarginalCrossEntropy;
                    else if (vstring.compare("datascaledmarginalcrossentropy") == 0)
                        nd._errorFunction = ErrorFunction::DataScaledMarginalCrossEntropy;
                    else
                    {
                        std::cout("LoadNeuralNetworkJSON: Invalid error function: %s\n", value.asString().c_str());
                        bValid = false;
                        goto exit;
                    }
                }

                else if (name.compare("layers") == 0)
                {
                    uint32_t size = value.isArray() ? value.size() : 1;
                    for (uint32_t i = 0; i < size; i++)
                    {
                        vector<WeightDescriptor> vSharedWeight;
                        LayerDescriptor ldl;
                        bool bSource = false;
                        Json::Value layer = value.isArray() ? value[i] : value;
                        bool bAutoSize = false;

                        if (i == 0)
                            ldl._kind = Layer::Kind::Input;
                        else if (i == size - 1)
                            ldl._kind = Layer::Kind::Output;
                        else
                            ldl._kind = Layer::Kind::Hidden;
                        ldl._type = Layer::Type::FullyCoected;


                        for (Json::ValueIterator litr = layer.begin(); litr != layer.end() ; litr++)
                        {
                            string lname = litr.name();
                            std::transform(lname.begin(), lname.end(), lname.begin(), ::tolower);
                            Json::Value lkey = litr.key();
                            Json::Value lvalue = *litr;

                            if (lname.compare("kind") == 0)
                            {
                                string s = lvalue.asString();
                                std::transform(s.begin(), s.end(), s.begin(), ::tolower);
                                if (s.compare("input") == 0)
                                    ldl._kind = Layer::Kind::Input;
                                else if (s.compare("hidden") == 0)
                                    ldl._kind = Layer::Kind::Hidden;
                                else if (s.compare("target") == 0)
                                    ldl._kind = Layer::Kind::Target;
                                else if (s.compare("output") == 0)
                                    ldl._kind = Layer::Kind::Output;
                                else
                                {
                                    std::cout("LoadNeuralNetworkJSON: Invalid layer kind: %s\n", lvalue.asString().c_str());
                                    bValid = false;
                                    goto exit;
                                }
                            }

                            else if (lname.compare("type") == 0)
                            {
                                string s = lvalue.asString();
                                std::transform(s.begin(), s.end(), s.begin(), ::tolower);
                                if (s.compare("fullycoected") == 0)
                                    ldl._type = Layer::Type::FullyCoected;
                                else if (s.compare("convolutional") == 0)
                                    ldl._type = Layer::Type::Convolutional;
                                else if (s.compare("pooling") == 0)
                                    ldl._type = Layer::Type::Pooling;
                                else
                                {
                                    std::cout("LoadNeuralNetworkJSON: Invalid layer type: %s\n", lvalue.asString().c_str());
                                    bValid = false;
                                    goto exit;
                                }
                            }
                        }

                        if ((ldl._type == Layer::Type::Pooling) || (ldl._type == Layer::Type::Convolutional))
                        {
                            ldl._bDimensionsProvided = false;
                        }

                        switch (ldl._kind)
                        {
                            case Layer::Kind::Input:
                                ldl._name = "Input" + to_string(nd._vLayerDescriptor.size());
                                break;

                            case Layer::Kind::Hidden:
                                ldl._name = "Hidden" + to_string(nd._vLayerDescriptor.size());
                                break;

                            case Layer::Kind::Output:
                                ldl._name = "Output" + to_string(nd._vLayerDescriptor.size());
                                break;

                            case Layer::Kind::Target:
                                ldl._name = "Target" + to_string(nd._vLayerDescriptor.size());
                                break;
                        }

                        for (Json::ValueIterator litr = layer.begin(); litr != layer.end() ; litr++)
                        {
                            string lname = litr.name();
                            std::transform(lname.begin(), lname.end(), lname.begin(), ::tolower);
                            Json::Value lkey = litr.key();
                            Json::Value lvalue = *litr;

                            if ((lname.compare("kind") == 0) || (lname.compare("type") == 0))
                            {
                                continue;
                            }

                            if (lname.compare("name") == 0)
                            {
                                ldl._name = lvalue.asString();
                                if (sLayer.find(ldl._name) != sLayer.end())
                                {
                                    std::cout("LoadNeuralNetworkJSON: Duplicate layer name detected: %s\n", ldl._name.c_str());
                                    bValid = false;
                                    goto exit;
                                }
                                sLayer.insert(ldl._name);
                                continue;
                            }

                            if (lname.compare("sparse") == 0)
                            {
                                if (lvalue.asBool())
                                    ldl._attributes|= Layer::Attributes::Sparse;
                                continue;
                            }
                            else if (lname.compare("n") == 0)
                            {
                                if (lvalue.isArray())
                                {
                                    if (lvalue.size() < 5)
                                    {
                                        ldl._dimensions     = lvalue.size();
                                        switch (lvalue.size())
                                        {
                                            case 4:
                                                ldl._Nw = lvalue[3].asInt();
                                            case 3:
                                                ldl._Nz = lvalue[2].asInt();
                                            case 2:
                                                ldl._Ny = lvalue[1].asInt();
                                            case 1:
                                                ldl._Nx = lvalue[0].asInt();
                                        }

                                    }
                                    else
                                    {
                                        std::cout("LoadNeuralNetworkJSON: >4 dimensions detected in layer: %s\n", ldl._name.c_str());
                                        bValid = false;
                                        goto exit;
                                    }

                                }
                                else if (lvalue.isString())
                                {
                                    string nstring = lvalue.asString();
                                    std::transform(nstring.begin(), nstring.end(), nstring.begin(), ::tolower);
                                    if ((ldl._kind != Layer::Kind::Hidden) && (nstring.compare("auto") == 0))
                                        bAutoSize = true;
                                    else if (nstring.compare("auto") == 0)
                                    {
                                        std::cout("LoadNeuralNetworkJSON: Illegal attempt to use auto for hidden layer: %s\n", ldl._name.c_str());
                                        bValid = false;
                                        goto exit;
                                    }
                                }
                                else
                                {
                                    ldl._Nx = lvalue.asInt();
                                    ldl._dimensions = 1;
                                }
                                continue;
                            }
                            else if (lname.compare("pdropout") == 0)
                            {
                                ldl._pDropout = lvalue.asFloat();
                                continue;
                            }


                            if (ldl._kind != Layer::Kind::Input)
                            {
                                if (lname.compare("source") == 0)
                                {
                                    uint32_t size = lvalue.isArray() ? lvalue.size() : 1;

#if 0
                                    if ((ldl._type == Layer::Type::Pooling) && (size > 1))
                                    {
                                            std::cout("LoadNeuralNetworkJSON: Pooling layer %s has multiple sources\n", ldl._name.c_str());
                                            bValid = false;
                                            goto exit;
                                    }
#endif

                                    for (uint32_t j = 0; j < size; j++)
                                    {
                                        Json::Value src = lvalue.isArray() ? lvalue[j] : lvalue;
                                        ldl._vSource.push_back(src.asString());
                                        bSource = true;
                                    }
                                    continue;
                                }

                                else if ((lname.compare("kernel") == 0) || (lname.compare("kernelstride") == 0))
                                {
                                    uint32_t x = 1;
                                    uint32_t y = 1;
                                    uint32_t z = 1;
                                    uint32_t dimensions = 1;
                                    if (lvalue.isArray())
                                    {
                                        if (lvalue.size() < 4)
                                        {
                                            dimensions = lvalue.size();
                                            switch (lvalue.size())
                                            {
                                                case 3:
                                                    z = lvalue[2].asInt();
                                                case 2:
                                                    y = lvalue[1].asInt();
                                                case 1:
                                                    x = lvalue[0].asInt();
                                            }
                                        }
                                        else
                                        {
                                            bValid = false;
                                            goto exit;
                                        }
                                    }
                                    else
                                    {
                                        x = lvalue.asInt();
                                    }

                                    if (lname.compare("kernel") == 0)
                                    {
                                        ldl._kernelX = x;
                                        ldl._kernelY = y;
                                        ldl._kernelZ = z;
                                        ldl._kernelDimensions = dimensions;
                                    }
                                    else
                                    {
                                        ldl._kernelStrideX = x;
                                        ldl._kernelStrideY = y;
                                        ldl._kernelStrideZ = z;
                                    }
                                    continue;
                                }
                            }




                            if (ldl._kind == Layer::Kind::Hidden)
                            {
                                if (lname.compare("batchnormalization") == 0)
                                {
                                    if (lvalue.asBool())
                                        ldl._attributes|= Layer::Attributes::BatchNormalization;
                                    continue;
                                }

                                else if (lname.compare("sparsenesspenalty") == 0)
                                {
                                    for (Json::ValueIterator pitr = lvalue.begin(); pitr != lvalue.end() ; pitr++)
                                    {
                                        string pname = pitr.name();
                                        std::transform(pname.begin(), pname.end(), pname.begin(), ::tolower);
                                        Json::Value pkey = pitr.key();
                                        Json::Value pvalue = *pitr;
                                        if (pname.compare("p") == 0)
                                            ldl._sparsenessPenalty_p = pvalue.asFloat();
                                        else if (pname.compare("beta") == 0)
                                            ldl._sparsenessPenalty_beta  = pvalue.asFloat();
                                        else
                                        {
                                            std::cout("LoadNeuralNetworkJSON: Invalid sparseness penalty parameter for hidden layer %s\n", ldl._name.c_str());
                                            bValid = false;
                                            goto exit;
                                        }
                                    }
                                    continue;
                                }
                            }

                            if (ldl._kind == Layer::Kind::Output)
                            {

                            }

                            if ((ldl._kind == Layer::Kind::Hidden) || (ldl._kind == Layer::Kind::Output))
                            {
                                if (ldl._type == Layer::Type::Pooling)
                                {
                                    if (lname.compare("function") == 0)
                                    {
                                        string s = lvalue.asString();
                                        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
                                        if (s.compare("max") == 0)
                                            ldl._poolingFunction = PoolingFunction::Max;
                                        else if (s.compare("maxout") == 0)
                                            ldl._poolingFunction = PoolingFunction::Maxout;
                                        else if (s.compare("dotproduct") == 0)
                                            ldl._poolingFunction = PoolingFunction::DotProduct;
                                        else if (s.compare("cosine") == 0)
                                            ldl._poolingFunction = PoolingFunction::Cosine;
                                        else if (s.compare("average") == 0)
                                            ldl._poolingFunction = PoolingFunction::Average;
                                        else if ((s.compare("lrn") == 0) || (s.compare("localresponsenormalization") == 0))
                                            ldl._poolingFunction = PoolingFunction::LRN;
                                        else
                                        {
                                            std::cout("LoadNeuralNetworkJSON: Invalid pooling function (%s) for pooling layer %s\n", lvalue.asString().c_str(), ldl._name.c_str());
                                            bValid = false;
                                            goto exit;
                                        }
                                        continue;
                                    }
                                }

                                if (lname.compare("skip") == 0)
                                {
                                    uint32_t size = lvalue.isArray() ? lvalue.size() : 1;
                                    for (uint32_t j = 0; j < size; j++)
                                    {
                                        Json::Value src = lvalue.isArray() ? lvalue[j] : lvalue;
                                        ldl._vSkip.push_back(src.asString());
                                    }
                                    continue;
                                }

                                else if (lname.compare("activation") == 0)
                                {
                                    string s = lvalue.asString();
                                    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
                                    if (s.compare("sigmoid") == 0)
                                        ldl._activation = Activation::Sigmoid;
                                    else if (s.compare("tanh") == 0)
                                        ldl._activation = Activation::Tanh;
                                    else if (s.compare("linear") == 0)
                                        ldl._activation = Activation::Linear;
                                    else if ((s.compare("relu") == 0) || (s.compare("rectifiedlinear") == 0))
                                        ldl._activation = Activation::RectifiedLinear;
                                    else if ((s.compare("lrelu") == 0) || (s.compare("leakyrectifiedlinear") == 0))
                                        ldl._activation = Activation::LeakyRectifiedLinear;
                                    else if ((s.compare("elu") == 0) || (s.compare("exponentiallinear") == 0))
                                        ldl._activation = Activation::ExponentialLinear;
                                    else if ((s.compare("selu") == 0) || (s.compare("scaledexponentiallinear") == 0))
                                        ldl._activation = Activation::ScaledExponentialLinear;
                                    else if (s.compare("softplus") == 0)
                                        ldl._activation = Activation::SoftPlus;
                                    else if (s.compare("softsign") == 0)
                                        ldl._activation = Activation::SoftSign;
                                    else if (s.compare("softmax") == 0)
                                        ldl._activation = Activation::SoftMax;
                                    else if (s.compare("relumax") == 0)
                                        ldl._activation = Activation::RELUMax;
                                    else if (s.compare("linearmax") == 0)
                                        ldl._activation = Activation::LinearMax;
                                    else
                                    {
                                        std::cout("LoadNeuralNetworkJSON: Invalid layer activation: %s\n", lvalue.asString().c_str());
                                        bValid = false;
                                        goto exit;
                                    }
                                    continue;
                                }

                                else if ((lname.compare("reluslope") == 0) || (lname.compare("slope") == 0))
                                {
                                    ldl._RELUSlope = lvalue.asFloat();
                                    continue;
                                }
                                else if (lname.compare("elualpha") == 0)
                                {
                                    ldl._ELUAlpha = lvalue.asFloat();
                                    continue;
                                }
                                else if (lname.compare("selulambda") == 0)
                                {
                                    ldl._SELULambda = lvalue.asFloat();
                                    continue;
                                }

                                else if (lname.compare("weightnorm") == 0)
                                {
                                    ldl._weightNorm = lvalue.asFloat();
                                    continue;
                                }

                                else if (lname.compare("deltanorm") == 0)
                                {
                                    ldl._deltaNorm = lvalue.asFloat();
                                    continue;
                                }

                                else if (lname.compare("weightinit") == 0)
                                {
                                    for (int i = 0; i < lvalue.size(); i++)
                                    {
                                        for (Json::ValueIterator witr = lvalue.begin(); witr != lvalue.end() ; witr++)
                                        {
                                            string wname = witr.name();
                                            std::transform(wname.begin(), wname.end(), wname.begin(), ::tolower);
                                            Json::Value wkey = witr.key();
                                            Json::Value wvalue = *witr;

                                            if (wname.compare("scheme") == 0)
                                            {
                                                string scheme = wvalue.asString();
                                                std::transform(scheme.begin(), scheme.end(), scheme.begin(), ::tolower);
                                                if (scheme.compare("xavier") == 0)
                                                    ldl._weightInit = Xavier;
                                                else if (scheme.compare("caffexavier") == 0)
                                                    ldl._weightInit = CaffeXavier;
                                                else if (scheme.compare("gaussian") == 0)
                                                    ldl._weightInit = Gaussian;
                                                else if (scheme.compare("uniform") == 0)
                                                    ldl._weightInit = Uniform;
                                                else if (scheme.compare("unitball") == 0)
                                                    ldl._weightInit = UnitBall;
                                                else if (scheme.compare("constant") == 0)
                                                    ldl._weightInit = Constant;
                                                else if (scheme.compare("selu") == 0)
                                                    ldl._weightInit = SELU;
                                                else
                                                {
                                                    std::cout("LoadNeuralNetworkJSON: Invalid weight initialization scheme: %s\n", scheme.c_str());
                                                    bValid = false;
                                                    goto exit;
                                                }
                                            }
                                            else if (wname.compare("scale") == 0)
                                            {
                                               ldl._weightInitScale = wvalue.asFloat();
                                            }
                                            else if (wname.compare("bias") == 0)
                                            {
                                               ldl._biasInit = wvalue.asFloat();
                                            }
                                            else
                                            {
                                                std::cout("LoadNeuralNetworkJSON: Invalid weight initialization field: %s\n", wname.c_str());
                                                bValid = false;
                                                goto exit;
                                            }
                                        }
                                    }
                                    continue;
                                }

                                else if (lname.compare("sharedweights") == 0)
                                {
                                    uint32_t size = lvalue.isArray() ? lvalue.size() : 1;
                                    for (uint32_t i = 0; i < size; i++)
                                    {
                                        WeightDescriptor nd;
                                        Json::Value share = lvalue.isArray() ? lvalue[i] : lvalue;
                                        for (Json::ValueIterator sitr = share.begin(); sitr != share.end() ; sitr++)
                                        {
                                            string sname = sitr.name();
                                            std::transform(sname.begin(), sname.end(), sname.begin(), ::tolower);
                                            Json::Value skey = sitr.key();
                                            Json::Value svalue = *sitr;

                                            if (sname.compare("sourceinputlayer") == 0)
                                            {
                                                nd._sourceInputLayer = svalue.asString();
                                            }
                                            else if (sname.compare("sourceoutputlayer") == 0)
                                            {
                                                nd._sourceOutputLayer = svalue.asString();
                                            }
                                            else if (sname.compare("inputlayer") == 0)
                                            {
                                                nd._inputLayer = svalue.asString();
                                            }
                                            else if (sname.compare("transposed") == 0)
                                            {
                                                nd._bTransposed = svalue.asBool();
                                            }
                                            else
                                            {
                                                std::cout("LoadNeuralNetworkJSON: Invalid shared weight field: %s\n", sname.c_str());
                                                bValid = false;
                                                goto exit;
                                            }
                                        }
                                        /**
                                         * @brief Sets the shared flag in the network description.
                                         *
                                         * @param nd The network description object to modify.
                                         *
                                         * @note This flag indicates that the network weights are shared.
                                         */
                                        nd._bShared = true;

                                        /**
                                         * @brief Adds the network description object to the vector of shared weights.
                                         *
                                         * @param vSharedWeight The vector of shared weights.
                                         *
                                         * @note The network description object is added to the vector for tracking shared weights.
                                         */
                                        vSharedWeight.push_back(nd);
                                    }
                                    continue;
                                }
                            }


                            if ((ldl._kind == Layer::Kind::Input) || (ldl._kind == Layer::Kind::Output))
                            {
                                if (lname.compare("dataset") == 0)
                                {
                                    ldl._dataSet = lvalue.asString();
                                    continue;
                                }

                            }

                            std::cout("LoadNeuralNetworkJSON: Unknown neural network layer field: %s\n", lname.c_str());
                            bValid = false;
                            goto exit;
                        }

                        if (bAutoSize)
                        {
                            bool bFound = false;
                            for (auto p : vDataSet)
                            {
                                if (p->_name.compare(ldl._dataSet) == 0)
                                {
                                    /**
                                     * @brief Sets the dimensions of a LayerDescriptor object based on the properties of a given parameter object.
                                     *
                                     * @param ldl The LayerDescriptor object to modify.
                                     * @param p The parameter object from which to retrieve the dimensions.
                                     */
                                    ldl._Nx = p->_width;

                                    /**
                                     * @brief Sets the dimensions of a LayerDescriptor object based on the properties of a given parameter object.
                                     *
                                     * @param ldl The LayerDescriptor object to modify.
                                     * @param p The parameter object from which to retrieve the dimensions.
                                     */
                                    ldl._Ny = p->_height;

                                    /**
                                     * @brief Sets the dimensions of a LayerDescriptor object based on the properties of a given parameter object.
                                     *
                                     * @param ldl The LayerDescriptor object to modify.
                                     * @param p The parameter object from which to retrieve the dimensions.
                                     */
                                    ldl._Nz = p->_length;

                                    /**
                                     * @brief Sets the dimensions of a LayerDescriptor object based on the properties of a given parameter object.
                                     *
                                     * @param ldl The LayerDescriptor object to modify.
                                     * @param p The parameter object from which to retrieve the dimensions.
                                     */
                                    ldl._dimensions = p->_dimensions;

                                    /**
                                     * @brief Marks the layer descriptor as found.
                                     *
                                     * @note This variable is typically used as a flag to indicate that a specific condition has been met.
                                     */
                                    bFound = true;
                                }
                            }
                            if (!bFound)
                            {
                                std::cout("LoadNeuralNetworkJSON: Unable to find data set %s to determine dimensions for layer: %s\n", ldl._dataSet.c_str(), ldl._name.c_str());
                                bValid = false;
                                goto exit;
                            }
                        }

                        if (!bSource && (ldl._kind != Layer::Kind::Input))
                        {
                            ldl._vSource.push_back(nd._vLayerDescriptor.back()._name);
                        }

                        if ((ldl._type == Layer::Type::Pooling) &&
                            (ldl._poolingFunction == PoolingFunction::DotProduct) || (ldl._poolingFunction == PoolingFunction::Cosine))
                        {
                            if (ldl._vSource.size() < 2)
                            {
                                std::cout("LoadNeuralNetworkJSON: Dot product layer %s must have 2 or more sources\n", ldl._name.c_str());
                                bValid = false;
                                goto exit;
                            }
                            ldl._Nx = ldl._vSource.size() - 1;
                            ldl._Ny = 1;
                            ldl._Nz = 1;
                            ldl._dimensions = 1;
                        }

                        if (ldl._type != Layer::Type::Pooling)
                        {

                            uint32_t sharedWeightsFound = 0;
                            for (uint32_t i = 0; i < ldl._vSource.size(); i++)
                            {
                                WeightDescriptor wd;
                                wd._inputLayer = ldl._vSource[i];
                                wd._outputLayer = ldl._name;
                                wd._norm = ldl._weightNorm;

                                for (uint32_t j = 0; j < vSharedWeight.size(); j++)
                                {
                                    if (vSharedWeight[j]._inputLayer == wd._inputLayer)
                                    {
                                        /**
                                         * @brief Sets the shared properties for a WeightData object.
                                         *
                                         * @param wd The WeightData object to modify.
                                         * @param vSharedWeight The vector of shared weights.
                                         * @param j The index of the shared weight to set.
                                         */
                                        wd._bShared = true;

                                        /**
                                         * @brief Sets the transposed flag for the shared weight.
                                         *
                                         * @note The transposed flag indicates whether the weight is transposed or not.
                                         */
                                        wd._bTransposed = vSharedWeight[j]._bTransposed;

                                        /**
                                         * @brief Sets the input layer source for the shared weight.
                                         *
                                         * @note The sourceInputLayer specifies the input layer from which the weight originates.
                                         */
                                        wd._sourceInputLayer = vSharedWeight[j]._sourceInputLayer;

                                        /**
                                         * @brief Sets the output layer source for the shared weight.
                                         *
                                         * @note The sourceOutputLayer specifies the output layer to which the weight connects.
                                         */
                                        wd._sourceOutputLayer = vSharedWeight[j]._sourceOutputLayer;

                                        /**
                                         * @brief Increments the count of shared weights found.
                                         *
                                         * @note sharedWeightsFound keeps track of the number of shared weights discovered.
                                         */
                                        sharedWeightsFound++;

                                        /**
                                         * @brief Breaks the loop after finding the shared weight.
                                         *
                                         * @note This statement breaks out of the loop once a shared weight is found.
                                         */
                                        break;
                                    }
                                }
                                nd._vWeightDescriptor.push_back(wd);
                            }

                            if (sharedWeightsFound < vSharedWeight.size())
                            {
                                std::cout("LoadNeuralNetworkJSON: Unable to locate all shared weights\n");
                                bValid = false;
                                goto exit;
                            }
                        }

                        if (ldl._dimensions < ldl._kernelDimensions)
                        {
                            ldl._bDimensionsProvided = false;
                        }

                        nd._vLayerDescriptor.push_back(ldl);
                    }
                }

                else
                {
                    std::cout("LoadNeuralNetworkJSON: Unknown neural network field: %s\n", name.c_str());
                    bValid = false;
                    goto exit;
                }
            }
        }

        if (nd._sparsenessPenalty_beta > (Float)0.0)
            nd._bSparsenessPenalty = true;

        if (nd._denoising_p > (Float)0.0)
        {
            nd._bDenoising = true;
            for (size_t i = 0; i <  nd._vLayerDescriptor.size(); i++)
            {
                if ((nd._vLayerDescriptor[i]._kind == Layer::Kind::Input) && ((nd._vLayerDescriptor[i]._attributes & Layer::Attributes::Sparse) != 0))
                {
                    nd._vLayerDescriptor[i]._attributes |= Layer::Attributes::Denoising;
                }
            }
        }
    }

    for (size_t i = 0; i <  nd._vLayerDescriptor.size(); i++)
    {
        /**
         * @brief Checks and updates the activation function parameters for a layer descriptor if they are NaN (Not a Number).
         *
         * @param nd The network description containing the layer descriptors.
         * @param i The index of the layer descriptor to check and update.
         *
         * @note If any activation function parameter in the layer descriptor is NaN, it will be replaced with the corresponding value from the network description.
         * @see NetworkDescription
         */
        if (isnan(nd._vLayerDescriptor[i]._RELUSlope))
            nd._vLayerDescriptor[i]._RELUSlope = nd._RELUSlope;

        /**
         * @brief Checks and updates the activation function parameters for a layer descriptor if they are NaN (Not a Number).
         *
         * @param nd The network description containing the layer descriptors.
         * @param i The index of the layer descriptor to check and update.
         *
         * @note If any activation function parameter in the layer descriptor is NaN, it will be replaced with the corresponding value from the network description.
         * @see NetworkDescription
         */
        if (isnan(nd._vLayerDescriptor[i]._ELUAlpha))
            nd._vLayerDescriptor[i]._ELUAlpha = nd._ELUAlpha;

        /**
         * @brief Checks and updates the activation function parameters for a layer descriptor if they are NaN (Not a Number).
         *
         * @param nd The network description containing the layer descriptors.
         * @param i The index of the layer descriptor to check and update.
         *
         * @note If any activation function parameter in the layer descriptor is NaN, it will be replaced with the corresponding value from the network description.
         * @see NetworkDescription
         */
        if (isnan(nd._vLayerDescriptor[i]._SELULambda))
            nd._vLayerDescriptor[i]._SELULambda = nd._SELULambda;
    }

    CalculateDerivedLayerDimensions(nd);

exit:
    MPI_Bcast(&bValid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!bValid)
    {
        getGpu().Shutdown();
        exit(-1);
    }


    MPI_Bcast_NetworkDescriptor(nd);

    if (getGpu()._id == 0)
    {
        std::cout << "LoadNeuralNetworkJSON: Enumerating network:" << std::endl;
        std::cout << nd << std::endl;
    }

    /**
     * @brief Creates a new Network object.
     *
     * @param nd The network description.
     * @param batch The batch size for the network.
     * @return A pointer to the newly created Network object.
     */
    pNetwork = new Network(nd, batch);

    /**
     * @brief Returns the pointer to the newly created Network object.
     *
     * @note The caller of this function is responsible for deleting the Network object.
     * @see deleteNetwork()
     */
    return pNetwork;
}

std::unique_ptr<Network> LoadNeuralNetworkNetCDF(const std::string& fname, const uint32_t batch)
{
    /**
     * @brief Creates a unique pointer to a Network object.
     */
    auto pNetwork = std::make_unique<Network>();

    /**
     * @brief Network descriptor object.
     */
    NetworkDescriptor nd;

    /**
     * @brief Flag indicating the result of an operation.
     */
    bool bResult = true;

    /**
     * @brief Version number.
     */
    float version = 0.0f;

    /**
     * @brief Number of layers in the network.
     */
    uint32_t layers = 0;

    /**
     * @brief Number of weights in the network.
     */
    uint32_t weights = 0;

    /**
     * @brief Broadcasts a string using MPI.
     * 
     * @param nd._name The string to be broadcasted.
     */
    MPI_Bcast_string(nd._name);

    /**
     * @brief Flag indicating whether convolutional layers are calculated.
     */
    nd._bConvLayersCalculated = true;

    if (getGpu()._id == 0)
    {
        bool bOpened = false;
        try
        {
            NcFile nc(fname, NcFile::read);
            bOpened = true;

            auto getAttribute = [&](const std::string& name, auto& value) {
                NcGroupAtt attribute = nc.getAtt(name);
                if (attribute.isNull())
                {
                    throw NC_EXCEPTION("NcException", "Network::etwork: No " + name + " supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                attribute.getValues(&value);
            };

            /**
             * Retrieves the value of the "version" attribute and assigns it to the variable "version".
             * @param attribute The name of the attribute to retrieve.
             * @param version [out] The variable to store the retrieved value.
             */
            getAttribute("version", version);

            /**
             * Retrieves the value of the "name" attribute and assigns it to the variable "nd._name".
             * @param attribute The name of the attribute to retrieve.
             * @param nd._name [out] The variable to store the retrieved value.
             */
            getAttribute("name", nd._name);

            /**
             * Retrieves the value of the "kind" attribute and assigns it to the variable "nd._kind".
             * @param attribute The name of the attribute to retrieve.
             * @param nd._kind [out] The variable to store the retrieved value.
             */
            getAttribute("kind", nd._kind);

            /**
             * Retrieves the value of the "errorFunction" attribute and assigns it to the variable "nd._errorFunction".
             * @param attribute The name of the attribute to retrieve.
             * @param nd._errorFunction [out] The variable to store the retrieved value.
             */
            getAttribute("errorFunction", nd._errorFunction);

            /**
             * Retrieves the value of the "decay" attribute and assigns it to the variable "nd._decay".
             * @param attribute The name of the attribute to retrieve.
             * @param nd._decay [out] The variable to store the retrieved value.
             */
            getAttribute("decay", nd._decay);

            /**
             * Retrieves the value of the "maxout_k" attribute and assigns it to the variable "nd._maxout_k".
             * @param attribute The name of the attribute to retrieve.
             * @param nd._maxout_k [out] The variable to store the retrieved value.
             */
            getAttribute("maxout_k", nd._maxout_k);

            /**
             * Retrieves the value of the "LRN_k" attribute and assigns it to the variable "nd._LRN_k".
             * @param attribute The name of the attribute to retrieve.
             * @param nd._LRN_k [out] The variable to store the retrieved value.
             */
            getAttribute("LRN_k", nd._LRN_k);

            /**
             * Retrieves the value of the "LRN_n" attribute and assigns it to the variable "nd._LRN_n".
             * @param attribute The name of the attribute to retrieve.
             * @param nd._LRN_n [out] The variable to store the retrieved value.
             */
            getAttribute("LRN_n", nd._LRN_n);

            /**
             * Retrieves the value of the "LRN_alpha" attribute and assigns it to the variable "nd._LRN_alpha".
             * @param attribute The name of the attribute to retrieve.
             * @param nd._LRN_alpha [out] The variable to store the retrieved value.
             */
            getAttribute("LRN_alpha", nd._LRN_alpha);

            /**
             * Retrieves the value of the "LRN_beta" attribute and assigns it to the variable "nd._LRN_beta".
             * @param attribute The name of the attribute to retrieve.
             * @param nd._LRN_beta [out] The variable to store the retrieved value.
             */
            getAttribute("LRN_beta", nd._LRN_beta);

            uint32_t bSparsenessPenalty;

            /**
             * Retrieves the value of the "bSparsenessPenalty" attribute and assigns it to the variable "bSparsenessPenalty".
             * @param attribute The name of the attribute to retrieve.
             * @param bSparsenessPenalty [out] The variable to store the retrieved value.
             */
            getAttribute("bSparsenessPenalty", bSparsenessPenalty);
            nd._bSparsenessPenalty = (bSparsenessPenalty != 0);

            /**
             * Retrieves the value of the "sparsenessPenalty_p" attribute and assigns it to the variable "nd._sparsenessPenalty_p".
             * @param attribute The name of the attribute to retrieve.
             * @param nd._sparsenessPenalty_p [out] The variable to store the retrieved value.
             */
            getAttribute("sparsenessPenalty_p", nd._sparsenessPenalty_p);

            /**
             * Retrieves the value of the "sparsenessPenalty_beta" attribute and assigns it to the variable "nd._sparsenessPenalty_beta".
             * @param attribute The name of the attribute to retrieve.
             * @param nd._sparsenessPenalty_beta [out] The variable to store the retrieved value.
             */
            getAttribute("sparsenessPenalty_beta", nd._sparsenessPenalty_beta);

            uint32_t bDenoising;

            /**
             * Retrieves the value of the "bDenoising" attribute and assigns it to the variable "bDenoising".
             * @param attribute The name of the attribute to retrieve.
             * @param bDenoising [out] The variable to store the retrieved value.
             */
            getAttribute("bDenoising", bDenoising);
            nd._bDenoising = (bDenoising != 0);

            /**
             * Retrieves the value of the "denoising_p" attribute and assigns it to the variable "nd._denoising_p".
             * @param attribute The name of the attribute to retrieve.
             * @param nd._denoising_p [out] The variable to store the retrieved value.
             */
            getAttribute("denoising_p", nd._denoising_p);

            /**
             * Retrieves the value of the "deltaBoost_one" attribute and assigns it to the variable "nd._deltaBoost_one".
             * @param attribute The name of the attribute to retrieve.
             * @param nd._deltaBoost_one [out] The variable to store the retrieved value.
             */
            getAttribute("deltaBoost_one", nd._deltaBoost_one);

            /**
             * Retrieves the value of the "deltaBoost_zero" attribute and assigns it to the variable "nd._deltaBoost_zero".
             * @param attribute The name of the attribute to retrieve.
             * @param nd._deltaBoost_zero [out] The variable to store the retrieved value.
             */
            getAttribute("deltaBoost_zero", nd._deltaBoost_zero);

            /**
             * Retrieves the value of the "SMCE_oneScale" attribute and assigns it to the variable "nd._SMCE_oneScale".
             * @param attribute The name of the attribute to retrieve.
             * @param nd._SMCE_oneScale [out] The variable to store the retrieved value.
             */
            getAttribute("SMCE_oneScale", nd._SMCE_oneScale);

            /**
             * Retrieves the value of the "SMCE_zeroScale" attribute and assigns it to the variable "nd._SMCE_zeroScale".
             * @param attribute The name of the attribute to retrieve.
             * @param nd._SMCE_zeroScale [out] The variable to store the retrieved value.
             */
            getAttribute("SMCE_zeroScale", nd._SMCE_zeroScale);

            /**
             * Retrieves the value of the "SMCE_oneTarget" attribute and assigns it to the variable "nd._SMCE_oneTarget".
             * @param attribute The name of the attribute to retrieve.
             * @param nd._SMCE_oneTarget [out] The variable to store the retrieved value.
             */
            getAttribute("SMCE_oneTarget", nd._SMCE_oneTarget);

            /**
             * Retrieves the value of the "SMCE_zeroTarget" attribute and assigns it to the variable "nd._SMCE_zeroTarget".
             * @param attribute The name of the attribute to retrieve.
             * @param nd._SMCE_zeroTarget [out] The variable to store the retrieved value.
             */
            getAttribute("SMCE_zeroTarget", nd._SMCE_zeroTarget);

            /**
             * Retrieves the "checkpoint_name" attribute from the NcGroupAtt object "checkpoint_nameAtt" and stores the values in the variable "nd._checkpoint_name".
             * @param checkpoint_nameAtt The NcGroupAtt object representing the "checkpoint_name" attribute.
             */
            NcGroupAtt checkpoint_nameAtt = nc.getAtt("checkpoint_name");
            if (!checkpoint_nameAtt.isNull())
            {
                checkpoint_nameAtt.getValues(nd._checkpoint_name);
            }

            /**
             * Retrieves the "checkpoint_interval" attribute from the NcGroupAtt object "checkpoint_intervalAtt" and stores the value in the variable "nd._checkpoint_interval".
             * @param checkpoint_intervalAtt The NcGroupAtt object representing the "checkpoint_interval" attribute.
             */
            NcGroupAtt checkpoint_intervalAtt = nc.getAtt("checkpoint_interval");
            if (!checkpoint_intervalAtt.isNull())
            {
                checkpoint_intervalAtt.getValues(&nd._checkpoint_interval);
            }

            /**
             * Retrieves the "checkpoint_epochs" attribute from the NcGroupAtt object "checkpoint_epochsAtt" and stores the value in the variable "nd._checkpoint_epochs".
             * @param checkpoint_epochsAtt The NcGroupAtt object representing the "checkpoint_epochs" attribute.
             */
            NcGroupAtt checkpoint_epochsAtt = nc.getAtt("checkpoint_epochs");
            if (!checkpoint_epochsAtt.isNull())
            {
                checkpoint_epochsAtt.getValues(&nd._checkpoint_epochs);
            }

            uint32_t bShuffleIndices;

            /**
             * Retrieves the value of the "ShuffleIndices" attribute and assigns it to the variable "bShuffleIndices".
             * @param attribute The name of the attribute to retrieve.
             * @param bShuffleIndices [out] The variable to store the retrieved value.
             */
            getAttribute("ShuffleIndices", bShuffleIndices);
            nd._bShuffleIndices = (bShuffleIndices != 0);

            /**
             * Retrieves the value of the "layers" attribute and assigns it to the variable "layers".
             * @param attribute The name of the attribute to retrieve.
             * @param layers [out] The variable to store the retrieved value.
             */
            getAttribute("layers", layers);

            for (uint32_t i = 0; i < layers; i++)
            {
                LayerDescriptor ld;
                if (!LoadLayerDescriptorNetCDF(fname, nc, i, ld))
                {
                    throw NC_EXCEPTION("NcException", "etwork::etwork: Error reading layer data in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                nd._vLayerDescriptor.push_back(ld);
            }

            getAttribute("weights", weights);

            for (uint32_t i = 0; i < weights; i++)
            {
                WeightDescriptor wd;
                if (!LoadWeightDescriptorNetCDF(fname, nc, i, wd))
                {
                    throw NC_EXCEPTION("NcException", "etwork::etwork: Error reading weight data in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                nd._vWeightDescriptor.push_back(wd);
            }
        }
        catch (NcException& e)
        {
            if (!bOpened)
            {
                std::cout << "Exception: etWork::etwork: Error opening NetCDF input file " << fname << std::endl;
            }
            else
            {
                std::cout << "Exception: " << e.what() << std::endl;
            }
            bResult = false;
        }
    }

    /**
     * Broadcasts the value of "bResult" from process 0 to all other processes in MPI_COMM_WORLD.
     * @param bResult The boolean value to broadcast.
     */
    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    /**
     * Checks the value of "bResult" and performs actions if it is false.
     * If "bResult" is false, it shuts down the GPU and exits the program with status -1.
     */
    if (!bResult)
    {
        getGpu().Shutdown();
        exit(-1);
    }

    /**
     * Broadcasts the NetworkDescriptor object "nd" to all processes in MPI_COMM_WORLD.
     * @param nd The NetworkDescriptor object to broadcast.
     */
    MPI_Bcast_NetworkDescriptor(nd);

    /**
     * Prints the message "LoadNeuralNetworkJSON: Enumerating network:" to the standard output stream.
     * Prints the NetworkDescriptor object "nd" to the standard output stream.
     * This is performed by the process with GPU ID 0.
     */
    if (getGpu()._id == 0)
    {
        std::cout << "LoadNeuralNetworkJSON: Enumerating network:" << std::endl;
        std::cout << nd << std::endl;
    }

    /**
     * Resets the neural network "pNetwork" with the provided NetworkDescriptor "nd" and batch size "batch".
     * @param nd The NetworkDescriptor object used to reset the neural network.
     * @param batch The batch size to use for the neural network.
     */
    pNetwork->Reset(nd, batch);

    /**
     * Refreshes the state of the neural network "pNetwork".
     */
    pNetwork->RefreshState();

    /**
     * Returns the pointer to the neural network "pNetwork".
     * @return A pointer to the neural network.
     */
    return pNetwork;
}
bool Network::P2P_Bcast(void* pBuffer, size_t size)
{
    cudaError_t status;

    if (getGpu()._numprocs > 1)
    {
        if (getGpu()._bP2P)
        {
            if (getGpu()._numprocs == 2)
            {
                if (getGpu()._id == 0)
                {
                    status = cudaMemcpy(GetPeerBackBuffer(), pBuffer, size, cudaMemcpyDefault);
                    RTERROR(status, "Network::P2P_Bcast: Failure to copy source data to P2P backbuffer");
                }
                cudaDeviceSynchronize();
                MPI_Barrier(MPI_COMM_WORLD);
            }
            else
            {
                if (getGpu()._id == 0)
                {
                    /**
                     * @brief Copies data from a source buffer to the P2P send buffer.
                     *
                     * @param pBuffer The source buffer containing the data to be copied.
                     * @param size The size of the data to be copied in bytes.
                     *
                     * @note The P2P send buffer must be preallocated and properly set up before calling this function.
                     * @see AllocateP2PBuffer()
                     */
                    status = cudaMemcpy(GetP2PSendBuffer(), pBuffer, size, cudaMemcpyDefault);

                    /**
                     * @brief Checks the status of the data copy operation and raises an error if it fails.
                     *
                     * @param status The status of the cudaMemcpy operation.
                     * @param errorMessage The error message to display if the operation fails.
                     *
                     * @note This function should be used after calling cudaMemcpy to check for errors.
                     * @see RTERROR()
                     */
                    RTERROR(status, "Network::P2P_Bcast: Failure to copy source data to P2P backbuffer");
                }

                uint32_t stages = 2 * getGpu()._numprocs - 2;
                uint32_t distance = (getGpu()._numprocs - getGpu()._id) % getGpu()._numprocs;
                uint64_t segment = 0;
                for (uint32_t i = 0; i < stages; i++)
                {
                    if ((getGpu()._id != 1) && (i >= distance) && (segment < getGpu()._numprocs))
                    {
                        /**
                         * @brief Perform P2P broadcast by copying data from the P2P send buffer to the P2P backbuffer.
                         * 
                         * @param size The size of the data to be broadcasted.
                         * @param segment The segment index of the broadcast operation.
                         */
                        size_t start = (size * segment) / getGpu()._numprocs;
                        size_t end = (size * (segment + 1)) / getGpu()._numprocs;
                        status = cudaMemcpy((char*)GetPeerBackBuffer() + start, (char*)GetP2PSendBuffer() + start, end - start, cudaMemcpyDefault);
                        RTERROR(status, "Network::P2P_Bcast: Failure to copy source data to P2P backbuffer");
                        segment++;
                    }

                    cudaDeviceSynchronize();
                    MPI_Barrier(MPI_COMM_WORLD);
                }
            }

            if (getGpu()._id > 0)
            {
                status = cudaMemcpy(pBuffer, GetP2PSendBuffer(), size, cudaMemcpyDefault);
                RTERROR(status, "Network::P2P_Bcast: Failure to copy source data from P2P sendbuffer");
            }
        }
        else
        {
            /**
             * Copies data from the source memory address "pBuffer" to the destination memory address "_pCPUBuffer.get()" using cudaMemcpy.
             * @param _pCPUBuffer.get() The destination memory address in the _pCPUBuffer.
             * @param pBuffer The source memory address.
             * @param size The size of the data to be copied in bytes.
             * @param cudaMemcpyDefault The cudaMemcpyKind specifying the type of memory copy.
             */
            cudaMemcpy(_pCPUBuffer.get(), pBuffer, size, cudaMemcpyDefault);

            /**
             * Broadcasts the data from the source memory address "_pCPUBuffer.get()" to all processes in MPI_COMM_WORLD.
             * @param _pCPUBuffer.get() The source memory address.
             * @param size The size of the data to be broadcasted.
             * @param MPI_BYTE The datatype of the elements being broadcasted.
             * @param 0 The root process that broadcasts the data.
             * @param MPI_COMM_WORLD The communicator representing all processes.
             */
            MPI_Bcast(_pCPUBuffer.get(), size, MPI_BYTE, 0, MPI_COMM_WORLD);

            /**
             * Copies data from the source memory address "_pCPUBuffer.get()" to the destination memory address "pBuffer" using cudaMemcpy.
             * @param pBuffer The destination memory address.
             * @param _pCPUBuffer.get() The source memory address in the _pCPUBuffer.
             * @param size The size of the data to be copied in bytes.
             * @param cudaMemcpyDefault The cudaMemcpyKind specifying the type of memory copy.
             */
            cudaMemcpy(pBuffer, _pCPUBuffer.get(), size, cudaMemcpyDefault);
        }
    }

    return true;
}
bool Network::P2P_Allreduce(Float* pBuffer, size_t size)
{
    if (getGpu()._numprocs > 1)
    {
        if (getGpu()._bP2P)
        {
            if (getGpu()._numprocs == 2)
            {
                /**
                 * Copies data from the source memory address "pBuffer" to the destination memory address "GetPeerBuffer()" using cudaMemcpy.
                 * @param GetPeerBuffer() The destination memory address in the PeerBuffer.
                 * @param pBuffer The source memory address.
                 * @param size The size of the data to be copied in bytes.
                 * @param cudaMemcpyDefault The cudaMemcpyKind specifying the type of memory copy.
                 */
                cudaMemcpy(GetPeerBuffer(), pBuffer, size * sizeof(Float), cudaMemcpyDefault);

                /**
                 * Synchronizes the current CUDA device.
                 */
                cudaDeviceSynchronize();

                /**
                 * Blocks until all processes in MPI_COMM_WORLD have reached this point.
                 */
                MPI_Barrier(MPI_COMM_WORLD);

                /**
                 * Adds the data from the source memory address "pBuffer" to the destination memory address "GetP2PReceiveBuffer()".
                 * @param pBuffer The source memory address.
                 * @param GetP2PReceiveBuffer() The destination memory address.
                 * @param size The size of the data to be added.
                 */
                kAddBuffers(pBuffer, GetP2PReceiveBuffer(), size);
            }
            else
            {
                /**
                 * Calculates the number of stages based on the total number of processes in getGpu()._numprocs.
                 * The number of stages is calculated as getGpu()._numprocs - 1.
                 * @param getGpu()._numprocs The total number of processes.
                 * @return The number of stages calculated.
                 */
                uint32_t stages = getGpu()._numprocs - 1;

                /**
                 * Calculates the segment index for the current process.
                 * Each process is assigned a unique segment index based on its getGpu()._id.
                 * @param getGpu()._id The ID of the current process.
                 * @return The segment index assigned to the current process.
                 */
                uint64_t segment = getGpu()._id;

                /**
                 * Calculates the start index for the current process's segment.
                 * The start index determines the starting position of the data for the current process's segment.
                 * @param size The total size of the data.
                 * @param segment The segment index assigned to the current process.
                 * @param getGpu()._numprocs The total number of processes.
                 * @return The start index for the current process's segment.
                 */
                uint64_t start = (size * segment) / getGpu()._numprocs;

                /**
                 * Calculates the end index for the current process's segment.
                 * The end index determines the ending position (exclusive) of the data for the current process's segment.
                 * @param size The total size of the data.
                 * @param segment The segment index assigned to the current process.
                 * @param getGpu()._numprocs The total number of processes.
                 * @return The end index for the current process's segment.
                 */
                uint64_t end = (size * (segment + 1)) / getGpu()._numprocs;

                for (uint32_t i = 0; i < stages; i++)
                {
                    if (i == 0)
                        cudaMemcpy(GetPeerBuffer(), pBuffer + start, (end - start) * sizeof(Float), cudaMemcpyDefault);
                    else
                        cudaMemcpy(GetPeerBuffer(), GetP2PSendBuffer(), (end - start) * sizeof(Float), cudaMemcpyDefault);

                    /**
                     * Synchronizes the current CUDA device.
                     */
                    cudaDeviceSynchronize();

                    /**
                     * Blocks until all processes in MPI_COMM_WORLD have reached this point.
                     */
                    MPI_Barrier(MPI_COMM_WORLD);

                    /**
                     * Swaps the contents of the PeerBuffer and P2PSendBuffer.
                     */
                    SwapPeerBuffers();

                    /**
                     * Updates the segment, start, and end values for the next data transfer.
                     * @param segment The current segment index.
                     * @param getGpu()._numprocs The total number of processes.
                     * @param size The total size of the data.
                     */
                    segment = (segment + 1) % getGpu()._numprocs;
                    start = (size * segment) / getGpu()._numprocs;
                    end = (size * (segment + 1)) / getGpu()._numprocs;

                    /**
                     * Adds the data from the source memory address "GetP2PSendBuffer()" to the destination memory address "pBuffer + start".
                     * @param GetP2PSendBuffer() The source memory address.
                     * @param pBuffer + start The destination memory address.
                     * @param end - start The size of the data to be added.
                     */
                    kAddBuffers(GetP2PSendBuffer(), pBuffer + start, end - start);
                }

                /**
                 * Copies data from the source memory address "GetP2PSendBuffer()" to the destination memory address "pBuffer + start" using cudaMemcpy.
                 * @param pBuffer + start The destination memory address.
                 * @param GetP2PSendBuffer() The source memory address.
                 * @param (end - start) * sizeof(Float) The size of the data to be copied in bytes.
                 * @param cudaMemcpyDefault The cudaMemcpyKind specifying the type of memory copy.
                 */
                cudaMemcpy(pBuffer + start, GetP2PSendBuffer(), (end - start) * sizeof(Float), cudaMemcpyDefault);
                /**
                 * Iterates over the stages to perform computation or data exchange.
                 * @param i The index of the current stage.
                 * @param stages The total number of stages.
                 */
                for (uint32_t i = 0; i < stages; i++)
                {
                    /**
                     * Copies data from the P2PSendBuffer to the PeerBuffer using cudaMemcpy.
                     * @param GetPeerBuffer() The destination memory address in the PeerBuffer.
                     * @param GetP2PSendBuffer() The source memory address in the P2PSendBuffer.
                     * @param (end - start) * sizeof(Float) The size of the data to be copied in bytes.
                     * @param cudaMemcpyDefault The cudaMemcpyKind specifying the type of memory copy.
                     */
                    cudaMemcpy(GetPeerBuffer(), GetP2PSendBuffer(), (end - start) * sizeof(Float), cudaMemcpyDefault);

                    /**
                     * Synchronizes the current CUDA device.
                     */
                    cudaDeviceSynchronize();

                    /**
                     * Blocks until all processes in MPI_COMM_WORLD have reached this point.
                     */
                    MPI_Barrier(MPI_COMM_WORLD);

                    /**
                     * Swaps the contents of the PeerBuffer and P2PSendBuffer.
                     */
                    SwapPeerBuffers();

                    /**
                     * Updates the segment, start, and end values for the next data transfer.
                     * @param segment The current segment index.
                     * @param getGpu()._numprocs The total number of processes.
                     * @param size The total size of the data.
                     */
                    segment = (segment + 1) % getGpu()._numprocs;
                    start = (size * segment) / getGpu()._numprocs;
                    end = (size * (segment + 1)) / getGpu()._numprocs;

                    /**
                     * Copies data from the P2PSendBuffer to the specified range in the pBuffer using cudaMemcpy.
                     * @param pBuffer The destination memory address in the pBuffer.
                     * @param GetP2PSendBuffer() The source memory address in the P2PSendBuffer.
                     * @param (end - start) * sizeof(Float) The size of the data to be copied in bytes.
                     * @param cudaMemcpyDefault The cudaMemcpyKind specifying the type of memory copy.
                     */
                    cudaMemcpy(pBuffer + start, GetP2PSendBuffer(), (end - start) * sizeof(Float), cudaMemcpyDefault);
                }
            }
        }
        else
        {
            /**
             * Copies data from the source memory address "pBuffer" to the destination memory address "_pCPUBuffer.get()" using cudaMemcpy.
             * @param _pCPUBuffer.get() The destination memory address in the _pCPUBuffer.
             * @param pBuffer The source memory address.
             * @param size The size of the data to be copied in bytes.
             * @param cudaMemcpyDefault The cudaMemcpyKind specifying the type of memory copy.
             */
            cudaMemcpy(_pCPUBuffer.get(), pBuffer, size * sizeof(Float), cudaMemcpyDefault);

            /**
             * Performs an allreduce operation on the data in _pCPUBuffer using MPI_Allreduce.
             * The result is stored in _pCPUBuffer.
             * @param MPI_IN_PLACE The input data is taken from the _pCPUBuffer itself.
             * @param _pCPUBuffer.get() The output buffer to store the result.
             * @param size The size of the data to be reduced.
             * @param MPI_FLOAT The datatype of the elements being reduced.
             * @param MPI_SUM The operation to be performed during reduction (summation).
             * @param MPI_COMM_WORLD The communicator representing all processes.
             */
            MPI_Allreduce(MPI_IN_PLACE, _pCPUBuffer.get(), size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

            /**
             * Copies data from the source memory address "_pCPUBuffer.get()" to the destination memory address "pBuffer" using cudaMemcpy.
             * @param pBuffer The destination memory address.
             * @param _pCPUBuffer.get() The source memory address in the _pCPUBuffer.
             * @param size The size of the data to be copied in bytes.
             * @param cudaMemcpyDefault The cudaMemcpyKind specifying the type of memory copy.
             */
            cudaMemcpy(pBuffer, _pCPUBuffer.get(), size * sizeof(Float), cudaMemcpyDefault);
        }
    }
    return true;
}
