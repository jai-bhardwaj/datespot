#include "GpuTypes.h"
#include "NcExcptionWrap.h"
#include "Types.h"
#include "kernels.h"
#include "Utils.h"
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

    Network::Kind _kind = Network::Kind::FeedForward;
    ErrorFunction _errorFunction = ErrorFunction::CrossEntropy;
    bool _bShuffleIndices = true;
    uint32_t _maxout_k = 2;
    Float _decay = 0.0;
    uint32_t _LRN_k = 2;
    uint32_t _LRN_n = 5;
    Float _LRN_alpha = 0.0001;
    Float _LRN_beta = 0.75;
    Float _RELUSlope = 1.0;
    Float _ELUAlpha = 1.0;
    Float _SELULambda = 1.050701;
    bool _bSparsenessPenalty = false;
    Float _sparsenessPenalty_p = 0.0;
    Float _sparsenessPenalty_beta = 0.0;
    bool _bDenoising = false;
    Float _denoising_p = 0.0;
    Float _deltaBoost_one = 1.0;
    Float _deltaBoost_zero = 1.0;
    Float _SMCE_oneTarget = 0.9;
    Float _SMCE_zeroTarget = 0.1;
    Float _SMCE_oneScale = 1.0;
    Float _SMCE_zeroScale = 1.0;
    std::string _name = "";
    std::string _checkpoint_name = "checkpoint";
    uint32_t _checkpoint_interval = 0;
    uint32_t _checkpoint_epochs = 0;
    bool _bConvLayersCalculated = false;
    std::vector<LayerDescriptor> _vLayerDescriptor;
    std::vector<WeightDescriptor> _vWeightDescriptor;

    friend std::ostream& operator<< (std::ostream& out, NetworkDescriptor& d);
};

std::ostream& operator<< (std::ostream& out, NetworkDescriptor& d)
{
    using namespace std::chrono;
    
    out << std::format("Name:                    {}\n", d._name);
    out << std::format("Kind:                    {}\n", d._kind);
    out << std::format("bShuffleIndices          {}\n", std::boolalpha(d._bShuffleIndices));
    out << std::format("Error Function:          {}\n", d._errorFunction);
    out << std::format("MaxOut_k:                {}\n", d._maxout_k);
    out << std::format("LRN_k:                   {}\n", d._LRN_k);
    out << std::format("LRN_n:                   {}\n", d._LRN_n);
    out << std::format("LRN_beta:                {}\n", d._LRN_beta);
    out << std::format("LRN_alpha:               {}\n", d._LRN_alpha);
    out << std::format("bSparsenessPenalty:      {}\n", std::boolalpha(d._bSparsenessPenalty));
    out << std::format("sparsenessPenalty_beta:  {}\n", d._sparsenessPenalty_beta);
    out << std::format("sparsenessPenalty_p:     {}\n", d._sparsenessPenalty_p);
    out << std::format("bDenoising:              {}\n", std::boolalpha(d._bDenoising));
    out << std::format("denoising_p:             {}\n", d._denoising_p);
    out << std::format("deltaBoost_one:          {}\n", d._deltaBoost_one);
    out << std::format("deltaBoost_zero:         {}\n", d._deltaBoost_zero);
    out << std::format("SMCE_oneTarget:          {}\n", d._SMCE_oneTarget);
    out << std::format("SMCE_zeroTarget:         {}\n", d._SMCE_zeroTarget);
    out << std::format("SMCE_oneScale:           {}\n", d._SMCE_oneScale);
    out << std::format("SMCE_zeroScale:          {}\n", d._SMCE_zeroScale);
    out << std::format("checkpoint_name:         {}\n", d._checkpoint_name);
    out << std::format("checkpoint_interval:     {}\n", d._checkpoint_interval);

    out << "\nLayers:\n";
    for (const auto& layer : d._vLayerDescriptor)
    {
        out << "Layer " << layer << '\n';
    }

    out << "\nWeights:\n";
    for (const auto& weight : d._vWeightDescriptor)
    {
        out << "Weight " << weight << '\n';
    }
    return out;
}
bool ValidateNetworkDescriptor(NetworkDescriptor& d)
{
    return true;
}

std::tuple<Float, uint32_t, Float, Float> Network::GetLRN() const
{
    return {_LRN_k, _LRN_n, _LRN_alpha, _LRN_beta};
}

std::tuple<Float> Network::GetDecay() const
{
    return {_decay};
}

std::tuple<uint32_t> Network::GetMaxout() const
{
    return {_maxout_k};
}

std::tuple<Float, Float> Network::GetSparsenessPenalty() const
{
    return {_sparsenessPenalty_p, _sparsenessPenalty_beta};
}

std::tuple<Float> Network::GetDenoising() const
{
    return {_denoising_p};
}

std::tuple<Float, Float> Network::GetDeltaBoost() const
{
    return {_deltaBoost_one, _deltaBoost_zero};
}

std::tuple<Float, Float, Float, Float> Network::GetSMCE() const
{
    return {_SMCE_oneTarget, _SMCE_zeroTarget, _SMCE_oneScale, _SMCE_zeroScale};
}

std::tuple<bool> Network::GetShuffleIndices() const
{
    return {_bShuffleIndices};
}

std::tuple<std::string, int32_t> Network::GetCheckPoint() const
{
    return {_checkpoint_name, _checkpoint_interval};
}

Layer* Network::GetLayer(const std::string& layer) const
{
    auto itr = _mLayer.find(layer);
    if (itr == _mLayer.end())
    {
        if (getGpu()._id == 0)
        {
            printf("Network::GetLayerDimensions: Unknown layer %s.\n", layer.c_str());
        }
        return nullptr;
    }
    return itr->second;
}
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

Float* Network::GetScratchBuffer(size_t size)
{
    if (size > _scratchBufferSize)
    {
        _pbScratchBuffer.reset(new GpuBuffer<Float>(size));
        _scratchBufferSize = size;
    }
    return _pbScratchBuffer->get();
}

void Network::SetCUDWorkspace(size_t size)
{
    if (size > _maxCUDWorkspaceSize)
    {
        _maxCUDWorkspaceSize = size;
    }
}

Float* Network::GetP2PSendBuffer()
{
    return _pbP2PBuffer[_sendIndex]->get();
}

Float* Network::GetP2PReceiveBuffer()
{
    return _pbP2PBuffer[_receiveIndex]->get();
}

Float* Network::GetP2PCPUBuffer()
{
    return _pCPUBuffer.get();
}

Float* Network::GetPeerBuffer()
{
    return _pPeerBuffer[_receiveIndex].get();
}

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
_name(d._name),
_kind(d._kind),
_mode(Prediction),
_trainingMode(SGD),
_batch(batch),
_localBatch(batch),
_position(0),
_localPosition(0),
_bShuffleIndices(d._bShuffleIndices),
_shuffleIndices(0),
_pShuffleIndex(nullptr),
_pShuffleIndexSort(),
_pbShuffleIndex(),
_bExamplesFound(false),
_bAllDataLoaded(true),
_examples(0),
_errorFunction(d._errorFunction),
_decay(d._decay),
_LRN_k(d._LRN_k),
_LRN_n(d._LRN_n),
_LRN_alpha(d._LRN_alpha),
_LRN_beta(d._LRN_beta),
_maxout_k(d._maxout_k),
_bSparsenessPenalty(d._bSparsenessPenalty),
_sparsenessPenalty_beta(d._sparsenessPenalty_beta),
_sparsenessPenalty_p(d._sparsenessPenalty_p),
_bDenoising(d._bDenoising),
_denoising_p(d._denoising_p),
_deltaBoost_one(d._deltaBoost_one),
_deltaBoost_zero(d._deltaBoost_zero),
_SMCE_oneTarget(d._SMCE_oneTarget),
_SMCE_zeroTarget(d._SMCE_zeroTarget),
_SMCE_oneScale(d._SMCE_oneScale),
_SMCE_zeroScale(d._SMCE_zeroScale),
_checkpoint_name(d._checkpoint_name),
_checkpoint_interval(d._checkpoint_interval),
_checkpoint_epochs(0),
_epochs(0),
_batches(0),
_bClearVelocity(true),
_bDirty(true),
_maxStride(0),
_scratchBufferSize(0),
_pbScratchBuffer(),
_pPeerBuffer{nullptr, nullptr},
_pbP2PBuffer(),
_pCPUBuffer(),
_sendIndex(0),
_receiveIndex(1),
_CUDWorkspaceSize(0),
_maxCUDWorkspaceSize(0),
_pbCUDWorkspace(),
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
                        std::memcpy(pDst, pSrc, pOutputLayer->_localStride * sizeof(Float));
                        pSrc += pOutputLayer->_stride;
                        pDst += pOutputLayer->_localStride;
                    }
                }
                else
                {
                    Float* pSrc = wd._vWeight.data() + pInputLayer->_minX * pOutputLayer->_stride;
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
                Float* pSrc = wd._vBias.data() + pOutputLayer->_minX;
                Float* pDst = pWeight->_vBias.data();
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
                                std::cout << "Network::Network: Transposed dimensions for shared weights between layers " <<
                                _vWeight[i]->_inputLayer._name << " and " << _vWeight[i]->_outputLayer._name << " do not match" << std::endl;
                            getGpu().Shutdown();
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