#ifndef NETWORK_H
#define NETWORK_H
#ifndef __NVCC__
#include <memory>
#include <map>
#include <vector>
#include <tuple>
#include <string>
#include <iostream>

struct NetworkDescriptor;

class Network {
public:
    friend class Layer;
    friend class Weight;
    friend void GpuContext::SetNeuralNetwork(Network* pNetwork);
    enum class Kind {
        FeedForward,
        AutoEncoder,
    };

    inline static std::pair<Kind, std::string> _sKindPair[] = {
        {Kind::FeedForward, "FeedForward"},
        {Kind::AutoEncoder, "AutoEncoder"}
    };

    inline static std::map<Kind, std::string> _sKindMap = {
        {Kind::FeedForward, "FeedForward"},
        {Kind::AutoEncoder, "AutoEncoder"}
    };

private:
    friend Network* LoadNeuralNetworkJSON(const std::string& fname, const uint32_t batch, const std::vector<DataSetBase*>& vDataSet);
    friend Network* LoadNeuralNetworkNetCDF(const std::string& fname, const uint32_t batch);
    friend Network* ImportAutoEncoder(const std::string& fname, uint32_t batch);
    std::string                      _name;
    uint32_t                    _batch;
    uint32_t                    _localBatch;
    uint32_t                    _position;
    uint32_t                    _localPosition;
    bool                        _bExamplesFound;
    bool                        _bAllDataLoaded;
    uint32_t                    _examples;
    const Kind                  _kind;
    ErrorFunction               _errorFunction;
    TrainingMode                _trainingMode;
    Mode                        _mode;
    uint32_t                    _epochs;
    uint32_t                    _indices;
    uint32_t                    _batches;
    float                       _decay;

    Float                     _LRN_k;
    uint32_t                    _LRN_n;
    Float                     _LRN_alpha;
    Float                     _LRN_beta;

    Float                     _RELUSlope;
    Float                     _ELUAlpha;
    Float                     _SELULambda;

    uint32_t                    _maxout_k;

    bool                        _bSparsenessPenalty;
    Float                     _sparsenessPenalty_p;
    Float                     _sparsenessPenalty_beta;

    bool                        _bDenoising;
    Float                     _denoising_p;

    Float                     _deltaBoost_one;
    Float                     _deltaBoost_zero;

    Float                     _SMCE_oneTarget;
    Float                     _SMCE_zeroTarget;
    Float                     _SMCE_oneScale;
    Float                     _SMCE_zeroScale;

    bool                        _bShuffleIndices;
    uint32_t                    _shuffleIndices;
    uint32_t*                   _pShuffleIndex;
    std::unique_ptr<GpuBuffer<uint32_t>> _pbShuffleIndex;
    std::unique_ptr<GpuSort<uint32_t, uint32_t>> _pShuffleIndexSort;

    std::string                      _checkpoint_name;
    int32_t                     _checkpoint_interval;
    int32_t                     _checkpoint_epochs;

    std::vector<Layer*>            _vLayer;
    std::vector<Layer*>            _vInputLayer;
    std::vector<Layer*>            _vOutputLayer;
    std::vector<Weight*>           _vWeight;
    std::vector<Weight*>           _vSharedWeight;
    std::vector<DataSetBase*>      _vData;
    std::vector<Layer*>            _vFPOrder;
    std::vector<Layer*>            _vBPOrder;
    std::map<std::string, Layer*>       _mLayer;
    bool                        _bDirty;
    bool                        _bClearVelocity;

    size_t                      _scratchBufferSize;
    std::unique_ptr<GpuBuffer<Float>> _pbScratchBuffer;


    uint32_t                    _maxStride;
    uint32_t                    _sendIndex;
    uint32_t                    _receiveIndex;
    std::unique_ptr<GpuBuffer<Float>> _pbP2PBuffer[2];
    Float*                    _pPeerBuffer[2];
    std::unique_ptr<Float[]>       _pCPUBuffer;

    size_t                      _CUDWorkspaceSize;
    size_t                      _maxCUDWorkspaceSize;
    std::unique_ptr<GpuBuffer<uint8_t>> _pbCUDWorkspace;

    bool                         _verbose;


public:

    ~Network();
    void ClearDataSets();

    void LoadDataSets(std::vector<DataSetBase*>& vData);
    void Randomize();
    bool Validate();
    float Train(uint32_t epochs = 1, Float alpha = (Float)0.1, Float lambda = (Float)0.001, Float lambda1 = (Float)0.0, Float mu = (Float)0.1,  Float mu1 = 0.0);
    void PredictBatch(uint32_t layers = 0);
    void CalculateOutput(const std::string& layer, uint32_t k, GpuBuffer<Float>* pbKey, GpuBuffer<uint32_t>* pbValue);
    void SaveBatch(std::string fname);
    void DumpBatch(FILE* fp);
    void SaveLayer(const std::string& fname, const std::string& layer);
    void DumpLayer(FILE* fp, const std::string& layer);
    void SaveWeights(const std::string& fname, const std::string& inputLayer, const std::string& outputLayer);
    bool LockWeights(const std::string& inputLayer, const std::string& outputLayer);
    bool UnlockWeights(const std::string& inputLayer, const std::string& outputLayer);
    void SetBatch(uint32_t batch);
    void SetPosition(uint32_t position);
    bool SetDecay(Float decay);
    void SetTrainingMode(TrainingMode mode);
    void SetShuffleIndices(bool bShuffleIndices);
    void SetCPUValidate(bool bValidate);
    void SetClearVelocity(bool bClear) { _bClearVelocity = bClear; };
    bool SaveNetCDF(const std::string& fname);

    unsigned int GetBatch() const;
    uint32_t GetExamples() const;
    uint32_t GetPosition() const;
    Weight* GetWeight(const std::string& inputLayer, const std::string& outputLayer) const;
    uint64_t GetBufferSize(const std::string& layer) const;
    Layer* GetLayer(const std::string &layer) const;

    std::vector<const Layer*>::iterator GetLayers(Layer::Kind layerKind, std::vector<const Layer*> &layers) const;
    std::vector<std::string> GetLayers() const;
    const std::string& GetName() const;
    std::tuple<Float, uint32_t, Float, Float> GetLRN() const;
    std::tuple<Float> GetDecay() const;
    std::tuple<uint32_t> GetMaxout() const;
    std::tuple<Float, Float> GetSparsenessPenalty() const;
    std::tuple<Float> GetDenoising() const;
    std::tuple<Float, Float> GetDeltaBoost() const;
    std::tuple<Float, Float, Float, Float> GetSMCE() const;
    std::tuple<bool> GetShuffleIndices() const;
    std::tuple<std::string, int32_t> GetCheckPoint() const;
    bool GetDebugLevel() const {return _verbose;}

    Float* GetUnitBuffer(const std::string& layer);
    Float* GetDeltaBuffer(const std::string& layer);
    Float* GetWeightBuffer(const std::string& inputLayer, const std::string& outputLayer);
    Float* GetScratchBuffer(size_t size = 0);
    Float* GetP2PSendBuffer();
    Float* GetP2PReceiveBuffer();
    Float* GetP2PCPUBuffer();
    Float* GetPeerBuffer();
    Float* GetPeerBackBuffer();
    bool P2P_Bcast(void* pBuffer, size_t size);
    bool P2P_Allreduce(Float* pBuffer, size_t size);


    bool SetLRN(Float k = (Float)2.0, uint32_t n = 5, Float alpha = (Float)0.0001, Float beta = (Float)0.75);
    bool SetMaxout(uint32_t k = 2);
    bool SetSparsenessPenalty(Float p = 0.0f, Float beta = 0.0f);
    bool SetDenoising(Float p = 0.0f);
    bool SetDeltaBoost(Float one = 1.0f, Float zero = 1.0f);
    bool SetSMCE(Float oneTarget = 0.9f, Float zeroTarget = 0.1f, Float oneScale = 1.0f, Float zeroScale = 1.0f);
    bool SetCheckpoint(std::string name, int32_t interval);
    void SetDebugLevel(bool verbose) {_verbose = verbose;}

private:
    void CalculatePropagationOrder();
    bool GenerateNetworkGraph();
    void AllocatePeerBuffers();
    void DeallocatePeerBuffers();
    void SwapPeerBuffers();
    void LoadBatch();
    void PredictTrainingBatch(uint32_t layers = 0);
    void PredictValidationBatch(uint32_t layers = 0);
    void RefreshShuffleBuffers();
    void ShuffleIndices();
    std::tuple<Float, Float> CalculateError(Float lambda, Float lambda1);
    void ClearUpdates();
    void BackPropagate();
    void UpdateWeights(Float alpha, Float lambda, Float lambda1, Float mu, Float mu1);
    Network(NetworkDescriptor& nd, uint32_t batch = DefaultBatch);
    void RefreshState();
    void Shuffle();
    void SetCUDWorkspace(size_t size);
};

std::ostream& operator<< (std::ostream& out, Network::Kind& k);


struct NetworkDescriptor
{
    std::string                      _name;
    Network::Kind             _kind;
    ErrorFunction               _errorFunction;
    std::vector<LayerDescriptor>   _vLayerDescriptor;
    std::vector<WeightDescriptor>  _vWeightDescriptor;
    bool                        _bShuffleIndices;
    Float                     _decay;
    uint32_t                    _maxout_k;
    Float                     _LRN_k;
    uint32_t                    _LRN_n;
    Float                     _LRN_alpha;
    Float                     _LRN_beta;
    Float                     _RELUSlope;
    Float                     _ELUAlpha;
    Float                     _SELULambda;
    bool                        _bSparsenessPenalty;
    Float                     _sparsenessPenalty_p;
    Float                     _sparsenessPenalty_beta;
    bool                        _bDenoising;
    Float                     _denoising_p;
    Float                     _deltaBoost_one;
    Float                     _deltaBoost_zero;
    Float                     _SMCE_oneTarget;
    Float                     _SMCE_zeroTarget;
    Float                     _SMCE_oneScale;
    Float                     _SMCE_zeroScale;
    std::string                      _checkpoint_name;
    int32_t                     _checkpoint_interval;
    int32_t                     _checkpoint_epochs;
    bool                        _bConvLayersCalculated;
    NetworkDescriptor();
};

std::ostream& operator<< (std::ostream& out, NetworkDescriptor& d);
Network* LoadNeuralNetworkNetCDF(const std::string& fname, const uint32_t batch = DefaultBatch);
Network* LoadNeuralNetworkJSON(const std::string& fname, const uint32_t batch = DefaultBatch, const std::vector<DataSetBase*>& vDataSet = std::vector<DataSetBase*>());
bool SaveNeuralNetworkJSON(const Network& net, const std::string& fname);
bool SaveNeuralNetworkNetCDF(const Network& net, const std::string& jname);
Network* ImportAutoEncoder(const std::string& fname, uint32_t batch = DefaultBatch);

#endif
#endif
