#include "GpuTypes.h"
#include "NcExcptionWrap.h"
#include "Types.h"
#include "kernels.h"
#include <cstdint>
#include <iostream>
#include <vector>
#include <cudnn.h>
#include <string>
#include <netcdf>
#include <mpi.h>

using namespace netCDF;
using namespace netCDF::exceptions;

class WeightDescriptor {
private:
    int _width; /**< The width of the object. */
    int _height; /**< The height of the object. */
    int _length; /**< The length of the object. */
    int _breadth; /**< The breadth of the object. */
    int _depth; /**< The depth of the object. */
    bool _bShared; /**< Flag indicating whether the object is shared. */
    bool _bTransposed; /**< Flag indicating whether the object is transposed. */
    bool _bLocked; /**< Flag indicating whether the object is locked. */
    Float _norm; /**< The norm of the object. */

public:
    WeightDescriptor() :
        _width(1), /**< The width of the object. Initialized to 1. */
        _height(1), /**< The height of the object. Initialized to 1. */
        _length(1), /**< The length of the object. Initialized to 1. */
        _breadth(1), /**< The breadth of the object. Initialized to 1. */
        _depth(1), /**< The depth of the object. Initialized to 1. */
        _bShared(false), /**< Flag indicating whether the object is shared. Initialized to false. */
        _bTransposed(false), /**< Flag indicating whether the object is transposed. Initialized to false. */
        _bLocked(false), /**< Flag indicating whether the object is locked. Initialized to false. */
        _norm((Float)0.0) /**< The norm of the object. Initialized to 0.0. */
    {

    }
};

/**
 * @brief Dumps information about a cudnnTensorDescriptor_t.
 *
 * This function retrieves and prints information about the given cudnnTensorDescriptor_t,
 * such as the data type, number of dimensions, dimensions, and strides.
 *
 * @param t The cudnnTensorDescriptor_t to dump.
 */
static void DumpTensor(cudnnTensorDescriptor_t t)
{
    cudnnDataType_t dataType;
    int ndims;
    std::vector<int> vDim(16);
    std::vector<int> vStride(16);

    cudnnStatus_t cudnnStatus = cudnnGetTensorNdDescriptor(t, 8, &dataType, &ndims, vDim.data(), vStride.data());
    CUDNNERROR(cudnnStatus, "cudnnGetTensorNdDescriptor error");

    std::cout << "Tensor:   " << ndims << " dimensions" << std::endl;
    std::cout << "DataType: " << dataType << std::endl;

    for (int i = 0; i < ndims; i++)
        std::cout << i << " " << vDim[i] << " " << vStride[i] << std::endl;

    std::cout << std::endl;
}

/**
 * @brief Dumps information about a cudnnFilterDescriptor_t.
 *
 * This function retrieves and prints information about the given cudnnFilterDescriptor_t,
 * such as the data type, format, number of dimensions, and dimensions.
 *
 * @param f The cudnnFilterDescriptor_t to dump.
 */
static void DumpFilter(cudnnFilterDescriptor_t f)
{
    cudnnDataType_t dataType;
    cudnnTensorFormat_t format;
    int ndims;
    std::vector<int> vDim(16);

    cudnnStatus_t cudnnStatus = cudnnGetFilterNdDescriptor(f, 5, &dataType, &format, &ndims, vDim.data());
    CUDNNERROR(cudnnStatus, "cudnnGetFilterNdDescriptor error");

    std::cout << "Filter:   " << ndims << " dimensions" << std::endl;
    std::cout << "DataType: " << dataType << std::endl;
    std::cout << "Format:   " << format << std::endl;

    for (int i = 0; i < ndims; i++)
        std::cout << i << " " << vDim[i] << " " << std::endl;

    std::cout << std::endl;
}

/**
 * @brief Dumps information about a cudnnConvolutionDescriptor_t.
 *
 * This function retrieves and prints information about the given cudnnConvolutionDescriptor_t,
 * such as the data type, mode, number of dimensions, padding, strides, and upscaling factors.
 *
 * @param c The cudnnConvolutionDescriptor_t to dump.
 */
static void DumpConvolution(cudnnConvolutionDescriptor_t c)
{
    cudnnDataType_t dataType;
    cudnnConvolutionMode_t mode;
    int ndims;
    std::vector<int> vPad(16);
    std::vector<int> vStride(16);
    std::vector<int> vUpscale(16);

    cudnnStatus_t cudnnStatus = cudnnGetConvolutionNdDescriptor(c, 5, &ndims, vPad.data(), vStride.data(), vUpscale.data(), &mode, &dataType);
    CUDNNERROR(cudnnStatus, "cudnnGetConvolutionNdDescriptor error");

    std::cout << "Convolution:   " << ndims << " dimensions" << std::endl;
    std::cout << "DataType:      " << dataType << std::endl;
    std::cout << "Mode:          " << mode << std::endl;

    for (int i = 0; i < ndims; i++)
        std::cout << i << " " << vPad[i] << " " << vStride[i] << " " << vUpscale[i] << std::endl;

    std::cout << std::endl;
}
/**
 * @brief Loads weight descriptor from a NetCDF file.
 *
 * This function loads the weight descriptor from a NetCDF file specified by `fname`.
 * It retrieves and populates the `WeightDescriptor` object `wd` with information such as
 * input layer, output layer, normalization value, shared flag, transposed flag, locked flag,
 * width, height, length, depth, breadth, bias values, and weight values (if not shared).
 *
 * @param fname The filename of the NetCDF file.
 * @param nc The netCDF::NcFile object representing the NetCDF file.
 * @param index The index of the weight to load.
 * @param wd The WeightDescriptor object to populate with the loaded information.
 * @return Returns `true` if the weight descriptor is loaded successfully, `false` otherwise.
 */
bool LoadWeightDescriptorNetCDF(const std::string& fname, netCDF::NcFile& nc, uint32_t index, WeightDescriptor& wd)
{
    bool bResult = true;

    if (getGpu()._id == 0)
    {
        std::string wstring = "weight" + std::to_string(index) + "_";
        try {
            netCDF::NcGroupAtt inputLayerAtt = nc.getAtt(wstring + "inputLayer");
            if (inputLayerAtt.isNull())
            {
                throw netCDF::exceptions::NcException("NcException", "No input layer supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            inputLayerAtt.getValues(wd._inputLayer);

            netCDF::NcGroupAtt outputLayerAtt = nc.getAtt(wstring + "outputLayer");
            if (outputLayerAtt.isNull())
            {
                throw netCDF::exceptions::NcException("NcException", "No output layer supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            outputLayerAtt.getValues(wd._outputLayer);

            netCDF::NcGroupAtt normAtt = nc.getAtt(wstring + "norm");
            if (normAtt.isNull())
            {
                wd._norm = static_cast<Float>(0.0);
            }
            else
                normAtt.getValues(&wd._norm);

            netCDF::NcGroupAtt bSharedAtt = nc.getAtt(wstring + "bShared");
            if (bSharedAtt.isNull())
            {
                throw netCDF::exceptions::NcException("NcException", "No bShared supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            uint32_t bShared;
            bSharedAtt.getValues(&bShared);
            wd._bShared = (bShared != 0);

            if (wd._bShared)
            {
                netCDF::NcGroupAtt sourceInputLayerAtt = nc.getAtt(wstring + "sourceInputLayer");
                if (sourceInputLayerAtt.isNull())
                {
                    throw netCDF::exceptions::NcException("NcException", "No sourceInputLayer for shared weights supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                sourceInputLayerAtt.getValues(wd._sourceInputLayer);
                netCDF::NcGroupAtt sourceOutputLayerAtt = nc.getAtt(wstring + "sourceOutputLayer");
                if (sourceInputLayerAtt.isNull())
                {
                    throw netCDF::exceptions::NcException("NcException", "No sourceOutputLayer for shared weights supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                sourceOutputLayerAtt.getValues(wd._sourceOutputLayer);
                netCDF::NcGroupAtt bTransposedAtt = nc.getAtt(wstring + "bTransposed");
                if (bTransposedAtt.isNull())
                {
                    throw netCDF::exceptions::NcException("NcException", "No bTransposed for shared weights supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                uint32_t bTransposed;
                bTransposedAtt.getValues(&bTransposed);
                wd._bTransposed = (bTransposed != 0);
            }

            netCDF::NcGroupAtt bLockedAtt = nc.getAtt(wstring + "bLocked");
            if (bLockedAtt.isNull())
            {
                wd._bLocked = false;
            }
            else
            {
                uint32_t bLocked;
                bLockedAtt.getValues(&bLocked);
                wd._bLocked = (bLocked != 0);
            }

            netCDF::NcGroupAtt widthAtt = nc.getAtt(wstring + "width");
            if (widthAtt.isNull())
            {
                throw netCDF::exceptions::NcException("NcException", "No weight width supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            widthAtt.getValues(&wd._width);

            netCDF::NcGroupAtt heightAtt = nc.getAtt(wstring + "height");
            if (heightAtt.isNull())
            {
                throw netCDF::exceptions::NcException("NcException", "No weight height supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            heightAtt.getValues(&wd._height);

            netCDF::NcGroupAtt lengthAtt = nc.getAtt(wstring + "length");
            if (lengthAtt.isNull())
            {
                throw netCDF::exceptions::NcException("NcException", "No weight length supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            lengthAtt.getValues(&wd._length);

            netCDF::NcGroupAtt depthAtt = nc.getAtt(wstring + "depth");
            if (depthAtt.isNull())
            {
                throw netCDF::exceptions::NcException("NcException", "No weight depth supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            depthAtt.getValues(&wd._depth);

            netCDF::NcGroupAtt breadthAtt = nc.getAtt(wstring + "breadth");
            if (breadthAtt.isNull())
            {
                throw netCDF::exceptions::NcException("NcException", "No weight breadth supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            breadthAtt.getValues(&wd._breadth);

            netCDF::NcDim biasDim = nc.getDim(wstring + "biasDim");
            netCDF::NcVar biasVar = nc.getVar(wstring + "bias");
            wd._vBias.resize(biasDim.getSize());
            biasVar.getVar(wd._vBias.data());

            if (!wd._bShared)
            {
                netCDF::NcDim weightDim = nc.getDim(wstring + "weightDim");
                netCDF::NcVar weightVar = nc.getVar(wstring + "weights");
                wd._vWeight.resize(weightDim.getSize());
                weightVar.getVar(wd._vWeight.data());
            }
#if 0
            cout("Weights %d %lu %lu\n", index, _vWeight.size(), _vBias.size());
            for (int i = 0; i < 20; i++)
                cout("%3d %16.8f %16.8f\n", i, _vWeight[i], _vBias[i]);
#endif
        }
        catch (netCDF::exceptions::NcException& e)
        {
            std::cout << "WeightDescriptor::WeightDescriptor: Exception: " << e.what() << std::endl;
            bResult = false;
        }

    }

    return bResult;
}
/**
 * @brief Broadcasts weight and bias data using MPI communication.
 *
 * This function broadcasts the weight and bias data of the WeightDescriptor `d`
 * using MPI communication. The size of the weight vector and bias vector is first
 * broadcasted, and then the actual data is broadcasted to all MPI processes.
 *
 * @param d The WeightDescriptor object containing the weight and bias data to broadcast.
 * @return Returns 0 upon successful broadcast.
 */
uint64_t weights = d._vWeight.size();
MPI_Bcast(&weights, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
d._vWeight.resize(weights);
MPI_Bcast(d._vWeight.data(), weights, MPI_FLOAT, 0, MPI_COMM_WORLD);
uint64_t biases = d._vBias.size();
MPI_Bcast(&biases, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
d._vBias.resize(biases);
MPI_Bcast(d._vBias.data(), biases, MPI_FLOAT, 0, MPI_COMM_WORLD);
return 0;
}

/**
 * @brief Overloaded output stream operator for printing WeightDescriptor.
 *
 * This overloaded operator<< allows printing the WeightDescriptor `d` to an output stream.
 * The information such as input layer, output layer, width, height, length, depth, breadth,
 * shared flag, transposed flag, locked flag, and normalization value are printed.
 *
 * @param out The output stream to write the WeightDescriptor to.
 * @param d The WeightDescriptor object to be printed.
 * @return Returns the reference to the output stream after writing the WeightDescriptor.
 */
std::ostream& operator<< (std::ostream& out, WeightDescriptor& d)
{
    if (getGpu()._id == 0)
    {
        out << "Input Layer:        " << d._inputLayer << std::endl;
        out << "Output Layer:       " << d._outputLayer << std::endl;
        out << "Width:              " << d._width << std::endl;
        out << "Height:             " << d._height << std::endl;
        out << "Length:             " << d._length << std::endl;
        out << "Depth:              " << d._depth << std::endl;
        out << "Breadth:            " << d._breadth << std::endl;
        out << "bShared:            " << std::boolalpha << d._bShared << std::endl;
        out << "bTransposed:        " << std::boolalpha << d._bTransposed << std::endl;
        if (d._bShared)
        {
            out << "sourceInputLayer:   " << d._sourceInputLayer << std::endl;
            out << "sourceOutputLayer:  " << d._sourceOutputLayer << std::endl;
        }
        out << "bLocked:            " << std::boolalpha << d._bLocked << std::endl;
        out << "norm:               " << d._norm << std::endl;
    }
    return out;
}

/**
 * @brief Weight constructor.
 *
 * This constructor initializes a Weight object with the specified input layer, output layer,
 * shared flag, transposed flag, locked flag, and normalization value. It sets up the necessary
 * descriptors and buffers based on the layer types and dimensions. For convolutional layers,
 * it creates and sets cudnn descriptors for convolution and filter. For linear layers,
 * it determines the appropriate dimensions and sizes for weight and bias buffers.
 *
 * @param inputLayer The input Layer object.
 * @param outputLayer The output Layer object.
 * @param bShared Flag indicating whether the weight is shared.
 * @param bTransposed Flag indicating whether the weight is transposed.
 * @param bLocked Flag indicating whether the weight is locked.
 * @param norm The normalization value.
 */
Weight::Weight(Layer& inputLayer, Layer& outputLayer, bool bShared, bool bTransposed, bool bLocked, Float norm) :
    _inputLayer(inputLayer),
    _outputLayer(outputLayer),
    _dimensionality(2),
    _width(1),
    _height(1),
    _length(1),
    _depth(1),
    _breadth(1),
    _sharingCount(1),
    _updateCount(0),
    _bShared(bShared),
    _bTransposed(bTransposed),
    _bLocked(bLocked),
    _norm(norm),
    _pSharedWeight(nullptr),
    _pbWeight(),
    _pbBias(),
    _pbWeightGradient(),
    _pbBiasGradient(),
    _pbWeightVelocity(),
    _pbBiasVelocity(),
    _pbWeightGradientVelocity(),
    _pbBiasGradientVelocity()
{
    inputLayer._vOutgoingLayer.push_back(&outputLayer);
    outputLayer._vIncomingLayer.push_back(&inputLayer);
    inputLayer._vOutgoingWeight.push_back(this);
    outputLayer._vIncomingWeight.push_back(this);

    if (_outputLayer._type == Layer::Type::Convolutional)
    {
        _transform = Convolution;

        cudnnStatus_t cudnnStatus = cudnnCreateTensorDescriptor(&_convBiasTensor);
        CUDNNERROR(cudnnStatus, "Weight::Weight: Unable to create tensor descriptor");
        cudnnStatus = cudnnCreateFilterDescriptor(&_convFilterDesc);
        CUDNNERROR(cudnnStatus, "Weight::Weight: Unable to create filter descriptor");
        cudnnStatus = cudnnCreateConvolutionDescriptor(&_convDesc);
        CUDNNERROR(cudnnStatus, "Weight::Weight: Unable to create convolution descriptor");

        std::vector<int> vFilterDim(5, 1);
        switch (_outputLayer._dimensions)
        {
            case 2:
                vFilterDim[0] = _outputLayer._Ny;
                vFilterDim[1] = _inputLayer._Ny;
                vFilterDim[2] = _inputLayer._kernelX;
                _dimensionality = 3;
                break;

            case 3:
                vFilterDim[0] = _outputLayer._Nz;
                vFilterDim[1] = _inputLayer._Nz;
                vFilterDim[2] = _outputLayer._kernelY;
                vFilterDim[3] = _outputLayer._kernelX;
                _dimensionality = 4;
                break;

            case 4:
                vFilterDim[0] = _outputLayer._Nw;
                vFilterDim[1] = _inputLayer._Nw;
                vFilterDim[2] = _outputLayer._kernelZ;
                vFilterDim[3] = _outputLayer._kernelY;
                vFilterDim[4] = _outputLayer._kernelX;
                _dimensionality = 5;
                break;
        }
        cudnnStatus = cudnnSetFilterNdDescriptor(_convFilterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, _outputLayer._dimensions + 1, vFilterDim.data());
        CUDNNERROR(cudnnStatus, "Weight::Weight: Unable to set filter descriptor");

        _width = vFilterDim[0];
        _height = vFilterDim[1];
        _length = vFilterDim[2];
        _depth = vFilterDim[3];
        _breadth = vFilterDim[4];

        std::vector<int> vConvPad(3, 0);
        std::vector<int> vConvStride(3, 1);
        std::vector<int> vConvUpscale(3, 1);
        switch (_outputLayer._dimensions)
        {
            case 2:
                vConvPad[0] = _outputLayer._kernelPaddingX;
                vConvStride[0] = _outputLayer._kernelStrideX;
                break;

            case 3:
                vConvPad[0] = _outputLayer._kernelPaddingY;
                vConvStride[0] = _outputLayer._kernelStrideY;
                vConvPad[1] = _outputLayer._kernelPaddingX;
                vConvStride[1] = _outputLayer._kernelStrideX;
                break;

            case 4:
                vConvPad[0] = _outputLayer._kernelPaddingZ;
                vConvStride[0] = _outputLayer._kernelStrideZ;
                vConvPad[1] = _outputLayer._kernelPaddingY;
                vConvStride[1] = _outputLayer._kernelStrideY;
                vConvPad[2] = _outputLayer._kernelPaddingX;
                vConvStride[2] = _outputLayer._kernelStrideX;
                break;
        }
        cudnnStatus = cudnnSetConvolutionNdDescriptor(_convDesc, _outputLayer._kernelDimensions, vConvPad.data(), vConvStride.data(), vConvUpscale.data(), CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
        CUDNNERROR(cudnnStatus, "Weight::Weight: cudnnSetConvolutionNdDescriptor failed.");

        std::vector<int> vBiasDim(5, 1);
        std::vector<int> vBiasStride(5, 1);
        vBiasDim[1] = vFilterDim[0];
        cudnnStatus = cudnnSetTensorNdDescriptor(_convBiasTensor, CUDNN_DATA_FLOAT, outputLayer._dimensions + 1, vBiasDim.data(), vBiasStride.data());
        CUDNNERROR(cudnnStatus, "Weight::Weight: Unable to set bias tensor descriptor");

        _size = vFilterDim[0] * vFilterDim[1] * _outputLayer._kernelX * _outputLayer._kernelY * _outputLayer._kernelZ;
        _biasSize = vFilterDim[0];
        _localSize = _size;
        _localBiasSize = _biasSize;

        if (getGpu()._id == 0)
        {
            std::cout("Weight::Weight: Allocating %" PRIu64 " bytes (%d x %d x %u", _localSize * sizeof(Float), vFilterDim[0], vFilterDim[1], _outputLayer._kernelX);
            if (_outputLayer._dimensions >= 3)
                std::cout(" x %u", _outputLayer._kernelY);
            if (_outputLayer._dimensions >= 4)
                std::cout(" x %u", _outputLayer._kernelZ);
            std::cout(") for convolutional weights between layers %s and %s\n", inputLayer._name.c_str(), outputLayer._name.c_str());
        }
    }
    else
    {
        _transform = Linear;

        uint32_t outgoingSize = outputLayer._stride * 3;
        uint32_t incomingSize = inputLayer._stride * 2;

        if (outgoingSize > incomingSize)
        {
            inputLayer._vOutgoingLargerLayer.push_back(&outputLayer);
            inputLayer._vOutgoingLargerWeight.push_back(this);
            _width = outputLayer._localStride;
            _height = inputLayer._stride;
        }
        else
        {
            outputLayer._vIncomingLargerLayer.push_back(&inputLayer);
            outputLayer._vIncomingLargerWeight.push_back(this);
            _width = outputLayer._stride;
            _height = inputLayer._localStride;
        }
        _localSize = _width * _height * _length * _depth * _breadth;
        _localBiasSize = outputLayer._localStride;
        _size = outputLayer._stride * inputLayer._stride * _length * _depth * _breadth;
        _biasSize = outputLayer._stride;

        if (getGpu()._id == 0)
            std::cout("Weight::Weight: Allocating %" PRIu64 " bytes (%" PRIu64 ", %" PRIu64 ") for fully connected weights between layers %s and %s\n", _localSize * sizeof(float), _width, _height, inputLayer._name.c_str(), outputLayer._name.c_str());
    }

    if (!_bShared)
    {
        _vWeight.resize(_localSize);
        _pbWeight.reset(new GpuBuffer<Float>(_localSize));
        _pbWeightGradient.reset(new GpuBuffer<Float>(_localSize));
    }

    _vBias.resize(_localBiasSize);
    _pbBias.reset(new GpuBuffer<Float>(_localBiasSize));

    if (_transform == Convolution)
    {
        _pbBiasGradient.reset(new GpuBuffer<Float>(_localBiasSize));
    }
}
/**
 * @brief Weight destructor.
 *
 * This destructor is responsible for cleaning up the resources used by the Weight object.
 * Currently, it does not perform any specific cleanup operations.
 */
Weight::~Weight()
{
}
/**
 * @brief Clear the velocity buffers.
 *
 * This function clears the velocity buffers used for weight and bias updates.
 * It sets all the elements in the buffers to zero.
 */
void Weight::ClearVelocity()
{
    std::memset(_pbWeightVelocity->_pDevData, 0, _localSize * sizeof(Float));
    std::memset(_pbBiasVelocity->_pDevData, 0, _localBiasSize * sizeof(Float));
    if (_pbWeightGradientVelocity)
        std::memset(_pbWeightGradientVelocity->_pDevData, 0, _localSize * sizeof(Float));
    if (_pbBiasGradientVelocity)
        std::memset(_pbBiasGradientVelocity->_pDevData, 0, _localBiasSize * sizeof(Float));
}
/**
 * @brief Clear the gradient buffer.
 *
 * This function clears the gradient buffer used for weight updates.
 * It sets all the elements in the buffer to zero.
 */
void Weight::ClearGradient()
{
    std::memset(_pbWeightGradient->_pDevData, 0, _localSize * sizeof(Float));
}
/**
 * @brief Randomize the weight values.
 *
 * This function randomizes the weight values based on the specified weight initialization method.
 * If the weight is not shared, it generates random values using a random number generator and applies
 * the appropriate scaling and biasing based on the selected weight initialization method.
 * It also sets the bias values to zero and applies the bias initialization value.
 */
void Weight::Randomize()
{
    if (!_bShared)
    {
        Float scale, bias;
        switch (_outputLayer._weightInit)
        {
        case CaffeXavier:
            curandGenerateUniform(getGpu()._RNG, _pbWeight->_pDevData, _localSize);
            scale = _outputLayer._weightInitScale * 2.0f * std::sqrt(3.0f / _outputLayer._stride);
            bias = 0.5f * scale;
            kScaleAndBias(_pbWeight->_pDevData, _localSize, scale, bias);
            break;

        case Xavier:
            curandGenerateUniform(getGpu()._RNG, _pbWeight->_pDevData, _localSize);
            scale = _outputLayer._weightInitScale * std::sqrt(6.0f / (_outputLayer._stride + _inputLayer._stride));
            bias = 0.5f * scale;
            kScaleAndBias(_pbWeight->_pDevData, _localSize, scale, bias);
            break;

        case Uniform:
            curandGenerateUniform(getGpu()._RNG, _pbWeight->_pDevData, _localSize);
            scale = 2.0f * _outputLayer._weightInitScale;
            bias = 0.5f * scale;
            kScaleAndBias(_pbWeight->_pDevData, _localSize, scale, bias);
            break;

        case Gaussian:
            curandGenerateNormal(getGpu()._RNG, _pbWeight->_pDevData, _localSize, 0.0f, _outputLayer._weightInitScale);
            break;

        case UnitBall:
            curandGenerateUniform(getGpu()._RNG, _pbWeight->_pDevData, _localSize);
            scale = _outputLayer._weightInitScale;
            kScaleAndBias(_pbWeight->_pDevData, _localSize, scale, 0.0f);
            break;

        case SELU:
            curandGenerateNormal(getGpu()._RNG, _pbWeight->_pDevData, _localSize, 0.0f, 1.0f / _inputLayer._stride);
            break;

        case Constant:
            std::memset(_pbWeight->_pDevData, 0, _localSize * sizeof(Float));
            kScaleAndBias(_pbWeight->_pDevData, _localSize, (Float)0.0, _outputLayer._weightInitScale);
            break;
        };
    }

    std::memset(_pbBias->_pDevData, 0, _localBiasSize * sizeof(Float));
    kScaleAndBias(_pbBias->_pDevData, _localBiasSize, (Float)0.0, -_outputLayer._biasInit);
}
/**
 * @brief Lock the weight.
 *
 * This function locks the weight by setting the _bLocked member variable to true.
 * When a weight is locked, its value is not updated during training.
 */
void Weight::Lock()
{
    _bLocked = true;
}
/**
 * @brief Unlock the weight.
 *
 * This function unlocks the weight by setting the _bLocked member variable to false.
 * When a weight is unlocked, its value can be updated during training.
 */
void Weight::Unlock()
{
    _bLocked = false;
}
/**
 * @brief Refresh the state of the weight.
 *
 * This function refreshes the state of the weight based on the specified network and training mode.
 * It initializes or resets the velocity buffers and gradient buffers based on the training mode.
 * If the weight is convolutional, it retrieves and sets the convolutional algorithms and workspace size.
 *
 * @param pNetwork Pointer to the network containing the weight.
 * @param mode Training mode.
 */
void Weight::RefreshState(Network* pNetwork, TrainingMode mode)
{
    if (mode != TrainingMode::SGD)
    {
        if (!_pbWeightVelocity)
            _pbWeightVelocity.reset(new GpuBuffer<Float>(_localSize));
        if (!_pbBiasVelocity)
            _pbBiasVelocity.reset(new GpuBuffer<Float>(_localBiasSize));

        if ((mode == TrainingMode::AdaDelta) || (mode == TrainingMode::Adam))
        {
            if (!_pbWeightGradientVelocity)
                _pbWeightGradientVelocity.reset(new GpuBuffer<Float>(_localSize));
            if (!_pbBiasGradientVelocity)
                _pbBiasGradientVelocity.reset(new GpuBuffer<Float>(_localBiasSize));
        }
        else
        {
            _pbWeightGradientVelocity.reset();
            _pbBiasGradientVelocity.reset();
        }
    }
    else
    {
        _pbWeightVelocity.reset();
        _pbBiasVelocity.reset();
        _pbWeightGradientVelocity.reset();
        _pbBiasGradientVelocity.reset();
    }

    if (_outputLayer._type == Layer::Type::Convolutional)
    {
        std::cout << "Getting algorithm between " << _inputLayer._name << " and " << _outputLayer._name << '\n';
        size_t workspaceSize;
        cudnnStatus_t cudnnStatus = cudnnGetConvolutionForwardAlgorithm(
            getGpu()._cuDNNHandle,
            _inputLayer._tensorDescriptor,
            _convFilterDesc,
            _convDesc,
            _outputLayer._tensorDescriptor,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            1,
            &_convFWAlgo
        );
        CUDNNERROR(cudnnStatus, "Weight::Refresh: cudnnGetConvolutionForwardAlgorithm failed.");

        cudnnStatus = cudnnGetConvolutionForwardWorkspaceSize(
            getGpu()._cuDNNHandle,
            _inputLayer._tensorDescriptor,
            _convFilterDesc,
            _convDesc,
            _outputLayer._tensorDescriptor,
            _convFWAlgo,
            &workspaceSize
        );
        CUDNNERROR(cudnnStatus, "Weight::Refresh: cudnnGetConvolutionForwardWorkspaceSize failed.");
        pNetwork->SetCUDNNWorkspace(workspaceSize);

        cudnnStatus = cudnnGetConvolutionBackwardFilterAlgorithm(
            getGpu()._cuDNNHandle,
            _inputLayer._tensorDescriptor,
            _outputLayer._tensorDescriptor,
            _convDesc,
            _convFilterDesc,
            CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
            0,
            &_convBWWeightAlgo
        );
        CUDNNERROR(cudnnStatus, "Weight::Refresh: cudnnGetConvolutionBackwardFilterAlgorithm failed.");

        cudnnStatus = cudnnGetConvolutionBackwardFilterWorkspaceSize(
            getGpu()._cuDNNHandle,
            _inputLayer._tensorDescriptor,
            _outputLayer._tensorDescriptor,
            _convDesc,
            _convFilterDesc,
            _convBWWeightAlgo,
            &workspaceSize
        );
        CUDNNERROR(cudnnStatus, "Weight::Refresh: cudnnGetConvolutionBackwardFilterWorkspaceSize failed.");
        pNetwork->SetCUDNNWorkspace(workspaceSize);

        cudnnStatus = cudnnGetConvolutionBackwardDataAlgorithm(
            getGpu()._cuDNNHandle,
            _convFilterDesc,
            _outputLayer._tensorDescriptor,
            _convDesc,
            _inputLayer._tensorDescriptor,
            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
            0,
            &_convBWDeltaAlgo
        );
        CUDNNERROR(cudnnStatus, "Weight::Refresh: cudnnGetConvolutionBackwardDataAlgorithm failed.");

        cudnnStatus = cudnnGetConvolutionBackwardDataWorkspaceSize(
            getGpu()._cuDNNHandle,
            _convFilterDesc,
            _outputLayer._tensorDescriptor,
            _convDesc,
            _inputLayer._tensorDescriptor,
            _convBWDeltaAlgo,
            &workspaceSize
        );
        CUDNNERROR(cudnnStatus, "Weight::Refresh: cudnnGetConvolutionBackwardDataWorkspaceSize failed.");
        pNetwork->SetCUDNNWorkspace(workspaceSize);

        std::vector<int> vOutputDim(8, 1);
        cudnnStatus = cudnnGetConvolutionNdForwardOutputDim(
            _convDesc,
            _inputLayer._tensorDescriptor,
            _convFilterDesc,
            _outputLayer._dimensions + 1,
            vOutputDim.data()
        );
        CUDNNERROR(cudnnStatus, "Weight::Refresh: cudnnGetConvolutionNdForwardOutputDim failed.");
        size_t dim = 1;
        for (size_t i = 0; i < _outputLayer._dimensions + 1; i++)
            dim *= vOutputDim[i];
        if (dim != _outputLayer._maxLocalStride * _outputLayer._localBatch)
        {
            if (getGpu()._id == 0)
                std::cout << "Output layer " << _outputLayer._name << " has incorrectly calculated dimensions for cuDNN.\n";
            getGpu().Shutdown();
        }
    }
}
/**
 * @brief Calculate the regularization error of the weight.
 *
 * This function calculates the regularization error of the weight using the specified lambda and lambda1 values.
 * If the weight is shared, the regularization error is 0.
 *
 * @param lambda Regularization parameter lambda.
 * @param lambda1 Regularization parameter lambda1.
 * @return Regularization error of the weight.
 */
Float Weight::CalculateRegularizationError(Float lambda, Float lambda1)
{
    if (_bShared)
        return 0;
    else
        return kCalculateRegularizationError(lambda, lambda1, _pbWeight->_pDevData, _localSize);
}
/**
 * @brief Update the weights of the weight.
 *
 * This function updates the weights of the weight based on the specified training mode, batch index, and learning parameters.
 * It performs different update operations based on the training mode and weight type (linear or convolutional).
 *
 * @param trainingMode Training mode.
 * @param batch Batch index.
 * @param alpha Learning rate.
 * @param lambda Regularization parameter lambda.
 * @param lambda1 Regularization parameter lambda1.
 * @param mu Momentum parameter mu.
 * @param mu1 Adam parameter mu1.
 * @param t Adam parameter t.
 */
void Weight::UpdateWeights(TrainingMode trainingMode, uint32_t batch, Float alpha, Float lambda, Float lambda1, Float mu, Float mu1, Float t)
{
    cublasStatus_t cstatus;

    if (_bLocked)
        return;

    if (!_bShared)
    {
        switch (trainingMode)
        {
            case TrainingMode::SGD:
                kSGDUpdateWeights(alpha, lambda, lambda1, _localSize, _pbWeightGradient->_pDevData, _pbWeight->_pDevData);
                break;

            case TrainingMode::Momentum:
                kMomentumUpdateWeights(alpha, lambda, lambda1, mu, _localSize, _pbWeightVelocity->_pDevData, _pbWeightGradient->_pDevData, _pbWeight->_pDevData);
                break;

            case TrainingMode::AdaGrad:
                kAdaGradUpdateWeights(alpha, lambda, lambda1, _localSize, _pbWeightVelocity->_pDevData, _pbWeightGradient->_pDevData, _pbWeight->_pDevData);
                break;

            case TrainingMode::Nesterov:
                kNesterovUpdateWeights(alpha, lambda, lambda1, mu, _localSize, _pbWeightVelocity->_pDevData, _pbWeightGradient->_pDevData, _pbWeight->_pDevData);
                break;

            case TrainingMode::RMSProp:
                kRMSPropUpdateWeights(alpha, lambda, lambda1, mu, _localSize, _pbWeightVelocity->_pDevData, _pbWeightGradient->_pDevData, _pbWeight->_pDevData);
                break;

            case TrainingMode::AdaDelta:
                kAdaDeltaUpdateWeights(lambda, lambda1, mu, _localSize, _pbWeightVelocity->_pDevData, _pbWeightGradient->_pDevData, _pbWeightGradientVelocity->_pDevData, _pbWeight->_pDevData);
                break;

            case TrainingMode::Adam:
                kAdamUpdateWeights(alpha, lambda, lambda1, mu, mu1, t, _localSize, _pbWeightVelocity->_pDevData, _pbWeightGradient->_pDevData, _pbWeightGradientVelocity->_pDevData, _pbWeight->_pDevData);
                break;
        }
    }

    if (_transform == Linear)
    {
        switch (trainingMode)
        {
            case TrainingMode::SGD:
                kSGDUpdateBiases(alpha, batch, _localBiasSize, _outputLayer._pbDelta->_pDevData, _pbBias->_pDevData);
                break;

            case TrainingMode::Momentum:
                kMomentumUpdateBiases(alpha, mu, batch, _localBiasSize, _outputLayer._pbDelta->_pDevData, _pbBiasVelocity->_pDevData, _pbBias->_pDevData);
                break;

            case TrainingMode::AdaGrad:
                kAdaGradUpdateBiases(alpha, batch, _localBiasSize, _outputLayer._pbDelta->_pDevData, _pbBiasVelocity->_pDevData, _pbBias->_pDevData);
                break;

            case TrainingMode::Nesterov:
                kNesterovUpdateBiases(alpha, mu, batch, _localBiasSize, _outputLayer._pbDelta->_pDevData, _pbBiasVelocity->_pDevData, _pbBias->_pDevData);
                break;

            case TrainingMode::RMSProp:
                kRMSPropUpdateBiases(alpha, mu, batch, _localBiasSize, _outputLayer._pbDelta->_pDevData, _pbBiasVelocity->_pDevData, _pbBias->_pDevData);
                break;

            case TrainingMode::AdaDelta:
                kAdaDeltaUpdateBiases(mu, batch, _localBiasSize, _outputLayer._pbDelta->_pDevData, _pbBiasVelocity->_pDevData, _pbBiasGradientVelocity->_pDevData, _pbBias->_pDevData);
                break;

            case TrainingMode::Adam:
                kAdamUpdateBiases(alpha, mu, mu1, t, batch, _localBiasSize, _outputLayer._pbDelta->_pDevData, _pbBiasVelocity->_pDevData, _pbBiasGradientVelocity->_pDevData, _pbBias->_pDevData);
                break;
        }
    }
    else
    {
        switch (trainingMode)
        {
            case TrainingMode::SGD:
                kSGDUpdateWeights(alpha, (Float)0.0, (Float)0.0, _localBiasSize, _pbBiasGradient->_pDevData, _pbBias->_pDevData);
                break;

            case TrainingMode::Momentum:
                kMomentumUpdateWeights(alpha, (Float)0.0, (Float)0.0, mu, _localBiasSize, _pbBiasVelocity->_pDevData, _pbBiasGradient->_pDevData, _pbBias->_pDevData);
                break;

            case TrainingMode::AdaGrad:
                kAdaGradUpdateWeights(alpha, (Float)0.0, (Float)0.0, _localBiasSize, _pbBiasVelocity->_pDevData, _pbBiasGradient->_pDevData, _pbBias->_pDevData);
                break;

            case TrainingMode::Nesterov:
                kNesterovUpdateWeights(alpha, (Float)0.0, (Float)0.0, mu, _localBiasSize, _pbBiasVelocity->_pDevData, _pbBiasGradient->_pDevData, _pbBias->_pDevData);
                break;

            case TrainingMode::RMSProp:
                kRMSPropUpdateWeights(alpha, (Float)0.0, (Float)0.0, mu, _localBiasSize, _pbBiasVelocity->_pDevData, _pbBiasGradient->_pDevData, _pbBias->_pDevData);
                break;

            case TrainingMode::AdaDelta:
                kAdaDeltaUpdateWeights((Float)0.0, (Float)0.0, mu, _localBiasSize, _pbBiasVelocity->_pDevData, _pbBiasGradient->_pDevData, _pbBiasGradientVelocity->_pDevData, _pbBias->_pDevData);
                break;

            case TrainingMode::Adam:
                kAdamUpdateWeights(alpha, (Float)0.0, (Float)0.0, mu, mu1, t, _localBiasSize, _pbBiasVelocity->_pDevData, _pbBiasGradient->_pDevData, _pbBiasGradientVelocity->_pDevData, _pbBias->_pDevData);
                break;
        }
    }

#if 0
    if (_width < 1024)
    {
        _pbBias->Download(&_vBias[0]);
        for (int i = 0; i < _width; i++)
            cout("%3d %16.8f\n", i, _vBias[i]);
    }
#endif

    if ((_norm > (Float)0.0) && (!_bShared))
    {
        if (getGpu()._numprocs == 1)
            kNormalizeWeights(_norm, _outputLayer._stride, _inputLayer._localStride, _pbWeight->_pDevData);
        else
        {
            Float* pMagnitude = getGpu()._pNetwork->GetScratchBuffer(_outputLayer._stride);
            kCalculateWeightMagnitudes(_outputLayer._stride, _inputLayer._localStride, _pbWeight->_pDevData, pMagnitude);
            getGpu()._pNetwork->P2P_Allreduce(pMagnitude, _outputLayer._stride);
            kNormalizeWeightMagnitudes(_norm, _outputLayer._stride, _inputLayer._localStride, _pbWeight->_pDevData, pMagnitude);
        }
    }
}
/**
 * @brief Write the weight to a NetCDF file.
 *
 * This function writes the weight information to a NetCDF file, including its attributes, dimensions, and values.
 * It also writes the bias information if available. The weight data is written as a variable named "weights" and the bias data is written as a variable named "bias".
 *
 * @param nc NetCDF file object.
 * @param index Index of the weight.
 * @param pWeight Pointer to the weight data. If nullptr, the weight data from the weight object will be used.
 * @param pBias Pointer to the bias data. If nullptr, the bias data from the weight object will be used.
 * @return True if writing to the NetCDF file is successful, false otherwise.
 */
bool Weight::WriteNetCDF(netCDF::NcFile& nc, uint32_t index, Float* pWeight, Float* pBias)
{
    bool bResult = true;
    if (getGpu()._id == 0)
    {
        std::string wstring = "weight" + std::to_string(index) + "_";
        nc.putAtt(wstring + "inputLayer", _inputLayer._name);
        nc.putAtt(wstring + "outputLayer", _outputLayer._name);

        nc.putAtt(wstring + "width", netCDF::ncUint64, static_cast<unsigned long long>(_width));
        nc.putAtt(wstring + "height", netCDF::ncUint64, static_cast<unsigned long long>(_height));
        nc.putAtt(wstring + "length", netCDF::ncUint64, static_cast<unsigned long long>(_length));
        nc.putAtt(wstring + "depth", netCDF::ncUint64, static_cast<unsigned long long>(_depth));
        nc.putAtt(wstring + "breadth", netCDF::ncUint64, static_cast<unsigned long long>(_breadth));

        nc.putAtt(wstring + "bShared", netCDF::ncUint, static_cast<uint32_t>(_bShared));
        nc.putAtt(wstring + "bLocked", netCDF::ncUint, static_cast<uint32_t>(_bLocked));
        nc.putAtt(wstring + "norm", netCDF::ncFloat, _norm);

        netCDF::NcDim biasDim = nc.addDim(wstring + "biasDim", _biasSize);
        netCDF::NcVar biasVar = nc.addVar(wstring + "bias", "float", biasDim.getName());
        if (pBias == nullptr)
            pBias = _vBias.data();
        biasVar.putVar(pBias);
        if (_bShared)
        {
            nc.putAtt(wstring + "bTransposed", netCDF::ncUint, static_cast<uint32_t>(_bTransposed));
            nc.putAtt(wstring + "sourceInputLayer", _pSharedWeight->_inputLayer._name);
            nc.putAtt(wstring + "sourceOutputLayer", _pSharedWeight->_outputLayer._name);
        }
        else
        {

#if 0
        cout("Weights %d %lu %lu\n", index, _vWeight.size(), _vBias.size());
        for (int i = 0; i < 20; i++)
            cout("%3d %16.8f %16.8f\n", i, _vWeight[i], _vBias[i]);
#endif
            netCDF::NcDim weightDim = nc.addDim(wstring + "weightDim", _size);
            netCDF::NcVar weightVar = nc.addVar(wstring + "weights", "float", weightDim.getName());
            if (!pWeight)
                pWeight = _vWeight.data();
            weightVar.putVar(pWeight);
        }
    }

    return bResult;
}
/**
 * @brief Copy weights from another weight object.
 *
 * This function copies the weights and biases from another weight object to this weight object.
 * If the weight object is shared, the weights are copied to the shared weight object.
 *
 * @param pSrcWeight Pointer to the source weight object.
 * @return True if the weights are successfully copied, false otherwise.
 */
bool Weight::CopyWeights(const Weight* pSrcWeight)
{
    bool bValid = true;
    Weight* pDstWeight = _bShared ? _pSharedWeight : this;

    if (!pSrcWeight)
    {
        if (getGpu()._id == 0)
            std::cout << "Weight::CopyWeights: Invalid weight pointer.\n";
        return false;
    }

    pSrcWeight = _bShared ? pSrcWeight->_pSharedWeight : pSrcWeight;
    if ((pSrcWeight->_width != pDstWeight->_width) || (pSrcWeight->_height != pDstWeight->_height) || (pSrcWeight->_length != pDstWeight->_length))
    {
        if (getGpu()._id == 0)
        {
            std::cout << "Weight::CopyWeights: Mismatched weight dimensions (" << pDstWeight->_width << " x " << pDstWeight->_height << " x " << pDstWeight->_length << ") versus (" << pSrcWeight->_width << " x " << pSrcWeight->_height << " x " << pSrcWeight->_length << ").\n";
        }
        bValid = false;
    }
    else
    {
        pDstWeight->_vWeight = pSrcWeight->_vWeight;
        _vBias = pSrcWeight->_vBias;
        if (pDstWeight->_pbWeight != nullptr)
            pDstWeight->_pbWeight->Upload(pDstWeight->_vWeight.data());
        if (_pbBias != nullptr)
            _pbBias->Upload(_vBias.data());
    }
    return bValid;
}
/**
 * @brief Set the weights using a vector of values.
 *
 * This function sets the weights of the weight object using the provided vector of values.
 * If the weight object is shared, the weights are set in the shared weight object.
 *
 * @param vWeight The vector of weight values.
 * @return True if the weights are successfully set, false otherwise.
 */
bool Weight::SetWeights(const std::vector<Float>& vWeight)
{
    bool bValid = true;
    Weight* pWeight = _bShared ? _pSharedWeight : this;

    if (getGpu()._numprocs == 1)
    {
        if (vWeight.size() < pWeight->_vWeight.size())
        {
            if (getGpu()._id == 0)
            {
                std::cout << "Weight::SetWeights: Input vector smaller than weight vector.\n";
            }
            bValid = false;
        }
        else
        {
            if (vWeight.size() > pWeight->_vWeight.size())
                std::copy(vWeight.data(), vWeight.data() + pWeight->_vWeight.size(), pWeight->_vWeight.data());
            else
                pWeight->_vWeight = vWeight;
            if (pWeight->_pbWeight != nullptr)
                pWeight->_pbWeight->Upload(_vWeight.data());
        }
    }
    else
    {

    }
    return bValid;
}
/**
 * @brief Set the biases using a vector of values.
 *
 * This function sets the biases of the weight object using the provided vector of values.
 *
 * @param vBias The vector of bias values.
 * @return True if the biases are successfully set, false otherwise.
 */
bool Weight::SetBiases(const std::vector<Float>& vBias)
{
    bool bValid = true;

    if (vBias.size() < _vBias.size())
    {
        if (getGpu()._id == 0)
        {
            std::cout << "Weight::SetBiases: Input vector smaller than bias vector.\n";
        }
        bValid = false;
    }
    else
    {
        if (vBias.size() > _vBias.size())
            std::copy(vBias.data(), vBias.data() + _vBias.size(), _vBias.data());
        else
            _vBias = vBias;
        if (_pbBias != nullptr)
            _pbBias->Upload(_vBias.data());
    }
    return bValid;
}
/**
 * @brief Get the weights as a vector of values.
 *
 * This function retrieves the weights of the weight object and stores them in the provided vector.
 *
 * @param vWeight The vector to store the weight values.
 * @return True if the weights are successfully retrieved, false otherwise.
 */
bool Weight::GetWeights(std::vector<Float>& vWeight)
{
    bool bValid = true;

    if (vWeight.size() < _vWeight.size())
    {
        vWeight.resize(_vWeight.size());
    }

    if (_pbWeight != nullptr)
    {
        _pbWeight->Download(vWeight.data());
    }
    else
    {
        vWeight = _vWeight;
    }
    return bValid;
}
/**
 * @brief Get the biases as a vector of values.
 *
 * This function retrieves the biases of the weight object and stores them in the provided vector.
 *
 * @param vBias The vector to store the bias values.
 * @return True if the biases are successfully retrieved, false otherwise.
 */
bool Weight::GetBiases(std::vector<Float>& vBias)
{
    bool bValid = true;

    if (getGpu()._numprocs == 1)
    {

        if (vBias.size() < _vBias.size())
        {
            vBias.resize(_vBias.size());
        }

        if (_pbBias != nullptr)
        {
            _pbBias->Download(vBias.data());
        }
        else
        {
            vBias = _vBias;
        }
    }
    else
    {

    }
    return bValid;
}
/**
 * @brief Get the dimensions of the weight.
 *
 * This function retrieves the dimensions of the weight object and stores them in the provided vector.
 *
 * @param dimensions The vector to store the dimensions of the weight.
 * @return True if the dimensions are successfully retrieved, false otherwise.
 */
bool Weight::GetDimensions(std::vector<uint64_t>& dimensions)
{
    if (_dimensionality < 2 || _dimensionality > 5) {
        std::cout << "Weight::GetDimensions: _dimensionality = " << _dimensionality << '\n';
        return false;
    }
    if (_dimensionality >= 1) dimensions.push_back(_width);
    if (_dimensionality >= 2) dimensions.push_back(_height);
    if (_dimensionality >= 3) dimensions.push_back(_length);
    if (_dimensionality >= 4) dimensions.push_back(_depth);
    if (_dimensionality == 5) dimensions.push_back(_breadth);
    return true;
}
/**
 * @brief Dump the weight data to a file.
 *
 * This function dumps the weight data to a file specified by the provided filename.
 *
 * @param fname The name of the file to dump the weight data to.
 * @param pBuffer A pointer to the buffer containing the weight data.
 */
void Weight::Dump(std::string fname, Float* pBuffer)
{
    std::vector<Float> vWeight;

    if (getGpu()._numprocs == 1)
    {
        vWeight.resize(_localSize);
        cudaMemcpy(vWeight.data(), pBuffer, _localSize * sizeof(Float), cudaMemcpyDefault);
    }
    else
    {
        if (getGpu()._id == 0)
            vWeight.resize(_outputLayer._stride * _inputLayer._stride);
        uint32_t outgoingSize = _outputLayer._stride * 3;
        uint32_t incomingSize = _inputLayer._stride * 2;
        cudaMemcpy(_vWeight.data(), pBuffer, _localSize * sizeof(Float), cudaMemcpyDefault);

        if (getGpu()._id == 0)
        {
            Float* pWeight = vWeight.data();
            if (outgoingSize > incomingSize)
            {
                cudaMemcpy2D(pWeight, _outputLayer._stride * sizeof(Float), _vWeight.data(), _outputLayer._localStride * sizeof(Float), _outputLayer._localStride * sizeof(Float), _inputLayer._stride, cudaMemcpyDefault);
                pWeight += _outputLayer._localStride;
                for (uint32_t i = 1; i < getGpu()._numprocs; i++)
                {
                    uint64_t size;
                    MPI_Status status;
                    MPI_Recv(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
                    std::vector<Float> vTemp(size);
                    MPI_Recv(vTemp.data(), size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
                    uint64_t lstride = size / _inputLayer._stride;
                    Float* pSrcWeight = vTemp.data();
                    Float* pDstWeight = pWeight;
                    for (uint32_t j = 0; j < _inputLayer._stride; j++)
                    {
                        std::memcpy(pDstWeight, pSrcWeight, lstride * sizeof(Float));
                        pSrcWeight += lstride;
                        pDstWeight += _outputLayer._stride;
                    }
                    pWeight += lstride;
                }
            }
            else
            {
                cudaMemcpy(pWeight, _vWeight.data(), _outputLayer._stride * _inputLayer._localStride * sizeof(Float), cudaMemcpyDefault);
                pWeight += _outputLayer._stride * _inputLayer._localStride;
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
            uint64_t size = _vWeight.size();
            MPI_Send(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD);
            MPI_Send(_vWeight.data(), size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }

    }

    if (getGpu()._id == 0)
    {
        FILE* fp = fopen(fname.c_str(), "w");
        Float* pData = vWeight.data();
        for (int i = 0; i < _inputLayer._stride; i++)
        {
            for (int j = 0; j < _outputLayer._stride; j++)
            {
                fprintf(fp, "%12.9f ", *pData);
                pData++;
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }
}
