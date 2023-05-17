#include <array>
#include <iostream>
#include <cudnn.h>
#include "GpuTypes.h"
#include "NcExcptionWrap.h"
#include "Types.h"
#include "kernels.h"
#include "Layer.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

using namespace netCDF;
using namespace netCDF::exceptions;

WeightDescriptor::WeightDescriptor() :
/**
 * @brief The width of the object.
 */
_width(1),

/**
 * @brief The height of the object.
 */
_height(1),

/**
 * @brief The length of the object.
 */
_length(1),

/**
 * @brief The breadth of the object.
 */
_breadth(1),

/**
 * @brief The depth of the object.
 */
_depth(1),

/**
 * @brief Indicates whether the object is shared.
 */
_bShared(false),

/**
 * @brief Indicates whether the object is transposed.
 */
_bTransposed(false),

/**
 * @brief Indicates whether the object is locked.
 */
_bLocked(false),

/**
 * @brief The norm value of the object.
 */
_norm((NNFloat)0.0)
{
    
}

/**
 * @brief Dumps information about a cudnnTensorDescriptor.
 *
 * @param t The cudnnTensorDescriptor to dump.
 */
static void DumpTensor(cudnnTensorDescriptor_t t)
{
    cudnnDataType_t dataType;
    int ndims;
    std::array<int, 16> vDim;
    std::array<int, 16> vStride;
    cudnnStatus_t cudnnStatus = cudnnGetTensorNdDescriptor(t, std::size(vDim), &dataType, &ndims, vDim.data(), vStride.data());
    CUDNNERROR(cudnnStatus, "cudnnGetTensorNdDescriptor error");    

    /**
     * @brief Prints the information of the tensor.
     */
    std::cout << "Tensor:   " << ndims << " dimensions" << std::endl;
    std::cout << "DataType: " << dataType << std::endl;
    for (auto i = 0; i < ndims; i++)
        std::cout << i << " " << vDim[i] << " " << vStride[i] << std::endl;
    std::cout << std::endl;
}

/**
 * @brief Dumps information about a cudnnFilterDescriptor.
 *
 * @param f The cudnnFilterDescriptor to dump.
 */
static void DumpFilter(cudnnFilterDescriptor_t f)
{
    cudnnDataType_t dataType;
    cudnnTensorFormat_t format;
    int ndims;
    std::array<int, 16> vDim;
    cudnnStatus_t cudnnStatus = cudnnGetFilterNdDescriptor(f, 5, &dataType, &format, &ndims, vDim.data());
    CUDNNERROR(cudnnStatus, "cudnnGetFilterNdDescriptor error");        

    /**
     * @brief Prints the information of the filter.
     */
    std::cout << "Filter:   " << ndims << " dimensions" << std::endl;
    std::cout << "DataType: " << dataType << std::endl;
    std::cout << "Format:   " << format << std::endl;    
    for (auto i = 0; i < ndims; i++)
        std::cout << i << " " << vDim[i] << std::endl;
    std::cout << std::endl;
}

/**
 * @brief Dumps information about a cudnnConvolutionDescriptor.
 *
 * @param c The cudnnConvolutionDescriptor to dump.
 */
static void DumpConvolution(cudnnConvolutionDescriptor_t c)
{
    cudnnDataType_t dataType;
    cudnnConvolutionMode_t mode;
    int ndims;
    std::array<int, 16> vPad;
    std::array<int, 16> vStride;
    std::array<int, 16> vUpscale;
    cudnnStatus_t cudnnStatus = cudnnGetConvolutionNdDescriptor(c, 5, &ndims, vPad.data(), vStride.data(), vUpscale.data(), &mode, &dataType);
    CUDNNERROR(cudnnStatus, "cudnnGetConvolutionNdDescriptor error");      

    /**
     * @brief Prints the information of the convolution.
     */
    std::cout << "Convolution:   " << ndims << " dimensions" << std::endl;
    std::cout << "DataType:      " << dataType << std::endl;
    std::cout << "Mode:          " << mode << std::endl;    
    for (auto i = 0; i < ndims; i++)
        std::cout << i << " " << vPad[i] << " " << vStride[i] << " " << vUpscale[i] << std::endl;
    std::cout << std::endl;
}


bool LoadWeightDescriptorNetCDF(const string& fname, netCDF::NcFile& nc, uint32_t index, WeightDescriptor& wd)
{
    bool bResult = true; 

    if (getGpu()._id == 0)
    {
        string wstring = "weight" + std::to_string(index) + "_";
        try {
                NcGroupAtt inputLayerAtt = nc.getAtt(wstring + "inputLayer");
                if (inputLayerAtt.isNull())
                {
                    /**
                     * @brief Throws an exception if the input layer attribute is missing in the NetCDF input file.
                     *
                     * The exception message includes the file name.
                     */
                    throw NC_EXCEPTION("NcException", "No input layer supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                inputLayerAtt.getValues(wd._inputLayer);  

                NcGroupAtt outputLayerAtt = nc.getAtt(wstring + "outputLayer");
                if (outputLayerAtt.isNull())
                {
                    /**
                     * @brief Throws an exception if the output layer attribute is missing in the NetCDF input file.
                     *
                     * The exception message includes the file name.
                     */
                    throw NC_EXCEPTION("NcException", "No output layer supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                outputLayerAtt.getValues(wd._outputLayer);

                NcGroupAtt normAtt = nc.getAtt(wstring + "norm");
                if (normAtt.isNull())
                {
                    /**
                     * @brief Sets the norm value to zero if the norm attribute is missing in the NetCDF input file.
                     */
                    wd._norm = (NNFloat)0.0;
                }
                else
                    normAtt.getValues(&wd._norm);

                NcGroupAtt bSharedAtt = nc.getAtt(wstring + "bShared");
                if (bSharedAtt.isNull())
                {
                    /**
                     * @brief Throws an exception if the bShared attribute is missing in the NetCDF input file.
                     *
                     * The exception message includes the file name.
                     */
                    throw NC_EXCEPTION("NcException", "No bShared supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                uint32_t bShared;
                bSharedAtt.getValues(&bShared);
                wd._bShared = (bShared != 0);

            if (wd._bShared)
            {
                NcGroupAtt sourceInputLayerAtt = nc.getAtt(wstring + "sourceInputLayer");
                if (sourceInputLayerAtt.isNull())
                {
                    /**
                     * @brief Throws an exception if the sourceInputLayer for shared weights attribute is missing in the NetCDF input file.
                     *
                     * The exception message includes the file name.
                     */
                    throw NC_EXCEPTION("NcException", "No sourceInputLayer for shared weights supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                sourceInputLayerAtt.getValues(wd._sourceInputLayer);

                NcGroupAtt sourceOutputLayerAtt = nc.getAtt(wstring + "sourceOutputLayer");
                if (sourceOutputLayerAtt.isNull())
                {
                    /**
                     * @brief Throws an exception if the sourceOutputLayer for shared weights attribute is missing in the NetCDF input file.
                     *
                     * The exception message includes the file name.
                     */
                    throw NC_EXCEPTION("NcException", "No sourceOutputLayer for shared weights supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                sourceOutputLayerAtt.getValues(wd._sourceOutputLayer);

                NcGroupAtt bTransposedAtt = nc.getAtt(wstring + "bTransposed");
                if (bTransposedAtt.isNull())
                {
                    /**
                     * @brief Throws an exception if the bTransposed for shared weights attribute is missing in the NetCDF input file.
                     *
                     * The exception message includes the file name.
                     */
                    throw NC_EXCEPTION("NcException", "No bTransposed for shared weights supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                /**
                 * Variable to store the transposed state.
                 */
                uint32_t bTransposed;

                /**
                 * Get the values of the 'bTransposed' attribute.
                 *
                 * @param bTransposedAtt Attribute object for the 'bTransposed' attribute.
                 */
                bTransposedAtt.getValues(&bTransposed);

                /**
                 * Set the transposed state of the WeightData object based on the value of 'bTransposed'.
                 *
                 * @param wd Reference to the WeightData object.
                 */
                wd._bTransposed = (bTransposed != 0);
            }

            /**
             * Retrieve the 'bLocked' attribute from the NetCDF group.
             *
             * @param bLockedAtt Attribute object for the 'bLocked' attribute.
             */
            NcGroupAtt bLockedAtt = nc.getAtt(wstring + "bLocked");

            /**
             * Check if the 'bLocked' attribute is null.
             * If it is null, set the locked state of the WeightData object to false.
             *
             * @param wd Reference to the WeightData object.
             */
            if (bLockedAtt.isNull())
            {
                wd._bLocked = false;
            }
            else
            {
                /**
                 * Variable to store the locked state.
                 */
                uint32_t bLocked;

                /**
                 * Get the values of the 'bLocked' attribute.
                 *
                 * @param bLockedAtt Attribute object for the 'bLocked' attribute.
                 */
                bLockedAtt.getValues(&bLocked);

                /**
                 * Set the locked state of the WeightData object based on the value of 'bLocked'.
                 *
                 * @param wd Reference to the WeightData object.
                 */
                wd._bLocked = (bLocked != 0);
            }

            NcGroupAtt widthAtt = nc.getAtt(wstring + "width");
            if (widthAtt.isNull())
            {
                /**
                 * @brief Throws an exception if the weight width attribute is missing in the NetCDF input file.
                 *
                 * The exception message includes the file name.
                 */
                throw NC_EXCEPTION("NcException", "No weight width supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            widthAtt.getValues(&wd._width);

            NcGroupAtt heightAtt = nc.getAtt(wstring + "height");
            if (heightAtt.isNull())
            {
                /**
                 * @brief Throws an exception if the weight height attribute is missing in the NetCDF input file.
                 *
                 * The exception message includes the file name.
                 */
                throw NC_EXCEPTION("NcException", "No weight height supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            heightAtt.getValues(&wd._height);

            NcGroupAtt lengthAtt = nc.getAtt(wstring + "length");
            if (lengthAtt.isNull())
            {
                /**
                 * @brief Throws an exception if the weight length attribute is missing in the NetCDF input file.
                 *
                 * The exception message includes the file name.
                 */
                throw NC_EXCEPTION("NcException", "No weight length supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            lengthAtt.getValues(&wd._length);

            NcGroupAtt depthAtt = nc.getAtt(wstring + "depth");
            if (depthAtt.isNull())
            {
                /**
                 * @brief Throws an exception if the weight depth attribute is missing in the NetCDF input file.
                 *
                 * The exception message includes the file name.
                 */
                throw NC_EXCEPTION("NcException", "No weight depth supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            depthAtt.getValues(&wd._depth);

            NcGroupAtt breadthAtt = nc.getAtt(wstring + "breadth");
            if (breadthAtt.isNull())
            {
                /**
                 * @brief Throws an exception if the weight breadth attribute is missing in the NetCDF input file.
                 *
                 * The exception message includes the file name.
                 */
                throw NC_EXCEPTION("NcException", "No weight breadth supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            breadthAtt.getValues(&wd._breadth);

            NcDim biasDim = nc.getDim(wstring + "biasDim");
            NcVar biasVar = nc.getVar(wstring + "bias");  
            wd._vBias.resize(biasDim.getSize()); 
            biasVar.getVar(wd._vBias.data());         

            if (!wd._bShared)
            {
                /**
                 * @brief Loads weight data from NetCDF file if the weights are not shared.
                 *
                 * Retrieves the weight dimension and variable from the NetCDF file,
                 * resizes the weight vector, and reads the weight values into it.
                 */
                NcDim weightDim = nc.getDim(wstring + "weightDim");
                NcVar weightVar = nc.getVar(wstring + "weights");
                wd._vWeight.resize(weightDim.getSize()); 
                weightVar.getVar(wd._vWeight.data());
            }
#if 0
            /**
             * @brief Prints the weights and biases information.
             *
             * @param index The index of the weights and biases.
             */
            std::cout << "Weights " << index << " " << _vWeight.size() << " " << _vBias.size() << std::endl;

            /**
             * @brief Prints the weights and biases values.
             */
            for (int i = 0; i < 20; i++)
                std::cout << std::setw(3) << i << std::fixed << std::setprecision(8) << std::setw(16) << _vWeight[i] << std::setw(16) << _vBias[i] << std::endl;

#endif
        }
        catch (std::NcException& e)
        {
            /**
             * @brief Handles the exception thrown by NcException.
             *
             * Prints the exception message and sets the result to false.
             */
            std::cout << "WeightDescriptor::WeightDescriptor: Exception: " << e.what() << std::endl;
            bResult = false;
        }

    }

    return bResult;
}

uint32_t MPI_Bcast_WeightDescriptor(WeightDescriptor& d)
{
    /**
     * @brief Broadcasts a string using MPI communication.
     *
     * @param str The string to be broadcasted.
     */
    MPI_Bcast_string(d._inputLayer);

    /**
     * @brief Broadcasts a string using MPI communication.
     *
     * @param str The string to be broadcasted.
     */
    MPI_Bcast_string(d._outputLayer);

    /**
     * @brief Broadcasts a boolean value using MPI communication.
     *
     * @param ptr A pointer to the boolean value to be broadcasted.
     */
    MPI_Bcast(reinterpret_cast<void*>(&d._bShared), 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    /**
     * @brief Broadcasts a boolean value using MPI communication.
     *
     * @param ptr A pointer to the boolean value to be broadcasted.
     */
    MPI_Bcast(reinterpret_cast<void*>(&d._bTransposed), 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    /**
     * @brief Broadcasts a boolean value using MPI communication.
     *
     * @param ptr A pointer to the boolean value to be broadcasted.
     */
    MPI_Bcast(reinterpret_cast<void*>(&d._bLocked), 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    /**
     * @brief Broadcasts a floating-point value using MPI communication.
     *
     * @param ptr A pointer to the floating-point value to be broadcasted.
     */
    MPI_Bcast(reinterpret_cast<void*>(&d._norm), 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /**
     * @brief Broadcasts a string using MPI communication.
     *
     * @param str The string to be broadcasted.
     */
    MPI_Bcast_string(d._sourceInputLayer);

    /**
     * @brief Broadcasts a string using MPI communication.
     *
     * @param str The string to be broadcasted.
     */
    MPI_Bcast_string(d._sourceOutputLayer);

    /**
     * @brief Broadcasts an unsigned 64-bit integer value using MPI communication.
     *
     * @param ptr A pointer to the unsigned 64-bit integer value to be broadcasted.
     */
    MPI_Bcast(reinterpret_cast<void*>(&d._width), 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    /**
     * @brief Broadcasts an unsigned 64-bit integer value using MPI communication.
     *
     * @param ptr A pointer to the unsigned 64-bit integer value to be broadcasted.
     */
    MPI_Bcast(reinterpret_cast<void*>(&d._height), 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    /**
     * @brief Broadcasts an unsigned 64-bit integer value using MPI communication.
     *
     * @param ptr A pointer to the unsigned 64-bit integer value to be broadcasted.
     */
    MPI_Bcast(reinterpret_cast<void*>(&d._length), 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    /**
     * @brief Broadcasts an unsigned 64-bit integer value using MPI communication.
     *
     * @param ptr A pointer to the unsigned 64-bit integer value to be broadcasted.
     */
    MPI_Bcast(reinterpret_cast<void*>(&d._depth), 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    /**
     * @brief Broadcasts an unsigned 64-bit integer value using MPI communication.
     *
     * @param ptr A pointer to the unsigned 64-bit integer value to be broadcasted.
     */
    MPI_Bcast(reinterpret_cast<void*>(&d._breadth), 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    
    /**
     * @brief Broadcasts the size of the weight vector using MPI communication.
     *
     * @param weights The size of the weight vector.
     */
    uint64_t weights = d._vWeight.size();
    MPI_Bcast(reinterpret_cast<void*>(&weights), 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    /**
     * @brief Resizes the weight vector to the broadcasted size and broadcasts its content using MPI communication.
     *
     * @param data A pointer to the memory location of the weight vector.
     * @param count The number of elements in the weight vector.
     * @param datatype The datatype of the elements in the weight vector.
     * @param root The rank of the process broadcasting the data.
     * @param comm The MPI communicator.
     */
    d._vWeight = std::vector<float>(weights);
    MPI_Bcast(reinterpret_cast<void*>(d._vWeight.data()), static_cast<int>(weights), MPI_FLOAT, 0, MPI_COMM_WORLD);

    /**
     * @brief Reduces the capacity of the weight vector to fit its size.
     */
    d._vWeight.shrink_to_fit();

    /**
     * @brief Broadcasts the size of the bias vector using MPI communication.
     *
     * @param biases The size of the bias vector.
     */
    uint64_t biases = d._vBias.size();
    MPI_Bcast(reinterpret_cast<void*>(&biases), 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    /**
     * @brief Resizes the bias vector to the broadcasted size and broadcasts its content using MPI communication.
     *
     * @param data A pointer to the memory location of the bias vector.
     * @param count The number of elements in the bias vector.
     * @param datatype The datatype of the elements in the bias vector.
     * @param root The rank of the process broadcasting the data.
     * @param comm The MPI communicator.
     */
    d._vBias = std::vector<float>(biases);
    MPI_Bcast(reinterpret_cast<void*>(d._vBias.data()), static_cast<int>(biases), MPI_FLOAT, 0, MPI_COMM_WORLD);

    /**
     * @brief Reduces the capacity of the bias vector to fit its size.
     */
    d._vBias.shrink_to_fit();

    /**
     * @brief Returns 0 to indicate successful completion of the function.
     */
    return 0;
}
std::ostream& operator<< (std::ostream& out, WeightDescriptor& d)
{
    if (getGpu()._id == 0)
    {
        /**
         * @brief Prints information about the layer's properties.
         * The input layer, output layer, dimensions, shared status, transposed status,
         * source input layer, source output layer, locked status, and norm are printed.
         */
        std::cout << "Input Layer:        " << d._inputLayer << '\n'
                << "Output Layer:       " << d._outputLayer << '\n'
                << "Width               " << d._width << '\n'
                << "Height              " << d._height << '\n'
                << "Length              " << d._length << '\n'
                << "Depth               " << d._depth << '\n'
                << "Breadth             " << d._breadth << '\n'
                << "bShared:            " << std::boolalpha << d._bShared << '\n'
                << "bTransposed:        " << std::boolalpha << d._bTransposed << '\n';

        if (d._bShared)
        {
            /**
             * @brief Prints the source input layer and source output layer if the layer is shared.
             */
            std::cout << "sourceInputLayer:   " << d._sourceInputLayer << '\n'
                    << "sourceOutputLayer:  " << d._sourceOutputLayer << '\n';
        }

        std::cout << "bLocked:            " << std::boolalpha << d._bLocked << '\n'
                << "norm:               " << d._norm << '\n';

        /**
         * @brief Resets the boolalpha format for the output stream to default.
         */
        std::cout << std::noboolalpha;
    }

    return out;
}


Weight::Weight(Layer& inputLayer, Layer& outputLayer, bool bShared, bool bTransposed, bool bLocked, NNFloat norm) :
/**
 * @brief Initializes the input layer of the weight.
 *
 * @param inputLayer The input layer for the weight.
 */
_inputLayer = inputLayer;

/**
 * @brief Initializes the output layer of the weight.
 *
 * @param outputLayer The output layer for the weight.
 */
_outputLayer = outputLayer;

/**
 * @brief Initializes the dimensionality of the weight.
 *
 * The dimensionality refers to the number of dimensions in the weight.
 * For this weight, the dimensionality is set to 2.
 */
_dimensionality = 2;

/**
 * @brief Initializes the width of the weight.
 *
 * The width represents the width dimension of the weight tensor.
 * For this weight, the initial width is set to 1.
 */
_width = 1;

/**
 * @brief Initializes the height of the weight.
 *
 * The height represents the height dimension of the weight tensor.
 * For this weight, the initial height is set to 1.
 */
_height = 1;

/**
 * @brief Initializes the length of the weight.
 *
 * The length represents the length dimension of the weight tensor.
 * For this weight, the initial length is set to 1.
 */
_length = 1;

/**
 * @brief Initializes the depth of the weight.
 *
 * The depth represents the depth dimension of the weight tensor.
 * For this weight, the initial depth is set to 1.
 */
_depth = 1;

/**
 * @brief Initializes the breadth of the weight.
 *
 * The breadth represents the breadth dimension of the weight tensor.
 * For this weight, the initial breadth is set to 1.
 */
_breadth = 1;

/**
 * @brief Initializes the sharing count of the weight.
 *
 * The sharing count represents the number of shared instances of the weight.
 * For this weight, the initial sharing count is set to 1.
 */
_sharingCount = 1;

/**
 * @brief Initializes the update count of the weight.
 *
 * The update count represents the number of times the weight has been updated.
 * For this weight, the initial update count is set to 0.
 */
_updateCount = 0;

/**
 * @brief Initializes the shared flag of the weight.
 *
 * The shared flag indicates whether the weight is shared among multiple instances.
 * For this weight, the initial shared flag is set to the value of bShared.
 *
 * @param bShared The flag indicating if the weight is shared.
 */
_bShared = bShared;

/**
 * @brief Initializes the transposed flag of the weight.
 *
 * The transposed flag indicates whether the weight is transposed.
 * For this weight, the initial transposed flag is set to the value of bTransposed.
 *
 * @param bTransposed The flag indicating if the weight is transposed.
 */
_bTransposed = bTransposed;

/**
 * @brief Initializes the locked flag of the weight.
 *
 * The locked flag indicates whether the weight is locked and should not be modified.
 * For this weight, the initial locked flag is set to the value of bLocked.
 *
 * @param bLocked The flag indicating if the weight is locked.
 */
_bLocked = bLocked;

/**
 * @brief Initializes the norm value of the weight.
 *
 * The norm represents the normalization value for the weight.
 * For this weight, the initial norm value is set to norm.
 *
 * @param norm The normalization value for the weight.
 */
_norm = norm;
/**
 * @brief Initializes the shared pointer to the weight's shared weight.
 *
 * The shared weight pointer refers to the shared weight instance used when the weight is shared.
 * For this weight, the initial shared weight pointer is set to NULL.
 */
_pSharedWeight = NULL;

/**
 * @brief Initializes the weight data for the weight.
 *
 * The weight data represents the actual values of the weight.
 * For this weight, the weight data is initialized to an empty vector.
 */
_pbWeight = std::vector<float>();

/**
 * @brief Initializes the bias data for the weight.
 *
 * The bias data represents the bias values associated with the weight.
 * For this weight, the bias data is initialized to an empty vector.
 */
_pbBias = std::vector<float>();

/**
 * @brief Initializes the weight gradient data for the weight.
 *
 * The weight gradient data represents the gradients of the weight values.
 * For this weight, the weight gradient data is initialized to an empty vector.
 */
_pbWeightGradient = std::vector<float>();

/**
 * @brief Initializes the bias gradient data for the weight.
 *
 * The bias gradient data represents the gradients of the bias values.
 * For this weight, the bias gradient data is initialized to an empty vector.
 */
_pbBiasGradient = std::vector<float>();

/**
 * @brief Initializes the weight velocity data for the weight.
 *
 * The weight velocity data represents the velocity of weight updates.
 * For this weight, the weight velocity data is initialized to an empty vector.
 */
_pbWeightVelocity = std::vector<float>();

/**
 * @brief Initializes the bias velocity data for the weight.
 *
 * The bias velocity data represents the velocity of bias updates.
 * For this weight, the bias velocity data is initialized to an empty vector.
 */
_pbBiasVelocity = std::vector<float>();

/**
 * @brief Initializes the weight gradient velocity data for the weight.
 *
 * The weight gradient velocity data represents the velocity of weight gradient updates.
 * For this weight, the weight gradient velocity data is initialized to an empty vector.
 */
_pbWeightGradientVelocity = std::vector<float>();

/**
 * @brief Initializes the bias gradient velocity data for the weight.
 *
 * The bias gradient velocity data represents the velocity of bias gradient updates.
 * For this weight, the bias gradient velocity data is initialized to an empty vector.
 */
_pbBiasGradientVelocity = std::vector<float>();
{
    /**
     * @brief Adds the output layer to the list of outgoing layers for the input layer.
     *
     * This function establishes a connection between the input layer and the output layer,
     * indicating that the output layer is one of the layers that the input layer sends its outputs to.
     *
     * @param outputLayer The output layer to be added as an outgoing layer of the input layer.
     */
    inputLayer._vOutgoingLayer.push_back(&outputLayer);

    /**
     * @brief Adds the input layer to the list of incoming layers for the output layer.
     *
     * This function establishes a connection between the output layer and the input layer,
     * indicating that the input layer is one of the layers that provides inputs to the output layer.
     *
     * @param inputLayer The input layer to be added as an incoming layer of the output layer.
     */
    outputLayer._vIncomingLayer.push_back(&inputLayer);

    /**
     * @brief Adds the weight to the list of outgoing weights for the input layer.
     *
     * This function establishes a connection between the input layer and the weight,
     * indicating that the weight is one of the weights associated with the input layer.
     *
     * @param this Pointer to the weight to be added as an outgoing weight of the input layer.
     */
    inputLayer._vOutgoingWeight.push_back(this);

    /**
     * @brief Adds the weight to the list of incoming weights for the output layer.
     *
     * This function establishes a connection between the output layer and the weight,
     * indicating that the weight is one of the weights associated with the output layer.
     *
     * @param this Pointer to the weight to be added as an incoming weight of the output layer.
     */
    outputLayer._vIncomingWeight.push_back(this);
    
    if (_outputLayer._type == Layer::Type::Convolutional)
    {
        /**
         * @brief Sets the transformation type to Convolution.
         */
        _transform = Convolution;

        /**
         * @brief Creates a CUDNN tensor descriptor for the convolution bias.
         *
         * @param descriptor The created tensor descriptor.
         * @return The CUDNN status indicating the success or failure of the operation.
         */
        cudnnStatus_t cudnnStatus = cudnnCreateTensorDescriptor(&_convBiasTensor);
        CUDNNERROR(cudnnStatus, "Weight::Weight: Unable to create tensor descriptor");

        /**
         * @brief Creates a CUDNN filter descriptor for the convolution filter.
         *
         * @param descriptor The created filter descriptor.
         * @return The CUDNN status indicating the success or failure of the operation.
         */
        cudnnStatus = cudnnCreateFilterDescriptor(&_convFilterDesc);
        CUDNNERROR(cudnnStatus, "Weight::Weight: Unable to create filter descriptor");

        /**
         * @brief Creates a CUDNN convolution descriptor for the convolution operation.
         *
         * @param descriptor The created convolution descriptor.
         * @return The CUDNN status indicating the success or failure of the operation.
         */
        cudnnStatus = cudnnCreateConvolutionDescriptor(&_convDesc);
        CUDNNERROR(cudnnStatus, "Weight::Weight: Unable to create convolution descriptor");


        vector<int> vFilterDim(5, 1);
        switch (_outputLayer._dimensions)
        {
            case 2:
                /**
                 * @brief Sets the filter dimensions for a 2-dimensional weight.
                 *
                 * This case corresponds to a weight with 2-dimensional filter dimensions,
                 * where the filter dimensions are based on the input and output layer properties.
                 * The dimensionality of the weight is set to 3.
                 *
                 * @param vFilterDim The array to store the filter dimensions.
                 */
                vFilterDim[0] = _outputLayer._Ny;
                vFilterDim[1] = _inputLayer._Ny;
                vFilterDim[2] = _inputLayer._kernelX;
                _dimensionality = 3;
                break;
                
            case 3:
                /**
                 * @brief Sets the filter dimensions for a 3-dimensional weight.
                 *
                 * This case corresponds to a weight with 3-dimensional filter dimensions,
                 * where the filter dimensions are based on the input and output layer properties.
                 * The dimensionality of the weight is set to 4.
                 *
                 * @param vFilterDim The array to store the filter dimensions.
                 */
                vFilterDim[0] = _outputLayer._Nz;
                vFilterDim[1] = _inputLayer._Nz;
                vFilterDim[2] = _outputLayer._kernelY;
                vFilterDim[3] = _outputLayer._kernelX;
                _dimensionality = 4;
                break;   
                            
            case 4:
                /**
                 * @brief Sets the filter dimensions for a 4-dimensional weight.
                 *
                 * This case corresponds to a weight with 4-dimensional filter dimensions,
                 * where the filter dimensions are based on the input and output layer properties.
                 * The dimensionality of the weight is set to 5.
                 *
                 * @param vFilterDim The array to store the filter dimensions.
                 */
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

        /**
         * @brief Sets the dimensions and data type of the convolution filter descriptor.
         * The dimensions are set based on the output layer dimensions and the filter dimensions.
         */
        _width = vFilterDim[0];
        _height = vFilterDim[1];
        _length = vFilterDim[2];
        _depth = vFilterDim[3];
        _breadth = vFilterDim[4];

        /**
         * @brief Initializes the convolution padding, stride, and upscale vectors with default values.
         */
        std::vector<int> vConvPad(3, 0);
        std::vector<int> vConvStride(3, 1);
        std::vector<int> vConvUpscale(3, 1);
        switch (_outputLayer._dimensions)
        {
            case 2:
                /**
                 * @brief Sets the convolution padding and stride values for 2-dimensional convolution.
                 */
                vConvPad[0] = _outputLayer._kernelPaddingX;
                vConvStride[0] = _outputLayer._kernelStrideX;
                break;

            case 3:
                /**
                 * @brief Sets the convolution padding and stride values for 3-dimensional convolution.
                 */
                vConvPad[0] = _outputLayer._kernelPaddingY;
                vConvStride[0] = _outputLayer._kernelStrideY;
                vConvPad[1] = _outputLayer._kernelPaddingX;
                vConvStride[1] = _outputLayer._kernelStrideX;
                break;

            case 4:
                /**
                 * @brief Sets the convolution padding and stride values for 4-dimensional convolution.
                 */
                vConvPad[0] = _outputLayer._kernelPaddingZ;
                vConvStride[0] = _outputLayer._kernelStrideZ;
                vConvPad[1] = _outputLayer._kernelPaddingY;
                vConvStride[1] = _outputLayer._kernelStrideY;
                vConvPad[2] = _outputLayer._kernelPaddingX;
                vConvStride[2] = _outputLayer._kernelStrideX;
                break;
        }
        /**
         * @brief Sets the convolution descriptor with the specified parameters.
         *
         * @param _outputLayer._kernelDimensions The number of dimensions of the kernel.
         * @param vConvPad The padding values for each dimension of the convolution.
         * @param vConvStride The stride values for each dimension of the convolution.
         * @param vConvUpscale The upscale values for each dimension of the convolution.
         */
        cudnnStatus = cudnnSetConvolutionNdDescriptor(_convDesc, _outputLayer._kernelDimensions, vConvPad.data(), vConvStride.data(), vConvUpscale.data(), CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
        CUDNNERROR(cudnnStatus, "Weight::Weight: cudnnSetConvolutionNdDescriptor failed.");

        /**
         * @brief Sets the tensor descriptor for the convolution bias.
         *
         * @param CUDNN_DATA_FLOAT The data type of the bias tensor.
         * @param outputLayer._dimensions + 1 The number of dimensions of the bias tensor.
         * @param vBiasDim The dimensions of the bias tensor.
         * @param vBiasStride The stride values for each dimension of the bias tensor.
         */
        std::vector<int> vBiasDim(5, 1);
        std::vector<int> vBiasStride(5, 1);
        vBiasDim[1] = vFilterDim[0];
        cudnnStatus = cudnnSetTensorNdDescriptor(_convBiasTensor, CUDNN_DATA_FLOAT, outputLayer._dimensions + 1, vBiasDim.data(), vBiasStride.data());
        CUDNNERROR(cudnnStatus, "Weight::Weight: Unable to set bias tensor descriptor");

        /**
         * @brief Calculates the size of the convolution weights and bias.
         * The size is calculated based on the filter dimensions and output layer parameters.
         */
        _size = vFilterDim[0] * vFilterDim[1] * _outputLayer._kernelX * _outputLayer._kernelY * _outputLayer._kernelZ;
        _biasSize = vFilterDim[0];
        _localSize = _size;
        _localBiasSize = _biasSize;
        
        if (getGpu()._id == 0)
        {
        /**
         * @brief Prints information about the allocation of memory for convolutional weights.
         * The size of the allocated memory is calculated based on the localSize, sizeof(NNFloat), vFilterDim, and output layer dimensions.
         * The names of the input and output layers are also printed.
         */
        std::printf("Weight::Weight: Allocating %" PRIu64 " bytes (%d x %d x %u", _localSize * sizeof(NNFloat), vFilterDim[0], vFilterDim[1], _outputLayer._kernelX);
        if (_outputLayer._dimensions >= 3)
            std::printf(" x %u", _outputLayer._kernelY);
        if (_outputLayer._dimensions >= 4)
            std::printf(" x %u", _outputLayer._kernelZ);
        std::printf(") for convolutional weights between layers %s and %s\n", inputLayer._name.c_str(), outputLayer._name.c_str());
        
    }
    else
    {
        /**
         * \brief Determines the type of transformation between layers.
         */
        _transform                  = Linear;

        /**
         * \brief Size of outgoing connections from this weight layer.
         */
        uint32_t outgoingSize       = outputLayer._stride * 3;

        /**
         * \brief Size of incoming connections to this weight layer.
         */
        uint32_t incomingSize       = inputLayer._stride * 2;

        if (outgoingSize > incomingSize)
        {
            /**
             * \brief Adds the output layer to the list of larger layers for the input layer.
             */
            inputLayer._vOutgoingLargerLayer.push_back(&outputLayer);

            /**
             * \brief Adds the weight layer to the list of larger weights for the input layer.
             */
            inputLayer._vOutgoingLargerWeight.push_back(this);

            /**
             * \brief Width of the weight layer (output layer's local stride).
             */
            _width                  = outputLayer._localStride;

            /**
             * \brief Height of the weight layer (input layer's stride).
             */
            _height                 = inputLayer._stride;
        }
        else
        {
            /**
             * \brief Adds the input layer to the list of larger layers for the output layer.
             */
            outputLayer._vIncomingLargerLayer.push_back(&inputLayer);

            /**
             * \brief Adds the weight layer to the list of larger weights for the output layer.
             */
            outputLayer._vIncomingLargerWeight.push_back(this);

            /**
             * \brief Width of the weight layer (output layer's stride).
             */
            _width                  = outputLayer._stride;

            /**
             * \brief Height of the weight layer (input layer's local stride).
             */
            _height                 = inputLayer._localStride;
        }

        /**
         * \brief Size of the weight layer in local memory.
         */
        _localSize                  = _width * _height * _length * _depth * _breadth;

        /**
         * \brief Size of the bias layer in local memory.
         */
        _localBiasSize              = outputLayer._localStride;

        /**
         * \brief Total size of the weight layer.
         */
        _size                       = outputLayer._stride * inputLayer._stride *_length * _depth * _breadth;

        /**
         * \brief Total size of the bias layer.
         */
        _biasSize                   = outputLayer._stride;

        if (getGpu()._id == 0)
        {
            /**
             * \brief Prints the memory allocation information for the weight layer.
             */
            printf("Weight::Weight: Allocating %" PRIu64 " bytes (%" PRIu64 ", %" PRIu64 ") for fully connected weights between layers %s and %s\n", _localSize * sizeof(float), _width, _height, inputLayer._name.c_str(), outputLayer._name.c_str());
        }
    }

    /**
     * @brief Resizes the bias vector and initializes the GPU buffer for bias.
     *
     * @param _localBiasSize The size of the local bias vector.
     */
    _vBias.resize(_localBiasSize);
    _pbBias.reset(new GpuBuffer<NNFloat>(_localBiasSize));

    /**
     * @brief Initializes the GPU buffer for bias gradients if the layer transformation is convolution.
     *
     * @param _localBiasSize The size of the local bias vector.
     */
    if (_transform == Convolution)
    {
        _pbBiasGradient.reset(new GpuBuffer<NNFloat>(_localBiasSize));
    }
}

Weight::~Weight()
{
}

/**
 * @brief Clears the velocity buffers used for weight updates.
 */
void Weight::ClearVelocity()
{
    /**
     * @brief Sets the device memory for weight velocity to zero.
     *
     * This function asynchronously sets the device memory for weight velocity (_pbWeightVelocity)
     * to zero using cudaMemsetAsync. The size of the memory region to be set is determined by _localSize.
     *
     * @param dest The destination pointer to the device memory.
     * @param value The value to set the memory to (zero in this case).
     * @param count The size of the memory region to be set, in bytes.
     */
    cudaMemsetAsync(_pbWeightVelocity->_pDevData, 0, _localSize * sizeof(*_pbWeightVelocity->_pDevData));

    /**
     * @brief Sets the device memory for bias velocity to zero.
     *
     * This function asynchronously sets the device memory for bias velocity (_pbBiasVelocity)
     * to zero using cudaMemsetAsync. The size of the memory region to be set is determined by _localBiasSize.
     *
     * @param dest The destination pointer to the device memory.
     * @param value The value to set the memory to (zero in this case).
     * @param count The size of the memory region to be set, in bytes.
     */
    cudaMemsetAsync(_pbBiasVelocity->_pDevData, 0, _localBiasSize * sizeof(*_pbBiasVelocity->_pDevData));

    /**
     * @brief Sets the device memory for weight gradient velocity to zero if it is non-null.
     *
     * This function asynchronously sets the device memory for weight gradient velocity (_pbWeightGradientVelocity)
     * to zero using cudaMemsetAsync if it is non-null. The size of the memory region to be set is determined by _localSize.
     * If _pbWeightGradientVelocity is null, the function call is a no-op.
     *
     * @param dest The destination pointer to the device memory.
     * @param value The value to set the memory to (zero in this case).
     * @param count The size of the memory region to be set, in bytes.
     */
    _pbWeightGradientVelocity ? cudaMemsetAsync(_pbWeightGradientVelocity->_pDevData, 0, _localSize * sizeof(*_pbWeightGradientVelocity->_pDevData)) : void(0);

    /**
     * @brief Sets the device memory for bias gradient velocity to zero if it is non-null.
     *
     * This function asynchronously sets the device memory for bias gradient velocity (_pbBiasGradientVelocity)
     * to zero using cudaMemsetAsync if it is non-null. The size of the memory region to be set is determined by _localBiasSize.
     * If _pbBiasGradientVelocity is null, the function call is a no-op.
     *
     * @param dest The destination pointer to the device memory.
     * @param value The value to set the memory to (zero in this case).
     * @param count The size of the memory region to be set, in bytes.
     */
    _pbBiasGradientVelocity ? cudaMemsetAsync(_pbBiasGradientVelocity->_pDevData, 0, _localBiasSize * sizeof(*_pbBiasGradientVelocity->_pDevData)) : void(0);

}

/**
 * @brief Clears the gradient buffer used for weight updates.
 */
void Weight::ClearGradient()
{
    cudaMemsetAsync(_pbWeightGradient->_pDevData, 0, _localSize * sizeof(*_pbWeightGradient->_pDevData));
}

/**
 * @brief Randomizes the weight matrix by generating random values based on the specified weight initialization method.
 *        This function is only applicable to non-shared weight matrices.
 */
void Weight::Randomize()
{
    if (!_bShared)
    {
        /**
         * @brief Initializes the weights of a neural network layer using various initialization methods.
         *
         * @param scale The scale factor for weight initialization.
         * @param bias The bias factor for weight initialization.
         * @param rng The random number generator.
         */
        NNFloat scale, bias;
        curandGenerator_t& rng = getGpu()._RNG;

        // Generate uniform random values for weights and biases
        curandGenerateUniformAsync(rng, _pbWeight->_pDevData, _localSize);
        curandGenerateUniformAsync(rng, _pbBias->_pDevData, _localBiasSize);
        cudaDeviceSynchronize();

        // Initialize weights based on the chosen weight initialization method
        switch (_outputLayer._weightInit)
        {
        case CaffeXavier:
            /**
             * @brief Xavier weight initialization method for Caffe models.
             * The scale and bias factors are calculated based on the weightInitScale and stride values of the output layer.
             */
            scale = _outputLayer._weightInitScale * 2.0f * sqrtf(3.0f / _outputLayer._stride);
            bias = 0.5f * scale;
            kScaleAndBias(_pbWeight->_pDevData, _localSize, scale, bias);
            break;

        case Xavier:
            /**
             * @brief Xavier weight initialization method.
             * The scale and bias factors are calculated based on the weightInitScale, stride, and input layer stride values.
             */
            scale = _outputLayer._weightInitScale * sqrtf(6.0f / (_outputLayer._stride + _inputLayer._stride));
            bias = 0.5f * scale;
            kScaleAndBias(_pbWeight->_pDevData, _localSize, scale, bias);
            break;

        case Uniform:
            /**
             * @brief Uniform weight initialization method.
             * The scale and bias factors are calculated based on the weightInitScale value.
             */
            scale = 2.0f * _outputLayer._weightInitScale;
            bias = 0.5f * scale;
            kScaleAndBias(_pbWeight->_pDevData, _localSize, scale, bias);
            break;

        case Gaussian:
            /**
             * @brief Gaussian weight initialization method.
             * Random values are generated from a normal distribution with mean 0 and standard deviation equal to weightInitScale.
             */
            curandGenerateNormalAsync(rng, _pbWeight->_pDevData, _localSize, 0.0f, _outputLayer._weightInitScale);
            break;

        case UnitBall:
            /**
             * @brief Unit ball weight initialization method.
             * The scale factor is set to weightInitScale, and bias is set to 0.
             */
            scale = _outputLayer._weightInitScale;
            kScaleAndBias(_pbWeight->_pDevData, _localSize, scale, 0.0f);
            break;

        case SELU:
            /**
             * @brief SELU weight initialization method.
             * Random values are generated from a normal distribution with mean 0 and standard deviation equal to 1.0 / input layer stride.
             */
            curandGenerateNormalAsync(rng, _pbWeight->_pDevData, _localSize, 0.0f, 1.0f / _inputLayer._stride);
            break;

        case Constant:
            /**
             * @brief Constant weight initialization method.
             * All weight values are set to 0, and the bias factor is set to weightInitScale.
             */
            cudaMemsetAsync(_pbWeight->_pDevData, 0, _localSize * sizeof(*_pbWeight->_pDevData));
            kScaleAndBias(_pbWeight->_pDevData, _localSize, static_cast<NNFloat>(0.0), _outputLayer._weightInitScale);
            break;
        };
    }
    
    /**
     * @brief Sets the bias memory to zero asynchronously.
     *
     * @param _pbBias->_pDevData Pointer to the bias data on the device.
     * @param _localBiasSize The size of the bias data.
     */
    cudaMemsetAsync(_pbBias->_pDevData, 0, _localBiasSize * sizeof(*_pbBias->_pDevData));

    /**
     * @brief Applies scaling and bias to the bias data.
     *
     * @param _pbBias->_pDevData Pointer to the bias data on the device.
     * @param _localBiasSize The size of the bias data.
     * @param scale The scaling factor.
     * @param bias The bias value.
     */
    kScaleAndBias(_pbBias->_pDevData, _localBiasSize, static_cast<NNFloat>(0.0), static_cast<NNFloat>(0.1));
 
}

/**
 * @brief Locks the weight matrix, preventing updates.
 */
void Weight::Lock()
{
    _bLocked = true;
}

/**
 * @brief Unlocks the weight matrix, allowing updates.
 */
void Weight::Unlock()
{
    _bLocked = false;
}
/**
 * @brief Refreshes the state of the weight matrix based on the network and training mode.
 *
 * @param pNetwork Pointer to the network.
 * @param mode The training mode.
 */
void Weight::RefreshState(Network* pNetwork, TrainingMode mode)
{
    if (mode != TrainingMode::SGD)
    {
        /**
         * @brief Initializes the weight velocity buffer if it is not already initialized.
         *
         * If the weight velocity buffer (_pbWeightVelocity) is not already initialized,
         * this function creates a new GpuBuffer object with the appropriate size (_localSize)
         * and assigns it to _pbWeightVelocity.
         */
        if (!_pbWeightVelocity)
            _pbWeightVelocity.reset(new GpuBuffer<NNFloat>(_localSize));

        /**
         * @brief Initializes the bias velocity buffer if it is not already initialized.
         *
         * If the bias velocity buffer (_pbBiasVelocity) is not already initialized,
         * this function creates a new GpuBuffer object with the appropriate size (_localBiasSize)
         * and assigns it to _pbBiasVelocity.
         */
        if (!_pbBiasVelocity)
            _pbBiasVelocity.reset(new GpuBuffer<NNFloat>(_localBiasSize));

        /**
         * @brief Initializes the weight gradient velocity buffer if required by the training mode.
         *
         * If the training mode is AdaDelta or Adam, this function initializes the weight gradient velocity buffer
         * (_pbWeightGradientVelocity) if it is not already initialized. It creates a new GpuBuffer object with the
         * appropriate size (_localSize) and assigns it to _pbWeightGradientVelocity. If the training mode is not
         * AdaDelta or Adam, the weight gradient velocity buffer is reset to nullptr.
         */
        if ((mode == TrainingMode::AdaDelta) || (mode == TrainingMode::Adam))
        {
            if (!_pbWeightGradientVelocity)
                _pbWeightGradientVelocity.reset(new GpuBuffer<NNFloat>(_localSize));
            if (!_pbBiasGradientVelocity)
                _pbBiasGradientVelocity.reset(new GpuBuffer<NNFloat>(_localBiasSize));
        }
        else
        {
            /**
             * @brief Resets the weight gradient velocity buffers.
             *
             * If the training mode is not AdaDelta or Adam, this function resets the weight gradient velocity buffers
             * (_pbWeightGradientVelocity and _pbBiasGradientVelocity) to nullptr, effectively deallocating the memory.
             */
            _pbWeightGradientVelocity.reset();
            _pbBiasGradientVelocity.reset();
        }
    }
    else
    {
        /**
         * @brief Resets the weight and bias velocity buffers.
         */
        _pbWeightVelocity.reset();
        _pbBiasVelocity.reset();

        /**
         * @brief Resets the weight and bias gradient velocity buffers.
         */
        _pbWeightGradientVelocity.reset();
        _pbBiasGradientVelocity.reset();
    }
    
    if (_outputLayer._type == Layer::Type::Convolutional)
    {
        /**
         * @brief Prints the information about getting the algorithm between the input layer and output layer.
         *
         * @param inputName The name of the input layer.
         * @param outputName The name of the output layer.
         */
        std::printf("Getting algorithm between %s and %s\n", _inputLayer._name.c_str(), _outputLayer._name.c_str());

        /**
         * @brief Gets the workspace size and convolution forward algorithm.
         */
        size_t workspaceSize;
        cudnnStatus_t cudnnStatus = cudnnGetConvolutionForwardAlgorithm_v7(getGpu()._cuDNNHandle(),
                                                                            _inputLayer.getTensorDescriptor(),
                                                                            _convFilterDesc,
                                                                            _convDesc,
                                                                            _outputLayer.getTensorDescriptor(),
                                                                            1,
                                                                            (cudnnConvolutionFwdAlgoPerfStruct*)&_convFWAlgo);
        CUDNNERROR(cudnnStatus, "Weight::Refresh: cudnnGetConvolutionForwardAlgorithm_v7 failed.");

            /**
             * @brief Gets the workspace size for convolution forward.
             */
            cudnnStatus = cudnnGetConvolutionForwardWorkspaceSize(getGpu()._cuDNNHandle(),
                                                                    _inputLayer.getTensorDescriptor(),
                                                                    _convFilterDesc,
                                                                    _convDesc,
                                                                    _outputLayer.getTensorDescriptor(),
                                                                    _convFWAlgo,
                                                                    &workspaceSize);
            CUDNNERROR(cudnnStatus, "Weight::Refresh: cudnnGetConvolutionForwardWorkspaceSize failed.");

            /**
             * @brief Sets the CUDNN workspace size in the network.
             *
             * @param workspaceSize The size of the workspace.
             */
            pNetwork->SetCUDNNWorkspace(workspaceSize);

            /**
             * @brief Gets the convolution backward filter algorithm.
             */
            cudnnStatus = cudnn::cudnnGetConvolutionBackwardFilterAlgorithm(getGpu()._cuDNNHandle(),
                                                                            _inputLayer.getTensorDescriptor(),
                                                                            _outputLayer.getTensorDescriptor(),
                                                                            _convDesc,
                                                                            _convFilterDesc,
                                                                            0,
                                                                            &_convBWWeightAlgo);
            CUDNNERROR(cudnnStatus, "Weight::Refresh: cudnnGetConvolutionBackwardFilterAlgorithm failed.");


            /**
             * @brief Gets the workspace size for convolution backward filter.
             */
            cudnnStatus = cudnnGetConvolutionBackwardFilterWorkspaceSize(getGpu()._cuDNNHandle(),
                                                                            _inputLayer.getTensorDescriptor(),
                                                                            _outputLayer.getTensorDescriptor(),
                                                                            _convDesc,
                                                                            _convFilterDesc,
                                                                            _convBWWeightAlgo,
                                                                            &workspaceSize);
            CUDNNERROR(cudnnStatus, "Weight::Refresh: cudnnGetConvolutionBackwardFilterWorkspaceSize failed.");

            /**
             * @brief Sets the CUDNN workspace size in the network.
             *
             * @param workspaceSize The size of the workspace.
             */
            pNetwork->SetCUDNNWorkspace(workspaceSize);

        #if 0
            /**
             * @brief Dumps information about the output layer tensor descriptor.
             */
            DumpTensor(_outputLayer.getTensorDescriptor());

            /**
             * @brief Dumps information about the input layer tensor descriptor.
             */
            DumpTensor(_inputLayer.getTensorDescriptor());

        #endif

            /**
             * @brief Gets the convolution backward data algorithm.
             */
            cudnnStatus = cudnnGetConvolutionBackwardDataAlgorithm_v7(getGpu()._cuDNNHandle(),
                                                                        _convFilterDesc,
                                                                        _outputLayer.getTensorDescriptor(),
                                                                        _convDesc,
                                                                        _inputLayer.getTensorDescriptor(),
                                                                        0,
                                                                        (cudnnConvolutionBwdDataAlgo_t*)&_convBWDeltaAlgo);
            CUDNNERROR(cudnnStatus, "Weight::Refresh: cudnnGetConvolutionBackwardDataAlgorithm failed.");


        #if 0
            cout << "Input ";
            DumpTensor(_inputLayer.getTensorDescriptor());
            cout << "Output ";
            DumpTensor(_outputLayer.getTensorDescriptor());
            DumpConvolution(_convDesc);
            DumpFilter(_convFilterDesc);
        #endif
                                        
        /**
         * @brief Gets the workspace size for convolution backward data.
         */
        cudnnStatus = cudnnGetConvolutionBackwardDataWorkspaceSize(getGpu()._cuDNNHandle(),
                                                                    _convFilterDesc,
                                                                    _outputLayer._tensorDescriptor,
                                                                    _convDesc,
                                                                    _inputLayer._tensorDescriptor,
                                                                    _convBWDeltaAlgo,
                                                                    &workspaceSize);
        CUDNNERROR(cudnnStatus, "Weight::Refresh: cudnnGetConvolutionBackwardDataWorkspaceSize failed.");

        /**
         * @brief Sets the CUDNN workspace size in the network.
         *
         * @param workspaceSize The size of the workspace.
         */
        pNetwork->SetCUDNNWorkspace(workspaceSize);

        
        /**
         * @brief Creates a vector with initial values of 1 for output dimensions.
         */
        vector<int> vOutputDim(8, 1);

        /**
         * @brief Gets the forward convolution output dimensions for the given descriptors.
         *
         * @param _convDesc The convolution descriptor.
         * @param _inputLayer._tensorDescriptor The input tensor descriptor.
         * @param _convFilterDesc The convolution filter descriptor.
         * @param _outputLayer._dimensions + 1 The number of dimensions of the output layer.
         * @param vOutputDim.data() Pointer to the data of the output dimensions vector.
         */
        cudnnStatus = cudnnGetConvolutionNdForwardOutputDim(_convDesc,
                                                            _inputLayer._tensorDescriptor,
                                                            _convFilterDesc,
                                                            _outputLayer._dimensions + 1,
                                                            vOutputDim.data());
        CUDNNERROR(cudnnStatus, "Weight::Refresh: cudnnGetConvolutionNdForwardOutputDim failed.");

        /**
         * @brief Calculates the total dimension by multiplying the output dimensions.
         */
        size_t dim = 1;
        for (size_t i = 0; i < _outputLayer._dimensions + 1; i++)
            dim *= vOutputDim[i];

        /**
         * @brief Checks if the calculated dimension matches the expected dimension.
         *        If not, it prints an error message and shuts down the GPU.
         */
        if (dim != _outputLayer._maxLocalStride * _outputLayer._localBatch)
        {
            if (getGpu()._id == 0)
                std::printf("Output layer %s has incorrectly calculated dimensions for cuDNN.\n", _outputLayer._name.c_str());
            getGpu().Shutdown();
        }
    }
}
/**
 * @brief Calculates the regularization error of the weight matrix using the specified regularization hyperparameters.
 *
 * @param lambda The weight decay hyperparameter.
 * @param lambda1 The L1 weight decay hyperparameter.
 * @return The regularization error.
 */
NNFloat Weight::CalculateRegularizationError(NNFloat lambda, NNFloat lambda1)
{
    if (_bShared)
        return 0;
    else
        return kCalculateRegularizationError(lambda, lambda1, _pbWeight->_pDevData, _localSize);
}
/**
 * @brief Updates the weights and biases of the weight matrix using the specified training mode and hyperparameters.
 *
 * @param trainingMode The training mode to use for weight and bias updates.
 * @param batch The current batch size.
 * @param alpha The learning rate hyperparameter.
 * @param lambda The weight decay hyperparameter.
 * @param lambda1 The L1 weight decay hyperparameter.
 * @param mu The momentum hyperparameter.
 * @param mu1 The Nesterov momentum hyperparameter.
 * @param t The current time step.
 */
void Weight::UpdateWeights(TrainingMode trainingMode, uint32_t batch, NNFloat alpha, NNFloat lambda, NNFloat lambda1, NNFloat mu, NNFloat mu1, NNFloat t)
{
    cublasStatus_t cstatus;

    if (_bLocked)
        return; 

    if (!_bShared)
    {
        switch (trainingMode)
        {
            case SGD:
                /**
                 * @brief Updates weights using the Stochastic Gradient Descent (SGD) algorithm.
                 *
                 * This case corresponds to the SGD training mode, where the weights are updated
                 * using the kSGDUpdateWeights kernel. The function passes the necessary parameters
                 * for the kernel to perform the weight update.
                 *
                 * @param alpha The learning rate.
                 * @param lambda The weight decay.
                 * @param lambda1 The L1 regularization.
                 * @param size The size of the weight array.
                 * @param gradient The device memory for the weight gradients.
                 * @param weights The device memory for the weights.
                 */
                kSGDUpdateWeights(alpha, lambda, lambda1, _localSize, _pbWeightGradient->_pDevData, _pbWeight->_pDevData);
                break;
                
            case Momentum:
                /**
                 * @brief Updates weights using the Momentum algorithm.
                 *
                 * This case corresponds to the Momentum training mode, where the weights are updated
                 * using the kMomentumUpdateWeights kernel. The function passes the necessary parameters
                 * for the kernel to perform the weight update.
                 *
                 * @param alpha The learning rate.
                 * @param lambda The weight decay.
                 * @param lambda1 The L1 regularization.
                 * @param mu The momentum factor.
                 * @param size The size of the weight array.
                 * @param velocity The device memory for the weight velocities.
                 * @param gradient The device memory for the weight gradients.
                 * @param weights The device memory for the weights.
                 */
                kMomentumUpdateWeights(alpha, lambda, lambda1, mu, _localSize, _pbWeightVelocity->_pDevData, _pbWeightGradient->_pDevData, _pbWeight->_pDevData);
                break;
                        
            case AdaGrad:
                /**
                 * @brief Updates weights using the AdaGrad algorithm.
                 *
                 * This case corresponds to the AdaGrad training mode, where the weights are updated
                 * using the kAdaGradUpdateWeights kernel. The function passes the necessary parameters
                 * for the kernel to perform the weight update.
                 *
                 * @param alpha The learning rate.
                 * @param lambda The weight decay.
                 * @param lambda1 The L1 regularization.
                 * @param size The size of the weight array.
                 * @param velocity The device memory for the weight velocities.
                 * @param gradient The device memory for the weight gradients.
                 * @param weights The device memory for the weights.
                 */
                kAdaGradUpdateWeights(alpha, lambda, lambda1, _localSize, _pbWeightVelocity->_pDevData, _pbWeightGradient->_pDevData, _pbWeight->_pDevData);
                break;

            case Nesterov:
                /**
                 * @brief Updates weights using the Nesterov algorithm.
                 *
                 * This case corresponds to the Nesterov training mode, where the weights are updated
                 * using the kNesterovUpdateWeights kernel. The function passes the necessary parameters
                 * for the kernel to perform the weight update.
                 *
                 * @param alpha The learning rate.
                 * @param lambda The weight decay.
                 * @param lambda1 The L1 regularization.
                 * @param mu The momentum factor.
                 * @param size The size of the weight array.
                 * @param velocity The device memory for the weight velocities.
                 * @param gradient The device memory for the weight gradients.
                 * @param weights The device memory for the weights.
                 */
                kNesterovUpdateWeights(alpha, lambda, lambda1, mu, _localSize, _pbWeightVelocity->_pDevData, _pbWeightGradient->_pDevData, _pbWeight->_pDevData);
                break;
                        
            case RMSProp:
                /**
                 * @brief Updates weights using the RMSProp algorithm.
                 *
                 * This case corresponds to the RMSProp training mode, where the weights are updated
                 * using the kRMSPropUpdateWeights kernel. The function passes the necessary parameters
                 * for the kernel to perform the weight update.
                 *
                 * @param alpha The learning rate.
                 * @param lambda The weight decay.
                 * @param lambda1 The L1 regularization.
                 * @param mu The momentum factor.
                 * @param size The size of the weight array.
                 * @param velocity The device memory for the weight velocities.
                 * @param gradient The device memory for the weight gradients.
                 * @param weights The device memory for the weights.
                 */
                kRMSPropUpdateWeights(alpha, lambda, lambda1, mu, _localSize, _pbWeightVelocity->_pDevData, _pbWeightGradient->_pDevData, _pbWeight->_pDevData);
                break;

            case AdaDelta:
                /**
                 * @brief Updates weights using the AdaDelta algorithm.
                 *
                 * This case corresponds to the AdaDelta training mode, where the weights are updated
                 * using the kAdaDeltaUpdateWeights kernel. The function passes the necessary parameters
                 * for the kernel to perform the weight update.
                 *
                 * @param lambda The weight decay.
                 * @param lambda1 The L1 regularization.
                 * @param mu The momentum factor.
                 * @param size The size of the weight array.
                 * @param velocity The device memory for the weight velocities.
                 * @param gradient The device memory for the weight gradients.
                 * @param gradientVelocity The device memory for the weight gradient velocities.
                 * @param weights The device memory for the weights.
                 */
                kAdaDeltaUpdateWeights(lambda, lambda1, mu, _localSize, _pbWeightVelocity->_pDevData, _pbWeightGradient->_pDevData, _pbWeightGradientVelocity->_pDevData, _pbWeight->_pDevData);
                break;  

            case Adam:
                /**
                 * @brief Updates weights using the Adam algorithm.
                 *
                 * This case corresponds to the Adam training mode, where the weights are updated
                 * using the kAdamUpdateWeights kernel. The function passes the necessary parameters
                 * for the kernel to perform the weight update.
                 *
                 * @param alpha The learning rate.
                 * @param lambda The weight decay.
                 * @param lambda1 The L1 regularization.
                 * @param mu The momentum factor.
                 * @param mu1 The first moment decay rate.
                 * @param t The time step count.
                 * @param size The size of the weight array.
                 * @param velocity The device memory for the weight velocities.
                 * @param gradient The device memory for the weight gradients.
                 * @param gradientVelocity The device memory for the weight gradient velocities.
                 * @param weights The device memory for the weights.
                 */
                kAdamUpdateWeights(alpha, lambda, lambda1, mu, mu1, t, _localSize, _pbWeightVelocity->_pDevData, _pbWeightGradient->_pDevData, _pbWeightGradientVelocity->_pDevData, _pbWeight->_pDevData);
                break;  
        }
    }

    if (_transform == Linear)
    {
        switch (trainingMode)
        {
            case SGD:
                /**
                 * @brief Updates biases using the Stochastic Gradient Descent (SGD) algorithm.
                 *
                 * This case corresponds to the SGD training mode, where the biases are updated
                 * using the kSGDUpdateBiases kernel. The function passes the necessary parameters
                 * for the kernel to perform the bias update.
                 *
                 * @param alpha The learning rate.
                 * @param batch The batch size.
                 * @param size The size of the bias array.
                 * @param delta The device memory for the delta values.
                 * @param biases The device memory for the biases.
                 */
                kSGDUpdateBiases(alpha, batch, _localBiasSize, _outputLayer._pbDelta->_pDevData, _pbBias->_pDevData);
                break;

            case Momentum:
                /**
                 * @brief Updates biases using the Momentum algorithm.
                 *
                 * This case corresponds to the Momentum training mode, where the biases are updated
                 * using the kMomentumUpdateBiases kernel. The function passes the necessary parameters
                 * for the kernel to perform the bias update.
                 *
                 * @param alpha The learning rate.
                 * @param mu The momentum factor.
                 * @param batch The batch size.
                 * @param size The size of the bias array.
                 * @param delta The device memory for the delta values.
                 * @param velocity The device memory for the bias velocities.
                 * @param biases The device memory for the biases.
                 */
                kMomentumUpdateBiases(alpha, mu, batch, _localBiasSize, _outputLayer._pbDelta->_pDevData, _pbBiasVelocity->_pDevData, _pbBias->_pDevData);
                break;
                    
            case AdaGrad:
                /**
                 * @brief Updates biases using the AdaGrad algorithm.
                 *
                 * This case corresponds to the AdaGrad training mode, where the biases are updated
                 * using the kAdaGradUpdateBiases kernel. The function passes the necessary parameters
                 * for the kernel to perform the bias update.
                 *
                 * @param alpha The learning rate.
                 * @param batch The batch size.
                 * @param size The size of the bias array.
                 * @param delta The device memory for the delta values.
                 * @param velocity The device memory for the bias velocities.
                 * @param biases The device memory for the biases.
                 */
                kAdaGradUpdateBiases(alpha, batch, _localBiasSize, _outputLayer._pbDelta->_pDevData, _pbBiasVelocity->_pDevData, _pbBias->_pDevData);
                break;

            case Nesterov:
                /**
                 * @brief Updates biases using the Nesterov algorithm.
                 *
                 * This case corresponds to the Nesterov training mode, where the biases are updated
                 * using the kNesterovUpdateBiases kernel. The function passes the necessary parameters
                 * for the kernel to perform the bias update.
                 *
                 * @param alpha The learning rate.
                 * @param mu The momentum factor.
                 * @param batch The batch size.
                 * @param size The size of the bias array.
                 * @param delta The device memory for the delta values.
                 * @param velocity The device memory for the bias velocities.
                 * @param biases The device memory for the biases.
                 */
                kNesterovUpdateBiases(alpha, mu, batch, _localBiasSize, _outputLayer._pbDelta->_pDevData, _pbBiasVelocity->_pDevData, _pbBias->_pDevData);
                break;
                    
            case RMSProp:
                /**
                 * @brief Updates biases using the RMSProp algorithm.
                 *
                 * This case corresponds to the RMSProp training mode, where the biases are updated
                 * using the kRMSPropUpdateBiases kernel. The function passes the necessary parameters
                 * for the kernel to perform the bias update.
                 *
                 * @param alpha The learning rate.
                 * @param mu The momentum factor.
                 * @param batch The batch size.
                 * @param size The size of the bias array.
                 * @param delta The device memory for the delta values.
                 * @param velocity The device memory for the bias velocities.
                 * @param biases The device memory for the biases.
                 */
                kRMSPropUpdateBiases(alpha, mu, batch, _localBiasSize, _outputLayer._pbDelta->_pDevData, _pbBiasVelocity->_pDevData, _pbBias->_pDevData);
                break;
                            
            case AdaDelta:
                /**
                 * @brief Updates biases using the AdaDelta algorithm.
                 *
                 * This case corresponds to the AdaDelta training mode, where the biases are updated
                 * using the kAdaDeltaUpdateBiases kernel. The function passes the necessary parameters
                 * for the kernel to perform the bias update.
                 *
                 * @param mu The momentum factor.
                 * @param batch The batch size.
                 * @param size The size of the bias array.
                 * @param delta The device memory for the delta values.
                 * @param velocity The device memory for the bias velocities.
                 * @param gradientVelocity The device memory for the bias gradient velocities.
                 * @param biases The device memory for the biases.
                 */
                kAdaDeltaUpdateBiases(mu, batch, _localBiasSize, _outputLayer._pbDelta->_pDevData, _pbBiasVelocity->_pDevData, _pbBiasGradientVelocity->_pDevData, _pbBias->_pDevData);
                break;          

            case Adam:
                /**
                 * @brief Updates biases using the Adam algorithm.
                 *
                 * This case corresponds to the Adam training mode, where the biases are updated
                 * using the kAdamUpdateBiases kernel. The function passes the necessary parameters
                 * for the kernel to perform the bias update.
                 *
                 * @param alpha The learning rate.
                 * @param mu The momentum factor.
                 * @param mu1 The first moment decay rate.
                 * @param t The time step count.
                 * @param batch The batch size.
                 * @param size The size of the bias array.
                 * @param delta The device memory for the delta values.
                 * @param velocity The device memory for the bias velocities.
                 * @param gradientVelocity The device memory for the bias gradient velocities.
                 * @param biases The device memory for the biases.
                 */
                kAdamUpdateBiases(alpha, mu, mu1, t, batch, _localBiasSize, _outputLayer._pbDelta->_pDevData, _pbBiasVelocity->_pDevData, _pbBiasGradientVelocity->_pDevData, _pbBias->_pDevData);
                break; 
        }
    }
    else
    {
        switch (trainingMode)
        {
            case SGD:
                /**
                 * @brief Updates biases using the Stochastic Gradient Descent (SGD) algorithm.
                 *
                 * This case corresponds to the SGD training mode, where the biases are updated
                 * using the kSGDUpdateWeights kernel. The function passes the necessary parameters
                 * for the kernel to perform the bias update.
                 *
                 * @param alpha The learning rate.
                 * @param lambda The weight decay (set to 0 for biases).
                 * @param lambda1 The L1 regularization (set to 0 for biases).
                 * @param size The size of the bias array.
                 * @param gradient The device memory for the bias gradients.
                 * @param biases The device memory for the biases.
                 */
                kSGDUpdateWeights(alpha, (NNFloat)0.0, (NNFloat)0.0, _localBiasSize, _pbBiasGradient->_pDevData, _pbBias->_pDevData);
                break;

            case Momentum:
                /**
                 * @brief Updates biases using the Momentum algorithm.
                 *
                 * This case corresponds to the Momentum training mode, where the biases are updated
                 * using the kMomentumUpdateWeights kernel. The function passes the necessary parameters
                 * for the kernel to perform the bias update.
                 *
                 * @param alpha The learning rate.
                 * @param lambda The weight decay (set to 0 for biases).
                 * @param lambda1 The L1 regularization (set to 0 for biases).
                 * @param mu The momentum factor.
                 * @param size The size of the bias array.
                 * @param velocity The device memory for the bias velocities.
                 * @param gradient The device memory for the bias gradients.
                 * @param biases The device memory for the biases.
                 */
                kMomentumUpdateWeights(alpha, (NNFloat)0.0, (NNFloat)0.0, mu, _localBiasSize, _pbBiasVelocity->_pDevData, _pbBiasGradient->_pDevData, _pbBias->_pDevData);
                break;
                    
            case AdaGrad:
                /**
                 * @brief Updates biases using the AdaGrad algorithm.
                 *
                 * This case corresponds to the AdaGrad training mode, where the biases are updated
                 * using the kAdaGradUpdateWeights kernel. The function passes the necessary parameters
                 * for the kernel to perform the bias update.
                 *
                 * @param alpha The learning rate.
                 * @param lambda The weight decay (set to 0 for biases).
                 * @param lambda1 The L1 regularization (set to 0 for biases).
                 * @param size The size of the bias array.
                 * @param velocity The device memory for the bias velocities.
                 * @param gradient The device memory for the bias gradients.
                 * @param biases The device memory for the biases.
                 */
                kAdaGradUpdateWeights(alpha, (NNFloat)0.0, (NNFloat)0.0, _localBiasSize, _pbBiasVelocity->_pDevData, _pbBiasGradient->_pDevData, _pbBias->_pDevData);
                break;

            case Nesterov:
                /**
                 * @brief Updates biases using the Nesterov algorithm.
                 *
                 * This case corresponds to the Nesterov training mode, where the biases are updated
                 * using the kNesterovUpdateWeights kernel. The function passes the necessary parameters
                 * for the kernel to perform the bias update.
                 *
                 * @param alpha The learning rate.
                 * @param lambda The weight decay (set to 0 for biases).
                 * @param lambda1 The L1 regularization (set to 0 for biases).
                 * @param mu The momentum factor.
                 * @param size The size of the bias array.
                 * @param velocity The device memory for the bias velocities.
                 * @param gradient The device memory for the bias gradients.
                 * @param biases The device memory for the biases.
                 */
                kNesterovUpdateWeights(alpha, (NNFloat)0.0, (NNFloat)0.0, mu, _localBiasSize, _pbBiasVelocity->_pDevData, _pbBiasGradient->_pDevData, _pbBias->_pDevData);
                break;
                        
            case RMSProp:
                /**
                 * @brief Updates biases using the RMSProp algorithm.
                 *
                 * This case corresponds to the RMSProp training mode, where the biases are updated
                 * using the kRMSPropUpdateWeights kernel. The function passes the necessary parameters
                 * for the kernel to perform the bias update.
                 *
                 * @param alpha The learning rate.
                 * @param lambda The weight decay (set to 0 for biases).
                 * @param lambda1 The L1 regularization (set to 0 for biases).
                 * @param mu The momentum factor.
                 * @param size The size of the bias array.
                 * @param velocity The device memory for the bias velocities.
                 * @param gradient The device memory for the bias gradients.
                 * @param biases The device memory for the biases.
                 */
                kRMSPropUpdateWeights(alpha, (NNFloat)0.0, (NNFloat)0.0, mu, _localBiasSize, _pbBiasVelocity->_pDevData, _pbBiasGradient->_pDevData, _pbBias->_pDevData);
                break;

            case AdaDelta:
                /**
                 * @brief Updates biases using the AdaDelta algorithm.
                 *
                 * This case corresponds to the AdaDelta training mode, where the biases are updated
                 * using the kAdaDeltaUpdateWeights kernel. The function passes the necessary parameters
                 * for the kernel to perform the bias update.
                 *
                 * @param lambda The weight decay (set to 0 for biases).
                 * @param lambda1 The L1 regularization (set to 0 for biases).
                 * @param mu The momentum factor.
                 * @param size The size of the bias array.
                 * @param velocity The device memory for the bias velocities.
                 * @param gradient The device memory for the bias gradients.
                 * @param gradientVelocity The device memory for the bias gradient velocities.
                 * @param biases The device memory for the biases.
                 */
                kAdaDeltaUpdateWeights((NNFloat)0.0, (NNFloat)0.0, mu, _localBiasSize, _pbBiasVelocity->_pDevData, _pbBiasGradient->_pDevData, _pbBiasGradientVelocity->_pDevData, _pbBias->_pDevData);
                break;

            case Adam:
                /**
                 * @brief Updates biases using the Adam algorithm.
                 *
                 * This case corresponds to the Adam training mode, where the biases are updated
                 * using the kAdamUpdateWeights kernel. The function passes the necessary parameters
                 * for the kernel to perform the bias update.
                 *
                 * @param alpha The learning rate.
                 * @param lambda The weight decay (set to 0 for biases).
                 * @param lambda1 The L1 regularization (set to 0 for biases).
                 * @param mu The momentum factor.
                 * @param mu1 The first moment decay rate.
                 * @param t The time step count.
                 * @param size The size of the bias array.
                 * @param velocity The device memory for the bias velocities.
                 * @param gradient The device memory for the bias gradients.
                 * @param gradientVelocity The device memory for the bias gradient velocities.
                 * @param biases The device memory for the biases.
                 */
                kAdamUpdateWeights(alpha, (NNFloat)0.0, (NNFloat)0.0, mu, mu1, t, _localBiasSize, _pbBiasVelocity->_pDevData, _pbBiasGradient->_pDevData, _pbBiasGradientVelocity->_pDevData, _pbBias->_pDevData);
                break;                                 
        }       
    }
#if 0
        /**
         * @brief Downloads the bias data from device to host and prints it if the width is less than 1024.
         *
         * If the width is less than 1024, the bias data is downloaded from the device to the host using `_pbBias->Download()`.
         * The downloaded data is then printed using a loop that iterates over the width of the bias vector.
         */
        if (_width < 1024)
        {
            _pbBias->Download(_vBias.data());

            for (int i = 0; i < _width; i++)
            {
                printf("%3d %16.8f\n", i, _vBias[i]);
            }
        }
#endif
          
    /**
     * @brief Normalizes the weights if the norm is greater than 0 and the weights are not shared.
     *
     * If the norm is greater than 0 and the weights are not shared, the weights are normalized.
     * If the number of processes is 1, the `kNormalizeWeights` kernel is called to normalize the weights on the current GPU.
     * Otherwise, the `kCalculateWeightMagnitudes` kernel is called to calculate the weight magnitudes,
     * the magnitudes are reduced across all processes using `P2P_Allreduce`,
     * and the `kNormalizeWeightMagnitudes` kernel is called to normalize the weights based on the magnitudes.
     */
    if ((_norm > (NNFloat)0.0) && (!_bShared))
    {
        if (getGpu()._numprocs == 1)
        {
            kNormalizeWeights(_norm, _outputLayer._stride, _inputLayer._localStride, _pbWeight->_pDevData);
        }
        else
        {
            NNFloat* pMagnitude = getGpu()._pNetwork->GetScratchBuffer(_outputLayer._stride);

            kCalculateWeightMagnitudes(_outputLayer._stride, _inputLayer._localStride, _pbWeight->_pDevData, pMagnitude);
            getGpu()._pNetwork->P2P_Allreduce(pMagnitude, _outputLayer._stride);
            kNormalizeWeightMagnitudes(_norm, _outputLayer._stride, _inputLayer._localStride, _pbWeight->_pDevData, pMagnitude);
        }
    }
}
/**
 * @brief Writes the weight matrix to a netCDF file.
 *
 * @param nc The netCDF file object to write to.
 * @param index The index of the weight matrix.
 * @param pWeight Pointer to the weight matrix data.
 * @param pBias Pointer to the bias vector data.
 * @return True if successful, false otherwise.
 */
bool Weight::WriteNetCDF(netCDF::NcFile& nc, uint32_t index, NNFloat* pWeight, NNFloat* pBias)
{
    bool bResult                = true;
    if (getGpu()._id == 0)
    {
        /**
         * @brief Constructs the weight attributes and stores them in the NetCDF file.
         *
         * Constructs the weight attributes based on the weight index and stores them in the NetCDF file.
         * The attributes include information about input layer, output layer, width, height, length, depth, breadth,
         * shared flag, locked flag, normalization, bias dimensions, and bias data.
         * If the weight is shared, additional attributes for transposed flag, source input layer, and source output layer are stored.
         *
         * @param index The index of the weight.
         * @param nc The NetCDF file object.
         */
        string wstring = "weight" + std::to_string(index) + "_";

        // Store attributes related to input layer, output layer, width, height, length, depth, and breadth
        nc.putAtt(wstring + "inputLayer", _inputLayer._name);
        nc.putAtt(wstring + "outputLayer", _outputLayer._name);
        nc.putAtt(wstring + "width", ncUint64, (unsigned long long int)_width);
        nc.putAtt(wstring + "height", ncUint64, (unsigned long long int)_height);
        nc.putAtt(wstring + "length", ncUint64, (unsigned long long int)_length);
        nc.putAtt(wstring + "depth", ncUint64, (unsigned long long int)_depth);
        nc.putAtt(wstring + "breadth", ncUint64, (unsigned long long int)_breadth);

        // Store attributes related to shared flag, locked flag, and normalization
        nc.putAtt(wstring + "bShared", ncUint, (uint32_t)_bShared);
        nc.putAtt(wstring + "bLocked", ncUint, (uint32_t)_bLocked);
        nc.putAtt(wstring + "norm", ncFloat, _norm);

        // Create a dimension for bias and store the bias data
        NcDim biasDim = nc.addDim(wstring + "biasDim", _biasSize);
        NcVar biasVar = nc.addVar(wstring + "bias", "float", biasDim.getName());
        if (pBias == NULL)
            pBias = _vBias.data();
        biasVar.putVar(pBias);

        if (_bShared)
        {
            // Store additional attributes related to shared weight
            nc.putAtt(wstring + "bTransposed", ncUint, (uint32_t)_bTransposed);
            nc.putAtt(wstring + "sourceInputLayer", _pSharedWeight->_inputLayer._name);
            nc.putAtt(wstring + "sourceOutputLayer", _pSharedWeight->_outputLayer._name);
        }
        else
        {

#if 0
        /**
         * Print weights and biases.
         *
         * @param index The index value.
         */
        std::printf("Weights %d %lu %lu\n", index, _vWeight.size(), _vBias.size());

        /**
         * Print the first 20 elements of weights and biases.
         */
        for (int i = 0; i < 20; i++)
            printf("%3d %16.8f %16.8f\n", i, _vWeight[i], _vBias[i]);

#endif
            /**
             * Add weight dimension and variable to the NetCDF file.
             *
             * @param wstring A string used to create the weight dimension and variable names.
             * @param _size The size of the weight dimension.
             * @return The weight dimension created.
             */
            NcDim weightDim = nc.addDim(wstring + "weightDim", _size);

            /**
             * Add weight variable to the NetCDF file.
             *
             * @param wstring A string used to create the weight dimension and variable names.
             * @param "float" The data type of the weight variable.
             * @param weightDim.getName() The name of the weight dimension.
             * @return The weight variable created.
             */
            NcVar weightVar = nc.addVar(wstring + "weights", "float", weightDim.getName());

            /**
             * Put the weight data into the weight variable.
             *
             * @param pWeight Pointer to the weight data.
             */
            if (!pWeight)
                pWeight = _vWeight.data();
            weightVar.putVar(pWeight);
        }
    }

    return bResult;
}
/**
 * @brief Copies the weights from another weight object.
 *
 * @param pSrcWeight Pointer to the source weight object.
 * @return True if successful, false otherwise.
 */
bool Weight::CopyWeights(const Weight* pSrcWeight)
{
    /**
     * Flag indicating the validity of weight copying operation.
     */
    bool bValid = true;

    /**
     * Pointer to the destination Weight object.
     * If _bShared is true, it points to _pSharedWeight, otherwise it points to 'this'.
     */
    Weight* pDstWeight = _bShared ? _pSharedWeight : this;

    /**
     * Check if the source Weight object is valid.
     *
     * @param pSrcWeight Pointer to the source Weight object.
     * @return False if the source Weight object is invalid, otherwise true.
     */
    if (!pSrcWeight)
    {
        if (getGpu()._id == 0)
            printf("Weight::CopyWeights: Invalid weight pointer.\n");
        return false;
    }

    /**
     * Update the source Weight object to its shared weight if _bShared is true.
     *
     * @param pSrcWeight Pointer to the source Weight object.
     */
    pSrcWeight = _bShared ? pSrcWeight->_pSharedWeight : pSrcWeight;

    /**
     * Check if the dimensions of the source and destination Weight objects match.
     * If the dimensions don't match, set 'bValid' to false.
     *
     * @param pSrcWeight Pointer to the source Weight object.
     * @param pDstWeight Pointer to the destination Weight object.
     */
    if ((pSrcWeight->_width != pDstWeight->_width) || (pSrcWeight->_height != pDstWeight->_height) || (pSrcWeight->_length != pDstWeight->_length))
    {
        if (getGpu()._id == 0)
        {
            std::printf("Weight::CopyWeights: Mismatched weight dimensions (%" PRIu64 " x %" PRIu64 " x %" PRIu64") versus (%" PRIu64 " x %" PRIu64 " x %" PRIu64 ").\n",
                pDstWeight->_width, pDstWeight->_height, pDstWeight->_length, pSrcWeight->_width, pSrcWeight->_height, pSrcWeight->_length);
        }
        bValid = false;
    }
    else
    {
        /**
         * Copy the weight and bias values from the source Weight object to the destination Weight object.
         *
         * @param pDstWeight Pointer to the destination Weight object.
         * @param pSrcWeight Pointer to the source Weight object.
         */
        pDstWeight->_vWeight = pSrcWeight->_vWeight;
        _vBias = pSrcWeight->_vBias;

        /**
         * Upload the weight data to the GPU memory if the destination Weight object has a valid weight buffer.
         *
         * @param pDstWeight Pointer to the destination Weight object.
         */
        if (pDstWeight->_pbWeight)
            pDstWeight->_pbWeight->Upload(pDstWeight->_vWeight.data());

        /**
         * Upload the bias data to the GPU memory if the bias buffer is valid.
         */
        if (_pbBias)
            _pbBias->Upload(_vBias.data());
    }
    return bValid;
}
/**
 * @brief Sets the weights of the weight matrix.
 *
 * @param vWeight Vector containing the weights.
 * @return True if successful, false otherwise.
 */
bool Weight::SetWeights(const vector<NNFloat>& vWeight)
{
    bool bValid                 = true;
    Weight* pWeight = _bShared ? _pSharedWeight : this;
    
    if (getGpu()._numprocs == 1)
    {
        if (vWeight.size() < pWeight->_vWeight.size())
        {
            if (getGpu()._id == 0)
            {
                printf("Weight::SetWeights: Input vector smaller than weight vector.\n");
            }
            bValid                  = false;        
        }
        else
        {
            if (vWeight.size() > pWeight->_vWeight.size())
                std::copy(vWeight.data(), vWeight.data() + pWeight->_vWeight.size(), pWeight->_vWeight.data());
            else
                pWeight->_vWeight       = vWeight;
            if (pWeight->_pbWeight != NULL)
                pWeight->_pbWeight->Upload(_vWeight.data());
        }
    }
    else
    {
        
    }
    return bValid;
}
/**
 * @brief Sets the biases of the weight matrix.
 *
 * @param vBias Vector containing the biases.
 * @return True if successful, false otherwise.
 */
bool Weight::SetBiases(const vector<NNFloat>& vBias)
{
    bool bValid                 = true;

    if (vBias.size() < _vBias.size())
    {
        if (getGpu()._id == 0)
        {
            printf("Weight::SetBiases: Input vector smaller than bias vector.\n");
        }
        bValid                  = false;        
    }
    else
    {
        if (vBias.size() > _vBias.size())
            std::copy(vBias.data(), vBias.data() + _vBias.size(), _vBias.data());
        else
            _vBias       = vBias;
        if (_pbBias != NULL)
            _pbBias->Upload(_vBias.data());
    }
    return bValid;
}
/**
 * @brief Retrieves the weights of the weight matrix.
 *
 * @param vWeight Vector to store the weights.
 * @return True if successful, false otherwise.
 */
bool Weight::GetWeights(vector<NNFloat>& vWeight)
{
    bool bValid                 = true;

    if (vWeight.size() < _vWeight.size())
    {
        vWeight.resize(_vWeight.size());
    }

    if (_pbWeight != NULL)
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
 * @brief Retrieves the biases of the weight matrix.
 *
 * @param vBias Vector to store the biases.
 * @return True if successful, false otherwise.
 */
bool Weight::GetBiases(vector<NNFloat>& vBias)
{
    bool bValid                 = true;

    if (getGpu()._numprocs == 1)
    {

        if (vBias.size() < _vBias.size())
        {
            vBias.resize(_vBias.size());
        }

        if (_pbBias != NULL)
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
 * @brief Retrieves the dimensions of the weight matrix.
 *
 * @param dimensions Vector to store the dimensions.
 * @return True if successful, false otherwise.
 */
bool Weight::GetDimensions(vector<uint64_t>& dimensions)
{
  if (_dimensionality < 2 || _dimensionality > 5) {
      printf("Weight::GetDimensions: _dimensionality = %u\n", _dimensionality);
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
 * @brief Dumps the weight matrix to a file.
 *
 * @param fname The name of the file to save the weight matrix.
 * @param pBuffer Pointer to the weight matrix data.
 */
void Weight::Dump(string fname, NNFloat* pBuffer)
{
    vector<NNFloat> vWeight;

    if (getGpu()._numprocs == 1)
    {
        vWeight.resize(_localSize);
        cudaMemcpy(vWeight.data(), pBuffer, _localSize * sizeof(NNFloat), cudaMemcpyDefault);
    }
    else
    {
        if (getGpu()._id == 0)
            vWeight.resize(_outputLayer._stride * _inputLayer._stride);        
        uint32_t outgoingSize       = _outputLayer._stride * 3;               
        uint32_t incomingSize       = _inputLayer._stride * 2;     
        cudaMemcpy(_vWeight.data(), pBuffer, _localSize * sizeof(NNFloat), cudaMemcpyDefault);

        if (getGpu()._id == 0)
        {
            NNFloat* pWeight            = vWeight.data();                    
            if (outgoingSize > incomingSize)
            {
                cudaMemcpy2D(pWeight, _outputLayer._stride * sizeof(NNFloat), _vWeight.data(), _outputLayer._localStride * sizeof(NNFloat), _outputLayer._localStride * sizeof(NNFloat), _inputLayer._stride, cudaMemcpyDefault);
                pWeight                += _outputLayer._localStride;
                for (uint32_t i = 1; i < getGpu()._numprocs; i++)
                {                        
                    uint64_t size;
                    MPI_Status status;                
                    MPI_Recv(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
                    vector<NNFloat> vTemp(size);
                    MPI_Recv(vTemp.data(), size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
                    uint64_t lstride    = size / _inputLayer._stride;
                    NNFloat* pSrcWeight = vTemp.data();
                    NNFloat* pDstWeight = pWeight;
                    for (uint32_t j = 0; j < _inputLayer._stride; j++)
                    {
                        memcpy(pDstWeight, pSrcWeight, lstride * sizeof(NNFloat));
                        pSrcWeight     += lstride;
                        pDstWeight     += _outputLayer._stride;
                    }                          
                    pWeight            += lstride;
                }
            }
            else
            {
                cudaMemcpy(pWeight, _vWeight.data(), _outputLayer._stride * _inputLayer._localStride * sizeof(NNFloat), cudaMemcpyDefault);
                pWeight                += _outputLayer._stride * _inputLayer._localStride;
                for (uint32_t i = 1; i < getGpu()._numprocs; i++)
                {
                    uint64_t size;
                    MPI_Status status;                
                    MPI_Recv(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
                    MPI_Recv(pWeight, size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
                    pWeight            += size;
                }                        
            }
        }              
        else
        {
            uint64_t size               = _vWeight.size();
            MPI_Send(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD);
            MPI_Send(_vWeight.data(), size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);                  
        }

    }

    if (getGpu()._id == 0)
    {
        FILE* fp                        = fopen(fname.c_str(), "w");
        NNFloat* pData                  = vWeight.data();
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
