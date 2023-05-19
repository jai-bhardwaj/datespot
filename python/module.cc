#include "module.h"
#include "calculate.h"
#include "CDLAccessors.h"
#include "utilities.h"
#include "NetworkFunctions.h"
#include "NetworkAccessors.h"
#include "LayerAccessors.h"
#include "WeightAccessors.h"
#include "DataSetAccessors.h"
#include "utilities.h"

static PyMethodDef tensorhubMethods[] = {
    
    {"GetCDLRandomSeed", CDLAccessors::GetRandomSeed, METH_VARARGS,
    /**
     * @brief Get random seed from the source CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Get random seed from the source CDL"},

    {"SetCDLRandomSeed", CDLAccessors::SetRandomSeed, METH_VARARGS,
    /**
     * @brief Set random seed in the destination CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set random seed in the destination CDL"},

    {"GetCDLEpochs", CDLAccessors::GetEpochs, METH_VARARGS,
    /**
     * @brief Get epochs from the source CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Get epochs from the source CDL"},

    {"SetCDLEpochs", CDLAccessors::SetEpochs, METH_VARARGS,
    /**
     * @brief Set epochs in the destination CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set epochs in the destination CDL"},
    
    {"GetCDLBatch", CDLAccessors::GetBatch, METH_VARARGS,
    /**
     * @brief Get batch from the source CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Get batch from the source CDL"},

    {"SetCDLBatch", CDLAccessors::SetBatch, METH_VARARGS,
    /**
     * @brief Set batch in the destination CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set batch in the destination CDL"},

    {"GetCDLCheckpointInterval", CDLAccessors::GetCheckpointInterval, METH_VARARGS,
    /**
     * @brief Get checkpoint interval from the source CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Get checkpoint interval from the source CDL"},

    {"SetCDLCheckpointInterval", CDLAccessors::SetCheckpointInterval, METH_VARARGS,
    /**
     * @brief Set checkpoint interval in the destination CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set checkpoint interval in the destination CDL"},
    
    {"GetCDLAlphaInterval", CDLAccessors::GetAlphaInterval, METH_VARARGS,
    /**
     * @brief Get alpha interval from the source CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Get alpha interval from the source CDL"},

    {"SetCDLAlphaInterval", CDLAccessors::SetAlphaInterval, METH_VARARGS,
    /**
     * @brief Set alpha interval in the destination CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set alpha interval in the destination CDL"},

    {"GetCDLShuffleIndexes", CDLAccessors::GetShuffleIndexes, METH_VARARGS,
    /**
     * @brief Get shuffle indexes from the source CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Get shuffle indexes from the source CDL"},

    {"SetCDLShuffleIndexes", CDLAccessors::SetShuffleIndexes, METH_VARARGS,
    /**
     * @brief Set shuffle indexes in the destination CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set shuffle indexes in the destination CDL"},

    {"GetCDLAlpha", CDLAccessors::GetAlpha, METH_VARARGS,
    /**
     * @brief Get alpha from the source CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Get alpha from the source CDL"},

    {"SetCDLAlpha", CDLAccessors::SetAlpha, METH_VARARGS,
    /**
     * @brief Set alpha in the destination CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set alpha in the destination CDL"},
    
    {"GetCDLLambda", CDLAccessors::GetLambda, METH_VARARGS,
    /**
     * @brief Get lambda from the source CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Get lambda from the source CDL"},

    {"SetCDLLambda", CDLAccessors::SetLambda, METH_VARARGS,
    /**
     * @brief Set lambda in the destination CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set lambda in the destination CDL"},

    {"GetCDLMu", CDLAccessors::GetMu, METH_VARARGS,
    /**
     * @brief Get mu from the source CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Get mu from the source CDL"},

    {"SetCDLMu", CDLAccessors::SetMu, METH_VARARGS,
    /**
     * @brief Set mu in the destination CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set mu in the destination CDL"},

    {"GetCDLAlphaMultiplier", CDLAccessors::GetAlphaMultiplier, METH_VARARGS,
    /**
     * @brief Get alpha multiplier from the source CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Get alpha multiplier from the source CDL"},

    {"SetCDLAlphaMultiplier", CDLAccessors::SetAlphaMultiplier, METH_VARARGS,
    /**
     * @brief Set alpha multiplier in the destination CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set alpha multiplier in the destination CDL"},
    
    {"GetCDLMode", CDLAccessors::GetMode, METH_VARARGS,
    /**
     * @brief Get the mode enumerator from the source CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Get the mode enumerator from the source CDL"},

    {"SetCDLMode", CDLAccessors::SetMode, METH_VARARGS,
    /**
     * @brief Set the mode enumerator in the destination CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set the mode enumerator in the destination CDL"},

    {"GetCDLOptimizer", CDLAccessors::GetOptimizer, METH_VARARGS,
    /**
     * @brief Get the training mode enumerator from the source CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Get the training mode enumerator from the source CDL"},

    {"SetCDLOptimizer", CDLAccessors::SetOptimizer, METH_VARARGS,
    /**
     * @brief Set the training mode enumerator in the destination CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set the training mode enumerator in the destination CDL"},

    {"GetCDLNetworkFileName", CDLAccessors::GetNetworkFileName, METH_VARARGS,
    /**
     * @brief Get network filename from the source CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Get network filename from the source CDL"},
    
    {"SetCDLNetworkFileName", CDLAccessors::SetNetworkFileName, METH_VARARGS,
    /**
     * @brief Set network filename in the destination CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set network filename in the destination CDL"},

    {"GetCDLDataFileName", CDLAccessors::GetDataFileName, METH_VARARGS,
    /**
     * @brief Get data filename from the source CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Get data filename from the source CDL"},

    {"SetCDLDataFileName", CDLAccessors::SetDataFileName, METH_VARARGS,
    /**
     * @brief Set data filename in the destination CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set data filename in the destination CDL"},

    {"GetCDLCheckpointFileName", CDLAccessors::GetCheckpointFileName, METH_VARARGS,
    /**
     * @brief Get checkpoint filename from the source CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Get checkpoint filename from the source CDL"},

    {"SetCDLCheckpointFileName", CDLAccessors::SetCheckpointFileName, METH_VARARGS,
    /**
     * @brief Set checkpoint filename in the destination CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set checkpoint filename in the destination CDL"},
    
    {"GetCDLResultsFileName", CDLAccessors::GetResultsFileName, METH_VARARGS,
    /**
     * @brief Get results filename from the source CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Get results filename from the source CDL"},

    {"SetCDLResultsFileName", CDLAccessors::SetResultsFileName, METH_VARARGS,
    /**
     * @brief Set results filename in the destination CDL.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set results filename in the destination CDL"},

    {"Startup", Utilities::Startup, METH_VARARGS,
    /**
     * @brief Initialize the GPUs and MPI.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Initialize the GPUs and MPI"},

    {"Shutdown", Utilities::Shutdown, METH_VARARGS,
    /**
     * @brief Shutdown the GPUs.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Shutdown the GPUs"},

    {"CreateCDLFromJSON", Utilities::CreateCDLFromJSON, METH_VARARGS,
    /**
     * @brief Create a CDL instance and initialize it from a JSON file.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Create a CDL instance and initialize it from a JSON file"},

    {"CreateCDLFromDefaults", Utilities::CreateCDLFromDefaults, METH_VARARGS,
    /**
     * @brief Create a CDL instance and initialize it with default values.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Create a CDL instance and initialize it with default values"},
    
    {"DeleteCDL", Utilities::DeleteCDL, METH_VARARGS,
    /**
     * @brief Delete a CDL instance.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Delete a CDL instance"},

    {"LoadNetCDF", Utilities::LoadDataSetFromNetCDF, METH_VARARGS,
    /**
     * @brief Load a Python array (i.e. a list) of data sets from a CDF file.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Load a Python array (i.e. a list) of data sets from a CDF file"},

    {"DeleteDataSet", Utilities::DeleteDataSet, METH_VARARGS,
    /**
     * @brief Delete a data set.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Delete a data set"},

    {"LoadNeuralNetworkNetCDF", Utilities::LoadNeuralNetworkFromNetCDF, METH_VARARGS,
    /**
     * @brief Load a neural network from a CDF file.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Load a neural network from a CDF file"},

    {"LoadNeuralNetworkJSON", Utilities::LoadNeuralNetworkFromJSON, METH_VARARGS,
    /**
     * @brief Load a neural network from a JSON config file, a batch number, and a Python list of data sets.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Load a neural network from a JSON config file, a batch number, and a Python list of data sets"},
    
    {"DeleteNetwork", Utilities::DeleteNetwork, METH_VARARGS,
    /**
     * @brief Delete a neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Delete a neural network"},

    {"OpenFile", Utilities::OpenFile, METH_VARARGS,
    /**
     * @brief Open a FILE* stream.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Open a FILE* stream"},

    {"CloseFile", Utilities::CloseFile, METH_VARARGS,
    /**
     * @brief Close a FILE* stream.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Close a FILE* stream"},

    {"SetRandomSeed", Utilities::SetRandomSeed, METH_VARARGS,
    /**
     * @brief Set random seed in the GPU.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set random seed in the GPU"},

    {"GetMemoryUsage", Utilities::GetMemoryUsage, METH_VARARGS,
    /**
     * @brief Get the GPU and CPU memory usage.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Get the GPU and CPU memory usage"},

    {"Transpose", Utilities::Transpose, METH_VARARGS,
    /**
     * @brief Transpose a NumPy 2D matrix to create a contiguous matrix.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Transpose a NumPy 2D matrix to create a contiguous matrix"},
    
    {"CreateFloatGpuBuffer", Utilities::CreateFloatGpuBuffer, METH_VARARGS,
    /**
     * @brief Create a GPU Buffer of type Float and of the specified size.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Create a GPU Buffer of type Float and of the specified size"},

    {"DeleteFloatGpuBuffer", Utilities::DeleteFloatGpuBuffer, METH_VARARGS,
    /**
     * @brief Delete a GPU Buffer of type Float.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Delete a GPU Buffer of type Float"},

    {"CreateUnsignedGpuBuffer", Utilities::CreateUnsignedGpuBuffer, METH_VARARGS,
    /**
     * @brief Create a GPU Buffer of type uint32_t and of the specified size.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Create a GPU Buffer of type uint32_t and of the specified size"},

    {"DeleteUnsignedGpuBuffer", Utilities::DeleteUnsignedGpuBuffer, METH_VARARGS,
    /**
     * @brief Delete a GPU Buffer of type uint32_t.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Delete a GPU Buffer of type uint32_t"},

    {"ClearDataSets", NetworkFunctions::ClearDataSets, METH_VARARGS,
    /**
     * @brief Clear the data sets from the neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Clear the data sets from the neural network"},

    {"LoadDataSets", NetworkFunctions::LoadDataSets, METH_VARARGS,
    /**
     * @brief Load a Python list of data sets into the neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Load a Python list of data sets into the neural network"},

    {"Randomize", NetworkFunctions::Randomize, METH_VARARGS,
    /**
     * @brief Randomize a neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Randomize a neural network"},
    
    {"Validate", NetworkFunctions::Validate, METH_VARARGS,
    /**
     * @brief Validate the network gradients numerically for the neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Validate the network gradients numerically for the neural network"},

    {"Train", NetworkFunctions::Train, METH_VARARGS,
    /**
     * @brief Train the neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Train the neural network"},

    {"PredictBatch", NetworkFunctions::PredictBatch, METH_VARARGS,
    /**
     * @brief Predict batch for the neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Predict batch for the neural network"},

    {"CalculateOutput", NetworkFunctions::CalculateOutput, METH_VARARGS,
    /**
     * @brief Calculate the top K results for the neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Calculate the top K results for the neural network"},

    {"PredictOutput", NetworkFunctions::PredictOutput, METH_VARARGS,
    /**
     * @brief Do a prediction calculation and return the top K results for a neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Do a prediction calculation and return the top K results for a neural network"},

    {"CalculateMRR", NetworkFunctions::CalculateMRR, METH_VARARGS,
    /**
     * @brief Calculate the MRR for a neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Calculate the MRR for a neural network"},
    
    {"SaveBatch", NetworkFunctions::SaveBatch, METH_VARARGS,
    /**
     * @brief Save the batch to a file for the neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Save the batch to a file for the neural network"},

    {"DumpBatch", NetworkFunctions::DumpBatch, METH_VARARGS,
    /**
     * @brief Dump the batch to a FILE for the neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Dump the batch to a FILE for the neural network"},

    {"SaveLayer", NetworkFunctions::SaveLayer, METH_VARARGS,
    /**
     * @brief Save the specified layer to a file for the neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Save the specified layer to a file for the neural network"},

    {"DumpLayer", NetworkFunctions::DumpLayer, METH_VARARGS,
    /**
     * @brief Dump the specified layer to a FILE for the neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Dump the specified layer to a FILE for the neural network"},

    {"SaveWeights", NetworkFunctions::SaveWeights, METH_VARARGS,
    /**
     * @brief Save the weights connecting two layers to a file for the neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Save the weights connecting two layers to a file for the neural network"},
    
    {"LockWeights", NetworkFunctions::LockWeights, METH_VARARGS,
    /**
     * @brief Lock the weights connecting two layers for the neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Lock the weights connecting two layers for the neural network"},

    {"UnlockWeights", NetworkFunctions::UnlockWeights, METH_VARARGS,
    /**
     * @brief Unlock the weights connecting two layers for the neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Unlock the weights connecting two layers for the neural network"},

    {"SaveNetCDF", NetworkFunctions::SaveNetCDF, METH_VARARGS,
    /**
     * @brief Save training results to a CDF file for the neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Save training results to a CDF file for the neural network"},

    {"P2P_Bcast", NetworkFunctions::P2P_Bcast, METH_VARARGS,
    /**
     * @brief Broadcast data from process 0 to all other processes for the neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Broadcast data from process 0 to all other processes for the neural network"},

    {"P2P_Allreduce", NetworkFunctions::P2P_Allreduce, METH_VARARGS,
    /**
     * @brief Reduce a buffer across all processes for the neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Reduce a buffer across all processes for the neural network"},

    {"GetBatch", NetworkAccessors::GetBatch, METH_VARARGS,
    /**
     * @brief Get the batch from the source neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The batch from the source neural network.
     */
    "Get the batch from the source neural network"},
    
    {"SetBatch", NetworkAccessors::SetBatch, METH_VARARGS,
    /**
     * @brief Set the batch in the destination neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set the batch in the destination neural network"},

    {"GetPosition", NetworkAccessors::GetPosition, METH_VARARGS,
    /**
     * @brief Get the position from the source neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The position from the source neural network.
     */
    "Get the position from the source neural network"},

    {"SetPosition", NetworkAccessors::SetPosition, METH_VARARGS,
    /**
     * @brief Set the position in the destination neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set the position in the destination neural network"},

    {"GetShuffleIndices", NetworkAccessors::GetShuffleIndices, METH_VARARGS,
    /**
     * @brief Get the shuffle indices from the source neural network; unwrap the unnecessary tuple.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The shuffle indices from the source neural network.
     */
    "Get the shuffle indices from the source neural network; unwrap the unnecessary tuple"},

    {"SetShuffleIndices", NetworkAccessors::SetShuffleIndices, METH_VARARGS,
    /**
     * @brief Set the shuffle indices boolean in the destination neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set the shuffle indices boolean in the destination neural network"},

    {"GetSparsenessPenalty", NetworkAccessors::GetSparsenessPenalty, METH_VARARGS,
    /**
     * @brief Get the sparseness penalty p and beta from the source neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The sparseness penalty p and beta from the source neural network.
     */
    "Get the sparseness penalty p and beta from the source neural network"},
    
    {"SetSparsenessPenalty", NetworkAccessors::SetSparsenessPenalty, METH_VARARGS,
    /**
     * @brief Set the sparseness penalty p and beta in the destination neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set the sparseness penalty p and beta in the destination neural network"},

    {"GetDenoising", NetworkAccessors::GetDenoising, METH_VARARGS,
    /**
     * @brief Get the denoising p from the source neural network and unwrap the unnecessary tuple.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The denoising p from the source neural network.
     */
    "Get the denoising p from the source neural network and unwrap the unnecessary tuple"},

    {"SetDenoising", NetworkAccessors::SetDenoising, METH_VARARGS,
    /**
     * @brief Set the denoising p in the destination neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set the denoising p in the destination neural network"},

    {"GetDeltaBoost", NetworkAccessors::GetDeltaBoost, METH_VARARGS,
    /**
     * @brief Get the delta boost one and zero from the source neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The delta boost one and zero from the source neural network.
     */
    "Get the delta boost one and zero from the source neural network"},

    {"SetDeltaBoost", NetworkAccessors::SetDeltaBoost, METH_VARARGS,
    /**
     * @brief Set the delta boost one and zero in the destination neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set the delta boost one and zero in the destination neural network"},

    {"GetDebugLevel", NetworkAccessors::GetDebugLevel, METH_VARARGS,
    /**
     * @brief Get the debug level from the source neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The debug level from the source neural network.
     */
    "Get the debug level from the source neural network"},
    
    {"SetDebugLevel", NetworkAccessors::SetDebugLevel, METH_VARARGS,
    /**
     * @brief Set the debug level in the destination neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set the debug level in the destination neural network"},

    {"GetCheckpoint", NetworkAccessors::GetCheckpoint, METH_VARARGS,
    /**
     * @brief Get the checkpoint file name and interval from the source neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The checkpoint file name and interval from the source neural network.
     */
    "Get the checkpoint file name and interval from the source neural network"},

    {"SetCheckpoint", NetworkAccessors::SetCheckpoint, METH_VARARGS,
    /**
     * @brief Set the checkpoint filename and interval in the destination neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set the checkpoint filename and interval in the destination neural network"},

    {"GetLRN", NetworkAccessors::GetLRN, METH_VARARGS,
    /**
     * @brief Get the local response network (LRN) k, n, alpha, and beta from the source neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The local response network (LRN) k, n, alpha, and beta from the source neural network.
     */
    "Get the local response network (LRN) k, n, alpha, and beta from the source neural network"},

    {"SetLRN", NetworkAccessors::SetLRN, METH_VARARGS,
    /**
     * @brief Set the local response network (LRN) k, n, alpha, and beta in the destination neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set the local response network (LRN) k, n, alpha, and beta in the destination neural network"},

    {"GetSMCE", NetworkAccessors::GetSMCE, METH_VARARGS,
    /**
     * @brief Get the scaled marginal cross entropy (SMCE) oneTarget, zeroTarget, oneScale, and zeroScale from the source neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The scaled marginal cross entropy (SMCE) oneTarget, zeroTarget, oneScale, and zeroScale from the source neural network.
     */
    "Get the scaled marginal cross entropy (SMCE) oneTarget, zeroTarget, oneScale, and zeroScale from the source neural network"},

    {"SetSMCE", NetworkAccessors::SetSMCE, METH_VARARGS,
    /**
     * @brief Set the scaled marginal cross entropy (SMCE) oneTarget, zeroTarget, oneScale, and zeroScale in the destination neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set the scaled marginal cross entropy (SMCE) oneTarget, zeroTarget, oneScale, and zeroScale in the destination neural network"},
    
    {"GetMaxout", NetworkAccessors::GetMaxout, METH_VARARGS,
    /**
     * @brief Get the maxout k from the source neural network and unwrap the unnecessary tuple.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The maxout k from the source neural network.
     */
    "Get the maxout k from the source neural network and unwrap the unnecessary tuple"},

    {"SetMaxout", NetworkAccessors::SetMaxout, METH_VARARGS,
    /**
     * @brief Set the maxout k in the destination neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set the maxout k in the destination neural network"},

    {"GetExamples", NetworkAccessors::GetExamples, METH_VARARGS,
    /**
     * @brief Get the examples from the source neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The examples from the source neural network.
     */
    "Get the examples from the source neural network"},

    {"GetWeight", NetworkAccessors::GetWeight, METH_VARARGS,
    /**
     * @brief Get the set of weights connecting the specified input and output layers from the source neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The set of weights connecting the specified input and output layers from the source neural network.
     */
    "Get the set of weights connecting the specified input and output layers from the source neural network"},

    {"GetBufferSize", NetworkAccessors::GetBufferSize, METH_VARARGS,
    /**
     * @brief Get the buffer size of the specified layer for the source neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The buffer size of the specified layer for the source neural network.
     */
    "Get the buffer size of the specified layer for the source neural network"},

    {"GetLayer", NetworkAccessors::GetLayer, METH_VARARGS,
    /**
     * @brief Get the specified layer from the source neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The specified layer from the source neural network.
     */
    "Get the specified layer from the source neural network"},

    {"GetLayers", NetworkAccessors::GetLayers, METH_VARARGS,
    /**
     * @brief Get the list of layer names from the source neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The list of layer names from the source neural network.
     */
    "Get the list of layer names from the source neural network"},
    
    {"GetName", NetworkAccessors::GetName, METH_VARARGS,
    /**
     * @brief Get the name of the source neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The name of the source neural network.
     */
    "Get the name of the source neural network"},

    {"GetUnitBuffer", NetworkAccessors::GetUnitBuffer, METH_VARARGS,
    /**
     * @brief Get the unit buffer for the specified layer from the source neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The unit buffer for the specified layer from the source neural network.
     */
    "Get the unit buffer for the specified layer from the source neural network"},

    {"GetDeltaBuffer", NetworkAccessors::GetDeltaBuffer, METH_VARARGS,
    /**
     * @brief Get the delta buffer for the specified layer from the source neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The delta buffer for the specified layer from the source neural network.
     */
    "Get the delta buffer for the specified layer from the source neural network"},

    {"GetWeightBuffer", NetworkAccessors::GetWeightBuffer, METH_VARARGS,
    /**
     * @brief Get the weight buffer for the specified input and output layers from the source neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The weight buffer for the specified input and output layers from the source neural network.
     */
    "Get the weight buffer for the specified input and output layers from the source neural network"},

    {"GetScratchBuffer", NetworkAccessors::GetScratchBuffer, METH_VARARGS,
    /**
     * @brief Get the current scratch buffer from the source neural network and resize it if necessary.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The current scratch buffer from the source neural network.
     */
    "Get the current scratch buffer from the source neural network and resize it if necessary"},

    {"GetP2PSendBuffer", NetworkAccessors::GetP2PSendBuffer, METH_VARARGS,
    /**
     * @brief Get the current local send buffer from the source neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The current local send buffer from the source neural network.
     */
    "Get the current local send buffer from the source neural network"},

    {"GetP2PReceiveBuffer", NetworkAccessors::GetP2PReceiveBuffer, METH_VARARGS,
    /**
     * @brief Get the current local receive buffer from the source neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The current local receive buffer from the source neural network.
     */
    "Get the current local receive buffer from the source neural network"},
    
    {"GetP2PCPUBuffer", NetworkAccessors::GetP2PCPUBuffer, METH_VARARGS,
    /**
     * @brief Get the system memory work buffer from the source neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The system memory work buffer from the source neural network.
     */
    "Get the system memory work buffer from the source neural network"},

    {"GetPeerBuffer", NetworkAccessors::GetPeerBuffer, METH_VARARGS,
    /**
     * @brief Get the current adjacent peer receive buffer from the source neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The current adjacent peer receive buffer from the source neural network.
     */
    "Get the current adjacent peer receive buffer from the source neural network"},

    {"GetPeerBackBuffer", NetworkAccessors::GetPeerBackBuffer, METH_VARARGS,
    /**
     * @brief Get the current adjacent peer send buffer from the source neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The current adjacent peer send buffer from the source neural network.
     */
    "Get the current adjacent peer send buffer from the source neural network"},

    {"SetClearVelocity", NetworkAccessors::SetClearVelocity, METH_VARARGS,
    /**
     * @brief Set the clear velocity flag in the destination neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set the clear velocity flag in the destination neural network"},

    {"SetTrainingMode", NetworkAccessors::SetTrainingMode, METH_VARARGS,
    /**
     * @brief Set the training mode enumerator in the destination neural network.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set the training mode enumerator in the destination neural network"},

    {"GetLayerName", LayerAccessors::GetName, METH_VARARGS,
    /**
     * @brief Get the name from the source layer.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The name from the source layer.
     */
    "Get the name from the source layer"},

    {"GetKind", LayerAccessors::GetKind, METH_VARARGS,
    /**
     * @brief Get the kind enumerator from the source layer.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The kind enumerator from the source layer.
     */
    "Get the kind enumerator from the source layer"},
    
    {"GetType", LayerAccessors::GetType, METH_VARARGS,
    /**
     * @brief Get the type enumerator from the source layer.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The type enumerator from the source layer.
     */
    "Get the type enumerator from the source layer"},

    {"GetAttributes", LayerAccessors::GetAttributes, METH_VARARGS,
    /**
     * @brief Get the attributes from the source layer.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The attributes from the source layer.
     */
    "Get the attributes from the source layer"},

    {"GetDataSetBase", LayerAccessors::GetDataSet, METH_VARARGS,
    /**
     * @brief Get the data set from the source layer.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The data set from the source layer.
     */
    "Get the data set from the source layer"},

    {"GetNumDimensions", LayerAccessors::GetNumDimensions, METH_VARARGS,
    /**
     * @brief Get the number of dimensions from the source layer.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The number of dimensions from the source layer.
     */
    "Get the number of dimensions from the source layer"},

    {"GetDimensions", LayerAccessors::GetDimensions, METH_VARARGS,
    /**
     * @brief Get dimensions from the source layer.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The dimensions from the source layer.
     */
    "Get dimensions from the source layer"},

    {"GetLocalDimensions", LayerAccessors::GetLocalDimensions, METH_VARARGS,
    /**
     * @brief Get local dimensions from the source layer.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The local dimensions from the source layer.
     */
    "Get local dimensions from the source layer"},
    
    {"GetKernelDimensions", LayerAccessors::GetKernelDimensions, METH_VARARGS,
    /**
     * @brief Get kernel dimensions from the source layer.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The kernel dimensions from the source layer.
     */
    "Get kernel dimensions from the source layer"},

    {"GetKernelStride", LayerAccessors::GetKernelStride, METH_VARARGS,
    /**
     * @brief Get kernel stride from the source layer.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The kernel stride from the source layer.
     */
    "Get kernel stride from the source layer"},

    {"GetUnits", LayerAccessors::GetUnits, METH_VARARGS,
    /**
     * @brief Modify the destination float32 NumPy array beginning at a specified index
     *        by copying the units from the source layer.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Modify the destination float32 NumPy array beginning at a specified index by copying the units from the source layer"},

    {"SetUnits", LayerAccessors::SetUnits, METH_VARARGS,
    /**
     * @brief Set the units of the destination layer by copying the units from a source float32 NumPy array.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set the units of the destination layer by copying the units from a source float32 NumPy array"},

    {"GetDeltas", LayerAccessors::GetDeltas, METH_VARARGS,
    /**
     * @brief Modify the destination float32 NumPy array beginning at a specified index
     *        by copying the deltas from the source layer.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Modify the destination float32 NumPy array beginning at a specified index by copying the deltas from the source layer"},

    {"SetDeltas", LayerAccessors::SetDeltas, METH_VARARGS,
    /**
     * @brief Set the deltas of the destination layer by copying the deltas from a source float32 NumPy array.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set the deltas of the destination layer by copying the deltas from a source float32 NumPy array"},

    
    {"CopyWeights", WeightAccessors::CopyWeights, METH_VARARGS,
    /**
     * @brief Copy the weights from the specified source weight to the destination weight.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Copy the weights from the specified source weight to the destination weight"},

    {"SetWeights", WeightAccessors::SetWeights, METH_VARARGS,
    /**
     * @brief Set the weights in the destination weight from a source NumPy array of weights.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set the weights in the destination weight from a source NumPy array of weights"},

    {"SetBiases", WeightAccessors::SetBiases, METH_VARARGS,
    /**
     * @brief Set the biases in the destination weight from a source NumPy array of biases.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set the biases in the destination weight from a source NumPy array of biases"},

    {"GetWeights", WeightAccessors::GetWeights, METH_VARARGS,
    /**
     * @brief Get the weights from the source weight.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The weights from the source weight.
     */
    "Get the weights from the source weight"},

    {"GetBiases", WeightAccessors::GetBiases, METH_VARARGS,
    /**
     * @brief Get the biases from the source weight.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The biases from the source weight.
     */
    "Get the biases from the source weight"},

    {"SetNorm", WeightAccessors::SetNorm, METH_VARARGS,
    /**
     * @brief Set the norm for the destination weight.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set the norm for the destination weight"},
    
    {"GetDataSetName", DataSetAccessors::GetDataSetName, METH_VARARGS,
    /**
     * @brief Get the name from the source DataSetBase*.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The name from the source DataSetBase*.
     */
    "Get the name from the source DataSetBase*"},

    {"CreateDenseDataSet", DataSetAccessors::CreateDenseDataSet, METH_VARARGS,
    /**
     * @brief Create an encapsulated DataSetBase* from a dense NumPy array.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The encapsulated DataSetBase* created from the dense NumPy array.
     */
    "Create an encapsulated DataSetBase* from a dense NumPy array"},

    {"CreateSparseDataSet", DataSetAccessors::CreateSparseDataSet, METH_VARARGS,
    /**
     * @brief Create an encapsulated DataSetBase* from a compressed sparse row (CSR) SciPy two-dimensional matrix.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The encapsulated DataSetBase* created from the compressed sparse row (CSR) matrix.
     */
    "Create an encapsulated DataSetBase* from a compressed sparse row (CSR) SciPy two-dimensional matrix"},

    {"SetStreaming", DataSetAccessors::SetStreaming, METH_VARARGS,
    /**
     * @brief Set the streaming flag for the destination DataSetBase*.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The result of the operation.
     */
    "Set the streaming flag for the destination DataSetBase*"},

    {"GetStreaming", DataSetAccessors::GetStreaming, METH_VARARGS,
    /**
     * @brief Get the streaming flag from the source DataSetBase*.
     *
     * @param self The module object.
     * @param args The Python tuple containing the arguments.
     * @return PyObject* The streaming flag from the source DataSetBase*.
     */
    "Get the streaming flag from the source DataSetBase*"},

    {nullptr, nullptr, 0, nullptr}
};

/**
 * @brief PyInit_tensorhub - Module initialization function for the tensorhub module.
 *
 * @return PyObject* The initialized module object, or nullptr on failure.
 * @note This function is called when the module is imported in Python.
 * @note The caller is responsible for managing the lifetime of the returned module object.
 */
PyMODINIT_FUNC PyInit_tensorhub(void) {
    PyObject* module = PyModule_Create(&tensorhubModule);
    if (module == nullptr) {
        return nullptr;
    }

    import_array();
    return module;
}
