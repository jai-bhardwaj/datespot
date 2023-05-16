#include <dlfcn.h>
#include <string>
#include <vector>

#include "jni_util.h"
#include "com_system_tensorhub_tensorhub.h"

#include "core/src/runtime/Context.h"

/**
 * Namespace for the TensorHub library.
 */
namespace tensorhub {

/**
 * Namespace for the JNI-related functionality of TensorHub.
 */
namespace jni {

/**
 * Class representing references used in JNI operations.
 */
class References {};

}  // namespace jni
}  // namespace tensorhub

namespace {

/**
 * Pointer to the loaded libmpi.so library.
 */
void* lib_mpi = nullptr;

/**
 * Filename of the libmpi.so library.
 */
constexpr char kLibMpiSo[] = "libmpi.so";

/**
 * References object for JNI operations.
 */
tensorhub::jni::References refs;

/**
 * Fully-qualified class name for the Layer Java class.
 */
constexpr char kLayer[] = "com/system/tensorhub/Layer";

/**
 * Fully-qualified class name for the DataSet Java class.
 */
constexpr char kDataSet[] = "com/system/tensorhub/DataSet";

/**
 * Fully-qualified class name for the Output Java class.
 */
constexpr char kOutput[] = "com/system/tensorhub/Output";

/**
 * Method ID for the add method of the ArrayList Java class.
 */
jmethodID java_array_list;

/**
 * Method ID for the add method of the ArrayList Java class.
 */
jmethodID java_array_list_add;

/**
 * Method ID for the Layer constructor.
 */
jmethodID layer;

/**
 * Method ID for the getName method of the DataSet Java class.
 */
jmethodID data_set_get_name;

/**
 * Method ID for the getLayerName method of the DataSet Java class.
 */
jmethodID data_set_get_layer_name;

/**
 * Method ID for the getAttribute method of the DataSet Java class.
 */
jmethodID data_set_get_attribute;

/**
 * Method ID for the getDataTypeOrdinal method of the DataSet Java class.
 */
jmethodID data_set_get_data_type_ordinal;

/**
 * Method ID for the getDimensions method of the DataSet Java class.
 */
jmethodID data_set_get_dimensions;

/**
 * Method ID for the getDimX method of the DataSet Java class.
 */
jmethodID data_set_get_dim_x;

/**
 * Method ID for the getDimY method of the DataSet Java class.
 */
jmethodID data_set_get_dim_y;

/**
 * Method ID for the getDimZ method of the DataSet Java class.
 */
jmethodID data_set_get_dim_z;

/**
 * Method ID for the getExamples method of the DataSet Java class.
 */
jmethodID data_set_get_examples;

/**
 * Method ID for the getStride method of the DataSet Java class.
 */
jmethodID data_set_get_stride;

/**
 * Method ID for the getSparseStart method of the DataSet Java class.
 */
jmethodID data_set_get_sparse_start;

/**
 * Method ID for the getSparseEnd method of the DataSet Java class.
 */
jmethodID data_set_get_sparse_end;

/**
 * Method ID for the getSparseIndex method of the DataSet Java class.
 */
jmethodID data_set_get_sparse_index;

/**
 * Method ID for the getData method of the DataSet Java class.
 */
jmethodID data_set_get_data;

/**
 * Method ID for the getName method of the Output Java class.
 */
jmethodID output_get_name;

/**
 * Method ID for the getLayerName method of the Output Java class.
 */
jmethodID output_get_layer_name;

/**
 * Method ID for the getIndexes method of the Output Java class.
 */
jmethodID output_get_indexes;

/**
 * Method ID for the getScores method of the Output Java class.
 */
jmethodID output_get_scores;

}  // namespace

/**
 * JNI library load function.
 *
 * @param vm Pointer to the JavaVM instance.
 * @param reserved Reserved parameter (unused).
 * @return The JNI version supported by the library.
 */
jint JNI_OnLoad(JavaVM *vm, void *reserved)
{
    LIB_MPI = dlopen(LIB_MPI_SO, RTLD_NOW | RTLD_GLOBAL);

    if (LIB_MPI == NULL)
    {
        std::cerr << "Failed to load libmpi.so" << std::endl;
        std::exit(1);
    }

    JNIEnv* env;
    if (vm->GetEnv((void **) &env, JNI_VERSION_1_8) != JNI_OK)
    {
        return JNI_ERR;
    } else
    {
        auto java_ArrayList_ = findConstructorId(env, REFS, ArrayList, NO_ARGS_CONSTRUCTOR);
        auto java_ArrayList_add = findMethodId(env, REFS, ArrayList, "add", "(Ljava/lang/Object;)Z");
        auto Layer_ = findConstructorId(env, REFS, _Layer, "(Ljava/lang/String;Ljava/lang/String;IIIIII)V");
        auto DataSet_getName = findMethodId(env, REFS, _DataSet, "getName", "()Ljava/lang/String;");
        auto DataSet_getLayerName = findMethodId(env, REFS, _DataSet, "getLayerName", "()Ljava/lang/String;");
        auto DataSet_getAttribute = findMethodId(env, REFS, _DataSet, "getAttribute", "()I");
        auto DataSet_getDataTypeOrdinal = findMethodId(env, REFS, _DataSet, "getDataTypeOrdinal", "()I");
        auto DataSet_getDimensions = findMethodId(env, REFS, _DataSet, "getDimensions", "()I");
        auto DataSet_getDimX = findMethodId(env, REFS, _DataSet, "getDimX", "()I");
        auto DataSet_getDimY = findMethodId(env, REFS, _DataSet, "getDimY", "()I");
        auto DataSet_getDimZ = findMethodId(env, REFS, _DataSet, "getDimZ", "()I");
        auto DataSet_getExamples = findMethodId(env, REFS, _DataSet, "getExamples", "()I");
        auto DataSet_getStride = findMethodId(env, REFS, _DataSet, "getStride", "()I");
        auto DataSet_getSparseStart = findMethodId(env, REFS, _DataSet, "getSparseStart", "()[J");
        auto DataSet_getSparseEnd = findMethodId(env, REFS, _DataSet, "getSparseEnd", "()[J");
        auto DataSet_getSparseIndex = findMethodId(env, REFS, _DataSet, "getSparseIndex", "()[J");
        auto DataSet_getData = findMethodId(env, REFS, _DataSet, "getData", "()Ljava/nio/ByteBuffer;");

        auto Output_getName = findMethodId(env, REFS, _Output, "getName", "()Ljava/lang/String;");
        auto Output_getLayerName = findMethodId(env, REFS, _Output, "getLayerName", "()Ljava/lang/String;");
        auto Output_getIndexes = findMethodId(env, REFS, _Output, "getIndexes", "()[J");
        auto Output_getScores = findMethodId(env, REFS, _Output, "getScores", "()[F");

        return JNI_VERSION_1_8;
    }
}

/**
 * JNI library unload function.
 *
 * @param vm Pointer to the JavaVM instance.
 * @param reserved Reserved parameter (unused).
 */
void JNI_OnUnload(JavaVM *vm, void *reserved)
{
    using namespace tensorhub;

    auto [env, result] = vm->GetEnv<JNIEnv*>(JNI_VERSION_1_8);
    if (result != JNI_OK)
    {
        return;
    }

    deleteReferences(env, REFS);
}

/**
 * Loads a TensorHub context with the specified network file, batch size, and maxK.
 *
 * @param env The JNI environment.
 * @param clazz The Java class associated with the native method.
 * @param jNetworkFileName The name of the network file.
 * @param batchSize The batch size for the network.
 * @param maxK The maximum K value.
 * @return A pointer to the loaded TensorHub context.
 */
JNIEXPORT jlong JNICALL Java_com_system_tensorhub_tensorhub_load(JNIEnv *env, jclass clazz, jstring jNetworkFileName,
                                                           jint batchSize, jint maxK)
{
    auto networkFileName = env->GetStringUTFChars(jNetworkFileName, 0);
    std::unique_ptr<tensorhubContext> dc(new tensorhubContext(networkFileName.get(), batchSize, maxK));
    return (jlong) dc.release();
}

/**
 * Retrieves the data dimensions of a Dataset object.
 *
 * @param env The JNI environment.
 * @param jDataset The Dataset object.
 * @return The dimensions of the Dataset.
 */
DataSetDimensions getDataDimensions(JNIEnv *env, jobject jDataset)
{
    DataSetDimensions dim;
    auto getDim = [&env, &jDataset](auto methodId) {
        return env->CallIntMethod(jDataset, methodId);
    };

    dim._width = getDim(DataSet_getDimX);
    dim._length = getDim(DataSet_getDimY);
    dim._height = getDim(DataSet_getDimZ);
    dim._dimensions = getDim(DataSet_getDimensions);
    return dim;
}

/**
 * Loads datasets into the TensorHub context.
 *
 * @param env The JNI environment.
 * @param clazz The Java class associated with the native method.
 * @param ptr Pointer to the TensorHub context.
 * @param jDatasets An array of Dataset objects to load.
 */
JNIEXPORT void JNICALL Java_com_system_tensorhub_tensorhub_load_1datasets(JNIEnv *env, jclass clazz, jlong ptr,
                                                                  jobjectArray jDatasets)
{
    using DataSetEnums::DataType;

    auto len = env->GetArrayLength(jDatasets);
    std::vector<DataSetDescriptor> datasetDescriptors;

    for (auto i = 0; i < len; ++i)
    {
        auto jDataset = env->GetObjectArrayElement(jDatasets, i);
        auto dataType = static_cast<DataType>(env->CallIntMethod(jDataset, DataSet_getDataTypeOrdinal));
        auto jName = (jstring) env->CallObjectMethod(jDataset, DataSet_getName);
        auto name = env->GetStringUTFChars(jName, NULL);
        auto attributes = env->CallIntMethod(jDataset, DataSet_getAttribute);
        auto examples = env->CallIntMethod(jDataset, DataSet_getExamples);
        auto stride = env->CallIntMethod(jDataset, DataSet_getStride);
        auto dim = getDataDimensions(env, jDataset);

        auto sparseDensity =  (double) stride / ((double) (dim._width * dim._length * dim._height));

        DataSetDescriptor descriptor;
        descriptor._name = name;
        descriptor._attributes = attributes;
        descriptor._dataType = dataType;
        descriptor._dim = dim;
        descriptor._examples = examples;
        descriptor._sparseDensity = sparseDensity;

        datasetDescriptors.push_back(descriptor);
        env->ReleaseStringUTFChars(jName, name);
    }

    auto dc = tensorhubContext::fromPtr(ptr);
    dc->initInputLayerDataSets(datasetDescriptors);
}

/**
 * Shuts down the TensorHub context associated with the given pointer.
 *
 * @param env The JNI environment.
 * @param clazz The Java class associated with the native method.
 * @param ptr Pointer to the TensorHub context.
 */
JNIEXPORT void JNICALL Java_com_system_tensorhub_tensorhub_shutdown(JNIEnv *env, jclass clazz, jlong ptr)
{
    std::unique_ptr<tensorhubContext> dc(tensorhubContext::fromPtr(ptr));
}

/**
 * Retrieves layers of a specified kind from the network.
 *
 * @param env The JNI environment.
 * @param clazz The Java class associated with the native method.
 * @param ptr Pointer to the TensorHub context.
 * @param kindOrdinal The ordinal value representing the layer kind.
 * @return An ArrayList of Layer objects.
 * @throws RuntimeException if no layers of the specified kind are found in the network.
 */
JNIEXPORT jobject JNICALL Java_com_system_tensorhub_tensorhub_get_1layers(JNIEnv *env, jclass clazz, jlong ptr,
                                                                    jint kindOrdinal)
{
    auto dc = tensorhubContext::fromPtr(ptr);
    auto network = dc->getNetwork();
    auto kind = static_cast<Layer::Kind>(kindOrdinal);

    auto layers = network->GetLayers(kind);
    if (layers.empty())
    {
        throwJavaException(env, RuntimeException, "No layers of type %s found in network %s", Layer::_sKindMap[kind],
                           network->GetName());
    }

    auto jLayers = newObject(env, REFS, ArrayList, java_ArrayList_);

    for (auto&& layer : layers)
    {
        auto name = layer->GetName();
        auto datasetName = layer->GetDataSetName();
        auto jName = env->NewStringUTF(name.c_str());
        auto jDatasetName = env->NewStringUTF(datasetName.c_str());
        auto layerKind = static_cast<int>(layer->GetKind());
        auto attributes = layer->GetAttributes();
        auto numDim = layer->GetNumDimensions();
        auto [lx, ly, lz, lw] = layer->GetDimensions();

        auto jInputLayer = newObject(env, REFS, _Layer, Layer_, jName, jDatasetName, layerKind, attributes, numDim,
                                        lx, ly, lz);

        env->CallBooleanMethod(jLayers, java_ArrayList_add, jInputLayer);
    }
    return jLayers;
}

/**
 * Checks if a dataset matches the specified attributes, data type, dimensions, and number of examples.
 *
 * @param env The JNI environment.
 * @param Dataset Pointer to the dataset to check.
 * @param attribute The expected attribute value.
 * @param dataType The expected data type.
 * @param dim The expected dataset dimensions.
 * @param examples The expected number of examples.
 * @throws IllegalArgumentException if any of the dataset attributes do not match the expected values.
 */
void checkDataset(JNIEnv *env, DataSetBase *Dataset, uint32_t attribute, DataType dataType,
                  const DataSetDimensions &dim, uint32_t examples)
{
    if (Dataset->_attributes != attribute)
    {
        throwJavaException(env, IllegalArgumentException, "Attribute mismatch in dataset %s", Dataset->_name.c_str());
    }
    if (Dataset->_dataType != dataType)
    {
        throwJavaException(env, IllegalArgumentException, "Data type mismatch in dataset %s", Dataset->_name.c_str());
    }
    if (Dataset->_dimensions != dim._dimensions)
    {
        throwJavaException(env, IllegalArgumentException, "Dimension mismatch in dataset %s", Dataset->_name.c_str());
    }
    if (Dataset->_width != dim._width || Dataset->_length != dim._length || Dataset->_height != dim._height)
    {
        throwJavaException(env, IllegalArgumentException, "Dimension mismatch in dataset %s", Dataset->_name.c_str());
    }
    if (Dataset->_examples != examples)
    {
        throwJavaException(env, IllegalArgumentException, "Examples mismatch in dataset %s", Dataset->_name.c_str());
    }
}

/**
 * Perform prediction using TensorHub.
 *
 * @param env The JNI environment.
 * @param clazz The Java class object.
 * @param ptr The pointer to the tensorhubContext.
 * @param k The value of k for top-k calculation.
 * @param jInputDatasets The array of input datasets.
 * @param jOutputDatasets The array of output datasets.
 */
JNIEXPORT void JNICALL Java_com_system_tensorhub_tensorhub_predict(JNIEnv *env, jclass clazz, jlong ptr, jint k,
                                                                 jobjectArray jInputDatasets, jobjectArray jOutputDatasets)
{
    /**
     * Retrieve the tensorhubContext and network objects from the pointer.
     */
    tensorhubContext *dc = tensorhubContext::fromPtr(ptr);
    Network *network = dc->getNetwork();

    /**
     * Retrieve input layers, batch size, and input datasets length.
     */
    vector<const Layer*> inputLayers;
    network->GetLayers(Layer::Kind::Input, inputLayers);
    uint32_t batchSize = network->GetBatch();
    jsize inputLen = env->GetArrayLength(jInputDatasets);

    /**
     * Import the DataSetEnums namespace.
     */
    using DataSetEnums::DataType;

    /**
     * Iterate over the input datasets.
     */
    for (jsize i = 0; i < inputLen; ++i)
    {
        /**
         * Get the i-th input dataset from the array.
         */
        jobject jInputDataset = env->GetObjectArrayElement(jInputDatasets, i);

        /**
         * Get the dataset name and layer name from the input dataset.
         */
        auto jDatasetName = env->CallObjectMethod(jInputDataset, DataSet_getName);
        auto jLayerName = env->CallObjectMethod(jInputDataset, DataSet_getLayerName);
        const char *datasetName = env->GetStringUTFChars(jDatasetName, NULL);
        const char *layerName = env->GetStringUTFChars(jLayerName, NULL);

        /**
         * Get the examples, dimensions, attribute, and data type of the input dataset.
         */
        uint32_t examples = env->CallIntMethod(jInputDataset, DataSet_getExamples);
        DataSetDimensions dim = getDataDimensions(env, jInputDataset);
        uint32_t attribute = env->CallIntMethod(jInputDataset, DataSet_getAttribute);
        DataType dataType = static_cast<DataType>(env->CallIntMethod(jInputDataset, DataSet_getDataTypeOrdinal));

        /**
         * Get the layer object based on the layer name.
         */
        const Layer *layer = network->GetLayer(layerName);
        if (!layer)
        {
            throwJavaException(env, IllegalArgumentException, "No matching layer found in network %s for dataset: %s",
                               network->GetName(), datasetName);
        }

        /**
         * Get the DataSetBase object from the layer and perform dataset checks.
         */
        DataSetBase *Dataset = layer->GetDataSet();
        checkDataset(env, Dataset, attribute, dataType, dim, examples);

        /**
         * Get the source data buffer from the input dataset.
         */
        auto srcByteBuffer = env->CallObjectMethod(jInputDataset, DataSet_getData);
        const void *srcDataNative = env->GetDirectBufferAddress(srcByteBuffer);

        if (Dataset->_attributes == Attributes::Sparse)
        {
            /**
             * If the dataset is sparse, retrieve the sparse start, end, and index arrays.
             */

            auto jSparseStart = env->CallObjectMethod(jInputDataset, DataSet_getSparseStart);
            auto jSparseEnd = env->CallObjectMethod(jInputDataset, DataSet_getSparseEnd);
            auto jSparseIndex = env->CallObjectMethod(jInputDataset, DataSet_getSparseIndex);

            /**
             * Get the sparse start, end, and index arrays as primitive long pointers.
             */
            long *sparseStart = (long*) env->GetPrimitiveArrayCritical(jSparseStart, NULL);
            long *sparseEnd = (long*) env->GetPrimitiveArrayCritical(jSparseEnd, NULL);
            long *sparseIndex = (long*) env->GetPrimitiveArrayCritical(jSparseIndex, NULL);

            /**
             * Load the sparse data into the dataset using the retrieved arrays.
             */
            Dataset->LoadSparseData(sparseStart, sparseEnd, srcDataNative, sparseIndex);

            /**
             * Release the sparse start, end, and index arrays.
             */
            env->ReleasePrimitiveArrayCritical(jSparseStart, sparseStart, JNI_ABORT);
            env->ReleasePrimitiveArrayCritical(jSparseEnd, sparseEnd, JNI_ABORT);
            env->ReleasePrimitiveArrayCritical(jSparseIndex, sparseIndex, JNI_ABORT);
        } else
        {
            /**
             * If the dataset is dense, load the dense data into the dataset.
             */
            Dataset->LoadDenseData(srcDataNative);
        }

        /**
         * Release the dataset name and layer name.
         */
        env->ReleaseStringUTFChars(jDatasetName, datasetName);
        env->ReleaseStringUTFChars(jLayerName, layerName);
    }

    /**
     * Set the position to 0 in the network and perform batch prediction.
     */
    network->SetPosition(0);
    network->PredictBatch();

/**
 * Process the output datasets.
 *
 * @param jOutputDatasets The array of output datasets.
 */
jsize outputLen = env->GetArrayLength(jOutputDatasets);
for (jsize i = 0; i < outputLen; ++i)
{
    /**
     * Get the i-th output dataset from the array.
     */
    jobject jOutputDataset = env->GetObjectArrayElement(jOutputDatasets, i);

    /**
     * Get the layer name of the output dataset.
     */
    jstring jLayerName = (jstring) env->CallObjectMethod(jOutputDataset, Output_getLayerName);
    const char *layerName = env->GetStringUTFChars(jLayerName, NULL);

    /**
     * Get the scores and indexes arrays from the output dataset.
     */
    jfloatArray jScores = (jfloatArray) env->CallObjectMethod(jOutputDataset, Output_getScores);
    jlongArray jIndexes = (jlongArray) env->CallObjectMethod(jOutputDataset, Output_getIndexes);

    /**
     * Get the output layer object based on the layer name.
     */
    Layer *outputLayer = network->GetLayer(layerName);

    /**
     * Get the dimensions of the output layer.
     */
    uint32_t x, y, z, w;
    tie(x, y, z, w) = outputLayer->GetDimensions();
    uint32_t stride = x * y * z;

    /**
     * Get the scores array as a primitive float pointer.
     */
    float *scores = (float*) env->GetPrimitiveArrayCritical(jScores, NULL);

    if (k > 0)
    {
        /**
         * If k is greater than 0, perform additional processing.
         */

        /**
         * Get the output unit buffer from the network.
         */
        Float *outputUnitBuffer = network->GetUnitBuffer(layerName);

        /**
         * Get the device scores and indexes buffers from dc.
         */
        long *indexes = (long*) env->GetPrimitiveArrayCritical(jIndexes, NULL);
        Float *dScores = dc->getOutputScoresBuffer(layerName)->_pDevData;
        uint32_t *dIndexes = dc->getOutputIndexesBuffer(layerName)->_pDevData;

        /**
         * Allocate a host buffer for indexes.
         */
        uint32_t *hIndexes = (uint32_t*) calloc(k * batchSize, sizeof(uint32_t));

        /**
         * Calculate the top-k scores and indexes using the provided function.
         */
        kCalculateTopK(outputUnitBuffer, dScores, dIndexes, batchSize, stride, k);

        /**
         * Copy the scores and indexes from the device to the host.
         */
        cudaMemcpy(scores, dScores, k * batchSize * sizeof(Float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hIndexes, dIndexes, k * batchSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        /**
         * Copy the indexes from the host buffer to the indexes array.
         */
        for (size_t i = 0; i < k * batchSize; ++i)
        {
            indexes[i] = (long) hIndexes[i];
        }
        free(hIndexes);

        /**
         * Release the indexes array.
         */
        env->ReleasePrimitiveArrayCritical(jIndexes, indexes, 0);
    } else
    {
        /**
         * If k is not greater than 0, simply get the output units from the output layer.
         */
        outputLayer->GetUnits((Float*) scores);
    }

        /**
         * Release the scores array.
         */
        env->ReleasePrimitiveArrayCritical(jScores, scores, 0);
        env->ReleaseStringUTFChars(jLayerName, layerName);
    }
}
