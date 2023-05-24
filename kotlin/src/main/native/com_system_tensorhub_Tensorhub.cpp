/**
 * @file jni_util.cpp
 * @brief JNI utility functions for Tensorhub library.
 */

#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <span>

#include "jni_util.h"
#include "com_system_tensorhub_Tensorhub.h"

#include "runtime/Context.h"

namespace fs = std::filesystem;
using namespace tensorhub;
using namespace tensorhub::jni;

using DataSetEnums::Attributes;
using DataSetEnums::DataType;

namespace {
    fs::dynamic_library LIB_MPI;
    const fs::path LIB_MPI_SO = "libmpi.so";

    References REFS;

    const std::string _Layer = "com/system/tensorhub/Layer";
    const std::string _DataSet = "com/system/tensorhub/DataSet";
    const std::string _Output = "com/system/tensorhub/Output";

    jmethodID java_ArrayList_;
    jmethodID java_ArrayList_add;

    jmethodID Layer_;

    jmethodID DataSet_getName;
    jmethodID DataSet_getLayerName;
    jmethodID DataSet_getAttribute;
    jmethodID DataSet_getDataTypeOrdinal;

    jmethodID DataSet_getDimensions;
    jmethodID DataSet_getDimX;
    jmethodID DataSet_getDimY;
    jmethodID DataSet_getDimZ;
    jmethodID DataSet_getExamples;
    jmethodID DataSet_getStride;

    jmethodID DataSet_getSparseStart;
    jmethodID DataSet_getSparseEnd;
    jmethodID DataSet_getSparseIndex;
    jmethodID DataSet_getData;

    jmethodID Output_getName;
    jmethodID Output_getLayerName;
    jmethodID Output_getIndexes;
    jmethodID Output_getScores;
}

/**
 * @brief JNI library initialization function.
 * 
 * This function is automatically called when the JNI library is loaded.
 * It initializes the necessary JNI references and loads the "libmpi.so" library.
 * 
 * @param vm A pointer to the JavaVM instance.
 * @param reserved Reserved for future use.
 * @return The JNI version supported by the library.
 */
jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    fs::path libPath = fs::current_path() / LIB_MPI_SO;

    LIB_MPI = fs::dynamic_library(libPath);

    if (!LIB_MPI.is_loaded()) {
        std::cerr << "Failed to load libmpi.so" << " at " << std::source_location::current() << std::endl;
        exit(1);
    }

    JNIEnv* env;
    if (vm->GetEnv((void**)&env, JNI_VERSION_10) != JNI_OK) {
        return JNI_ERR;
    }
    else {
        java_ArrayList_ = findConstructorId(env, REFS, ArrayList, NO_ARGS_CONSTRUCTOR);
        java_ArrayList_add = findMethodId(env, REFS, ArrayList, "add", "(Ljava/lang/Object;)Z");

        Layer_ = findConstructorId(env, REFS, _Layer, "(Ljava/lang/String;Ljava/lang/String;IIIIII)V");

        DataSet_getName = findMethodId(env, REFS, _DataSet, "getName", "()Ljava/lang/String;");
        DataSet_getLayerName = findMethodId(env, REFS, _DataSet, "getLayerName", "()Ljava/lang/String;");
        DataSet_getAttribute = findMethodId(env, REFS, _DataSet, "getAttribute", "()I");
        DataSet_getDataTypeOrdinal = findMethodId(env, REFS, _DataSet, "getDataTypeOrdinal", "()I");
        DataSet_getDimensions = findMethodId(env, REFS, _DataSet, "getDimensions", "()I");
        DataSet_getDimX = findMethodId(env, REFS, _DataSet, "getDimX", "()I");
        DataSet_getDimY = findMethodId(env, REFS, _DataSet, "getDimY", "()I");
        DataSet_getDimZ = findMethodId(env, REFS, _DataSet, "getDimZ", "()I");
        DataSet_getExamples = findMethodId(env, REFS, _DataSet, "getExamples", "()I");
        DataSet_getStride = findMethodId(env, REFS, _DataSet, "getStride", "()I");
        DataSet_getSparseStart = findMethodId(env, REFS, _DataSet, "getSparseStart", "()[J");
        DataSet_getSparseEnd = findMethodId(env, REFS, _DataSet, "getSparseEnd", "()[J");
        DataSet_getSparseIndex = findMethodId(env, REFS, _DataSet, "getSparseIndex", "()[J");
        DataSet_getData = findMethodId(env, REFS, _DataSet, "getData", "()Ljava/nio/ByteBuffer;");

        Output_getName = findMethodId(env, REFS, _Output, "getName", "()Ljava/lang/String;");
        Output_getLayerName = findMethodId(env, REFS, _Output, "getLayerName", "()Ljava/lang/String;");
        Output_getIndexes = findMethodId(env, REFS, _Output, "getIndexes", "()[J");
        Output_getScores = findMethodId(env, REFS, _Output, "getScores", "()[F");

        return JNI_VERSION_10;
    }
}

/**
 * @brief JNI library cleanup function.
 * 
 * This function is automatically called when the JNI library is unloaded.
 * It releases the allocated JNI references.
 * 
 * @param vm A pointer to the JavaVM instance.
 * @param reserved Reserved for future use.
 */
void JNI_OnUnload(JavaVM* vm, void* reserved) {
    using namespace tensorhub;

    JNIEnv* env;
    if (vm->GetEnv((void**)&env, JNI_VERSION_10) != JNI_OK) {
        return;
    }
    else {
        deleteReferences(env, REFS);
    }
}

/**
 * @brief JNI method for loading a network.
 * 
 * This method is called from Java to load a network into the Tensorhub library.
 * It creates a new Context object and returns its pointer as a jlong value.
 * 
 * @param env The JNIEnv pointer.
 * @param clazz The Java class object.
 * @param jNetworkFileName The network file name as a Java string.
 * @param batchSize The batch size for the network.
 * @param maxK The maximum value of K for the network.
 * @return The pointer to the created Context object.
 */
JNIEXPORT jlong JNICALL Java_com_system_tensorhub_tensorhub_load(JNIEnv* env, jclass clazz, jstring jNetworkFileName,
                                                                jint batchSize, jint maxK) {
    std::string networkFileName = env->GetStringUTFChars(jNetworkFileName, nullptr);
    std::unique_ptr<Context> dc = std::make_unique<Context>(networkFileName, batchSize, maxK);
    env->ReleaseStringUTFChars(jNetworkFileName, networkFileName.c_str());
    return reinterpret_cast<jlong>(dc.release());
}

/**
 * @brief Utility function to get the dimensions of a dataset.
 * 
 * This function extracts the dimensions of a dataset object in Java and returns them as a DataSetDimensions struct.
 * 
 * @param env The JNIEnv pointer.
 * @param jDataset The Java dataset object.
 * @return The dimensions of the dataset.
 */
DataSetDimensions getDataDimensions(JNIEnv* env, jobject jDataset) {
    DataSetDimensions dim;
    dim._width = env->CallIntMethod(jDataset, DataSet_getDimX);
    dim._length = env->CallIntMethod(jDataset, DataSet_getDimY);
    dim._height = env->CallIntMethod(jDataset, DataSet_getDimZ);
    dim._dimensions = env->CallIntMethod(jDataset, DataSet_getDimensions);
    return dim;
}

/**
 * @brief JNI method for loading datasets into the network.
 * 
 * This method is called from Java to load datasets into the network in the Tensorhub library.
 * It retrieves the dataset information from Java objects and initializes the input layer datasets in the Context.
 * 
 * @param env The JNIEnv pointer.
 * @param clazz The Java class object.
 * @param ptr The pointer to the Context object.
 * @param jDatasets The array of dataset objects in Java.
 */
JNIEXPORT void JNICALL Java_com_system_tensorhub_tensorhub_load_1datasets(JNIEnv* env, jclass clazz, jlong ptr,
                                                                          jobjectArray jDatasets) {
    using DataSetEnums::DataType;

    jsize len = env->GetArrayLength(jDatasets);
    std::vector<DataSetDescriptor> datasetDescriptors;

    for (jsize i = 0; i < len; ++i) {
        jobject jDataset = env->GetObjectArrayElement(jDatasets, i);
        DataType dataType = static_cast<DataType>(env->CallIntMethod(jDataset, DataSet_getDataTypeOrdinal));
        jstring jName = static_cast<jstring>(env->CallObjectMethod(jDataset, DataSet_getName));
        std::string name = env->GetStringUTFChars(jName, nullptr);
        jint attributes = env->CallIntMethod(jDataset, DataSet_getAttribute);
        int examples = env->CallIntMethod(jDataset, DataSet_getExamples);
        int stride = env->CallIntMethod(jDataset, DataSet_getStride);
        DataSetDimensions dim = getDataDimensions(env, jDataset);

        float sparseDensity = static_cast<float>(stride) / (dim._width * dim._length * dim._height);

        DataSetDescriptor descriptor;
        descriptor._name = name;
        descriptor._attributes = attributes;
        descriptor._dataType = dataType;
        descriptor._dim = dim;
        descriptor._examples = examples;
        descriptor._sparseDensity = sparseDensity;

        datasetDescriptors.push_back(descriptor);
        env->ReleaseStringUTFChars(jName, name.c_str());
    }

    std::unique_ptr<Context> dc(reinterpret_cast<Context*>(ptr));
    dc->initInputLayerDataSets(datasetDescriptors);
}

/**
 * @brief JNI method for shutting down the network.
 * 
 * This method is called from Java to shut down the network in the Tensorhub library.
 * It releases the resources associated with the network.
 * 
 * @param env The JNIEnv pointer.
 * @param clazz The Java class object.
 * @param ptr The pointer to the Context object.
 */
JNIEXPORT void JNICALL Java_com_system_tensorhub_tensorhub_shutdown(JNIEnv* env, jclass clazz, jlong ptr) {
    std::unique_ptr<Context> dc(reinterpret_cast<Context*>(ptr));
}

/**
 * @brief JNI method for getting layers of a specific kind.
 * 
 * This method is called from Java to retrieve layers of a specific kind from the network in the Tensorhub library.
 * It returns a list of layer objects as an ArrayList in Java.
 * 
 * @param env The JNIEnv pointer.
 * @param clazz The Java class object.
 * @param ptr The pointer to the Context object.
 * @param kindOrdinal The ordinal value of the layer kind.
 * @return The ArrayList containing the layer objects.
 */
JNIEXPORT jobject JNICALL Java_com_system_tensorhub_tensorhub_get_1layers(JNIEnv* env, jclass clazz, jlong ptr,
                                                                         jint kindOrdinal) {
    std::unique_ptr<Context> dc(reinterpret_cast<Context*>(ptr));
    Network* network = dc->getNetwork();
    Layer::Kind kind = static_cast<Layer::Kind>(kindOrdinal);

    std::vector<const Layer*> layers;
    std::vector<const Layer*>::iterator it = network->GetLayers(kind, layers);
    if (it == layers.end()) {
        throwJavaException(env, RuntimeException, "No layers of type %s found in network %s", Layer::_sKindMap[kind],
                           network->GetName());
    }

    jobject jLayers = newObject(env, REFS, ArrayList, java_ArrayList_);

    for (; it != layers.end(); ++it) {
        const Layer* layer = *it;
        const std::string& name = layer->GetName();
        const std::string& datasetName = layer->GetDataSetName();
        jstring jName = env->NewStringUTF(name.c_str());
        jstring jDatasetName = env->NewStringUTF(datasetName.c_str());
        int kind = static_cast<int>(layer->GetKind());
        uint32_t attributes = layer->GetAttributes();

        uint32_t numDim = layer->GetNumDimensions();

        uint32_t lx, ly, lz, lw;
        std::tie(lx, ly, lz, lw) = layer->GetDimensions();

        jobject jInputLayer =
            newObject(env, REFS, _Layer, Layer_, jName, jDatasetName, kind, attributes, numDim, lx, ly, lz);

        env->CallBooleanMethod(jLayers, java_ArrayList_add, jInputLayer);
    }
    return jLayers;
}

/**
 * @brief Utility function to check if a dataset matches the expected attributes, data type, and dimensions.
 * 
 * This function is used to check if a dataset object retrieved from Java matches the expected attributes, data type,
 * and dimensions in C++. It throws a Java exception if any mismatch is detected.
 * 
 * @param env The JNIEnv pointer.
 * @param Dataset The dataset object to check.
 * @param attribute The expected attribute value.
 * @param dataType The expected data type value.
 * @param dim The expected dimensions.
 * @param examples The expected number of examples.
 */
void checkDataset(JNIEnv* env, DataSetBase* Dataset, uint32_t attribute, DataType dataType,
                  const DataSetDimensions& dim, uint32_t examples) {
    if (Dataset->_attributes != attribute) {
        throwJavaException(env, IllegalArgumentException, "Attribute mismatch in dataset %s",
                           Dataset->_name.c_str());
    }
    if (Dataset->_dataType != dataType) {
        throwJavaException(env, IllegalArgumentException, "Data type mismatch in dataset %s",
                           Dataset->_name.c_str());
    }
    if (Dataset->_dimensions != dim._dimensions) {
        throwJavaException(env, IllegalArgumentException, "Dimension mismatch in dataset %s",
                           Dataset->_name.c_str());
    }
    if (Dataset->_width != dim._width) {
        throwJavaException(env, IllegalArgumentException, "Width mismatch in dataset %s",
                           Dataset->_name.c_str());
    }
    if (Dataset->_length != dim._length) {
        throwJavaException(env, IllegalArgumentException, "Length mismatch in dataset %s",
                           Dataset->_name.c_str());
    }
    if (Dataset->_height != dim._height) {
        throwJavaException(env, IllegalArgumentException, "Height mismatch in dataset %s",
                           Dataset->_name.c_str());
    }
    if (Dataset->_examples != examples) {
        throwJavaException(env, IllegalArgumentException, "Examples mismatch in dataset %s",
                           Dataset->_name.c_str());
    }
}

/**
 * @brief JNI method for performing prediction on the network.
 * 
 * This method is called from Java to perform prediction on the network in the Tensorhub library.
 * It retrieves input datasets and output datasets from Java objects and performs the prediction.
 * 
 * @param env The JNIEnv pointer.
 * @param clazz The Java class object.
 * @param ptr The pointer to the Context object.
 * @param k The value of K for top-K prediction. If k <= 0, all output units are returned.
 * @param jInputDatasets The array of input dataset objects in Java.
 * @param jOutputDatasets The array of output dataset objects in Java.
 */
JNIEXPORT void JNICALL Java_com_system_tensorhub_tensorhub_predict(JNIEnv* env, jclass clazz, jlong ptr, jint k,
                                                                  jobjectArray jInputDatasets,
                                                                  jobjectArray jOutputDatasets) {
    std::unique_ptr<Context> dc(reinterpret_cast<Context*>(ptr));
    Network* network = dc->getNetwork();

    std::vector<const Layer*> inputLayers;
    network->GetLayers(Layer::Kind::Input, inputLayers);
    uint32_t batchSize = network->GetBatch();

    jsize inputLen = env->GetArrayLength(jInputDatasets);

    using DataSetEnums::DataType;

    for (jsize i = 0; i < inputLen; ++i) {
        jobject jInputDataset = env->GetObjectArrayElement(jInputDatasets, i);
        jstring jDatasetName = static_cast<jstring>(env->CallObjectMethod(jInputDataset, DataSet_getName));
        jstring jLayerName = static_cast<jstring>(env->CallObjectMethod(jInputDataset, DataSet_getLayerName));
        std::string datasetName = env->GetStringUTFChars(jDatasetName, nullptr);
        std::string layerName = env->GetStringUTFChars(jLayerName, nullptr);
        uint32_t examples = env->CallIntMethod(jInputDataset, DataSet_getExamples);
        DataSetDimensions dim = getDataDimensions(env, jInputDataset);
        uint32_t attribute = env->CallIntMethod(jInputDataset, DataSet_getAttribute);
        DataType dataType = static_cast<DataType>(env->CallIntMethod(jInputDataset, DataSet_getDataTypeOrdinal));

        const Layer* layer = network->GetLayer(layerName);
        if (!layer) {
            throwJavaException(env, IllegalArgumentException,
                               "No matching layer found in network %s for dataset: %s", network->GetName(),
                               datasetName);
        }

        DataSetBase* Dataset = layer->GetDataSet();
        checkDataset(env, Dataset, attribute, dataType, dim, examples);

        jobject srcByteBuffer = env->CallObjectMethod(jInputDataset, DataSet_getData);
        const void* srcDataNative = env->GetDirectBufferAddress(srcByteBuffer);

        if (Dataset->_attributes == Attributes::Sparse) {
            jlongArray jSparseStart = static_cast<jlongArray>(env->CallObjectMethod(jInputDataset, DataSet_getSparseStart));
            jlongArray jSparseEnd = static_cast<jlongArray>(env->CallObjectMethod(jInputDataset, DataSet_getSparseEnd));
            jlongArray jSparseIndex = static_cast<jlongArray>(env->CallObjectMethod(jInputDataset, DataSet_getSparseIndex));

            std::span<long> sparseStart(static_cast<long*>(env->GetPrimitiveArrayCritical(jSparseStart, nullptr)),
                                        env->GetArrayLength(jSparseStart));
            std::span<long> sparseEnd(static_cast<long*>(env->GetPrimitiveArrayCritical(jSparseEnd, nullptr)),
                                      env->GetArrayLength(jSparseEnd));
            std::span<long> sparseIndex(static_cast<long*>(env->GetPrimitiveArrayCritical(jSparseIndex, nullptr)),
                                        env->GetArrayLength(jSparseIndex));

            Dataset->LoadSparseData(sparseStart.data(), sparseEnd.data(), srcDataNative, sparseIndex.data());

            env->ReleasePrimitiveArrayCritical(jSparseStart, sparseStart.data(), JNI_ABORT);
            env->ReleasePrimitiveArrayCritical(jSparseEnd, sparseEnd.data(), JNI_ABORT);
            env->ReleasePrimitiveArrayCritical(jSparseIndex, sparseIndex.data(), JNI_ABORT);
        }
        else {
            Dataset->LoadDenseData(srcDataNative);
        }

        env->ReleaseStringUTFChars(jDatasetName, datasetName.c_str());
        env->ReleaseStringUTFChars(jLayerName, layerName.c_str());
    }

    network->SetPosition(0);
    network->PredictBatch();

    jsize outputLen = env->GetArrayLength(jOutputDatasets);
    for (jsize i = 0; i < outputLen; ++i) {
        jobject jOutputDataset = env->GetObjectArrayElement(jOutputDatasets, i);

        jstring jLayerName = static_cast<jstring>(env->CallObjectMethod(jOutputDataset, Output_getLayerName));
        std::string layerName = env->GetStringUTFChars(jLayerName, nullptr);

        jfloatArray jScores = static_cast<jfloatArray>(env->CallObjectMethod(jOutputDataset, Output_getScores));
        jlongArray jIndexes = static_cast<jlongArray>(env->CallObjectMethod(jOutputDataset, Output_getIndexes));

        Layer* outputLayer = network->GetLayer(layerName.data());

        uint32_t x, y, z, w;
        std::tie(x, y, z, w) = outputLayer->GetDimensions();
        uint32_t stride = x * y * z;

        std::vector<float> scores(env->GetArrayLength(jScores));
        env->GetFloatArrayRegion(jScores, 0, scores.size(), scores.data());

        if (k > 0) {
            float* outputUnitBuffer = network->GetUnitBuffer(layerName.data());
            std::vector<long> indexes(env->GetArrayLength(jIndexes));
            env->GetLongArrayRegion(jIndexes, 0, indexes.size(), indexes.data());
            float* dScores = dc->getOutputScoresBuffer(layerName.data())->_pDevData;
            uint32_t* dIndexes = dc->getOutputIndexesBuffer(layerName.data())->_pDevData;
            std::vector<uint32_t> hIndexes(k * batchSize, 0);

            kCalculateOutput(outputUnitBuffer, dScores, dIndexes, batchSize, stride, k);

            cudaMemcpy(scores.data(), dScores, k * batchSize * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(hIndexes.data(), dIndexes, k * batchSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);

            for (size_t i = 0; i < k * batchSize; ++i) {
                indexes[i] = static_cast<long>(hIndexes[i]);
            }

            env->SetLongArrayRegion(jIndexes, 0, indexes.size(), indexes.data());
        }
        else {
            outputLayer->GetUnits(reinterpret_cast<float*>(scores.data()));
        }

        env->SetFloatArrayRegion(jScores, 0, scores.size(), scores.data());
        env->ReleaseStringUTFChars(jLayerName, layerName.c_str());
    }
}
