#include <dlfcn.h>
#include <string>
#include <vector>
#include <memory>
#include <iostream>

#include "jni_util.h"
#include "com_system_tensorhub_tensorhub.h"

#include "core/src/runtime/Context.h"

namespace tensorhub::jni {
    class References {};
}

namespace {
    void* lib_mpi = nullptr;
    constexpr char kLibMpiSo[] = "libmpi.so";
    tensorhub::jni::References refs;

    constexpr char kLayer[] = "com/system/tensorhub/Layer";
    constexpr char kDataSet[] = "com/system/tensorhub/DataSet";
    constexpr char kOutput[] = "com/system/tensorhub/Output";

    jmethodID java_array_list;
    jmethodID java_array_list_add;
    jmethodID layer;
    jmethodID data_set_get_name;
    jmethodID data_set_get_layer_name;
    jmethodID data_set_get_attribute;
    jmethodID data_set_get_data_type_ordinal;
    jmethodID data_set_get_dimensions;
    jmethodID data_set_get_dim_x;
    jmethodID data_set_get_dim_y;
    jmethodID data_set_get_dim_z;
    jmethodID data_set_get_examples;
    jmethodID data_set_get_stride;
    jmethodID data_set_get_sparse_start;
    jmethodID data_set_get_sparse_end;
    jmethodID data_set_get_sparse_index;
    jmethodID data_set_get_data;
    jmethodID output_get_name;
    jmethodID output_get_layer_name;
    jmethodID output_get_indexes;
    jmethodID output_get_scores;
}

jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    lib_mpi = dlopen(kLibMpiSo, RTLD_NOW | RTLD_GLOBAL);

    if (lib_mpi == nullptr) {
        std::cerr << "Failed to load libmpi.so" << std::endl;
        std::exit(1);
    }

    JNIEnv* env;
    if (vm->GetEnv((void **)&env, JNI_VERSION_1_8) != JNI_OK) {
        return JNI_ERR;
    } else {
        auto java_ArrayList_ = findConstructorId(env, refs, kLayer, NO_ARGS_CONSTRUCTOR);
        auto java_ArrayList_add = findMethodId(env, refs, kDataSet, "add", "(Ljava/lang/Object;)Z");
        auto Layer_ = findConstructorId(env, refs, kLayer, "(Ljava/lang/String;Ljava/lang/String;IIIIII)V");
        auto DataSet_getName = findMethodId(env, refs, kDataSet, "getName", "()Ljava/lang/String;");
        auto DataSet_getLayerName = findMethodId(env, refs, kDataSet, "getLayerName", "()Ljava/lang/String;");
        auto DataSet_getAttribute = findMethodId(env, refs, kDataSet, "getAttribute", "()I");
        auto DataSet_getDataTypeOrdinal = findMethodId(env, refs, kDataSet, "getDataTypeOrdinal", "()I");
        auto DataSet_getDimensions = findMethodId(env, refs, kDataSet, "getDimensions", "()I");
        auto DataSet_getDimX = findMethodId(env, refs, kDataSet, "getDimX", "()I");
        auto DataSet_getDimY = findMethodId(env, refs, kDataSet, "getDimY", "()I");
        auto DataSet_getDimZ = findMethodId(env, refs, kDataSet, "getDimZ", "()I");
        auto DataSet_getExamples = findMethodId(env, refs, kDataSet, "getExamples", "()I");
        auto DataSet_getStride = findMethodId(env, refs, kDataSet, "getStride", "()I");
        auto DataSet_getSparseStart = findMethodId(env, refs, kDataSet, "getSparseStart", "()[J");
        auto DataSet_getSparseEnd = findMethodId(env, refs, kDataSet, "getSparseEnd", "()[J");
        auto DataSet_getSparseIndex = findMethodId(env, refs, kDataSet, "getSparseIndex", "()[J");
        auto DataSet_getData = findMethodId(env, refs, kDataSet, "getData", "()Ljava/nio/ByteBuffer;");

        auto Output_getName = findMethodId(env, refs, kOutput, "getName", "()Ljava/lang/String;");
        auto Output_getLayerName = findMethodId(env, refs, kOutput, "getLayerName", "()Ljava/lang/String;");
        auto Output_getIndexes = findMethodId(env, refs, kOutput, "getIndexes", "()[J");
        auto Output_getScores = findMethodId(env, refs, kOutput, "getScores", "()[F");

        return JNI_VERSION_1_8;
    }
}

void JNI_OnUnload(JavaVM *vm, void *reserved) {
    using namespace tensorhub;

    JNIEnv *env;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_8) != JNI_OK) {
        return;
    }

    deleteReferences(env, refs);
}

JNIEXPORT jlong JNICALL Java_com_system_tensorhub_tensorhub_load(JNIEnv *env, jclass clazz, jstring jNetworkFileName, jint batchSize, jint maxK) {
    const char *networkFileName = env->GetStringUTFChars(jNetworkFileName, nullptr);
    std::unique_ptr<tensorhubContext> dc(new tensorhubContext(networkFileName, batchSize, maxK));
    env->ReleaseStringUTFChars(jNetworkFileName, networkFileName);
    return reinterpret_cast<jlong>(dc.release());
}

#include <vector>
#include <string>
#include <span>

// Helper function to convert jstring to std::string
std::string jstringToString(JNIEnv *env, jstring jstr) {
    const char *strChars = env->GetStringUTFChars(jstr, nullptr);
    std::string str(strChars);
    env->ReleaseStringUTFChars(jstr, strChars);
    return str;
}

// Helper function to release JNI critical arrays
template<typename T>
void releaseCriticalArray(JNIEnv *env, T *array, jint mode = 0) {
    env->ReleasePrimitiveArrayCritical(array, array, mode);
}

// Helper function to get array elements as std::vector
template<typename T>
std::vector<T> getArrayElements(JNIEnv *env, jobjectArray jarray) {
    jsize length = env->GetArrayLength(jarray);
    std::vector<T> elements(length);
    for (jsize i = 0; i < length; ++i) {
        elements[i] = env->GetObjectArrayElement(jarray, i);
    }
    return elements;
}

// Helper function to release JNI objects
template<typename T>
void releaseJNIObjects(JNIEnv *env, const std::vector<T> &objects) {
    for (const auto &obj : objects) {
        env->DeleteLocalRef(obj);
    }
}

// Helper function to release JNI strings
void releaseJNIStrings(JNIEnv *env, const std::vector<jstring> &strings) {
    releaseJNIObjects(env, strings);
}

DataSetDimensions getDataDimensions(JNIEnv *env, jobject jDataset) {
    DataSetDimensions dim;

    auto getDim = [&env, &jDataset](jmethodID methodId) {
        return env->CallIntMethod(jDataset, methodId);
    };

    dim._width = getDim(DataSet_getDimX);
    dim._length = getDim(DataSet_getDimY);
    dim._height = getDim(DataSet_getDimZ);
    dim._dimensions = getDim(DataSet_getDimensions);
    return dim;
}

JNIEXPORT void JNICALL Java_com_system_tensorhub_tensorhub_load_1datasets(JNIEnv *env, jclass clazz, jlong ptr,
                                                                         jobjectArray jDatasets) {
    using DataSetEnums::DataType;

    jsize len = env->GetArrayLength(jDatasets);
    std::vector<DataSetDescriptor> datasetDescriptors;

    for (jsize i = 0; i < len; ++i) {
        jobject jDataset = env->GetObjectArrayElement(jDatasets, i);
        DataType dataType = static_cast<DataType>(env->CallIntMethod(jDataset, DataSet_getDataTypeOrdinal));
        jstring jName = (jstring) env->CallObjectMethod(jDataset, DataSet_getName);
        std::string name = jstringToString(env, jName);
        jint attributes = env->CallIntMethod(jDataset, DataSet_getAttribute);
        jint examples = env->CallIntMethod(jDataset, DataSet_getExamples);
        jint stride = env->CallIntMethod(jDataset, DataSet_getStride);
        DataSetDimensions dim = getDataDimensions(env, jDataset);

        double sparseDensity = static_cast<double>(stride) / (dim._width * dim._length * dim._height);

        DataSetDescriptor descriptor;
        descriptor._name = name;
        descriptor._attributes = attributes;
        descriptor._dataType = dataType;
        descriptor._dim = dim;
        descriptor._examples = examples;
        descriptor._sparseDensity = sparseDensity;

        datasetDescriptors.push_back(descriptor);
    }

    auto dc = tensorhubContext::fromPtr(ptr);
    dc->initInputLayerDataSets(datasetDescriptors);
}

JNIEXPORT void JNICALL Java_com_system_tensorhub_tensorhub_shutdown(JNIEnv *env, jclass clazz, jlong ptr) {
    std::unique_ptr<tensorhubContext> dc(tensorhubContext::fromPtr(ptr));
}

JNIEXPORT jobject JNICALL Java_com_system_tensorhub_tensorhub_get_1layers(JNIEnv *env, jclass clazz, jlong ptr,
                                                                          jint kindOrdinal) {
    auto dc = tensorhubContext::fromPtr(ptr);
    auto network = dc->getNetwork();
    auto kind = static_cast<Layer::Kind>(kindOrdinal);

    auto layers = network->GetLayers(kind);
    if (layers.empty()) {
        throwJavaException(env, RuntimeException, "No layers of type %s found in network %s",
                           Layer::_sKindMap[kind], network->GetName());
    }

    jclass arrayListClass = env->FindClass("java/util/ArrayList");
    jmethodID arrayListCtor = env->GetMethodID(arrayListClass, "<init>", "()V");
    jmethodID arrayListAdd = env->GetMethodID(arrayListClass, "add", "(Ljava/lang/Object;)Z");
    jobject jLayers = env->NewObject(arrayListClass, arrayListCtor);

    for (const auto &layer : layers) {
        jstring jName = env->NewStringUTF(layer->GetName().c_str());
        jstring jDatasetName = env->NewStringUTF(layer->GetDataSetName().c_str());
        jint layerKind = static_cast<jint>(layer->GetKind());
        jint attributes = layer->GetAttributes();
        jint numDim = layer->GetNumDimensions();
        jint lx, ly, lz, lw;
        std::tie(lx, ly, lz, lw) = layer->GetDimensions();

        jobject jInputLayer = env->NewObject(_Layer, Layer_, jName, jDatasetName, layerKind, attributes, numDim,
                                             lx, ly, lz);

        env->CallBooleanMethod(jLayers, arrayListAdd, jInputLayer);

        env->DeleteLocalRef(jName);
        env->DeleteLocalRef(jDatasetName);
        env->DeleteLocalRef(jInputLayer);
    }

    env->DeleteLocalRef(arrayListClass);

    return jLayers;
}

void checkDataset(JNIEnv *env, DataSetBase *Dataset, uint32_t attribute, DataType dataType,
                  const DataSetDimensions &dim, uint32_t examples) {
    if (Dataset->_attributes != attribute) {
        throwJavaException(env, IllegalArgumentException, "Attribute mismatch in dataset %s", Dataset->_name.c_str());
    }
    if (Dataset->_dataType != dataType) {
        throwJavaException(env, IllegalArgumentException, "Data type mismatch in dataset %s", Dataset->_name.c_str());
    }
    if (Dataset->_dimensions != dim._dimensions) {
        throwJavaException(env, IllegalArgumentException, "Dimension mismatch in dataset %s", Dataset->_name.c_str());
    }
    if (Dataset->_width != dim._width || Dataset->_length != dim._length || Dataset->_height != dim._height) {
        throwJavaException(env, IllegalArgumentException, "Dimension mismatch in dataset %s", Dataset->_name.c_str());
    }
    if (Dataset->_examples != examples) {
        throwJavaException(env, IllegalArgumentException, "Examples mismatch in dataset %s", Dataset->_name.c_str());
    }
}

JNIEXPORT void JNICALL Java_com_system_tensorhub_tensorhub_predict(JNIEnv *env, jclass clazz, jlong ptr, jint k,
                                                                   jobjectArray jInputDatasets,
                                                                   jobjectArray jOutputDatasets) {
    tensorhubContext *dc = tensorhubContext::fromPtr(ptr);
    Network *network = dc->getNetwork();

    std::vector<jobject> jInputDatasetsElements = getArrayElements<jobject>(env, jInputDatasets);

    using DataSetEnums::DataType;

    for (const auto &jInputDataset : jInputDatasetsElements) {
        jstring jDatasetName = (jstring) env->CallObjectMethod(jInputDataset, DataSet_getName);
        jstring jLayerName = (jstring) env->CallObjectMethod(jInputDataset, DataSet_getLayerName);
        std::string datasetName = jstringToString(env, jDatasetName);
        std::string layerName = jstringToString(env, jLayerName);

        jint examples = env->CallIntMethod(jInputDataset, DataSet_getExamples);
        DataSetDimensions dim = getDataDimensions(env, jInputDataset);
        jint attribute = env->CallIntMethod(jInputDataset, DataSet_getAttribute);
        DataType dataType = static_cast<DataType>(env->CallIntMethod(jInputDataset, DataSet_getDataTypeOrdinal));

        const Layer *layer = network->GetLayer(layerName);
        if (!layer) {
            throwJavaException(env, IllegalArgumentException, "No matching layer found in network %s for dataset: %s",
                               network->GetName(), datasetName.c_str());
        }

        DataSetBase *Dataset = layer->GetDataSet();
        checkDataset(env, Dataset, attribute, dataType, dim, examples);

        jobject srcByteBuffer = env->CallObjectMethod(jInputDataset, DataSet_getData);
        const void *srcDataNative = env->GetDirectBufferAddress(srcByteBuffer);

        if (Dataset->_attributes == Attributes::Sparse) {
            jobject jSparseStart = env->CallObjectMethod(jInputDataset, DataSet_getSparseStart);
            jobject jSparseEnd = env->CallObjectMethod(jInputDataset, DataSet_getSparseEnd);
            jobject jSparseIndex = env->CallObjectMethod(jInputDataset, DataSet_getSparseIndex);

            jlong *sparseStart = env->GetPrimitiveArrayCritical<jlong>(jSparseStart, nullptr);
            jlong *sparseEnd = env->GetPrimitiveArrayCritical<jlong>(jSparseEnd, nullptr);
            jlong *sparseIndex = env->GetPrimitiveArrayCritical<jlong>(jSparseIndex, nullptr);

            Dataset->LoadSparseData(sparseStart, sparseEnd, srcDataNative, sparseIndex);

            releaseCriticalArray(env, sparseStart);
            releaseCriticalArray(env, sparseEnd);
            releaseCriticalArray(env, sparseIndex);
        } else {
            Dataset->LoadDenseData(srcDataNative);
        }

        env->DeleteLocalRef(jDatasetName);
        env->DeleteLocalRef(jLayerName);
        env->DeleteLocalRef(srcByteBuffer);
        env->DeleteLocalRef(jInputDataset);
    }

    network->SetPosition(0);
    network->PredictBatch();

    std::vector<jobject> jOutputDatasetsElements = getArrayElements<jobject>(env, jOutputDatasets);

    for (const auto &jOutputDataset : jOutputDatasetsElements) {
        jstring jLayerName = (jstring) env->CallObjectMethod(jOutputDataset, Output_getLayerName);
        std::string layerName = jstringToString(env, jLayerName);

        jfloatArray jScores = (jfloatArray) env->CallObjectMethod(jOutputDataset, Output_getScores);
        jlongArray jIndexes = (jlongArray) env->CallObjectMethod(jOutputDataset, Output_getIndexes);

        Layer *outputLayer = network->GetLayer(layerName);

        jint x, y, z, w;
        std::tie(x, y, z, w) = outputLayer->GetDimensions();
        jint stride = x * y * z;

        jfloat *scores = env->GetPrimitiveArrayCritical<jfloat>(jScores, nullptr);

        if (k > 0) {
            Float *outputUnitBuffer = network->GetUnitBuffer(layerName);

            jlong *indexes = env->GetPrimitiveArrayCritical<jlong>(jIndexes, nullptr);
            Float *dScores = dc->getOutputScoresBuffer(layerName)->_pDevData;
            uint32_t *dIndexes = dc->getOutputIndexesBuffer(layerName)->_pDevData;

            uint32_t *hIndexes = (uint32_t*) calloc(k * stride, sizeof(uint32_t));

            kCalculateTopK(outputUnitBuffer, dScores, dIndexes, stride, k);

            cudaMemcpy(scores, dScores, k * stride * sizeof(Float), cudaMemcpyDeviceToHost);
            cudaMemcpy(hIndexes, dIndexes, k * stride * sizeof(uint32_t), cudaMemcpyDeviceToHost);

            for (size_t i = 0; i < k * stride; ++i) {
                indexes[i] = hIndexes[i];
            }
            free(hIndexes);

            releaseCriticalArray(env, indexes);
        } else {
            outputLayer->GetUnits((Float*) scores);
        }

        releaseCriticalArray(env, scores);
        env->DeleteLocalRef(jLayerName);
        env->DeleteLocalRef(jScores);
        env->DeleteLocalRef(jIndexes);
        env->DeleteLocalRef(jOutputDataset);
    }

    releaseJNIObjects(env, jInputDatasetsElements);
    releaseJNIObjects(env, jOutputDatasetsElements);
}