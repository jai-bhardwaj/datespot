#include <gtest/gtest.h>
#include "GpuTypes.h"
#include "Types.h"
#include "Layer.h"

class TestDataSetFixture : public ::testing::Test {
protected:
    DataSetDimensions datasetDim = DataSetDimensions(128, 1, 1);
    uint32_t examples = 32;
    uint32_t uniqueExamples = 16;
    double sparseDensity = 0.1;
    size_t stride = datasetDim._height * datasetDim._width * datasetDim._length;
    size_t dataLength = stride * examples;

    void SetUp() override {
    }

    void TearDown() override {
    }
};

TEST_F(TestDataSetFixture, testCreateDenseDataset) {
    DataSet<uint32_t> dataset(32, datasetDim);

    EXPECT_EQ(dataset._stride, 128);
    EXPECT_FALSE(dataset._bIndexed);
    EXPECT_FALSE(dataset._bStreaming);
    EXPECT_EQ(dataset._dimensions, 1);
    EXPECT_EQ(dataset._attributes, DataSetEnums::None);
    EXPECT_EQ(dataset._dataType, DataSetEnums::DataType::UInt);
    EXPECT_EQ(dataset._width, 128);
    EXPECT_EQ(dataset._height, 1);
    EXPECT_EQ(dataset._length, 1);
    EXPECT_EQ(dataset._examples, 32);
    EXPECT_EQ(dataset._uniqueExamples, 32);
    EXPECT_EQ(dataset._localExamples, 32);
    EXPECT_EQ(dataset._sparseDataSize, 0);
}

TEST_F(TestDataSetFixture, testCreateDenseIndexedDataset) {
    DataSet<uint32_t> dataset(examples, uniqueExamples, datasetDim);

    EXPECT_EQ(dataset._stride, 128);
    EXPECT_TRUE(dataset._bIndexed);
    EXPECT_FALSE(dataset._bStreaming);
    EXPECT_EQ(dataset._dimensions, 1);
    EXPECT_EQ(dataset._attributes, DataSetEnums::Indexed);
    EXPECT_EQ(dataset._dataType, DataSetEnums::DataType::UInt);
    EXPECT_EQ(dataset._width, 128);
    EXPECT_EQ(dataset._height, 1);
    EXPECT_EQ(dataset._length, 1);
    EXPECT_EQ(dataset._examples, examples);
    EXPECT_EQ(dataset._uniqueExamples, uniqueExamples);
    EXPECT_EQ(dataset._localExamples, examples);
    EXPECT_EQ(dataset._sparseDataSize, 0);
}

TEST_F(TestDataSetFixture, testCreateSparseDataset) {
    bool isWeighted = false;
    DataSet<int> dataset(examples, sparseDensity, datasetDim, isWeighted);

    EXPECT_EQ(dataset._stride, 0);
    EXPECT_FALSE(dataset._bIndexed);
    EXPECT_FALSE(dataset._bStreaming);
    EXPECT_EQ(dataset._dimensions, 1);
    EXPECT_EQ(dataset._attributes, DataSetEnums::Sparse);
    EXPECT_EQ(dataset._dataType, DataSetEnums::DataType::Int);
    EXPECT_EQ(dataset._width, 128);
    EXPECT_EQ(dataset._height, 1);
    EXPECT_EQ(dataset._length, 1);
    EXPECT_EQ(dataset._examples, examples);
    EXPECT_EQ(dataset._uniqueExamples, examples);
    EXPECT_EQ(dataset._localExamples, examples);
    EXPECT_EQ(dataset._sparseDataSize, static_cast<uint64_t>(128.0 * 0.1 * 32.0));
}

TEST_F(TestDataSetFixture, testCreateSparseWeightedDataset) {
    bool isWeighted = true;
    DataSet<int> dataset(examples, sparseDensity, datasetDim, isWeighted);

    EXPECT_EQ(dataset._stride, 0);
    EXPECT_FALSE(dataset._bIndexed);
    EXPECT_FALSE(dataset._bStreaming);
    EXPECT_EQ(dataset._dimensions, 1);
    EXPECT_EQ(dataset._attributes, (DataSetEnums::Sparse | DataSetEnums::Weighted));
    EXPECT_EQ(dataset._dataType, DataSetEnums::DataType::Int);
    EXPECT_EQ(dataset._width, 128);
    EXPECT_EQ(dataset._height, 1);
    EXPECT_EQ(dataset._length, 1);
    EXPECT_EQ(dataset._examples, examples);
    EXPECT_EQ(dataset._uniqueExamples, examples);
    EXPECT_EQ(dataset._localExamples, examples);
    EXPECT_EQ(dataset._sparseDataSize, static_cast<uint64_t>(128.0 * 0.1 * 32.0));
}

TEST_F(TestDataSetFixture, testCreateSparseIndexedDataset) {
    size_t sparseDataSize = 128 * uniqueExamples / 10;
    bool isIndexed = true;
    bool isWeighted = false;
    DataSet<long> dataset(examples, uniqueExamples, sparseDataSize, datasetDim, isIndexed, isWeighted);

    EXPECT_EQ(dataset._stride, 0);
    EXPECT_TRUE(dataset._bIndexed);
    EXPECT_FALSE(dataset._bStreaming);
    EXPECT_EQ(dataset._dimensions, 1);
    EXPECT_EQ(dataset._attributes, (DataSetEnums::Sparse | DataSetEnums::Indexed));
    EXPECT_EQ(dataset._dataType, DataSetEnums::DataType::LLInt);
    EXPECT_EQ(dataset._width, 128);
    EXPECT_EQ(dataset._height, 1);
    EXPECT_EQ(dataset._length, 1);
    EXPECT_EQ(dataset._examples, examples);
    EXPECT_EQ(dataset._uniqueExamples, uniqueExamples);
    EXPECT_EQ(dataset._localExamples, examples);
    EXPECT_EQ(dataset._sparseDataSize, sparseDataSize);
}

TEST_F(TestDataSetFixture, testCreateSparseWeightedIndexedDataset) {
    size_t sparseDataSize = 128 * uniqueExamples / 10;
    bool isIndexed = true;
    bool isWeighted = true;
    DataSet<long> dataset(examples, uniqueExamples, sparseDataSize, datasetDim, isIndexed, isWeighted);

    EXPECT_EQ(dataset._stride, 0);
    EXPECT_TRUE(dataset._bIndexed);
    EXPECT_FALSE(dataset._bStreaming);
    EXPECT_EQ(dataset._dimensions, 1);
    EXPECT_EQ(dataset._attributes, (DataSetEnums::Sparse | DataSetEnums::Indexed | DataSetEnums::Weighted));
    EXPECT_EQ(dataset._dataType, DataSetEnums::DataType::LLInt);
    EXPECT_EQ(dataset._width, 128);
    EXPECT_EQ(dataset._height, 1);
    EXPECT_EQ(dataset._length, 1);
    EXPECT_EQ(dataset._examples, examples);
    EXPECT_EQ(dataset._uniqueExamples, uniqueExamples);
    EXPECT_EQ(dataset._localExamples, examples);
    EXPECT_EQ(dataset._sparseDataSize, static_cast<uint64_t>(128.0 * 0.1 * 16.0));
}

TEST_F(TestDataSetFixture, testLoadDenseData) {
    DataSet<uint32_t> dataset(examples, datasetDim);

    uint32_t srcData[dataLength];
    for (size_t i = 0; i < dataLength; ++i) {
        srcData[i] = i;
    }

    dataset.LoadDenseData(srcData);

    for (size_t i = 0; i < examples; ++i) {
        for (size_t j = 0; j < stride; ++j) {
            EXPECT_EQ(static_cast<uint32_t>(i * stride + j), dataset.GetDataPoint(i, j));
        }
    }
}

TEST_F(TestDataSetFixture, testSetDenseData_OnSparseDataset) {
    DataSet<uint32_t> dataset(examples, sparseDensity, datasetDim, false);
    uint32_t srcData[dataLength];
    EXPECT_THROW(dataset.LoadDenseData(srcData), std::runtime_error);
}

TEST_F(TestDataSetFixture, testLoadSparseData) {
    DataSet<Float> dataset(examples, sparseDensity, datasetDim, false);

    DataSetDimensions dim = dataset.GetDimensions();
    size_t sparseDataSize = static_cast<size_t>(((double) dim._height * dim._width * dim._length) * examples * sparseDensity);
    uint64_t sparseStart[examples];
    uint64_t sparseEnd[examples];
    Float sparseData[sparseDataSize];
    uint32_t sparseIndex[sparseDataSize];

    size_t sparseExampleSize = (sparseDataSize + examples - 1) / examples;

    sparseStart[0] = 0;
    sparseEnd[0] = sparseDataSize - (sparseExampleSize * (examples - 1));
    for (uint32_t i = 1; i < examples; i++) {
        sparseStart[i] = sparseEnd[i - 1];
        sparseEnd[i] = sparseStart[i] + sparseExampleSize;
    }

    for (uint32_t i = 0; i < sparseDataSize; ++i) {
        sparseData[i] = static_cast<Float>(i + 1);
    }

    for (size_t i = 0; i < sparseDataSize; ++i) {
        sparseIndex[i] = static_cast<uint32_t>(i);
    }

    dataset.LoadSparseData(sparseStart, sparseEnd, sparseData, sparseIndex);

    EXPECT_EQ(sparseEnd[0], dataset.GetSparseDataPoints(0));
    for (uint32_t i = 0; i < sparseEnd[0]; ++i) {
        EXPECT_EQ(static_cast<Float>(i + 1), dataset.GetSparseDataPoint(0, i));
        EXPECT_EQ(static_cast<uint32_t>(i), dataset.GetSparseIndex(0, i));
    }
}

TEST_F(TestDataSetFixture, testLoadSparseData_Overflow) {
    DataSet<Float> dataset(examples, sparseDensity, datasetDim, false);

    DataSetDimensions dim = dataset.GetDimensions();
    size_t sparseDataSize = static_cast<size_t>(((double) dim._height * dim._width * dim._length) * examples * sparseDensity);
    uint64_t sparseStart[examples];
    sparseStart[0] = 0;
    uint64_t sparseEnd[examples];
    Float sparseData[1];
    uint32_t sparseIndex[1];
    sparseEnd[examples - 1] = sparseDataSize + 1;
    EXPECT_THROW(dataset.LoadSparseData(sparseStart, sparseEnd, sparseData, sparseIndex), std::length_error);
}

TEST_F(TestDataSetFixture, testLoadSparseData_SparseStartNotZeroIndexed) {
    DataSet<Float> dataset(examples, sparseDensity, datasetDim, false);

    DataSetDimensions dim = dataset.GetDimensions();
    size_t sparseDataSize = static_cast<size_t>(((double) dim._height * dim._width * dim._length) * examples * sparseDensity);
    uint64_t sparseStart[examples];
    sparseStart[0] = 1;
    uint64_t sparseEnd[examples];
    Float sparseData[1];
    uint32_t sparseIndex[1];
    sparseEnd[examples - 1] = sparseDataSize + 1;
    EXPECT_THROW(dataset.LoadSparseData(sparseStart, sparseEnd, sparseData, sparseIndex), std::runtime_error);
}

TEST_F(TestDataSetFixture, testLoadSparseData_OnDenseDataset) {
    DataSet<Float> dataset(examples, datasetDim);
    uint64_t sparseStart[examples];
    uint64_t sparseEnd[examples];
    Float sparseData[1];
    uint32_t sparseIndex[1];
    EXPECT_THROW(dataset.LoadSparseData(sparseStart, sparseEnd, sparseData, sparseIndex), std::runtime_error);
}

TEST_F(TestDataSetFixture, testLoadIndexedData) {
    DataSet<Float> dataset(examples, uniqueExamples, datasetDim);
    uint32_t indexedData[uniqueExamples];
    for (uint32_t i = 0; i < uniqueExamples; ++i) {
        indexedData[i] = i;
    }

    dataset.LoadIndexedData(indexedData);

    for (uint32_t i = 0; i < uniqueExamples; ++i) {
        EXPECT_EQ(i, dataset._vIndex[i]);
    }
}

TEST_F(TestDataSetFixture, testLoadIndexedData_OotIndexedDataset) {
    DataSet<Float> dataset(examples, datasetDim);
    uint32_t indexedData[examples];
    EXPECT_THROW(dataset.LoadIndexedData(indexedData), std::runtime_error);
}

TEST_F(TestDataSetFixture, testLoadDataWeights) {
    DataSet<uint32_t> dataset(examples, sparseDensity, datasetDim, true);

    Float dataWeights[examples];
    for (uint32_t i = 0; i < examples; ++i) {
        dataWeights[i] = static_cast<Float>(i);
    }

    dataset.LoadDataWeight(dataWeights);

    for (uint32_t i = 0; i < examples; ++i) {
        EXPECT_EQ(static_cast<Float>(i), dataset._vDataWeight[i]);
    }
}

TEST_F(TestDataSetFixture, testLoadDataWeights_OotWeightedDataset) {
    DataSet<uint32_t> dataset(examples, sparseDensity, datasetDim, false);

    Float dataWeights[examples];
    EXPECT_THROW(dataset.LoadDataWeight(dataWeights), std::runtime_error);
}

TEST_F(TestDataSetFixture, testDataSetTypes) {
    DataSet<Float> floatDataset(examples, datasetDim);
    DataSet<double> doubleDataset(examples, datasetDim);
    DataSet<unsigned char> unsignedCharDataset(examples, datasetDim);
    DataSet<char> charDataset(examples, datasetDim);
    DataSet<uint32_t> unsignedIntDataset(examples, datasetDim);
    DataSet<uint64_t> unsignedLongDataset(examples, datasetDim);
    DataSet<int32_t> intDataset(examples, datasetDim);
    DataSet<int64_t> longDataset(examples, datasetDim);

}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

