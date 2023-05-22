#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <memory>
#include <variant>

#include "GpuTypes.h"
#include "Types.h"
#include "CDFhelper.h"

#define TEST_DATA_PATH "../../../../tests/test_data/"

struct DataParameters {
    constexpr static int numberOfSamples = 1024;
    constexpr static int inpFeatureDimensionality = 1;
    constexpr static int outFeatureDimensionality = 1;
    constexpr static float W0 = -2.f;
    constexpr static float B0 = 3.f;
};

/**
 * Generates test data.
 *
 * @param path The path to save the generated test data.
 * @param testDataType The type of test data to generate.
 * @param dataParameters The parameters for generating the test data.
 * @param out The output stream to write log messages.
 */
void generateTestData(const std::string& path, const std::variant<int, std::string>& testDataType, const DataParameters& dataParameters, std::ostream& out) {
    std::vector<std::vector<unsigned int>> vSampleTestInput, vSampleTestInputTime;
    std::vector<std::vector<float>> vSampleTestInputData;
    std::vector<std::vector<unsigned int>> vSampleTestOutput, vSampleTestOutputTime;
    std::vector<std::vector<float>> vSampleTestOutputData;
    std::vector<std::string> vSamplesName(dataParameters.numberOfSamples);
    std::map<std::string, unsigned int> mFeatureNameToIndex;

    for (int d = 0; d < dataParameters.inpFeatureDimensionality; d++) {
        std::string feature_name = "feature" + std::to_string(d);
        mFeatureNameToIndex[feature_name] = d;
    }

    for (int s = 0; s < dataParameters.numberOfSamples; s++) {
        vSamplesName[s] = "sample" + std::to_string(s);
    }

    for (int s = 0; s < dataParameters.numberOfSamples; s++) {
        std::vector<unsigned int> inpFeatureIndex, inpTime;
        std::vector<float> inpFeatureValue;
        std::vector<unsigned int> outFeatureIndex, outTime;
        std::vector<float> outFeatureValue;

        std::visit([&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, int>) {
                if (arg == 1) {
                    for (int d = 0; d < dataParameters.inpFeatureDimensionality; d++) {
                        inpFeatureIndex.push_back(d);
                        inpFeatureValue.push_back(static_cast<float>(s));
                        inpTime.push_back(s);
                    }

                    for (int d = 0; d < dataParameters.outFeatureDimensionality; d++) {
                        outFeatureIndex.push_back(d);
                        outFeatureValue.push_back(dataParameters.W0 * inpFeatureValue[d] + dataParameters.B0);
                        outTime.push_back(s);
                    }

                    vSampleTestInput.push_back(inpFeatureIndex);
                    vSampleTestInputData.push_back(inpFeatureValue);
                    vSampleTestInputTime.push_back(inpTime);
                    vSampleTestOutput.push_back(outFeatureIndex);
                    vSampleTestOutputData.push_back(outFeatureValue);
                    vSampleTestOutputTime.push_back(outTime);
                } else if (arg == 2) {
                    inpFeatureIndex.push_back(s % dataParameters.inpFeatureDimensionality);
                    inpTime.push_back(s);

                    outFeatureIndex.push_back(s % dataParameters.outFeatureDimensionality);
                    outTime.push_back(s);

                    vSampleTestInput.push_back(inpFeatureIndex);
                    vSampleTestInputTime.push_back(inpTime);
                    vSampleTestOutput.push_back(outFeatureIndex);
                    vSampleTestOutputTime.push_back(outTime);
                } else if (arg == 3) {
                    inpFeatureIndex.push_back(s % dataParameters.inpFeatureDimensionality);
                    inpFeatureValue.push_back(static_cast<float>(s));
                    inpTime.push_back(s);

                    for (int d = 0; d < dataParameters.outFeatureDimensionality; d++) {
                        outFeatureIndex.push_back(d);
                        outFeatureValue.push_back(((s + d) % 2) + 1);
                        outTime.push_back(s);
                    }
                    vSampleTestInput.push_back(inpFeatureIndex);
                    vSampleTestInputData.push_back(inpFeatureValue);
                    vSampleTestInputTime.push_back(inpTime);

                    vSampleTestOutput.push_back(outFeatureIndex);
                    vSampleTestOutputData.push_back(outFeatureValue);
                    vSampleTestOutputTime.push_back(outTime);
                } else {
                    out << "unsupported mode";
                    exit(2);
                }
            }
        }, testDataType);
    }

    int minInpDate = std::numeric_limits<int>::max(), maxInpDate = std::numeric_limits<int>::min();
    int minOutDate = std::numeric_limits<int>::max(), maxOutDate = std::numeric_limits<int>::min();
    const bool alignFeatureNumber = false;
    writeNETCDF(path + "test.nc", vSamplesName, mFeatureNameToIndex, vSampleTestInput, vSampleTestInputTime,
        vSampleTestInputData, mFeatureNameToIndex, vSampleTestOutput, vSampleTestOutputTime,
        vSampleTestOutputData, minInpDate, maxInpDate, minOutDate, maxOutDate, alignFeatureNumber, 2);
}

/**
 * Validates a neural network model.
 *
 * @param batch The batch size.
 * @param modelPath The path to the neural network model file.
 * @param testDataType The type of test data to use for validation.
 * @param dataParameters The parameters for generating the test data.
 * @param out The output stream to write log messages.
 *
 * @return true if the validation is successful, false otherwise.
 */
bool validateNeuralNetwork(const size_t batch, const std::string& modelPath, const std::variant<int, std::string>& testDataType, const DataParameters& dataParameters, std::ostream& out) {
    out << "start validation of " << modelPath << std::endl;

    std::unique_ptr<Network> pNetwork = nullptr;
    std::vector<std::unique_ptr<DataSetBase>> vDataSet;
    const std::string dataName = "test.nc";
    const std::string dataPath(TEST_DATA_PATH);
    generateTestData(dataPath, testDataType, dataParameters, out);
    vDataSet = LoadNetCDF(dataPath + dataName);
    pNetwork = std::make_unique<Network>(LoadNeuralNetworkJSON(modelPath, batch, vDataSet.get()));
    pNetwork->LoadDataSets(vDataSet.get());
    pNetwork->SetCheckpoint("check", 1);
    pNetwork->SetTrainingMode(SGD);
    bool valid = pNetwork->Validate();
    if (valid) {
        out << "SUCCESSFUL validation" << std::endl;
    } else {
        out << "FAILED validation" << std::endl;
    }

    int totalGPUMemory, totalCPUMemory;
    getGpu().GetMemoryUsage(&totalGPUMemory, &totalCPUMemory);
    out << "GPU Memory Usage: " << totalGPUMemory << " KB" << std::endl;
    out << "CPU Memory Usage: " << totalCPUMemory << " KB" << std::endl;

    return valid;
}
