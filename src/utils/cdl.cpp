#include "GpuTypes.h"
#include "Types.h"
#include "cdl.h"
#include <map>
#include <string>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <format>
#include <json/json.h>
#include <chrono>

/**
 * @brief Map used for mapping optimization strings to corresponding TrainingMode enum values.
 */
static const std::map<std::string, TrainingMode> sOptimizationMap = {
    {"sgd",         TrainingMode::SGD},
    {"nesterov",    TrainingMode::Nesterov}
};

CDL::CDL()
    : _randomSeed(std::chrono::system_clock::now().time_since_epoch().count()),
      _alphaInterval(0),
      _alphaMultiplier(0.5f),
      _batch(1024),
      _checkpointInterval(1),
      _checkpointFileName("check"),
      _shuffleIndexes(false),
      _resultsFileName("network.nc"),
      _alpha(0.1f),
      _lambda(0.001f),
      _mu(0.9f),
      _optimizer(TrainingMode::SGD)
{
}

int CDL::Load_JSON(const std::string& fname)
{
    // Indicates whether the network is set.
    bool networkSet = false;
    // Indicates whether the command is set.
    bool commandSet = false;
    // Indicates whether the data set is set.
    bool dataSet = false;
    // Indicates whether the number of epochs is set.
    bool epochsSet = false;

    // JSON reader for parsing the file.
    Json::CharReaderBuilder readerBuilder;
    readerBuilder["collectComments"] = false;
    std::unique_ptr<Json::CharReader> reader(readerBuilder.newCharReader());

    // JSON value to store the parsed data.
    Json::Value index;

    // Opens the JSON file for reading.
    std::ifstream stream(fname, std::ifstream::binary);
    if (!stream.is_open()) {
        std::cout << "CDL::Load_JSON: Failed to open JSON file: " << fname << '\n';
        return -1;
    }

    // Performs parsing and stores the result in the `index` variable.
    std::string errors;
    if (!reader->parse(stream, stream, &index, &errors)) {
        std::cout << "CDL::Load_JSON: Failed to parse JSON file: " << fname << ", error: " << errors << '\n';
        return -1;
    }

    // Iterates over the values in the JSON object.
    for (const auto& itr : index)
    {
        // Retrieves the name from the JSON iterator and converts it to lowercase.
        std::string name = itr.name();
        std::transform(name.begin(), name.end(), name.begin(), ::tolower);

        // Retrieves the key from the JSON iterator.
        const Json::Value& key = itr.key();

        // Retrieves the value from the JSON iterator.
        const Json::Value& value = itr;

        // Retrieves the string representation of the value, converted to lowercase.
        std::string vstring = value.isString() ? value.asString() : "";
        std::transform(vstring.begin(), vstring.end(), vstring.begin(), ::tolower);

        // Compares the name with "version" and assigns the float value.
        if (name == "version")
        {
            float version = value.asFloat();
        }
        // Compares the name with "network" and assigns the network file name.
        // Sets the networkSet flag to true.
        else if (name == "network")
        {
            _networkFileName = value.asString();
            networkSet = true;
        }
        // Compares the name with "data" and assigns the data file name.
        // Sets the dataSet flag to true.
        else if (name == "data")
        {
            _dataFileName = value.asString();
            dataSet = true;
        }
        // Compares the name with "randomseed" and assigns the random seed value.
        else if (name == "randomseed")
        {
            _randomSeed = value.asInt();
        }
        // Compares the name with "command" and assigns the appropriate mode based on the string value.
        // Sets the commandSet flag to true.
        // If the string value is not recognized, displays an error message and returns -1.
        else if (name == "command")
        {
            if (vstring == "train")
                _mode = Mode::Training;
            else if (vstring == "predict")
                _mode = Mode::Prediction;
            else if (vstring == "validate")
                _mode = Mode::Validation;
            else
            {
                std::cout << "CDL::Load_JSON: Failed to parse JSON file: " << fname << ", error: Invalid command\n";
                return -1;
            }
            commandSet = true;
        }
        else if (name == "trainingparameters")
        {
            for (const auto& pitr : value)
            {
                // Retrieves the name from the JSON iterator and converts it to lowercase.
                std::string pname = pitr.name();
                std::transform(pname.begin(), pname.end(), pname.begin(), ::tolower);

                // Retrieves the value from the JSON iterator.
                const Json::Value& pvalue = pitr;

                if (pname == "epochs")
                {
                    _epochs = pvalue.asInt();
                    epochsSet = true;
                }
                else if (pname == "alpha")
                {
                    _alpha = pvalue.asFloat();
                }
                else if (pname == "alphainterval")
                {
                    _alphaInterval = pvalue.asFloat();
                }
                else if (pname == "alphamultiplier")
                {
                    _alphaMultiplier = pvalue.asFloat();
                }
                else if (pname == "mu")
                {
                    _mu = pvalue.asFloat();
                }
                else if (pname == "lambda")
                {
                    _lambda = pvalue.asFloat();
                }
                else if (pname == "checkpointinterval")
                {
                    _checkpointInterval = pvalue.asFloat();
                }
                else if (pname == "checkpointname")
                {
                    _checkpointFileName = pvalue.asString();
                }
                else if (pname == "optimizer")
                {
                    // Retrieves the string value from the JSON iterator, converts it to lowercase, and assigns it to pstring.
                    std::string pstring = pvalue.isString() ? pvalue.asString() : "";
                    std::transform(pstring.begin(), pstring.end(), pstring.begin(), ::tolower);

                    // Searches for the lowercase pstring in the optimization map.
                    // If found, assigns the corresponding optimizer value to _optimizer.
                    // If not found, displays an error message and returns -1.
                    auto it = sOptimizationMap.find(pstring);
                    if (it != sOptimizationMap.end())
                        _optimizer = it->second;
                    else
                    {
                        std::cout << "CDL::Load_JSON: Invalid TrainingParameter Optimizer: " << pstring << '\n';
                        return -1;
                    }
                }
                else if (pname == "results")
                {
                    _resultsFileName = pvalue.asString();
                }
                else
                {
                    // Retrieves the name from the JSON iterator.
                    name = pitr.name();

                    // Displays an error message indicating an invalid TrainingParameter.
                    std::cout << "CDL::Load_JSON: Invalid TrainingParameter: " << name << '\n';
                    return -1;
                }
            }
        }
        else
        {
            // Displays an error message indicating an unknown keyword.
            std::cout << "*** CDL::Load_JSON: Unknown keyword: " << name << '\n';
            return -1;
        }
    }

    // Checks if the alpha interval is zero.
    // If true, assigns default values to _alphaInterval and _alphaMultiplier.
    if (_alphaInterval == 0)
    {
        _alphaInterval = 20;
        _alphaMultiplier = 1;
    }

    // Checks if the network is not set.
    // If true, displays an error message and returns -1.
    if (!networkSet)
    {
        std::cout << "CDL::Load_JSON: Network is required to be set, none found\n";
        return -1;
    }

    // Checks if the command is not set.
    // If true, displays an error message and returns -1.
    if (!commandSet)
    {
        std::cout << "CDL::Load_JSON: Command is required, none found\n";
        return -1;
    }

    // Checks if the data set is not set.
    // If true, displays an error message and returns -1.
    if (!dataSet)
    {
        std::cout << "CDL::Load_JSON: Data source file is required to be set\n";
        return -1;
    }

    // Checks if the mode is set to Training and the number of epochs is not set.
    // If true, displays an error message and returns -1.
    if (_mode == Mode::Training && !epochsSet)
    {
        std::cout << "CDL::Load_JSON: Mode set to Training, requires number of epochs to be set\n";
        return -1;
    }

    // Displays a success message indicating the successful parsing of the JSON file.
    std::cout << "CDL::Load_JSON: " << fname << " successfully parsed\n";
    return 0;
}
