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
    : _randomSeed(std::chrono::steady_clock::now().time_since_epoch().count()),
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
        std::cout << std::format("CDL::Load_JSON: Failed to open JSON file: {}\n", fname);
        return -1;
    }

    // Performs parsing and stores the result in the `index` variable.
    std::string errors;
    if (!reader->parse(stream, stream, &index, &errors)) {
        std::cout << std::format("CDL::Load_JSON: Failed to parse JSON file: {}, error: {}\n", fname, errors);
        return -1;
    }

    // Iterates over the values in the JSON object.
    for (const auto& [name, value] : index)
    {
        // Converts the name to lowercase.
        std::string lname = name;
        std::transform(lname.begin(), lname.end(), lname.begin(), ::tolower);

        // Converts the string value to lowercase.
        std::string vstring = value.isString() ? value.asString() : "";
        std::transform(vstring.begin(), vstring.end(), vstring.begin(), ::tolower);

        // Compares the name with "version" and assigns the float value.
        if (lname == "version")
        {
            float version = value.asFloat();
        }
        // Compares the name with "network" and assigns the network file name.
        // Sets the networkSet flag to true.
        else if (lname == "network")
        {
            _networkFileName = value.asString();
            networkSet = true;
        }
        // Compares the name with "data" and assigns the data file name.
        // Sets the dataSet flag to true.
        else if (lname == "data")
        {
            _dataFileName = value.asString();
            dataSet = true;
        }
        // Compares the name with "randomseed" and assigns the random seed value.
        else if (lname == "randomseed")
        {
            _randomSeed = value.asInt();
        }
        // Compares the name with "command" and assigns the appropriate mode based on the string value.
        // Sets the commandSet flag to true.
        // If the string value is not recognized, displays an error message and returns -1.
        else if (lname == "command")
        {
            if (vstring == "train")
                _mode = Mode::Training;
            else if (vstring == "predict")
                _mode = Mode::Prediction;
            else if (vstring == "validate")
                _mode = Mode::Validation;
            else
            {
                std::cout << std::format("CDL::Load_JSON: Failed to parse JSON file: {}, error: Invalid command\n", fname);
                return -1;
            }
            commandSet = true;
        }
        else if (lname == "trainingparameters")
        {
            for (const auto& [pname, pvalue] : value)
            {
                // Converts the pname to lowercase.
                std::string lpname = pname;
                std::transform(lpname.begin(), lpname.end(), lpname.begin(), ::tolower);

                if (lpname == "epochs")
                {
                    _epochs = pvalue.asInt();
                    epochsSet = true;
                }
                else if (lpname == "alpha")
                {
                    _alpha = pvalue.asFloat();
                }
                else if (lpname == "alphainterval")
                {
                    _alphaInterval = pvalue.asFloat();
                }
                else if (lpname == "alphamultiplier")
                {
                    _alphaMultiplier = pvalue.asFloat();
                }
                else if (lpname == "mu")
                {
                    _mu = pvalue.asFloat();
                }
                else if (lpname == "lambda")
                {
                    _lambda = pvalue.asFloat();
                }
                else if (lpname == "checkpointinterval")
                {
                    _checkpointInterval = pvalue.asFloat();
                }
                else if (lpname == "checkpointname")
                {
                    _checkpointFileName = pvalue.asString();
                }
                else if (lpname == "optimizer")
                {
                    // Converts the string value to lowercase.
                    std::string pstring = pvalue.isString() ? pvalue.asString() : "";
                    std::transform(pstring.begin(), pstring.end(), pstring.begin(), ::tolower);

                    // Checks if the optimization map contains the lowercase pstring.
                    // If found, assigns the corresponding optimizer value to _optimizer.
                    // If not found, displays an error message and returns -1.
                    if (sOptimizationMap.contains(pstring))
                        _optimizer = sOptimizationMap.at(pstring);
                    else
                    {
                        std::cout << std::format("CDL::Load_JSON: Invalid TrainingParameter Optimizer: {}\n", pstring);
                        return -1;
                    }
                }
                else if (lpname == "results")
                {
                    _resultsFileName = pvalue.asString();
                }
                else
                {
                    // Displays an error message indicating an invalid TrainingParameter.
                    std::cout << std::format("CDL::Load_JSON: Invalid TrainingParameter: {}\n", pname);
                    return -1;
                }
            }
        }
        else
        {
            // Displays an error message indicating an unknown keyword.
            std::cout << std::format("*** CDL::Load_JSON: Unknown keyword: {}\n", name);
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
    std::cout << std::format("CDL::Load_JSON: {} successfully parsed\n", fname);
    return 0;
}
