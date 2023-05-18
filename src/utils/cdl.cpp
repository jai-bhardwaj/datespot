#include "GpuTypes.h"
#include "Types.h"
#include "cdl.h"
#include <map>
#include <string>



/**
 * @brief Map used for mapping optimization strings to corresponding TrainingMode enum values.
 */
static std::map<std::string, TrainingMode> sOptimizationMap = {
    {"sgd",         TrainingMode::SGD},
    {"nesterov",    TrainingMode::Nesterov}
};


CDL::CDL()
{
    /**
     * @brief Sets the random seed to the current time.
     */
    _randomSeed = time(nullptr);

    /**
     * @brief Sets the interval for alpha.
     *        Default value: 0
     */
    _alphaInterval = 0;

    /**
     * @brief Sets the multiplier for alpha.
     *        Default value: 0.5f
     */
    _alphaMultiplier = 0.5f;

    /**
     * @brief Sets the batch size.
     *        Default value: 1024
     */
    _batch = 1024;

    /**
     * @brief Sets the checkpoint interval.
     *        Default value: 1
     */
    _checkpointInterval = 1;

    /**
     * @brief Sets the checkpoint file name.
     *        Default value: "check"
     */
    _checkpointFileName = "check";

    /**
     * @brief Indicates whether to shuffle indexes.
     *        Default value: false
     */
    _shuffleIndexes = false;

    /**
     * @brief Sets the results file name.
     *        Default value: "network.nc"
     */
    _resultsFileName = "network.nc";

    /**
     * @brief Sets the alpha value.
     *        Default value: 0.1f
     */
    _alpha = 0.1f;

    /**
     * @brief Sets the lambda value.
     *        Default value: 0.001f
     */
    _lambda = 0.001f;

    /**
     * @brief Sets the mu value.
     *        Default value: 0.9f
     */
    _mu = 0.9f;

    /**
     * @brief Sets the optimizer for training.
     *        Default value: TrainingMode::SGD
     */
    _optimizer = TrainingMode::SGD;
}


int CDL::Load_JSON(const string& fname)
{
    /**
     * @brief Indicates whether the network is set.
     *        Default value: false
     */
    bool networkSet = false;

    /**
     * @brief Indicates whether the command is set.
     *        Default value: false
     */
    bool commandSet = false;

    /**
     * @brief Indicates whether the data set is set.
     *        Default value: false
     */
    bool dataSet = false;

    /**
     * @brief Indicates whether the number of epochs is set.
     *        Default value: false
     */
    bool epochsSet = false;

    /**
     * @brief JSON reader for parsing the file.
     */
    Json::Reader reader;

    /**
     * @brief JSON value to store the parsed data.
     */
    Json::Value index;

    /**
     * @brief Opens the JSON file for reading.
     *        Performs parsing and stores the result in the `index` variable.
     *        If parsing fails, an error message is displayed and the function returns -1.
     */
    std::ifstream stream(fname, std::ifstream::binary);
    bool parsedSuccess = reader.parse(stream, index, false);
    if (!parsedSuccess)
    {
        std::cout << std::format("CDL::Load_JSON: Failed to parse JSON file: {}, error: {}\n", fname, reader.getFormattedErrorMessages());
        return -1;
    }

    /**
     * @brief Iterates over the values in the JSON object.
     *        The loop continues until the end of the JSON object is reached.
     */
    for (Json::ValueIterator itr = index.begin(); itr != index.end() ; itr++)
    {
        /**
         * @brief Retrieves the name from the JSON iterator and converts it to lowercase.
         */
        std::string name = itr.name();
        std::transform(name.begin(), name.end(), name.begin(), ::tolower);

        /**
         * @brief Retrieves the key from the JSON iterator.
         */
        Json::Value key = itr.key();

        /**
         * @brief Retrieves the value from the JSON iterator.
         */
        Json::Value value = *itr;

        /**
         * @brief Retrieves the string representation of the value, converted to lowercase.
         */
        std::string vstring = value.isString() ? value.asString() : "";
        std::transform(vstring.begin(), vstring.end(), vstring.begin(), ::tolower);

        /**
         * @brief Compares the name with "version" and assigns the float value.
         */
        if (name.compare("version") == 0)
        {
            float version = value.asFloat();
        }

        /**
         * @brief Compares the name with "network" and assigns the network file name.
         *        Sets the networkSet flag to true.
         */
        else if (name.compare("network") == 0)
        {
            _networkFileName = value.asString();
            networkSet = true;
        }

        /**
         * @brief Compares the name with "data" and assigns the data file name.
         *        Sets the dataSet flag to true.
         */
        else if (name.compare("data") == 0)
        {
            _dataFileName = value.asString();
            dataSet = true;
        }

        /**
         * @brief Compares the name with "randomseed" and assigns the random seed value.
         */
        else if (name.compare("randomseed") == 0)
        {
            _randomSeed = value.asInt();
        }

        /**
         * @brief Compares the name with "command" and assigns the appropriate mode based on the string value.
         *        Sets the commandSet flag to true.
         *        If the string value is not recognized, displays an error message and returns -1.
         */
        else if (name.compare("command") == 0)
        {
            /**
             * @brief Compares the string value with "train", "predict", and "validate" to determine the mode.
             *        Sets the appropriate mode enum value based on the comparison.
             *        If the string value is not recognized, displays an error message and returns -1.
             */
            if (vstring.compare("train") == 0)
                _mode = Mode::Training;
            else if (vstring.compare("predict") == 0)
                _mode = Mode::Prediction;
            else if (vstring.compare("validate") == 0)
                _mode = Mode::Validation;
            else
            {
                std::cout << "CDL::Load_JSON: Failed to parse JSON file: {}, error: {}\n"s
                        .format(fname, reader.getFormattedErrorMessages());
                return -1;
            }

            /**
             * @brief Sets the commandSet flag to true indicating that the command has been successfully set.
             */
            commandSet = true;
        }
        else if (name.compare("trainingparameters") == 0)
        {
            for (Json::ValueIterator pitr = value.begin(); pitr != value.end() ; pitr++)
            {
                /**
                 * @brief Retrieves the name from the JSON iterator and converts it to lowercase.
                 */
                std::string pname = pitr.name();
                std::transform(pname.begin(), pname.end(), pname.begin(), ::tolower);

                /**
                 * @brief Retrieves the key from the JSON iterator.
                 */
                Json::Value pkey = pitr.key();

                /**
                 * @brief Retrieves the value from the JSON iterator.
                 */
                Json::Value pvalue = *pitr;

                /**
                 * @brief Compares the name with "epochs" and assigns the integer value to _epochs.
                 *        Sets the epochsSet flag to true.
                 */
                if (pname.compare("epochs") == 0)
                {
                    _epochs = pvalue.asInt();
                    epochsSet = true;
                }

                /**
                 * @brief Compares the name with "alpha" and assigns the float value to _alpha.
                 */
                else if (pname.compare("alpha") == 0)
                {
                    _alpha = pvalue.asFloat();
                }

                /**
                 * @brief Compares the name with "alphainterval" and assigns the float value to _alphaInterval.
                 */
                else if (pname.compare("alphainterval") == 0)
                {
                    _alphaInterval = pvalue.asFloat();
                }

                /**
                 * @brief Compares the name with "alphamultiplier" and assigns the float value to _alphaMultiplier.
                 */
                else if (pname.compare("alphamultiplier") == 0)
                {
                    _alphaMultiplier = pvalue.asFloat();
                }

                /**
                 * @brief Compares the name with "mu" and assigns the float value to _mu.
                 */
                else if (pname.compare("mu") == 0)
                {
                    _mu = pvalue.asFloat();
                }

                /**
                 * @brief Compares the name with "lambda" and assigns the float value to _lambda.
                 */
                else if (pname.compare("lambda") == 0)
                {
                    _lambda = pvalue.asFloat();
                }

                /**
                 * @brief Compares the name with "checkpointinterval" and assigns the float value to _checkpointInterval.
                 */
                else if (pname.compare("checkpointinterval") == 0)
                {
                    _checkpointInterval = pvalue.asFloat();
                }

                /**
                 * @brief Compares the name with "checkpointname" and assigns the string value to _checkpointFileName.
                 */
                else if (pname.compare("checkpointname") == 0)
                {
                    _checkpointFileName = pvalue.asString();
                }

                /**
                 * @brief Compares the name with "optimizer" and assigns the appropriate optimizer based on the string value.
                 *        Converts the string value to lowercase before the comparison.
                 *        If the string value is recognized, assigns the corresponding optimizer value from the optimization map.
                 *        If the string value is not recognized, displays an error message and returns -1.
                 */
                else if (pname.compare("optimizer") == 0)
                {
                    /**
                     * @brief Retrieves the string value from the JSON iterator, converts it to lowercase, and assigns it to pstring.
                     */
                    std::string pstring = pvalue.isString() ? pvalue.asString() : "";
                    std::transform(pstring.begin(), pstring.end(), pstring.begin(), ::tolower);

                    /**
                     * @brief Searches for the lowercase pstring in the optimization map.
                     *        If found, assigns the corresponding optimizer value to _optimizer.
                     *        If not found, displays an error message and returns -1.
                     */
                    auto it = sOptimizationMap.find(pstring);
                    if (it != sOptimizationMap.end())
                        _optimizer = it->second;
                    else
                    {
                        /**
                         * @brief Displays an error message indicating an invalid TrainingParameter Optimizer.
                         *        Prints the invalid TrainingParameter Optimizer using std::string_view.
                         *        Returns -1 to indicate an error.
                         */
                        std::cout << "CDL::Load_JSON: Invalid TrainingParameter Optimizer: " << std::string_view(pstring) << '\n';
                        return -1;
                    }
                else if (pname.compare("results") == 0) {
                    auto _resultsFileName = pvalue.asString();
                } else {
                    /**
                     * @brief Retrieves the name from the JSON iterator.
                     */
                    name = pitr.name();

                    /**
                     * @brief Displays an error message indicating an invalid TrainingParameter.
                     *        Prints the invalid TrainingParameter using std::string_view.
                     *        Returns -1 to indicate an error.
                     */
                    std::cout << "CDL::Load_JSON: Invalid TrainingParameter: " << std::string_view(name) << '\n';
                    return -1;
                }
            }
        }
        else
        {
            /**
             * @brief Displays an error message indicating an unknown keyword.
             *        Prints the unknown keyword using std::string_view.
             *        Returns -1 to indicate an error.
             */
            std::cout << "*** CDL::Load_JSON: Unknown keyword: " << std::string_view(name) << '\n';
            return -1;
        }
    }

    /**
     * @brief Checks if the alpha interval is zero.
     *        If true, assigns default values to _alphaInterval and _alphaMultiplier.
     */
    if (_alphaInterval == 0)
    {
        _alphaInterval = 20;
        _alphaMultiplier = 1;
    }

    /**
     * @brief Checks if the network is not set.
     *        If true, displays an error message and returns -1.
     */
    if (!networkSet)
    {
        std::cout << "CDL::Load_JSON: Network is required to be set, none found\n";
        return -1;
    }

    /**
     * @brief Checks if the command is not set.
     *        If true, displays an error message and returns -1.
     */
    if (!commandSet)
    {
        std::cout << "CDL::Load_JSON: Command is required, none found\n";
        return -1;
    }

    /**
     * @brief Checks if the data set is not set.
     *        If true, displays an error message and returns -1.
     */
    if (!dataSet)
    {
        std::cout << "CDL::Load_JSON: Data source file is required to be set\n";
        return -1;
    }

    /**
     * @brief Checks if the mode is set to Training and the number of epochs is not set.
     *        If true, displays an error message and returns -1.
     */
    if (_mode == Mode::Training && !epochsSet)
    {
        std::cout << "CDL::Load_JSON: Mode set to Training, requires number of epochs to be set\n";
        return -1;
    }

    /**
     * @brief Displays a success message indicating the successful parsing of the JSON file.
     *        Returns 0 to indicate success.
     */
    std::cout << "CDL::Load_JSON: " << std::string_view(fname) << " successfully parsed\n";
    return 0;
}
