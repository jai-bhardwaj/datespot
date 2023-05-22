#ifndef CW_METRIC_H
#define CW_METRIC_H

#include <string_view>
#include <unordered_map>

class CWMetric
{
public:
    /**
     * @brief Updates the metrics with the given metric and value.
     * 
     * @param metric The metric to update.
     * @param value The value of the metric.
     */
    void updateMetrics(std::string_view metric, std::string_view value)
    {
        metrics[std::string(metric)] = std::string(value);
    }

private:
    std::unordered_map<std::string, std::string> metrics;
};

#endif // CW_METRIC_H