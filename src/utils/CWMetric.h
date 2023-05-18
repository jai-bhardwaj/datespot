#ifndef CW_METRIC_H
#define CW_METRIC_H

#include <string_view>

class CWMetric
{
public:
    /**
     * @brief Updates the metrics with the given metric and value.
     * 
     * @param metric The metric to update.
     * @param value The value of the metric.
     */
    void updateMetrics(std::string_view metric, std::string_view value);
};

#endif // CW_METRIC_H