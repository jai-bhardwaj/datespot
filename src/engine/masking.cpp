#include "Masking.h"
#include <vector>
#include <algorithm>
#include <iterator>

class Masking {
public:
    static void createPaddingMask(const std::vector<float>& input, std::vector<std::vector<float>>& mask) {
        mask.clear();
        mask.reserve(input.size());

        std::transform(input.begin(), input.end(), std::back_inserter(mask),
            [&input](float value) { return value == 0.0f ? std::vector<float>(input.size(), 0.0f) : std::vector<float>(input.size(), 1.0f); });
    }
};

