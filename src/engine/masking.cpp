#include "Masking.h"

void Masking::createPaddingMask(const std::vector<float>& input, std::vector<std::vector<float>>& mask) {
    auto size = input.size();
    mask.clear();
    mask.reserve(size);

    auto i = 0;
    for (const auto& value : input) {
        if (value == 0.0f) {
            mask.emplace_back(size, 0.0f);
        } else {
            mask.emplace_back(size, 1.0f);
        }
        ++i;
    }
}
