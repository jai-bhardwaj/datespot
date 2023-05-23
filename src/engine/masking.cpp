#include "Masking.h"

std::vector<std::vector<float>> Masking::createPaddingMask(const std::vector<float>& input) {
    std::vector<std::vector<float>> mask(input.size(), std::vector<float>(input.size(), 1.0f));

    for (int i = 0; i < input.size(); ++i) {
        if (input[i] == 0.0f) {
            for (int j = 0; j < input.size(); ++j) {
                mask[i][j] = 0.0f;
                mask[j][i] = 0.0f;
            }
        }
    }

    return mask;
}
