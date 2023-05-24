#ifndef MASKING_H
#define MASKING_H

#include <vector>

class Masking {
public:
    static std::vector<std::vector<float>> createPaddingMask(const std::vector<float>& input);
    static void createPaddingMask(const std::vector<float>& input, std::vector<std::vector<float>>& mask);
private:
    static float maskingValue_;
};

#endif // MASKING_H
