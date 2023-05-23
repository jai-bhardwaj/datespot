#ifndef MASKING_H
#define MASKING_H

#include <vector>

class Masking {
public:
    static std::vector<std::vector<float>> createPaddingMask(const std::vector<float>& input);
};

#endif // MASKING_H
