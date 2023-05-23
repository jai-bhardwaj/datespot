#ifndef MASKING_H
#define MASKING_H

#include <vector>

class Masking {
public:
    void createPaddingMask(const std::vector<float>& input, std::vector<std::vector<float>>& mask);
};

#endif // MASKING_H
