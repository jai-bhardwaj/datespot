#ifndef ATTENTION_H
#define ATTENTION_H

#include <vector>

class Attention {
public:
    static void applyAttentionMask(const std::vector<float>& input, const std::vector<float>& mask, std::vector<float>& maskedInput);

private:
    // TODO Private member functions and variables

};

#endif  // ATTENTION_H