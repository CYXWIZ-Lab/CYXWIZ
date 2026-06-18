#pragma once

#include <cstddef>
#include <vector>

namespace cyxwiz {

struct DebugSmokeSampleSelection {
    std::vector<size_t> indices;
    bool stratified = false;
};

class DebugSmokeSampleSelector {
public:
    DebugSmokeSampleSelection SelectDeterministic(
        size_t sample_count,
        size_t max_samples) const;

    DebugSmokeSampleSelection SelectStratified(
        const std::vector<int>& labels,
        size_t max_samples) const;
};

} // namespace cyxwiz
