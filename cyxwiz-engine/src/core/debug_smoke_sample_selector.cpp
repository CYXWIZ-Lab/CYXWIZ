#include "debug_smoke_sample_selector.h"

#include <algorithm>
#include <map>

namespace cyxwiz {

DebugSmokeSampleSelection DebugSmokeSampleSelector::SelectDeterministic(
    size_t sample_count,
    size_t max_samples) const {
    DebugSmokeSampleSelection out;
    const size_t n = std::min(sample_count, max_samples);
    out.indices.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        out.indices.push_back(i);
    }
    return out;
}

DebugSmokeSampleSelection DebugSmokeSampleSelector::SelectStratified(
    const std::vector<int>& labels,
    size_t max_samples) const {
    DebugSmokeSampleSelection out;
    if (labels.empty() || max_samples == 0) {
        return out;
    }

    std::map<int, std::vector<size_t>> by_label;
    for (size_t i = 0; i < labels.size(); ++i) {
        by_label[labels[i]].push_back(i);
    }

    out.stratified = by_label.size() > 1;
    out.indices.reserve(std::min(labels.size(), max_samples));

    if (!out.stratified) {
        return SelectDeterministic(labels.size(), max_samples);
    }

    size_t offset = 0;
    while (out.indices.size() < max_samples) {
        bool added = false;
        for (const auto& [label, indices] : by_label) {
            (void)label;
            if (offset < indices.size()) {
                out.indices.push_back(indices[offset]);
                added = true;
                if (out.indices.size() == max_samples) {
                    break;
                }
            }
        }
        if (!added) {
            break;
        }
        ++offset;
    }

    return out;
}

} // namespace cyxwiz
