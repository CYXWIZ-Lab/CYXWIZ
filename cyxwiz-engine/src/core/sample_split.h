#pragma once

// The train / validation / test parts of a set of labelled samples (images,
// audio clips; TOFIX178 1.5): the split rule every training path uses
// (split_partitioning.h, by class when stratified, seeded and portable), over
// the samples training reads. The batchers, the compiler's counts and Data
// Studio's ML readiness call this one function, so they always agree.

#include "split_partitioning.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cyxwiz {

struct SampleSplitSettings {
    float train = 0.8f;
    float val = 0.1f;
    float test = 0.1f;
    bool stratified = true;
    uint32_t seed = 42;
};

struct SampleSplitParts {
    std::vector<size_t> train, val, test;  // sample numbers, in set order
};

// `usable`: the sample numbers training reads (unreadable and left-out ones
// already taken out), ascending. `labels`: every sample's class, by sample
// number (a label below 0 counts as one more class).
inline SampleSplitParts SplitSamples(const std::vector<size_t>& usable, const std::vector<int>& labels,
                                   const SampleSplitSettings& split) {
    std::vector<uint32_t> class_of_row;
    if (split.stratified) {
        uint32_t unlabeled = 0;
        for (const size_t sample : usable) {
            const int label = sample < labels.size() ? labels[sample] : -1;
            if (label >= 0) unlabeled = std::max<uint32_t>(unlabeled, static_cast<uint32_t>(label) + 1);
        }
        class_of_row.reserve(usable.size());
        for (const size_t sample : usable) {
            const int label = sample < labels.size() ? labels[sample] : -1;
            class_of_row.push_back(label >= 0 ? static_cast<uint32_t>(label) : unlabeled);
        }
    }
    const auto partitions =
        AssignSplitPartitions(usable.size(), class_of_row, split.train, split.val, split.test, true, split.seed);
    SampleSplitParts parts;
    for (size_t row = 0; row < usable.size(); ++row) {
        auto& part = partitions[row] == 0 ? parts.train : partitions[row] == 1 ? parts.val : parts.test;
        part.push_back(usable[row]);
    }
    return parts;
}

}  // namespace cyxwiz
