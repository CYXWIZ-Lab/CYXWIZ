#pragma once

// How an image dataset splits into train / validation / test before
// training (TOFIX178 1.5): its index table (path, label, readable) through
// the same rule training uses (SplitSamples), so the counts the compiler and
// Data Studio show are the parts training reads.

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace cyxwiz {

struct SampleSplitSettings;  // sample_split.h (Arrow headers; this one is light)

struct ImageSplitCounts {
    bool known = false;
    std::string reason;                       // why not, when not known
    std::vector<std::string> class_names;     // as the labels give them, by class index
    std::vector<std::string> shown_names;     // as the screens show them (renamed on the Data Input)
    size_t images = 0;                        // the images training reads
    size_t unreadable = 0;                    // left out: the file cannot be read
    size_t left_out = 0;                      // left out: a Quality node's rejects
    std::array<size_t, 3> parts{};            // train, validation, test
    std::vector<std::array<size_t, 3>> per_class;  // [class][part]; an unlabelled class last
};

// `image_dataset`: the image Data Input's dataset name (its index table is
// "<name>_index" in the DataRegistry). `excluded_files`: the paths a Quality
// node leaves out.
ImageSplitCounts CountImageSplit(const std::string& image_dataset, const SampleSplitSettings& split,
                                 const std::vector<std::string>& excluded_files = {});

}  // namespace cyxwiz
