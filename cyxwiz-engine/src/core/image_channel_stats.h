#pragma once

// Per-channel mean and standard deviation of a prepared image set (TOFIX178
// 1.3e): what Normalize receives, since the set holds the images after the
// Resize and the fixed transforms. Measured on the ArrayFire device in
// chunks; unreadable files (zero rows) are left out, as training leaves them.

#include "prepared_image_set.h"

#include <atomic>
#include <cstddef>
#include <functional>
#include <vector>

namespace cyxwiz {

struct ImageChannelStats {
    std::vector<double> mean;    // per channel, in 0..1
    std::vector<double> std_dev;
    size_t images = 0;           // the images measured
};

// progress(done, total) after each chunk; empty when cancel became true.
ImageChannelStats MeasureImageChannelStats(const PreparedImageSet& set,
                                           const std::function<void(size_t, size_t)>& progress = {},
                                           const std::atomic<bool>* cancel = nullptr);

}  // namespace cyxwiz
