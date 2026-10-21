#include "cyxwiz/dataloader.h"

#include <random>

namespace cyxwiz {

// ============================================================================
// SyntheticDataset Implementation
// ============================================================================

SyntheticDataset::SyntheticDataset(size_t num_samples, const std::vector<size_t>& input_shape,
                                   size_t num_classes, int seed)
    : num_samples_(num_samples), shape_(input_shape), num_classes_(num_classes), rng_(seed)
{
    sample_size_ = 1;
    for (size_t dim : shape_) {
        sample_size_ *= dim;
    }
}

std::pair<std::vector<float>, int> SyntheticDataset::GetItem(size_t index) const {
    // Generate deterministic random data based on index
    std::mt19937 local_rng(static_cast<unsigned>(index));
    std::uniform_real_distribution<float> data_dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> label_dist(0, static_cast<int>(num_classes_) - 1);

    std::vector<float> data(sample_size_);
    for (size_t i = 0; i < sample_size_; i++) {
        data[i] = data_dist(local_rng);
    }

    int label = label_dist(local_rng);
    return {data, label};
}

} // namespace cyxwiz
