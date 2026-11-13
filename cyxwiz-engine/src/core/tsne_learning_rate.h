#pragma once

// t-SNE's learning rate: the one typed, or 0 for scikit-learn's
// learning_rate="auto" (its default since 1.2): max(rows / early
// exaggeration (12) / 4, 50). A fixed 200 overshoots on small tables and in
// 3D (TOFIX129 D8: three clusters of 60 points did not come apart). Shared by
// the t-SNE node, Explore > Projection and the word map.

#include <algorithm>
#include <cstddef>

namespace cyxwiz {

inline double TsneLearningRate(double typed, size_t rows) {
    return typed > 0.0 ? typed : std::max(static_cast<double>(rows) / 48.0, 50.0);
}

}  // namespace cyxwiz
