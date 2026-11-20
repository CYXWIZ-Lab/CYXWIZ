#pragma once

// A class label the model has no output for is refused, in every batcher
// that builds one-hot targets (TOFIX223 0.4). An all-zero target row would
// otherwise be trained on with no error: labels 1..K with K outputs, or
// -1 / +1 labels, lose a class silently.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace cyxwiz {

[[noreturn]] inline void ThrowClassLabelOutsideWidth(const char* batcher, int64_t label, size_t num_classes) {
    throw std::runtime_error(std::string(batcher) + ": class label " + std::to_string(label) +
                             " is outside the model's " + std::to_string(num_classes) +
                             " outputs (class labels must be 0 to " +
                             std::to_string(num_classes == 0 ? 0 : num_classes - 1) +
                             "; check the label column and the last layer's units)");
}

inline void CheckClassLabelInWidth(const char* batcher, int64_t label, size_t num_classes) {
    if (label < 0 || static_cast<uint64_t>(label) >= num_classes) ThrowClassLabelOutsideWidth(batcher, label, num_classes);
}

}  // namespace cyxwiz
