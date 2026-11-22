#pragma once

// A model's stored values as one flat float32 vector with a fixed layout, so
// two machines that built the same graph can exchange them (group training).
// Pure host memory; no file or network access.

#include <cyxwiz/sequential.h>

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace cyxwiz {

class IExecutableModel;

struct StateSegment {
    static constexpr size_t kNotExchanged = static_cast<size_t>(-1);

    std::string name;                  // "layer<i>.<parameter>"
    std::vector<size_t> shape;
    ModelStateRole role = ModelStateRole::Trainable;
    size_t count = 0;                  // values in the segment
    size_t offset = 0;                 // in the full state (every segment)
    size_t exchanged_offset = kNotExchanged;  // in the exchanged state; a frozen segment has none
};

struct ModelStateLayout {
    std::vector<StateSegment> segments;  // layer order, then parameter name
    size_t exchanged_count = 0;          // trainable + statistic values: what a training round moves
    size_t total_count = 0;              // + frozen values: the full state
    std::string hash;                    // SHA-256 of every segment's name, role and shape
};

// The layout of `model`. Fails for a value that is not float32.
bool DescribeModelState(IExecutableModel& model, ModelStateLayout& layout, std::string& error);

// The model's values in layout order: the full state, or without the frozen
// segments. Fails when the model no longer has this layout.
bool ReadModelState(IExecutableModel& model, const ModelStateLayout& layout, bool include_frozen,
                    std::vector<float>& values, std::string& error);

// Puts `values` (as ReadModelState returns them) into the model. Nothing is
// written unless the model has this layout, `values` has the layout's length
// and every value is a finite number.
bool WriteModelState(IExecutableModel& model, const ModelStateLayout& layout, std::span<const float> values,
                     bool include_frozen, std::string& error);

}  // namespace cyxwiz
