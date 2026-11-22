#include "model_state_layout.h"

#include "executable_model.h"
#include "sha256_digest.h"

#include <cmath>
#include <cstring>
#include <map>

namespace cyxwiz {

namespace {

const char* RoleName(ModelStateRole role) {
    switch (role) {
        case ModelStateRole::Trainable: return "trainable";
        case ModelStateRole::Statistic: return "statistic";
        case ModelStateRole::Frozen: return "frozen";
    }
    return "unknown";
}

std::string ShapeText(const std::vector<size_t>& shape) {
    std::string text;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) text += 'x';
        text += std::to_string(shape[i]);
    }
    return text;
}

// The model's entries, checked against the layout one by one.
bool MatchingState(IExecutableModel& model, const ModelStateLayout& layout, std::vector<ModelStateEntry>& entries,
                   std::string& error) {
    entries = model.GetState();
    if (entries.size() != layout.segments.size()) {
        error = "the model has " + std::to_string(entries.size()) + " stored tensors, the layout has " +
                std::to_string(layout.segments.size());
        return false;
    }
    for (size_t i = 0; i < entries.size(); ++i) {
        const ModelStateEntry& entry = entries[i];
        const StateSegment& segment = layout.segments[i];
        if (entry.name != segment.name) {
            error = "the model has " + entry.name + " where the layout has " + segment.name;
            return false;
        }
        if (entry.value.GetDataType() != DataType::Float32 || entry.value.Shape() != segment.shape) {
            error = entry.name + " has shape " + ShapeText(entry.value.Shape()) + ", the layout has " +
                    ShapeText(segment.shape);
            return false;
        }
        if (entry.role != segment.role) {
            error = entry.name + " is " + RoleName(entry.role) + ", the layout has it as " + RoleName(segment.role);
            return false;
        }
    }
    return true;
}

size_t StateLength(const ModelStateLayout& layout, bool include_frozen) {
    return include_frozen ? layout.total_count : layout.exchanged_count;
}

}  // namespace

bool DescribeModelState(IExecutableModel& model, ModelStateLayout& layout, std::string& error) {
    layout = {};
    error.clear();
    Sha256Hasher hasher;
    for (const ModelStateEntry& entry : model.GetState()) {
        if (entry.value.GetDataType() != DataType::Float32) {
            error = entry.name + " is not float32; only float32 models can be exchanged";
            return false;
        }
        StateSegment segment;
        segment.name = entry.name;
        segment.shape = entry.value.Shape();
        segment.role = entry.role;
        segment.count = entry.value.NumElements();
        segment.offset = layout.total_count;
        layout.total_count += segment.count;
        if (segment.role != ModelStateRole::Frozen) {
            segment.exchanged_offset = layout.exchanged_count;
            layout.exchanged_count += segment.count;
        }
        // One line per segment; a name holds no tab or line break.
        const std::string line = segment.name + '\t' + RoleName(segment.role) + '\t' + ShapeText(segment.shape) + '\n';
        if (!hasher.Update(line, error)) return false;
        layout.segments.push_back(std::move(segment));
    }
    return hasher.Finish(layout.hash, error);
}

bool ReadModelState(IExecutableModel& model, const ModelStateLayout& layout, bool include_frozen,
                    std::vector<float>& values, std::string& error) {
    values.clear();
    error.clear();
    std::vector<ModelStateEntry> entries;
    if (!MatchingState(model, layout, entries, error)) return false;
    values.resize(StateLength(layout, include_frozen));
    for (size_t i = 0; i < entries.size(); ++i) {
        const StateSegment& segment = layout.segments[i];
        if (segment.count == 0) continue;
        if (segment.role == ModelStateRole::Frozen && !include_frozen) continue;
        const size_t offset = include_frozen ? segment.offset : segment.exchanged_offset;
        std::memcpy(values.data() + offset, entries[i].value.ReadData<float>(), segment.count * sizeof(float));
    }
    return true;
}

bool WriteModelState(IExecutableModel& model, const ModelStateLayout& layout, std::span<const float> values,
                     bool include_frozen, std::string& error) {
    error.clear();
    if (values.size() != StateLength(layout, include_frozen)) {
        error = "the state has " + std::to_string(values.size()) + " values, the layout needs " +
                std::to_string(StateLength(layout, include_frozen));
        return false;
    }
    std::vector<ModelStateEntry> entries;
    if (!MatchingState(model, layout, entries, error)) return false;

    std::map<std::string, Tensor> tensors;
    for (const StateSegment& segment : layout.segments) {
        if (segment.role == ModelStateRole::Frozen && !include_frozen) continue;
        const float* first = values.data() + (include_frozen ? segment.offset : segment.exchanged_offset);
        for (size_t i = 0; i < segment.count; ++i) {
            if (!std::isfinite(first[i])) {
                error = segment.name + " holds a value that is not a finite number";
                return false;
            }
        }
        tensors.emplace(segment.name, Tensor(segment.shape, first, DataType::Float32));
    }
    model.SetParameters(tensors);
    return true;
}

}  // namespace cyxwiz
