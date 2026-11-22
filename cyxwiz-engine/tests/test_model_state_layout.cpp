#include "../src/core/executable_model.h"
#include "../src/core/model_state_layout.h"

#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

using cyxwiz::ModelStateLayout;
using cyxwiz::ModelStateRole;
using cyxwiz::StateSegment;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

// layer0 Linear(3->2), layer1 ReLU, layer2 BatchNorm(2), layers 3..9 ReLU,
// layer10 Linear(2->1). Eleven layers, so "layer10" sorts before "layer2" in a
// name-ordered map and the layout has to restore the layer order.
std::shared_ptr<cyxwiz::SequentialModel> BuildModel(bool freeze_last, size_t hidden = 2) {
    auto model = std::make_shared<cyxwiz::SequentialModel>();
    model->Add<cyxwiz::LinearModule>(3, hidden);
    model->Add<cyxwiz::ReLUModule>();
    model->Add<cyxwiz::BatchNormModule>(hidden);
    for (int i = 3; i <= 9; ++i) model->Add<cyxwiz::ReLUModule>();
    model->Add<cyxwiz::LinearModule>(hidden, 1);
    if (freeze_last) model->GetModule(10)->Freeze();
    return model;
}

const StateSegment& Segment(const ModelStateLayout& layout, const std::string& name) {
    for (const StateSegment& segment : layout.segments) {
        if (segment.name == name) return segment;
    }
    std::cerr << "FAIL: the layout has no segment " << name << "\n";
    std::exit(1);
}

ModelStateLayout Describe(cyxwiz::IExecutableModel& model) {
    ModelStateLayout layout;
    std::string error;
    Check(cyxwiz::DescribeModelState(model, layout, error), "the layout should be described: " + error);
    return layout;
}

std::vector<float> Read(cyxwiz::IExecutableModel& model, const ModelStateLayout& layout, bool include_frozen) {
    std::vector<float> values;
    std::string error;
    Check(cyxwiz::ReadModelState(model, layout, include_frozen, values, error), "the state should be read: " + error);
    return values;
}

} // namespace

int main() {
    cyxwiz::SequentialExecutableModel model(BuildModel(true));
    const ModelStateLayout layout = Describe(model);

    // Order, roles and counts.
    {
        std::vector<std::string> layers;
        for (const StateSegment& segment : layout.segments) {
            const std::string layer = segment.name.substr(0, segment.name.find('.'));
            if (layers.empty() || layers.back() != layer) layers.push_back(layer);
        }
        Check(layers == std::vector<std::string>({"layer0", "layer2", "layer10"}),
              "segments should be in layer order, not in name order");

        const StateSegment& mean = Segment(layout, "layer2.running_mean");
        const StateSegment& variance = Segment(layout, "layer2.running_var");
        Check(mean.role == ModelStateRole::Statistic && variance.role == ModelStateRole::Statistic,
              "BatchNorm's running mean and variance should be statistics");
        Check(Segment(layout, "layer2.gamma").role == ModelStateRole::Trainable, "BatchNorm's gamma should be trainable");
        Check(mean.shape == std::vector<size_t>({2}) && mean.count == 2, "the running mean should hold one value per feature");

        size_t total = 0;
        size_t exchanged = 0;
        size_t frozen_segments = 0;
        for (const StateSegment& segment : layout.segments) {
            Check(segment.offset == total, segment.name + " should start where the segment before it ends");
            total += segment.count;
            const bool frozen = segment.role == ModelStateRole::Frozen;
            Check(frozen == (segment.name.rfind("layer10.", 0) == 0), segment.name + " should be frozen only in the frozen layer");
            if (frozen) {
                ++frozen_segments;
                Check(segment.exchanged_offset == StateSegment::kNotExchanged, "a frozen segment should have no exchanged offset");
            } else {
                Check(segment.exchanged_offset == exchanged, segment.name + " should follow the exchanged segment before it");
                exchanged += segment.count;
            }
        }
        Check(frozen_segments > 0, "the frozen layer's values should be in the layout");
        Check(layout.total_count == total && layout.exchanged_count == exchanged, "the layout's counts should match its segments");
        // Linear(3->2): 6 + 2; BatchNorm(2): 4 x 2; frozen Linear(2->1): 2 + 1.
        Check(layout.exchanged_count == 16 && layout.total_count == 19, "the counts should match the model");
        Check(layout.hash.size() == 64, "the hash should be a SHA-256 in hex");
    }

    // The hash names the layout: the same build gives the same hash, a
    // different role or shape another one.
    {
        cyxwiz::SequentialExecutableModel same(BuildModel(true));
        cyxwiz::SequentialExecutableModel unfrozen(BuildModel(false));
        cyxwiz::SequentialExecutableModel wider(BuildModel(true, 4));
        Check(Describe(same).hash == layout.hash, "two builds of one model should have one hash");
        Check(Describe(unfrozen).hash != layout.hash, "a layer frozen on one side only should change the hash");
        Check(Describe(wider).hash != layout.hash, "another shape should change the hash");
        Check(Describe(unfrozen).exchanged_count == 19, "with nothing frozen every value is exchanged");
    }

    // Read, write and read again: the full state, in row-major order.
    {
        const std::vector<float> before = Read(model, layout, true);
        Check(before.size() == layout.total_count, "the full state should have the layout's length");
        for (const cyxwiz::ModelStateEntry& entry : model.GetState()) {
            const StateSegment& segment = Segment(layout, entry.name);
            const float* host = entry.value.ReadData<float>();
            Check(std::equal(host, host + segment.count, before.begin() + static_cast<std::ptrdiff_t>(segment.offset)),
                  entry.name + " should be read in the tensor's own order");
        }

        std::vector<float> wanted(layout.total_count);
        for (size_t i = 0; i < wanted.size(); ++i) wanted[i] = 0.25f * static_cast<float>(i) - 1.0f;
        std::string error;
        Check(cyxwiz::WriteModelState(model, layout, wanted, true, error), "the full state should be written: " + error);
        Check(Read(model, layout, true) == wanted, "the model should hold what was written");
        const StateSegment& weight = Segment(layout, "layer0.weight");
        Check(weight.shape.size() == 2, "the first layer's weight should be a matrix");
        const cyxwiz::Tensor stored = model.GetParameters().at("layer0.weight");
        Check(stored.Shape() == weight.shape && stored.At(1, 0) == wanted[weight.offset + weight.shape[1]],
              "a written matrix should keep its shape and row-major order");
    }

    // The exchanged state leaves the frozen layer alone.
    {
        const std::vector<float> full_before = Read(model, layout, true);
        std::vector<float> exchanged = Read(model, layout, false);
        Check(exchanged.size() == layout.exchanged_count, "the exchanged state should leave the frozen values out");
        for (float& value : exchanged) value += 3.0f;
        std::string error;
        Check(cyxwiz::WriteModelState(model, layout, exchanged, false, error), "the exchanged state should be written: " + error);
        const std::vector<float> full_after = Read(model, layout, true);
        for (const StateSegment& segment : layout.segments) {
            const bool frozen = segment.role == ModelStateRole::Frozen;
            for (size_t i = 0; i < segment.count; ++i) {
                const float expected = full_before[segment.offset + i] + (frozen ? 0.0f : 3.0f);
                Check(full_after[segment.offset + i] == expected, segment.name + " should change only when it is exchanged");
            }
        }
    }

    // What must be refused, with the model left as it was.
    {
        const std::vector<float> before = Read(model, layout, true);
        std::string error;

        std::vector<float> short_state(layout.total_count - 1, 0.0f);
        Check(!cyxwiz::WriteModelState(model, layout, short_state, true, error) && !error.empty(),
              "a state of the wrong length should be refused");

        std::vector<float> not_finite = before;
        const StateSegment& last = layout.segments.back();
        not_finite[last.offset] = std::numeric_limits<float>::quiet_NaN();
        Check(!cyxwiz::WriteModelState(model, layout, not_finite, true, error), "a state with a NaN should be refused");
        Check(error.find(last.name) != std::string::npos, "the refusal should name the segment: " + error);
        not_finite[last.offset] = std::numeric_limits<float>::infinity();
        Check(!cyxwiz::WriteModelState(model, layout, not_finite, true, error), "a state with an infinity should be refused");

        cyxwiz::SequentialExecutableModel wider(BuildModel(true, 4));
        std::vector<float> values;
        Check(!cyxwiz::ReadModelState(wider, layout, true, values, error) && error.find("layer0") != std::string::npos,
              "reading a model with another layout should be refused and say where: " + error);
        Check(!cyxwiz::WriteModelState(wider, layout, before, true, error), "writing a model with another layout should be refused");
        cyxwiz::SequentialExecutableModel unfrozen(BuildModel(false));
        Check(!cyxwiz::ReadModelState(unfrozen, layout, true, values, error) && error.find("frozen") != std::string::npos,
              "a layer frozen on one side only should be refused: " + error);

        Check(Read(model, layout, true) == before, "a refused write should leave the model as it was");
    }

    // A statistic moves in training without a gradient, and the state shows it.
    {
        cyxwiz::SequentialExecutableModel fresh(BuildModel(false));
        const ModelStateLayout fresh_layout = Describe(fresh);
        const StateSegment& mean = Segment(fresh_layout, "layer2.running_mean");
        // Known positive weights, so the layer before BatchNorm cannot output zeros.
        const std::vector<float> before(fresh_layout.exchanged_count, 0.5f);
        std::string error;
        Check(cyxwiz::WriteModelState(fresh, fresh_layout, before, false, error), "the state should be written: " + error);
        fresh.SetTraining(true);
        const std::vector<float> batch = {1.0f, 2.0f, 3.0f, 2.0f, 5.0f, 1.0f, 4.0f, 0.5f, 2.5f, 3.0f, 1.0f, 6.0f};
        fresh.Forward(cyxwiz::Tensor({4, 3}, batch.data(), cyxwiz::DataType::Float32));
        const std::vector<float> after = Read(fresh, fresh_layout, false);
        bool mean_moved = false;
        for (size_t i = 0; i < mean.count; ++i) {
            mean_moved = mean_moved || after[mean.exchanged_offset + i] != before[mean.exchanged_offset + i];
        }
        Check(mean_moved, "a training forward pass should move BatchNorm's running mean");
        Check(fresh.GetGradients().count("layer2.running_mean") == 0, "a statistic should have no gradient");
    }

    std::cout << "model state layout: all checks passed\n";
    return 0;
}
