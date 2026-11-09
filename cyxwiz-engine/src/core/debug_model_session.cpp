#include "debug_model_session.h"

#include "model_builder.h"

#include <cyxwiz/sequential.h>

#include <stdexcept>

namespace cyxwiz {

namespace {

std::string ShapeText(const std::vector<size_t>& shape) {
    std::string text = "[";
    for (size_t i = 0; i < shape.size(); ++i) text += (i ? ", " : "") + std::to_string(shape[i]);
    return text + "]";
}

}  // namespace

DebugModelSession::DebugModelSession(std::unique_ptr<IExecutableModel> model) : model_(std::move(model)) {
    model_->SetTraining(false);
}

std::unique_ptr<DebugModelSession> DebugModelSession::FromCapture(const TrainingConfiguration& config,
                                                                  const LayerCapture& capture, std::string* error) {
    const auto fail = [&](std::string why) -> std::unique_ptr<DebugModelSession> {
        if (error) *error = std::move(why);
        return nullptr;
    };
    auto built = BuildExecutableFromConfig(config);
    if (!built.ok())
        return fail(built.error_message.empty() ? "the model could not be built" : built.error_message);

    IExecutableModel& model = *built.model;
    // The graph must be the run's: each captured layer is the same node in
    // the rebuilt model (a saved run is rebuilt from the canvas as it is now).
    for (const auto& provenance : built.module_provenance) {
        if (!provenance.module_index) continue;
        for (const auto& l : capture.Layers()) {
            if (l.module_index != *provenance.module_index || l.node_id < 0 || l.node_id == provenance.node_id)
                continue;
            return fail("the graph is not the one this run used: layer " + std::to_string(l.module_index) +
                        " was " + l.node_name + ", it is " + provenance.node_name + " now");
        }
    }
    for (size_t i = 0; i < model.LayerCount(); ++i) {
        Module* module = model.LayerModule(i);
        if (!module) continue;
        const auto wanted = module->GetParameters();
        if (wanted.empty()) continue;
        const CapturedLayer* layer = nullptr;
        for (const auto& l : capture.Layers())
            if (l.module_index == i) layer = &l;
        const std::string name = layer && !layer->node_name.empty() ? layer->node_name
                                                                    : "layer " + std::to_string(i);
        std::map<std::string, Tensor> weights;
        for (const auto& [key, tensor] : wanted) {
            const CapturedTensor* kept = layer ? layer->Find(CapturedTensorRole::Parameter, key) : nullptr;
            if (!kept && layer) kept = layer->Find(CapturedTensorRole::State, key);
            if (!kept)
                return fail("the run did not keep " + name + "'s " + key +
                            " (Layer capture: keep the weights, within the size limit)");
            if (kept->tensor.Shape() != tensor.Shape())
                return fail(name + "'s " + key + " was kept as " + ShapeText(kept->tensor.Shape()) +
                            ", the model has " + ShapeText(tensor.Shape()));
            weights[key] = kept->tensor;
        }
        module->SetParameters(weights);
    }
    return std::unique_ptr<DebugModelSession>(new DebugModelSession(std::move(built.model)));
}

Tensor DebugModelSession::Forward(const Tensor& input) {
    return model_->Forward(input);
}

Tensor DebugModelSession::InputGradient(const Tensor& input, const Tensor& target_weights) {
    const Tensor output = model_->Forward(input);
    if (output.Shape() != target_weights.Shape())
        throw std::runtime_error("the target " + ShapeText(target_weights.Shape()) +
                                 " does not have the output's shape " + ShapeText(output.Shape()));
    return model_->Backward(target_weights);
}

}  // namespace cyxwiz
