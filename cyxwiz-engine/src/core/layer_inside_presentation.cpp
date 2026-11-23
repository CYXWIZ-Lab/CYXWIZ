#include "layer_inside_presentation.h"

#include "compiled_node_presentation.h"
#include "spatial_layer_shapes.h"

#include <stdexcept>

namespace cyxwiz {

namespace {

const char* ActivationLabel(const std::string& value) {
    if (value == "hardswish") return "Hardswish";
    if (value == "swish") return "Swish";
    if (value == "hardsigmoid") return "Hardsigmoid";
    if (value == "sigmoid") return "Sigmoid";
    return "ReLU";
}

LayerInside SqueezeExcitationInside(const std::map<std::string, std::string>& parameters,
                                    const std::vector<size_t>& input_sample) {
    LayerInside inside;
    inside.applies = true;
    const bool known = input_sample.size() == 3 && input_sample[2] > 0;
    const size_t channels = known ? input_sample[2] : 1;
    spatial::SqueezeExcitationSettings settings;
    try {
        settings = spatial::ResolveSqueezeExcitation(channels, parameters);
    } catch (const std::invalid_argument& error) {
        inside.steps = {{"Mean over the map", "", ""}, {"Dense", "", ""}, {"Activation", "", ""},
                        {"Dense", "", ""}, {"Gate", "", ""}, {"Multiply the map", "", ""}};
        inside.note = std::string("These settings do not compile: ") + error.what() + ".";
        return inside;
    }
    const char* activation = ActivationLabel(settings.inner_activation);
    const char* gate = ActivationLabel(settings.gate);
    if (!known) {
        inside.steps = {{"Mean over the map", "", ""},
                        {"Dense, channels to hidden width", "", ""},
                        {activation, "", ""},
                        {"Dense, hidden width to channels", "", ""},
                        {gate, "", ""},
                        {"Multiply the map", "", ""}};
        inside.note = "Sizes appear when the graph compiles.";
        return inside;
    }
    const size_t squeeze = settings.squeeze;
    const long long reduce = static_cast<long long>(channels * squeeze + (settings.use_bias ? squeeze : 0));
    const long long expand = static_cast<long long>(squeeze * channels + (settings.use_bias ? channels : 0));
    const std::string c = std::to_string(channels), s = std::to_string(squeeze);
    const std::string wide = FormatCompiledShape({channels}), narrow = FormatCompiledShape({squeeze});
    inside.steps = {{"Mean over the map", wide, ""},
                    {"Dense " + c + " to " + s, narrow, FormatCompiledCount(reduce)},
                    {activation, narrow, ""},
                    {"Dense " + s + " to " + c, wide, FormatCompiledCount(expand)},
                    {gate, wide, ""},
                    {"Multiply the map", FormatCompiledShape(input_sample), ""}};
    inside.total = FormatCompiledCount(reduce + expand);
    if (settings.exact_width) {
        inside.note = "Hidden width " + s + ", typed.";
    } else {
        inside.note = "Hidden width " + s + " = " + c + " channels / reduction " + std::to_string(settings.reduction) +
                      (channels < static_cast<size_t>(settings.reduction) ? " (at least 1)." : ".");
    }
    inside.note += settings.use_bias ? " Shapes are for one sample." : " No biases. Shapes are for one sample.";
    return inside;
}

}  // namespace

LayerInside BuildLayerInside(gui::NodeType type,
                             const std::map<std::string, std::string>& parameters,
                             const std::vector<size_t>& input_sample) {
    if (type == gui::NodeType::SqueezeExcitation) return SqueezeExcitationInside(parameters, input_sample);
    return {};
}

}  // namespace cyxwiz
