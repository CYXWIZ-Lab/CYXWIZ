#include "../src/core/graph_executor.h"
#include "../src/core/simulation_runtime_capabilities.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

bool Near(float actual, float expected, float tolerance = 1e-5f) {
    return std::abs(actual - expected) <= tolerance;
}

gui::NodePin Pin(int id, const char* name, bool is_input) {
    gui::NodePin pin{};
    pin.id = id;
    pin.type = gui::PinType::Tensor;
    pin.name = name;
    pin.is_input = is_input;
    return pin;
}

gui::MLNode Source(int id,
                   gui::NodeType type,
                   const char* name,
                   int output_pin) {
    gui::MLNode node{};
    node.id = id;
    node.type = type;
    node.category = type == gui::NodeType::Constant
                        ? gui::NodeCategory::Utility
                        : gui::NodeCategory::Signal;
    node.name = name;
    node.outputs.push_back(Pin(output_pin, type == gui::NodeType::Constant ||
                                              type == gui::NodeType::SignalSlider
                                          ? "Value"
                                          : "Signal",
                               false));
    return node;
}

float Scalar(const cyxwiz::GraphExecutor& executor, int pin_id) {
    cyxwiz::NodeValue value;
    Check(executor.TryGetPinValue(pin_id, value),
          "expected pin value " + std::to_string(pin_id));
    const auto* scalar = std::get_if<float>(&value);
    Check(scalar != nullptr,
          "expected scalar pin value " + std::to_string(pin_id));
    return *scalar;
}

void CheckCapabilityCatalog() {
    Check(cyxwiz::kBuiltInSimulationRuntimeCapabilities.size() == 6,
          "built-in simulation capability count should remain explicit");
    for (const auto& capability :
         cyxwiz::kBuiltInSimulationRuntimeCapabilities) {
        Check(cyxwiz::IsBuiltInSimulationRuntimeNode(capability.node_type),
              std::string("capability should resolve: ") +
                  capability.runtime_name);
    }
    Check(!cyxwiz::IsBuiltInSimulationRuntimeNode(gui::NodeType::Dense),
          "training nodes must not enter the simulation runtime catalog");
}

void CheckConstantAndLiveSlider() {
    auto constant = Source(1, gui::NodeType::Constant, "Constant", 101);
    constant.parameters["value"] = "2.5";
    cyxwiz::GraphExecutor constant_executor;
    Check(constant_executor.Build({constant}, {}),
          constant_executor.GetError());
    Check(constant_executor.Tick(0.1f), constant_executor.GetError());
    Check(Near(Scalar(constant_executor, 101), 2.5f),
          "Constant should emit its configured scalar");

    auto slider = Source(2, gui::NodeType::SignalSlider, "Signal Slider", 102);
    slider.parameters = {{"value", "0.25"}, {"min", "-1"}, {"max", "1"}};
    cyxwiz::GraphExecutor slider_executor;
    Check(slider_executor.Build({slider}, {}), slider_executor.GetError());
    Check(slider_executor.Tick(0.1f), slider_executor.GetError());
    Check(Near(Scalar(slider_executor, 102), 0.25f),
          "Slider should read the canonical value key");
    Check(slider_executor.SetNodeParameter(2, "value", "-0.75"),
          "live slider parameter update should resolve the node");
    Check(slider_executor.Tick(0.1f), slider_executor.GetError());
    Check(Near(Scalar(slider_executor, 102), -0.75f),
          "live slider update should reach the next tick");
}

void CheckTimeVaryingSources() {
    auto sine = Source(3, gui::NodeType::SineWave, "Sine Wave", 103);
    sine.parameters = {{"amplitude", "2"}, {"frequency", "1"},
                       {"phase", "0"}, {"offset", "3"}};
    cyxwiz::GraphExecutor sine_executor;
    Check(sine_executor.Build({sine}, {}), sine_executor.GetError());
    Check(sine_executor.Tick(0.25f), sine_executor.GetError());
    Check(Near(Scalar(sine_executor, 103), 3.0f),
          "Sine should include offset at time zero");
    Check(sine_executor.Tick(0.25f), sine_executor.GetError());
    Check(Near(Scalar(sine_executor, 103), 5.0f),
          "Sine should use hertz, seconds, and radians");

    auto step = Source(4, gui::NodeType::StepSignal, "Step Signal", 104);
    step.parameters = {{"step_time", "1"}, {"initial_value", "-2"},
                       {"final_value", "4"}};
    cyxwiz::GraphExecutor step_executor;
    Check(step_executor.Build({step}, {}), step_executor.GetError());
    Check(step_executor.Tick(1.0f), step_executor.GetError());
    Check(Near(Scalar(step_executor, 104), -2.0f),
          "Step should emit initial_value before the transition");
    Check(step_executor.Tick(0.0f), step_executor.GetError());
    Check(Near(Scalar(step_executor, 104), 4.0f),
          "Step should emit final_value at the transition");

    auto legacy_step = Source(5, gui::NodeType::StepSignal,
                              "Legacy Step Signal", 105);
    legacy_step.parameters = {{"step_time", "0"},
                              {"initial_value", "-1"}, {"value", "6"}};
    cyxwiz::GraphExecutor legacy_executor;
    Check(legacy_executor.Build({legacy_step}, {}),
          legacy_executor.GetError());
    Check(legacy_executor.Tick(0.0f), legacy_executor.GetError());
    Check(Near(Scalar(legacy_executor, 105), 6.0f),
          "legacy Step value should migrate at the runtime boundary");

    auto ramp = Source(6, gui::NodeType::RampSignal, "Ramp Signal", 106);
    ramp.parameters = {{"start_value", "2"}, {"end_value", "-2"},
                       {"duration", "2"}};
    cyxwiz::GraphExecutor ramp_executor;
    Check(ramp_executor.Build({ramp}, {}), ramp_executor.GetError());
    Check(ramp_executor.Tick(1.0f), ramp_executor.GetError());
    Check(Near(Scalar(ramp_executor, 106), 2.0f),
          "Ramp should start at start_value");
    Check(ramp_executor.Tick(1.0f), ramp_executor.GetError());
    Check(Near(Scalar(ramp_executor, 106), 0.0f),
          "descending Ramp should interpolate without reversed clamp bounds");
    Check(ramp_executor.Tick(0.0f), ramp_executor.GetError());
    Check(Near(Scalar(ramp_executor, 106), -2.0f),
          "Ramp should hold end_value after duration");
}

void CheckScopeAndPluginRouting() {
    auto source = Source(10, gui::NodeType::Constant, "Constant", 201);
    source.parameters["value"] = "7";
    gui::MLNode scope{};
    scope.id = 11;
    scope.type = gui::NodeType::SignalScope;
    scope.category = gui::NodeCategory::Signal;
    scope.name = "Signal Scope";
    scope.inputs.push_back(Pin(202, "Signal", true));
    gui::NodeLink link{1, 10, 201, 11, 202};

    cyxwiz::GraphExecutor executor;
    Check(executor.Build({source, scope}, {link}), executor.GetError());
    Check(executor.Tick(0.1f), executor.GetError());
    Check(Near(Scalar(executor, 202), 7.0f),
          "Scope should retain its connected live scalar input");

    gui::MLNode plugin{};
    plugin.id = 12;
    plugin.type = gui::NodeType::PluginCustom;
    plugin.category = gui::NodeCategory::Plugin;
    plugin.name = "External Plant";
    plugin.plugin_qualified_name = "test:Plant";
    plugin.outputs.push_back(Pin(203, "sensor", false));
    cyxwiz::GraphExecutor missing_provider;
    Check(missing_provider.Build({plugin}, {}), missing_provider.GetError());
    Check(!missing_provider.Tick(0.1f),
          "plugin simulation should fail closed without a provider callback");
    Check(missing_provider.GetError().find("no simulation provider") !=
              std::string::npos,
          "missing plugin provider error should be actionable");
}

void CheckInvalidInputs() {
    auto ramp = Source(20, gui::NodeType::RampSignal, "Bad Ramp", 301);
    ramp.parameters = {{"start_value", "0"}, {"end_value", "1"},
                       {"duration", "0"}};
    cyxwiz::GraphExecutor executor;
    Check(executor.Build({ramp}, {}), executor.GetError());
    Check(!executor.Tick(0.1f), "zero-duration Ramp should fail closed");
    Check(executor.GetError().find("greater than zero") != std::string::npos,
          "invalid duration error should identify the constraint");

    ramp.parameters["duration"] = "not-a-number";
    Check(executor.Build({ramp}, {}), executor.GetError());
    Check(!executor.Tick(0.1f), "non-numeric Ramp should fail closed");
    Check(executor.GetError().find("finite number") != std::string::npos,
          "invalid numeric parameter error should be actionable");

    Check(executor.Build({}, {}), executor.GetError());
    Check(!executor.Tick(std::numeric_limits<float>::quiet_NaN()),
          "non-finite simulation timestep should fail closed");

    auto source = Source(21, gui::NodeType::Constant, "Constant", 302);
    source.parameters["value"] = "1";
    gui::NodeLink missing_target{2, 21, 302, 999, 9991};
    Check(!executor.Build({source}, {missing_target}),
          "links to missing nodes should fail during graph build");
    Check(executor.GetError().find("missing node") != std::string::npos,
          "malformed-link error should identify the missing node");
}

} // namespace

int main() {
    CheckCapabilityCatalog();
    CheckConstantAndLiveSlider();
    CheckTimeVaryingSources();
    CheckScopeAndPluginRouting();
    CheckInvalidInputs();
    std::cout << "GraphExecutor simulation contract tests passed\n";
    return 0;
}
