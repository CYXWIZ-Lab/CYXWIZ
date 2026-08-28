#include "graph_executor.h"
#include "simulation_runtime_capabilities.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <queue>
#include <spdlog/spdlog.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cyxwiz {
namespace {

bool TryParseFiniteFloat(const std::string& text, float& value) {
    errno = 0;
    char* end = nullptr;
    const float parsed = std::strtof(text.c_str(), &end);
    if (end == text.c_str() || errno == ERANGE || !std::isfinite(parsed)) {
        return false;
    }
    while (*end != '\0' && std::isspace(static_cast<unsigned char>(*end))) {
        ++end;
    }
    if (*end != '\0') {
        return false;
    }
    value = parsed;
    return true;
}

} // namespace

bool GraphExecutor::Build(const std::vector<gui::MLNode>& nodes,
                          const std::vector<gui::NodeLink>& links) {
    nodes_ = nodes;
    links_ = links;
    error_.clear();
    sim_time_.store(0.0f);
    {
        std::lock_guard<std::mutex> lock(values_mutex_);
        pin_values_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(parameters_mutex_);
        parameter_overrides_.clear();
    }
    if (!BuildExecutionOrder()) {
        return false;
    }

    spdlog::info("GraphExecutor: Built execution plan with {} nodes", execution_order_.size());
    return true;
}

bool GraphExecutor::BuildExecutionOrder() {
    execution_order_.clear();

    // Build adjacency and in-degree maps
    std::map<int, std::vector<int>> adj;       // node_id -> downstream node_ids
    std::map<int, int> in_degree;

    for (const auto& node : nodes_) {
        if (!in_degree.emplace(node.id, 0).second) {
            error_ = "Graph contains duplicate node id " +
                     std::to_string(node.id);
            return false;
        }
        adj[node.id];
    }

    std::map<int, int> input_connection_count;
    for (const auto& link : links_) {
        const auto* from_node = FindNode(link.from_node);
        const auto* to_node = FindNode(link.to_node);
        if (!from_node || !to_node) {
            error_ = "Graph link " + std::to_string(link.id) +
                     " references a missing node";
            return false;
        }
        const auto from_pin = std::find_if(
            from_node->outputs.begin(), from_node->outputs.end(),
            [&link](const gui::NodePin& pin) {
                return pin.id == link.from_pin;
            });
        const auto to_pin = std::find_if(
            to_node->inputs.begin(), to_node->inputs.end(),
            [&link](const gui::NodePin& pin) {
                return pin.id == link.to_pin;
            });
        if (from_pin == from_node->outputs.end() ||
            to_pin == to_node->inputs.end()) {
            error_ = "Graph link " + std::to_string(link.id) +
                     " references a missing or wrong-direction pin";
            return false;
        }
        const int connection_count = ++input_connection_count[link.to_pin];
        if (to_pin->max_connections != gui::PIN_UNLIMITED &&
            connection_count > to_pin->max_connections) {
            error_ = "Graph input pin " + std::to_string(link.to_pin) +
                     " exceeds its connection capacity";
            return false;
        }
        adj[link.from_node].push_back(link.to_node);
        in_degree[link.to_node]++;
    }

    // Kahn's algorithm
    std::queue<int> q;
    for (const auto& [id, deg] : in_degree) {
        if (deg == 0) q.push(id);
    }

    while (!q.empty()) {
        int id = q.front();
        q.pop();
        execution_order_.push_back(id);

        for (int next : adj[id]) {
            if (--in_degree[next] == 0) {
                q.push(next);
            }
        }
    }

    if (execution_order_.size() != nodes_.size()) {
        error_ = "Graph contains a cycle";
        return false;
    }

    return true;
}

bool GraphExecutor::Tick(float dt) {
    error_.clear();

    if (!std::isfinite(dt) || dt < 0.0f) {
        error_ = "Simulation time step must be finite and non-negative";
        return false;
    }

    for (const int node_id : execution_order_) {
        const auto* node = FindNode(node_id);
        if (!node) {
            error_ = "Simulation execution plan references a missing node";
            return false;
        }
        if (!EvaluateNode(*node, dt)) return false;
    }

    sim_time_.store(sim_time_.load() + dt);
    return true;
}

void GraphExecutor::Reset() {
    std::lock_guard<std::mutex> lock(values_mutex_);
    pin_values_.clear();
    sim_time_ = 0.0f;
}

NodeValue GraphExecutor::GetPinValue(int pin_id) const {
    std::lock_guard<std::mutex> lock(values_mutex_);
    auto it = pin_values_.find(pin_id);
    if (it != pin_values_.end()) return it->second;
    return 0.0f;
}

bool GraphExecutor::TryGetPinValue(int pin_id, NodeValue& value) const {
    std::lock_guard<std::mutex> lock(values_mutex_);
    const auto it = pin_values_.find(pin_id);
    if (it == pin_values_.end()) return false;
    value = it->second;
    return true;
}

void GraphExecutor::SetPinValue(int pin_id, NodeValue value) {
    std::lock_guard<std::mutex> lock(values_mutex_);
    pin_values_[pin_id] = std::move(value);
}

bool GraphExecutor::SetNodeParameter(int node_id,
                                     const std::string& key,
                                     const std::string& value) {
    if (!FindNode(node_id)) return false;
    std::lock_guard<std::mutex> lock(parameters_mutex_);
    parameter_overrides_[{node_id, key}] = value;
    return true;
}

bool GraphExecutor::EvaluateNode(const gui::MLNode& node, float dt) {
    if (node.type == gui::NodeType::PluginCustom) {
        return EvaluatePluginNode(node, dt);
    }
    if (!IsBuiltInSimulationRuntimeNode(node.type)) {
        return true;
    }
    return EvaluateSignalNode(node, dt);
}

bool GraphExecutor::EvaluateSignalNode(const gui::MLNode& node, float dt) {
    (void)dt;

    switch (node.type) {
        case gui::NodeType::Constant: {
            float value = 1.0f;
            return ReadFloatParameter(node, "value", 1.0f, value) &&
                   StoreScalarOutput(node, value);
        }

        case gui::NodeType::SignalSlider: {
            float value = 0.0f;
            float minimum = -1.0f;
            float maximum = 1.0f;
            if (!ReadFloatParameter(node, "value", 0.0f, value) ||
                !ReadFloatParameter(node, "min", -1.0f, minimum) ||
                !ReadFloatParameter(node, "max", 1.0f, maximum)) {
                return false;
            }
            if (minimum > maximum || value < minimum || value > maximum) {
                error_ = node.name + ": slider requires min <= value <= max";
                return false;
            }
            return StoreScalarOutput(node, value);
        }

        case gui::NodeType::SineWave: {
            float amplitude = 1.0f;
            float frequency = 1.0f;
            float phase = 0.0f;
            float offset = 0.0f;
            if (!ReadFloatParameter(node, "amplitude", 1.0f, amplitude) ||
                !ReadFloatParameter(node, "frequency", 1.0f, frequency) ||
                !ReadFloatParameter(node, "phase", 0.0f, phase) ||
                !ReadFloatParameter(node, "offset", 0.0f, offset)) {
                return false;
            }
            const float value = amplitude * std::sin(
                2.0f * static_cast<float>(M_PI) * frequency *
                    sim_time_.load() +
                phase) + offset;
            return StoreScalarOutput(node, value);
        }

        case gui::NodeType::StepSignal: {
            float step_time = 1.0f;
            float initial_value = 0.0f;
            float final_value = 1.0f;
            if (!ReadFloatParameter(node, "step_time", 1.0f, step_time) ||
                !ReadFloatParameter(node, "initial_value", 0.0f,
                                    initial_value) ||
                !ReadFloatParameter(node, "final_value", 1.0f, final_value,
                                    "value")) {
                return false;
            }
            if (step_time < 0.0f) {
                error_ = node.name +
                         ": step_time must be greater than or equal to zero";
                return false;
            }
            return StoreScalarOutput(node, sim_time_.load() >= step_time
                                               ? final_value
                                               : initial_value);
        }

        case gui::NodeType::RampSignal: {
            float start_value = 0.0f;
            float end_value = 1.0f;
            float duration = 5.0f;
            if (!ReadFloatParameter(node, "start_value", 0.0f,
                                    start_value) ||
                !ReadFloatParameter(node, "end_value", 1.0f, end_value) ||
                !ReadFloatParameter(node, "duration", 5.0f, duration)) {
                return false;
            }
            if (duration <= 0.0f) {
                error_ = node.name + ": duration must be greater than zero";
                return false;
            }
            const float progress =
                std::clamp(sim_time_.load() / duration, 0.0f, 1.0f);
            return StoreScalarOutput(
                node, start_value + progress * (end_value - start_value));
        }

        case gui::NodeType::SignalScope: {
            if (node.inputs.size() != 1) {
                error_ = node.name + ": scope requires exactly one scalar input";
                return false;
            }
            const auto inputs = GatherInputs(node);
            const auto input = inputs.find(node.inputs[0].name);
            if (input == inputs.end()) {
                error_ = node.name + ": scope input is not connected";
                return false;
            }
            const auto* scalar = std::get_if<float>(&input->second);
            if (!scalar) {
                error_ = node.name + ": scope input must be a scalar signal";
                return false;
            }
            std::lock_guard<std::mutex> lock(values_mutex_);
            pin_values_[node.inputs[0].id] = *scalar;
            return true;
        }

        default:
            error_ = node.name + ": no built-in simulation evaluator";
            return false;
    }
}

std::string GraphExecutor::GetParameterValue(
    const gui::MLNode& node,
    const char* key,
    const char* legacy_key,
    const std::string& fallback) const {
    std::lock_guard<std::mutex> lock(parameters_mutex_);
    const auto find_value = [&](const char* candidate) -> const std::string* {
        if (!candidate) return nullptr;
        const auto override_it =
            parameter_overrides_.find({node.id, candidate});
        if (override_it != parameter_overrides_.end() &&
            !override_it->second.empty()) {
            return &override_it->second;
        }
        const auto parameter_it = node.parameters.find(candidate);
        if (parameter_it != node.parameters.end() &&
            !parameter_it->second.empty()) {
            return &parameter_it->second;
        }
        return nullptr;
    };
    if (const auto* value = find_value(key)) return *value;
    if (const auto* value = find_value(legacy_key)) return *value;
    return fallback;
}

bool GraphExecutor::ReadFloatParameter(const gui::MLNode& node,
                                       const char* key,
                                       float fallback,
                                       float& value,
                                       const char* legacy_key) {
    const std::string text = GetParameterValue(
        node, key, legacy_key, std::to_string(fallback));
    if (TryParseFiniteFloat(text, value)) return true;
    error_ = node.name + ": parameter '" + key +
             "' must be a finite number (received '" + text + "')";
    return false;
}

bool GraphExecutor::StoreScalarOutput(const gui::MLNode& node, float value) {
    if (node.outputs.size() != 1) {
        error_ = node.name + ": scalar source requires exactly one output";
        return false;
    }
    std::lock_guard<std::mutex> lock(values_mutex_);
    pin_values_[node.outputs[0].id] = value;
    return true;
}

bool GraphExecutor::EvaluatePluginNode(const gui::MLNode& node, float dt) {
    if (!plugin_eval_callback_) {
        error_ = "Plugin node '" + node.name +
                 "': no simulation provider callback is registered";
        return false;
    }

    // Build eval context
    NodeEvalContext ctx;
    ctx.parameters = node.parameters;
    ctx.sim_time = sim_time_.load();
    ctx.dt = dt;

    // Extract type name from plugin_qualified_name ("mujoco:MuJoCoPlant" -> "MuJoCoPlant")
    auto colon = node.plugin_qualified_name.find(':');
    if (colon != std::string::npos) {
        ctx.node_type_name = node.plugin_qualified_name.substr(colon + 1);
    } else {
        ctx.node_type_name = node.plugin_qualified_name;
    }

    // Gather inputs by pin name
    ctx.input_values = GatherInputs(node);

    // Call plugin
    auto result = plugin_eval_callback_(node.plugin_qualified_name, ctx);

    if (!result.success) {
        error_ = "Plugin node '" + node.name + "': " + result.error_message;
        spdlog::warn("GraphExecutor: {}", error_);
        return false;
    }

    // Store output values by matching pin name to pin ID
    {
        std::lock_guard<std::mutex> lock(values_mutex_);
        for (const auto& [pin_name, value] : result.output_values) {
            for (const auto& pin : node.outputs) {
                if (pin.name == pin_name) {
                    pin_values_[pin.id] = value;
                    break;
                }
            }
        }
    }

    return true;
}

std::map<std::string, NodeValue> GraphExecutor::GatherInputs(const gui::MLNode& node) const {
    std::map<std::string, NodeValue> inputs;

    for (const auto& input_pin : node.inputs) {
        // Find link targeting this input pin
        for (const auto& link : links_) {
            if (link.to_pin == input_pin.id) {
                std::lock_guard<std::mutex> lock(values_mutex_);
                auto it = pin_values_.find(link.from_pin);
                if (it != pin_values_.end()) {
                    inputs[input_pin.name] = it->second;
                }
                break;
            }
        }
    }

    return inputs;
}

const gui::MLNode* GraphExecutor::FindNode(int node_id) const {
    for (const auto& n : nodes_) {
        if (n.id == node_id) return &n;
    }
    return nullptr;
}

} // namespace cyxwiz
