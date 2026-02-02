#include "graph_executor.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <spdlog/spdlog.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cyxwiz {

bool GraphExecutor::Build(const std::vector<gui::MLNode>& nodes,
                          const std::vector<gui::NodeLink>& links) {
    nodes_ = nodes;
    links_ = links;
    error_.clear();
    sim_time_ = 0.0f;
    pin_values_.clear();
    dirty_nodes_.clear();
    full_eval_ = true;

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
        adj[node.id];  // ensure entry
        in_degree[node.id] = 0;
    }

    for (const auto& link : links_) {
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

    if (full_eval_) {
        // Evaluate all nodes in order
        for (int node_id : execution_order_) {
            const auto* node = FindNode(node_id);
            if (!node) continue;
            if (!EvaluateNode(*node, dt)) return false;
        }
        full_eval_ = false;
    } else if (!dirty_nodes_.empty()) {
        // Partial evaluation: only dirty nodes (in execution order)
        for (int node_id : execution_order_) {
            if (dirty_nodes_.count(node_id)) {
                const auto* node = FindNode(node_id);
                if (!node) continue;
                if (!EvaluateNode(*node, dt)) return false;
            }
        }
        dirty_nodes_.clear();
    } else {
        // No dirty nodes — still evaluate time-varying nodes (SineWave, etc.)
        for (int node_id : execution_order_) {
            const auto* node = FindNode(node_id);
            if (!node) continue;

            bool is_time_varying = (node->type == gui::NodeType::SineWave ||
                                    node->type == gui::NodeType::RampSignal ||
                                    node->type == gui::NodeType::StepSignal ||
                                    node->type == gui::NodeType::PluginCustom);
            if (is_time_varying) {
                if (!EvaluateNode(*node, dt)) return false;
            }
        }
    }

    sim_time_ += dt;
    return true;
}

void GraphExecutor::Reset() {
    std::lock_guard<std::mutex> lock(values_mutex_);
    pin_values_.clear();
    sim_time_ = 0.0f;
    dirty_nodes_.clear();
    full_eval_ = true;
}

NodeValue GraphExecutor::GetPinValue(int pin_id) const {
    std::lock_guard<std::mutex> lock(values_mutex_);
    auto it = pin_values_.find(pin_id);
    if (it != pin_values_.end()) return it->second;
    return 0.0f;
}

void GraphExecutor::SetPinValue(int pin_id, NodeValue value) {
    std::lock_guard<std::mutex> lock(values_mutex_);
    pin_values_[pin_id] = std::move(value);
}

void GraphExecutor::MarkPinDirty(int pin_id) {
    std::set<int> visited_pins;
    MarkPinDirtyImpl(pin_id, visited_pins);
}

void GraphExecutor::MarkPinDirtyImpl(int pin_id, std::set<int>& visited_pins) {
    if (!visited_pins.insert(pin_id).second) return; // cycle detection

    for (const auto& link : links_) {
        if (link.from_pin == pin_id) {
            dirty_nodes_.insert(link.to_node);
            for (const auto& link2 : links_) {
                if (link2.from_node == link.to_node) {
                    MarkPinDirtyImpl(link2.from_pin, visited_pins);
                }
            }
        }
    }
}

bool GraphExecutor::EvaluateNode(const gui::MLNode& node, float dt) {
    if (node.type == gui::NodeType::PluginCustom) {
        return EvaluatePluginNode(node, dt);
    }
    return EvaluateSignalNode(node, dt);
}

bool GraphExecutor::EvaluateSignalNode(const gui::MLNode& node, float dt) {
    auto get_param = [&](const std::string& key, const std::string& fallback = "0") -> std::string {
        auto it = node.parameters.find(key);
        return (it != node.parameters.end() && !it->second.empty()) ? it->second : fallback;
    };

    auto get_float = [&](const std::string& key, float fallback = 0.0f) -> float {
        try { return std::stof(get_param(key, std::to_string(fallback))); }
        catch (...) { return fallback; }
    };

    switch (node.type) {
        case gui::NodeType::Constant: {
            float value = get_float("value", 0.0f);
            if (!node.outputs.empty()) {
                std::lock_guard<std::mutex> lock(values_mutex_);
                pin_values_[node.outputs[0].id] = value;
            }
            return true;
        }

        case gui::NodeType::SignalSlider: {
            float value = get_float("current_value", 0.0f);
            if (!node.outputs.empty()) {
                int pin_id = node.outputs[0].id;
                bool changed = false;
                {
                    std::lock_guard<std::mutex> lock(values_mutex_);
                    auto it = pin_values_.find(pin_id);
                    if (it == pin_values_.end()) {
                        changed = true;
                    } else if (auto* prev = std::get_if<float>(&it->second)) {
                        changed = (std::abs(*prev - value) > 1e-7f);
                    } else {
                        changed = true;
                    }
                    pin_values_[pin_id] = value;
                }
                if (changed) {
                    MarkPinDirty(pin_id);
                }
            }
            return true;
        }

        case gui::NodeType::SineWave: {
            float amplitude = get_float("amplitude", 1.0f);
            float frequency = get_float("frequency", 1.0f);
            float phase = get_float("phase", 0.0f);
            float value = amplitude * std::sin(2.0f * static_cast<float>(M_PI) * frequency * sim_time_ + phase);
            if (!node.outputs.empty()) {
                std::lock_guard<std::mutex> lock(values_mutex_);
                pin_values_[node.outputs[0].id] = value;
            }
            return true;
        }

        case gui::NodeType::StepSignal: {
            float step_time = get_float("step_time", 1.0f);
            float value = get_float("value", 1.0f);
            float output = (sim_time_ >= step_time) ? value : 0.0f;
            if (!node.outputs.empty()) {
                std::lock_guard<std::mutex> lock(values_mutex_);
                pin_values_[node.outputs[0].id] = output;
            }
            return true;
        }

        case gui::NodeType::RampSignal: {
            float start_time = get_float("start_time", 0.0f);
            float end_time = get_float("end_time", 1.0f);
            float start_value = get_float("start_value", 0.0f);
            float end_value = get_float("end_value", 1.0f);
            float t = std::clamp((sim_time_ - start_time) / std::max(end_time - start_time, 0.001f), 0.0f, 1.0f);
            float output = start_value + t * (end_value - start_value);
            if (!node.outputs.empty()) {
                std::lock_guard<std::mutex> lock(values_mutex_);
                pin_values_[node.outputs[0].id] = output;
            }
            return true;
        }

        case gui::NodeType::SignalScope: {
            // Scope is a sink — read input and store (caller reads via GetPinValue on input pin)
            // No output pins to write
            return true;
        }

        default:
            // Non-signal nodes are silently skipped during simulation
            return true;
    }
}

bool GraphExecutor::EvaluatePluginNode(const gui::MLNode& node, float dt) {
    if (!plugin_eval_callback_) {
        // No plugin callback — skip silently
        return true;
    }

    // Build eval context
    NodeEvalContext ctx;
    ctx.parameters = node.parameters;
    ctx.sim_time = sim_time_;
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
