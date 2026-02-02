#pragma once

// GraphExecutor — Evaluates node graph at runtime for live simulation
//
// Two workflows:
//   1. Manual Control: Signal sources (Slider, SineWave) → MuJoCo Plant → Scope
//   2. RL Training: Separate executor (RLTrainingExecutor) — not handled here
//
// Usage:
//   GraphExecutor executor;
//   executor.Build(nodes, links);
//   executor.SetPluginEvalCallback(...);
//   while (simulating) {
//       executor.Tick(dt);  // Evaluate entire graph once
//       auto val = executor.GetPinValue(scope_pin_id);
//   }

#include "../gui/node_editor.h"

#include <vector>
#include <map>
#include <set>
#include <variant>
#include <functional>
#include <mutex>
#include <string>

namespace cyxwiz {

// Runtime value flowing through graph pins
using NodeValue = std::variant<
    float,                      // Scalar signal
    std::vector<float>,         // Vector (sensor outputs, qpos, qvel)
    std::string                 // Handles, identifiers
>;

// Context passed to plugin for node evaluation
struct NodeEvalContext {
    std::string node_type_name;                           // e.g., "MuJoCoPlant"
    std::map<std::string, std::string> parameters;        // All node parameters
    std::map<std::string, NodeValue> input_values;        // pin_name -> value
    float sim_time = 0.0f;                                // Current simulation time
    float dt = 0.0f;                                      // Time step
};

// Result from evaluating a plugin node
struct NodeEvalResult {
    std::map<std::string, NodeValue> output_values;       // pin_name -> value
    bool success = true;
    std::string error_message;
};

// Callback for plugin node evaluation (routed via PluginManager)
using PluginEvalCallback = std::function<NodeEvalResult(
    const std::string& plugin_qualified_name,   // "mujoco:MuJoCoPlant"
    const NodeEvalContext& ctx
)>;

class GraphExecutor {
public:
    GraphExecutor() = default;

    // Build execution plan from current node graph
    bool Build(const std::vector<gui::MLNode>& nodes,
               const std::vector<gui::NodeLink>& links);

    // Set callback for evaluating plugin nodes
    void SetPluginEvalCallback(PluginEvalCallback callback) {
        plugin_eval_callback_ = std::move(callback);
    }

    // Advance one simulation tick
    bool Tick(float dt);

    // Reset all state (sim time, pin values)
    void Reset();

    // Value access (thread-safe)
    NodeValue GetPinValue(int pin_id) const;
    void SetPinValue(int pin_id, NodeValue value);

    // Errors
    bool HasError() const { return !error_.empty(); }
    std::string GetError() const { return error_; }

    // Sim time
    float GetSimTime() const { return sim_time_; }

    // Dirty flag: mark a pin as changed (for partial evaluation)
    void MarkPinDirty(int pin_id);

private:
    bool BuildExecutionOrder();
    bool EvaluateNode(const gui::MLNode& node, float dt);
    bool EvaluateSignalNode(const gui::MLNode& node, float dt);
    bool EvaluatePluginNode(const gui::MLNode& node, float dt);

    // Gather input pin values for a node by following links
    std::map<std::string, NodeValue> GatherInputs(const gui::MLNode& node) const;

    // Find node by ID
    const gui::MLNode* FindNode(int node_id) const;

    // Graph data (copies)
    std::vector<gui::MLNode> nodes_;
    std::vector<gui::NodeLink> links_;

    // Execution order (node IDs in topological order)
    std::vector<int> execution_order_;

    // Pin values
    mutable std::mutex values_mutex_;
    std::map<int, NodeValue> pin_values_;

    // Dirty tracking for partial evaluation
    std::set<int> dirty_nodes_;
    bool full_eval_ = true;  // First tick always evaluates everything

    // Simulation state
    float sim_time_ = 0.0f;

    // Plugin callback
    PluginEvalCallback plugin_eval_callback_;

    // Error state
    std::string error_;
};

} // namespace cyxwiz
