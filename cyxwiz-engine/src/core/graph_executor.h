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
#include <atomic>
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
    bool TryGetPinValue(int pin_id, NodeValue& value) const;
    void SetPinValue(int pin_id, NodeValue value);

    // Publish a live Properties edit to the executor-owned graph snapshot.
    bool SetNodeParameter(int node_id,
                          const std::string& key,
                          const std::string& value);

    // Errors
    bool HasError() const { return !error_.empty(); }
    std::string GetError() const { return error_; }

    // Sim time
    float GetSimTime() const { return sim_time_.load(); }

private:
    bool BuildExecutionOrder();
    bool EvaluateNode(const gui::MLNode& node, float dt);
    bool EvaluateSignalNode(const gui::MLNode& node, float dt);
    bool EvaluatePluginNode(const gui::MLNode& node, float dt);
    bool ReadFloatParameter(const gui::MLNode& node,
                            const char* key,
                            float fallback,
                            float& value,
                            const char* legacy_key = nullptr);
    std::string GetParameterValue(const gui::MLNode& node,
                                  const char* key,
                                  const char* legacy_key,
                                  const std::string& fallback) const;
    bool StoreScalarOutput(const gui::MLNode& node, float value);

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

    // Live parameter overrides are separate from the immutable build snapshot.
    mutable std::mutex parameters_mutex_;
    std::map<std::pair<int, std::string>, std::string> parameter_overrides_;

    // Simulation state
    std::atomic<float> sim_time_{0.0f};

    // Plugin callback
    PluginEvalCallback plugin_eval_callback_;

    // Error state
    std::string error_;
};

} // namespace cyxwiz
