// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

// Graph simulation runs with a plugin's nodes (plugin API 1.4). The Engine
// finds this through IPlugin::QueryInterface("ISimulationProvider") when the
// plugin starts (UIModify permission). A plugin node runs in a graph
// simulation only when its plugin names it here: INodeProvider::EvaluateNode
// stays the per-tick call, IPanelProvider stays the drawing.
//
// When the user runs a graph that holds such nodes, the Engine calls
// BeginGraphRun before the first tick and EndGraphRun after the last one has
// finished. Both are called on the UI thread. Between them EvaluateNode is
// called from the run's worker thread.

#include <map>
#include <string>
#include <vector>

namespace cyxwiz::plugin {

struct PluginSimulationNodeInfo {
    std::string node_type;  // one of the plugin's INodeProvider node types ("MuJoCoPlant")
    // A panel of the plugin the Engine shows when a run with this node
    // starts ("mujoco_viewport"); empty for none.
    std::string panel_id;
};

// One node of the plugin in the graph being started, with its settings.
struct PluginSimulationGraphNode {
    int node_id = 0;
    std::string node_type;
    std::map<std::string, std::string> parameters;
};

struct PluginSimulationBeginResult {
    bool ok = true;
    // When not ok: what stops the run and what to do ("The viewport is
    // running the model. Stop it, then run the graph.").
    std::string reason;
};

class ISimulationProvider {
public:
    virtual ~ISimulationProvider() = default;

    virtual std::vector<PluginSimulationNodeInfo> GetSimulationNodes() = 0;

    // Check the nodes (a model loaded, settings that make sense) and take
    // what the run needs, such as the right to step the physics, which
    // nothing else may hold meanwhile. Not ok: the run does not start and
    // EndGraphRun is not called.
    virtual PluginSimulationBeginResult BeginGraphRun(const std::vector<PluginSimulationGraphNode>& nodes) = 0;

    // The run ended (Stop, a failed tick, the graph closed, the plugin
    // stopping) and its worker has finished: release what BeginGraphRun took.
    virtual void EndGraphRun() = 0;
};

} // namespace cyxwiz::plugin
