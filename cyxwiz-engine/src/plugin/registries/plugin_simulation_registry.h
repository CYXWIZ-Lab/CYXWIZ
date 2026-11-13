#pragma once

// Plugin nodes that run in a graph simulation (ISimulationProvider, plugin
// API 1.4), and the start of a run that holds them. Keyed by the node's
// extension type id ("<plugin id>:<node type>").

#include "../interfaces/i_simulation_provider.h"
#include "../../core/graph_model.h"

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace cyxwiz::plugin {

// What a started run holds from its plugins. Ending it (the last copy going)
// calls each plugin's EndGraphRun: drop it on the UI thread, after the run's
// worker has finished.
class SimulationRunLease {
public:
    ~SimulationRunLease();
    SimulationRunLease(const SimulationRunLease&) = delete;
    SimulationRunLease& operator=(const SimulationRunLease&) = delete;

    const std::set<std::string>& PluginIds() const { return plugin_ids_; }
    // The panels to show for this run: (plugin id, panel id), each once.
    const std::vector<std::pair<std::string, std::string>>& Panels() const { return panels_; }

private:
    friend class PluginSimulationRegistry;
    SimulationRunLease() = default;

    struct Begun {
        std::string plugin_id;
        ISimulationProvider* provider = nullptr;  // non-owning; the registry entry outlives the run
    };
    std::vector<Begun> begun_;
    std::set<std::string> plugin_ids_;
    std::vector<std::pair<std::string, std::string>> panels_;
    std::vector<std::shared_ptr<void>> library_leases_;  // keep the libraries mapped meanwhile
};

class PluginSimulationRegistry {
public:
    static PluginSimulationRegistry& Instance();

    PluginSimulationRegistry(const PluginSimulationRegistry&) = delete;
    PluginSimulationRegistry& operator=(const PluginSimulationRegistry&) = delete;

    // The info is copied into the Engine's own strings.
    void Register(const std::string& plugin_id, const PluginSimulationNodeInfo& info, ISimulationProvider* provider);
    void RemoveByPlugin(const std::string& plugin_id);

    bool IsSimulationNode(const std::string& type_id) const;

    // Asks each plugin with nodes in the graph to begin the run. Null with
    // the reason when a plugin node cannot run in a simulation or a plugin
    // says no; plugins that had begun are ended again.
    std::shared_ptr<SimulationRunLease> BeginGraphRun(const std::vector<::gui::MLNode>& nodes, std::string& reason);

    size_t GetCount() const;

private:
    PluginSimulationRegistry() = default;

    struct Registered {
        std::string plugin_id;
        PluginSimulationNodeInfo info;
        ISimulationProvider* provider = nullptr;  // non-owning
    };

    mutable std::mutex mutex_;
    std::map<std::string, Registered> nodes_;
};

} // namespace cyxwiz::plugin
