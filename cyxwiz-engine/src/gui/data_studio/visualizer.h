#pragma once

// Data Studio Visualize tab (TOFIX134 P3.11, approved board 15): a quick
// one-widget dashboard on the dataset picked in the Data Studio header. Every
// plot type, the shared plot view and the dashboard's settings panel; Add to
// Dashboard copies a plot into a Dashboard node on the same data. Each dataset
// keeps its own list of plots for the session; Keep as Plot node (TOFIX148,
// approved board) adds the selected plot to the graph as a Plot node. UI
// thread only.

#include <functional>
#include <map>
#include <memory>
#include <string>

namespace cyxwiz {

namespace dashboard {
class DashboardWindow;
}

class Visualizer {
public:
    Visualizer();
    ~Visualizer();

    void Render();
    void SetActiveDataset(const std::string& dataset_name);

    // Keep as Plot node: why the dataset's plot cannot become a node ("" when
    // it can), and adding it (returns what was done). Set by Data Studio.
    std::function<std::string(const std::string& dataset)> keep_blocker;
    std::function<std::string(const std::string& dataset, const std::string& title, const std::string& spec_json)> on_keep;

private:
    std::string current_dataset_;
    std::map<std::string, std::unique_ptr<dashboard::DashboardWindow>> plots_;  // per dataset
};

}  // namespace cyxwiz
