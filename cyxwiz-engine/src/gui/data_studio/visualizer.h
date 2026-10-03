#pragma once

// Data Studio Visualize tab (TOFIX134 P3.11, approved board 15): a quick
// one-widget dashboard on the dataset picked in the Data Studio header. Every
// plot type, the shared plot view and the dashboard's settings panel; Add to
// Dashboard copies a plot into a Dashboard node on the same data. Each dataset
// keeps its own list of plots for the session. UI thread only.

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

private:
    std::string current_dataset_;
    std::map<std::string, std::unique_ptr<dashboard::DashboardWindow>> plots_;  // per dataset
};

}  // namespace cyxwiz
