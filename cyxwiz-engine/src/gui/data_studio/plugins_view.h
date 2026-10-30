#pragma once

// Data Studio > Explore > Plugins (TOFIX181 step 4, approved mockup
// tofix181/step4_plugins_tab_mockup.html): the dataset analytics installed
// plugins add (IAnalyticsProvider). Pick an analytic and the columns it
// reads, Run, and the plugin's answer shows as bars, a note and a table.
// The chosen columns stream to the plugin in the background, from memory or
// from disk (core/dataset_column_stream); core/explore_plugins_presentation
// decides the wording.

#include "explore_tab.h"
#include "../../core/explore_plugins_presentation.h"
#include "../plot/column_picker.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

class PluginsView {
public:
    void Render(const ExploreContext& context);
    bool IsRunning() const { return job_.Running(); }

    // Shows the Plugin Manager (no plugin with analytics, or its plugin is gone).
    std::function<void()> open_plugin_manager;

    struct Answer {
        explore_plugins::Result result;
        std::string error;
        bool ok() const { return error.empty(); }
    };

private:
    void RefreshAnalytics();
    void RenderPicker(const std::vector<explore_plugins::Column>& columns);
    void RenderRun(const ExploreContext& context);
    // `changed`: the choices differ from the answer's; dimmed too when its plugin is gone.
    void RenderAnswer(const ExploreContext& context, const Answer& a, bool changed);
    void Run(const ExploreContext& context);
    void Export(const Answer& a);
    bool Loaded(const std::string& analytic_id) const;

    uint64_t generation_ = ~0ull;  // of the analytics registry, when the list was read
    std::vector<explore_plugins::Analytic> analytics_;
    explore_plugins::Analytic chosen_;  // kept when its plugin goes, so the answer can say so
    std::string dataset_;                // the columns below were picked for it
    std::vector<std::string> picked_;
    bool picked_by_user_ = false;        // else the default columns follow the analytic
    std::string asked_key_;              // the question the job answers
    std::shared_ptr<std::atomic<int64_t>> rows_read_ = std::make_shared<std::atomic<int64_t>>(0);
    int64_t rows_total_ = 0;
    std::string message_;
    bool message_error_ = false;
    plot::ColumnPicker picker_;
    ExploreJob<Answer> job_;
};

}  // namespace cyxwiz
