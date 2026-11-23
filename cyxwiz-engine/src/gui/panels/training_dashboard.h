#pragma once

// RL Training Dashboard (approved board 3, TOFIX134 P1 step 1.7; redrawn on
// the Training Dashboard's parts in TOFIX129 B4-8, mockup RL-1..RL-6):
// episode and policy figures a run reports, drawn with the shared PlotView
// in the theme colours. Two sources: a script (pycyxwiz.rl_update_metric, one
// value per report) and the Engine's own PPO run (TOFIX195 S4.3, R9: one row
// per update by environment step, progress, speed, Pause / Stop, and the
// Stable-Baselines3 reference of the same settings). The words come from
// core/rl_dashboard_presentation. UI thread only.

#include "../panel.h"
#include "../plot/plot_view.h"
#include "../../core/rl/ppo_reference.h"
#include "../../core/rl/ppo_trainer.h"
#include "../../core/rl_dashboard_presentation.h"

#include <imgui.h>

#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

class TrainingDashboardPanel : public Panel {
public:
    TrainingDashboardPanel();
    ~TrainingDashboardPanel() override = default;

    void Render() override;

    void UpdateCustomMetric(const std::string& name, float value);
    // Clears the figures, the charts and the run (More > Clear all...)
    void ResetRLMetrics();

    // The Engine's PPO run
    struct PpoRunInfo {
        std::string environment;  // "CartPole-v1"
        int copies = 1;
        long long total_timesteps = 0;
        float target_return = 0.0f;  // NaN: no target
        bool continuous = false;
        std::optional<rl::PpoReference> reference;
    };
    struct PpoRunControls {
        std::function<void()> pause;
        std::function<void()> resume;
        std::function<void()> stop;
        std::function<bool()> paused;
        std::function<bool()> waiting;  // for another training run's GPU (SetPpoWaiting says why)
        // After the run: its trained policy (TOFIX195 S5)
        std::function<bool()> has_policy;
        std::function<void()> play;
        std::function<void()> save;
        // Watching the run while it trains (TOFIX212 P5): the choice and the
        // interval, applied every frame (0 off, 1 live preview, 2 raw training
        // copy); the next update's networks now; what the preview plays; the
        // plugin's view
        struct WatchView {
            std::string unavailable;  // why there is nothing to watch in (no view); empty: there is
            std::string view_label;   // "Show the MuJoCo Viewport"
            std::string preview_of;   // "update 1,580 (810,240 steps)"
            std::string episodes;     // "3 played · last return 1912.4 (update 1,560)"
            std::string message;      // why the preview stopped
        };
        std::function<void(int watch, int every_updates)> watch;
        std::function<void()> watch_now;
        std::function<WatchView()> watch_view;
        std::function<void()> show_view;
    };
    void BeginPpoRun(PpoRunInfo info, PpoRunControls controls);
    // The device the run trains on, once its preflight is done, and the note
    // under the header (core/rl_device_presentation DashboardNote)
    void SetPpoDevice(const rl_device::RunDevice& device, std::string note);
    // Why the run waits for the GPU (another training run holds it)
    void SetPpoWaiting(std::string why) { waiting_ = std::move(why); }
    // How the run computes (fused kernels or ArrayFire operations), under Run details
    void SetPpoComputePath(std::string path) { compute_path_ = std::move(path); }
    void SetPpoNote(std::string note) { device_note_ = std::move(note); }
    void AddPpoUpdate(const rl::PpoUpdateMetrics& m);
    // Watch progress set from outside (a policy playback took the view: 0, off)
    void SetWatch(int watch) { watch_ = watch; }
    // message: the run's own words (the failure's reason)
    void EndPpoRun(rl_dashboard::End end, const std::string& message);

private:
    void RegisterCustomPlot(const std::string& name, const std::string& display_name);
    struct Metric {
        std::string display_name;
        std::vector<double> history;  // one value per report
        std::vector<double> steps;    // the environment step of each value (a PPO run), else empty
        bool dirty = true;            // the plot needs new data
        std::unique_ptr<plot::PlotView> view;
    };

    rl_dashboard::RunFacts Facts() const;
    bool Running() const { return run_.has_value() && end_ == rl_dashboard::End::None; }
    void RenderHeader(const rl_dashboard::RunFacts& facts, const rl_dashboard::Header& header);
    void RenderCards(const std::vector<rl_dashboard::Card>& cards);
    void RenderWatch();  // Watch progress (a PPO run)
    void RenderCharts();
    void RenderMetric(const std::string& name, float height);
    // A chart's options, the same on the dashboard and in its own window:
    // its series colour and, for the return of a run with a target, the target line
    plot::PlotView::Options MetricOptions(const std::string& name) const;
    void RunPendingExport();
    void Push(const std::string& name, double value, double steps);

    // Long runs keep every value; the plot draws a reduced copy.
    static constexpr size_t kMaxHistory = 200000;

    std::map<std::string, Metric> metrics_;
    std::vector<std::string> order_;  // registration order (series colours)
    std::chrono::steady_clock::time_point last_script_value_{};

    // The Engine's PPO run, when one reports here
    std::optional<PpoRunInfo> run_;
    PpoRunControls controls_;
    rl_device::RunDevice device_;
    std::string waiting_;
    std::string device_note_;
    std::string compute_path_;
    std::vector<rl_dashboard::Update> updates_;
    std::optional<rl::PpoUpdateMetrics> last_update_;  // its environment cards
    rl_dashboard::End end_ = rl_dashboard::End::None;
    std::string end_message_;
    // Watch progress: from Preferences > Simulation at each run's start
    int watch_ = 0;
    int watch_every_ = 20;

    int chart_tab_ = 0;  // Episodes, Policy diagnostics
    bool run_details_open_ = false;
    bool open_clear_all_ = false;
    bool pending_export_ = false;
    std::string export_note_;
    bool export_failed_ = false;
};

}  // namespace cyxwiz
