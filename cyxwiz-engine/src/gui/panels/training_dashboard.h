#pragma once

// RL Training Dashboard (approved board 3, TOFIX134 P1 step 1.7; redrawn on
// the Training Dashboard's parts in TOFIX129 B4-8, mockup RL-1..RL-6):
// episode and policy figures a run reports, drawn with the shared PlotView
// in the theme colours. Two sources: a script (pycyxwiz.rl_update_metric, one
// value per report) and the Engine's own run (TOFIX195 S4.3, R9; SAC since
// TOFIX216 S4, TD3 since U4, DQN since V4): one row per PPO update or SAC /
// TD3 / DQN report by environment step, progress, speed, Pause / Stop, the
// learner's own cards (SAC: critic and actor loss, entropy weight, replay
// buffer) and the Stable-Baselines3 reference of the same settings. The words
// come from core/rl_dashboard_presentation. UI thread only.

#include "../panel.h"
#include "../plot/plot_view.h"
#include "../../core/rl/off_policy_trainer.h"
#include "../../core/rl/ppo_trainer.h"
#include "../../core/rl/rl_reference.h"
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

    // The Engine's run (PPO, SAC, TD3 or DQN)
    struct RunInfo {
        std::string learner = "PPO";  // "PPO", "SAC", "TD3", "DQN"
        std::string environment;      // "CartPole-v1"
        int copies = 1;
        long long total_timesteps = 0;
        float target_return = 0.0f;  // NaN: no target
        bool continuous = false;
        std::optional<rl::RLReference> reference;
        // A SAC, TD3 or DQN run: its own cards and progress
        bool off_policy = false;
        long long learning_starts = 0;  // random actions before
        int report_every = 0;           // steps between two reports (the watch's unit)
        double updates_per_step = 1.0;  // gradient steps per environment step
        // SAC's entropy weight
        bool ent_coef_auto = true;
        float ent_coef_start = 1.0f;
        float target_entropy = 0.0f;    // resolved (auto: minus the actions)
        // TD3's exploration and policy delay (td3: the run is TD3's)
        bool td3 = false;
        bool action_noise = true;
        float action_noise_std = 0.1f;
        float target_policy_noise = 0.2f;
        float target_noise_clip = 0.5f;
        int policy_delay = 1;
        // DQN's epsilon schedule (dqn: the run is DQN's)
        bool dqn = false;
        float exploration_initial_eps = 1.0f;
        float exploration_final_eps = 0.05f;
        float exploration_fraction = 0.1f;
    };
    struct RunControls {
        std::function<void()> pause;
        std::function<void()> resume;
        std::function<void()> stop;
        std::function<bool()> paused;
        std::function<bool()> waiting;  // for another training run's GPU (SetRunWaiting says why)
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
    void BeginRun(RunInfo info, RunControls controls);
    // The device the run trains on, once its preflight is done, and the note
    // under the header (core/rl_device_presentation DashboardNote)
    void SetRunDevice(const rl_device::RunDevice& device, std::string note);
    // Why the run waits for the GPU (another training run holds it)
    void SetRunWaiting(std::string why) { waiting_ = std::move(why); }
    // Where a SAC, TD3 or DQN run's replay buffer lives: "the GPU", "the computer's memory"
    void SetRunMemoryPlace(std::string place) { memory_place_ = std::move(place); }
    // How the run computes (fused kernels or ArrayFire operations), under Run details
    void SetRunComputePath(std::string path) { compute_path_ = std::move(path); }
    void SetRunNote(std::string note) { device_note_ = std::move(note); }
    void AddPpoUpdate(const rl::PpoUpdateMetrics& m);
    void AddOffPolicyReport(const rl::OffPolicyUpdateMetrics& m);
    // Watch progress set from outside (a policy playback took the view: 0, off)
    void SetWatch(int watch) { watch_ = watch; }
    // message: the run's own words (the failure's reason)
    void EndRun(rl_dashboard::End end, const std::string& message);

private:
    void RegisterCustomPlot(const std::string& name, const std::string& display_name);
    struct Metric {
        std::string display_name;
        std::vector<double> history;  // one value per report
        std::vector<double> steps;    // the environment step of each value (an Engine run), else empty
        bool dirty = true;            // the plot needs new data
        std::unique_ptr<plot::PlotView> view;
    };

    rl_dashboard::RunFacts Facts() const;
    bool Running() const { return run_.has_value() && end_ == rl_dashboard::End::None; }
    void RenderHeader(const rl_dashboard::RunFacts& facts, const rl_dashboard::Header& header);
    void RenderCards(const std::vector<rl_dashboard::Card>& cards);
    void RenderWatch();  // Watch progress (an Engine run)
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

    // The Engine's run, when one reports here
    std::optional<RunInfo> run_;
    RunControls controls_;
    rl_device::RunDevice device_;
    std::string waiting_;
    std::string device_note_;
    std::string compute_path_;
    std::string memory_place_;
    // A PPO run's updates, or a SAC, TD3 or DQN run's reports: one row each
    std::vector<rl_dashboard::Update> updates_;
    std::optional<rl::PpoUpdateMetrics> last_update_;         // its environment cards (PPO)
    std::optional<rl::OffPolicyUpdateMetrics> last_report_;  // its environment cards (SAC, TD3, DQN)
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
