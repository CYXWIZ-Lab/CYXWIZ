#pragma once

// RL Training Dashboard (approved board 3, TOFIX134 P1 step 1.7): episode
// and policy metrics a run reports, drawn with the shared PlotView in the
// theme colours. Two sources: a script (pycyxwiz.rl_update_metric, one value
// per report) and the Engine's own PPO run (TOFIX195 S4.3, R9: one row per
// update by environment step, progress, speed, Pause / Stop, and the
// Stable-Baselines3 reference of the same settings). UI thread only, except
// SetRLTrainingState (a script's completion callback may call it from the
// script worker).

#include "../panel.h"
#include "../plot/plot_view.h"
#include "../../core/rl/ppo_reference.h"

#include <imgui.h>

#include <atomic>
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

    void RegisterCustomPlot(const std::string& name, const std::string& display_name);
    void UpdateCustomMetric(const std::string& name, float value);

    void SetTrainingState(bool is_training);
    void SetRLTrainingState(bool is_rl_training);
    void ResetRLMetrics();

    // The Engine's PPO run
    struct PpoRunInfo {
        std::string title;  // "PPO on CartPole-v1 x 8"
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
        std::function<bool()> waiting;  // for another training run's GPU (the device chip says why)
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
    // The device the run trains on, once its preflight is done
    // (core/rl_device_presentation): the chip, its route on hover, a note
    void SetPpoDevice(std::string chip, std::string tooltip, bool fallback, std::string note);
    // How the run computes (fused kernels or ArrayFire operations), shown with the chip's route
    void SetPpoComputePath(std::string path) { device_compute_path_ = std::move(path); }
    void SetPpoNote(std::string note) { device_note_ = std::move(note); }
    void AddPpoUpdate(const rl::PpoUpdateMetrics& m);
    // Watch progress set from outside (a policy playback took the view: 0, off)
    void SetWatch(int watch) { watch_ = watch; }
    void EndPpoRun(const std::string& outcome, bool failed);

private:
    struct Metric {
        std::string display_name;
        std::vector<double> history;  // one value per report
        std::vector<double> steps;    // the environment step of each value (a PPO run), else empty
        bool dirty = true;            // the plot needs new data
        std::unique_ptr<plot::PlotView> view;
    };

    void RenderRunHeader();
    void RenderKpis();
    void RenderWatch();  // Watch progress (a PPO run)
    void RenderMetric(const std::string& name, float height);
    void Push(const std::string& name, double value, double steps);

    // Long runs keep every value; the plot draws a reduced copy.
    static constexpr size_t kMaxHistory = 200000;

    std::atomic<bool> training_{false};
    std::map<std::string, Metric> metrics_;
    std::vector<std::string> order_;  // registration order (series colours)

    // The Engine's PPO run, when one reports here
    std::optional<PpoRunInfo> run_;
    PpoRunControls controls_;
    std::string device_chip_;
    std::string device_tooltip_;
    bool device_fallback_ = false;
    std::string device_note_;
    std::string device_compute_path_;
    std::optional<rl::PpoUpdateMetrics> last_update_;
    // Watch progress: from Preferences > Simulation at each run's start
    int watch_ = 0;
    int watch_every_ = 20;
    std::string outcome_;
    bool failed_ = false;
};

}  // namespace cyxwiz
