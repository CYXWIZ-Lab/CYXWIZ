#pragma once

// MjViewportPanel — ImGui panel displaying the MuJoCo 3D viewport.
//
// One control row (Run / Pause, Step, Reset, Stop, speed, status, View menu)
// that drives the simulation executor, the rendered scene with camera input
// and its time/speed overlays, and the episode strip below it. Drawn with the
// Engine's widgets (PluginContext::GetUi, plugin API 1.4).

#include "mj_renderer.h"
#include "mj_env_manager.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <string>

namespace cyxwiz::plugin {
class PluginContext;
}

namespace cyxwiz::plugin::mujoco {

// Forward declaration
class MjSimulationExecutor;

class MjViewportPanel {
public:
    // Opens a model file the user picked (the plugin's LoadEnvironment).
    using OpenModelFn = std::function<bool(const std::string& mjcf_path)>;

    MjViewportPanel() = default;
    ~MjViewportPanel() = default;

    // The Engine's widgets, and the Environment Library panel to open.
    void SetContext(PluginContext* ctx) { ctx_ = ctx; }
    void SetOpenModel(OpenModelFn fn) { open_model_ = std::move(fn); }

    // Render the panel. Call each frame from IPanelProvider::RenderPanel.
    // |env| and |renderer| must be valid. |visible| controls panel open state.
    void Render(MjEnvManager& env, MjRenderer& renderer, bool* visible);

    // Set simulation executor for Run/Pause/Step controls
    void SetSimExecutor(MjSimulationExecutor* exec) { sim_executor_ = exec; }

    // The model shown in the control row: its name and "Built-in · Locomotion".
    void SetModelLabel(std::string title, std::string subtitle) {
        model_title_ = std::move(title);
        model_subtitle_ = std::move(subtitle);
    }

    // Graph simulation driving physics (disables local sim controls)
    void SetGraphSimActive(bool active) { graph_sim_active_ = active; graph_sim_frames_ = 0; }
    bool IsGraphSimActive() const { return graph_sim_active_; }

    // The run controls, shared by the control row and the Simulation menu
    static constexpr const char* kGraphReason =
        "The graph run is stepping this model. Stop the graph run to control it here.";
    struct RunState {
        bool graph = false;      // a graph run steps the model
        bool active = false;     // a viewport run exists (running or paused)
        bool running = false;
        bool paused = false;
        bool can_drive = false;  // a model is loaded and no graph run holds it
    };
    RunState GetRunState() const;
    void RunOrResume();  // starts a run, or resumes a paused one
    void PauseRun();
    void StepOnce();     // pauses a running model, then advances one step
    void StopRun();
    // A model file to open before the next frame's ImGui calls (the panel
    // must be shown for it to load)
    void RequestOpenModel(const std::string& path) { pending_open_path_ = path; }

private:
    void RenderEmpty();
    void RenderControls(MjEnvManager& env, MjRenderer& renderer);
    void RenderViewMenu(MjEnvManager& env, MjRenderer& renderer);
    void RenderScene(MjEnvManager& env, MjRenderer& renderer);
    void RenderEpisodeStrip(MjEnvManager& env);
    // Achieved speed from simulation time while the model is being stepped
    void SampleSpeed(bool stepping, double sim_time);

    PluginContext* ctx_ = nullptr;
    OpenModelFn open_model_;
    MjSimulationExecutor* sim_executor_ = nullptr;
    std::string model_title_;
    std::string model_subtitle_;
    std::string pending_open_path_;  // Opened before the next frame's ImGui calls
    std::atomic<bool> graph_sim_active_{false};  // set from the graph simulation thread
    std::atomic<int> graph_sim_frames_{0};       // frames since last SetGraphSimActive(true)

    // Speed: real-time factor the user asks for, or as fast as possible
    double speed_ = 1.0;
    bool unlimited_ = false;

    // Achieved speed, sampled twice a second from simulation and wall time
    std::chrono::steady_clock::time_point speed_sample_wall_{};
    double speed_sample_sim_ = 0.0;
    double achieved_speed_ = 0.0;

    // Resize debouncing — only apply after user stops resizing
    int pending_w_ = 0;
    int pending_h_ = 0;
    int stable_frames_ = 0;  // Frames since last size change
    static constexpr int kResizeDebounceFrames = 10;
};

} // namespace cyxwiz::plugin::mujoco
