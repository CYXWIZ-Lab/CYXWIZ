#pragma once

// MjViewportPanel — ImGui panel displaying the MuJoCo 3D viewport.
//
// One control row (Run / Pause, Step, Reset, Stop, speed, the Camera and
// Record menus, status, the View menu) that drives the simulation executor,
// the rendered scene with camera input and its overlays (time and speed; the
// live episode numbers during policy playback), and the episode strip below
// it. Drawn with the Engine's widgets (PluginContext::GetUi, plugin API 1.4).
// The menus are in mj_viewport_menus.cpp.

#include "mj_renderer.h"
#include "mj_env_manager.h"
#include "mj_recorder.h"
#include "mj_viewport_settings.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

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
    void SetGraphSimActive(bool active) { graph_sim_active_ = active; }
    bool IsGraphSimActive() const { return graph_sim_active_; }

    // The Engine plays a trained policy in a copy shown here (ShowCopy):
    // the scene follows it and the local controls are off. label: what the
    // episode strip says ("Inverted Pendulum · copy 0").
    void SetPlayback(bool active, std::string label = {}) {
        if (active && !playback_) camera_default_pending_ = true;  // playback starts with its camera
        playback_ = active;
        playback_label_ = std::move(label);
    }
    // The played copy's episode, read from the training batch each frame
    struct PlaybackNumbers {
        int episode = 1;
        int step = 0;
        int max_steps = 0;
        double step_seconds = 0.0;  // one environment step of the played task
        double episode_return = 0.0;
        bool has_last_return = false;
        double last_return = 0.0;
        std::vector<float> action;
    };
    void SetPlaybackNumbers(PlaybackNumbers numbers) { numbers_ = std::move(numbers); }
    // The renderer (re)started for a model: apply the saved scene options and
    // the camera to start with
    void OnRendererReady() {
        scene_pending_ = true;
        camera_default_pending_ = true;
    }
    bool IsPlayback() const { return playback_; }
    const std::string& ModelTitle() const { return model_title_; }
    const std::string& ModelSubtitle() const { return model_subtitle_; }

    // The run controls, shared by the control row and the Simulation menu
    static constexpr const char* kGraphReason =
        "The graph run is stepping this model. Stop the graph run to control it here.";
    static constexpr const char* kPlaybackReason =
        "Policy playback is showing this scene. Stop the playback in the Engine to control it here.";
    struct RunState {
        bool graph = false;      // a graph run steps the model
        bool playback = false;   // policy playback shows a training copy
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
    void RenderScene(MjEnvManager& env, MjRenderer& renderer);
    void RenderEpisodeStrip(MjEnvManager& env);
    void RenderLiveNumbers(float scene_left, float scene_top);
    // The menus and the Record dialog (mj_viewport_menus.cpp)
    void RenderCameraMenu(MjEnvManager& env, MjRenderer& renderer);
    void RenderViewMenu(MjEnvManager& env, MjRenderer& renderer);
    void RenderRecordMenu(MjEnvManager& env, MjRenderer& renderer);
    void RenderRecordDialog(MjEnvManager& env);
    std::string CameraLabel(const mjModel* model, const MjRenderer& renderer) const;
    // The camera a model starts with: playback's saved choice, else the free camera
    void ApplyStartCamera(const mjModel* model, const mjData* data, MjRenderer& renderer);
    // Screenshots and recordings go to <project>/exports; empty without a project
    std::filesystem::path ExportsFolder() const;
    std::string ExportName() const;  // "half_cheetah_20261112-064512"
    void TakeScreenshot(MjRenderer& renderer);
    void StartRecording(MjEnvManager& env);
    // Feeds the recorder after a frame was drawn; stops it at the episode's end
    void RecordFrame(MjEnvManager& env, MjRenderer& renderer);
    int CurrentEpisode(MjEnvManager& env) const;
    // A short line on the scene for a few seconds ("Saved exports/...")
    void Notify(std::string text, bool failed = false);
    // Achieved speed from simulation time while the model is being stepped
    void SampleSpeed(bool stepping, double sim_time);

    PluginContext* ctx_ = nullptr;
    OpenModelFn open_model_;
    MjSimulationExecutor* sim_executor_ = nullptr;
    std::string model_title_;
    std::string model_subtitle_;
    std::string pending_open_path_;  // Opened before the next frame's ImGui calls
    std::atomic<bool> graph_sim_active_{false};  // from BeginGraphRun to EndGraphRun
    bool playback_ = false;
    std::string playback_label_;
    PlaybackNumbers numbers_;

    // Saved choices (Preferences > Simulation), read when a model or a
    // playback starts
    ViewportDefaults defaults_;
    bool defaults_loaded_ = false;
    bool live_numbers_ = true;
    bool scene_pending_ = false;           // apply defaults_.scene once the renderer runs
    bool camera_default_pending_ = false;  // apply the start camera once the renderer runs

    // Record
    MjRecorder recorder_;
    bool open_record_dialog_ = false;
    int record_format_choice_ = 0;  // 0 GIF, 1 PNG frames
    int record_length_choice_ = 0;  // 0 until the episode ends, 1 until Stop
    bool record_until_stop_ = false;
    int record_episode_ = 0;
    std::string notice_;
    bool notice_failed_ = false;
    std::chrono::steady_clock::time_point notice_until_{};

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

// The Viewport's section in Preferences > Simulation (IPreferencesProvider):
// the camera for playback, the live numbers, the Record defaults, the scene options.
void RenderViewportPreferences(PluginContext& ctx);

} // namespace cyxwiz::plugin::mujoco
