#pragma once

// MjViewportPanel — ImGui panel displaying the MuJoCo 3D viewport.
//
// One control row (Run / Pause, Step, Reset, Stop, speed, the Camera and
// Record menus, status, the View menu) that drives the simulation executor,
// the rendered scene with camera input and its overlays (time and speed; the
// live episode numbers during policy playback), and the episode strip below
// it. Drawn with the Engine's widgets (PluginContext::GetUi, plugin API 1.4).
// The menus are in mj_viewport_menus.cpp; the Stress panel beside the scene
// (kicks, thrown objects, the ground, Ctrl+drag pushes) in mj_viewport_stress.cpp.

#include "env/mj_stress.h"
#include "env/mj_task.h"
#include "mj_renderer.h"
#include "mj_env_manager.h"
#include "mj_recorder.h"
#include "env/mj_inspect.h"
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
    // Reads the open model file again (Inspect's Reload); false with the reason
    using ReloadModelFn = std::function<bool(std::string& error)>;

    MjViewportPanel() = default;
    ~MjViewportPanel() = default;

    // The Engine's widgets, and the Environment Library panel to open.
    void SetContext(PluginContext* ctx) { ctx_ = ctx; }
    void SetOpenModel(OpenModelFn fn) { open_model_ = std::move(fn); }
    void SetReloadModel(ReloadModelFn fn) { reload_model_ = std::move(fn); }

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
        bool last_ended_by_task = false;  // the episode before ended by the task's end (a fall)
        std::vector<float> action;
        // The Engine's words for the copy (LabelShownCopy): the status ("Training
        // preview"; empty: "Policy playback") and rows put first in the live numbers
        std::string status;
        std::vector<std::pair<std::string, std::string>> rows;
        bool watch_only = false;  // a training's own copy: the Stress panel is off
    };
    void SetPlaybackNumbers(PlaybackNumbers numbers) { numbers_ = std::move(numbers); }
    // The played copy's status word: the Engine's, or "Policy playback"
    const char* PlaybackStatus() const { return numbers_.status.empty() ? "Policy playback" : numbers_.status.c_str(); }
    // The renderer (re)started for a model: apply the saved scene options and
    // the camera to start with
    void OnRendererReady() {
        scene_pending_ = true;
        camera_default_pending_ = true;
    }
    bool IsPlayback() const { return playback_; }

    // The Stress panel's requests since the last frame: the plugin sends them
    // to the played copy, or applies them to the Viewport's own run
    std::vector<MjStressRequest> TakeStressRequests() {
        std::vector<MjStressRequest> requests;
        requests.swap(stress_requests_);
        return requests;
    }
    // A request could not be done (shown on the scene for a few seconds)
    void StressFailed(const std::string& why) { Notify("Stress: " + why, true); }
    // Another model was opened (or the file read again): the events, the held
    // push, the selected body and the held controls belong to the old one
    void ResetStress();
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
    void RenderLiveNumbers(float scene_left, float scene_top, const MjGround& ground);
    // The Inspect panel (mj_viewport_inspect.cpp): the selected body, its
    // joints, the actuators, keyframes, Reload
    void RenderInspectPanel(MjEnvManager& env, MjRenderer& renderer);
    // Why Inspect cannot change the model now; empty: it can
    std::string InspectBlocked(MjEnvManager& env) const;
    // The held controls to the run (under the physics lock)
    void ApplyHeldControls(MjEnvManager& env);
    // The Stress panel (mj_viewport_stress.cpp)
    void RenderStressPanel(MjEnvManager& env);
    // Why the Stress panel cannot act now; empty: it can
    std::string StressBlocked(MjEnvManager& env) const;
    // Each frame: the event outcomes, the automatic kicks
    void WatchStress(MjEnvManager& env);
    // Ctrl+drag on the scene: start, move and end a push
    void HandlePush(MjEnvManager& env, MjRenderer& renderer, bool scene_active, float x, float y, float height);
    struct StressClock {
        double time = 0.0;
        int episode = 1;
    };
    StressClock ShownClock(MjEnvManager& env) const;
    // A kick's velocity change: 0 back, 1 forward, 2 up, 3 any way (random)
    std::array<double, 3> KickVelocity(int way);
    // A throw of the panel's object as the panel says; random: varied by the
    // auto-throw ranges, from any side
    void QueueThrow(MjEnvManager& env, bool random);
    // The top-down dial on the scene that sets where throws come from
    void RenderThrowDial(float right, float top);
    void Queue(MjStressRequest request, MjEnvManager& env, std::string what);
    void LogStress(MjEnvManager& env, std::string what);  // an event to watch, at the shown copy's time
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
    ReloadModelFn reload_model_;
    // Inspect
    InspectOptions inspect_;
    int selected_body_ = -1;              // double-clicked; -1: none
    std::vector<double> typed_controls_;  // the actuator controls typed in Inspect
    std::vector<MjKeyframe> keyframes_;   // saved for keyframes_model_ (in the project)
    std::string keyframes_model_;
    char keyframe_name_[64] = "";
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

    // Stress panel (TOFIX212 P4): its choices this session (Preferences hold
    // how it starts), the requests for the plugin, the events, the push held
    StressOptions stress_;
    std::vector<MjStressRequest> stress_requests_;
    MjStressLog stress_log_;
    bool pushing_ = false;
    MjGround asked_ground_;          // the ground last asked for (the next episode's until it starts)
    bool ground_asked_ = false;
    MjGround shown_ground_;          // the ground of the episode shown last frame
    bool shown_ground_known_ = false;
    double next_auto_kick_ = -1.0;   // simulated time of the next automatic kick; -1: not scheduled
    double next_auto_throw_ = -1.0;  // and throw
    double last_watch_time_ = 0.0;
    int stress_episode_ = -1;        // the episode the random stream was seeded for
    MjRandom stress_random_{7};      // random kicks, throws and tumbles: seeded by the ground seed each episode

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
