#include "mujoco_plugin.h"
#include "mj_mjcf_parser.h"
#include "rl/reward_shaper.h"
#include "rl/observation_filter.h"
#include "env/mj_env_batch.h"

#include <spdlog/spdlog.h>
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string_view>

namespace cyxwiz::plugin::mujoco {

namespace {

// Two spellings of one model file (relative/absolute, slashes) are the same model.
bool SameModelFile(const std::string& a, const std::string& b) {
    if (a == b) return true;
    if (a.empty() || b.empty()) return false;
    std::error_code ec_a, ec_b;
    const auto ca = std::filesystem::weakly_canonical(std::filesystem::path(a), ec_a);
    const auto cb = std::filesystem::weakly_canonical(std::filesystem::path(b), ec_b);
    return !ec_a && !ec_b && ca == cb;
}

// The actuator a Plant input pin drives: by name, or "actuator_<i>" for an
// unnamed actuator (the name ParseMjcfFile gives it).
int ActuatorId(const mjModel* m, const std::string& pin_name) {
    const int id = mj_name2id(m, mjOBJ_ACTUATOR, pin_name.c_str());
    if (id >= 0) return id;
    constexpr std::string_view kPrefix = "actuator_";
    if (pin_name.rfind(kPrefix, 0) != 0) return -1;
    const std::string digits = pin_name.substr(kPrefix.size());
    if (digits.empty() || digits.size() > 9 ||
        !std::all_of(digits.begin(), digits.end(), [](unsigned char ch) { return std::isdigit(ch) != 0; }))
        return -1;
    const int index = std::stoi(digits);
    if (index < m->nu && !mj_id2name(m, mjOBJ_ACTUATOR, index)) return index;
    return -1;
}

// MuJoCo state (mjtNum, double) as the float values node pins carry.
std::vector<float> ToFloats(const mjtNum* values, mjtSize count) {
    std::vector<float> out(static_cast<size_t>(count));
    for (mjtSize i = 0; i < count; ++i) out[static_cast<size_t>(i)] = static_cast<float>(values[i]);
    return out;
}

}  // namespace

// =============================================================================
// IPlugin Lifecycle
// =============================================================================

bool MuJoCoPlugin::OnLoad(PluginContext& ctx) {
    ctx.LogInfo("MuJoCo Simulation: OnLoad");

    ctx.LogInfo(std::string("MuJoCo Simulation: Using MuJoCo version ") + mj_versionString());

    // Capture main GLFW window for renderer context management.
    main_window_ = glfwGetCurrentContext();

    // The built-in environments ship in the plugin's own folder (next to plugin.json).
    assets_dir_ = (ctx.GetPluginDir() / "assets").string();
    env_library_.SetAssetsDir(assets_dir_);

    // Scan for previously downloaded Menagerie models and earlier imports
    env_library_.ScanDownloadedModels();
    env_library_.ScanImportedModels();

    // Wire browser panel load callback and downloader; the viewport shows the
    // loaded model's name and where it comes from.
    browser_panel_.SetLoadCallback([this](const std::string& path, const EnvInfo& info) {
        if (!LoadEnvironment(path)) return false;
        if (info.source == EnvSource::Imported) {
            viewport_panel_.SetModelLabel(info.name, "Imported from URL");
        } else {
            const char* source = info.source == EnvSource::Menagerie ? "Menagerie" : "Built-in";
            viewport_panel_.SetModelLabel(info.name, std::string(source) + " \xC2\xB7 " + info.category);
        }
        return true;
    });
    browser_panel_.SetDownloader(&menagerie_downloader_);
    browser_panel_.SetContext(&ctx);
    // One model at a time: while a graph run steps it, the library loads no other
    // (loading stops a viewport run, which the user started here)
    browser_panel_.SetLoadBlocker([this]() -> std::string {
        return viewport_panel_.IsGraphSimActive()
                   ? "A graph run is stepping the loaded model. Stop the graph run to load another."
                   : std::string();
    });

    ctx_ = &ctx;
    // The viewport draws the Engine's widgets and opens the library; a model
    // file it opens is named after the file.
    viewport_panel_.SetContext(&ctx);
    viewport_panel_.SetOpenModel([this](const std::string& path) {
        if (!LoadEnvironment(path)) return false;
        viewport_panel_.SetModelLabel(std::filesystem::path(path).stem().string(), "Your file");
        browser_panel_.ClearLoaded();
        return true;
    });

    state_ = PluginState::Loaded;
    return true;
}

bool MuJoCoPlugin::OnInitialize(PluginContext& ctx) {
    ctx.LogInfo("MuJoCo Simulation: OnInitialize");

    // Note: Registration of interfaces (IPanelProvider, INodeProvider)
    // is handled engine-side via QueryInterface in PluginManager to avoid DLL singleton
    // duplication issues. Do NOT call ctx.Register*() here.

    state_ = PluginState::Active;
    return true;
}

void MuJoCoPlugin::OnShutdown(PluginContext& ctx) {
    ctx.LogInfo("MuJoCo Simulation: OnShutdown");
    sim_executor_.Stop();  // Must stop before freeing model/renderer
    // A running download reports to the library panel: finish it while the panel exists
    menagerie_downloader_.CancelDownload();
    renderer_.Shutdown();
    {
        std::lock_guard phys_lock(env_manager_.GetPhysicsMutex());
        env_manager_.Close();
        loaded_mjcf_path_.clear();
    }
    state_ = PluginState::Loaded;
}

void MuJoCoPlugin::OnUnload(PluginContext& ctx) {
    ctx.LogInfo("MuJoCo Simulation: OnUnload");
    ctx_ = nullptr;
    state_ = PluginState::Unloaded;
}

// =============================================================================
// Environment Management
// =============================================================================

bool MuJoCoPlugin::LoadEnvironment(const std::string& mjcf_path) {
    // Shutdown existing renderer if switching environments
    if (renderer_.IsInitialized()) {
        renderer_.Shutdown();
    }

    // Nothing may step the old model while it is replaced: the executor stops
    // first, and the graph simulation's EvaluateNode waits on the physics lock.
    sim_executor_.Stop();
    {
        std::lock_guard phys_lock(env_manager_.GetPhysicsMutex());
        if (!env_manager_.LoadModel(mjcf_path)) {
            loaded_mjcf_path_.clear();
            return false;
        }
        // The loaded path, for ResolveDynamicPins and the Plant's model check
        loaded_mjcf_path_ = mjcf_path;
        env_manager_.Reset();
    }

    // Wire simulation executor and viewport
    sim_executor_.SetEnvManager(&env_manager_);
    viewport_panel_.SetSimExecutor(&sim_executor_);

    // NOTE: Renderer is NOT initialized here. It will be lazily initialized
    // by the viewport panel on first render, using the engine's GL context
    // directly (avoids cross-DLL glad/GL function pointer issues).
    renderer_needs_init_ = true;

    spdlog::info("[MuJoCo] Environment loaded — obs_dim={}, act_dim={}, max_steps={}",
                 env_manager_.GetObservationDim(),
                 env_manager_.GetActionDim(),
                 env_manager_.GetMaxEpisodeSteps());
    return true;
}

// =============================================================================
// IPanelProvider
// =============================================================================

std::vector<PluginPanelInfo> MuJoCoPlugin::GetPanels() {
    PluginPanelInfo viewport;
    viewport.panel_id = "mujoco_viewport";
    viewport.title = "MuJoCo Viewport";
    viewport.category = "Simulation";
    viewport.show_by_default = false;

    PluginPanelInfo browser;
    browser.panel_id = "mujoco_env_browser";
    browser.title = "Environment Library";
    browser.category = "Simulation";
    browser.show_by_default = false;

    return { viewport, browser };
}

void MuJoCoPlugin::SyncPlayback() {
    std::string path, title;
    std::vector<mjtNum> state;
    int copy = 0;
    bool active = false;
    MjViewportPanel::PlaybackNumbers numbers;
    {
        std::lock_guard<std::mutex> lock(view_slot_->mutex);
        active = view_slot_->owner != nullptr;
        if (active) {
            path = view_slot_->model_path;
            title = view_slot_->title;
            copy = view_slot_->copy;
            numbers.episode = view_slot_->episode;
            numbers.step = view_slot_->step;
            numbers.max_steps = view_slot_->max_steps;
            numbers.step_seconds = view_slot_->step_seconds;
            numbers.episode_return = view_slot_->episode_return;
            numbers.has_last_return = view_slot_->has_last_return;
            numbers.last_return = view_slot_->last_return;
            numbers.action = view_slot_->action;
            if (view_slot_->version != shown_version_) {
                shown_version_ = view_slot_->version;
                state = view_slot_->state;
            }
        }
    }
    if (active) {
        if (viewport_panel_.IsGraphSimActive()) return;  // the graph run keeps its model
        if (!playback_) {
            playback_ = true;
            restore_path_ = loaded_mjcf_path_;
            restore_title_ = viewport_panel_.ModelTitle();
            restore_subtitle_ = viewport_panel_.ModelSubtitle();
        }
        if (loaded_mjcf_path_ != path) {
            if (!LoadEnvironment(path)) {
                spdlog::warn("[MuJoCo] Policy playback: the model {} did not load", path);
                return;
            }
            state.clear();
            shown_version_ = 0;  // apply the latest state next frame, after the renderer starts
        }
        // Also when the played model was open already (a second playback)
        if (viewport_panel_.ModelSubtitle() != "Policy playback") viewport_panel_.SetModelLabel(title, "Policy playback");
        viewport_panel_.SetPlayback(true, title + " \xC2\xB7 copy " + std::to_string(copy));
        viewport_panel_.SetPlaybackNumbers(std::move(numbers));
        if (!state.empty()) {
            std::lock_guard phys_lock(env_manager_.GetPhysicsMutex());
            const mjModel* m = env_manager_.GetModel();
            mjData* d = env_manager_.GetMutableData();
            if (m && d && static_cast<size_t>(mj_stateSize(m, mjSTATE_FULLPHYSICS)) == state.size()) {
                mj_setState(m, d, state.data(), mjSTATE_FULLPHYSICS);
                mj_forward(m, d);
            }
        }
    } else if (playback_) {
        playback_ = false;
        viewport_panel_.SetPlayback(false);
        if (!restore_path_.empty() && restore_path_ != loaded_mjcf_path_ && LoadEnvironment(restore_path_))
            viewport_panel_.SetModelLabel(restore_title_, restore_subtitle_);
        else if (restore_path_.empty())  // nothing before it: the played model stays
            viewport_panel_.SetModelLabel(viewport_panel_.ModelTitle(), "From the last policy playback");
    }
}

void MuJoCoPlugin::RenderPanel(const std::string& panel_id, bool* visible) {
    if (panel_id == "mujoco_viewport") SyncPlayback();

    // Lazy renderer init — deferred from LoadEnvironment to avoid
    // cross-DLL GL context/glad issues. Init happens here in the
    // engine's render loop where the GL context is guaranteed current.
    if (renderer_needs_init_ && env_manager_.IsLoaded() && main_window_) {
        renderer_needs_init_ = false;
        renderer_.SetPhysicsMutex(&env_manager_.GetPhysicsMutex());
        if (!renderer_.Initialize(env_manager_.GetModel(), 640, 480, main_window_)) {
            spdlog::warn("[MuJoCo] Renderer initialization failed — viewport disabled");
        } else {
            renderer_.ResetCamera(env_manager_.GetModel());
            viewport_panel_.OnRendererReady();
        }
    }

    if (panel_id == "mujoco_viewport") {
        viewport_panel_.Render(env_manager_, renderer_, visible);
    } else if (panel_id == "mujoco_env_browser") {
        browser_panel_.Render(env_library_, env_manager_, visible);
    }
}

// =============================================================================
// IPreferencesProvider: the Viewport's section in Preferences > Simulation
// =============================================================================

std::vector<PluginPreferencesSection> MuJoCoPlugin::GetPreferenceSections() {
    return {{"viewport", "Simulation", "MuJoCo Viewport",
             "How the Viewport starts. Changing these in the Viewport does not change them here."}};
}

void MuJoCoPlugin::RenderPreferences(const std::string& section_id) {
    if (section_id == "viewport" && ctx_) RenderViewportPreferences(*ctx_);
}

// =============================================================================
// ICommandProvider: the Simulation menu
// =============================================================================

namespace {
constexpr const char* kViewportPanelId = "mujoco_viewport";
}  // namespace

std::vector<PluginCommandInfo> MuJoCoPlugin::GetCommands() {
    return {
        // The viewport's own run gives way to a graph run in the menu
        {"run", "Simulation", "run", "Run Simulation", "Runs the loaded model in the viewport, or resumes it.", false, true},
        {"pause", "Simulation", "run", "Pause", "Pauses the running model.", false, true},
        {"stop", "Simulation", "run", "Stop", "Stops the run.", false, true},
        {"step", "Simulation", "run", "Step Once", "Pauses the model and advances it one step.", false, true},
        {"load", "Simulation", "model", "Load MJCF Model...", "Opens a MuJoCo model file in the viewport.", false, false},
    };
}

PluginCommandState MuJoCoPlugin::GetCommandState(const std::string& command_id) {
    const auto run = viewport_panel_.GetRunState();
    auto disabled = [](const char* reason) { return PluginCommandState{false, reason, false}; };
    if (command_id == "load") {
        if (run.playback) return disabled(MjViewportPanel::kPlaybackReason);
        return run.graph ? disabled("A graph run is stepping the loaded model. Stop the graph run to load another.")
                         : PluginCommandState{};
    }
    if (run.graph) return disabled(MjViewportPanel::kGraphReason);
    if (run.playback) return disabled(MjViewportPanel::kPlaybackReason);
    if (!env_manager_.IsLoaded()) return disabled("Load a model first: Environment Library or Load MJCF Model...");
    if (command_id == "run") return run.running ? disabled("The model is running.") : PluginCommandState{};
    if (command_id == "pause") return run.running ? PluginCommandState{} : disabled("Nothing is running.");
    if (command_id == "stop") return run.active ? PluginCommandState{} : disabled("Nothing is running.");
    return PluginCommandState{};  // step
}

void MuJoCoPlugin::RunCommand(const std::string& command_id) {
    if (command_id == "run") viewport_panel_.RunOrResume();
    else if (command_id == "pause") viewport_panel_.PauseRun();
    else if (command_id == "stop") viewport_panel_.StopRun();
    else if (command_id == "step") viewport_panel_.StepOnce();
    else if (command_id == "load" && ctx_ && ctx_->GetUi()) {
        const std::string path = ctx_->GetUi()->OpenFileDialog("Open MJCF model", "MJCF model", "xml");
        if (path.empty()) return;
        // The viewport opens it before its next frame: show it
        viewport_panel_.RequestOpenModel(path);
    }
    // The run and the model are shown in the viewport
    if (ctx_) ctx_->ShowPanel(kViewportPanelId);
}

// =============================================================================
// INodeProvider - RL Simulation Nodes
// =============================================================================

std::vector<PluginNodeTypeInfo> MuJoCoPlugin::GetNodeTypes() {
    std::vector<PluginNodeTypeInfo> nodes;

    // --- Reward Function node ---
    // Configurable reward shaping for RL training
    {
        PluginNodeTypeInfo n;
        n.type_name    = "RewardFunction";
        n.display_name = "Reward Function";
        n.category     = "RL / Simulation";
        n.description  = "Define reward shaping for RL training (alive bonus, control penalty, etc.)";
        n.color        = 0xFF44AADD;  // Orange-gold

        n.pins.push_back({"qpos", "Tensor", true});
        n.pins.push_back({"qvel", "Tensor", true});
        n.pins.push_back({"ctrl", "Tensor", true});
        n.pins.push_back({"sensor", "Tensor", true});
        n.pins.push_back({"reward", "Float", false});

        n.default_parameters["alive_bonus"]    = "1.0";
        n.default_parameters["ctrl_cost_weight"] = "0.1";
        n.default_parameters["velocity_reward"] = "true";
        n.default_parameters["height_penalty_threshold"] = "0.0";
        n.default_parameters["height_penalty_value"] = "10.0";
        nodes.push_back(std::move(n));
    }

    // --- Observation Filter node ---
    // Select/normalize observations from the environment
    {
        PluginNodeTypeInfo n;
        n.type_name    = "ObservationFilter";
        n.display_name = "Observation Filter";
        n.category     = "RL / Simulation";
        n.description  = "Filter and normalize environment observations (qpos, qvel, sensors)";
        n.color        = 0xFFAAAA44;  // Blue-ish

        n.pins.push_back({"qpos", "Tensor", true});
        n.pins.push_back({"qvel", "Tensor", true});
        n.pins.push_back({"sensor", "Tensor", true});
        n.pins.push_back({"obs", "Tensor", false});

        n.default_parameters["include_qpos"] = "true";
        n.default_parameters["include_qvel"] = "true";
        n.default_parameters["include_sensors"] = "false";
        n.default_parameters["normalize"]    = "true";
        nodes.push_back(std::move(n));
    }

    // --- MuJoCo Plant node (Simulink-style) ---
    // Loads an MJCF model; pins are dynamically resolved per-actuator/sensor
    {
        PluginNodeTypeInfo n;
        n.type_name    = "MuJoCoPlant";
        n.display_name = "MuJoCo Plant";
        n.category     = "Simulation / Control";
        n.description  = "Simulink-style plant block: per-actuator inputs, sensor/image outputs. "
                         "Set mjcf_path to auto-discover actuators and sensors.";
        n.color        = 0xFF22BB66;  // Green

        // Default pins (before MJCF is loaded)
        n.pins.push_back({"u", "Tensor", true});        // control vector input
        n.pins.push_back({"sensor", "Tensor", false});   // sensor output
        n.pins.push_back({"qpos", "Tensor", false});     // joint positions
        n.pins.push_back({"qvel", "Tensor", false});     // joint velocities
        n.pins.push_back({"rgb", "Image", false});       // camera image
        n.pins.push_back({"depth", "Image", false});     // depth image

        n.default_parameters["mjcf_path"]    = "";
        n.default_parameters["timestep"]     = "0.002";
        n.default_parameters["frame_skip"]   = "1";
        n.default_parameters["interface"]    = "bus";   // "bus" = per-actuator pins, "vector" = single u
        n.default_parameters["camera"]       = "0";

        n.supports_dynamic_pins = true;
        n.dynamic_pin_trigger   = "mjcf_path";

        nodes.push_back(std::move(n));
    }

    return nodes;
}

std::string MuJoCoPlugin::GenerateCode(
    const std::string& node_type_name,
    const std::map<std::string, std::string>& params,
    const std::string& framework)
{
    auto get = [&](const std::string& key, const std::string& fallback = "") -> std::string {
        auto it = params.find(key);
        return (it != params.end()) ? it->second : fallback;
    };

    if (framework != "pytorch") {
        return "# MuJoCo simulation nodes currently support PyTorch only\n";
    }

    if (node_type_name == "RewardFunction") {
        std::string alive = get("alive_bonus", "1.0");
        std::string ctrl_w = get("ctrl_cost_weight", "0.1");
        std::string vel = get("velocity_reward", "true");
        std::string h_thresh = get("height_penalty_threshold", "0.0");
        std::string h_val = get("height_penalty_value", "10.0");
        std::string code =
            "import numpy as np\n\n"
            "def compute_reward(obs, action, next_obs, info):\n"
            "    reward = " + alive + "  # alive bonus\n"
            "    reward -= " + ctrl_w + " * np.sum(action ** 2)  # control cost\n";
        if (vel == "true")
            code += "    reward += next_obs[0]  # forward velocity reward\n";
        if (h_thresh != "0.0" && h_thresh != "0")
            code += "    if len(next_obs) > 2 and next_obs[2] < " + h_thresh + ":\n"
                    "        reward -= " + h_val + "  # height penalty\n";
        code += "    return reward\n";
        return code;
    }

    if (node_type_name == "ObservationFilter") {
        std::string norm = get("normalize", "true");
        std::string qpos = get("include_qpos", "true");
        std::string qvel = get("include_qvel", "true");
        std::string sens = get("include_sensors", "false");
        return
            "import numpy as np\n"
            "\n"
            "def observation_filter(obs):\n"
            "    parts = []\n" +
            std::string(qpos == "true" ? "    parts.append(obs['qpos'])  # joint positions\n" : "") +
            std::string(qvel == "true" ? "    parts.append(obs['qvel'])  # joint velocities\n" : "") +
            std::string(sens == "true" ? "    parts.append(obs['sensors'])  # sensor readings\n" : "") +
            "    filtered = np.concatenate(parts)\n" +
            std::string(norm == "true"
                ? "    # Running mean/std normalization\n"
                  "    filtered = (filtered - obs_mean) / (obs_std + 1e-8)\n"
                : "") +
            "    return filtered\n";
    }

    if (node_type_name == "MuJoCoPlant") {
        std::string mjcf = get("mjcf_path", "model.xml");
        std::string timestep = get("timestep", "0.002");
        std::string frame_skip = get("frame_skip", "1");
        std::string iface = get("interface", "bus");
        return
            "import mujoco\n"
            "import numpy as np\n"
            "\n"
            "model = mujoco.MjModel.from_xml_path('" + mjcf + "')\n"
            "data = mujoco.MjData(model)\n"
            "\n"
            "# Simulation step\n"
            "def plant_step(ctrl):\n"
            "    data.ctrl[:] = ctrl\n"
            "    for _ in range(" + frame_skip + "):\n"
            "        mujoco.mj_step(model, data)\n"
            "    sensor = data.sensordata.copy()\n"
            "    return sensor\n";
    }

    return "# Unknown MuJoCo node type: " + node_type_name + "\n";
}

std::vector<PluginSimulationNodeInfo> MuJoCoPlugin::GetSimulationNodes() {
    return {{"MuJoCoPlant", kViewportPanelId}, {"RewardFunction", ""}, {"ObservationFilter", ""}};
}

PluginSimulationBeginResult MuJoCoPlugin::BeginGraphRun(const std::vector<PluginSimulationGraphNode>& nodes) {
    obs_filters_.clear();  // each run starts its running normalisation again
    const auto refuse = [](std::string reason) { return PluginSimulationBeginResult{false, std::move(reason)}; };
    bool steps_physics = false;
    for (const auto& node : nodes) {
        if (node.node_type != "MuJoCoPlant") continue;
        steps_physics = true;
        std::lock_guard phys_lock(env_manager_.GetPhysicsMutex());
        if (!env_manager_.IsLoaded())
            return refuse("No MuJoCo model is loaded. Load the MuJoCo Plant's model from the Environment Library, "
                          "then run the graph.");
        // The node's model is the model it steps: never another one silently.
        // Its own mjcf_path, or else the model its pins were synced from.
        const auto param = [&](const char* key) {
            const auto it = node.parameters.find(key);
            return it == node.parameters.end() ? std::string() : it->second;
        };
        const std::string node_model = param("mjcf_path");
        const std::string synced_model = param("_meta_loaded_path");
        if (!node_model.empty() && !SameModelFile(node_model, loaded_mjcf_path_))
            return refuse("The MuJoCo Plant's model '" + node_model + "' is not the model loaded in the viewport ('" +
                          loaded_mjcf_path_ + "'). Load it from the Environment Library.");
        if (node_model.empty() && !synced_model.empty() && !SameModelFile(synced_model, loaded_mjcf_path_))
            return refuse("The MuJoCo Plant's pins were made for '" + synced_model + "' but the viewport has '" +
                          loaded_mjcf_path_ + "' loaded. Sync the node again or load that model.");
    }
    if (!steps_physics) return {};
    // One driver at a time: the viewport's own run is not stopped behind the user's back.
    if (sim_executor_.GetMode() != SimMode::Stopped)
        return refuse("The MuJoCo Viewport is running the model. Stop it there, then run the graph.");
    {
        std::lock_guard phys_lock(env_manager_.GetPhysicsMutex());
        graph_clock_.Begin(env_manager_.GetData()->time);
    }
    viewport_panel_.SetGraphSimActive(true);
    return {};
}

void MuJoCoPlugin::EndGraphRun() {
    viewport_panel_.SetGraphSimActive(false);
}

std::vector<PluginEnvironmentInfo> MuJoCoPlugin::GetEnvironments() {
    std::vector<PluginEnvironmentInfo> environments;
    for (const auto& task : MjTasks()) {
        PluginEnvironmentInfo info;
        std::string error;
        if (DescribeMjTask(task, (std::filesystem::path(assets_dir_) / task.model_file).string(), info, error))
            environments.push_back(std::move(info));
        else if (ctx_)
            ctx_->LogWarn("MuJoCo Simulation: " + task.id + " is not offered: " + error);
    }
    return environments;
}

PluginEnvironmentCreateResult MuJoCoPlugin::CreateEnvironments(const std::string& id,
                                                               const std::map<std::string, std::string>& settings,
                                                               int count) {
    const MjTaskSpec* task = FindMjTask(id);
    if (!task) return {nullptr, "The MuJoCo plugin has no environment '" + id + "'."};
    std::string error;
    IEnvironmentBatch* batch = MjEnvBatch::Create(
        *task, (std::filesystem::path(assets_dir_) / task->model_file).string(), settings, count, view_slot_, error);
    return {batch, error};
}

PluginNodeEvalResult MuJoCoPlugin::EvaluateNode(const PluginNodeEvalContext& ctx) {
    PluginNodeEvalResult result;

    if (ctx.node_type_name == "MuJoCoPlant") {
        // Only a graph run that BeginGraphRun accepted steps the physics.
        if (!viewport_panel_.IsGraphSimActive()) {
            result.success = false;
            result.error_message = "The MuJoCo Plant steps the physics only inside a graph run";
            return result;
        }

        // One physics transaction: the model cannot be replaced (LoadEnvironment)
        // or rendered while this tick applies, steps and reads it. Loading
        // another model is refused while the graph run holds this one.
        std::lock_guard phys_lock(env_manager_.GetPhysicsMutex());
        if (!env_manager_.IsLoaded()) {
            result.success = false;
            result.error_message = "No MuJoCo model loaded";
            return result;
        }

        mjData* d = env_manager_.GetMutableData();
        const mjModel* m = env_manager_.GetModel();
        if (!d || !m) {
            result.success = false;
            result.error_message = "MuJoCo model/data not available";
            return result;
        }

        // Controls from the input pins: "u" is the whole control vector by
        // actuator index; any other pin is one actuator by name (the pin names
        // ResolveDynamicPins gave it, "actuator_<i>" for an unnamed one).
        std::vector<std::pair<int, float>> controls;
        for (const auto& [pin_name, value] : ctx.input_values) {
            if (pin_name == "u") {
                if (std::holds_alternative<std::vector<float>>(value)) {
                    const auto& vec = std::get<std::vector<float>>(value);
                    for (size_t i = 0; i < vec.size() && static_cast<int>(i) < m->nu; ++i)
                        controls.emplace_back(static_cast<int>(i), vec[i]);
                } else if (std::holds_alternative<float>(value) && m->nu > 0) {
                    controls.emplace_back(0, std::get<float>(value));
                }
            } else if (std::holds_alternative<float>(value)) {
                const int id = ActuatorId(m, pin_name);
                if (id >= 0) controls.emplace_back(id, std::get<float>(value));
            }
        }

        // The physics follows the graph's clock (mj_graph_clock.h)
        for (const auto& [id, v] : controls) d->ctrl[id] = v;
        const int steps = graph_clock_.StepsForTick(d->time, m->opt.timestep, ctx.sim_time, ctx.dt);
        for (int i = 0; i < steps; ++i) mj_step(m, d);

        for (int i = 0; i < m->nsensor; i++) {
            const char* name = mj_id2name(m, mjOBJ_SENSOR, i);
            std::string sname = name ? name : ("sensor_" + std::to_string(i));
            int dim = m->sensor_dim[i];
            int adr = m->sensor_adr[i];
            if (dim == 1 && adr < m->nsensordata) {
                result.output_values[sname] = static_cast<float>(d->sensordata[adr]);
            } else {
                std::vector<float> vals(dim);
                for (int j = 0; j < dim && (adr + j) < m->nsensordata; j++) {
                    vals[j] = static_cast<float>(d->sensordata[adr + j]);
                }
                result.output_values[sname] = std::move(vals);
            }
        }
        result.output_values["qpos"] = ToFloats(d->qpos, m->nq);
        result.output_values["qvel"] = ToFloats(d->qvel, m->nv);
        if (m->nsensor > 0) {
            result.output_values["sensor"] = ToFloats(d->sensordata, m->nsensordata);
        }

        return result;
    }

    // --- RewardFunction node ---
    if (ctx.node_type_name == "RewardFunction") {
        rl::RewardConfig cfg = rl::RewardConfig::FromParameters(ctx.parameters);
        rl::RewardShaper shaper(cfg);

        rl::EnvState state;
        auto get_vec = [&](const std::string& key) -> std::vector<float> {
            auto it = ctx.input_values.find(key);
            if (it == ctx.input_values.end()) return {};
            if (std::holds_alternative<std::vector<float>>(it->second))
                return std::get<std::vector<float>>(it->second);
            if (std::holds_alternative<float>(it->second))
                return {std::get<float>(it->second)};
            return {};
        };

        state.qpos = get_vec("qpos");
        state.qvel = get_vec("qvel");
        state.ctrl = get_vec("ctrl");
        state.sensor = get_vec("sensor");
        state.sim_time = ctx.sim_time;

        std::lock_guard phys_lock(env_manager_.GetPhysicsMutex());
        const mjData* d = env_manager_.GetData();
        const mjModel* m = env_manager_.GetModel();
        if (d && m) {
            if (state.qpos.empty()) state.qpos = ToFloats(d->qpos, m->nq);
            if (state.qvel.empty()) state.qvel = ToFloats(d->qvel, m->nv);
            if (state.ctrl.empty()) state.ctrl = ToFloats(d->ctrl, m->nu);
            if (state.sensor.empty() && m->nsensordata > 0)
                state.sensor = ToFloats(d->sensordata, m->nsensordata);
        }

        float reward = shaper.ComputeReward(state);
        result.output_values["reward"] = reward;
        return result;
    }

    // --- ObservationFilter node ---
    if (ctx.node_type_name == "ObservationFilter") {
        rl::ObsFilterConfig cfg = rl::ObsFilterConfig::FromParameters(ctx.parameters);

        auto get_vec = [&](const std::string& key) -> std::vector<float> {
            auto it = ctx.input_values.find(key);
            if (it == ctx.input_values.end()) return {};
            if (std::holds_alternative<std::vector<float>>(it->second))
                return std::get<std::vector<float>>(it->second);
            if (std::holds_alternative<float>(it->second))
                return {std::get<float>(it->second)};
            return {};
        };

        std::vector<float> qpos = get_vec("qpos");
        std::vector<float> qvel = get_vec("qvel");
        std::vector<float> sensors = get_vec("sensor");

        std::lock_guard phys_lock(env_manager_.GetPhysicsMutex());
        const mjData* d = env_manager_.GetData();
        const mjModel* m = env_manager_.GetModel();
        if (d && m) {
            if (qpos.empty()) qpos = ToFloats(d->qpos, m->nq);
            if (qvel.empty()) qvel = ToFloats(d->qvel, m->nv);
            if (sensors.empty() && m->nsensordata > 0)
                sensors = ToFloats(d->sensordata, m->nsensordata);
        }

        // Keep the running normalisation across the ticks of a run, one filter
        // per settings; BeginGraphRun starts them again for each run.
        // Nodes with the same settings share one filter: the evaluation
        // context does not say which node is evaluated.
        const std::string key = std::string(cfg.include_qpos ? "p" : "-") + (cfg.include_qvel ? "v" : "-") +
                                (cfg.include_sensors ? "s" : "-") + (cfg.normalize ? "n" : "-");
        auto filter = obs_filters_.try_emplace(key, cfg).first;
        std::vector<float> obs = filter->second.Filter(qpos, qvel, sensors);
        result.output_values["obs"] = std::move(obs);
        return result;
    }

    result.success = false;
    result.error_message = "EvaluateNode not implemented for: " + ctx.node_type_name;
    return result;
}

DynamicPinResult MuJoCoPlugin::ResolveDynamicPins(
    const std::string& node_type_name,
    const std::map<std::string, std::string>& parameters)
{
    if (node_type_name != "MuJoCoPlant") return {};

    auto it = parameters.find("mjcf_path");
    std::string mjcf_path;
    if (it != parameters.end() && !it->second.empty()) {
        mjcf_path = it->second;
    } else if (!loaded_mjcf_path_.empty()) {
        mjcf_path = loaded_mjcf_path_;
    } else {
        return {};
    }

    // Parse the MJCF file to discover actuators and sensors
    MjcfModelInfo info = ParseMjcfFile(mjcf_path);
    if (!info.valid) {
        spdlog::warn("MuJoCoPlant: Failed to parse MJCF '{}': {}", mjcf_path, info.error);
        return {};
    }

    DynamicPinResult result;

    // Check interface mode
    auto iface_it = parameters.find("interface");
    std::string iface = (iface_it != parameters.end()) ? iface_it->second : "bus";

    if (iface == "bus") {
        // Per-actuator input pins
        for (const auto& act : info.actuators) {
            PluginNodeTypeInfo::PinInfo pin;
            pin.name = act.name;
            pin.type = "Scalar";
            pin.is_input = true;
            result.pins.push_back(std::move(pin));
        }
    } else {
        // Single vector input
        PluginNodeTypeInfo::PinInfo pin;
        pin.name = "u";
        pin.type = "Tensor";
        pin.is_input = true;
        result.pins.push_back(std::move(pin));
    }

    // Per-sensor output pins (bus mode) or single sensor vector
    if (iface == "bus") {
        for (const auto& sens : info.sensors) {
            PluginNodeTypeInfo::PinInfo pin;
            pin.name = sens.name;
            pin.type = (sens.dim == 1) ? "Scalar" : "Tensor";
            pin.is_input = false;
            result.pins.push_back(std::move(pin));
        }
    } else {
        PluginNodeTypeInfo::PinInfo pin;
        pin.name = "sensor";
        pin.type = "Tensor";
        pin.is_input = false;
        result.pins.push_back(std::move(pin));
    }

    // Always add state and image outputs
    result.pins.push_back({"qpos", "Tensor", false});
    result.pins.push_back({"qvel", "Tensor", false});
    result.pins.push_back({"rgb", "Image", false});
    result.pins.push_back({"depth", "Image", false});

    // Metadata
    result.metadata["model_name"] = info.model_name;
    result.metadata["loaded_path"] = mjcf_path;
    result.metadata["nu"] = std::to_string(info.nu);
    result.metadata["nq"] = std::to_string(info.nq);
    result.metadata["nv"] = std::to_string(info.nv);
    result.metadata["nsensor"] = std::to_string(info.nsensor);
    result.metadata["timestep"] = std::to_string(info.timestep);

    spdlog::info("MuJoCoPlant: Resolved {} pins for '{}' ({})",
                 result.pins.size(), mjcf_path, info.model_name);
    return result;
}

} // namespace cyxwiz::plugin::mujoco

// =============================================================================
// Plugin Entry Point
// =============================================================================

CYXWIZ_PLUGIN_ENTRY(cyxwiz::plugin::mujoco::MuJoCoPlugin)
