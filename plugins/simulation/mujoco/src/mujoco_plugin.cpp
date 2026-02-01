#include "mujoco_plugin.h"
#include "mj_mjcf_parser.h"

#include <spdlog/spdlog.h>
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

namespace cyxwiz::plugin::mujoco {

// =============================================================================
// IPlugin Lifecycle
// =============================================================================

bool MuJoCoPlugin::OnLoad(PluginContext& ctx) {
    ctx.LogInfo("MuJoCo Simulation: OnLoad");

    // Log MuJoCo version (mj_versionString available since MuJoCo 2.3.0)
    int ver = mj_version();
    ctx.LogInfo("MuJoCo Simulation: Using MuJoCo version " +
                std::to_string(ver / 100) + "." +
                std::to_string((ver / 10) % 10) + "." +
                std::to_string(ver % 10));

    // Capture main GLFW window for renderer context management.
    main_window_ = glfwGetCurrentContext();

    // Set assets directory for the environment library.
    // Assets are in the source tree: plugins/simulation/mujoco/assets/
    // At runtime, resolve relative to known plugin search paths.
    env_library_.SetAssetsDir("plugins/simulation/mujoco/assets");

    // Scan for previously downloaded Menagerie models
    env_library_.ScanDownloadedModels();

    // Wire browser panel load callback and downloader
    browser_panel_.SetLoadCallback([this](const std::string& path) {
        return LoadEnvironment(path);
    });
    browser_panel_.SetDownloader(&menagerie_downloader_);

    state_ = PluginState::Loaded;
    return true;
}

bool MuJoCoPlugin::OnInitialize(PluginContext& ctx) {
    ctx.LogInfo("MuJoCo Simulation: OnInitialize");

    // Note: Registration of interfaces (IPanelProvider, INodeProvider, ITrainingHook)
    // is handled engine-side via QueryInterface in PluginManager to avoid DLL singleton
    // duplication issues. Do NOT call ctx.Register*() here.

    state_ = PluginState::Active;
    return true;
}

void MuJoCoPlugin::OnShutdown(PluginContext& ctx) {
    ctx.LogInfo("MuJoCo Simulation: OnShutdown");
    renderer_.Shutdown();
    env_manager_.Close();
    current_env_id_.clear();
    state_ = PluginState::Loaded;
}

void MuJoCoPlugin::OnUnload(PluginContext& ctx) {
    ctx.LogInfo("MuJoCo Simulation: OnUnload");
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

    if (!env_manager_.LoadModel(mjcf_path)) {
        return false;
    }

    // Reset environment to get initial state
    env_manager_.Reset();

    // Wire simulation executor and viewport
    sim_executor_.Stop();
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

void MuJoCoPlugin::RenderPanel(const std::string& panel_id, bool* visible) {
    // Lazy renderer init — deferred from LoadEnvironment to avoid
    // cross-DLL GL context/glad issues. Init happens here in the
    // engine's render loop where the GL context is guaranteed current.
    if (renderer_needs_init_ && env_manager_.IsLoaded() && main_window_) {
        renderer_needs_init_ = false;
        if (!renderer_.Initialize(env_manager_.GetModel(), 640, 480, main_window_)) {
            spdlog::warn("[MuJoCo] Renderer initialization failed — viewport disabled");
        } else {
            renderer_.ResetCamera(env_manager_.GetModel());
        }
    }

    if (panel_id == "mujoco_viewport") {
        viewport_panel_.Render(env_manager_, renderer_, visible);
    } else if (panel_id == "mujoco_env_browser") {
        browser_panel_.Render(env_library_, env_manager_, visible);
    }
}

// =============================================================================
// ITrainingHook - RL Metric Logging
// =============================================================================

void MuJoCoPlugin::OnTrainingStart(TrainingContext& ctx) {
    spdlog::info("[MuJoCo] RL training started — env='{}', epochs={}",
                 current_env_id_.empty() ? "(none)" : current_env_id_,
                 ctx.total_epochs);
}

void MuJoCoPlugin::OnTrainingEnd([[maybe_unused]] TrainingContext& ctx) {
    spdlog::info("[MuJoCo] RL training ended — {} episodes completed, "
                 "mean_reward={:.2f}, mean_length={:.1f}",
                 env_manager_.GetEpisodeCount(),
                 env_manager_.GetMeanEpisodeReward(),
                 env_manager_.GetMeanEpisodeLength());
}

void MuJoCoPlugin::OnEpochEnd(TrainingContext& ctx) {
    // Inject RL-specific metrics into training context
    if (env_manager_.IsLoaded()) {
        ctx.custom_metrics["mean_episode_reward"] = env_manager_.GetMeanEpisodeReward();
        ctx.custom_metrics["mean_episode_length"] = env_manager_.GetMeanEpisodeLength();
        ctx.custom_metrics["total_episodes"] = static_cast<float>(env_manager_.GetEpisodeCount());
        ctx.custom_metrics["total_env_steps"] = static_cast<float>(env_manager_.GetCurrentStep());
    }
}

bool MuJoCoPlugin::ShouldStopEarly([[maybe_unused]] const TrainingContext& ctx) {
    // Stop if reward exceeds the solved threshold for this environment
    if (reward_threshold_ < std::numeric_limits<float>::max()) {
        float mean_reward = env_manager_.GetMeanEpisodeReward();
        if (mean_reward >= reward_threshold_) {
            spdlog::info("[MuJoCo] Environment solved! mean_reward={:.2f} >= threshold={:.2f}",
                         mean_reward, reward_threshold_);
            return true;
        }
    }
    return false;
}

// =============================================================================
// INodeProvider - RL Simulation Nodes
// =============================================================================

std::vector<PluginNodeTypeInfo> MuJoCoPlugin::GetNodeTypes() {
    std::vector<PluginNodeTypeInfo> nodes;

    // --- MuJoCo Environment node ---
    // Loads an MJCF model and provides the Gymnasium-compatible env interface
    {
        PluginNodeTypeInfo n;
        n.type_name    = "MuJoCoEnv";
        n.display_name = "MuJoCo Environment";
        n.category     = "RL / Simulation";
        n.description  = "Load a MuJoCo MJCF model as a Gymnasium-compatible environment";
        n.color        = 0xFF44AA88;  // Teal

        n.pins.push_back({"env", "Environment", false});

        n.default_parameters["mjcf_path"]       = "inverted_pendulum.xml";
        n.default_parameters["max_steps"]        = "1000";
        n.default_parameters["frame_skip"]       = "1";
        n.default_parameters["reward_threshold"] = "inf";
        nodes.push_back(std::move(n));
    }

    // --- Reward Function node ---
    // Configurable reward shaping for RL training
    {
        PluginNodeTypeInfo n;
        n.type_name    = "RewardFunction";
        n.display_name = "Reward Function";
        n.category     = "RL / Simulation";
        n.description  = "Define reward shaping for RL training (alive bonus, control penalty, etc.)";
        n.color        = 0xFF44AADD;  // Orange-gold

        n.pins.push_back({"env", "Environment", true});
        n.pins.push_back({"env_out", "Environment", false});

        n.default_parameters["alive_bonus"]    = "1.0";
        n.default_parameters["ctrl_cost_weight"] = "0.1";
        n.default_parameters["velocity_reward"] = "true";
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

        n.pins.push_back({"env", "Environment", true});
        n.pins.push_back({"obs", "Tensor", false});

        n.default_parameters["include_qpos"] = "true";
        n.default_parameters["include_qvel"] = "true";
        n.default_parameters["include_sensors"] = "false";
        n.default_parameters["normalize"]    = "true";
        nodes.push_back(std::move(n));
    }

    // --- RL Agent node ---
    // PPO/SAC agent that interacts with the environment
    {
        PluginNodeTypeInfo n;
        n.type_name    = "RLAgent";
        n.display_name = "RL Agent";
        n.category     = "RL / Simulation";
        n.description  = "Reinforcement learning agent (PPO or SAC) for MuJoCo environments";
        n.color        = 0xFF4488DD;  // Red-orange

        n.pins.push_back({"env", "Environment", true});
        n.pins.push_back({"policy", "Model", false});

        n.default_parameters["algorithm"]       = "PPO";
        n.default_parameters["learning_rate"]   = "3e-4";
        n.default_parameters["gamma"]           = "0.99";
        n.default_parameters["gae_lambda"]      = "0.95";
        n.default_parameters["clip_range"]      = "0.2";
        n.default_parameters["n_steps"]         = "2048";
        n.default_parameters["batch_size"]      = "64";
        n.default_parameters["n_epochs"]        = "10";
        n.default_parameters["hidden_sizes"]    = "64,64";
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
        n.pins.push_back({"u", "Tensor", true});       // control vector input
        n.pins.push_back({"sensor", "Tensor", false});  // sensor output
        n.pins.push_back({"rgb", "Image", false});      // camera image
        n.pins.push_back({"depth", "Image", false});    // depth image

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

    if (node_type_name == "MuJoCoEnv") {
        std::string mjcf = get("mjcf_path", "inverted_pendulum.xml");
        std::string max_steps = get("max_steps", "1000");
        std::string frame_skip = get("frame_skip", "1");
        return
            "import gymnasium as gym\n"
            "import mujoco\n"
            "\n"
            "env = gym.make(\n"
            "    'InvertedPendulum-v5',  # or custom: gym.make('MuJoCoEnv', xml_file='" + mjcf + "')\n"
            "    max_episode_steps=" + max_steps + ",\n"
            "    frame_skip=" + frame_skip + ",\n"
            ")\n"
            "obs, info = env.reset()\n";
    }

    if (node_type_name == "RewardFunction") {
        std::string alive = get("alive_bonus", "1.0");
        std::string ctrl_w = get("ctrl_cost_weight", "0.1");
        std::string vel = get("velocity_reward", "true");
        return
            "def compute_reward(obs, action, next_obs, info):\n"
            "    reward = " + alive + "  # alive bonus\n"
            "    reward -= " + ctrl_w + " * (action ** 2).sum()  # control cost\n" +
            (vel == "true"
                ? "    reward += next_obs[0]  # forward velocity reward\n"
                : "") +
            "    return reward\n"
            "\n"
            "env = gym.wrappers.TransformReward(env, compute_reward)\n";
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

    if (node_type_name == "RLAgent") {
        std::string algo = get("algorithm", "PPO");
        std::string lr = get("learning_rate", "3e-4");
        std::string gamma = get("gamma", "0.99");
        std::string hidden = get("hidden_sizes", "64,64");
        std::string n_steps = get("n_steps", "2048");
        std::string batch = get("batch_size", "64");
        std::string n_epochs = get("n_epochs", "10");
        std::string clip = get("clip_range", "0.2");
        std::string gae = get("gae_lambda", "0.95");

        // Convert "64,64" to [64, 64]
        std::string policy_kwargs = "dict(net_arch=[" + hidden + "])";

        if (algo == "PPO") {
            return
                "from stable_baselines3 import PPO\n"
                "\n"
                "model = PPO(\n"
                "    'MlpPolicy',\n"
                "    env,\n"
                "    learning_rate=" + lr + ",\n"
                "    gamma=" + gamma + ",\n"
                "    gae_lambda=" + gae + ",\n"
                "    clip_range=" + clip + ",\n"
                "    n_steps=" + n_steps + ",\n"
                "    batch_size=" + batch + ",\n"
                "    n_epochs=" + n_epochs + ",\n"
                "    policy_kwargs=" + policy_kwargs + ",\n"
                "    verbose=1,\n"
                ")\n"
                "model.learn(total_timesteps=1_000_000)\n";
        } else {
            return
                "from stable_baselines3 import SAC\n"
                "\n"
                "model = SAC(\n"
                "    'MlpPolicy',\n"
                "    env,\n"
                "    learning_rate=" + lr + ",\n"
                "    gamma=" + gamma + ",\n"
                "    batch_size=" + batch + ",\n"
                "    policy_kwargs=" + policy_kwargs + ",\n"
                "    verbose=1,\n"
                ")\n"
                "model.learn(total_timesteps=1_000_000)\n";
        }
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

DynamicPinResult MuJoCoPlugin::ResolveDynamicPins(
    const std::string& node_type_name,
    const std::map<std::string, std::string>& parameters)
{
    if (node_type_name != "MuJoCoPlant") return {};

    auto it = parameters.find("mjcf_path");
    if (it == parameters.end() || it->second.empty()) return {};

    // Parse the MJCF file to discover actuators and sensors
    MjcfModelInfo info = ParseMjcfFile(it->second);
    if (!info.valid) {
        spdlog::warn("MuJoCoPlant: Failed to parse MJCF '{}': {}", it->second, info.error);
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

    // Always add image outputs
    result.pins.push_back({"rgb", "Image", false});
    result.pins.push_back({"depth", "Image", false});

    // Metadata
    result.metadata["model_name"] = info.model_name;
    result.metadata["nu"] = std::to_string(info.nu);
    result.metadata["nq"] = std::to_string(info.nq);
    result.metadata["nv"] = std::to_string(info.nv);
    result.metadata["nsensor"] = std::to_string(info.nsensor);
    result.metadata["timestep"] = std::to_string(info.timestep);

    spdlog::info("MuJoCoPlant: Resolved {} pins for '{}' ({})",
                 result.pins.size(), it->second, info.model_name);
    return result;
}

} // namespace cyxwiz::plugin::mujoco

// =============================================================================
// Plugin Entry Point
// =============================================================================

CYXWIZ_PLUGIN_ENTRY(cyxwiz::plugin::mujoco::MuJoCoPlugin)
