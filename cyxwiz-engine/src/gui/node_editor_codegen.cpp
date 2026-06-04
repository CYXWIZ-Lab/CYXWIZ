#include "node_editor.h"
#include "panels/script_editor.h"
#include "../core/async_task_manager.h"
#include "../plugin/registries/plugin_node_registry.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <set>
#include <map>
#include <queue>
#include <functional>
#include <vector>

namespace gui {

namespace {

std::string TrimCopy(const std::string& value) {
    auto begin = std::find_if_not(value.begin(), value.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    });
    auto end = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    }).base();

    if (begin >= end) {
        return "";
    }
    return std::string(begin, end);
}

std::string CsvToPythonListLiteral(const std::string& csv, const std::string& fallback) {
    std::stringstream ss(csv.empty() ? fallback : csv);
    std::string item;
    std::vector<std::string> values;

    while (std::getline(ss, item, ',')) {
        item = TrimCopy(item);
        if (!item.empty()) {
            values.push_back(item);
        }
    }

    if (values.empty()) {
        return "[" + fallback + "]";
    }

    std::string out = "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out += ", ";
        out += values[i];
    }
    out += "]";
    return out;
}

std::string GetParamOrDefault(const MLNode& node, const std::string& key, const std::string& fallback) {
    auto it = node.parameters.find(key);
    if (it == node.parameters.end() || TrimCopy(it->second).empty()) {
        return fallback;
    }
    return TrimCopy(it->second);
}

std::string PythonBoolLiteral(const std::string& value) {
    std::string lower = TrimCopy(value);
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return (lower == "true" || lower == "1" || lower == "yes") ? "True" : "False";
}

} // namespace

void NodeEditor::GeneratePythonCode() {
    // Validate graph before generating code
    std::string error_message;
    if (!ValidateGraph(error_message)) {
        spdlog::error("Graph validation failed: {}", error_message);
        // TODO: Show error dialog to user
        return;
    }

    GenerateCodeForFramework(selected_framework_);
}

void NodeEditor::GenerateCodeForFramework(CodeFramework framework) {
    spdlog::info("Generating code from node graph (async)...");

    if (nodes_.empty()) {
        spdlog::warn("No nodes in graph - cannot generate code");
        return;
    }

    // Get topologically sorted node order (do this synchronously for validation)
    std::vector<int> sorted_ids = TopologicalSort();
    if (sorted_ids.empty()) {
        spdlog::error("Failed to perform topological sort - graph may have cycles");
        return;
    }

    // Copy graph data for thread safety
    std::vector<MLNode> nodes_copy = nodes_;
    std::vector<NodeLink> links_copy = links_;
    size_t total_nodes = sorted_ids.size();

    // Determine framework name
    std::string framework_name;
    switch (framework) {
        case CodeFramework::PyTorch: framework_name = "PyTorch"; break;
        case CodeFramework::TensorFlow: framework_name = "TensorFlow"; break;
        case CodeFramework::Keras: framework_name = "Keras"; break;
        case CodeFramework::PyCyxWiz: framework_name = "PyCyxWiz"; break;
        default: framework_name = "Unknown"; break;
    }

    // Store result for completion callback
    auto result = std::make_shared<std::string>();
    auto fw_name = std::make_shared<std::string>(framework_name);

    // Capture script_editor_ for completion callback
    auto script_editor = script_editor_;

    // Run code generation async
    cyxwiz::AsyncTaskManager::Instance().RunAsync(
        "Generate " + framework_name + " Code",
        [this, framework, sorted_ids, nodes_copy, total_nodes, result, fw_name](cyxwiz::LambdaTask& task) {
            task.ReportProgress(0.0f, "Starting code generation...");

            std::string code;

            // Generate code based on selected framework
            task.ReportProgress(0.1f, "Generating " + *fw_name + " code...");

            switch (framework) {
                case CodeFramework::PyTorch:
                    code = GeneratePyTorchCode(sorted_ids);
                    break;
                case CodeFramework::TensorFlow:
                    code = GenerateTensorFlowCode(sorted_ids);
                    break;
                case CodeFramework::Keras:
                    code = GenerateKerasCode(sorted_ids);
                    break;
                case CodeFramework::PyCyxWiz:
                    code = GeneratePyCyxWizCode(sorted_ids);
                    break;
                default:
                    task.MarkFailed("Unknown framework selected");
                    return;
            }

            if (task.ShouldStop()) {
                task.MarkFailed("Code generation cancelled");
                return;
            }

            task.ReportProgress(0.9f, "Finalizing...");

            // Store result
            *result = std::move(code);

            task.ReportProgress(1.0f, "Complete!");
            spdlog::info("Generated {} code ({} lines)", *fw_name, std::count(result->begin(), result->end(), '\n'));
        },
        // Progress callback (optional - can be used for detailed UI updates)
        nullptr,
        // Completion callback - runs on main thread
        [script_editor, result, fw_name](bool success, const std::string& error) {
            if (success && script_editor) {
                script_editor->LoadGeneratedCode(*result, fw_name->c_str());
                script_editor->SetVisible(true);
                spdlog::info("Code sent to Script Editor panel");
            } else if (!success) {
                spdlog::error("Code generation failed: {}", error);
            } else {
                spdlog::warn("Script Editor panel not available");
            }
        }
    );
}

// Helper to check if graph contains RL nodes
bool NodeEditor::IsRLGraph(const std::vector<int>& sorted_ids) const {
    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;
        if (node->type == NodeType::GymEnvironment ||
            node->type == NodeType::RLTraining ||
            node->type == NodeType::ReplayBufferNode) {
            return true;
        }
    }
    return false;
}

// Generate RL-specific PyTorch code using gymnasium + stable-baselines3
std::string NodeEditor::GenerateRLPyTorchCode(const std::vector<int>& sorted_ids) const {
    std::string code;

    // Header with RL imports
    code += "# Auto-generated Reinforcement Learning code from CyxWiz Node Editor\n";
    code += "# Generated at: " + std::string(__DATE__) + " " + std::string(__TIME__) + "\n\n";
    code += "import gymnasium as gym\n";
    code += "import numpy as np\n";
    code += "import torch\n";
    code += "import torch.nn as nn\n";
    code += "from stable_baselines3 import PPO, A2C, SAC, TD3, DQN\n";
    code += "from stable_baselines3.common.env_util import make_vec_env\n";
    code += "from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback\n";
    code += "from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv\n\n";

    // Extract parameters from nodes (with enhanced defaults)
    std::string env_name = "CartPole-v1";
    std::string env_type = "classic_control";
    std::string render_mode = "rgb_array";
    std::string max_episode_steps = "500";
    std::string n_envs = "4";
    std::string seed = "42";
    bool normalize_obs = false;
    bool normalize_reward = false;
    std::string frame_stack = "1";
    bool render = false;

    std::string algorithm = "PPO";
    std::string total_timesteps = "100000";
    std::string learning_rate = "3e-4";
    std::string batch_size = "64";
    std::string n_epochs = "10";
    std::string gamma = "0.99";
    std::string gae_lambda = "0.95";
    std::string clip_range = "0.2";
    std::string ent_coef = "0.0";
    std::string vf_coef = "0.5";
    std::string max_grad_norm = "0.5";
    std::string tau = "0.005";
    std::string eval_freq = "5000";
    std::string save_freq = "10000";
    bool tensorboard = true;

    std::string hidden_sizes = "128,128";
    std::string activation = "ReLU";
    std::string action_space = "discrete";
    bool ortho_init = true;

    std::string buffer_capacity = "100000";
    bool prioritized = false;
    std::string alpha = "0.6";
    std::string beta_start = "0.4";
    std::string n_step = "1";

    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;

        if (node->type == NodeType::GymEnvironment) {
            auto it = node->parameters.find("env_name");
            if (it != node->parameters.end()) env_name = it->second;
            it = node->parameters.find("env_type");
            if (it != node->parameters.end()) env_type = it->second;
            it = node->parameters.find("render");
            if (it != node->parameters.end()) render = (it->second == "true");
            it = node->parameters.find("render_mode");
            if (it != node->parameters.end()) render_mode = it->second;
            it = node->parameters.find("max_episode_steps");
            if (it != node->parameters.end()) max_episode_steps = it->second;
            it = node->parameters.find("n_envs");
            if (it != node->parameters.end()) n_envs = it->second;
            it = node->parameters.find("seed");
            if (it != node->parameters.end()) seed = it->second;
            it = node->parameters.find("normalize_obs");
            if (it != node->parameters.end()) normalize_obs = (it->second == "true");
            it = node->parameters.find("normalize_reward");
            if (it != node->parameters.end()) normalize_reward = (it->second == "true");
            it = node->parameters.find("frame_stack");
            if (it != node->parameters.end()) frame_stack = it->second;
        }
        else if (node->type == NodeType::RLTraining) {
            auto it = node->parameters.find("algorithm");
            if (it != node->parameters.end()) algorithm = it->second;
            it = node->parameters.find("total_timesteps");
            if (it != node->parameters.end()) total_timesteps = it->second;
            it = node->parameters.find("learning_rate");
            if (it != node->parameters.end()) learning_rate = it->second;
            it = node->parameters.find("batch_size");
            if (it != node->parameters.end()) batch_size = it->second;
            it = node->parameters.find("n_epochs");
            if (it != node->parameters.end()) n_epochs = it->second;
            it = node->parameters.find("gamma");
            if (it != node->parameters.end()) gamma = it->second;
            it = node->parameters.find("gae_lambda");
            if (it != node->parameters.end()) gae_lambda = it->second;
            it = node->parameters.find("clip_range");
            if (it != node->parameters.end()) clip_range = it->second;
            it = node->parameters.find("ent_coef");
            if (it != node->parameters.end()) ent_coef = it->second;
            it = node->parameters.find("vf_coef");
            if (it != node->parameters.end()) vf_coef = it->second;
            it = node->parameters.find("max_grad_norm");
            if (it != node->parameters.end()) max_grad_norm = it->second;
            it = node->parameters.find("tau");
            if (it != node->parameters.end()) tau = it->second;
            it = node->parameters.find("eval_freq");
            if (it != node->parameters.end()) eval_freq = it->second;
            it = node->parameters.find("save_freq");
            if (it != node->parameters.end()) save_freq = it->second;
            it = node->parameters.find("tensorboard");
            if (it != node->parameters.end()) tensorboard = (it->second == "true");
        }
        else if (node->type == NodeType::PolicyNetwork) {
            auto it = node->parameters.find("hidden_sizes");
            if (it != node->parameters.end()) hidden_sizes = it->second;
            it = node->parameters.find("activation");
            if (it != node->parameters.end()) activation = it->second;
            it = node->parameters.find("action_space");
            if (it != node->parameters.end()) action_space = it->second;
            it = node->parameters.find("ortho_init");
            if (it != node->parameters.end()) ortho_init = (it->second == "true");
        }
        else if (node->type == NodeType::ReplayBufferNode) {
            auto it = node->parameters.find("capacity");
            if (it != node->parameters.end()) buffer_capacity = it->second;
            it = node->parameters.find("batch_size");
            if (it != node->parameters.end()) batch_size = it->second;
            it = node->parameters.find("prioritized");
            if (it != node->parameters.end()) prioritized = (it->second == "true");
            it = node->parameters.find("alpha");
            if (it != node->parameters.end()) alpha = it->second;
            it = node->parameters.find("beta_start");
            if (it != node->parameters.end()) beta_start = it->second;
            it = node->parameters.find("n_step");
            if (it != node->parameters.end()) n_step = it->second;
        }
    }

    // Configuration section with enhanced parameters
    code += "# =============== Configuration ===============\n";
    code += "# Environment\n";
    code += "ENV_NAME = \"" + env_name + "\"\n";
    code += "ENV_TYPE = \"" + env_type + "\"  # classic_control, box2d, mujoco, atari\n";
    code += "RENDER = " + std::string(render ? "True" : "False") + "\n";
    code += "RENDER_MODE = \"" + render_mode + "\"\n";
    code += "MAX_EPISODE_STEPS = " + max_episode_steps + "\n";
    code += "N_ENVS = " + n_envs + "\n";
    code += "SEED = " + seed + "\n";
    code += "NORMALIZE_OBS = " + std::string(normalize_obs ? "True" : "False") + "\n";
    code += "NORMALIZE_REWARD = " + std::string(normalize_reward ? "True" : "False") + "\n";
    code += "FRAME_STACK = " + frame_stack + "\n\n";

    code += "# Algorithm\n";
    code += "ALGORITHM = \"" + algorithm + "\"\n";
    code += "TOTAL_TIMESTEPS = " + total_timesteps + "\n";
    code += "LEARNING_RATE = " + learning_rate + "\n";
    code += "BATCH_SIZE = " + batch_size + "\n";
    code += "N_EPOCHS = " + n_epochs + "  # PPO epochs per update\n\n";

    code += "# Discount and advantage\n";
    code += "GAMMA = " + gamma + "\n";
    code += "GAE_LAMBDA = " + gae_lambda + "\n\n";

    code += "# PPO-specific\n";
    code += "CLIP_RANGE = " + clip_range + "\n";
    code += "ENT_COEF = " + ent_coef + "  # Entropy coefficient\n";
    code += "VF_COEF = " + vf_coef + "  # Value function coefficient\n";
    code += "MAX_GRAD_NORM = " + max_grad_norm + "\n\n";

    code += "# SAC/TD3-specific\n";
    code += "TAU = " + tau + "  # Soft update coefficient\n\n";

    code += "# Network architecture\n";
    code += "HIDDEN_SIZES = [" + hidden_sizes + "]\n";
    code += "ACTIVATION = \"" + activation + "\"\n";
    code += "ACTION_SPACE = \"" + action_space + "\"\n";
    code += "ORTHO_INIT = " + std::string(ortho_init ? "True" : "False") + "\n\n";

    code += "# Replay buffer (for off-policy algorithms)\n";
    code += "BUFFER_SIZE = " + buffer_capacity + "\n";
    code += "PRIORITIZED = " + std::string(prioritized ? "True" : "False") + "\n";
    code += "PER_ALPHA = " + alpha + "\n";
    code += "PER_BETA_START = " + beta_start + "\n";
    code += "N_STEP = " + n_step + "\n\n";

    code += "# Logging and checkpointing\n";
    code += "EVAL_FREQ = " + eval_freq + "\n";
    code += "SAVE_FREQ = " + save_freq + "\n";
    code += "TENSORBOARD = " + std::string(tensorboard ? "True" : "False") + "\n\n";

    // Custom network architecture
    code += "# =============== Custom Policy Network ===============\n";
    code += "from stable_baselines3.common.policies import ActorCriticPolicy\n";
    code += "from stable_baselines3.common.torch_layers import BaseFeaturesExtractor\n\n";

    code += "# Activation function mapping\n";
    code += "ACTIVATION_FNS = {\n";
    code += "    \"ReLU\": nn.ReLU,\n";
    code += "    \"Tanh\": nn.Tanh,\n";
    code += "    \"GELU\": nn.GELU,\n";
    code += "    \"Swish\": nn.SiLU,\n";
    code += "    \"LeakyReLU\": nn.LeakyReLU,\n";
    code += "}\n\n";

    code += "class CustomNetwork(BaseFeaturesExtractor):\n";
    code += "    \"\"\"Custom feature extractor based on node editor configuration.\"\"\"\n";
    code += "    def __init__(self, observation_space, features_dim=None):\n";
    code += "        features_dim = features_dim or HIDDEN_SIZES[-1]\n";
    code += "        super().__init__(observation_space, features_dim)\n";
    code += "        n_input = observation_space.shape[0]\n";
    code += "        \n";
    code += "        layers = []\n";
    code += "        in_features = n_input\n";
    code += "        activation_cls = ACTIVATION_FNS.get(ACTIVATION, nn.ReLU)\n";
    code += "        \n";
    code += "        for hidden_size in HIDDEN_SIZES:\n";
    code += "            layers.append(nn.Linear(in_features, hidden_size))\n";
    code += "            layers.append(activation_cls())\n";
    code += "            in_features = hidden_size\n";
    code += "        \n";
    code += "        self.net = nn.Sequential(*layers)\n";
    code += "        \n";
    code += "        # Apply orthogonal initialization if enabled\n";
    code += "        if ORTHO_INIT:\n";
    code += "            for layer in self.net:\n";
    code += "                if isinstance(layer, nn.Linear):\n";
    code += "                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))\n";
    code += "                    nn.init.constant_(layer.bias, 0.0)\n";
    code += "    \n";
    code += "    def forward(self, observations):\n";
    code += "        return self.net(observations)\n\n";

    // Policy kwargs
    code += "# Policy kwargs with custom network\n";
    code += "policy_kwargs = dict(\n";
    code += "    features_extractor_class=CustomNetwork,\n";
    code += "    features_extractor_kwargs=dict(features_dim=HIDDEN_SIZES[-1]),\n";
    code += "    net_arch=[dict(pi=HIDDEN_SIZES, vf=HIDDEN_SIZES)],\n";
    code += "    ortho_init=ORTHO_INIT,\n";
    code += ")\n\n";

    // Main training code
    code += "# =============== Training ===============\n";
    code += "from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack\n\n";

    code += "def create_env(n_envs=N_ENVS, seed=SEED):\n";
    code += "    \"\"\"Create and wrap environment with optional normalization.\"\"\"\n";
    code += "    env = make_vec_env(ENV_NAME, n_envs=n_envs, seed=seed)\n";
    code += "    \n";
    code += "    # Apply frame stacking for image-based environments\n";
    code += "    if FRAME_STACK > 1:\n";
    code += "        env = VecFrameStack(env, n_stack=FRAME_STACK)\n";
    code += "    \n";
    code += "    # Apply observation and reward normalization\n";
    code += "    if NORMALIZE_OBS or NORMALIZE_REWARD:\n";
    code += "        env = VecNormalize(env, norm_obs=NORMALIZE_OBS, norm_reward=NORMALIZE_REWARD)\n";
    code += "    \n";
    code += "    return env\n\n";

    code += "def train():\n";
    code += "    # Create vectorized environment with wrappers\n";
    code += "    env = create_env(n_envs=N_ENVS, seed=SEED)\n";
    code += "    \n";
    code += "    # Create evaluation environment\n";
    code += "    eval_env = create_env(n_envs=1, seed=SEED + 1000)\n";
    code += "    \n";
    code += "    # Tensorboard logging path\n";
    code += "    tb_log = \"./tb_logs/\" if TENSORBOARD else None\n";
    code += "    \n";
    code += "    # Select algorithm with full hyperparameters\n";
    code += "    if ALGORITHM == \"PPO\":\n";
    code += "        model = PPO(\n";
    code += "            \"MlpPolicy\",\n";
    code += "            env,\n";
    code += "            policy_kwargs=policy_kwargs,\n";
    code += "            learning_rate=LEARNING_RATE,\n";
    code += "            n_steps=2048,\n";
    code += "            batch_size=BATCH_SIZE,\n";
    code += "            n_epochs=N_EPOCHS,\n";
    code += "            gamma=GAMMA,\n";
    code += "            gae_lambda=GAE_LAMBDA,\n";
    code += "            clip_range=CLIP_RANGE,\n";
    code += "            ent_coef=ENT_COEF,\n";
    code += "            vf_coef=VF_COEF,\n";
    code += "            max_grad_norm=MAX_GRAD_NORM,\n";
    code += "            seed=SEED,\n";
    code += "            verbose=1,\n";
    code += "            tensorboard_log=tb_log\n";
    code += "        )\n";
    code += "    elif ALGORITHM == \"A2C\":\n";
    code += "        model = A2C(\n";
    code += "            \"MlpPolicy\",\n";
    code += "            env,\n";
    code += "            policy_kwargs=policy_kwargs,\n";
    code += "            learning_rate=LEARNING_RATE,\n";
    code += "            gamma=GAMMA,\n";
    code += "            gae_lambda=GAE_LAMBDA,\n";
    code += "            ent_coef=ENT_COEF,\n";
    code += "            vf_coef=VF_COEF,\n";
    code += "            max_grad_norm=MAX_GRAD_NORM,\n";
    code += "            seed=SEED,\n";
    code += "            verbose=1,\n";
    code += "            tensorboard_log=tb_log\n";
    code += "        )\n";
    code += "    elif ALGORITHM == \"SAC\":\n";
    code += "        model = SAC(\n";
    code += "            \"MlpPolicy\",\n";
    code += "            env,\n";
    code += "            learning_rate=LEARNING_RATE,\n";
    code += "            buffer_size=BUFFER_SIZE,\n";
    code += "            batch_size=BATCH_SIZE,\n";
    code += "            gamma=GAMMA,\n";
    code += "            tau=TAU,\n";
    code += "            ent_coef=\"auto\",\n";
    code += "            seed=SEED,\n";
    code += "            verbose=1,\n";
    code += "            tensorboard_log=tb_log\n";
    code += "        )\n";
    code += "    elif ALGORITHM == \"TD3\":\n";
    code += "        model = TD3(\n";
    code += "            \"MlpPolicy\",\n";
    code += "            env,\n";
    code += "            learning_rate=LEARNING_RATE,\n";
    code += "            buffer_size=BUFFER_SIZE,\n";
    code += "            batch_size=BATCH_SIZE,\n";
    code += "            gamma=GAMMA,\n";
    code += "            tau=TAU,\n";
    code += "            seed=SEED,\n";
    code += "            verbose=1,\n";
    code += "            tensorboard_log=tb_log\n";
    code += "        )\n";
    code += "    elif ALGORITHM == \"DQN\":\n";
    code += "        model = DQN(\n";
    code += "            \"MlpPolicy\",\n";
    code += "            env,\n";
    code += "            learning_rate=LEARNING_RATE,\n";
    code += "            buffer_size=BUFFER_SIZE,\n";
    code += "            batch_size=BATCH_SIZE,\n";
    code += "            gamma=GAMMA,\n";
    code += "            tau=TAU,\n";
    code += "            seed=SEED,\n";
    code += "            verbose=1,\n";
    code += "            tensorboard_log=tb_log\n";
    code += "        )\n";
    code += "    else:\n";
    code += "        raise ValueError(f\"Unknown algorithm: {ALGORITHM}\")\n";
    code += "    \n";
    code += "    # Setup callbacks\n";
    code += "    eval_callback = EvalCallback(\n";
    code += "        eval_env,\n";
    code += "        best_model_save_path=\"./best_model/\",\n";
    code += "        log_path=\"./eval_logs/\",\n";
    code += "        eval_freq=EVAL_FREQ,\n";
    code += "        deterministic=True,\n";
    code += "        render=False\n";
    code += "    )\n";
    code += "    \n";
    code += "    checkpoint_callback = CheckpointCallback(\n";
    code += "        save_freq=SAVE_FREQ,\n";
    code += "        save_path=\"./checkpoints/\",\n";
    code += "        name_prefix=\"rl_model\"\n";
    code += "    )\n";
    code += "    \n";
    code += "    # Train the model\n";
    code += "    print(f\"Training {ALGORITHM} on {ENV_NAME}...\")\n";
    code += "    print(f\"Total timesteps: {TOTAL_TIMESTEPS}\")\n";
    code += "    print(f\"Network: {HIDDEN_SIZES}, Activation: {ACTIVATION}\")\n";
    code += "    model.learn(\n";
    code += "        total_timesteps=TOTAL_TIMESTEPS,\n";
    code += "        callback=[eval_callback, checkpoint_callback],\n";
    code += "        progress_bar=True\n";
    code += "    )\n";
    code += "    \n";
    code += "    # Save the final model\n";
    code += "    model.save(f\"{ALGORITHM}_{ENV_NAME}\")\n";
    code += "    print(f\"Model saved to {ALGORITHM}_{ENV_NAME}.zip\")\n";
    code += "    \n";
    code += "    env.close()\n";
    code += "    eval_env.close()\n";
    code += "    return model\n\n";

    // Evaluation code
    code += "# =============== Evaluation ===============\n";
    code += "def evaluate(model=None, model_path=None, num_episodes=10):\n";
    code += "    \"\"\"Evaluate a trained model.\n";
    code += "    \n";
    code += "    Args:\n";
    code += "        model: A trained model object (from train())\n";
    code += "        model_path: Path to load a saved model from\n";
    code += "        num_episodes: Number of evaluation episodes\n";
    code += "    \"\"\"\n";
    code += "    render_mode = \"human\" if RENDER else None\n";
    code += "    env = gym.make(ENV_NAME, render_mode=render_mode)\n";
    code += "    \n";
    code += "    # Load model from path if provided\n";
    code += "    if model_path:\n";
    code += "        if ALGORITHM == \"PPO\":\n";
    code += "            model = PPO.load(model_path)\n";
    code += "        elif ALGORITHM == \"A2C\":\n";
    code += "            model = A2C.load(model_path)\n";
    code += "        elif ALGORITHM == \"SAC\":\n";
    code += "            model = SAC.load(model_path)\n";
    code += "        elif ALGORITHM == \"TD3\":\n";
    code += "            model = TD3.load(model_path)\n";
    code += "        elif ALGORITHM == \"DQN\":\n";
    code += "            model = DQN.load(model_path)\n";
    code += "    \n";
    code += "    if model is None:\n";
    code += "        raise ValueError(\"Either 'model' or 'model_path' must be provided\")\n";
    code += "    \n";
    code += "    episode_rewards = []\n";
    code += "    for ep in range(num_episodes):\n";
    code += "        obs, info = env.reset()\n";
    code += "        total_reward = 0\n";
    code += "        done = False\n";
    code += "        \n";
    code += "        while not done:\n";
    code += "            action, _ = model.predict(obs, deterministic=True)\n";
    code += "            obs, reward, terminated, truncated, info = env.step(action)\n";
    code += "            total_reward += reward\n";
    code += "            done = terminated or truncated\n";
    code += "        \n";
    code += "        episode_rewards.append(total_reward)\n";
    code += "        print(f\"Episode {ep+1}: Reward = {total_reward:.2f}\")\n";
    code += "    \n";
    code += "    env.close()\n";
    code += "    print(f\"\\nMean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}\")\n";
    code += "    return episode_rewards\n\n";

    // Main entry point
    code += "# =============== Main ===============\n";
    code += "if __name__ == \"__main__\":\n";
    code += "    import argparse\n";
    code += "    parser = argparse.ArgumentParser()\n";
    code += "    parser.add_argument(\"--train\", action=\"store_true\", help=\"Train the model\")\n";
    code += "    parser.add_argument(\"--eval\", type=str, default=None, help=\"Evaluate model from path\")\n";
    code += "    parser.add_argument(\"--episodes\", type=int, default=10, help=\"Number of evaluation episodes\")\n";
    code += "    args = parser.parse_args()\n";
    code += "    \n";
    code += "    if args.train:\n";
    code += "        model = train()\n";
    code += "        evaluate(model=model, num_episodes=args.episodes)\n";
    code += "    elif args.eval:\n";
    code += "        evaluate(model_path=args.eval, num_episodes=args.episodes)\n";
    code += "    else:\n";
    code += "        # Default: train and evaluate\n";
    code += "        model = train()\n";
    code += "        evaluate(model=model, num_episodes=args.episodes)\n";

    return code;
}

std::string NodeEditor::GeneratePyTorchCode(const std::vector<int>& sorted_ids) {
    // Check if this is an RL graph
    if (IsRLGraph(sorted_ids)) {
        return GenerateRLPyTorchCode(sorted_ids);
    }

    std::string code;

    // Header
    code += "# Auto-generated PyTorch model from CyxWiz Node Editor\n";
    code += "# Generated at: " + std::string(__DATE__) + " " + std::string(__TIME__) + "\n\n";
    code += "import torch\n";
    code += "import torch.nn as nn\n";
    code += "import torch.nn.functional as F\n";
    code += "import torch.optim as optim\n";
    code += "import numpy as np\n\n";

    // Model class
    code += "class GeneratedModel(nn.Module):\n";
    code += "    def __init__(self):\n";
    code += "        super(GeneratedModel, self).__init__()\n";

    // Generate layer definitions
    int layer_idx = 0;
    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;

        // Skip DatasetInput and Output nodes in __init__ (they don't have layers)
        if (node->type == NodeType::DatasetInput || node->type == NodeType::Output) {
            continue;
        }

        std::string layer_code = NodeTypeToPythonLayer(*node);
        if (!layer_code.empty()) {
            code += "        self.layer" + std::to_string(layer_idx) + " = " + layer_code + "\n";
            layer_idx++;
        }
    }

    code += "\n";

    // Forward pass
    code += "    def forward(self, x):\n";
    layer_idx = 0;
    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;

        switch (node->type) {
            case NodeType::DatasetInput:
                code += "        # Dataset input layer (x is already the input)\n";
                break;

            case NodeType::Dense:
                code += "        x = self.layer" + std::to_string(layer_idx++) + "(x)\n";
                break;

            case NodeType::ReLU:
                code += "        x = F.relu(x)\n";
                break;

            case NodeType::Sigmoid:
                code += "        x = torch.sigmoid(x)\n";
                break;

            case NodeType::Tanh:
                code += "        x = torch.tanh(x)\n";
                break;

            case NodeType::Softmax:
                code += "        x = F.softmax(x, dim=1)\n";
                break;

            case NodeType::Dropout:
                code += "        x = F.dropout(x, p=0.5, training=self.training)\n";
                break;

            case NodeType::Flatten:
                code += "        x = torch.flatten(x, 1)\n";
                break;

            case NodeType::Output:
                code += "        # Output layer\n";
                break;

            // ===== Transformer Layers =====
            case NodeType::TransformerEncoder:
                code += "        x = self.layer" + std::to_string(layer_idx++) + "(x)\n";
                break;

            case NodeType::TransformerDecoder:
                code += "        x = self.layer" + std::to_string(layer_idx++) + "(x, memory)  # memory from encoder\n";
                break;

            case NodeType::PositionalEncoding:
                code += "        x = self.layer" + std::to_string(layer_idx++) + "(x)\n";
                break;

            // ===== RNN Layers =====
            case NodeType::LSTM:
                code += "        x, (h_n, c_n) = self.layer" + std::to_string(layer_idx++) + "(x)\n";
                break;

            case NodeType::GRU:
            case NodeType::RNN:
                code += "        x, h_n = self.layer" + std::to_string(layer_idx++) + "(x)\n";
                break;

            // ===== Additional Activations =====
            case NodeType::GELU:
                code += "        x = F.gelu(x)\n";
                break;

            case NodeType::LeakyReLU:
                code += "        x = F.leaky_relu(x, 0.01)\n";
                break;

            case NodeType::Swish:
                code += "        x = F.silu(x)\n";
                break;

            case NodeType::Mish:
                code += "        x = F.mish(x)\n";
                break;

            // ===== Normalization Layers =====
            case NodeType::BatchNorm:
            case NodeType::LayerNorm:
            case NodeType::GroupNorm:
            case NodeType::InstanceNorm:
                code += "        x = self.layer" + std::to_string(layer_idx++) + "(x)\n";
                break;

            // ===== Embedding =====
            case NodeType::Embedding:
                code += "        x = self.layer" + std::to_string(layer_idx++) + "(x)\n";
                break;

            // ===== Attention =====
            case NodeType::MultiHeadAttention:
                code += "        x, _ = self.layer" + std::to_string(layer_idx++) + "(x, x, x)  # self-attention\n";
                break;

            case NodeType::LinearAttention:
                code += "        x = self.layer" + std::to_string(layer_idx++) + "(x)\n";
                break;

            // ===== Signal / Control Nodes =====
            case NodeType::Constant: {
                std::string val = "0.0";
                auto it = node->parameters.find("value");
                if (it != node->parameters.end() && !it->second.empty()) val = it->second;
                code += "        signal_" + std::to_string(node->id) + " = " + val + "\n";
                break;
            }
            case NodeType::SignalSlider: {
                std::string val = "0.0";
                auto it = node->parameters.find("value");
                if (it != node->parameters.end() && !it->second.empty()) val = it->second;
                std::string mn = "-1.0", mx = "1.0";
                it = node->parameters.find("min");
                if (it != node->parameters.end() && !it->second.empty()) mn = it->second;
                it = node->parameters.find("max");
                if (it != node->parameters.end() && !it->second.empty()) mx = it->second;
                code += "        signal_" + std::to_string(node->id) + " = " + val + "  # slider [" + mn + ", " + mx + "]\n";
                break;
            }
            case NodeType::SineWave: {
                std::string amp = "1.0", freq = "1.0", phase = "0.0", offset = "0.0";
                auto it = node->parameters.find("amplitude");
                if (it != node->parameters.end() && !it->second.empty()) amp = it->second;
                it = node->parameters.find("frequency");
                if (it != node->parameters.end() && !it->second.empty()) freq = it->second;
                it = node->parameters.find("phase");
                if (it != node->parameters.end() && !it->second.empty()) phase = it->second;
                it = node->parameters.find("offset");
                if (it != node->parameters.end() && !it->second.empty()) offset = it->second;
                code += "        signal_" + std::to_string(node->id) + " = " + amp + " * np.sin(2 * np.pi * " + freq + " * t + " + phase + ") + " + offset + "\n";
                break;
            }
            case NodeType::StepSignal: {
                std::string step_t = "0.5", init = "0.0", final_v = "1.0";
                auto it = node->parameters.find("step_time");
                if (it != node->parameters.end() && !it->second.empty()) step_t = it->second;
                it = node->parameters.find("initial_value");
                if (it != node->parameters.end() && !it->second.empty()) init = it->second;
                it = node->parameters.find("final_value");
                if (it != node->parameters.end() && !it->second.empty()) final_v = it->second;
                code += "        signal_" + std::to_string(node->id) + " = " + final_v + " if t >= " + step_t + " else " + init + "\n";
                break;
            }
            case NodeType::RampSignal: {
                std::string start = "0.0", end = "1.0", dur = "5.0";
                auto it = node->parameters.find("start_value");
                if (it != node->parameters.end() && !it->second.empty()) start = it->second;
                it = node->parameters.find("end_value");
                if (it != node->parameters.end() && !it->second.empty()) end = it->second;
                it = node->parameters.find("duration");
                if (it != node->parameters.end() && !it->second.empty()) dur = it->second;
                code += "        signal_" + std::to_string(node->id) + " = np.clip(" + start + " + (" + end + " - " + start + ") * t / " + dur + ", " + start + ", " + end + ")\n";
                break;
            }
            case NodeType::SignalScope:
                code += "        # Scope: visualize connected signal\n";
                break;

            // ===== Reinforcement Learning Nodes =====
            case NodeType::GymEnvironment:
            case NodeType::ReplayBufferNode:
            case NodeType::PolicyNetwork:
            case NodeType::ValueNetwork:
            case NodeType::RLTraining:
                // RL nodes handled separately - skip in supervised forward pass
                break;

            default:
                break;
        }
    }
    code += "        return x\n\n";

    // Training code
    code += "# Training setup\n";
    code += "if __name__ == '__main__':\n";
    code += "    # Create model\n";
    code += "    model = GeneratedModel()\n";
    code += "    print(model)\n\n";

    code += "    # Loss and optimizer\n";
    code += "    criterion = nn.CrossEntropyLoss()\n";
    code += "    optimizer = optim.Adam(model.parameters(), lr=0.001)\n\n";

    code += "    # TODO: Add your training data here\n";
    code += "    # Example training loop:\n";
    code += "    # for epoch in range(num_epochs):\n";
    code += "    #     for batch_idx, (data, target) in enumerate(train_loader):\n";
    code += "    #         optimizer.zero_grad()\n";
    code += "    #         output = model(data)\n";
    code += "    #         loss = criterion(output, target)\n";
    code += "    #         loss.backward()\n";
    code += "    #         optimizer.step()\n";

    return code;
}

std::string NodeEditor::GenerateTensorFlowCode(const std::vector<int>& sorted_ids) {
    // Check if this is an RL graph - redirect to PyTorch RL code (SB3 is the standard)
    if (IsRLGraph(sorted_ids)) {
        std::string code;
        code += "# RL training is best supported with Stable-Baselines3 (PyTorch-based)\n";
        code += "# Please select PyTorch framework for RL code generation.\n";
        code += "# Alternatively, use TF-Agents for TensorFlow-based RL:\n\n";
        code += "# pip install tf-agents\n";
        code += "# See: https://www.tensorflow.org/agents\n\n";
        code += GenerateRLPyTorchCode(sorted_ids);  // Fallback to SB3 code
        return code;
    }

    std::string code;

    // Header
    code += "# Auto-generated TensorFlow model from CyxWiz Node Editor\n";
    code += "# Generated at: " + std::string(__DATE__) + " " + std::string(__TIME__) + "\n\n";
    code += "import tensorflow as tf\n";
    code += "from tensorflow.keras import layers, models, optimizers\n\n";

    // Model class using tf.keras
    code += "class GeneratedModel(tf.keras.Model):\n";
    code += "    def __init__(self):\n";
    code += "        super(GeneratedModel, self).__init__()\n";

    // Generate layer definitions
    int layer_idx = 0;
    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;

        // Skip DatasetInput and Output nodes in __init__
        if (node->type == NodeType::DatasetInput || node->type == NodeType::Output) {
            continue;
        }

        std::string layer_code = NodeTypeToTensorFlowLayer(*node, layer_idx);
        if (!layer_code.empty()) {
            code += "        self.layer" + std::to_string(layer_idx) + " = " + layer_code + "\n";
            layer_idx++;
        }
    }

    code += "\n";

    // Call method (forward pass in TensorFlow)
    code += "    def call(self, x, training=False):\n";
    layer_idx = 0;
    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;

        switch (node->type) {
            case NodeType::DatasetInput:
                code += "        # Dataset input layer (x is already the input)\n";
                break;

            case NodeType::Dense:
                code += "        x = self.layer" + std::to_string(layer_idx++) + "(x)\n";
                break;

            case NodeType::ReLU:
                code += "        x = tf.nn.relu(x)\n";
                break;

            case NodeType::Sigmoid:
                code += "        x = tf.nn.sigmoid(x)\n";
                break;

            case NodeType::Tanh:
                code += "        x = tf.nn.tanh(x)\n";
                break;

            case NodeType::Softmax:
                code += "        x = tf.nn.softmax(x)\n";
                break;

            case NodeType::Dropout:
                code += "        x = tf.keras.layers.Dropout(0.5)(x, training=training)\n";
                break;

            case NodeType::Flatten:
                code += "        x = tf.keras.layers.Flatten()(x)\n";
                break;

            case NodeType::Output:
                code += "        # Output layer\n";
                break;

            default:
                break;
        }
    }
    code += "        return x\n\n";

    // Training code
    code += "# Training setup\n";
    code += "if __name__ == '__main__':\n";
    code += "    # Create model\n";
    code += "    model = GeneratedModel()\n";
    code += "    model.build(input_shape=(None, 784))  # Adjust input shape as needed\n";
    code += "    model.summary()\n\n";

    code += "    # Compile model\n";
    code += "    model.compile(\n";
    code += "        optimizer='adam',\n";
    code += "        loss='sparse_categorical_crossentropy',\n";
    code += "        metrics=['accuracy']\n";
    code += "    )\n\n";

    code += "    # TODO: Add your training data here\n";
    code += "    # Example training:\n";
    code += "    # model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)\n";

    return code;
}

std::string NodeEditor::GenerateKerasCode(const std::vector<int>& sorted_ids) {
    // Check if this is an RL graph - redirect to PyTorch RL code (SB3 is the standard)
    if (IsRLGraph(sorted_ids)) {
        std::string code;
        code += "# RL training is best supported with Stable-Baselines3 (PyTorch-based)\n";
        code += "# Please select PyTorch framework for RL code generation.\n\n";
        code += GenerateRLPyTorchCode(sorted_ids);  // Fallback to SB3 code
        return code;
    }

    std::string code;

    // Header
    code += "# Auto-generated Keras model from CyxWiz Node Editor\n";
    code += "# Generated at: " + std::string(__DATE__) + " " + std::string(__TIME__) + "\n\n";
    code += "from tensorflow import keras\n";
    code += "from tensorflow.keras import layers\n\n";

    // Sequential model approach
    code += "# Build model using Sequential API\n";
    code += "model = keras.Sequential([\n";

    bool first_layer = true;
    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;

        // Skip DatasetInput node
        if (node->type == NodeType::DatasetInput) {
            continue;
        }

        std::string layer_code = NodeTypeToKerasLayer(*node);
        if (!layer_code.empty()) {
            if (!first_layer) {
                code += ",\n";
            }
            code += "    " + layer_code;
            first_layer = false;
        }
    }

    code += "\n])\n\n";

    // Model summary and compilation
    code += "# Model configuration\n";
    code += "model.build(input_shape=(None, 784))  # Adjust input shape as needed\n";
    code += "model.summary()\n\n";

    code += "# Compile model\n";
    code += "model.compile(\n";
    code += "    optimizer='adam',\n";
    code += "    loss='sparse_categorical_crossentropy',\n";
    code += "    metrics=['accuracy']\n";
    code += ")\n\n";

    code += "# TODO: Add your training data here\n";
    code += "# Example training:\n";
    code += "# history = model.fit(\n";
    code += "#     x_train, y_train,\n";
    code += "#     epochs=10,\n";
    code += "#     batch_size=32,\n";
    code += "#     validation_split=0.2\n";
    code += "# )\n";

    return code;
}

std::string NodeEditor::GeneratePyCyxWizCode(const std::vector<int>& sorted_ids) {
    // Check if this is an RL graph
    if (IsRLGraph(sorted_ids)) {
        return GenerateRLPyCyxWizCode(sorted_ids);
    }

    std::string code;

    // Header
    code += "# Auto-generated PyCyxWiz model from CyxWiz Node Editor\n";
    code += "# Generated at: " + std::string(__DATE__) + " " + std::string(__TIME__) + "\n\n";
    code += "import pycyxwiz as cx\n";
    code += "import numpy as np\n\n";

    // Model class using pycyxwiz
    code += "class GeneratedModel:\n";
    code += "    def __init__(self):\n";

    // Generate layer definitions
    int layer_idx = 0;
    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;

        // Skip DatasetInput and Output nodes in __init__
        if (node->type == NodeType::DatasetInput || node->type == NodeType::Output) {
            continue;
        }

        std::string layer_code = NodeTypeToPyCyxWizLayer(*node);
        if (!layer_code.empty()) {
            code += "        self.layer" + std::to_string(layer_idx) + " = " + layer_code + "\n";
            layer_idx++;
        }
    }

    code += "\n";

    // Forward method
    code += "    def forward(self, x):\n";
    layer_idx = 0;
    std::map<int, std::string> node_outputs;
    std::map<int, std::string> pin_outputs;
    std::string last_expr = "x";

    auto node_var = [](int node_id) {
        return "v" + std::to_string(node_id);
    };

    auto input_expr = [&](const MLNode& node, size_t input_index) {
        if (input_index >= node.inputs.size()) {
            return last_expr;
        }

        const int target_pin = node.inputs[input_index].id;
        for (const auto& link : links_) {
            if (link.to_node == node.id && link.to_pin == target_pin) {
                auto pin_it = pin_outputs.find(link.from_pin);
                if (pin_it != pin_outputs.end()) {
                    return pin_it->second;
                }
                auto it = node_outputs.find(link.from_node);
                if (it != node_outputs.end()) {
                    return it->second;
                }
                return std::string("x");
            }
        }

        return last_expr;
    };

    auto all_input_exprs = [&](const MLNode& node) {
        std::vector<std::string> inputs;
        for (size_t i = 0; i < node.inputs.size(); ++i) {
            const int target_pin = node.inputs[i].id;
            for (const auto& link : links_) {
                if (link.to_node == node.id && link.to_pin == target_pin) {
                    auto pin_it = pin_outputs.find(link.from_pin);
                    if (pin_it != pin_outputs.end()) {
                        inputs.push_back(pin_it->second);
                        continue;
                    }
                    auto it = node_outputs.find(link.from_node);
                    inputs.push_back(it != node_outputs.end() ? it->second : "x");
                }
            }
        }
        if (inputs.empty()) {
            inputs.push_back(last_expr);
        }
        return inputs;
    };

    auto python_list = [](const std::vector<std::string>& values) {
        std::string out = "[";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i > 0) out += ", ";
            out += values[i];
        }
        out += "]";
        return out;
    };

    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;

        std::string out = node_var(node->id);
        auto record_output = [&](const std::string& expr) {
            node_outputs[node->id] = expr;
            for (const auto& pin : node->outputs) {
                pin_outputs[pin.id] = expr;
            }
            last_expr = expr;
        };

        switch (node->type) {
            case NodeType::DatasetInput:
                code += "        " + out + " = x\n";
                record_output(out);
                break;

            case NodeType::Dense:
                code += "        " + out + " = self.layer" + std::to_string(layer_idx++) + ".forward(" + input_expr(*node, 0) + ")\n";
                record_output(out);
                break;

            case NodeType::ReLU:
                code += "        " + out + " = cx.relu(" + input_expr(*node, 0) + ")\n";
                record_output(out);
                break;

            case NodeType::Sigmoid:
                code += "        " + out + " = cx.sigmoid(" + input_expr(*node, 0) + ")\n";
                record_output(out);
                break;

            case NodeType::Tanh:
                code += "        " + out + " = cx.tanh(" + input_expr(*node, 0) + ")\n";
                record_output(out);
                break;

            case NodeType::Softmax:
                code += "        " + out + " = cx.softmax(" + input_expr(*node, 0) + ")\n";
                record_output(out);
                break;

            case NodeType::Dropout:
                code += "        " + out + " = cx.dropout(" + input_expr(*node, 0) + ", p=0.5)\n";
                record_output(out);
                break;

            case NodeType::Flatten:
                code += "        " + out + " = " + input_expr(*node, 0) + ".flatten()\n";
                record_output(out);
                break;

            case NodeType::Reshape:
            case NodeType::View: {
                std::string shape = CsvToPythonListLiteral(GetParamOrDefault(*node, "shape", "-1"), "-1");
                code += "        " + out + " = " + input_expr(*node, 0) + ".view(" + shape + ")\n";
                record_output(out);
                break;
            }

            case NodeType::Permute: {
                std::string dims = CsvToPythonListLiteral(GetParamOrDefault(*node, "dims", "0,2,1"), "0,2,1");
                code += "        " + out + " = " + input_expr(*node, 0) + ".permute(" + dims + ")\n";
                record_output(out);
                break;
            }

            case NodeType::Squeeze: {
                std::string dim = GetParamOrDefault(*node, "dim", "-1");
                if (dim == "-1") {
                    code += "        " + out + " = " + input_expr(*node, 0) + ".squeeze()\n";
                } else {
                    code += "        " + out + " = " + input_expr(*node, 0) + ".squeeze(" + dim + ")\n";
                }
                record_output(out);
                break;
            }

            case NodeType::Unsqueeze: {
                std::string dim = GetParamOrDefault(*node, "dim", "0");
                code += "        " + out + " = " + input_expr(*node, 0) + ".unsqueeze(" + dim + ")\n";
                record_output(out);
                break;
            }

            case NodeType::Split: {
                std::string split_size = GetParamOrDefault(*node, "split_size", "2");
                std::string dim = GetParamOrDefault(*node, "dim", "0");
                code += "        " + out + " = " + input_expr(*node, 0) + ".split(" + split_size + ", " + dim + ")\n";
                node_outputs[node->id] = out + "[0]";
                for (size_t i = 0; i < node->outputs.size(); ++i) {
                    pin_outputs[node->outputs[i].id] = out + "[" + std::to_string(i) + "]";
                }
                last_expr = out + "[0]";
                break;
            }

            case NodeType::Concatenate: {
                std::string dim = GetParamOrDefault(*node, "dim", "1");
                code += "        " + out + " = cx.Tensor.cat(" + python_list(all_input_exprs(*node)) + ", " + dim + ")\n";
                record_output(out);
                break;
            }

            case NodeType::Add: {
                auto inputs = all_input_exprs(*node);
                code += "        " + out + " = " + inputs[0];
                for (size_t i = 1; i < inputs.size(); ++i) {
                    code += " + " + inputs[i];
                }
                code += "\n";
                record_output(out);
                break;
            }

            case NodeType::Multiply: {
                auto inputs = all_input_exprs(*node);
                code += "        " + out + " = " + inputs[0];
                for (size_t i = 1; i < inputs.size(); ++i) {
                    code += " * " + inputs[i];
                }
                code += "\n";
                record_output(out);
                break;
            }

            case NodeType::Average: {
                auto inputs = all_input_exprs(*node);
                code += "        " + out + " = (" + inputs[0];
                for (size_t i = 1; i < inputs.size(); ++i) {
                    code += " + " + inputs[i];
                }
                code += ") / " + std::to_string(inputs.size()) + ".0\n";
                record_output(out);
                break;
            }

            case NodeType::TensorSum:
            case NodeType::TensorMean:
            case NodeType::TensorMax:
            case NodeType::TensorMin:
            case NodeType::TensorProd:
            case NodeType::TensorVar:
            case NodeType::TensorStd: {
                std::string method;
                switch (node->type) {
                    case NodeType::TensorSum: method = "sum"; break;
                    case NodeType::TensorMean: method = "mean"; break;
                    case NodeType::TensorMax: method = "max"; break;
                    case NodeType::TensorMin: method = "min"; break;
                    case NodeType::TensorProd: method = "prod"; break;
                    case NodeType::TensorVar: method = "var"; break;
                    case NodeType::TensorStd: method = "std"; break;
                    default: break;
                }

                std::string dim = GetParamOrDefault(*node, "dim", "-1");
                std::string keepdim = PythonBoolLiteral(GetParamOrDefault(*node, "keepdim", "false"));
                code += "        " + out + " = " + input_expr(*node, 0) + "." + method + "(";
                if (dim == "-1") {
                    code += "keepdim=" + keepdim;
                } else {
                    code += dim + ", keepdim=" + keepdim;
                }
                code += ")\n";
                record_output(out);
                break;
            }

            case NodeType::TensorBroadcastTo:
            case NodeType::TensorExpand: {
                std::string shape = CsvToPythonListLiteral(GetParamOrDefault(*node, "shape", "-1"), "-1");
                std::string method = node->type == NodeType::TensorBroadcastTo ? "broadcast_to" : "expand";
                code += "        " + out + " = " + input_expr(*node, 0) + "." + method + "(" + shape + ")\n";
                record_output(out);
                break;
            }

            case NodeType::TensorIndexSelect: {
                std::string dim = GetParamOrDefault(*node, "dim", "0");
                std::string indices = CsvToPythonListLiteral(GetParamOrDefault(*node, "indices", "0"), "0");
                code += "        " + out + " = " + input_expr(*node, 0) + ".index_select(" + dim + ", " + indices + ")\n";
                record_output(out);
                break;
            }

            case NodeType::TensorPow: {
                std::string exponent = GetParamOrDefault(*node, "exponent", "2.0");
                code += "        " + out + " = " + input_expr(*node, 0) + ".pow(" + exponent + ")\n";
                record_output(out);
                break;
            }

            case NodeType::TensorSqrt:
            case NodeType::TensorExp:
            case NodeType::TensorLog:
            case NodeType::TensorAbs:
            case NodeType::TensorSign: {
                std::string method;
                switch (node->type) {
                    case NodeType::TensorSqrt: method = "sqrt"; break;
                    case NodeType::TensorExp: method = "exp"; break;
                    case NodeType::TensorLog: method = "log"; break;
                    case NodeType::TensorAbs: method = "abs"; break;
                    case NodeType::TensorSign: method = "sign"; break;
                    default: break;
                }
                code += "        " + out + " = " + input_expr(*node, 0) + "." + method + "()\n";
                record_output(out);
                break;
            }

            case NodeType::TensorClip: {
                std::string min_value = GetParamOrDefault(*node, "min", "0.0");
                std::string max_value = GetParamOrDefault(*node, "max", "1.0");
                code += "        " + out + " = " + input_expr(*node, 0) + ".clip(" + min_value + ", " + max_value + ")\n";
                record_output(out);
                break;
            }

            case NodeType::TensorDot:
            case NodeType::TensorBatchMatMul: {
                auto inputs = all_input_exprs(*node);
                std::string rhs = inputs.size() > 1 ? inputs[1] : input_expr(*node, 0);
                std::string method = node->type == NodeType::TensorDot ? "dot" : "batch_matmul";
                code += "        " + out + " = " + input_expr(*node, 0) + "." + method + "(" + rhs + ")\n";
                record_output(out);
                break;
            }

            case NodeType::TensorCompare: {
                std::string rhs = GetParamOrDefault(*node, "scalar", "0.0");
                std::string op = GetParamOrDefault(*node, "op", ">");
                code += "        " + out + " = " + input_expr(*node, 0) + " " + op + " " + rhs + "\n";
                record_output(out);
                break;
            }

            case NodeType::TensorLogicalMask: {
                code += "        " + out + " = ~" + input_expr(*node, 0) + "\n";
                record_output(out);
                break;
            }

            case NodeType::Output:
                last_expr = input_expr(*node, 0);
                break;

            default:
                break;
        }
    }
    code += "        return " + last_expr + "\n\n";

    code += "    def train(self, x_train, y_train, epochs=10, learning_rate=0.001):\n";
    code += "        \"\"\"Training loop using CyxWiz backend\"\"\"\n";
    code += "        optimizer = cx.Adam(learning_rate=learning_rate)\n";
    code += "        loss_fn = cx.CrossEntropyLoss()\n\n";
    code += "        for epoch in range(epochs):\n";
    code += "            # Forward pass\n";
    code += "            predictions = self.forward(x_train)\n";
    code += "            loss = loss_fn(predictions, y_train)\n\n";
    code += "            # Backward pass\n";
    code += "            loss.backward()\n";
    code += "            optimizer.step()\n";
    code += "            optimizer.zero_grad()\n\n";
    code += "            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')\n\n";

    // Training setup
    code += "# Training setup\n";
    code += "if __name__ == '__main__':\n";
    code += "    # Initialize CyxWiz backend\n";
    code += "    cx.initialize()\n\n";
    code += "    # Select device (GPU if available)\n";
    code += "    device = cx.get_device(cx.DeviceType.CUDA if cx.cuda_available() else cx.DeviceType.CPU)\n";
    code += "    cx.set_device(device)\n";
    code += "    print(f'Using device: {device.name()}')\n\n";

    code += "    # Create model\n";
    code += "    model = GeneratedModel()\n\n";

    code += "    # TODO: Load your training data here\n";
    code += "    # x_train = cx.Tensor(your_data)\n";
    code += "    # y_train = cx.Tensor(your_labels)\n";
    code += "    # model.train(x_train, y_train, epochs=10)\n";

    return code;
}

std::string NodeEditor::NodeTypeToPythonLayer(const MLNode& node) {
    std::string code;

    switch (node.type) {
        case NodeType::Dense: {
            std::string units = "128";
            auto it = node.parameters.find("units");
            if (it != node.parameters.end()) {
                units = it->second;
            }
            // Note: input size needs to be determined from graph connections
            code = "nn.Linear(in_features=AUTO, out_features=" + units + ")";
            break;
        }

        case NodeType::Conv2D:
            code = "nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)";
            break;

        case NodeType::MaxPool2D:
            code = "nn.MaxPool2d(kernel_size=2)";
            break;

        case NodeType::BatchNorm:
            code = "nn.BatchNorm2d(num_features=AUTO)";
            break;

        case NodeType::Dropout: {
            code = "nn.Dropout(p=0.5)";
            break;
        }

        case NodeType::LinearAttention: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            std::string feature_map = "elu";
            std::string eps = "1e-6";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            it = node.parameters.find("feature_map");
            if (it != node.parameters.end()) feature_map = it->second;
            it = node.parameters.find("eps");
            if (it != node.parameters.end()) eps = it->second;
            // Linear attention with O(n) complexity (Performer-style)
            // Requires: pip install performer-pytorch or custom implementation
            code = "LinearAttention(dim=" + embed_dim + ", heads=" + num_heads +
                   ", dim_head=" + embed_dim + "//" + num_heads +
                   ", feature_map='" + feature_map + "', eps=" + eps + ")";
            break;
        }

        case NodeType::MultiHeadAttention: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            code = "nn.MultiheadAttention(embed_dim=" + embed_dim + ", num_heads=" + num_heads + ")";
            break;
        }

        case NodeType::LayerNorm: {
            std::string normalized_shape = "512";
            auto it = node.parameters.find("normalized_shape");
            if (it != node.parameters.end()) normalized_shape = it->second;
            code = "nn.LayerNorm(" + normalized_shape + ")";
            break;
        }

        case NodeType::Embedding: {
            std::string num_embeddings = "10000";
            std::string embedding_dim = "512";
            auto it = node.parameters.find("num_embeddings");
            if (it != node.parameters.end()) num_embeddings = it->second;
            it = node.parameters.find("embedding_dim");
            if (it != node.parameters.end()) embedding_dim = it->second;
            code = "nn.Embedding(num_embeddings=" + num_embeddings + ", embedding_dim=" + embedding_dim + ")";
            break;
        }

        case NodeType::GELU:
            code = "nn.GELU()";
            break;

        case NodeType::ReLU:
            code = "nn.ReLU()";
            break;

        case NodeType::LeakyReLU: {
            std::string negative_slope = "0.01";
            auto it = node.parameters.find("negative_slope");
            if (it != node.parameters.end()) negative_slope = it->second;
            code = "nn.LeakyReLU(negative_slope=" + negative_slope + ")";
            break;
        }

        case NodeType::Swish:
            code = "nn.SiLU()";
            break;

        case NodeType::Mish:
            code = "nn.Mish()";
            break;

        // ===== Transformer Layers =====
        case NodeType::TransformerEncoder: {
            std::string d_model = "512";
            std::string nhead = "8";
            std::string dim_feedforward = "2048";
            std::string dropout = "0.1";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) d_model = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) nhead = it->second;
            it = node.parameters.find("ff_dim");
            if (it != node.parameters.end()) dim_feedforward = it->second;
            it = node.parameters.find("dropout");
            if (it != node.parameters.end()) dropout = it->second;
            code = "nn.TransformerEncoderLayer(d_model=" + d_model + ", nhead=" + nhead +
                   ", dim_feedforward=" + dim_feedforward + ", dropout=" + dropout + ", batch_first=True)";
            break;
        }

        case NodeType::TransformerDecoder: {
            std::string d_model = "512";
            std::string nhead = "8";
            std::string dim_feedforward = "2048";
            std::string dropout = "0.1";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) d_model = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) nhead = it->second;
            it = node.parameters.find("ff_dim");
            if (it != node.parameters.end()) dim_feedforward = it->second;
            it = node.parameters.find("dropout");
            if (it != node.parameters.end()) dropout = it->second;
            code = "nn.TransformerDecoderLayer(d_model=" + d_model + ", nhead=" + nhead +
                   ", dim_feedforward=" + dim_feedforward + ", dropout=" + dropout + ", batch_first=True)";
            break;
        }

        case NodeType::PositionalEncoding: {
            std::string d_model = "512";
            std::string max_len = "5000";
            auto it = node.parameters.find("d_model");
            if (it != node.parameters.end()) d_model = it->second;
            it = node.parameters.find("max_len");
            if (it != node.parameters.end()) max_len = it->second;
            code = "PositionalEncoding(d_model=" + d_model + ", max_len=" + max_len + ")";
            break;
        }

        // ===== RNN Layers =====
        case NodeType::LSTM: {
            std::string input_size = "512";
            std::string hidden_size = "256";
            std::string num_layers = "1";
            std::string bidirectional = "False";
            std::string dropout = "0.0";
            auto normalize_bool = [](const std::string& value) -> std::string {
                if (value == "true" || value == "1" || value == "True") {
                    return "True";
                }
                if (value == "false" || value == "0" || value == "False") {
                    return "False";
                }
                return value;
            };
            auto it = node.parameters.find("input_size");
            if (it != node.parameters.end()) input_size = it->second;
            it = node.parameters.find("hidden_size");
            if (it != node.parameters.end()) hidden_size = it->second;
            it = node.parameters.find("num_layers");
            if (it != node.parameters.end()) num_layers = it->second;
            it = node.parameters.find("bidirectional");
            if (it != node.parameters.end()) bidirectional = normalize_bool(it->second);
            it = node.parameters.find("dropout");
            if (it != node.parameters.end()) dropout = it->second;
            code = "nn.LSTM(input_size=" + input_size + ", hidden_size=" + hidden_size +
                   ", num_layers=" + num_layers + ", batch_first=True, bidirectional=" + bidirectional +
                   ", dropout=" + dropout + ")";
            break;
        }

        case NodeType::GRU: {
            std::string input_size = "512";
            std::string hidden_size = "256";
            std::string num_layers = "1";
            std::string bidirectional = "False";
            std::string dropout = "0.0";
            auto normalize_bool = [](const std::string& value) -> std::string {
                if (value == "true" || value == "1" || value == "True") {
                    return "True";
                }
                if (value == "false" || value == "0" || value == "False") {
                    return "False";
                }
                return value;
            };
            auto it = node.parameters.find("input_size");
            if (it != node.parameters.end()) input_size = it->second;
            it = node.parameters.find("hidden_size");
            if (it != node.parameters.end()) hidden_size = it->second;
            it = node.parameters.find("num_layers");
            if (it != node.parameters.end()) num_layers = it->second;
            it = node.parameters.find("bidirectional");
            if (it != node.parameters.end()) bidirectional = normalize_bool(it->second);
            it = node.parameters.find("dropout");
            if (it != node.parameters.end()) dropout = it->second;
            code = "nn.GRU(input_size=" + input_size + ", hidden_size=" + hidden_size +
                   ", num_layers=" + num_layers + ", batch_first=True, bidirectional=" + bidirectional +
                   ", dropout=" + dropout + ")";
            break;
        }

        case NodeType::RNN: {
            std::string input_size = "512";
            std::string hidden_size = "256";
            std::string num_layers = "1";
            std::string nonlinearity = "tanh";
            auto it = node.parameters.find("input_size");
            if (it != node.parameters.end()) input_size = it->second;
            it = node.parameters.find("hidden_size");
            if (it != node.parameters.end()) hidden_size = it->second;
            it = node.parameters.find("num_layers");
            if (it != node.parameters.end()) num_layers = it->second;
            it = node.parameters.find("nonlinearity");
            if (it != node.parameters.end()) nonlinearity = it->second;
            code = "nn.RNN(input_size=" + input_size + ", hidden_size=" + hidden_size +
                   ", num_layers=" + num_layers + ", nonlinearity='" + nonlinearity + "', batch_first=True)";
            break;
        }

        case NodeType::PluginCustom: {
            auto it = node.parameters.find("plugin_qualified_name");
            if (it != node.parameters.end())
                code = cyxwiz::plugin::PluginNodeRegistry::Instance().GenerateCode(it->second, node.parameters, "pytorch");
            break;
        }

        default:
            // Activation functions and others don't need layers in __init__
            code = "";
            break;
    }

    return code;
}

std::string NodeEditor::NodeTypeToTensorFlowLayer(const MLNode& node, int /*layer_idx*/) {
    std::string code;

    switch (node.type) {
        case NodeType::Dense: {
            std::string units = "128";
            auto it = node.parameters.find("units");
            if (it != node.parameters.end()) {
                units = it->second;
            }
            code = "layers.Dense(" + units + ")";
            break;
        }

        case NodeType::Conv2D:
            code = "layers.Conv2D(32, kernel_size=3)";
            break;

        case NodeType::MaxPool2D:
            code = "layers.MaxPool2D(pool_size=2)";
            break;

        case NodeType::BatchNorm:
            code = "layers.BatchNormalization()";
            break;

        case NodeType::Dropout:
            code = "layers.Dropout(0.5)";
            break;

        case NodeType::LinearAttention: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            // TensorFlow doesn't have native linear attention - use MultiHeadAttention or custom layer
            // Comment indicates O(n) linear attention should be used
            code = "# LinearAttention (O(n)) - requires tensorflow-addons or custom impl\n"
                   "        layers.MultiHeadAttention(key_dim=" + embed_dim + "//" + num_heads +
                   ", num_heads=" + num_heads + ")  # Replace with linear attention";
            break;
        }

        case NodeType::MultiHeadAttention: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            code = "layers.MultiHeadAttention(key_dim=" + embed_dim + "//" + num_heads +
                   ", num_heads=" + num_heads + ")";
            break;
        }

        case NodeType::LayerNorm: {
            std::string normalized_shape = "512";
            auto it = node.parameters.find("normalized_shape");
            if (it != node.parameters.end()) normalized_shape = it->second;
            code = "layers.LayerNormalization()";
            break;
        }

        case NodeType::Embedding: {
            std::string num_embeddings = "10000";
            std::string embedding_dim = "512";
            auto it = node.parameters.find("num_embeddings");
            if (it != node.parameters.end()) num_embeddings = it->second;
            it = node.parameters.find("embedding_dim");
            if (it != node.parameters.end()) embedding_dim = it->second;
            code = "layers.Embedding(input_dim=" + num_embeddings + ", output_dim=" + embedding_dim + ")";
            break;
        }

        case NodeType::GELU:
            code = "layers.Activation('gelu')";
            break;

        case NodeType::ReLU:
            code = "layers.ReLU()";
            break;

        case NodeType::PluginCustom: {
            auto it = node.parameters.find("plugin_qualified_name");
            if (it != node.parameters.end())
                code = cyxwiz::plugin::PluginNodeRegistry::Instance().GenerateCode(it->second, node.parameters, "tensorflow");
            break;
        }

        default:
            // Activation functions and others don't need layers in __init__
            code = "";
            break;
    }

    return code;
}

std::string NodeEditor::NodeTypeToKerasLayer(const MLNode& node) {
    std::string code;

    switch (node.type) {
        case NodeType::Dense: {
            std::string units = "128";
            auto it = node.parameters.find("units");
            if (it != node.parameters.end()) {
                units = it->second;
            }
            code = "layers.Dense(" + units + ")";
            break;
        }

        case NodeType::Conv2D:
            code = "layers.Conv2D(32, kernel_size=3)";
            break;

        case NodeType::MaxPool2D:
            code = "layers.MaxPool2D(pool_size=2)";
            break;

        case NodeType::Flatten:
            code = "layers.Flatten()";
            break;

        case NodeType::Dropout:
            code = "layers.Dropout(0.5)";
            break;

        case NodeType::BatchNorm:
            code = "layers.BatchNormalization()";
            break;

        case NodeType::ReLU:
            code = "layers.ReLU()";
            break;

        case NodeType::Sigmoid:
            code = "layers.Activation('sigmoid')";
            break;

        case NodeType::Tanh:
            code = "layers.Activation('tanh')";
            break;

        case NodeType::Softmax:
            code = "layers.Activation('softmax')";
            break;

        case NodeType::Output: {
            std::string units = "10";
            auto it = node.parameters.find("units");
            if (it != node.parameters.end()) {
                units = it->second;
            }
            code = "layers.Dense(" + units + ", activation='softmax')";
            break;
        }

        case NodeType::LinearAttention: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            // Keras uses same MultiHeadAttention as TensorFlow
            code = "# LinearAttention (O(n)) - requires custom implementation\n"
                   "        layers.MultiHeadAttention(key_dim=" + embed_dim + "//" + num_heads +
                   ", num_heads=" + num_heads + ")  # Replace with linear attention";
            break;
        }

        case NodeType::MultiHeadAttention: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            code = "layers.MultiHeadAttention(key_dim=" + embed_dim + "//" + num_heads +
                   ", num_heads=" + num_heads + ")";
            break;
        }

        case NodeType::LayerNorm:
            code = "layers.LayerNormalization()";
            break;

        case NodeType::Embedding: {
            std::string num_embeddings = "10000";
            std::string embedding_dim = "512";
            auto it = node.parameters.find("num_embeddings");
            if (it != node.parameters.end()) num_embeddings = it->second;
            it = node.parameters.find("embedding_dim");
            if (it != node.parameters.end()) embedding_dim = it->second;
            code = "layers.Embedding(input_dim=" + num_embeddings + ", output_dim=" + embedding_dim + ")";
            break;
        }

        case NodeType::GELU:
            code = "layers.Activation('gelu')";
            break;

        case NodeType::PluginCustom: {
            auto it = node.parameters.find("plugin_qualified_name");
            if (it != node.parameters.end())
                code = cyxwiz::plugin::PluginNodeRegistry::Instance().GenerateCode(it->second, node.parameters, "keras");
            break;
        }

        default:
            code = "";
            break;
    }

    return code;
}

std::string NodeEditor::NodeTypeToPyCyxWizLayer(const MLNode& node) {
    std::string code;

    switch (node.type) {
        case NodeType::Dense: {
            std::string units = "128";
            auto it = node.parameters.find("units");
            if (it != node.parameters.end()) {
                units = it->second;
            }
            // Note: pycyxwiz Dense layer requires input size determination from graph
            code = "cx.Dense(in_features=AUTO, out_features=" + units + ")";
            break;
        }

        case NodeType::Conv2D:
            code = "cx.Conv2D(in_channels=1, out_channels=32, kernel_size=3)";
            break;

        case NodeType::MaxPool2D:
            code = "cx.MaxPool2D(kernel_size=2)";
            break;

        case NodeType::BatchNorm:
            code = "cx.BatchNorm()";
            break;

        case NodeType::Dropout:
            code = "cx.Dropout(p=0.5)";
            break;

        // ===== Attention & Transformer Layers =====
        case NodeType::LinearAttention: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            code = "cx.LinearAttention(dim=" + embed_dim + ", heads=" + num_heads + ")";
            break;
        }

        case NodeType::MultiHeadAttention: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            code = "cx.MultiHeadAttention(embed_dim=" + embed_dim + ", num_heads=" + num_heads + ")";
            break;
        }

        case NodeType::SelfAttention: {
            std::string embed_dim = "512";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            code = "cx.SelfAttention(embed_dim=" + embed_dim + ")";
            break;
        }

        case NodeType::CrossAttention: {
            std::string embed_dim = "512";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            code = "cx.CrossAttention(embed_dim=" + embed_dim + ")";
            break;
        }

        case NodeType::TransformerEncoder: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            std::string ff_dim = "2048";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            it = node.parameters.find("ff_dim");
            if (it != node.parameters.end()) ff_dim = it->second;
            code = "cx.TransformerEncoder(d_model=" + embed_dim + ", nhead=" + num_heads + ", dim_feedforward=" + ff_dim + ")";
            break;
        }

        case NodeType::TransformerDecoder: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            std::string ff_dim = "2048";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            it = node.parameters.find("ff_dim");
            if (it != node.parameters.end()) ff_dim = it->second;
            code = "cx.TransformerDecoder(d_model=" + embed_dim + ", nhead=" + num_heads + ", dim_feedforward=" + ff_dim + ")";
            break;
        }

        // ===== Normalization Layers =====
        case NodeType::LayerNorm: {
            std::string normalized_shape = "512";
            auto it = node.parameters.find("normalized_shape");
            if (it != node.parameters.end()) normalized_shape = it->second;
            code = "cx.LayerNorm(normalized_shape=" + normalized_shape + ")";
            break;
        }

        case NodeType::GroupNorm: {
            std::string num_groups = "32";
            std::string num_channels = "256";
            auto it = node.parameters.find("num_groups");
            if (it != node.parameters.end()) num_groups = it->second;
            it = node.parameters.find("num_channels");
            if (it != node.parameters.end()) num_channels = it->second;
            code = "cx.GroupNorm(num_groups=" + num_groups + ", num_channels=" + num_channels + ")";
            break;
        }

        case NodeType::InstanceNorm:
            code = "cx.InstanceNorm()";
            break;

        // ===== Embedding Layer =====
        case NodeType::Embedding: {
            std::string num_embeddings = "10000";
            std::string embedding_dim = "512";
            auto it = node.parameters.find("num_embeddings");
            if (it != node.parameters.end()) num_embeddings = it->second;
            it = node.parameters.find("embedding_dim");
            if (it != node.parameters.end()) embedding_dim = it->second;
            code = "cx.Embedding(num_embeddings=" + num_embeddings + ", embedding_dim=" + embedding_dim + ")";
            break;
        }

        case NodeType::PositionalEncoding: {
            std::string max_len = "5000";
            std::string d_model = "512";
            auto it = node.parameters.find("max_len");
            if (it != node.parameters.end()) max_len = it->second;
            it = node.parameters.find("d_model");
            if (it != node.parameters.end()) d_model = it->second;
            code = "cx.PositionalEncoding(d_model=" + d_model + ", max_len=" + max_len + ")";
            break;
        }

        // ===== Activation Functions =====
        case NodeType::ReLU:
            code = "cx.ReLU()";
            break;

        case NodeType::GELU:
            code = "cx.GELU()";
            break;

        case NodeType::LeakyReLU: {
            std::string negative_slope = "0.01";
            auto it = node.parameters.find("negative_slope");
            if (it != node.parameters.end()) negative_slope = it->second;
            code = "cx.LeakyReLU(negative_slope=" + negative_slope + ")";
            break;
        }

        case NodeType::Swish:
            code = "cx.Swish()";
            break;

        case NodeType::Mish:
            code = "cx.Mish()";
            break;

        case NodeType::Sigmoid:
            code = "cx.Sigmoid()";
            break;

        case NodeType::Tanh:
            code = "cx.Tanh()";
            break;

        case NodeType::Softmax: {
            std::string dim = "-1";
            auto it = node.parameters.find("dim");
            if (it != node.parameters.end()) dim = it->second;
            code = "cx.Softmax(dim=" + dim + ")";
            break;
        }

        // ===== Recurrent Layers =====
        case NodeType::LSTM: {
            std::string input_size = "512";
            std::string hidden_size = "256";
            std::string num_layers = "1";
            auto normalize_bool = [](const std::string& value) -> std::string {
                if (value == "true" || value == "1" || value == "True") {
                    return "True";
                }
                if (value == "false" || value == "0" || value == "False") {
                    return "False";
                }
                return value;
            };
            auto it = node.parameters.find("input_size");
            if (it != node.parameters.end()) input_size = it->second;
            it = node.parameters.find("hidden_size");
            if (it != node.parameters.end()) hidden_size = it->second;
            it = node.parameters.find("num_layers");
            if (it != node.parameters.end()) num_layers = it->second;
            std::string bidirectional = "false";
            std::string dropout = "0.0";
            it = node.parameters.find("bidirectional");
            if (it != node.parameters.end()) bidirectional = normalize_bool(it->second);
            it = node.parameters.find("dropout");
            if (it != node.parameters.end()) dropout = it->second;
            code = "cx.LSTM(input_size=" + input_size + ", hidden_size=" + hidden_size +
                   ", num_layers=" + num_layers + ", bidirectional=" + bidirectional +
                   ", dropout=" + dropout + ")";
            break;
        }

        case NodeType::GRU: {
            std::string input_size = "512";
            std::string hidden_size = "256";
            std::string num_layers = "1";
            std::string bidirectional = "false";
            auto normalize_bool = [](const std::string& value) -> std::string {
                if (value == "true" || value == "1" || value == "True") {
                    return "True";
                }
                if (value == "false" || value == "0" || value == "False") {
                    return "False";
                }
                return value;
            };
            auto it = node.parameters.find("input_size");
            if (it != node.parameters.end()) input_size = it->second;
            it = node.parameters.find("hidden_size");
            if (it != node.parameters.end()) hidden_size = it->second;
            it = node.parameters.find("num_layers");
            if (it != node.parameters.end()) num_layers = it->second;
            it = node.parameters.find("bidirectional");
            if (it != node.parameters.end()) bidirectional = normalize_bool(it->second);
            code = "cx.GRU(input_size=" + input_size +
                   ", hidden_size=" + hidden_size +
                   ", num_layers=" + num_layers +
                   ", batch_first=True, bidirectional=" + bidirectional + ")";
            break;
        }

        // Tensor shape/merge ops are emitted directly in forward(). They
        // are tensor methods/operators, not layer objects.
        case NodeType::Flatten:
        case NodeType::Reshape:
        case NodeType::View:
        case NodeType::Permute:
        case NodeType::Squeeze:
        case NodeType::Unsqueeze:
        case NodeType::Split:
        case NodeType::Concatenate:
        case NodeType::Add:
        case NodeType::Multiply:
        case NodeType::Average:
        case NodeType::TensorSum:
        case NodeType::TensorMean:
        case NodeType::TensorMax:
        case NodeType::TensorMin:
        case NodeType::TensorProd:
        case NodeType::TensorVar:
        case NodeType::TensorStd:
        case NodeType::TensorBroadcastTo:
        case NodeType::TensorExpand:
        case NodeType::TensorPow:
        case NodeType::TensorSqrt:
        case NodeType::TensorExp:
        case NodeType::TensorLog:
        case NodeType::TensorAbs:
        case NodeType::TensorSign:
        case NodeType::TensorClip:
        case NodeType::TensorDot:
        case NodeType::TensorBatchMatMul:
        case NodeType::TensorCompare:
        case NodeType::TensorLogicalMask:
        case NodeType::TensorIndexSelect:
            code = "";
            break;

        case NodeType::PluginCustom: {
            auto it = node.parameters.find("plugin_qualified_name");
            if (it != node.parameters.end())
                code = cyxwiz::plugin::PluginNodeRegistry::Instance().GenerateCode(it->second, node.parameters, "pycyxwiz");
            break;
        }

        default:
            // Other node types handled in forward pass or not yet implemented
            code = "";
            break;
    }

    return code;
}

// Generate RL code using PyCyxWiz backend with Gymnasium integration
std::string NodeEditor::GenerateRLPyCyxWizCode(const std::vector<int>& sorted_ids) const {
    std::string code;

    // Header with RL imports
    code += "# Auto-generated RL code using CyxWiz Backend with Gymnasium\n";
    code += "# Generated at: " + std::string(__DATE__) + " " + std::string(__TIME__) + "\n\n";
    code += "import pycyxwiz as cx\n";
    code += "import gymnasium as gym\n";
    code += "import numpy as np\n\n";

    // Extract parameters from nodes (matches enhanced node definitions)
    std::string env_name = "CartPole-v1";
    std::string algorithm = "PPO";
    std::string total_timesteps = "100000";
    std::string learning_rate = "3e-4";
    std::string gamma = "0.99";
    std::string clip_range = "0.2";
    std::string hidden_sizes = "128,128";
    std::string activation = "ReLU";
    std::string buffer_capacity = "100000";
    std::string batch_size = "256";
    bool render = false;

    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;

        if (node->type == NodeType::GymEnvironment) {
            auto it = node->parameters.find("env_name");
            if (it != node->parameters.end()) env_name = it->second;
            it = node->parameters.find("render");
            if (it != node->parameters.end()) render = (it->second == "true");
        }
        else if (node->type == NodeType::RLTraining) {
            auto it = node->parameters.find("algorithm");
            if (it != node->parameters.end()) algorithm = it->second;
            it = node->parameters.find("total_timesteps");
            if (it != node->parameters.end()) total_timesteps = it->second;
            it = node->parameters.find("learning_rate");
            if (it != node->parameters.end()) learning_rate = it->second;
            it = node->parameters.find("gamma");
            if (it != node->parameters.end()) gamma = it->second;
            it = node->parameters.find("clip_range");
            if (it != node->parameters.end()) clip_range = it->second;
        }
        else if (node->type == NodeType::PolicyNetwork) {
            auto it = node->parameters.find("hidden_sizes");
            if (it != node->parameters.end()) hidden_sizes = it->second;
            it = node->parameters.find("activation");
            if (it != node->parameters.end()) activation = it->second;
        }
        else if (node->type == NodeType::ReplayBufferNode) {
            auto it = node->parameters.find("capacity");
            if (it != node->parameters.end()) buffer_capacity = it->second;
            it = node->parameters.find("batch_size");
            if (it != node->parameters.end()) batch_size = it->second;
        }
    }

    // Configuration (aligned with PyTorch version)
    code += "# =============== Configuration ===============\n";
    code += "ENV_NAME = \"" + env_name + "\"\n";
    code += "ALGORITHM = \"" + algorithm + "\"\n";
    code += "TOTAL_TIMESTEPS = " + total_timesteps + "\n";
    code += "LEARNING_RATE = " + learning_rate + "\n";
    code += "GAMMA = " + gamma + "\n";
    code += "CLIP_RANGE = " + clip_range + "  # PPO clip parameter\n";
    code += "HIDDEN_SIZES = [" + hidden_sizes + "]\n";
    code += "ACTIVATION = \"" + activation + "\"\n";
    code += "BUFFER_CAPACITY = " + buffer_capacity + "\n";
    code += "BATCH_SIZE = " + batch_size + "\n";
    code += "RENDER = " + std::string(render ? "True" : "False") + "\n\n";

    // Activation function mapping
    code += "# Activation function mapping\n";
    code += "ACTIVATION_FNS = {\n";
    code += "    \"ReLU\": cx.relu,\n";
    code += "    \"Tanh\": cx.tanh,\n";
    code += "    \"Sigmoid\": cx.sigmoid,\n";
    code += "    \"GELU\": cx.gelu,\n";
    code += "    \"Swish\": cx.swish,\n";
    code += "}\n";
    code += "activation_fn = ACTIVATION_FNS.get(ACTIVATION, cx.relu)\n\n";

    // Policy Network using CyxWiz
    code += "# =============== Policy Network (CyxWiz Backend) ===============\n";
    code += "class PolicyNetwork:\n";
    code += "    def __init__(self, obs_dim, action_dim):\n";
    code += "        self.layers = []\n";
    code += "        in_features = obs_dim\n";
    code += "        for hidden_size in HIDDEN_SIZES:\n";
    code += "            self.layers.append(cx.Dense(in_features, hidden_size))\n";
    code += "            in_features = hidden_size\n";
    code += "        self.output_layer = cx.Dense(in_features, action_dim)\n";
    code += "    \n";
    code += "    def forward(self, x):\n";
    code += "        for layer in self.layers:\n";
    code += "            x = activation_fn(layer.forward(x))\n";
    code += "        return cx.softmax(self.output_layer.forward(x))\n";
    code += "    \n";
    code += "    def get_action(self, obs):\n";
    code += "        obs_tensor = cx.Tensor(obs.reshape(1, -1))\n";
    code += "        probs = self.forward(obs_tensor).numpy().flatten()\n";
    code += "        return np.random.choice(len(probs), p=probs)\n\n";

    // Value Network using CyxWiz
    code += "class ValueNetwork:\n";
    code += "    def __init__(self, obs_dim):\n";
    code += "        self.layers = []\n";
    code += "        in_features = obs_dim\n";
    code += "        for hidden_size in HIDDEN_SIZES:\n";
    code += "            self.layers.append(cx.Dense(in_features, hidden_size))\n";
    code += "            in_features = hidden_size\n";
    code += "        self.output_layer = cx.Dense(in_features, 1)\n";
    code += "    \n";
    code += "    def forward(self, x):\n";
    code += "        for layer in self.layers:\n";
    code += "            x = activation_fn(layer.forward(x))\n";
    code += "        return self.output_layer.forward(x)\n\n";

    // Replay Buffer
    code += "# =============== Replay Buffer ===============\n";
    code += "class ReplayBuffer:\n";
    code += "    def __init__(self, capacity=BUFFER_CAPACITY):\n";
    code += "        self.capacity = capacity\n";
    code += "        self.buffer = []\n";
    code += "        self.position = 0\n";
    code += "    \n";
    code += "    def push(self, state, action, reward, next_state, done):\n";
    code += "        if len(self.buffer) < self.capacity:\n";
    code += "            self.buffer.append(None)\n";
    code += "        self.buffer[self.position] = (state, action, reward, next_state, done)\n";
    code += "        self.position = (self.position + 1) % self.capacity\n";
    code += "    \n";
    code += "    def sample(self, batch_size=BATCH_SIZE):\n";
    code += "        if len(self.buffer) < batch_size:\n";
    code += "            return None  # Not enough samples yet\n";
    code += "        indices = np.random.choice(len(self.buffer), batch_size, replace=False)\n";
    code += "        batch = [self.buffer[i] for i in indices]\n";
    code += "        states, actions, rewards, next_states, dones = zip(*batch)\n";
    code += "        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)\n";
    code += "    \n";
    code += "    def can_sample(self, batch_size=BATCH_SIZE):\n";
    code += "        return len(self.buffer) >= batch_size\n";
    code += "    \n";
    code += "    def __len__(self):\n";
    code += "        return len(self.buffer)\n\n";

    // Training loop
    code += "# =============== Training ===============\n";
    code += "def train():\n";
    code += "    # Initialize CyxWiz backend\n";
    code += "    cx.initialize()\n";
    code += "    device = cx.get_device(cx.DeviceType.CUDA if cx.cuda_available() else cx.DeviceType.CPU)\n";
    code += "    cx.set_device(device)\n";
    code += "    print(f'Using device: {device.name()}')\n\n";
    code += "    # Create environment\n";
    code += "    render_mode = \"human\" if RENDER else None\n";
    code += "    env = gym.make(ENV_NAME, render_mode=render_mode)\n";
    code += "    obs_dim = env.observation_space.shape[0]\n";
    code += "    action_dim = env.action_space.n\n\n";
    code += "    # Create networks\n";
    code += "    policy = PolicyNetwork(obs_dim, action_dim)\n";
    code += "    value = ValueNetwork(obs_dim)\n";
    code += "    buffer = ReplayBuffer()\n\n";
    code += "    # Training loop (timestep-based like SB3)\n";
    code += "    episode_rewards = []\n";
    code += "    total_steps = 0\n";
    code += "    episode = 0\n";
    code += "    \n";
    code += "    while total_steps < TOTAL_TIMESTEPS:\n";
    code += "        obs, info = env.reset()\n";
    code += "        episode_reward = 0\n";
    code += "        done = False\n";
    code += "        \n";
    code += "        while not done and total_steps < TOTAL_TIMESTEPS:\n";
    code += "            # Select action\n";
    code += "            action = policy.get_action(obs)\n";
    code += "            \n";
    code += "            # Environment step\n";
    code += "            next_obs, reward, terminated, truncated, info = env.step(action)\n";
    code += "            done = terminated or truncated\n";
    code += "            episode_reward += reward\n";
    code += "            total_steps += 1\n";
    code += "            \n";
    code += "            # Store transition\n";
    code += "            buffer.push(obs, action, reward, next_obs, done)\n";
    code += "            obs = next_obs\n";
    code += "            \n";
    code += "            # Learn from buffer (simplified - real PPO is more complex)\n";
    code += "            if buffer.can_sample(BATCH_SIZE):\n";
    code += "                batch = buffer.sample(BATCH_SIZE)\n";
    code += "                # Placeholder for actual policy update\n";
    code += "                pass\n";
    code += "        \n";
    code += "        episode += 1\n";
    code += "        episode_rewards.append(episode_reward)\n";
    code += "        \n";
    code += "        if episode % 10 == 0:\n";
    code += "            avg_reward = np.mean(episode_rewards[-10:])\n";
    code += "            print(f'Episode {episode}, Steps: {total_steps}/{TOTAL_TIMESTEPS}, Avg Reward: {avg_reward:.2f}')\n";
    code += "    \n";
    code += "    env.close()\n";
    code += "    print(f'Training complete after {episode} episodes and {total_steps} steps.')\n";
    code += "    print(f'Final avg reward: {np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards):.2f}')\n";
    code += "    return policy, value\n\n";

    // Main
    code += "# =============== Main ===============\n";
    code += "if __name__ == \"__main__\":\n";
    code += "    # Note: For production RL training, use Stable-Baselines3 (PyTorch framework)\n";
    code += "    # This code demonstrates CyxWiz backend integration with Gymnasium\n";
    code += "    policy, value = train()\n";

    return code;
}

std::vector<int> NodeEditor::TopologicalSort() {
    std::vector<int> result;
    std::map<int, int> in_degree;
    std::map<int, std::vector<int>> adj_list;

    // Initialize in-degree for all nodes
    for (const auto& node : nodes_) {
        in_degree[node.id] = 0;
        adj_list[node.id] = {};
    }

    // Build adjacency list and calculate in-degrees
    for (const auto& link : links_) {
        adj_list[link.from_node].push_back(link.to_node);
        in_degree[link.to_node]++;
    }

    // Find all nodes with in-degree 0 (starting nodes)
    std::vector<int> queue;
    for (const auto& [node_id, degree] : in_degree) {
        if (degree == 0) {
            queue.push_back(node_id);
        }
    }

    // Process nodes
    while (!queue.empty()) {
        int current = queue.front();
        queue.erase(queue.begin());
        result.push_back(current);

        // Reduce in-degree for neighbors
        for (int neighbor : adj_list[current]) {
            in_degree[neighbor]--;
            if (in_degree[neighbor] == 0) {
                queue.push_back(neighbor);
            }
        }
    }

    // Check if all nodes were processed (no cycles)
    if (result.size() != nodes_.size()) {
        spdlog::error("Graph has cycles - cannot generate code");
        return {};
    }

    return result;
}


} // namespace gui
