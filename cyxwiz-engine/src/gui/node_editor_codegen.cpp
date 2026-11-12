#include "node_editor.h"
#include "../plugin/plugin_manager.h"
#include "../core/rl_graph_compile.h"
#include <cstdlib>
#include "activation_codegen_contract.h"
#include "panels/script_editor.h"
#include "../core/file_dialogs.h"
#include "../core/project_manager.h"
#include "../core/studio_toolbar_presentation.h"
#include "../core/pipeline_runtime_capabilities.h"
#include "../core/extension_node_metadata.h"
#include "../core/extension_node_registry.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <ctime>
#include <fstream>
#include <sstream>
#include <set>
#include <map>
#include <queue>
#include <functional>
#include <initializer_list>
#include <vector>

namespace gui {

namespace {

// The local time now, for a generated file's header and the result line
// (the time the code was generated, not when the Engine was built).
std::string ClockText(const char* format) {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char text[64] = {};
    std::strftime(text, sizeof(text), format, &local);
    return text;
}

const char* FrameworkName(CodeFramework framework) {
    switch (framework) {
        case CodeFramework::PyTorch: return "PyTorch";
        case CodeFramework::TensorFlow: return "TensorFlow";
        case CodeFramework::Keras: return "Keras";
        case CodeFramework::PyCyxWiz: return "PyCyxWiz";
    }
    return "PyTorch";
}

int LineCount(const std::string& code) {
    return static_cast<int>(std::count(code.begin(), code.end(), '\n')) + (code.empty() || code.back() == '\n' ? 0 : 1);
}

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

std::string GetParamOrAliases(const MLNode& node,
                              std::initializer_list<const char*> keys,
                              const std::string& fallback) {
    for (const char* key : keys) {
        const auto it = node.parameters.find(key);
        if (it != node.parameters.end() && !TrimCopy(it->second).empty()) {
            return TrimCopy(it->second);
        }
    }
    return fallback;
}

std::string PythonBoolLiteral(const std::string& value) {
    std::string lower = TrimCopy(value);
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return (lower == "true" || lower == "1" || lower == "yes") ? "True" : "False";
}

std::string BlockedCodegenConfiguration(
    const char* framework,
    const std::string& error) {
    std::string escaped_error;
    escaped_error.reserve(error.size());
    for (const char character : error) {
        switch (character) {
            case '\\': escaped_error += "\\\\"; break;
            case '\'': escaped_error += "\\\'"; break;
            case '\n': escaped_error += "\\n"; break;
            case '\r': escaped_error += "\\r"; break;
            case '\t': escaped_error += "\\t"; break;
            default: escaped_error += character; break;
        }
    }
    return std::string("# ") + framework +
        " export is blocked by Studio contract validation.\n"
        "raise ValueError('" + escaped_error + "')\n";
}

} // namespace

std::string NodeEditor::CodeProblem() {
    if (nodes_.empty()) return "The canvas is empty. Add a node first.";
    std::string error;
    if (!ValidateGraph(error)) return error.empty() ? std::string("The graph is not complete.") : error;
    return {};
}

std::string NodeEditor::GenerateCodeText(CodeFramework framework) {
    if (nodes_.empty()) return {};
    const std::vector<int> sorted_ids = TopologicalSort();
    if (sorted_ids.empty()) {
        spdlog::error("Failed to perform topological sort - graph may have cycles");
        return {};
    }
    switch (framework) {
        case CodeFramework::PyTorch: return GeneratePyTorchCode(sorted_ids);
        case CodeFramework::TensorFlow: return GenerateTensorFlowCode(sorted_ids);
        case CodeFramework::Keras: return GenerateKerasCode(sorted_ids);
        case CodeFramework::PyCyxWiz: return GeneratePyCyxWizCode(sorted_ids);
        default: return {};
    }
}

namespace {

// What Generate and Save as .py report.
cyxwiz::studio_toolbar::CodeResult StartResult(CodeFramework framework) {
    cyxwiz::studio_toolbar::CodeResult result;
    result.framework = FrameworkName(framework);
    result.at = ClockText("%H:%M");
    return result;
}

}  // namespace

// On the interface thread: the code is built from the graph as it is now, so
// no worker reads the graph while it is being edited.
void NodeEditor::GenerateCodeIntoEditor() {
    namespace st = cyxwiz::studio_toolbar;
    st::CodeResult result = StartResult(selected_framework_);
    const std::string problem = CodeProblem();
    const std::string code = problem.empty() ? GenerateCodeText(selected_framework_) : std::string();
    st::CodeInputs in;
    in.framework = result.framework;
    in.graph = current_file_path_.empty() ? std::string() : GetGraphStem();
    const std::string file_name = st::CodeMenuFor(in).file_name;
    if (!problem.empty()) {
        result.error = problem;
    } else if (code.empty()) {
        result.error = "the graph has a cycle, or no node in it writes " + result.framework + " code";
    } else if (!script_editor_) {
        result.error = "the Script Editor is not available";
    } else {
        script_editor_->LoadGeneratedCode(code, file_name);
        script_editor_->SetVisible(true);
        result.ok = true;
        result.lines = LineCount(code);
        result.where = file_name;
    }
    code_result_ = st::CodeResultText(result);
    code_result_ok_ = result.ok;
    if (result.ok) spdlog::info("{}", code_result_);
    else spdlog::warn("{}", code_result_);
}

void NodeEditor::SaveCodeAsPython() {
    namespace st = cyxwiz::studio_toolbar;
    st::CodeResult result = StartResult(selected_framework_);
    result.to_file = true;
    const std::string problem = CodeProblem();
    const std::string code = problem.empty() ? GenerateCodeText(selected_framework_) : std::string();
    if (!problem.empty() || code.empty()) {
        result.error = problem.empty() ? "the graph has a cycle, or no node in it writes " + result.framework + " code"
                                       : problem;
    } else {
        st::CodeInputs in;
        in.framework = result.framework;
        in.graph = current_file_path_.empty() ? std::string() : GetGraphStem();
        const std::string file_name = st::CodeMenuFor(in).file_name;
        auto& project = cyxwiz::ProjectManager::Instance();
        const std::string start = project.HasActiveProject() ? project.GetScriptsPath() : std::string();
        const std::string title = "Save " + result.framework + " code";
        const auto path = cyxwiz::FileDialogs::SaveFile(title.c_str(), {{"Python", "py"}},
                                                        start.empty() ? nullptr : start.c_str(), file_name.c_str());
        if (!path) return;  // cancelled: nothing to report
        std::ofstream file(*path, std::ios::binary | std::ios::trunc);
        if (!file || !(file << code)) {
            result.error = "cannot write " + *path;
        } else {
            result.ok = true;
            result.lines = LineCount(code);
            result.where = *path;
        }
    }
    code_result_ = st::CodeResultText(result);
    code_result_ok_ = result.ok;
    if (result.ok) spdlog::info("{}", code_result_);
    else spdlog::warn("{}", code_result_);
}

// An RL graph: Environment, Policy Head, Value Head, PPO (TOFIX195 S4.3)
bool NodeEditor::IsRLGraph(const std::vector<int>& sorted_ids) const {
    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (node && cyxwiz::IsRLGraphNode(node->type)) return true;
    }
    return false;
}

namespace {

std::string PyNumber(double v) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.7g", v);  // the settings are floats
    return buf;
}

std::string PyString(const std::string& s) {
    std::string out = "\"";
    for (const char c : s) {
        if (c == '\\' || c == '"') out += '\\';
        out += c;
    }
    return out + "\"";
}

// SB3 MlpPolicy takes the Dense widths and one activation function
std::string NetArch(const std::vector<cyxwiz::CompiledLayer>& layers, std::string& activation,
                    std::vector<std::string>& notes) {
    std::string widths;
    for (const auto& l : layers) {
        if (l.type == gui::NodeType::Dense) {
            widths += (widths.empty() ? "" : ", ") + std::to_string(l.units);
            continue;
        }
        const char* fn = l.type == gui::NodeType::Tanh        ? "nn.Tanh"
                         : l.type == gui::NodeType::ReLU      ? "nn.ReLU"
                         : l.type == gui::NodeType::LeakyReLU ? "nn.LeakyReLU"
                         : l.type == gui::NodeType::GELU      ? "nn.GELU"
                         : l.type == gui::NodeType::ELU       ? "nn.ELU"
                         : l.type == gui::NodeType::Sigmoid   ? "nn.Sigmoid"
                         : l.type == gui::NodeType::Swish     ? "nn.SiLU"
                                                              : nullptr;
        if (!fn) {
            notes.push_back(l.name + " has no MlpPolicy equivalent and is left out");
            continue;
        }
        if (activation.empty()) activation = fn;
        else if (activation != fn) notes.push_back(l.name + ": MlpPolicy uses one activation, " + activation);
    }
    return "[" + widths + "]";
}

}  // namespace

// Stable-Baselines3's PPO on the graph's environment (through pycyxwiz.rl,
// the Engine's own physics), layers and settings: the code of the run Train
// starts in the Engine.
std::string NodeEditor::GenerateRLCode(const std::vector<int>& /*sorted_ids*/) const {
    cyxwiz::GraphCompiler compiler;
    const cyxwiz::TrainingConfiguration config = compiler.Compile(nodes_, links_);
    if (!config.rl) {
        std::string code = "# The RL graph does not compile yet:\n";
        for (const auto& issue : config.issues)
            if (issue.level == cyxwiz::IssueLevel::Error)
                code += "#   " + (issue.node_name.empty() ? "" : issue.node_name + ": ") + issue.message + "\n";
        return code;
    }
    const auto& rl = *config.rl;
    const auto& c = rl.ppo;
    std::vector<std::string> notes;
    std::string policy_act, value_act;
    const std::string pi = NetArch(rl.policy_layers, policy_act, notes);
    const std::string vf = NetArch(rl.value_layers, value_act, notes);
    if (!value_act.empty() && !policy_act.empty() && value_act != policy_act)
        notes.push_back("the value layers' " + value_act + " becomes " + policy_act + " (MlpPolicy has one activation)");
    const std::string activation = policy_act.empty() ? (value_act.empty() ? "nn.Identity" : value_act) : policy_act;

    const size_t colon = rl.environment_key.find(':');
    const std::string provider = colon == std::string::npos ? "" : rl.environment_key.substr(0, colon);
    std::string plugin_dir;
    if (const auto* loaded = cyxwiz::plugin::PluginManager::Instance().GetLoadedPlugin(provider))
        plugin_dir = loaded->plugin_dir.string();

    std::string settings;
    for (const auto& [k, v] : rl.environment_settings) settings += ", " + k + "=" + PyString(v);

    std::string code;
    code += "# PPO on " + rl.environment.id + " x " + std::to_string(rl.copies) + ", from the CyxWiz graph.\n";
    code += "# Stable-Baselines3 on the Engine's own environments (pycyxwiz.rl): the same physics,\n";
    code += "# layers and settings the Engine's PPO trains with when you press Train.\n";
    code += "# Needs numpy, gymnasium, torch and stable_baselines3 (the Engine's python-embed has them).\n\n";
    code += "import torch.nn as nn\n";
    code += "from stable_baselines3 import PPO\n";
    code += "from stable_baselines3.common.callbacks import BaseCallback\n";
    code += "from stable_baselines3.common.env_util import make_vec_env\n";
    if (c.normalize_observations || c.normalize_rewards)
        code += "from stable_baselines3.common.vec_env import VecNormalize\n";
    code += "from pycyxwiz import rl\n\n";
    if (!provider.empty() && provider != "cyxwiz.builtin") {
        code += plugin_dir.empty() ? "# Load the plugin that offers it: rl.load_plugin(<its folder>)\n"
                                   : "rl.load_plugin(r" + PyString(plugin_dir) + ")\n";
    }
    code += "env = make_vec_env(lambda: rl.Env(" + PyString(rl.environment.id) + settings + "), n_envs=" +
            std::to_string(rl.copies) + ", seed=" + std::to_string(c.seed) + ")\n";
    if (c.normalize_observations || c.normalize_rewards)
        code += std::string("env = VecNormalize(env, norm_obs=") + (c.normalize_observations ? "True" : "False") +
                ", norm_reward=" + (c.normalize_rewards ? "True" : "False") + ", gamma=" + PyNumber(c.gamma) + ")\n";
    code += "\n";
    for (const auto& n : notes) code += "# " + n + "\n";
    code += "model = PPO(\n";
    code += "    \"MlpPolicy\",\n    env,\n";
    code += "    n_steps=" + std::to_string(c.n_steps) + ",\n";
    code += "    batch_size=" + std::to_string(c.batch_size) + ",\n";
    code += "    n_epochs=" + std::to_string(c.n_epochs) + ",\n";
    code += "    learning_rate=" + PyNumber(c.learning_rate) + ",\n";
    code += "    gamma=" + PyNumber(c.gamma) + ",\n";
    code += "    gae_lambda=" + PyNumber(c.gae_lambda) + ",\n";
    code += "    clip_range=" + PyNumber(c.loss.clip_range) + ",\n";
    code += "    clip_range_vf=" + (c.loss.clip_range_vf > 0.0f ? PyNumber(c.loss.clip_range_vf) : std::string("None")) + ",\n";
    code += "    ent_coef=" + PyNumber(c.loss.ent_coef) + ",\n";
    code += "    vf_coef=" + PyNumber(c.loss.vf_coef) + ",\n";
    code += "    max_grad_norm=" + PyNumber(c.max_grad_norm > 0.0f ? c.max_grad_norm : 1e9) + ",\n";
    code += std::string("    normalize_advantage=") + (c.loss.normalize_advantage ? "True" : "False") + ",\n";
    code += "    policy_kwargs=dict(net_arch=dict(pi=" + pi + ", vf=" + vf + "), activation_fn=" + activation +
            (rl.environment.discrete_actions ? std::string() : ", log_std_init=" + PyNumber(c.log_std_init)) + "),\n";
    code += "    seed=" + std::to_string(c.seed) + ",\n";
    code += "    device=\"cpu\",\n    verbose=1,\n)\n\n";
    if (!std::isnan(c.target_return)) {
        code += "\nclass StopAtReturn(BaseCallback):\n";
        code += "    \"\"\"Stop once the last 100 episodes average the target.\"\"\"\n\n";
        code += "    def _on_step(self):\n";
        code += "        buffer = self.model.ep_info_buffer\n";
        code += "        return not (len(buffer) >= 100 and sum(e[\"r\"] for e in buffer) / len(buffer) >= " +
                PyNumber(c.target_return) + ")\n\n\n";
        code += "model.learn(total_timesteps=" + std::to_string(c.total_timesteps) + ", callback=StopAtReturn())\n";
    } else {
        code += "model.learn(total_timesteps=" + std::to_string(c.total_timesteps) + ")\n";
    }
    code += "model.save(" + PyString("ppo_" + rl.environment.id) + ")\n";
    if (c.normalize_observations || c.normalize_rewards)
        code += "env.save(" + PyString("ppo_" + rl.environment.id + "_vecnormalize.pkl") + ")\n";
    return code;
}

std::optional<std::string>
NodeEditor::FindDenseActivationConfigurationError(
    const std::vector<int>& sorted_ids) const {
    for (const int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;
        if (const auto error =
                cyxwiz::ResolveInvalidDenseActivationConfigurationReason(
                    node->type, node->parameters)) {
            return EffectiveNodeName(*node) + ": " + *error;
        }
    }
    return std::nullopt;
}

std::optional<std::string>
NodeEditor::FindUnsupportedSequentialLayerError(
    const std::vector<int>& sorted_ids) const {
    for (const int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;
        if (const char* reason =
                cyxwiz::ResolvePipelineUnsupportedSequentialModelLayerReason(
                    node->type)) {
            return EffectiveNodeName(*node) + ": " + reason;
        }
        if (node->type == NodeType::CrossAttention) {
            return EffectiveNodeName(*node) +
                   ": code export does not support Cross Attention (a two-input layer) yet; train it in Studio";
        }
        if (node->type == NodeType::LinearAttention) {
            return EffectiveNodeName(*node) +
                   ": code export does not support Linear Attention yet (frameworks have no built-in layer for it); "
                   "train it in Studio";
        }
        if (node->type == NodeType::Conv3D) {
            return EffectiveNodeName(*node) +
                   ": code export does not support Conv3D (volume rows) yet; train it in Studio";
        }
        if (auto refusal = cyxwiz::ExtensionCodeExportRefusal(*node)) return refusal;
        switch (node->type) {
            case NodeType::CenterCrop:
            case NodeType::RandomCrop:
            case NodeType::HorizontalFlip:
            case NodeType::VerticalFlip:
            case NodeType::ImageRotate:
            case NodeType::ColorJitter:
            case NodeType::ImageGaussianBlur:
            case NodeType::Grayscale:
            case NodeType::MorphologyTransform:
            case NodeType::AdvancedAugment:
            case NodeType::AugmentationPreset:
            case NodeType::ImageContrast:
            case NodeType::ImageDenoise:
            case NodeType::ImageSharpen:
            case NodeType::ImageEdges:
            case NodeType::ImageSmooth:
            case NodeType::ImageWarp:
            case NodeType::ImagePyramid:
            case NodeType::RandomResizedCrop:
            case NodeType::RandomAffine:
            case NodeType::RandomPerspective:
            case NodeType::GaussianNoise:
            case NodeType::RandomGrayscale:
                return EffectiveNodeName(*node) +
                       ": code export does not include image transforms yet; train it in Studio";
            case NodeType::QualityAnalyzer:
                return EffectiveNodeName(*node) +
                       ": code export does not include the Quality Analyzer's filter yet; train it in Studio";
            default:
                break;
        }
    }
    return std::nullopt;
}

std::string NodeEditor::GeneratePyTorchCode(const std::vector<int>& sorted_ids) {
    // Check if this is an RL graph
    if (IsRLGraph(sorted_ids)) {
        return GenerateRLCode(sorted_ids);
    }
    if (const auto error = FindUnsupportedSequentialLayerError(sorted_ids)) {
        return BlockedCodegenConfiguration("PyTorch", *error);
    }
    if (const auto error = FindDenseActivationConfigurationError(sorted_ids)) {
        return BlockedCodegenConfiguration("PyTorch", *error);
    }

    std::string code;

    // Header
    code += "# Auto-generated PyTorch model from CyxWiz Node Editor\n";
    code += "# Generated at: " + ClockText("%Y-%m-%d %H:%M:%S") + "\n\n";
    code += "import torch\n";
    code += "import torch.nn as nn\n";
    code += "import torch.nn.functional as F\n";
    code += "import torch.optim as optim\n";
    code += "import math\n";
    code += "import numpy as np\n\n";

    code += "class PositionalEncoding(nn.Module):\n";
    code += "    def __init__(self, d_model, max_len=5000):\n";
    code += "        super().__init__()\n";
    code += "        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)\n";
    code += "        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))\n";
    code += "        encoding = torch.zeros(1, max_len, d_model)\n";
    code += "        encoding[0, :, 0::2] = torch.sin(position * div_term)\n";
    code += "        encoding[0, :, 1::2] = torch.cos(position * div_term[:encoding[0, :, 1::2].shape[-1]])\n";
    code += "        self.register_buffer('encoding', encoding)\n\n";
    code += "    def forward(self, x):\n";
    code += "        if x.size(1) > self.encoding.size(1):\n";
    code += "            raise ValueError('sequence length exceeds max_sequence_length')\n";
    code += "        return x + self.encoding[:, :x.size(1), :]\n\n";

    code += "def configure_ffn_dropout(block, probability):\n";
    code += "    block.dropout.p = probability\n";
    code += "    return block\n\n";
    code += "class CausalTransformerDecoderBlock(nn.Module):\n";
    code += "    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, norm_first=False, ffn_dropout=0.0):\n";
    code += "        super().__init__()\n";
    code += "        self.block = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True, norm_first=norm_first)\n";
    code += "        self.block.dropout.p = ffn_dropout\n\n";
    code += "    def forward(self, x):\n";
    code += "        causal_mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device, dtype=torch.bool), diagonal=1)\n";
    code += "        return self.block(x, src_mask=causal_mask)\n\n";
    // Configurable decoder block (tofix112): RMSNorm / gated GLU-family FFN /
    // activation / biases / RoPE / QK-norm. Emitted as explicit PyTorch so exports
    // match the backend math (gelu = tanh approximation, SwiGLU = act(gate(x)) * up(x)).
    // verify_transformer_block_options.py executes exactly these lines as its
    // PyTorch reference, so keep them one `code +=` per source line.
    code += "class SquaredReLU(nn.Module):\n";
    code += "    def forward(self, x):\n";
    code += "        return torch.relu(x).square()\n";
    code += "\n";
    code += "class ConfigurableAttention(nn.Module):\n";
    code += "    # Self-attention with the CyxWiz options; returns (output, None) like nn.MultiheadAttention (key/value args unused).\n";
    code += "    def __init__(self, d_model, nhead, dropout=0.0, bias=True, num_kv_heads=0, causal=True, qk_norm=False,\n";
    code += "                 norm_eps=1e-5, position_encoding='none', rope_base=10000.0, rope_fraction=1.0,\n";
    code += "                 sliding_window=0, logit_softcap=0.0):\n";
    code += "        super().__init__()\n";
    code += "        self.nhead, self.kv_heads, self.head_dim = nhead, num_kv_heads or nhead, d_model // nhead\n";
    code += "        self.q_proj = nn.Linear(d_model, d_model, bias=bias)\n";
    code += "        self.k_proj = nn.Linear(d_model, self.kv_heads * self.head_dim, bias=bias)\n";
    code += "        self.v_proj = nn.Linear(d_model, self.kv_heads * self.head_dim, bias=bias)\n";
    code += "        self.out_proj = nn.Linear(d_model, d_model, bias=bias)\n";
    code += "        # QK-norm: per-head RMSNorm of queries and keys before the scores (OLMo 2, Gemma 3, Qwen3)\n";
    code += "        self.q_norm = nn.RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else None\n";
    code += "        self.k_norm = nn.RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else None\n";
    code += "        self.causal, self.position_encoding, self.rope_base = causal, position_encoding, rope_base\n";
    code += "        self.rope_dims = int(rope_fraction * self.head_dim / 2 + 1e-6) * 2\n";
    code += "        self.sliding_window, self.logit_softcap, self.dropout = sliding_window, logit_softcap, nn.Dropout(dropout)\n";
    code += "\n";
    code += "    def _rotary(self, x):\n";
    code += "        # half-split RoPE on the first rope_dims features: x*cos + rotate_half(x)*sin, base^(-2i/rope_dims)\n";
    code += "        seq, dim = x.size(-2), self.rope_dims\n";
    code += "        inv_freq = self.rope_base ** (-torch.arange(0, dim, 2, device=x.device, dtype=x.dtype) / dim)\n";
    code += "        angle = torch.arange(seq, device=x.device, dtype=x.dtype)[:, None] * inv_freq[None, :]\n";
    code += "        cos, sin = torch.cat([angle.cos(), angle.cos()], -1), torch.cat([angle.sin(), angle.sin()], -1)\n";
    code += "        rot, rest = x[..., :dim], x[..., dim:]\n";
    code += "        x1, x2 = rot[..., :dim // 2], rot[..., dim // 2:]\n";
    code += "        return torch.cat([rot * cos + torch.cat([-x2, x1], -1) * sin, rest], -1)\n";
    code += "\n";
    code += "    def _alibi_slopes(self):\n";
    code += "        # Press et al. 2022, including head counts that are not a power of two\n";
    code += "        power_of_two = lambda n: [(2.0 ** (-8.0 / n)) ** i for i in range(1, n + 1)]\n";
    code += "        closest = 2 ** math.floor(math.log2(self.nhead))\n";
    code += "        return power_of_two(closest) + power_of_two(2 * closest)[0::2][:self.nhead - closest]\n";
    code += "\n";
    code += "    def forward(self, x, key=None, value=None, need_weights=False):\n";
    code += "        batch, seq, width = x.shape\n";
    code += "        q = self.q_proj(x).view(batch, seq, self.nhead, self.head_dim).transpose(1, 2)\n";
    code += "        k = self.k_proj(x).view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)\n";
    code += "        v = self.v_proj(x).view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)\n";
    code += "        if self.q_norm is not None:\n";
    code += "            q, k = self.q_norm(q), self.k_norm(k)\n";
    code += "        if self.position_encoding == 'rope':\n";
    code += "            q, k = self._rotary(q), self._rotary(k)\n";
    code += "        groups = self.nhead // self.kv_heads\n";
    code += "        k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)\n";
    code += "        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)\n";
    code += "        if self.logit_softcap > 0:\n";
    code += "            scores = self.logit_softcap * torch.tanh(scores / self.logit_softcap)\n";
    code += "        pos = torch.arange(seq, device=x.device)\n";
    code += "        if self.position_encoding == 'alibi':\n";
    code += "            slopes = torch.tensor(self._alibi_slopes(), device=x.device, dtype=x.dtype)\n";
    code += "            scores = scores + slopes[:, None, None] * (pos[None, :] - pos[:, None]).to(x.dtype)\n";
    code += "        if self.causal:\n";
    code += "            hidden = pos[None, :] > pos[:, None]\n";
    code += "            if self.sliding_window > 0:\n";
    code += "                hidden = hidden | (pos[:, None] - pos[None, :] >= self.sliding_window)\n";
    code += "            scores = scores.masked_fill(hidden, float('-inf'))\n";
    code += "        context = self.dropout(torch.softmax(scores, -1)) @ v\n";
    code += "        return self.out_proj(context.transpose(1, 2).reshape(batch, seq, width)), None\n";
    code += "\n";
    code += "class ConfigurableCausalDecoderBlock(nn.Module):\n";
    code += "    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0, norm_first=False, ffn_dropout=0.0,\n";
    code += "                 norm_type='layer_norm', norm_eps=1e-5, ffn_type='mlp', ffn_activation='relu', ffn_bias=True,\n";
    code += "                 position_encoding='external', rope_base=10000.0, attention_bias=True, qk_norm=False,\n";
    code += "                 rope_fraction=1.0, block_layout='sequential', sandwich_norm=False, residual_init_scale=1.0,\n";
    code += "                 attn_logit_softcap=0.0, sliding_window=0, num_kv_heads=0):\n";
    code += "        super().__init__()\n";
    code += "        self.norm_first, self.parallel = norm_first, block_layout == 'parallel'\n";
    code += "        self.self_attn = ConfigurableAttention(d_model, nhead, dropout, attention_bias, num_kv_heads, True, qk_norm,\n";
    code += "                                               norm_eps, 'none' if position_encoding == 'external' else position_encoding,\n";
    code += "                                               rope_base, rope_fraction, sliding_window, attn_logit_softcap)\n";
    code += "        make_norm = (lambda: nn.RMSNorm(d_model, eps=norm_eps)) if norm_type == 'rms_norm' else (lambda: nn.LayerNorm(d_model, eps=norm_eps))\n";
    code += "        self.norm1 = make_norm()\n";
    code += "        self.norm2 = None if self.parallel else make_norm()  # parallel blocks share norm1\n";
    code += "        # sandwich norm (Gemma 2/3): also normalize each sub-layer output before the residual add\n";
    code += "        self.post_attn_norm = make_norm() if sandwich_norm else None\n";
    code += "        self.post_ffn_norm = make_norm() if sandwich_norm else None\n";
    code += "        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=ffn_bias)  # up\n";
    code += "        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=ffn_bias)  # down\n";
    code += "        self.gate = nn.Linear(d_model, dim_feedforward, bias=ffn_bias) if ffn_type == 'gated' else None\n";
    code += "        self.act = {'relu': nn.ReLU(), 'gelu': nn.GELU(approximate='tanh'), 'gelu_exact': nn.GELU(), 'silu': nn.SiLU(),\n";
    code += "                    'mish': nn.Mish(), 'elu': nn.ELU(), 'selu': nn.SELU(), 'leaky_relu': nn.LeakyReLU(0.01),\n";
    code += "                    'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh(), 'hardswish': nn.Hardswish(),\n";
    code += "                    'squared_relu': SquaredReLU()}[ffn_activation]\n";
    code += "        self.dropout1, self.dropout3, self.ffn_dropout = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(ffn_dropout)\n";
    code += "        if residual_init_scale != 1.0:\n";
    code += "            with torch.no_grad():  # GPT-2 style scaled init of the residual output projections\n";
    code += "                self.self_attn.out_proj.weight.mul_(residual_init_scale)\n";
    code += "                self.linear2.weight.mul_(residual_init_scale)\n";
    code += "\n";
    code += "    def _ffn(self, x):\n";
    code += "        hidden = self.act(self.gate(x)) * self.linear1(x) if self.gate is not None else self.act(self.linear1(x))\n";
    code += "        return self.linear2(self.ffn_dropout(hidden))\n";
    code += "\n";
    code += "    def forward(self, x):\n";
    code += "        attend = lambda h: self.self_attn(h)[0]\n";
    code += "        if self.parallel:\n";
    code += "            h = self.norm1(x)\n";
    code += "            return x + self.dropout1(attend(h)) + self.dropout3(self._ffn(h))\n";
    code += "        if self.post_attn_norm is not None:\n";
    code += "            x = x + self.dropout1(self.post_attn_norm(attend(self.norm1(x))))\n";
    code += "            return x + self.dropout3(self.post_ffn_norm(self._ffn(self.norm2(x))))\n";
    code += "        if self.norm_first:\n";
    code += "            x = x + self.dropout1(attend(self.norm1(x)))\n";
    code += "            return x + self.dropout3(self._ffn(self.norm2(x)))\n";
    code += "        x = self.norm1(x + self.dropout1(attend(x)))\n";
    code += "        return self.norm2(x + self.dropout3(self._ffn(x)))\n";
    code += "\n";
    code += "class LearnedPositionalEncoding(nn.Module):\n";
    code += "    # GPT-2 learned absolute positions: x + weight[:seq_len]\n";
    code += "    def __init__(self, d_model, max_len=512):\n";
    code += "        super().__init__()\n";
    code += "        self.weight = nn.Parameter(torch.randn(max_len, d_model) * 0.02)\n";
    code += "\n";
    code += "    def forward(self, x):\n";
    code += "        return x + self.weight[:x.size(1)]\n\n";

    // Model class
    code += "class GeneratedModel(nn.Module):\n";
    code += "    def __init__(self):\n";
    code += "        super(GeneratedModel, self).__init__()\n";

    // Generate layer definitions
    int layer_idx = 0;
    int embedding_layer_idx = -1;  // for tie_embedding on a TimeDistributed head
    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;

        // Skip DatasetInput and Output nodes in __init__ (they don't have layers)
        if (node->type == NodeType::DatasetInput ||
            node->type == NodeType::Output ||
            node->type == NodeType::SequenceTagOutput) {
            continue;
        }

        if (node->type == NodeType::PluginCustom) {
            code += cyxwiz::SimulationNodeCodeComment(*node, "pytorch", "        ");
            continue;
        }
        std::string layer_code = NodeTypeToPythonLayer(*node);
        if (!layer_code.empty()) {
            code += "        self.layer" + std::to_string(layer_idx) + " = " + layer_code + "\n";
            if (node->type == NodeType::Embedding) {
                embedding_layer_idx = layer_idx;
            }
            if (node->type == NodeType::TimeDistributed && embedding_layer_idx >= 0 &&
                PythonBoolLiteral(GetParamOrDefault(*node, "tie_embedding", "false")) == "True") {
                code += "        self.layer" + std::to_string(layer_idx) + ".weight = self.layer" +
                        std::to_string(embedding_layer_idx) + ".weight  # tied input/output embeddings\n";
            }
            layer_idx++;
        }
    }

    code += "\n";

    // Word + POS fusion (the compiler's SequenceFeatureFusion route): a
    // Concatenate whose Input 1 and Input 2 are Embedding nodes. The model
    // input is the packed [batch, seq, 2] word/POS ids CyxWiz trains on.
    const auto fusion_input_embedding = [this](const MLNode& concat, size_t input_index) -> const MLNode* {
        if (concat.type != NodeType::Concatenate || concat.inputs.size() <= input_index) return nullptr;
        for (const auto& link : links_) {
            if (link.to_node != concat.id || link.to_pin != concat.inputs[input_index].id) continue;
            const MLNode* source = FindNodeById(link.from_node);
            return source && source->type == NodeType::Embedding ? source : nullptr;
        }
        return nullptr;
    };
    std::map<int, int> fusion_embedding_layer;  // Embedding node id -> layer index
    for (const auto& candidate : nodes_) {
        if (fusion_input_embedding(candidate, 0) && fusion_input_embedding(candidate, 1)) {
            fusion_embedding_layer[fusion_input_embedding(candidate, 0)->id] = -1;
            fusion_embedding_layer[fusion_input_embedding(candidate, 1)->id] = -1;
        }
    }

    // Forward pass
    code += "    def forward(self, x):\n";
    layer_idx = 0;
    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;

        if (node->type == NodeType::Embedding && fusion_embedding_layer.count(node->id)) {
            fusion_embedding_layer[node->id] = layer_idx++;  // applied at the Concatenate
            continue;
        }
        if (const MLNode* word = fusion_input_embedding(*node, 0)) {
            if (const MLNode* pos = fusion_input_embedding(*node, 1)) {
                code += "        # Word + POS ids packed as [batch, seq, 2]\n";
                code += "        x = torch.cat([self.layer" + std::to_string(fusion_embedding_layer[word->id]) +
                        "(x[..., 0].long()), self.layer" + std::to_string(fusion_embedding_layer[pos->id]) +
                        "(x[..., 1].long())], dim=-1)\n";
                continue;
            }
        }

        const auto activation = BuildActivationCodegen(
            ActivationCodegenTarget::PyTorch,
            node->type,
            node->parameters,
            "x");
        if (activation.handled) {
            code += "        x = " + activation.expression + "\n";
            continue;
        }

        switch (node->type) {
            case NodeType::DatasetInput:
                code += "        # Dataset input layer (x is already the input)\n";
                break;

            case NodeType::Dense:
            case NodeType::PReLU:
                code += "        x = self.layer" + std::to_string(layer_idx++) + "(x)\n";
                break;

            case NodeType::Dropout:
                code += "        x = F.dropout(x, p=" +
                        GetParamOrDefault(*node, "rate", "0.5") +
                        ", training=self.training)\n";
                break;

            case NodeType::Flatten:
                code += "        x = torch.flatten(x, 1)\n";
                break;

            case NodeType::Output:
            case NodeType::SequenceTagOutput:
                code += "        # Output layer\n";
                break;

            // ===== Transformer Layers =====
            case NodeType::TransformerEncoder:
                code += "        x = self.layer" + std::to_string(layer_idx++) + "(x)\n";
                break;

            case NodeType::TransformerDecoder:
                code += "        x = self.layer" + std::to_string(layer_idx++) + "(x)\n";
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

            // ===== Normalization Layers =====
            case NodeType::BatchNorm:
            case NodeType::LayerNorm:
            case NodeType::GroupNorm:
            case NodeType::InstanceNorm:
                code += "        x = self.layer" + std::to_string(layer_idx++) + "(x)\n";
                break;

            // ===== Embedding =====
            case NodeType::Embedding:
            case NodeType::TimeDistributed:
                code += "        x = self.layer" + std::to_string(layer_idx++) + "(x)\n";
                break;

            // ===== Attention =====
            case NodeType::MultiHeadAttention:
                code += "        x, _ = self.layer" + std::to_string(layer_idx++) + "(x, x, x)  # self-attention\n";
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
                code += "        signal_" + std::to_string(node->id) + " = " + start + " + np.clip(t / " + dur + ", 0.0, 1.0) * (" + end + " - " + start + ")\n";
                break;
            }
            case NodeType::SignalScope:
                code += "        # Scope: visualize connected signal\n";
                break;

            // ===== Reinforcement Learning Nodes (GenerateRLCode) =====
            case NodeType::RLEnvironment:
            case NodeType::PolicyHead:
            case NodeType::ValueHead:
            case NodeType::PPO:
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

    code += "    # Loss and optimizer (from the graph's optimizer node)\n";
    code += "    criterion = nn.CrossEntropyLoss()\n";
    code += PyTorchOptimizerSetup();

    code += "    # TODO: Add your training data here\n";
    code += "    # Example training loop:\n";
    code += "    # for epoch in range(num_epochs):\n";
    code += "    #     for batch_idx, (data, target) in enumerate(train_loader):\n";
    code += "    #         optimizer.zero_grad()\n";
    code += "    #         output = model(data)\n";
    code += "    #         loss = criterion(output, target)\n";
    code += PyTorchPenaltyLines();
    code += "    #         loss.backward()\n";
    code += PyTorchStepLines();

    return code;
}

// Training recipe in PyTorch terms, read from the first optimizer node
// (tofix112): optimizer + hyperparameters, AdamW no-decay groups, per-update
// warmup/decay LambdaLR matching WarmupDecayLR, and global-norm clipping.
const MLNode* NodeEditor::FindExportOptimizerNode() const {
    for (const auto& node : nodes_) {
        switch (node.type) {
            case NodeType::SGD: case NodeType::Adam: case NodeType::AdamW:
            case NodeType::RMSprop: case NodeType::Adagrad: case NodeType::NAdam:
                return &node;
            default:
                break;
        }
    }
    return nullptr;
}

// The scheduler node fed by the export optimizer's State output (TOFIX140
// A3), or nullptr. The compiler trains the same one.
const MLNode* NodeEditor::FindExportSchedulerNode() const {
    const MLNode* optimizer = FindExportOptimizerNode();
    if (!optimizer) return nullptr;
    for (const auto& link : links_) {
        if (link.from_node != optimizer->id) continue;
        for (const auto& node : nodes_) {
            if (node.id != link.to_node) continue;
            switch (node.type) {
                case NodeType::StepLR: case NodeType::CosineAnnealing: case NodeType::ReduceOnPlateau:
                case NodeType::ExponentialLR: case NodeType::WarmupScheduler:
                    return &node;
                default:
                    break;
            }
        }
    }
    return nullptr;
}

std::string NodeEditor::PyTorchOptimizerSetup() const {
    const MLNode* node = FindExportOptimizerNode();
    if (!node) {
        return "    optimizer = optim.Adam(model.parameters(), lr=0.001)  # no optimizer node in the graph\n\n";
    }
    const std::string lr = GetParamOrAliases(*node, {"learning_rate", "lr"}, "0.001");
    const std::string betas = "(" + GetParamOrDefault(*node, "beta1", "0.9") + ", " +
                              GetParamOrDefault(*node, "beta2", "0.999") + ")";
    const std::string eps = GetParamOrDefault(*node, "epsilon", "1e-8");
    std::string code;
    std::string params = "model.parameters()";
    switch (node->type) {
        case NodeType::SGD:
            code += "    optimizer = optim.SGD(" + params + ", lr=" + lr + ", momentum=" +
                    GetParamOrDefault(*node, "momentum", "0.9") + ")\n";
            break;
        case NodeType::AdamW: {
            const std::string decay = GetParamOrDefault(*node, "weight_decay", "0.01");
            const std::string exclude = GetParamOrDefault(*node, "weight_decay_exclude", "none");
            if (exclude != "none") {
                code += "    # weight_decay_exclude=" + exclude + ": 1-D tensors (biases, norm scales)";
                code += exclude == "norms_biases_embeddings" ? " and embedding tables skip decay\n" : " skip decay\n";
                code += "    embedding_ids = {id(m.weight) for m in model.modules() if isinstance(m, (nn.Embedding, LearnedPositionalEncoding))}\n";
                code += "    no_decay = [p for p in model.parameters() if p.ndim < 2";
                code += exclude == "norms_biases_embeddings" ? " or id(p) in embedding_ids]\n" : "]\n";
                code += "    decay = [p for p in model.parameters() if all(p is not q for q in no_decay)]\n";
                params = "[{'params': decay, 'weight_decay': " + decay + "}, {'params': no_decay, 'weight_decay': 0.0}]";
            }
            code += "    optimizer = optim.AdamW(" + params + ", lr=" + lr + ", betas=" + betas + ", eps=" + eps +
                    ", weight_decay=" + decay + ")\n";
            break;
        }
        case NodeType::RMSprop:
            code += "    optimizer = optim.RMSprop(" + params + ", lr=" + lr + ", alpha=" +
                    GetParamOrDefault(*node, "alpha", "0.99") + ", eps=" + eps + ", momentum=" +
                    GetParamOrDefault(*node, "momentum", "0.0") + ")\n";
            break;
        case NodeType::Adagrad:
            code += "    optimizer = optim.Adagrad(" + params + ", lr=" + lr + ", eps=" +
                    GetParamOrDefault(*node, "epsilon", "1e-10") + ")\n";
            break;
        case NodeType::NAdam:
            code += "    optimizer = optim.NAdam(" + params + ", lr=" + lr + ", betas=" + betas + ", eps=" + eps + ")\n";
            break;
        default:
            code += "    optimizer = optim.Adam(" + params + ", lr=" + lr + ", betas=" + betas + ", eps=" + eps + ")\n";
            break;
    }
    const std::string schedule = GetParamOrDefault(*node, "lr_schedule", "none");
    if (schedule != "none") {
        const std::string decay = schedule == "warmup_linear" ? "linear"
                                : schedule == "warmup_constant" ? "constant" : "cosine";
        code += "    # lr_schedule=" + schedule + " per optimizer update (CyxWiz WarmupDecayLR)\n";
        code += "    total_updates = 1000  # TODO: epochs * ceil(batches_per_epoch / grad_accum_steps)\n";
        code += "    warmup = round(" + GetParamOrDefault(*node, "warmup_ratio", "0.02") + " * total_updates)\n";
        code += "    min_ratio = " + GetParamOrDefault(*node, "min_lr_ratio", "0.1") + "\n";
        code += "    def lr_factor(done):  # done = completed updates; returns the factor for the next one\n";
        code += "        u = min(max(done + 1, 1), total_updates)\n";
        code += "        if u <= warmup:\n";
        code += "            return u / warmup\n";
        if (decay == "constant") {
            code += "        return 1.0\n";
        } else {
            code += "        p = (u - warmup) / max(1, total_updates - warmup)\n";
            code += decay == "linear"
                ? "        return min_ratio + (1 - min_ratio) * (1 - p)\n"
                : "        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))\n";
        }
        code += "    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_factor)\n";
    } else if (const MLNode* scheduler = FindExportSchedulerNode()) {
        const auto param = [scheduler](const char* key, const char* fallback) {
            return GetParamOrDefault(*scheduler, key, fallback);
        };
        code += "    # Scheduler node: stepped once per epoch\n";
        switch (scheduler->type) {
            case NodeType::StepLR:
                code += "    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=" + param("step_size", "10") +
                        ", gamma=" + param("gamma", "0.1") + ")\n";
                break;
            case NodeType::CosineAnnealing:
                code += "    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=" + param("T_max", "100") +
                        ", eta_min=" + param("eta_min", "0.0") + ")\n";
                break;
            case NodeType::ReduceOnPlateau:
                code += "    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=" +
                        param("factor", "0.1") + ", patience=" + param("patience", "10") + ", threshold=" +
                        param("threshold", "0.0001") + ", threshold_mode='abs', min_lr=" + param("min_lr", "0.0") +
                        ")\n";
                break;
            case NodeType::ExponentialLR:
                code += "    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=" + param("gamma", "0.95") +
                        ")\n";
                break;
            default:
                code += "    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=" +
                        param("start_factor", "0.1") + ", end_factor=1.0, total_iters=" +
                        param("warmup_epochs", "5") + ")\n";
                break;
        }
    }
    return code + "\n";
}

// The regularization node feeding the export optimizer's Loss input
// (TOFIX140 A4) as a loss term; the compiler trains the same one.
std::string NodeEditor::PyTorchPenaltyLines() const {
    const MLNode* optimizer = FindExportOptimizerNode();
    if (!optimizer) return {};
    for (const auto& link : links_) {
        if (link.to_node != optimizer->id) continue;
        for (const auto& node : nodes_) {
            if (node.id != link.from_node) continue;
            const std::string lambda = GetParamOrDefault(node, "lambda", "0.01");
            const std::string l1 = "sum(p.abs().sum() for p in model.parameters())";
            const std::string l2 = "sum(p.pow(2).sum() for p in model.parameters())";
            switch (node.type) {
                case NodeType::L1Regularization:
                    return "    #         loss = loss + " + lambda + " * " + l1 + "  # L1 Regularization node\n";
                case NodeType::L2Regularization:
                    return "    #         loss = loss + " + lambda + " * " + l2 + "  # L2 Regularization node\n";
                case NodeType::ElasticNet: {
                    const std::string ratio = GetParamOrDefault(node, "l1_ratio", "0.5");
                    return "    #         loss = loss + " + lambda + " * (" + ratio + " * " + l1 + " + (1 - " +
                           ratio + ") * " + l2 + ")  # Elastic Net node\n";
                }
                default:
                    break;
            }
        }
    }
    return {};
}

std::string NodeEditor::PyTorchStepLines() const {
    const MLNode* node = FindExportOptimizerNode();
    std::string code;
    if (node) {
        const std::string clip = GetParamOrDefault(*node, "grad_clip_norm", "0");
        if (std::strtod(clip.c_str(), nullptr) > 0.0) {
            code += "    #         torch.nn.utils.clip_grad_norm_(model.parameters(), " + clip + ")\n";
        }
    }
    code += "    #         optimizer.step()\n";
    if (node && GetParamOrDefault(*node, "lr_schedule", "none") != "none") {
        code += "    #         scheduler.step()\n";
    } else if (const MLNode* scheduler = FindExportSchedulerNode()) {
        code += scheduler->type == NodeType::ReduceOnPlateau
            ? "    #     scheduler.step(val_loss)  # after validating the epoch\n"
            : "    #     scheduler.step()  # after each epoch\n";
    }
    return code;
}

std::string NodeEditor::GenerateTensorFlowCode(const std::vector<int>& sorted_ids) {
    // An RL graph: Stable-Baselines3 (PyTorch) is the code
    if (IsRLGraph(sorted_ids)) {
        return "# RL graphs generate Stable-Baselines3 (PyTorch) code.\n\n" + GenerateRLCode(sorted_ids);
    }
    if (const auto error = FindUnsupportedSequentialLayerError(sorted_ids)) {
        return BlockedCodegenConfiguration("TensorFlow", *error);
    }
    if (const auto error = FindDenseActivationConfigurationError(sorted_ids)) {
        return BlockedCodegenConfiguration("TensorFlow", *error);
    }

    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (node && (node->type == NodeType::TransformerEncoder ||
                     node->type == NodeType::TransformerDecoder ||
                     node->type == NodeType::PositionalEncoding)) {
            return "# TensorFlow export is blocked for the current CyxWiz "
                   "Transformer/PositionalEncoding runtime contract.\n"
                   "raise NotImplementedError('Use PyTorch or PyCyxWiz export "
                   "for this graph; TensorFlow parity is not yet proven.')\n";
        }
        if (node && node->type == NodeType::LayerNorm &&
            !GetParamOrDefault(*node, "normalized_shape", "").empty()) {
            return "# TensorFlow export is blocked for explicit Studio "
                   "LayerNorm normalized_shape.\n"
                   "raise NotImplementedError('TensorFlow LayerNormalization "
                   "selects axes but does not preserve the configured Studio "
                   "shape-size contract.')\n";
        }
    }

    std::string code;

    // Header
    code += "# Auto-generated TensorFlow model from CyxWiz Node Editor\n";
    code += "# Generated at: " + ClockText("%Y-%m-%d %H:%M:%S") + "\n\n";
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
        if (node->type == NodeType::DatasetInput ||
            node->type == NodeType::Output ||
            node->type == NodeType::SequenceTagOutput) {
            continue;
        }

        if (node->type == NodeType::PluginCustom) {
            code += cyxwiz::SimulationNodeCodeComment(*node, "tensorflow", "        ");
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

        const auto activation = BuildActivationCodegen(
            ActivationCodegenTarget::TensorFlow,
            node->type,
            node->parameters,
            "x");
        if (activation.handled) {
            code += "        x = " + activation.expression + "\n";
            continue;
        }

        switch (node->type) {
            case NodeType::DatasetInput:
                code += "        # Dataset input layer (x is already the input)\n";
                break;

            case NodeType::Dense:
            case NodeType::PReLU:
                code += "        x = self.layer" + std::to_string(layer_idx++) + "(x)\n";
                break;

            case NodeType::Dropout:
                code += "        x = tf.keras.layers.Dropout(" +
                        GetParamOrDefault(*node, "rate", "0.5") +
                        ")(x, training=training)\n";
                break;

            case NodeType::BatchNorm:
            case NodeType::LayerNorm:
                code += "        x = self.layer" +
                        std::to_string(layer_idx++) +
                        "(x, training=training)\n";
                break;

            case NodeType::Flatten:
                code += "        x = tf.keras.layers.Flatten()(x)\n";
                break;

            case NodeType::Output:
            case NodeType::SequenceTagOutput:
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
    // An RL graph: Stable-Baselines3 (PyTorch) is the code
    if (IsRLGraph(sorted_ids)) {
        return "# RL graphs generate Stable-Baselines3 (PyTorch) code.\n\n" + GenerateRLCode(sorted_ids);
    }
    if (const auto error = FindUnsupportedSequentialLayerError(sorted_ids)) {
        return BlockedCodegenConfiguration("Keras", *error);
    }
    if (const auto error = FindDenseActivationConfigurationError(sorted_ids)) {
        return BlockedCodegenConfiguration("Keras", *error);
    }

    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (node && (node->type == NodeType::TransformerEncoder ||
                     node->type == NodeType::TransformerDecoder ||
                     node->type == NodeType::PositionalEncoding)) {
            return "# Keras export is blocked for the current CyxWiz "
                   "Transformer/PositionalEncoding runtime contract.\n"
                   "raise NotImplementedError('Use PyTorch or PyCyxWiz export "
                   "for this graph; Keras parity is not yet proven.')\n";
        }
        if (node && node->type == NodeType::LayerNorm &&
            !GetParamOrDefault(*node, "normalized_shape", "").empty()) {
            return "# Keras export is blocked for explicit Studio LayerNorm "
                   "normalized_shape.\n"
                   "raise NotImplementedError('Keras LayerNormalization "
                   "selects axes but does not preserve the configured Studio "
                   "shape-size contract.')\n";
        }
    }

    std::string code;

    // Header
    code += "# Auto-generated Keras model from CyxWiz Node Editor\n";
    code += "# Generated at: " + ClockText("%Y-%m-%d %H:%M:%S") + "\n\n";
    code += "import tensorflow as tf\n";
    code += "from tensorflow import keras\n";
    code += "from tensorflow.keras import layers\n\n";

    // Sequential model approach
    code += "# Build model using Sequential API\n";
    code += "model = keras.Sequential([\n";

    bool first_layer = true;
    std::string simulation_comments;  // after the layer list, outside it
    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;

        // Skip DatasetInput node
        if (node->type == NodeType::DatasetInput) {
            continue;
        }
        if (node->type == NodeType::PluginCustom) {
            simulation_comments += cyxwiz::SimulationNodeCodeComment(*node, "keras", "");
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
    if (!simulation_comments.empty()) code += simulation_comments + "\n";

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
        return GenerateRLCode(sorted_ids);
    }
    if (const auto error = FindUnsupportedSequentialLayerError(sorted_ids)) {
        return BlockedCodegenConfiguration("PyCyxWiz", *error);
    }
    if (const auto error = FindDenseActivationConfigurationError(sorted_ids)) {
        return BlockedCodegenConfiguration("PyCyxWiz", *error);
    }

    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (node && node->type == NodeType::BatchNorm) {
            return "# PyCyxWiz export is blocked for Studio BatchNorm1D.\n"
                   "raise NotImplementedError('The current cx.BatchNorm binding "
                   "is spatial BatchNorm2D; use Engine, PyTorch, or TensorFlow "
                   "until a BatchNorm1D module binding owns this contract.')\n";
        }
        if (node && node->type == NodeType::LayerNorm &&
            GetParamOrDefault(*node, "normalized_shape", "").empty()) {
            return "# PyCyxWiz export is blocked for automatic Studio "
                   "LayerNorm shape.\n"
                   "raise NotImplementedError('Set normalized_shape explicitly "
                   "or use Engine/PyTorch; PyCyxWiz has no lazy LayerNorm "
                   "constructor.')\n";
        }
    }

    std::string code;

    // Header
    code += "# Auto-generated PyCyxWiz model from CyxWiz Node Editor\n";
    code += "# Generated at: " + ClockText("%Y-%m-%d %H:%M:%S") + "\n\n";
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
        if (node->type == NodeType::DatasetInput ||
            node->type == NodeType::Output ||
            node->type == NodeType::SequenceTagOutput) {
            continue;
        }

        if (node->type == NodeType::PluginCustom) {
            code += cyxwiz::SimulationNodeCodeComment(*node, "pycyxwiz", "        ");
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

        const auto activation = BuildActivationCodegen(
            ActivationCodegenTarget::PyCyxWiz,
            node->type,
            node->parameters,
            input_expr(*node, 0));
        if (activation.handled) {
            code += "        " + out + " = " + activation.expression + "\n";
            record_output(out);
            continue;
        }

        switch (node->type) {
            case NodeType::DatasetInput:
                code += "        " + out + " = x\n";
                record_output(out);
                break;

            case NodeType::Dense:
            case NodeType::PReLU:
                code += "        " + out + " = self.layer" + std::to_string(layer_idx++) + ".forward(" + input_expr(*node, 0) + ")\n";
                record_output(out);
                break;

            case NodeType::Dropout:
                code += "        " + out + " = cx.dropout(" +
                        input_expr(*node, 0) + ", p=" +
                        GetParamOrDefault(*node, "rate", "0.5") + ")\n";
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
                // Output 1 = the first split_size entries, Output 2 = the rest.
                std::string split_size = GetParamOrDefault(*node, "split_size", "2");
                std::string dim = GetParamOrDefault(*node, "dim", "1");
                const std::string source = input_expr(*node, 0);
                code += "        " + out + " = " + source + ".split([" + split_size + ", " + source +
                        ".shape()[" + dim + "] - " + split_size + "], " + dim + ")\n";
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
                auto inputs = all_input_exprs(*node);
                std::string rhs = inputs.size() > 1
                    ? inputs[1]
                    : GetParamOrDefault(*node, "scalar", "0.0");
                std::string op = GetParamOrDefault(*node, "op", ">");
                code += "        " + out + " = " + input_expr(*node, 0) + " " + op + " " + rhs + "\n";
                record_output(out);
                break;
            }

            case NodeType::TensorLogicalMask: {
                auto inputs = all_input_exprs(*node);
                std::string op = GetParamOrDefault(*node, "op", "not");
                if ((op == "and" || op == "or") && inputs.size() > 1) {
                    const std::string py_op = op == "and" ? " & " : " | ";
                    code += "        " + out + " = (" + input_expr(*node, 0) + ")" + py_op + "(" + inputs[1] + ")\n";
                } else {
                    code += "        " + out + " = ~" + input_expr(*node, 0) + "\n";
                }
                record_output(out);
                break;
            }

            case NodeType::MultiHeadAttention:
            case NodeType::TransformerEncoder:
            case NodeType::TransformerDecoder:
            case NodeType::PositionalEncoding:
            case NodeType::LayerNorm:
            case NodeType::Embedding: {
                code += "        " + out + " = self.layer" +
                        std::to_string(layer_idx++) + ".forward(" +
                        input_expr(*node, 0) + ")\n";
                record_output(out);
                break;
            }

            case NodeType::Output:
            case NodeType::SequenceTagOutput:
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

std::string NodeEditor::InferredInputWidth(const MLNode& node) const {
    if (node.inputs.empty()) return {};
    const int input_pin = node.inputs.front().id;
    const MLNode* source = nullptr;
    for (const auto& link : links_) {
        if (link.to_node != node.id || link.to_pin != input_pin) continue;
        for (const auto& other : nodes_) {
            if (other.id == link.from_node) source = &other;
        }
    }
    if (!source) return {};
    const auto param = [source](const char* key) {
        const auto it = source->parameters.find(key);
        return it == source->parameters.end() ? std::string{} : it->second;
    };
    switch (source->type) {
        case NodeType::Embedding:
            return param("embedding_dim");
        case NodeType::LSTM:
        case NodeType::GRU:
        case NodeType::RNN: {
            const std::string hidden = param("hidden_size");
            const std::string bidirectional = param("bidirectional");
            if (hidden.empty()) return {};
            return (bidirectional == "true" || bidirectional == "True" || bidirectional == "1")
                       ? std::to_string(2 * std::atoi(hidden.c_str()))
                       : hidden;
        }
        case NodeType::Dense:
        case NodeType::TimeDistributed:
            return param("units");
        default:
            return {};
    }
}

std::string NodeEditor::NodeTypeToPythonLayer(const MLNode& node) {
    std::string code;

    if (BuildActivationCodegen(
            ActivationCodegenTarget::PyTorch,
            node.type,
            node.parameters).handled) {
        return code;
    }

    switch (node.type) {
        case NodeType::TimeDistributed: {
            // Per-token Dense head; nn.Linear acts on the last axis of [batch, seq, features].
            const std::string units = GetParamOrAliases(node, {"units", "out_features"}, "128");
            if (PythonBoolLiteral(GetParamOrDefault(node, "tie_embedding", "false")) == "True") {
                std::string width = "AUTO";
                for (const auto& other : nodes_) {
                    if (other.type == NodeType::Embedding) {
                        width = GetParamOrDefault(other, "embedding_dim", "512");
                        break;
                    }
                }
                code = "nn.Linear(" + width + ", " + units + ", bias=False)";
            } else {
                code = "nn.LazyLinear(" + units + ")";
            }
            break;
        }

        case NodeType::Dense: {
            std::string units = "64";
            auto it = node.parameters.find("units");
            if (it != node.parameters.end()) {
                units = it->second;
            }
            // The input width is read on the first forward pass (as the
            // TimeDistributed head does); a fixed guess broke every export.
            code = "nn.LazyLinear(out_features=" + units + ")";
            break;
        }

        case NodeType::Conv2D:
            code = "nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)";
            break;

        case NodeType::MaxPool2D:
            code = "nn.MaxPool2d(kernel_size=2)";
            break;

        case NodeType::BatchNorm:
            code = "nn.LazyBatchNorm1d(eps=" +
                   GetParamOrAliases(node, {"eps", "epsilon"}, "1e-5") +
                   ", momentum=" +
                   GetParamOrDefault(node, "momentum", "0.1") + ")";
            break;

        case NodeType::Dropout: {
            // The generated forward uses functional dropout so training mode
            // and the exact Studio rate remain explicit without consuming a
            // layer index.
            code = "";
            break;
        }

        case NodeType::MultiHeadAttention: {
            const std::string embed_dim = GetParamOrAliases(
                node, {"embed_dim", "d_model"}, "512");
            const std::string num_heads = GetParamOrAliases(
                node, {"num_heads", "heads"}, "8");
            const std::string dropout = GetParamOrAliases(
                node, {"dropout", "dropout_rate"}, "0.0");
            const std::string use_bias = PythonBoolLiteral(
                GetParamOrDefault(node, "use_bias", "true"));
            const std::string causal = PythonBoolLiteral(GetParamOrDefault(node, "causal", "false"));
            const std::string qk_norm = PythonBoolLiteral(GetParamOrDefault(node, "qk_norm", "false"));
            const std::string position = GetParamOrDefault(node, "position_encoding", "none");
            const std::string kv_heads = GetParamOrDefault(node, "num_kv_heads", "0");
            const std::string window = GetParamOrDefault(node, "sliding_window", "0");
            const std::string softcap = GetParamOrDefault(node, "attn_logit_softcap", "0");
            const bool plain = causal == "False" && qk_norm == "False" && position == "none" &&
                               std::strtod(kv_heads.c_str(), nullptr) == 0.0 &&
                               std::strtod(window.c_str(), nullptr) == 0.0 &&
                               std::strtod(softcap.c_str(), nullptr) == 0.0;
            if (plain) {
                code = "nn.MultiheadAttention(embed_dim=" + embed_dim +
                       ", num_heads=" + num_heads + ", dropout=" + dropout +
                       ", bias=" + use_bias + ", batch_first=True)";
            } else {
                code = "ConfigurableAttention(" + embed_dim + ", " + num_heads + ", dropout=" + dropout +
                       ", bias=" + use_bias + ", num_kv_heads=" + kv_heads + ", causal=" + causal +
                       ", qk_norm=" + qk_norm + ", position_encoding='" + position + "', rope_base=" +
                       GetParamOrDefault(node, "rope_base", "10000") + ", rope_fraction=" +
                       GetParamOrDefault(node, "rope_fraction", "1.0") + ", sliding_window=" + window +
                       ", logit_softcap=" + softcap + ")";
            }
            break;
        }

        case NodeType::LayerNorm: {
            const std::string normalized_shape =
                GetParamOrDefault(node, "normalized_shape", "AUTO");
            const std::string shape_literal = normalized_shape == "AUTO"
                ? "AUTO"
                : CsvToPythonListLiteral(normalized_shape, normalized_shape);
            code = "nn.LayerNorm(normalized_shape=" + shape_literal +
                   ", eps=" +
                   GetParamOrAliases(node, {"eps", "epsilon"}, "1e-5") +
                   ", elementwise_affine=" + PythonBoolLiteral(
                       GetParamOrDefault(node, "elementwise_affine", "true")) +
                   ")";
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

        // ===== Transformer Layers =====
        case NodeType::TransformerEncoder: {
            const std::string d_model = GetParamOrAliases(
                node, {"d_model", "embed_dim"}, "512");
            const std::string nhead = GetParamOrAliases(
                node, {"num_heads", "nhead"}, "8");
            const std::string dim_feedforward = GetParamOrAliases(
                node, {"dim_feedforward", "ff_dim", "d_ff"}, "2048");
            const std::string dropout = GetParamOrAliases(
                node, {"dropout", "dropout_rate"}, "0.1");
            const std::string norm_first = PythonBoolLiteral(
                GetParamOrDefault(node, "norm_first", "false"));
            code = "configure_ffn_dropout(nn.TransformerEncoderLayer(d_model=" + d_model + ", nhead=" + nhead +
                   ", dim_feedforward=" + dim_feedforward + ", dropout=" + dropout +
                   ", batch_first=True, norm_first=" + norm_first + "), " +
                   GetParamOrDefault(node, "ffn_dropout", "0.0") + ")";
            break;
        }

        case NodeType::TransformerDecoder: {
            const std::string d_model = GetParamOrAliases(
                node, {"d_model", "embed_dim"}, "512");
            const std::string nhead = GetParamOrAliases(
                node, {"num_heads", "nhead"}, "8");
            const std::string dim_feedforward = GetParamOrAliases(
                node, {"dim_feedforward", "ff_dim", "d_ff"}, "2048");
            const std::string dropout = GetParamOrAliases(
                node, {"dropout", "dropout_rate"}, "0.1");
            const std::string norm_first = PythonBoolLiteral(
                GetParamOrDefault(node, "norm_first", "false"));
            const std::string norm_type = GetParamOrDefault(node, "norm_type", "layer_norm");
            const std::string norm_eps = GetParamOrDefault(node, "norm_eps", "0.00001");
            const std::string ffn_type = GetParamOrDefault(node, "ffn_type", "mlp");
            const std::string ffn_activation = GetParamOrDefault(node, "ffn_activation", "relu");
            const std::string ffn_bias = PythonBoolLiteral(GetParamOrDefault(node, "ffn_bias", "true"));
            const std::string position_encoding = GetParamOrDefault(node, "position_encoding", "external");
            const std::string rope_base = GetParamOrDefault(node, "rope_base", "10000");
            const std::string attention_bias = PythonBoolLiteral(GetParamOrDefault(node, "attention_bias", "true"));
            const std::string qk_norm = PythonBoolLiteral(GetParamOrDefault(node, "qk_norm", "false"));
            const std::string rope_fraction = GetParamOrDefault(node, "rope_fraction", "1.0");
            const std::string block_layout = GetParamOrDefault(node, "block_layout", "sequential");
            const std::string sandwich_norm = PythonBoolLiteral(GetParamOrDefault(node, "sandwich_norm", "false"));
            const std::string residual_init_scale = GetParamOrDefault(node, "residual_init_scale", "1.0");
            const std::string attn_logit_softcap = GetParamOrDefault(node, "attn_logit_softcap", "0");
            const std::string sliding_window = GetParamOrDefault(node, "sliding_window", "0");
            const std::string num_kv_heads = GetParamOrDefault(node, "num_kv_heads", "0");
            const auto is_one = [](const std::string& v) { return std::strtod(v.c_str(), nullptr) == 1.0; };
            const auto is_zero = [](const std::string& v) { return std::strtod(v.c_str(), nullptr) == 0.0; };
            const bool classic = norm_type == "layer_norm" && ffn_type == "mlp" &&
                                 ffn_activation == "relu" && ffn_bias == "True" &&
                                 position_encoding == "external" &&
                                 attention_bias == "True" && qk_norm == "False" &&
                                 is_one(rope_fraction) && block_layout == "sequential" &&
                                 sandwich_norm == "False" && is_one(residual_init_scale) &&
                                 is_zero(attn_logit_softcap) && is_zero(sliding_window) && is_zero(num_kv_heads) &&
                                 (norm_eps == "0.00001" || norm_eps == "1e-5" || norm_eps == "1e-05");
            if (classic) {
                code = "CausalTransformerDecoderBlock(d_model=" + d_model + ", nhead=" + nhead +
                       ", dim_feedforward=" + dim_feedforward + ", dropout=" + dropout +
                       ", norm_first=" + norm_first + ", ffn_dropout=" +
                       GetParamOrDefault(node, "ffn_dropout", "0.0") + ")";
            } else {
                code = "ConfigurableCausalDecoderBlock(d_model=" + d_model + ", nhead=" + nhead +
                       ", dim_feedforward=" + dim_feedforward + ", dropout=" + dropout +
                       ", norm_first=" + norm_first + ", ffn_dropout=" +
                       GetParamOrDefault(node, "ffn_dropout", "0.0") + ", norm_type='" + norm_type +
                       "', norm_eps=" + norm_eps + ", ffn_type='" + ffn_type + "', ffn_activation='" +
                       ffn_activation + "', ffn_bias=" + ffn_bias + ", position_encoding='" +
                       position_encoding + "', rope_base=" + rope_base + ", attention_bias=" +
                       attention_bias + ", qk_norm=" + qk_norm + ", rope_fraction=" + rope_fraction +
                       ", block_layout='" + block_layout + "', sandwich_norm=" + sandwich_norm +
                       ", residual_init_scale=" + residual_init_scale + ", attn_logit_softcap=" +
                       attn_logit_softcap + ", sliding_window=" + sliding_window +
                       ", num_kv_heads=" + num_kv_heads + ")";
            }
            break;
        }

        case NodeType::PositionalEncoding: {
            const std::string d_model = GetParamOrAliases(
                node, {"d_model", "embed_dim"}, "512");
            const std::string max_len = GetParamOrAliases(
                node, {"max_sequence_length", "max_len", "max_length", "max_seq_len"}, "5000");
            code = GetParamOrDefault(node, "encoding_type", "sinusoidal") == "learned"
                ? "LearnedPositionalEncoding(d_model=" + d_model + ", max_len=" + max_len + ")"
                : "PositionalEncoding(d_model=" + d_model + ", max_len=" + max_len + ")";
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
            if (input_size.empty() || input_size == "0") {
                const std::string inferred = InferredInputWidth(node);
                if (!inferred.empty()) input_size = inferred;
            }
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
            if (input_size.empty() || input_size == "0") {
                const std::string inferred = InferredInputWidth(node);
                if (!inferred.empty()) input_size = inferred;
            }
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
            if (input_size.empty() || input_size == "0") {
                const std::string inferred = InferredInputWidth(node);
                if (!inferred.empty()) input_size = inferred;
            }
            it = node.parameters.find("hidden_size");
            if (it != node.parameters.end()) hidden_size = it->second;
            it = node.parameters.find("num_layers");
            if (it != node.parameters.end()) num_layers = it->second;
            it = node.parameters.find("nonlinearity");
            if (it != node.parameters.end() && !it->second.empty()) nonlinearity = it->second;
            code = "nn.RNN(input_size=" + input_size + ", hidden_size=" + hidden_size +
                   ", num_layers=" + num_layers + ", nonlinearity='" + nonlinearity +
                   "', batch_first=True)";
            break;
        }

        case NodeType::PReLU:
            code = "nn.PReLU(num_parameters=" + GetParamOrDefault(node, "num_parameters", "1") +
                   ", init=" + GetParamOrDefault(node, "init", "0.25") + ")";
            break;

        default:
            // Activation functions and others don't need layers in __init__
            code = "";
            break;
    }

    return code;
}

std::string NodeEditor::NodeTypeToTensorFlowLayer(const MLNode& node, int /*layer_idx*/) {
    std::string code;

    if (BuildActivationCodegen(
            ActivationCodegenTarget::TensorFlow,
            node.type,
            node.parameters).handled) {
        return code;
    }

    switch (node.type) {
        case NodeType::Dense: {
            std::string units = "64";
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
            code = "layers.BatchNormalization(epsilon=" +
                   GetParamOrAliases(node, {"eps", "epsilon"}, "1e-5") +
                   ", momentum=(1.0 - " +
                   GetParamOrDefault(node, "momentum", "0.1") + "))";
            break;

        case NodeType::Dropout:
            // Generated call uses the configured functional layer directly.
            code = "";
            break;

        case NodeType::MultiHeadAttention: {
            const std::string embed_dim = GetParamOrAliases(
                node, {"embed_dim", "d_model"}, "512");
            const std::string num_heads = GetParamOrAliases(
                node, {"num_heads", "heads"}, "8");
            const std::string dropout = GetParamOrAliases(
                node, {"dropout", "dropout_rate"}, "0.0");
            const std::string use_bias = PythonBoolLiteral(
                GetParamOrDefault(node, "use_bias", "true"));
            code = "layers.MultiHeadAttention(key_dim=" + embed_dim + "//" + num_heads +
                   ", num_heads=" + num_heads + ", dropout=" + dropout +
                   ", use_bias=" + use_bias + ")";
            break;
        }

        case NodeType::LayerNorm: {
            code = "layers.LayerNormalization(epsilon=" +
                   GetParamOrAliases(node, {"eps", "epsilon"}, "1e-5") +
                   ", center=" + PythonBoolLiteral(GetParamOrDefault(
                       node, "elementwise_affine", "true")) +
                   ", scale=" + PythonBoolLiteral(GetParamOrDefault(
                       node, "elementwise_affine", "true")) + ")";
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

        case NodeType::PReLU: {
            // One shared slope shares the alpha over every non-batch axis.
            const std::string init = GetParamOrDefault(node, "init", "0.25");
            code = GetParamOrDefault(node, "num_parameters", "1") == "1"
                ? "layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(" + init + "), shared_axes=[1])"
                : "layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(" + init + "))";
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

    const auto activation = BuildActivationCodegen(
        ActivationCodegenTarget::Keras,
        node.type,
        node.parameters);
    if (activation.handled) {
        return activation.expression;
    }

    switch (node.type) {
        case NodeType::Dense: {
            std::string units = "64";
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
            code = "layers.Dropout(" +
                   GetParamOrDefault(node, "rate", "0.5") + ")";
            break;

        case NodeType::BatchNorm:
            code = "layers.BatchNormalization(epsilon=" +
                   GetParamOrAliases(node, {"eps", "epsilon"}, "1e-5") +
                   ", momentum=(1.0 - " +
                   GetParamOrDefault(node, "momentum", "0.1") + "))";
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

        case NodeType::SequenceTagOutput:
            code = "# SequenceTagOutput marker";
            break;

        case NodeType::MultiHeadAttention: {
            const std::string embed_dim = GetParamOrAliases(
                node, {"embed_dim", "d_model"}, "512");
            const std::string num_heads = GetParamOrAliases(
                node, {"num_heads", "heads"}, "8");
            const std::string dropout = GetParamOrAliases(
                node, {"dropout", "dropout_rate"}, "0.0");
            const std::string use_bias = PythonBoolLiteral(
                GetParamOrDefault(node, "use_bias", "true"));
            code = "layers.MultiHeadAttention(key_dim=" + embed_dim + "//" + num_heads +
                   ", num_heads=" + num_heads + ", dropout=" + dropout +
                   ", use_bias=" + use_bias + ")";
            break;
        }

        case NodeType::LayerNorm: {
            const std::string affine = PythonBoolLiteral(
                GetParamOrDefault(node, "elementwise_affine", "true"));
            code = "layers.LayerNormalization(epsilon=" +
                   GetParamOrAliases(node, {"eps", "epsilon"}, "1e-5") +
                   ", center=" + affine + ", scale=" + affine + ")";
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

        case NodeType::PReLU: {
            // One shared slope shares the alpha over every non-batch axis.
            const std::string init = GetParamOrDefault(node, "init", "0.25");
            code = GetParamOrDefault(node, "num_parameters", "1") == "1"
                ? "layers.PReLU(alpha_initializer=keras.initializers.Constant(" + init + "), shared_axes=[1])"
                : "layers.PReLU(alpha_initializer=keras.initializers.Constant(" + init + "))";
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

    if (BuildActivationCodegen(
            ActivationCodegenTarget::PyCyxWiz,
            node.type,
            node.parameters).handled) {
        return code;
    }

    switch (node.type) {
        case NodeType::Dense: {
            std::string units = "64";
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
            // Generated forward uses cx.dropout with the exact Studio rate.
            code = "";
            break;

        // ===== Attention & Transformer Layers =====
        case NodeType::MultiHeadAttention: {
            const std::string embed_dim = GetParamOrAliases(
                node, {"embed_dim", "d_model"}, "512");
            const std::string num_heads = GetParamOrAliases(
                node, {"num_heads", "heads"}, "8");
            const std::string dropout = GetParamOrAliases(
                node, {"dropout", "dropout_rate"}, "0.0");
            const std::string use_bias = PythonBoolLiteral(
                GetParamOrDefault(node, "use_bias", "true"));
            code = "cx.AttentionModule(embed_dim=" + embed_dim +
                   ", num_heads=" + num_heads + ", dropout=" + dropout +
                   ", use_bias=" + use_bias +
                   ", causal=" + PythonBoolLiteral(GetParamOrDefault(node, "causal", "false")) +
                   ", qk_norm=" + PythonBoolLiteral(GetParamOrDefault(node, "qk_norm", "false")) +
                   ", position_encoding='" + GetParamOrDefault(node, "position_encoding", "none") +
                   "', rope_base=" + GetParamOrDefault(node, "rope_base", "10000") +
                   ", rope_fraction=" + GetParamOrDefault(node, "rope_fraction", "1.0") +
                   ", num_kv_heads=" + GetParamOrDefault(node, "num_kv_heads", "0") +
                   ", sliding_window=" + GetParamOrDefault(node, "sliding_window", "0") +
                   ", logit_softcap=" + GetParamOrDefault(node, "attn_logit_softcap", "0") + ")";
            break;
        }

        case NodeType::TransformerEncoder: {
            const std::string d_model = GetParamOrAliases(
                node, {"d_model", "embed_dim"}, "512");
            const std::string num_heads = GetParamOrAliases(
                node, {"num_heads", "nhead"}, "8");
            const std::string ff_dim = GetParamOrAliases(
                node, {"dim_feedforward", "ff_dim", "d_ff"}, "2048");
            const std::string dropout = GetParamOrAliases(
                node, {"dropout", "dropout_rate"}, "0.1");
            const std::string norm_first = PythonBoolLiteral(
                GetParamOrDefault(node, "norm_first", "false"));
            code = "cx.TransformerEncoderLayer(d_model=" + d_model +
                   ", nhead=" + num_heads + ", dim_feedforward=" + ff_dim +
                   ", dropout=" + dropout + ", norm_first=" + norm_first + ", ffn_dropout=" +
                   GetParamOrDefault(node, "ffn_dropout", "0.0") + ")";
            break;
        }

        case NodeType::TransformerDecoder: {
            const std::string d_model = GetParamOrAliases(
                node, {"d_model", "embed_dim"}, "512");
            const std::string num_heads = GetParamOrAliases(
                node, {"num_heads", "nhead"}, "8");
            const std::string ff_dim = GetParamOrAliases(
                node, {"dim_feedforward", "ff_dim", "d_ff"}, "2048");
            const std::string dropout = GetParamOrAliases(
                node, {"dropout", "dropout_rate"}, "0.1");
            const std::string norm_first = PythonBoolLiteral(
                GetParamOrDefault(node, "norm_first", "false"));
            code = "cx.TransformerDecoderLayer(d_model=" + d_model +
                   ", nhead=" + num_heads + ", dim_feedforward=" + ff_dim +
                   ", dropout=" + dropout + ", norm_first=" + norm_first + ", ffn_dropout=" +
                   GetParamOrDefault(node, "ffn_dropout", "0.0") +
                   ", norm_type='" + GetParamOrDefault(node, "norm_type", "layer_norm") +
                   "', norm_eps=" + GetParamOrDefault(node, "norm_eps", "0.00001") +
                   ", ffn_type='" + GetParamOrDefault(node, "ffn_type", "mlp") +
                   "', ffn_activation='" + GetParamOrDefault(node, "ffn_activation", "relu") +
                   "', ffn_bias=" + PythonBoolLiteral(GetParamOrDefault(node, "ffn_bias", "true")) +
                   ", position_encoding='" + GetParamOrDefault(node, "position_encoding", "external") +
                   "', rope_base=" + GetParamOrDefault(node, "rope_base", "10000") +
                   ", attention_bias=" + PythonBoolLiteral(GetParamOrDefault(node, "attention_bias", "true")) +
                   ", qk_norm=" + PythonBoolLiteral(GetParamOrDefault(node, "qk_norm", "false")) +
                   ", rope_fraction=" + GetParamOrDefault(node, "rope_fraction", "1.0") +
                   ", block_layout='" + GetParamOrDefault(node, "block_layout", "sequential") +
                   "', sandwich_norm=" + PythonBoolLiteral(GetParamOrDefault(node, "sandwich_norm", "false")) +
                   ", residual_init_scale=" + GetParamOrDefault(node, "residual_init_scale", "1.0") +
                   ", attn_logit_softcap=" + GetParamOrDefault(node, "attn_logit_softcap", "0") +
                   ", sliding_window=" + GetParamOrDefault(node, "sliding_window", "0") +
                   ", num_kv_heads=" + GetParamOrDefault(node, "num_kv_heads", "0") + ")";
            break;
        }

        // ===== Normalization Layers =====
        case NodeType::LayerNorm: {
            const std::string normalized_shape =
                GetParamOrDefault(node, "normalized_shape", "AUTO");
            const std::string shape_literal = normalized_shape == "AUTO"
                ? "[AUTO]"
                : CsvToPythonListLiteral(normalized_shape, normalized_shape);
            code = "cx.LayerNorm(normalized_shape=" + shape_literal +
                   ", eps=" +
                   GetParamOrAliases(node, {"eps", "epsilon"}, "1e-5") +
                   ", elementwise_affine=" + PythonBoolLiteral(
                       GetParamOrDefault(node, "elementwise_affine", "true")) +
                   ")";
            break;
        }

        case NodeType::GroupNorm: {
            std::string num_groups = "32";
            std::string num_channels = "256";
            std::string eps = "1e-5";
            std::string affine = "true";
            auto it = node.parameters.find("num_groups");
            if (it != node.parameters.end()) num_groups = it->second;
            it = node.parameters.find("num_channels");
            if (it != node.parameters.end()) num_channels = it->second;
            it = node.parameters.find("eps");
            if (it != node.parameters.end()) eps = it->second;
            it = node.parameters.find("affine");
            if (it != node.parameters.end()) affine = it->second;
            code = "cx.GroupNorm(num_groups=" + num_groups +
                   ", num_channels=" + num_channels +
                   ", eps=" + eps + ", affine=" +
                   (affine == "true" ? "True" : "False") + ")";
            break;
        }

        case NodeType::InstanceNorm: {
            std::string num_features = "64";
            std::string eps = "1e-5";
            std::string affine = "false";
            auto it = node.parameters.find("num_features");
            if (it != node.parameters.end()) num_features = it->second;
            it = node.parameters.find("eps");
            if (it != node.parameters.end()) eps = it->second;
            it = node.parameters.find("affine");
            if (it != node.parameters.end()) affine = it->second;
            code = "cx.InstanceNorm2D(num_features=" + num_features +
                   ", eps=" + eps + ", affine=" +
                   (affine == "true" ? "True" : "False") + ")";
            break;
        }

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
            const std::string d_model = GetParamOrAliases(
                node, {"d_model", "embed_dim"}, "512");
            const std::string max_len = GetParamOrAliases(
                node, {"max_sequence_length", "max_len", "max_length", "max_seq_len"}, "5000");
            code = std::string(GetParamOrDefault(node, "encoding_type", "sinusoidal") == "learned"
                                   ? "cx.LearnedPositionalEmbedding(d_model="
                                   : "cx.PositionalEncoding(d_model=") + d_model +
                   ", max_sequence_length=" + max_len + ")";
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
            if (it != node.parameters.end() && !it->second.empty()) nonlinearity = it->second;
            code = "cx.RNN(input_size=" + input_size +
                   ", hidden_size=" + hidden_size +
                   ", num_layers=" + num_layers +
                   ", batch_first=True, bidirectional=False, nonlinearity='" +
                   nonlinearity + "')";
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

        case NodeType::PReLU:
            code = "cx.PReLU(num_parameters=" + GetParamOrDefault(node, "num_parameters", "1") +
                   ", init=" + GetParamOrDefault(node, "init", "0.25") + ")";
            break;

        default:
            // Other node types handled in forward pass or not yet implemented
            code = "";
            break;
    }

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
