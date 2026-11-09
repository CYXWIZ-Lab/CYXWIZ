// Properties panel: the per-node editors (TOFIX129 A7). Only the node types
// whose metadata says Custom come here (the NER vocabularies and sequence
// nodes, the simulation signals) plus the fallback for types with no
// metadata parameters (activations, Flatten, Reshape, Augmentation, plugin
// nodes, and the generic key/value editor). Every row goes through
// properties_rows so it carries its truth chip and the shared look.
#include "properties_node_editors.h"
#include "node_editor.h"
#include "../core/rl_graph_presentation.h"
#include "../plugin/builtin/classic_control_environments.h"
#include "../plugin/plugin_manager.h"
#include "../plugin/registries/plugin_environment_registry.h"
#include "properties_rows.h"
#include "simulation_scope_plot.h"
#include "ui_buttons.h"
#include "ui_widgets.h"
#include "ui_tokens.h"
#include <imgui.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace gui::properties_node_editors {
namespace {

std::string ParamOr(const MLNode& node,
                    const char* key,
                    const char* fallback = "") {
    auto it = node.parameters.find(key);
    return it != node.parameters.end() ? it->second : fallback;
}

float ParseFiniteFloatOr(const std::string& text, float fallback) {
    try {
        size_t consumed = 0;
        const float value = std::stof(text, &consumed);
        return consumed == text.size() && std::isfinite(value)
                   ? value
                   : fallback;
    } catch (...) {
        return fallback;
    }
}

int ParseIntOr(const std::string& text, int fallback) {
    try {
        size_t consumed = 0;
        const int value = std::stoi(text, &consumed);
        return consumed == text.size() ? value : fallback;
    } catch (...) {
        return fallback;
    }
}

// The rows below show the stored value, or `fallback` when the node has
// none, and write the node only on a change (displaying never writes).

bool TextRow(MLNode& node,
             const char* key,
             const char* label,
             const char* fallback = "",
             ImGuiInputTextFlags flags = 0) {
    const std::string value = ParamOr(node, key, fallback);
    char buffer[256] = {};
    std::strncpy(buffer, value.c_str(), sizeof(buffer) - 1);
    properties_rows::Label(label);
    ImGui::PushID(key);
    const bool changed = ImGui::InputText("##v", buffer, sizeof(buffer), flags);
    if (changed) node.parameters[key] = buffer;
    ImGui::PopID();
    properties_rows::Status(key);
    return changed;
}

bool BoolRow(MLNode& node, const char* key, const char* label, bool fallback) {
    const std::string value = ParamOr(node, key, fallback ? "true" : "false");
    bool enabled = value == "true";
    properties_rows::Label(label);
    ImGui::PushID(key);
    const bool changed = ImGui::Checkbox("##v", &enabled);
    if (changed) node.parameters[key] = enabled ? "true" : "false";
    ImGui::PopID();
    properties_rows::Status(key);
    return changed;
}

bool EnumRow(MLNode& node,
             const char* key,
             const char* label,
             const char* const* values,
             int value_count,
             const char* fallback) {
    const std::string value = ParamOr(node, key, fallback);
    int current = 0;
    for (int i = 0; i < value_count; ++i) {
        if (value == values[i]) current = i;
    }
    properties_rows::Label(label);
    ImGui::PushID(key);
    const bool changed = ImGui::Combo("##v", &current, values, value_count);
    if (changed) node.parameters[key] = values[current];
    ImGui::PopID();
    properties_rows::Status(key);
    return changed;
}

// InputFloat with `min_value` as the floor; `format` is the stored precision.
bool FloatRow(MLNode& node,
              const char* key,
              const char* label,
              float fallback,
              float min_value = -INFINITY,
              float step = 0.1f,
              const char* format = "%.3f") {
    float v = std::max(ParseFiniteFloatOr(ParamOr(node, key), fallback), min_value);
    properties_rows::Label(label);
    ImGui::PushID(key);
    const bool changed = ImGui::InputFloat("##v", &v, step, step * 10.0f, format);
    if (changed) {
        v = std::max(v, min_value);
        char buf[32];
        std::snprintf(buf, sizeof(buf), format, v);
        node.parameters[key] = buf;
    }
    ImGui::PopID();
    properties_rows::Status(key);
    return changed;
}


void Note(const char* text) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim, "%s", text);
    ImGui::PopTextWrapPos();
}

// A one-line description of a node type with no settings (an activation).
void Formula(const char* what, const char* formula) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    ImGui::TextUnformatted(what);
    ImGui::TextColored(t.text_dim, "%s", formula);
}

void RenderSignalScope(MLNode& node, RenderNodePropertiesContext& context) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    int win = std::clamp(ParseIntOr(ParamOr(node, "window_size"), 500), 10, 100000);
    bool auto_scale = ParamOr(node, "auto_scale") == "true";
    {
        properties_rows::Rows rows("##scope");
        if (rows.ok) {
            properties_rows::Label("Window size");
            if (ImGui::InputInt("##scope_win", &win)) {
                win = std::clamp(win, 10, 100000);
                node.parameters["window_size"] = std::to_string(win);
                if (context.node_editor)
                    context.node_editor->GetSimulationSession().SetNodeParameter(node.id, "window_size",
                                                                                 node.parameters["window_size"]);
            }
            properties_rows::Status("window_size");
            properties_rows::Label("Auto scale");
            if (ImGui::Checkbox("##scope_auto", &auto_scale)) node.parameters["auto_scale"] = auto_scale ? "true" : "false";
            properties_rows::Status("auto_scale");
        }
    }

    // The run's history of this Scope, collected by the simulation session
    // whether or not this panel is drawn.
    std::vector<float> t_arr;
    std::vector<float> v_arr;
    if (context.node_editor) context.node_editor->GetSimulationSession().CopyScope(node.id, t_arr, v_arr);
    if (t_arr.empty()) Note("Connect one scalar signal and run the simulation to view data.");

    DrawScopeTrace("##scope_plot", "Signal", t_arr, v_arr, ImVec2(-1, 200), auto_scale, t.series[0]);
    if (cyxwiz::ui::SecondaryButton("Clear") && context.node_editor)
        context.node_editor->GetSimulationSession().ClearScope(node.id);
    ImGui::SameLine();
    ImGui::TextColored(t.text_dim, "Samples: %d", static_cast<int>(t_arr.size()));
}

// ========== Reinforcement learning (TOFIX195 S4.3) ==========

std::vector<cyxwiz::rl_graph::ListedEnvironment> ListedEnvironments() {
    std::vector<cyxwiz::rl_graph::ListedEnvironment> out;
    auto& plugins = cyxwiz::plugin::PluginManager::Instance();
    for (const auto& e : cyxwiz::plugin::PluginEnvironmentRegistry::Instance().Environments()) {
        std::string provider = "CyxWiz (built in)";
        if (e.plugin_id != cyxwiz::plugin::kBuiltinEnvironmentProviderId) {
            const auto* loaded = plugins.GetLoadedPlugin(e.plugin_id);
            provider = loaded && !loaded->manifest.name.empty() ? loaded->manifest.name : e.plugin_id;
        }
        out.push_back({e.key, provider, e.info});
    }
    return out;
}

void FactRows(const std::vector<cyxwiz::rl_graph::Fact>& facts) {
    for (const auto& f : facts) properties_rows::ReadOnly(f.label.c_str(), f.value, "");
}

// Environment: the environment list, its facts, the copies and the plugin's
// own settings (stored as setting.<key>)
bool RenderEnvironment(MLNode& node) {
    using cyxwiz::rl_graph::kSettingPrefix;
    bool edited = false;
    const std::string current = ParamOr(node, "environment", "cyxwiz.builtin:CartPole-v1");
    const auto listed = ListedEnvironments();
    const auto choices = cyxwiz::rl_graph::EnvironmentChoices(listed, current);
    properties_rows::Rows rows("##rl_env");
    if (!rows.ok) return false;
    properties_rows::Label("Environment", true);
    std::string shown = current;
    for (const auto& c : choices)
        if (c.key == current) shown = c.label;
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::BeginCombo("##environment", shown.c_str())) {
        for (const auto& c : choices) {
            if (ImGui::Selectable(c.label.c_str(), c.key == current) && c.key != current) {
                node.parameters["environment"] = c.key;
                for (auto it = node.parameters.begin(); it != node.parameters.end();)
                    it = it->first.rfind(kSettingPrefix, 0) == 0 ? node.parameters.erase(it) : std::next(it);
                edited = true;
            }
        }
        ImGui::EndCombo();
    }
    properties_rows::Status("environment");

    const cyxwiz::rl_graph::ListedEnvironment* env = nullptr;
    for (const auto& e : listed)
        if (e.key == ParamOr(node, "environment", current.c_str())) env = &e;
    int copies = std::clamp(ParseIntOr(ParamOr(node, "copies", "8"), 8), 1, 1024);
    properties_rows::Label("Copies", true, "Copies stepped side by side; one rollout is copies x steps per copy");
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::InputInt("##copies", &copies)) {
        node.parameters["copies"] = std::to_string(std::clamp(copies, 1, 1024));
        edited = true;
    }
    properties_rows::Status("copies");
    if (!env) {
        properties_rows::Note("No loaded plugin offers this environment: load its plugin in Plugin Manager.");
        return edited;
    }
    FactRows(cyxwiz::rl_graph::EnvironmentFacts(env->info));
    for (const auto& s : env->info.settings) {
        const std::string key = kSettingPrefix + s.key;
        const std::string value = ParamOr(node, key.c_str(), s.default_value.c_str());
        char* end = nullptr;
        const double number = std::strtod(value.c_str(), &end);
        properties_rows::Label(s.label.c_str(), false, s.description.c_str());
        ImGui::PushID(key.c_str());
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (!value.empty() && end && *end == '\0') {
            double v = number;
            const double step = std::floor(number) == number ? 1.0 : 0.01;
            if (ImGui::InputDouble("##v", &v, step, step * 10.0, "%.6g")) {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.6g", v);
                node.parameters[key] = buf;
                edited = true;
            }
        } else {
            char buf[256] = {};
            std::strncpy(buf, value.c_str(), sizeof(buf) - 1);
            if (ImGui::InputText("##v", buf, sizeof(buf))) {
                node.parameters[key] = buf;
                edited = true;
            }
        }
        ImGui::PopID();
        properties_rows::Status(key.c_str());
    }
    return edited;
}

}  // namespace

void RenderRlNodeFacts(MLNode& node, RenderNodePropertiesContext context) {
    if (!context.node_editor) return;
    auto& editor = *context.node_editor;
    const auto& nodes = editor.GetNodes();
    const auto& links = editor.GetLinks();
    const MLNode* env_node = cyxwiz::rl_graph::UpstreamEnvironment(nodes, links, node.id);
    cyxwiz::plugin::PluginEnvironmentRegistry::Environment env;
    const bool known = env_node && cyxwiz::plugin::PluginEnvironmentRegistry::Instance().Find(
                                       ParamOr(*env_node, "environment"), env);
    properties_rows::Rows rows("##rl_facts");
    if (!rows.ok) return;
    if (node.type == NodeType::ValueHead) {
        FactRows(cyxwiz::rl_graph::ValueHeadFacts(cyxwiz::rl_graph::UpstreamUnits(nodes, links, node.id)));
        return;
    }
    if (!known) {
        properties_rows::Note(env_node ? "Its Environment is not offered by a loaded plugin."
                                       : "Connect it from an Environment's Observation through layers to see what "
                                         "it adds.");
        return;
    }
    if (node.type == NodeType::PolicyHead) {
        FactRows(cyxwiz::rl_graph::PolicyHeadFacts(env.info, cyxwiz::rl_graph::UpstreamUnits(nodes, links, node.id)));
        return;
    }
    // PPO: the tuned settings for its environment
    const auto preset = cyxwiz::rl_graph::TunedPpoSettings(env.info.id);
    if (!preset) return;
    properties_rows::Label("Tuned for", false, "Load tuned settings for this environment");
    if (cyxwiz::ui::SecondaryButton(env.info.id.c_str())) {
        for (const auto& [k, v] : preset->ppo) node.parameters[k] = v;
        if (MLNode* e = editor.GetNodeForConfiguration(env_node->id))
            for (const auto& [k, v] : preset->environment) e->parameters[k] = v;
        for (const auto& l : links) {
            if (l.to_node != node.id) continue;
            MLNode* head = editor.GetNodeForConfiguration(l.from_node);
            if (head && head->type == NodeType::PolicyHead)
                for (const auto& [k, v] : preset->policy_head) head->parameters[k] = v;
        }
        if (context.invalidate_shapes) context.invalidate_shapes();
    }
    cyxwiz::ui::Tooltip((preset->source + (preset->layers_note.empty() ? "" : ". " + preset->layers_note)).c_str());
    properties_rows::Status("");
}

void RenderNodeProperties(MLNode& node, RenderNodePropertiesContext context) {
    const auto publish_simulation_parameter =
        [&](const char* key) {
            if (!context.node_editor) return;
            const auto value = node.parameters.find(key);
            if (value != node.parameters.end()) {
                context.node_editor->GetSimulationSession().SetNodeParameter(
                    node.id, key, value->second);
            }
        };
    bool edited = false;

    switch (node.type) {
        case NodeType::SequenceTagOutput: {
            Note("Declares token-level logits and BIO decode metadata.");
            properties_rows::Rows rows("##rows");
            if (!rows.ok) break;
            edited |= TextRow(node, "num_tags", "Number of tags", "0", ImGuiInputTextFlags_CharsDecimal);
            edited |= TextRow(node, "tag_vocab_file", "Tag vocabulary", "");
            static const char* decode_schemes[] = {"BIO"};
            edited |= EnumRow(node, "decode_scheme", "Decode scheme", decode_schemes, 1, "BIO");
            properties_rows::Note("Uses the number of tags for CrossEntropy class-count validation.");
            break;
        }

        case NodeType::NERSequenceBuilder: {
            Note("Consumes sentence-level string-list columns and emits padded id tensors.");
            properties_rows::Rows rows("##rows");
            if (!rows.ok) break;
            edited |= TextRow(node, "token_column", "Token column", "tokens");
            edited |= TextRow(node, "pos_column", "POS column", "");
            edited |= TextRow(node, "tag_column", "Tag column", "ner_tags");
            edited |= TextRow(node, "sentence_id_column", "Sentence id", "");
            properties_rows::Note("Required at launch: token and tag columns. POS and sentence id are optional.");
            edited |= TextRow(node, "max_sequence_length", "Max sequence length", "0", ImGuiInputTextFlags_CharsDecimal);
            edited |= TextRow(node, "ignore_index", "Padding label", "-100");
            edited |= BoolRow(node, "create_attention_mask", "Create attention mask", true);
            properties_rows::Note("Outputs: word_ids, pos_ids, tag_ids, attention_mask, sequence_length.");
            break;
        }

        case NodeType::TokenVocabulary:
        case NodeType::POSVocabulary:
        case NodeType::NERTagVocabulary: {
            const bool is_token = node.type == NodeType::TokenVocabulary;
            const bool is_tag = node.type == NodeType::NERTagVocabulary;
            const char* default_column = is_token ? "tokens" : (is_tag ? "ner_tags" : "pos_tags");
            Note("Builds a deterministic value,id table from one sequence column.");
            properties_rows::Rows rows("##rows");
            if (!rows.ok) break;
            edited |= TextRow(node, "column", "Source column", default_column);
            edited |= TextRow(node, "min_freq", "Minimum frequency", "1", ImGuiInputTextFlags_CharsDecimal);
            edited |= TextRow(node, "max_vocab_size", "Max vocab size", "0", ImGuiInputTextFlags_CharsDecimal);
            edited |= TextRow(node, "vocab_file", "Vocabulary file", "");
            if (is_tag) {
                edited |= TextRow(node, "outside_tag", "Outside tag", "O");
                edited |= TextRow(node, "bio_scheme", "Tag scheme", "BIO");
                properties_rows::Note("BIO tags are ordered deterministically with the outside tag first.");
            } else {
                edited |= BoolRow(node, "lowercase", "Lowercase values", is_token);
                edited |= TextRow(node, "pad_token", "Padding token", "[PAD]");
                edited |= TextRow(node, "unk_token", "Unknown token", "[UNK]");
                properties_rows::Note("Padding and unknown tokens are reserved before observed values.");
            }
            break;
        }

        // ========== Activation Functions ==========
        case NodeType::ReLU:
            Formula("ReLU activation", "f(x) = max(0, x)");
            break;
        case NodeType::Sigmoid:
            Formula("Sigmoid activation", "f(x) = 1 / (1 + exp(-x))");
            break;
        case NodeType::Tanh:
            Formula("Tanh activation", "f(x) = tanh(x)");
            break;
        case NodeType::Softmax:
            Formula("Softmax activation", "f(x_i) = exp(x_i) / sum(exp(x))");
            break;
        case NodeType::Flatten:
            Formula("Flattens the input to one vector", "[H, W, C] -> [H * W * C]");
            break;

        // ========== Simulation signals ==========
        case NodeType::SignalSlider: {
            float val = ParseFiniteFloatOr(ParamOr(node, "value"), 0.0f);
            float mn = ParseFiniteFloatOr(ParamOr(node, "min"), -1.0f);
            float mx = ParseFiniteFloatOr(ParamOr(node, "max"), 1.0f);
            if (mn > mx) std::swap(mn, mx);
            val = std::clamp(val, mn, mx);
            const auto store = [&](const char* key, float v, const char* format) {
                char buf[32];
                std::snprintf(buf, sizeof(buf), format, v);
                node.parameters[key] = buf;
                publish_simulation_parameter(key);
            };
            properties_rows::Rows rows("##rows");
            if (!rows.ok) break;
            properties_rows::Label("Value");
            // A typed field with - / + a hundredth of the range (owner
            // 2026-10-21: no sliders), kept in the range.
            const float step = mx > mn ? (mx - mn) / 100.0f : 0.01f;
            ImGui::SetNextItemWidth(-FLT_MIN);
            if (ImGui::InputFloat("##slider_val", &val, step, step * 10.0f, "%.4g")) {
                store("value", std::clamp(val, mn, mx), "%.6g");
                edited = true;
            }
            properties_rows::Status("value");
            properties_rows::Label("Range");
            const float half = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
            ImGui::SetNextItemWidth(half);
            if (ImGui::InputFloat("##slider_min", &mn, 0, 0, "%.2f")) {
                if (mn > mx) {
                    mx = mn;
                    store("max", mx, "%.2f");
                }
                store("min", mn, "%.2f");
                store("value", std::clamp(val, mn, mx), "%.4f");
                edited = true;
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth(half);
            if (ImGui::InputFloat("##slider_max", &mx, 0, 0, "%.2f")) {
                if (mx < mn) {
                    mn = mx;
                    store("min", mn, "%.2f");
                }
                store("max", mx, "%.2f");
                store("value", std::clamp(val, mn, mx), "%.4f");
                edited = true;
            }
            properties_rows::Status("min");
            break;
        }

        case NodeType::SineWave: {
            Note("A*sin(2*pi*f*t + phase) + offset");
            properties_rows::Rows rows("##rows");
            if (!rows.ok) break;
            for (const auto& [label, key] : {std::pair{"Amplitude", "amplitude"}, std::pair{"Frequency", "frequency"},
                                             std::pair{"Phase", "phase"}, std::pair{"Offset", "offset"}}) {
                if (FloatRow(node, key, label, 0.0f)) {
                    publish_simulation_parameter(key);
                    edited = true;
                }
            }
            break;
        }

        case NodeType::StepSignal: {
            properties_rows::Rows rows("##rows");
            if (!rows.ok) break;
            if (FloatRow(node, "step_time", "Step time", 0.0f, 0.0f)) { publish_simulation_parameter("step_time"); edited = true; }
            if (FloatRow(node, "initial_value", "Initial value", 0.0f)) { publish_simulation_parameter("initial_value"); edited = true; }
            if (FloatRow(node, "final_value", "Final value", 0.0f)) { publish_simulation_parameter("final_value"); edited = true; }
            break;
        }

        case NodeType::RampSignal: {
            properties_rows::Rows rows("##rows");
            if (!rows.ok) break;
            if (FloatRow(node, "start_value", "Start value", 0.0f)) { publish_simulation_parameter("start_value"); edited = true; }
            if (FloatRow(node, "end_value", "End value", 0.0f)) { publish_simulation_parameter("end_value"); edited = true; }
            if (FloatRow(node, "duration", "Duration", 0.0f, 0.001f)) { publish_simulation_parameter("duration"); edited = true; }
            break;
        }

        case NodeType::SignalScope:
            Note("Plots incoming signal values in real time");
            RenderSignalScope(node, context);
            break;

        case NodeType::PluginCustom:
            RenderPluginCustomNodeProperties(node, context);
            break;

        case NodeType::RLEnvironment:
            edited |= RenderEnvironment(node);
            break;

        case NodeType::ValueHead:
            RenderRlNodeFacts(node, context);
            break;

        default: {
            // Generic editor for a type with no metadata parameters and no
            // case above: one text row per stored key.
            if (node.parameters.empty()) {
                Note("No editable settings for this node type");
                break;
            }
            properties_rows::Rows rows("##rows");
            if (!rows.ok) break;
            std::vector<std::string> keys;
            for (const auto& [key, value] : node.parameters) keys.push_back(key);
            for (const auto& key : keys) {
                edited |= TextRow(node, key.c_str(), key.c_str(), "");
            }
            break;
        }
    }
    if (edited && context.invalidate_shapes) context.invalidate_shapes();
}
} // namespace gui::properties_node_editors
