// Properties panel: the per-node editors (TOFIX129 A7). Only the node types
// whose metadata says Custom come here (the NER vocabularies and sequence
// nodes, the simulation signals) plus the fallback for types with no
// metadata parameters (activations, Flatten, Reshape, Augmentation, plugin
// nodes, and the generic key/value editor). Every row goes through
// properties_rows so it carries its truth chip and the shared look.
#include "properties_node_editors.h"
#include "node_editor.h"
#include "properties_rows.h"
#include "ui_buttons.h"
#include "ui_tokens.h"
#include <imgui.h>
#include <implot.h>
#include <algorithm>
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

bool SliderRow(MLNode& node, const char* key, const char* label, float fallback, float min_value, float max_value,
               const char* format = "%.2f") {
    float v = std::clamp(ParseFiniteFloatOr(ParamOr(node, key), fallback), min_value, max_value);
    properties_rows::Label(label);
    ImGui::PushID(key);
    const bool changed = ImGui::SliderFloat("##v", &v, min_value, max_value, format);
    if (changed) {
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
            }
            properties_rows::Status("window_size");
            properties_rows::Label("Auto scale");
            if (ImGui::Checkbox("##scope_auto", &auto_scale)) node.parameters["auto_scale"] = auto_scale ? "true" : "false";
            properties_rows::Status("auto_scale");
        }
    }

    // Real-time signal plot
    auto& buf = context.scope_buffers[node.id];
    buf.max_samples = win;

    float live_value = 0.0f;
    const bool has_live_value =
        context.node_editor &&
        context.node_editor->IsGraphSimulationRunning() &&
        node.inputs.size() == 1 &&
        context.node_editor->TryGetSimulationScalar(node.inputs[0].id, live_value);
    if (has_live_value) {
        const float sample_time = context.node_editor->GetSimulationTime();
        if (buf.times.empty() || sample_time > buf.times.back()) buf.Push(sample_time, live_value);
    } else if (buf.times.empty()) {
        Note("Connect one scalar signal and run the simulation to view data.");
    }

    if (!buf.times.empty()) {
        // Copy deque to contiguous arrays for ImPlot
        std::vector<float> t_arr(buf.times.begin(), buf.times.end());
        std::vector<float> v_arr(buf.values.begin(), buf.values.end());
        if (ImPlot::BeginPlot("##scope_plot", ImVec2(-1, 200), ImPlotFlags_NoTitle)) {
            ImPlotAxisFlags x_flags = ImPlotAxisFlags_NoLabel;
            ImPlotAxisFlags y_flags = auto_scale ? (ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_NoLabel) : ImPlotAxisFlags_NoLabel;
            ImPlot::SetupAxes("Time (s)", "Value", x_flags, y_flags);
            // Auto-scroll X axis to follow latest data
            const float t_max = t_arr.back();
            float t_window = t_arr.size() > 1 ? t_arr.back() - t_arr.front() : 2.0f;
            if (t_window < 2.0f) t_window = 2.0f;
            ImPlot::SetupAxisLimits(ImAxis_X1, t_max - t_window, t_max, ImGuiCond_Always);
            ImPlot::PushStyleColor(ImPlotCol_Line, t.series[0]);
            ImPlot::PlotLine("Signal", t_arr.data(), v_arr.data(), static_cast<int>(t_arr.size()));
            ImPlot::PopStyleColor();
            ImPlot::EndPlot();
        }
    }
    if (cyxwiz::ui::SecondaryButton("Clear")) buf.Clear();
    ImGui::SameLine();
    ImGui::TextColored(t.text_dim, "Samples: %d", static_cast<int>(buf.times.size()));
}

}  // namespace

void ScopeBuffer::Push(float t, float v) {
    times.push_back(t);
    values.push_back(v);
    while (static_cast<int>(times.size()) > max_samples) {
        times.pop_front();
        values.pop_front();
    }
}

void ScopeBuffer::Clear() {
    times.clear();
    values.clear();
}

void RenderNodeProperties(MLNode& node, RenderNodePropertiesContext context) {
    const auto publish_simulation_parameter =
        [&](const char* key) {
            if (!context.node_editor) return;
            const auto value = node.parameters.find(key);
            if (value != node.parameters.end()) {
                context.node_editor->SetSimulationNodeParameter(
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

        case NodeType::Augmentation: {
            properties_rows::Rows rows("##rows");
            if (!rows.ok) break;
            edited |= TextRow(node, "transforms", "Transforms", "RandomFlip,Normalize");
            properties_rows::Note("Comma-separated list");
            edited |= SliderRow(node, "flip_prob", "Flip probability", 0.5f, 0.0f, 1.0f);
            edited |= TextRow(node, "normalize_mean", "Normalize mean", "0.0");
            edited |= TextRow(node, "normalize_std", "Normalize std", "1.0");
            break;
        }

        case NodeType::TensorReshape: {
            properties_rows::Rows rows("##rows");
            if (!rows.ok) break;
            edited |= TextRow(node, "shape", "Target shape", "-1,28,28,1");
            properties_rows::Note("Use -1 for the batch dimension");
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
            if (ImGui::SliderFloat("##slider_val", &val, mn, mx)) {
                store("value", val, "%.4f");
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
