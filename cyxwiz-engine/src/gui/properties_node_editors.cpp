#include "properties_node_editors.h"
#include "node_editor.h"
#include "properties_truth.h"
#include <imgui.h>
#include <implot.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
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

void RenderPathLine(const char* label, const std::string& value) {
    ImGui::TextUnformatted(label);
    ImGui::SameLine(120.0f);
    if (value.empty()) {
        ImGui::TextDisabled("<not set>");
    } else {
        ImGui::TextWrapped("%s", value.c_str());
    }
}

bool RenderTextParameter(MLNode& node,
                         const char* key,
                         const char* label,
                         const char* fallback = "",
                         ImGuiInputTextFlags flags = 0,
                         bool create_default = true) {
    auto existing = node.parameters.find(key);
    if (existing == node.parameters.end() && create_default) {
        existing = node.parameters.emplace(key, fallback).first;
    }

    const std::string value =
        existing != node.parameters.end() ? existing->second : fallback;
    char buffer[256] = {};
    strncpy(buffer, value.c_str(), sizeof(buffer) - 1);

    ImGui::Text("%s:", label);
    ImGui::SameLine(150.0f);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 10.0f);
    const std::string imgui_id = std::string("##") + key;
    if (ImGui::InputText(imgui_id.c_str(), buffer, sizeof(buffer), flags)) {
        node.parameters[key] = buffer;
        return true;
    }
    return false;
}

bool RenderBoolParameter(MLNode& node,
                         const char* key,
                         const char* label,
                         bool fallback) {
    std::string& value = node.parameters[key];
    if (value.empty()) {
        value = fallback ? "true" : "false";
    }
    bool enabled = value == "true";
    if (ImGui::Checkbox(label, &enabled)) {
        value = enabled ? "true" : "false";
        return true;
    }
    return false;
}

bool RenderEnumParameter(MLNode& node,
                         const char* key,
                         const char* label,
                         const char* const* values,
                         int value_count,
                         const char* fallback) {
    std::string& value = node.parameters[key];
    if (value.empty()) {
        value = fallback;
    }
    int current = 0;
    for (int i = 0; i < value_count; ++i) {
        if (value == values[i]) {
            current = i;
            break;
        }
    }

    ImGui::Text("%s:", label);
    ImGui::SameLine(150.0f);
    ImGui::SetNextItemWidth(180.0f);
    const std::string imgui_id = std::string("##") + key;
    if (ImGui::Combo(imgui_id.c_str(), &current, values, value_count)) {
        value = values[current];
        return true;
    }
    return false;
}

bool RenderFloatParameter(MLNode& node,
                          const char* key,
                          const char* label,
                          const char* fallback,
                          float min_value = 0.0f) {
    std::string& value = node.parameters[key];
    if (value.empty()) {
        value = fallback;
    }
    float parsed = min_value;
    try {
        parsed = std::stof(value);
    } catch (...) {
        parsed = std::stof(fallback);
    }
    if (!std::isfinite(parsed) || parsed < min_value) {
        parsed = std::stof(fallback);
    }

    ImGui::Text("%s:", label);
    ImGui::SameLine(150.0f);
    ImGui::SetNextItemWidth(120.0f);
    const std::string imgui_id = std::string("##") + key;
    if (ImGui::InputFloat(imgui_id.c_str(), &parsed, 0.0f, 0.0f, "%.4f")) {
        if (parsed < min_value) {
            parsed = min_value;
        }
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "%.6g", parsed);
        value = buffer;
        return true;
    }
    return false;
}

bool ParseNonNegativeFloatVector(const std::string& raw,
                                 std::vector<float>& values,
                                 std::string& error) {
    std::string text = raw;
    for (char& c : text) {
        if (c == '[' || c == ']' || c == '(' || c == ')' ||
            c == ',' || c == ';') {
            c = ' ';
        }
    }

    values.clear();
    std::istringstream in(text);
    std::string token;
    while (in >> token) {
        try {
            size_t parsed = 0;
            const float weight = std::stof(token, &parsed);
            if (parsed != token.size() || !std::isfinite(weight) ||
                weight < 0.0f) {
                throw std::runtime_error("invalid weight");
            }
            values.push_back(weight);
        } catch (...) {
            error = "invalid non-negative number '" + token + "'";
            return false;
        }
    }

    if (values.empty()) {
        error = "enter at least one class weight";
        return false;
    }
    return true;
}

size_t ParsePositiveSizeOrZero(const std::string& value) {
    try {
        size_t parsed = 0;
        const size_t count = std::stoul(value, &parsed);
        if (parsed == value.size() && count > 0) {
            return count;
        }
    } catch (...) {
    }
    return 0;
}

size_t FindExpectedClassCount(const RenderNodePropertiesContext& context) {
    if (!context.node_editor) {
        return 0;
    }

    for (const auto& graph_node : context.node_editor->GetNodes()) {
        if (graph_node.type != NodeType::Output) {
            continue;
        }
        auto it = graph_node.parameters.find("num_classes");
        if (it != graph_node.parameters.end()) {
            const size_t count = ParsePositiveSizeOrZero(it->second);
            if (count > 0) {
                return count;
            }
        }
    }
    return 0;
}

void RenderClassWeightsValidation(const MLNode& node,
                                  const RenderNodePropertiesContext& context) {
    const std::string weights = ParamOr(node, "class_weights", "");
    std::vector<float> parsed_weights;
    std::string error;
    if (!ParseNonNegativeFloatVector(weights, parsed_weights, error)) {
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f),
                           "  Invalid weight vector: %s", error.c_str());
        return;
    }

    const size_t expected_classes = FindExpectedClassCount(context);
    if (expected_classes > 0 && parsed_weights.size() != expected_classes) {
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f),
                           "  Weight count %zu does not match expected class count %zu.",
                           parsed_weights.size(), expected_classes);
        return;
    }

    if (expected_classes > 0) {
        ImGui::TextDisabled("  %zu weights parsed. Expected class count: %zu.",
                            parsed_weights.size(), expected_classes);
    } else {
        ImGui::TextDisabled("  %zu weights parsed. Expected class count is unknown until compile.",
                            parsed_weights.size());
    }
}

void RenderLossReduction(MLNode& node) {
    static const char* reductions[] = {"mean", "sum", "none"};
    RenderEnumParameter(node, "reduction", "Reduction",
                        reductions, IM_ARRAYSIZE(reductions), "mean");
}

void RenderSimpleLossProperties(MLNode& node,
                                const RenderNodePropertiesContext& context) {
    ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "%s", node.name.c_str());
    ImGui::Separator();
    ImGui::Spacing();

    RenderLossReduction(node);

    switch (node.type) {
        case NodeType::CrossEntropyLoss: {
            RenderTextParameter(node, "ignore_index", "Ignore index", "-100",
                                ImGuiInputTextFlags_CharsDecimal);
            RenderFloatParameter(node, "label_smoothing", "Label smoothing",
                                 "0.0", 0.0f);
            ImGui::TextDisabled("  Must be less than 1.0. Smooths hard class labels.");
            static const char* weight_modes[] = {"none", "manual", "balanced"};
            RenderEnumParameter(node, "class_weight", "Class weights",
                                weight_modes, IM_ARRAYSIZE(weight_modes),
                                "none");
            const std::string mode = ParamOr(node, "class_weight", "none");
            if (mode == "manual") {
                RenderTextParameter(node, "class_weights", "Weight vector", "");
                ImGui::TextDisabled("  Example: [1.0, 2.5, 1.0]. Length must match output classes.");
                RenderClassWeightsValidation(node, context);
            } else if (mode == "balanced") {
                ImGui::TextDisabled("  Auto-computed from Arrow/text train labels when supported.");
                ImGui::TextDisabled("  Unsupported dataset paths fall back to unweighted loss.");
            }
            break;
        }
        case NodeType::BCEWithLogits:
            RenderFloatParameter(node, "pos_weight", "Positive weight",
                                 "1.0", 0.000001f);
            ImGui::TextDisabled("  Scales positive examples for imbalanced binary labels.");
            break;
        case NodeType::FocalLoss:
            RenderFloatParameter(node, "alpha", "Alpha", "0.25", 0.0f);
            RenderFloatParameter(node, "gamma", "Gamma", "2.0", 0.0f);
            break;
        case NodeType::SoftDiceLoss:
            RenderFloatParameter(node, "smooth", "Smooth", "1.0", 0.0f);
            ImGui::TextDisabled("  Expects probability masks and same-shaped Float32 targets.");
            break;
        case NodeType::TverskyLoss:
            RenderFloatParameter(node, "alpha", "Alpha", "0.5", 0.0f);
            RenderFloatParameter(node, "beta", "Beta", "0.5", 0.0f);
            RenderFloatParameter(node, "smooth", "Smooth", "1.0", 0.0f);
            ImGui::TextDisabled("  Alpha penalizes false positives; beta penalizes false negatives.");
            break;
        case NodeType::JaccardLoss:
            RenderFloatParameter(node, "smooth", "Smooth", "1.0", 0.0f);
            ImGui::TextDisabled("  IoU-style overlap loss for same-shaped Float32 masks.");
            break;
        case NodeType::SmoothL1Loss:
        case NodeType::HuberLoss:
            RenderFloatParameter(node, "beta", "Beta", "1.0", 0.000001f);
            break;
        case NodeType::NLLLoss:
            RenderTextParameter(node, "ignore_index", "Ignore index", "-100",
                                ImGuiInputTextFlags_CharsDecimal);
            break;
        default:
            break;
    }
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
    // Render editable parameters based on node type
    switch (node.type) {
        case NodeType::Dense:
        case NodeType::TimeDistributed: {
            // Units
            std::string& units = node.parameters["units"];
            if (units.empty()) units = "64";
            char u_buffer[16];
            strncpy(u_buffer, units.c_str(), sizeof(u_buffer) - 1);
            u_buffer[sizeof(u_buffer) - 1] = '\0';

            ImGui::Text("Units:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##units", u_buffer, sizeof(u_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                units = u_buffer;
                context.invalidate_shapes();
            }

            ImGui::Spacing();

            if (node.type == NodeType::TimeDistributed) {
                break;
            }

            // Activation function
            std::string& activation = node.parameters["activation"];
            if (activation.empty()) activation = "relu";

            const char* activations[] = { "none", "relu", "sigmoid", "tanh", "softmax", "leaky_relu" };
            int current_activation = 0;
            for (int i = 0; i < 6; i++) {
                if (activation == activations[i]) {
                    current_activation = i;
                    break;
                }
            }

            ImGui::Text("Activation:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(150.0f);
            if (ImGui::Combo("##activation", &current_activation, activations, 6)) {
                activation = activations[current_activation];
            }
            break;
        }

        case NodeType::Conv2D: {
            // Filters
            std::string& filters = node.parameters["filters"];
            if (filters.empty()) filters = "32";
            char f_buffer[16];
            strncpy(f_buffer, filters.c_str(), sizeof(f_buffer) - 1);
            f_buffer[sizeof(f_buffer) - 1] = '\0';

            ImGui::Text("Filters:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##filters", f_buffer, sizeof(f_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                filters = f_buffer;
                context.invalidate_shapes();
            }

            ImGui::Spacing();

            // Kernel Size
            std::string& kernel = node.parameters["kernel_size"];
            if (kernel.empty()) kernel = "3";
            char k_buffer[16];
            strncpy(k_buffer, kernel.c_str(), sizeof(k_buffer) - 1);
            k_buffer[sizeof(k_buffer) - 1] = '\0';

            ImGui::Text("Kernel Size:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##kernel", k_buffer, sizeof(k_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                kernel = k_buffer;
                context.invalidate_shapes();
            }

            ImGui::Spacing();

            // Stride
            std::string& stride = node.parameters["stride"];
            if (stride.empty()) stride = "1";
            char s_buffer[16];
            strncpy(s_buffer, stride.c_str(), sizeof(s_buffer) - 1);
            s_buffer[sizeof(s_buffer) - 1] = '\0';

            ImGui::Text("Stride:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##stride", s_buffer, sizeof(s_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                stride = s_buffer;
                context.invalidate_shapes();
            }

            ImGui::Spacing();

            // Padding
            std::string& padding = node.parameters["padding"];
            if (padding.empty()) padding = "same";

            const char* paddings[] = { "same", "valid" };
            int current_padding = (padding == "valid") ? 1 : 0;

            ImGui::Text("Padding:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(150.0f);
            if (ImGui::Combo("##padding", &current_padding, paddings, 2)) {
                padding = paddings[current_padding];
                context.invalidate_shapes();
            }

            ImGui::Spacing();

            // Activation function
            std::string& activation = node.parameters["activation"];
            if (activation.empty()) activation = "relu";

            const char* activations[] = { "none", "relu", "sigmoid", "tanh", "softmax", "leaky_relu" };
            int current_activation = 0;
            for (int i = 0; i < 6; i++) {
                if (activation == activations[i]) {
                    current_activation = i;
                    break;
                }
            }

            ImGui::Text("Activation:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(150.0f);
            if (ImGui::Combo("##activation_conv", &current_activation, activations, 6)) {
                activation = activations[current_activation];
            }
            break;
        }

        case NodeType::MaxPool2D: {
            // Pool Size
            std::string& pool_size = node.parameters["pool_size"];
            if (pool_size.empty()) pool_size = "2";
            char p_buffer[16];
            strncpy(p_buffer, pool_size.c_str(), sizeof(p_buffer) - 1);
            p_buffer[sizeof(p_buffer) - 1] = '\0';

            ImGui::Text("Pool Size:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##pool_size", p_buffer, sizeof(p_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                pool_size = p_buffer;
                context.invalidate_shapes();
            }

            ImGui::Spacing();

            // Stride
            std::string& stride = node.parameters["stride"];
            if (stride.empty()) stride = "2";
            char s_buffer[16];
            strncpy(s_buffer, stride.c_str(), sizeof(s_buffer) - 1);
            s_buffer[sizeof(s_buffer) - 1] = '\0';

            ImGui::Text("Stride:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##stride_pool", s_buffer, sizeof(s_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                stride = s_buffer;
                context.invalidate_shapes();
            }
            break;
        }

        case NodeType::Dropout: {
            std::string& rate_str = node.parameters["rate"];
            if (rate_str.empty()) rate_str = "0.5";

            float rate = std::stof(rate_str);
            ImGui::Text("Drop Rate:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##rate", &rate, 0.0f, 0.9f, "%.2f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.2f", rate);
                rate_str = buf;
            }
            break;
        }

        case NodeType::BatchNorm: {
            // Momentum
            std::string& momentum_str = node.parameters["momentum"];
            if (momentum_str.empty()) momentum_str = "0.99";

            float momentum = std::stof(momentum_str);
            ImGui::Text("Momentum:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##momentum", &momentum, 0.0f, 1.0f, "%.3f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.3f", momentum);
                momentum_str = buf;
            }

            ImGui::Spacing();

            // Epsilon
            std::string& epsilon_str = node.parameters["epsilon"];
            if (epsilon_str.empty()) epsilon_str = "0.001";

            float epsilon = std::stof(epsilon_str);
            ImGui::Text("Epsilon:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##epsilon", &epsilon, 0.0001f, 0.01f, "%.4f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.4f", epsilon);
                epsilon_str = buf;
            }
            break;
        }

        case NodeType::Output: {
            std::string classes = ParamOr(
                node,
                "num_classes",
                ParamOr(node, "classes", "10").c_str());
            char c_buffer[16];
            strncpy(c_buffer, classes.c_str(), sizeof(c_buffer) - 1);
            c_buffer[sizeof(c_buffer) - 1] = '\0';

            ImGui::Text("Classes:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##classes", c_buffer, sizeof(c_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                properties_truth::WriteCanonicalAndAliases(
                    node, "num_classes", c_buffer);
                context.invalidate_shapes();
            }
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Number of output classes");
            break;
        }

        case NodeType::SequenceTagOutput: {
            ImGui::TextColored(ImVec4(0.45f, 0.9f, 0.85f, 1.0f),
                               "Sequence tag output");
            ImGui::TextColored(ImVec4(0.65f, 0.65f, 0.65f, 1.0f),
                               "Declares token-level logits and BIO decode metadata.");
            ImGui::Separator();
            ImGui::Spacing();

            bool shape_changed = false;
            shape_changed |= RenderTextParameter(
                node, "num_tags", "Number of tags", "0",
                ImGuiInputTextFlags_CharsDecimal);
            RenderTextParameter(node, "tag_vocab_file", "Tag vocabulary", "");

            std::string& decode_scheme = node.parameters["decode_scheme"];
            if (decode_scheme.empty()) {
                decode_scheme = "BIO";
            }
            const char* decode_schemes[] = {"BIO"};
            int current_scheme = 0;
            ImGui::Text("Decode scheme:");
            ImGui::SameLine(150.0f);
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::Combo("##decode_scheme", &current_scheme,
                             decode_schemes, 1)) {
                decode_scheme = decode_schemes[current_scheme];
            }

            if (shape_changed) {
                context.invalidate_shapes();
            }

            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.65f, 0.65f, 0.65f, 1.0f),
                               "Uses num_tags for CrossEntropy class-count validation.");
            break;
        }

        case NodeType::NERSequenceBuilder: {
            ImGui::TextColored(ImVec4(0.45f, 0.9f, 0.85f, 1.0f),
                               "NER sequence materializer");
            ImGui::TextColored(ImVec4(0.65f, 0.65f, 0.65f, 1.0f),
                               "Consumes sentence-level string-list columns and emits padded id tensors.");
            ImGui::Separator();
            ImGui::Spacing();

            RenderTextParameter(node, "token_column", "Token column", "tokens");
            RenderTextParameter(node, "pos_column", "POS column", "");
            RenderTextParameter(node, "tag_column", "Tag column", "ner_tags");
            RenderTextParameter(node, "sentence_id_column", "Sentence id", "");

            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.65f, 0.65f, 0.65f, 1.0f),
                               "Required at launch: token and tag columns. POS and sentence id are optional.");
            ImGui::Spacing();

            bool shape_changed = false;
            shape_changed |= RenderTextParameter(
                node, "max_sequence_length", "Max sequence length", "0",
                ImGuiInputTextFlags_CharsDecimal);
            RenderTextParameter(node, "ignore_index", "Padding label", "-100");
            RenderBoolParameter(node, "create_attention_mask",
                                "Create attention mask", true);

            if (shape_changed) {
                context.invalidate_shapes();
            }

            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.65f, 0.65f, 0.65f, 1.0f),
                               "Outputs: word_ids, pos_ids, tag_ids, attention_mask, sequence_length.");
            break;
        }

        case NodeType::TokenVocabulary:
        case NodeType::POSVocabulary:
        case NodeType::NERTagVocabulary: {
            const bool is_token = node.type == NodeType::TokenVocabulary;
            const bool is_pos = node.type == NodeType::POSVocabulary;
            const bool is_tag = node.type == NodeType::NERTagVocabulary;
            const char* title = is_token ? "Token vocabulary"
                              : (is_pos ? "POS vocabulary"
                                        : "NER tag vocabulary");
            const char* default_column = is_token ? "tokens"
                                       : (is_pos ? "pos_tags" : "ner_tags");

            ImGui::TextColored(ImVec4(0.45f, 0.9f, 0.85f, 1.0f), "%s", title);
            ImGui::TextColored(ImVec4(0.65f, 0.65f, 0.65f, 1.0f),
                               "Builds a deterministic value,id table from one sequence column.");
            ImGui::Separator();
            ImGui::Spacing();

            RenderTextParameter(node, "column", "Source column", default_column);
            RenderTextParameter(node, "min_freq", "Minimum frequency", "1",
                                ImGuiInputTextFlags_CharsDecimal);
            RenderTextParameter(node, "max_vocab_size", "Max vocab size", "0",
                                ImGuiInputTextFlags_CharsDecimal);
            RenderTextParameter(node, "vocab_file", "Vocabulary file", "", 0,
                                false);

            ImGui::Spacing();
            if (is_tag) {
                RenderTextParameter(node, "outside_tag", "Outside tag", "O");
                RenderTextParameter(node, "bio_scheme", "Tag scheme", "BIO");
                ImGui::TextColored(ImVec4(0.65f, 0.65f, 0.65f, 1.0f),
                                   "BIO tags are ordered deterministically with the outside tag first.");
            } else {
                RenderBoolParameter(node, "lowercase", "Lowercase values",
                                    is_token);
                RenderTextParameter(node, "pad_token", "Padding token", "[PAD]");
                RenderTextParameter(node, "unk_token", "Unknown token", "[UNK]");
                ImGui::TextColored(ImVec4(0.65f, 0.65f, 0.65f, 1.0f),
                                   "Padding and unknown tokens are reserved before observed values.");
            }
            break;
        }

        // ========== Data Pipeline Nodes ==========

        case NodeType::DatasetInput:
        case NodeType::DataLoader:
        case NodeType::Augmentation:
        case NodeType::DataSplit:
            RenderDataPipelineNodeProperties(node, context);
            break;
        case NodeType::TensorReshape: {
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "Reshape Node");
            ImGui::Separator();
            ImGui::Spacing();

            // Shape
            std::string& shape = node.parameters["shape"];
            if (shape.empty()) shape = "-1,28,28,1";
            char shape_buffer[64];
            strncpy(shape_buffer, shape.c_str(), sizeof(shape_buffer) - 1);
            shape_buffer[sizeof(shape_buffer) - 1] = '\0';

            ImGui::Text("Target Shape:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::InputText("##reshape", shape_buffer, sizeof(shape_buffer))) {
                shape = shape_buffer;
                context.invalidate_shapes();
            }
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Use -1 for batch dimension");
            break;
        }

        case NodeType::Normalize: {
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "Normalize Node");
            ImGui::Separator();
            ImGui::Spacing();

            // Mean
            std::string& mean_str = node.parameters["mean"];
            if (mean_str.empty()) mean_str = "0.0";
            char mean_buffer[32];
            strncpy(mean_buffer, mean_str.c_str(), sizeof(mean_buffer) - 1);
            mean_buffer[sizeof(mean_buffer) - 1] = '\0';

            ImGui::Text("Mean:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(150.0f);
            if (ImGui::InputText("##mean", mean_buffer, sizeof(mean_buffer))) {
                mean_str = mean_buffer;
            }

            ImGui::Spacing();

            // Std
            std::string& std_str = node.parameters["std"];
            if (std_str.empty()) std_str = "1.0";
            char std_buffer[32];
            strncpy(std_buffer, std_str.c_str(), sizeof(std_buffer) - 1);
            std_buffer[sizeof(std_buffer) - 1] = '\0';

            ImGui::Text("Standard Deviation:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(150.0f);
            if (ImGui::InputText("##std", std_buffer, sizeof(std_buffer))) {
                std_str = std_buffer;
            }

            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Common values:");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "  MNIST: mean=0.1307, std=0.3081");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "  ImageNet: mean=0.485,0.456,0.406");
            break;
        }

        case NodeType::OneHotEncode: {
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "One-Hot Encode Node");
            ImGui::Separator();
            ImGui::Spacing();

            // Num classes
            std::string& num_classes = node.parameters["num_classes"];
            if (num_classes.empty()) num_classes = "10";
            char classes_buffer[16];
            strncpy(classes_buffer, num_classes.c_str(), sizeof(classes_buffer) - 1);
            classes_buffer[sizeof(classes_buffer) - 1] = '\0';

            ImGui::Text("Number of Classes:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##num_classes", classes_buffer, sizeof(classes_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                num_classes = classes_buffer;
                context.invalidate_shapes();
            }
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "MNIST=10, CIFAR-10=10, ImageNet=1000");
            break;
        }

        // ========== Activation Functions ==========
        case NodeType::ReLU:
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "ReLU Activation");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "f(x) = max(0, x)");
            break;

        case NodeType::Sigmoid:
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Sigmoid Activation");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "f(x) = 1 / (1 + exp(-x))");
            break;

        case NodeType::Tanh:
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Tanh Activation");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "f(x) = tanh(x)");
            break;

        case NodeType::Softmax:
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Softmax Activation");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "f(x_i) = exp(x_i) / sum(exp(x))");
            break;

        case NodeType::LeakyReLU: {
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Leaky ReLU Activation");

            std::string& slope_str = node.parameters["negative_slope"];
            if (slope_str.empty()) slope_str = "0.01";
            float slope = std::stof(slope_str);

            ImGui::Text("Negative Slope:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##neg_slope", &slope, 0.001f, 0.3f, "%.3f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.3f", slope);
                slope_str = buf;
            }
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "f(x) = max(slope*x, x)");
            break;
        }

        // ========== Loss Functions ==========
        case NodeType::MSELoss:
        case NodeType::CrossEntropyLoss:
        case NodeType::FocalLoss:
        case NodeType::SoftDiceLoss:
        case NodeType::TverskyLoss:
        case NodeType::JaccardLoss:
        case NodeType::BCELoss:
        case NodeType::BCEWithLogits:
        case NodeType::L1Loss:
        case NodeType::SmoothL1Loss:
        case NodeType::HuberLoss:
        case NodeType::NLLLoss:
            RenderSimpleLossProperties(node, context);
            break;

        // ========== Optimizers ==========
        case NodeType::SGD: {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 1.0f, 1.0f), "SGD Optimizer");
            ImGui::Separator();
            ImGui::Spacing();

            std::string& lr_str = node.parameters["learning_rate"];
            if (lr_str.empty()) lr_str = "0.01";
            float lr = std::stof(lr_str);

            ImGui::Text("Learning Rate:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##lr_sgd", &lr, 0.0001f, 1.0f, "%.4f", ImGuiSliderFlags_Logarithmic)) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.4f", lr);
                lr_str = buf;
            }

            std::string& momentum_str = node.parameters["momentum"];
            if (momentum_str.empty()) momentum_str = "0.9";
            float momentum = std::stof(momentum_str);

            ImGui::Text("Momentum:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##momentum_sgd", &momentum, 0.0f, 0.99f, "%.2f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.2f", momentum);
                momentum_str = buf;
            }
            break;
        }

        case NodeType::Adam: {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 1.0f, 1.0f), "Adam Optimizer");
            ImGui::Separator();
            ImGui::Spacing();

            std::string& lr_str = node.parameters["learning_rate"];
            if (lr_str.empty()) lr_str = "0.001";
            float lr = std::stof(lr_str);

            ImGui::Text("Learning Rate:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##lr_adam", &lr, 0.00001f, 0.1f, "%.5f", ImGuiSliderFlags_Logarithmic)) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.5f", lr);
                lr_str = buf;
            }

            std::string& beta1_str = node.parameters["beta1"];
            if (beta1_str.empty()) beta1_str = "0.9";
            float beta1 = std::stof(beta1_str);

            ImGui::Text("Beta1:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##beta1", &beta1, 0.0f, 0.999f, "%.3f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.3f", beta1);
                beta1_str = buf;
            }

            std::string& beta2_str = node.parameters["beta2"];
            if (beta2_str.empty()) beta2_str = "0.999";
            float beta2 = std::stof(beta2_str);

            ImGui::Text("Beta2:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##beta2", &beta2, 0.0f, 0.9999f, "%.4f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.4f", beta2);
                beta2_str = buf;
            }
            break;
        }

        case NodeType::AdamW: {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 1.0f, 1.0f), "AdamW Optimizer");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Adam with decoupled weight decay");
            ImGui::Separator();
            ImGui::Spacing();

            std::string& lr_str = node.parameters["learning_rate"];
            if (lr_str.empty()) lr_str = "0.001";
            float lr = std::stof(lr_str);

            ImGui::Text("Learning Rate:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##lr_adamw", &lr, 0.00001f, 0.1f, "%.5f", ImGuiSliderFlags_Logarithmic)) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.5f", lr);
                lr_str = buf;
            }

            std::string& beta1_str = node.parameters["beta1"];
            if (beta1_str.empty()) beta1_str = "0.9";
            float beta1 = std::stof(beta1_str);

            ImGui::Text("Beta1:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##beta1_w", &beta1, 0.0f, 0.999f, "%.3f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.3f", beta1);
                beta1_str = buf;
            }

            std::string& beta2_str = node.parameters["beta2"];
            if (beta2_str.empty()) beta2_str = "0.999";
            float beta2 = std::stof(beta2_str);

            ImGui::Text("Beta2:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##beta2_w", &beta2, 0.0f, 0.9999f, "%.4f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.4f", beta2);
                beta2_str = buf;
            }

            std::string& wd_str = node.parameters["weight_decay"];
            if (wd_str.empty()) wd_str = "0.01";
            float wd = std::stof(wd_str);

            ImGui::Text("Weight Decay:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##wd", &wd, 0.0f, 0.1f, "%.4f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.4f", wd);
                wd_str = buf;
            }
            break;
        }

        case NodeType::Flatten:
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "Flatten Layer");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Flattens input to 1D vector");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "[H, W, C] -> [H * W * C]");
            break;

        case NodeType::SignalSlider: {
            ImGui::TextColored(ImVec4(0.0f, 0.9f, 0.8f, 1.0f), "Signal Slider");
            ImGui::Spacing();

            std::string& val_str = node.parameters["value"];
            std::string& min_str = node.parameters["min"];
            std::string& max_str = node.parameters["max"];
            float val = std::stof(val_str.empty() ? "0" : val_str);
            float mn = std::stof(min_str.empty() ? "-1" : min_str);
            float mx = std::stof(max_str.empty() ? "1" : max_str);

            ImGui::Text("Value:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##slider_val", &val, mn, mx)) {
                char buf[32]; snprintf(buf, sizeof(buf), "%.4f", val);
                val_str = buf;
            }

            ImGui::Text("Range:");
            ImGui::SetNextItemWidth(90.0f);
            if (ImGui::InputFloat("##slider_min", &mn, 0, 0, "%.2f")) {
                char buf[32]; snprintf(buf, sizeof(buf), "%.2f", mn);
                min_str = buf;
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth(90.0f);
            if (ImGui::InputFloat("##slider_max", &mx, 0, 0, "%.2f")) {
                char buf[32]; snprintf(buf, sizeof(buf), "%.2f", mx);
                max_str = buf;
            }
            break;
        }

        case NodeType::SineWave: {
            ImGui::TextColored(ImVec4(0.0f, 0.9f, 0.8f, 1.0f), "Sine Wave Generator");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "A*sin(2*pi*f*t + phase) + offset");
            ImGui::Spacing();

            auto floatParam = [&](const char* label, const char* key, float step = 0.1f) {
                std::string& s = node.parameters[key];
                float v = std::stof(s.empty() ? "0" : s);
                ImGui::Text("%s:", label);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(120.0f);
                std::string id = std::string("##sine_") + key;
                if (ImGui::InputFloat(id.c_str(), &v, step, step * 10, "%.3f")) {
                    char buf[32]; snprintf(buf, sizeof(buf), "%.3f", v);
                    s = buf;
                }
            };
            floatParam("Amplitude", "amplitude");
            floatParam("Frequency", "frequency");
            floatParam("Phase", "phase");
            floatParam("Offset", "offset");
            break;
        }

        case NodeType::StepSignal: {
            ImGui::TextColored(ImVec4(0.0f, 0.9f, 0.8f, 1.0f), "Step Signal");
            ImGui::Spacing();

            auto floatParam = [&](const char* label, const char* key) {
                std::string& s = node.parameters[key];
                float v = std::stof(s.empty() ? "0" : s);
                ImGui::Text("%s:", label);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(120.0f);
                std::string id = std::string("##step_") + key;
                if (ImGui::InputFloat(id.c_str(), &v, 0.1f, 1.0f, "%.3f")) {
                    char buf[32]; snprintf(buf, sizeof(buf), "%.3f", v);
                    s = buf;
                }
            };
            floatParam("Step Time", "step_time");
            floatParam("Initial Value", "initial_value");
            floatParam("Final Value", "final_value");
            break;
        }

        case NodeType::RampSignal: {
            ImGui::TextColored(ImVec4(0.0f, 0.9f, 0.8f, 1.0f), "Ramp Signal");
            ImGui::Spacing();

            auto floatParam = [&](const char* label, const char* key) {
                std::string& s = node.parameters[key];
                float v = std::stof(s.empty() ? "0" : s);
                ImGui::Text("%s:", label);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(120.0f);
                std::string id = std::string("##ramp_") + key;
                if (ImGui::InputFloat(id.c_str(), &v, 0.1f, 1.0f, "%.3f")) {
                    char buf[32]; snprintf(buf, sizeof(buf), "%.3f", v);
                    s = buf;
                }
            };
            floatParam("Start Value", "start_value");
            floatParam("End Value", "end_value");
            floatParam("Duration", "duration");
            break;
        }

        case NodeType::SignalScope: {
            ImGui::TextColored(ImVec4(0.0f, 0.9f, 0.8f, 1.0f), "Signal Scope");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Plots incoming signal values in real-time");
            ImGui::Spacing();

            std::string& ws = node.parameters["window_size"];
            int win = std::stoi(ws.empty() ? "500" : ws);
            ImGui::Text("Window Size:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputInt("##scope_win", &win)) {
                if (win < 10) win = 10;
                ws = std::to_string(win);
            }

            std::string& as = node.parameters["auto_scale"];
            bool auto_s = (as == "true");
            if (ImGui::Checkbox("Auto Scale", &auto_s)) {
                as = auto_s ? "true" : "false";
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Real-time signal plot
            auto& buf = context.scope_buffers[node.id];
            buf.max_samples = win;

            // Generate demo data when no simulation is running
            // (will be replaced by real simulation data when connected)
            context.scope_demo_time += ImGui::GetIO().DeltaTime;
            buf.Push(context.scope_demo_time, std::sin(2.0f * 3.14159f * 0.5f * context.scope_demo_time));

            if (!buf.times.empty()) {
                // Copy deque to contiguous arrays for ImPlot
                std::vector<float> t_arr(buf.times.begin(), buf.times.end());
                std::vector<float> v_arr(buf.values.begin(), buf.values.end());

                ImPlotFlags plot_flags = ImPlotFlags_NoTitle;
                if (ImPlot::BeginPlot("##scope_plot", ImVec2(-1, 200), plot_flags)) {
                    ImPlotAxisFlags x_flags = ImPlotAxisFlags_NoLabel;
                    ImPlotAxisFlags y_flags = auto_s ? (ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_NoLabel) : ImPlotAxisFlags_NoLabel;
                    ImPlot::SetupAxes("Time (s)", "Value", x_flags, y_flags);

                    // Auto-scroll X axis to follow latest data
                    if (!t_arr.empty()) {
                        float t_max = t_arr.back();
                        float t_window = win * 0.016f;  // Approximate window in seconds
                        if (t_window < 2.0f) t_window = 2.0f;
                        ImPlot::SetupAxisLimits(ImAxis_X1, t_max - t_window, t_max, ImGuiCond_Always);
                    }

                    ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.0f, 0.9f, 0.8f, 1.0f));
                    ImPlot::PlotLine("Signal", t_arr.data(), v_arr.data(), static_cast<int>(t_arr.size()));
                    ImPlot::PopStyleColor();
                    ImPlot::EndPlot();
                }
            }

            // Controls
            if (ImGui::Button("Clear")) {
                buf.Clear();
                context.scope_demo_time = 0.0f;
            }
            ImGui::SameLine();
            ImGui::TextDisabled("Samples: %d", static_cast<int>(buf.times.size()));

            break;
        }

        case NodeType::PluginCustom:
            RenderPluginCustomNodeProperties(node, context);
            break;
        // ========== Smart I/O Nodes (Dialog-only configuration) ==========
        case NodeType::DataInput:
        case NodeType::DataOutput:
            // These nodes are configured via the Open Dialog button only
            ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "Use 'Open Dialog' to configure");
            break;

        case NodeType::DataConvert: {
            const std::string status = ParamOr(node, "status", "Not run");
            const std::string output = ParamOr(
                node, "converted_output_path",
                ParamOr(node, "parquet_output_path",
                        ParamOr(node, "output_path").c_str()).c_str());
            const std::string manifest = ParamOr(node, "manifest_path");
            const std::string rows = ParamOr(node, "rows_written", "0");

            ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f),
                               "Use 'Open Dialog' to configure and run");
            ImGui::Spacing();
            RenderPathLine("Source:", ParamOr(node, "input_path"));
            RenderPathLine("Output:", output);
            if (!manifest.empty()) {
                RenderPathLine("Manifest:", manifest);
            }
            ImGui::Text("Rows written:");
            ImGui::SameLine(120.0f);
            ImGui::TextUnformatted(rows.c_str());
            ImGui::Text("Status:");
            ImGui::SameLine(120.0f);
            ImGui::TextWrapped("%s", status.c_str());
            break;
        }

        default: {
            // Generic parameter editor for nodes that don't have a
            // custom case above (e.g. the new Image Transform nodes).
            // Renders each parameter as an editable text field. Nodes
            // with no parameters at all show a "no parameters" hint.
            if (node.parameters.empty()) {
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                                   "No editable parameters for this node type");
            } else {
                for (auto& [key, value] : node.parameters) {
                    char buf[256] = {};
                    strncpy(buf, value.c_str(), sizeof(buf) - 1);

                    ImGui::Text("%s:", key.c_str());
                    ImGui::SameLine(140);
                    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 10);
                    std::string label = "##param_" + key;
                    if (ImGui::InputText(label.c_str(), buf, sizeof(buf))) {
                        value = buf;
                    }
                }
            }
            break;
        }
    }
}
} // namespace gui::properties_node_editors
