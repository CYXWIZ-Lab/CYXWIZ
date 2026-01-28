#include "json_tree.h"
#include <imgui.h>
#include <spdlog/spdlog.h>

namespace gui {

bool JSONTreeRenderer::SetContent(const std::string& json_str) {
    try {
        json_ = nlohmann::json::parse(json_str);
        is_valid_ = true;
        error_.clear();
        return true;
    } catch (const nlohmann::json::parse_error& e) {
        is_valid_ = false;
        error_ = e.what();
        return false;
    }
}

void JSONTreeRenderer::SetJSON(const nlohmann::json& json) {
    json_ = json;
    is_valid_ = true;
    error_.clear();
}

void JSONTreeRenderer::Render() {
    if (!is_valid_) {
        if (!error_.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "JSON Parse Error:");
            ImGui::TextWrapped("%s", error_.c_str());
        } else {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No JSON content loaded.");
        }
        return;
    }

    // Info bar
    if (json_.is_object()) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Object (%zu keys)", json_.size());
    } else if (json_.is_array()) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Array (%zu items)", json_.size());
    }
    ImGui::Separator();

    ImGui::BeginChild("##JSONTree", ImVec2(0, 0), false);
    node_id_ = 0;
    RenderNode("root", json_, true);
    ImGui::EndChild();
}

void JSONTreeRenderer::RenderNode(const std::string& key, const nlohmann::json& value, bool is_root) {
    node_id_++;
    ImGui::PushID(node_id_);

    if (value.is_object()) {
        // Object node
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth;
        if (is_root) flags |= ImGuiTreeNodeFlags_DefaultOpen;

        bool label_shown = !is_root;
        std::string label;
        if (label_shown) {
            label = key + " {" + std::to_string(value.size()) + "}";
        } else {
            label = "{" + std::to_string(value.size()) + " keys}";
            flags |= ImGuiTreeNodeFlags_DefaultOpen;
        }

        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.7f, 0.4f, 1.0f)); // Orange for keys
        bool open = ImGui::TreeNodeEx(label.c_str(), flags);
        ImGui::PopStyleColor();

        RenderContextMenu(value);

        if (open) {
            for (auto& [k, v] : value.items()) {
                RenderNode(k, v);
            }
            ImGui::TreePop();
        }
    } else if (value.is_array()) {
        // Array node
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth;
        if (is_root) flags |= ImGuiTreeNodeFlags_DefaultOpen;

        std::string label;
        if (!is_root) {
            label = key + " [" + std::to_string(value.size()) + "]";
        } else {
            label = "[" + std::to_string(value.size()) + " items]";
            flags |= ImGuiTreeNodeFlags_DefaultOpen;
        }

        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.8f, 1.0f, 1.0f)); // Blue for arrays
        bool open = ImGui::TreeNodeEx(label.c_str(), flags);
        ImGui::PopStyleColor();

        RenderContextMenu(value);

        if (open) {
            // For large arrays, only show first 100 items
            size_t max_items = std::min(value.size(), static_cast<size_t>(100));
            for (size_t i = 0; i < max_items; i++) {
                RenderNode("[" + std::to_string(i) + "]", value[i]);
            }
            if (value.size() > 100) {
                ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                    "... and %zu more items", value.size() - 100);
            }
            ImGui::TreePop();
        }
    } else {
        // Leaf node (value)
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f, 0.8f, 0.8f, 1.0f));
        ImGui::Text("%s: ", key.c_str());
        ImGui::PopStyleColor();
        ImGui::SameLine();
        RenderValue(value);
        RenderContextMenu(value);
    }

    ImGui::PopID();
}

void JSONTreeRenderer::RenderValue(const nlohmann::json& value) {
    if (value.is_string()) {
        std::string s = value.get<std::string>();
        // Truncate long strings
        if (s.size() > 200) {
            s = s.substr(0, 200) + "...";
        }
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.9f, 0.4f, 1.0f)); // Green
        ImGui::Text("\"%s\"", s.c_str());
        ImGui::PopStyleColor();
    } else if (value.is_number_integer()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.8f, 1.0f, 1.0f)); // Cyan
        ImGui::Text("%lld", value.get<int64_t>());
        ImGui::PopStyleColor();
    } else if (value.is_number_float()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.8f, 1.0f, 1.0f)); // Cyan
        ImGui::Text("%.6g", value.get<double>());
        ImGui::PopStyleColor();
    } else if (value.is_boolean()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.85f, 0.3f, 1.0f)); // Yellow
        ImGui::Text("%s", value.get<bool>() ? "true" : "false");
        ImGui::PopStyleColor();
    } else if (value.is_null()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.6f, 1.0f)); // Gray
        ImGui::Text("null");
        ImGui::PopStyleColor();
    }
}

void JSONTreeRenderer::RenderContextMenu(const nlohmann::json& value) {
    if (ImGui::BeginPopupContextItem()) {
        if (ImGui::MenuItem("Copy Value")) {
            std::string text;
            if (value.is_string()) {
                text = value.get<std::string>();
            } else {
                text = value.dump(2);
            }
            ImGui::SetClipboardText(text.c_str());
        }
        if (ImGui::MenuItem("Copy as JSON")) {
            ImGui::SetClipboardText(value.dump(2).c_str());
        }
        ImGui::EndPopup();
    }
}

} // namespace gui
