#include "pattern_browser.h"
#include "../icons.h"
#include "../node_editor.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstring>

namespace cyxwiz {

// Category names for display
static const char* GetCategoryName(::gui::patterns::PatternCategory category) {
    switch (category) {
        case ::gui::patterns::PatternCategory::Basic:         return "Basic";
        case ::gui::patterns::PatternCategory::CNN:           return "CNN";
        case ::gui::patterns::PatternCategory::RNN:           return "RNN";
        case ::gui::patterns::PatternCategory::Transformer:   return "Transformer";
        case ::gui::patterns::PatternCategory::Generative:    return "Generative";
        case ::gui::patterns::PatternCategory::BuildingBlocks: return "Blocks";
        case ::gui::patterns::PatternCategory::Custom:        return "Custom";
        default: return "Unknown";
    }
}

// Category icons
static const char* GetCategoryIcon(::gui::patterns::PatternCategory category) {
    switch (category) {
        case ::gui::patterns::PatternCategory::Basic:         return ICON_FA_LAYER_GROUP;
        case ::gui::patterns::PatternCategory::CNN:           return ICON_FA_IMAGE;
        case ::gui::patterns::PatternCategory::RNN:           return ICON_FA_CHART_LINE;
        case ::gui::patterns::PatternCategory::Transformer:   return ICON_FA_BRAIN;
        case ::gui::patterns::PatternCategory::Generative:    return ICON_FA_WAND_MAGIC_SPARKLES;
        case ::gui::patterns::PatternCategory::BuildingBlocks: return ICON_FA_CUBES;
        case ::gui::patterns::PatternCategory::Custom:        return ICON_FA_USER;
        default: return ICON_FA_CUBE;
    }
}

PatternBrowserPanel::PatternBrowserPanel()
    : Panel("Pattern Browser", false)  // Hidden by default
{
    std::memset(search_buffer_, 0, sizeof(search_buffer_));

    // Initialize the pattern library
    ::gui::patterns::PatternLibrary::Instance().Initialize();
}

void PatternBrowserPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(400, 500), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(GetName(), &visible_, ImGuiWindowFlags_NoCollapse)) {
        focused_ = ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);

        // Search bar at top
        RenderSearchBar();

        ImGui::Separator();

        // Category tabs
        RenderCategoryTabs();

        ImGui::Separator();

        // Pattern list
        ImGui::BeginChild("PatternList", ImVec2(0, 0), false);
        RenderPatternList();
        ImGui::EndChild();
    }
    ImGui::End();
}

void PatternBrowserPanel::RenderSearchBar() {
    ImGui::PushItemWidth(-1);

    // Search input with icon
    ImGui::Text(ICON_FA_MAGNIFYING_GLASS);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-1);

    if (ImGui::InputTextWithHint("##PatternSearch", "Search patterns...",
                                  search_buffer_, sizeof(search_buffer_))) {
        // Search is handled in RenderPatternList
    }

    ImGui::PopItemWidth();
}

void PatternBrowserPanel::RenderCategoryTabs() {
    // Get all categories that have patterns
    auto& library = ::gui::patterns::PatternLibrary::Instance();

    // Define all categories in order
    static const ::gui::patterns::PatternCategory categories[] = {
        ::gui::patterns::PatternCategory::Basic,
        ::gui::patterns::PatternCategory::CNN,
        ::gui::patterns::PatternCategory::RNN,
        ::gui::patterns::PatternCategory::Transformer,
        ::gui::patterns::PatternCategory::Generative,
        ::gui::patterns::PatternCategory::BuildingBlocks,
        ::gui::patterns::PatternCategory::Custom
    };

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 0));

    for (int i = 0; i < IM_ARRAYSIZE(categories); i++) {
        auto category = categories[i];

        // Count patterns in this category
        auto patterns = library.GetByCategory(category);

        // Only show categories that have patterns (or always show for discoverability)
        bool is_selected = (selected_category_ == category);

        // Category button styling
        if (is_selected) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
        }

        // Create button with icon and name
        std::string label = std::string(GetCategoryIcon(category)) + " " + GetCategoryName(category);

        if (i > 0) ImGui::SameLine();

        if (ImGui::Button(label.c_str())) {
            selected_category_ = category;
            expanded_pattern_id_.clear();  // Collapse any expanded pattern
        }

        // Tooltip showing pattern count
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s patterns: %zu", GetCategoryName(category), patterns.size());
        }

        if (is_selected) {
            ImGui::PopStyleColor();
        }
    }

    ImGui::PopStyleVar();
}

void PatternBrowserPanel::RenderPatternList() {
    auto& library = ::gui::patterns::PatternLibrary::Instance();

    // Get patterns based on search or category
    std::vector<::gui::patterns::Pattern> patterns;

    if (strlen(search_buffer_) > 0) {
        // Search across all categories
        patterns = library.Search(search_buffer_);
    } else {
        // Filter by selected category
        patterns = library.GetByCategory(selected_category_);
    }

    if (patterns.empty()) {
        ImGui::TextDisabled("No patterns found");
        if (strlen(search_buffer_) > 0) {
            ImGui::TextDisabled("Try a different search term");
        }
        return;
    }

    // Sort patterns alphabetically
    std::sort(patterns.begin(), patterns.end(),
              [](const ::gui::patterns::Pattern& a, const ::gui::patterns::Pattern& b) {
                  return a.name < b.name;
              });

    // Render each pattern card
    for (const auto& pattern : patterns) {
        RenderPatternCard(pattern);
        ImGui::Spacing();
    }
}

void PatternBrowserPanel::RenderPatternCard(const ::gui::patterns::Pattern& pattern) {
    bool is_expanded = (expanded_pattern_id_ == pattern.id);
    bool is_selected = (selected_pattern_id_ == pattern.id);

    ImGui::PushID(pattern.id.c_str());

    // Card background
    ImVec2 card_min = ImGui::GetCursorScreenPos();
    float card_width = ImGui::GetContentRegionAvail().x;

    // Calculate card height based on content
    float card_height = is_expanded ? 0.0f : 60.0f;  // Base height for collapsed

    // Draw card background
    ImU32 bg_color = is_selected ?
        ImGui::GetColorU32(ImGuiCol_FrameBgActive) :
        ImGui::GetColorU32(ImGuiCol_FrameBg);

    if (!is_expanded) {
        ImGui::GetWindowDrawList()->AddRectFilled(
            card_min,
            ImVec2(card_min.x + card_width, card_min.y + card_height),
            bg_color,
            4.0f  // Rounded corners
        );
    }

    // Card content
    ImGui::BeginGroup();

    // Pattern header with expand button
    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImGui::GetStyleColorVec4(ImGuiCol_FrameBgHovered));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImGui::GetStyleColorVec4(ImGuiCol_FrameBgActive));

    // Collapsing header for pattern
    bool header_open = ImGui::CollapsingHeader(
        (std::string(GetCategoryIcon(pattern.category)) + " " + pattern.name).c_str(),
        is_expanded ? ImGuiTreeNodeFlags_DefaultOpen : 0
    );

    ImGui::PopStyleColor(3);

    // Update expanded state
    if (header_open != is_expanded) {
        if (header_open) {
            expanded_pattern_id_ = pattern.id;
            selected_pattern_id_ = pattern.id;

            // Initialize parameter values with defaults
            param_values_.clear();
            for (const auto& param : pattern.parameters) {
                param_values_[param.name] = param.default_value;
            }
        } else {
            expanded_pattern_id_.clear();
        }
    }

    // Description (always visible)
    if (!header_open) {
        ImGui::Indent(20.0f);
        ImGui::TextDisabled("%s", pattern.description.c_str());
        ImGui::Unindent(20.0f);
    }

    // Expanded content
    if (header_open) {
        ImGui::Indent(20.0f);

        // Description
        ImGui::TextWrapped("%s", pattern.description.c_str());
        ImGui::Spacing();

        // Tags
        if (!pattern.tags.empty()) {
            ImGui::TextDisabled("Tags:");
            ImGui::SameLine();
            for (size_t i = 0; i < pattern.tags.size(); i++) {
                if (i > 0) ImGui::SameLine();
                ImGui::TextDisabled("[%s]", pattern.tags[i].c_str());
            }
            ImGui::Spacing();
        }

        // Parameters
        if (!pattern.parameters.empty()) {
            ImGui::Separator();
            ImGui::Text("Parameters:");
            ImGui::Spacing();

            RenderParameterInputs(pattern);

            ImGui::Spacing();
        }

        // Insert button
        ImGui::Separator();
        ImGui::Spacing();

        float button_width = ImGui::GetContentRegionAvail().x;
        if (ImGui::Button((std::string(ICON_FA_PLUS) + " Insert Pattern").c_str(),
                          ImVec2(button_width, 0))) {
            InsertPattern(pattern.id);
        }

        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Insert this pattern into the node editor");
        }

        ImGui::Unindent(20.0f);
    }

    ImGui::EndGroup();

    ImGui::PopID();
}

void PatternBrowserPanel::RenderParameterInputs(const ::gui::patterns::Pattern& pattern) {
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x * 0.5f);

    for (const auto& param : pattern.parameters) {
        ImGui::PushID(param.name.c_str());

        // Parameter label with tooltip
        ImGui::Text("%s:", param.name.c_str());
        if (ImGui::IsItemHovered() && !param.description.empty()) {
            ImGui::SetTooltip("%s", param.description.c_str());
        }
        ImGui::SameLine();

        // Get current value
        auto& value = param_values_[param.name];

        // Input based on type
        switch (param.type) {
            case ::gui::patterns::ParameterType::Int: {
                int int_val = std::stoi(value.empty() ? "0" : value);
                int min_val = param.min_value.empty() ? 0 : std::stoi(param.min_value);
                int max_val = param.max_value.empty() ? 10000 : std::stoi(param.max_value);

                if (ImGui::DragInt("##value", &int_val, 1.0f, min_val, max_val)) {
                    value = std::to_string(int_val);
                }
                break;
            }

            case ::gui::patterns::ParameterType::Float: {
                float float_val = std::stof(value.empty() ? "0.0" : value);
                float min_val = param.min_value.empty() ? 0.0f : std::stof(param.min_value);
                float max_val = param.max_value.empty() ? 1.0f : std::stof(param.max_value);

                if (ImGui::DragFloat("##value", &float_val, 0.01f, min_val, max_val, "%.3f")) {
                    value = std::to_string(float_val);
                }
                break;
            }

            case ::gui::patterns::ParameterType::String: {
                char buffer[256];
                std::strncpy(buffer, value.c_str(), sizeof(buffer) - 1);
                buffer[sizeof(buffer) - 1] = '\0';

                if (ImGui::InputText("##value", buffer, sizeof(buffer))) {
                    value = buffer;
                }
                break;
            }

            case ::gui::patterns::ParameterType::Bool: {
                bool bool_val = (value == "true" || value == "1");
                if (ImGui::Checkbox("##value", &bool_val)) {
                    value = bool_val ? "true" : "false";
                }
                break;
            }

            case ::gui::patterns::ParameterType::NodeType: {
                // Dropdown for node type selection
                if (!param.options.empty()) {
                    int current_idx = 0;
                    for (size_t i = 0; i < param.options.size(); i++) {
                        if (param.options[i] == value) {
                            current_idx = static_cast<int>(i);
                            break;
                        }
                    }

                    if (ImGui::BeginCombo("##value", value.c_str())) {
                        for (size_t i = 0; i < param.options.size(); i++) {
                            bool is_selected = (current_idx == static_cast<int>(i));
                            if (ImGui::Selectable(param.options[i].c_str(), is_selected)) {
                                value = param.options[i];
                            }
                            if (is_selected) {
                                ImGui::SetItemDefaultFocus();
                            }
                        }
                        ImGui::EndCombo();
                    }
                } else {
                    // Fallback to text input
                    char buffer[256];
                    std::strncpy(buffer, value.c_str(), sizeof(buffer) - 1);
                    buffer[sizeof(buffer) - 1] = '\0';

                    if (ImGui::InputText("##value", buffer, sizeof(buffer))) {
                        value = buffer;
                    }
                }
                break;
            }

            default: {
                // Default text input for unknown types
                char buffer[256];
                std::strncpy(buffer, value.c_str(), sizeof(buffer) - 1);
                buffer[sizeof(buffer) - 1] = '\0';

                if (ImGui::InputText("##value", buffer, sizeof(buffer))) {
                    value = buffer;
                }
                break;
            }
        }

        ImGui::PopID();
    }

    ImGui::PopItemWidth();
}

void PatternBrowserPanel::InsertPattern(const std::string& pattern_id) {
    auto& library = ::gui::patterns::PatternLibrary::Instance();

    // Prepare output containers
    std::vector<::gui::MLNode> nodes;
    std::vector<::gui::NodeLink> links;

    // Get next IDs from node editor (we'll need to coordinate this)
    // For now, use a starting ID that's likely to be unique
    static int base_id = 10000;
    int next_node_id = base_id;
    int next_pin_id = base_id * 10;
    int next_link_id = base_id * 100;

    // Default position (center of canvas area)
    ImVec2 base_pos(200.0f, 200.0f);

    // Instantiate the pattern
    bool success = library.InstantiatePattern(
        pattern_id,
        param_values_,
        nodes,
        links,
        next_node_id,
        next_pin_id,
        next_link_id,
        base_pos
    );

    if (success) {
        // Use callback if set
        if (insert_callback_) {
            insert_callback_(nodes, links);
        }

        // Update base ID for next insertion
        base_id = next_node_id + 100;

        spdlog::info("Inserted pattern '{}' with {} nodes", pattern_id, nodes.size());
    } else {
        spdlog::error("Failed to instantiate pattern: {}", pattern_id);
    }
}

void PatternBrowserPanel::OpenWithPattern(const std::string& pattern_id) {
    // Make panel visible
    Show();

    // Find and select the pattern
    auto& library = ::gui::patterns::PatternLibrary::Instance();
    const auto* pattern = library.GetPattern(pattern_id);

    if (pattern) {
        // Set category
        selected_category_ = pattern->category;

        // Expand this pattern
        expanded_pattern_id_ = pattern_id;
        selected_pattern_id_ = pattern_id;

        // Initialize parameter values
        param_values_.clear();
        for (const auto& param : pattern->parameters) {
            param_values_[param.name] = param.default_value;
        }
    }
}

} // namespace cyxwiz
