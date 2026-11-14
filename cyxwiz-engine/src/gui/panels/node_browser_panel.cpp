#include "node_browser_panel.h"
#include "../python_bundle_ui.h"
#include "../node_editor.h"
#include "../../core/node_metadata_registry.h"
#include "../../core/node_palette.h"
#include "../../core/extension_node_registry.h"
#include "../../core/engine_config.h"
#include "../../plugin/plugin_manager.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../patterns/pattern_library.h"
#include "../icons.h"
#include <imgui.h>
#include <imgui_internal.h>
#include <algorithm>
#include <cctype>
#include <string>
#include <vector>
#include <map>
#include <cmath>

namespace gui {

namespace ui = cyxwiz::ui;

namespace {

// A dimmed section title with a soft line running to the right edge.
void SectionTitle(const char* title, const char* count = nullptr) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
    ImGui::TextUnformatted(title);
    float text_width = ImGui::CalcTextSize(title).x + 8;
    if (count) {
        ImGui::SameLine(0, 6);
        ImGui::TextDisabled("%s", count);
        text_width += ImGui::CalcTextSize(count).x + 6;
    }
    ImGui::PopStyleColor();
    ImVec2 line_start = ImGui::GetCursorScreenPos();
    line_start.y -= 2;
    const ImVec2 line_end(line_start.x + ImGui::GetContentRegionAvail().x, line_start.y);
    line_start.x += text_width;
    ImGui::GetWindowDrawList()->AddLine(line_start, line_end, ui::ToU32(t.border_soft), 1.0f);
}

std::vector<int> TypeNumbers(const std::vector<cyxwiz::NodeType>& types) {
    std::vector<int> out;
    out.reserve(types.size());
    for (auto type : types) out.push_back(static_cast<int>(type));
    return out;
}

std::vector<cyxwiz::NodeType> NodeTypes(const std::vector<int>& numbers) {
    std::vector<cyxwiz::NodeType> out;
    out.reserve(numbers.size());
    for (int n : numbers) out.push_back(static_cast<cyxwiz::NodeType>(n));
    return out;
}

bool HasSupportedAxis(const cyxwiz::NodeMetadata* metadata,
                      const char* axis_name) {
    if (!metadata) return false;
    return std::any_of(
        metadata->support_axes.begin(),
        metadata->support_axes.end(),
        [axis_name](const cyxwiz::SupportAxisDefinition& axis) {
            return axis.name == axis_name && axis.supported;
        });
}

bool HasAxis(const cyxwiz::NodeMetadata* metadata, const char* axis_name) {
    if (!metadata) return false;
    return std::any_of(
        metadata->support_axes.begin(),
        metadata->support_axes.end(),
        [axis_name](const cyxwiz::SupportAxisDefinition& axis) {
            return axis.name == axis_name;
        });
}

const cyxwiz::SupportAxisDefinition* FindAxis(
    const cyxwiz::NodeMetadata* metadata,
    const char* axis_name) {
    if (!metadata) return nullptr;
    auto it = std::find_if(
        metadata->support_axes.begin(),
        metadata->support_axes.end(),
        [axis_name](const cyxwiz::SupportAxisDefinition& axis) {
            return axis.name == axis_name;
        });
    return it != metadata->support_axes.end() ? &*it : nullptr;
}

const cyxwiz::SupportAxisDefinition* FindSupportState(
    const cyxwiz::NodeMetadata* metadata) {
    return FindAxis(metadata, "Support State");
}

std::string FirstUnsupportedSupportReason(
    const cyxwiz::NodeMetadata* metadata) {
    if (!metadata) return {};

    if (const auto* support_state = FindSupportState(metadata)) {
        if (!support_state->reason.empty() &&
            (!support_state->supported || support_state->value == "blocked")) {
            return support_state->reason;
        }
    }

    for (const auto& axis : metadata->support_axes) {
        if (!axis.supported && !axis.reason.empty()) {
            return axis.name + ": " + axis.reason;
        }
    }

    return {};
}

ImVec4 SupportStateBadgeColor(const std::string& state) {
    const ui::Tokens& t = ui::CurrentTokens();
    const float alpha = 235.0f / 255.0f;
    if (state == "real") return ui::WithAlpha(t.success, alpha);
    if (state == "partial") return ui::WithAlpha(t.warning, alpha);
    if (state == "blocked") return ui::WithAlpha(t.error, alpha);
    return ui::WithAlpha(t.pending, alpha);
}

std::string SupportStateLabel(const std::string& state) {
    if (state == "real") return "Real";
    if (state == "partial") return "Partial";
    if (state == "blocked") return "Blocked";
    return state.empty() ? "Unknown" : state;
}

std::string SupportAxisValueLabel(const std::string& value) {
    if (value == "real") return "Real";
    if (value == "partial") return "Partial";
    if (value == "blocked") return "Blocked";
    if (value == "none") return "None";
    if (value == "training_backend") return "Training backend";
    if (value == "pipeline_executor") return "Pipeline executor";
    if (value == "pipeline_operator_factory") return "Pipeline operator factory";
    if (value == "ui_only") return "UI-only";
    if (value == "recipe_lowering") return "Recipe lowering";
    if (value == "classic_ml") return "Classic ML";
    if (value == "deep_learning") return "Deep learning";
    if (value == "model_layer") return "Model layer";
    if (value == "activation") return "Activation";
    if (value == "loss") return "Loss";
    if (value == "optimizer") return "Optimizer";
    if (value == "regression") return "Regression";
    if (value == "multiclass_classification") return "Multiclass classification";
    if (value == "binary_classification") return "Binary classification";
    if (value == "supported") return "Supported";
    if (value == "unsupported") return "Unsupported";
    if (value == "fail_closed") return "Fail closed";
    if (value == "hard_fail") return "Hard fail";

    std::string label = value.empty() ? std::string("Unknown") : value;
    std::replace(label.begin(), label.end(), '_', ' ');
    return label;
}

bool IsTrainingCategory(cyxwiz::NodeCategory category) {
    switch (category) {
    case cyxwiz::NodeCategory::Layers:
    case cyxwiz::NodeCategory::Activation:
    case cyxwiz::NodeCategory::Pooling:
    case cyxwiz::NodeCategory::Normalization:
    case cyxwiz::NodeCategory::Attention:
    case cyxwiz::NodeCategory::Recurrent:
    case cyxwiz::NodeCategory::ShapeOps:
    case cyxwiz::NodeCategory::MergeOps:
    case cyxwiz::NodeCategory::Training:
    case cyxwiz::NodeCategory::Regularization:
        return true;
    default:
        return false;
    }
}

} // namespace

NodeBrowserPanel::NodeBrowserPanel() {
    // Initialize category show_all state to false (collapsed)
    auto& registry = cyxwiz::NodeMetadataRegistry::Instance();
    if (!registry.IsInitialized()) {
        registry.Initialize();
    }
    for (auto category : registry.GetCategories()) {
        category_show_all_[category] = false;
    }
}

void NodeBrowserPanel::Render() {
    auto& registry = cyxwiz::NodeMetadataRegistry::Instance();
    if (!registry.IsInitialized()) {
        registry.Initialize();
    }
    // Also while hidden: nodes added from the menus and the canvas count.
    SyncUserNodes();
    if (!visible_) return;

    ImGui::SetNextWindowSizeConstraints(ImVec2(PANEL_MIN_WIDTH, 300), ImVec2(450, FLT_MAX));

    if (focus_window_next_frame_) {
        ImGui::SetNextWindowFocus();
        focus_window_next_frame_ = false;
    }
    if (ImGui::Begin("Nodes", &visible_)) {
        RenderHeader();
        RenderSearchBar();

        RenderFilterTags();

        ImGui::Separator();

        // Main content area with scrolling
        ImGui::BeginChild("NodeGrid", ImVec2(0, 0), false);

        // Favourites, Recent and Studio Tools (at the top when not searching
        // or in a category)
        if (search_query_.empty() && !showing_all_in_category_) {
            if (support_filter_mode_ == SupportFilterMode::All) {
                RenderFavouritesSection();
                RenderRecentSection();
            }
            RenderStudioSection();
            ImGui::Spacing();
        }

        if (!search_query_.empty()) {
            // Search results mode - show as grid
            auto results = registry.Search(search_query_, true);
            results = ApplySupportFilter(results);
            if (support_filter_mode_ == SupportFilterMode::All) {
                const auto rank = [](const cyxwiz::NodeMetadata* m) {
                    switch (cyxwiz::palette::PlaceNode(*m).place) {
                    case cyxwiz::palette::Place::Category: return 0;
                    case cyxwiz::palette::Place::Advanced: return 1;
                    case cyxwiz::palette::Place::CantRun: return 2;
                    case cyxwiz::palette::Place::Hidden: return 3;
                    }
                    return 3;
                };
                results.erase(std::remove_if(results.begin(), results.end(),
                                             [&](const cyxwiz::NodeMetadata* m) { return rank(m) == 3; }),
                              results.end());
                // The registry's order (exact name, name start, use) within each place.
                std::stable_sort(results.begin(), results.end(),
                                 [&](const cyxwiz::NodeMetadata* a, const cyxwiz::NodeMetadata* b) { return rank(a) < rank(b); });
            }
            // The generic extension entry: plugin nodes are listed below.
            results.erase(std::remove_if(results.begin(), results.end(),
                                         [](const cyxwiz::NodeMetadata* m) {
                                             return m->type == cyxwiz::NodeType::PluginCustom;
                                         }),
                          results.end());
            // Nodes named by the query first; nodes that only mention it (a
            // keyword or the description) come after, under a "Related" line.
            const auto lower = [](std::string text) {
                std::transform(text.begin(), text.end(), text.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                return text;
            };
            const std::string query = lower(search_query_);
            const auto named = std::stable_partition(results.begin(), results.end(), [&](const cyxwiz::NodeMetadata* m) {
                return lower(m->name).find(query) != std::string::npos;
            });
            std::vector<const cyxwiz::NodeMetadata*> related(named, results.end());
            results.erase(named, results.end());
            // The whole name, then the name's start, before the place order.
            const auto strength = [&](const cyxwiz::NodeMetadata* m) {
                const std::string name = lower(m->name);
                return name == query ? 2 : name.rfind(query, 0) == 0 ? 1 : 0;
            };
            std::stable_sort(results.begin(), results.end(),
                             [&](const cyxwiz::NodeMetadata* a, const cyxwiz::NodeMetadata* b) { return strength(a) > strength(b); });
            std::vector<cyxwiz::ExtensionPaletteEntry> extension_results;
            if (support_filter_mode_ == SupportFilterMode::All) {
                for (auto& entry : cyxwiz::BuildExtensionPalette(
                         cyxwiz::ExtensionNodeRegistry::Instance().All())) {
                    if (cyxwiz::ExtensionPaletteEntryMatches(entry, search_query_)) {
                        extension_results.push_back(std::move(entry));
                    }
                }
            }
            if (results.empty() && extension_results.empty() && related.empty()) {
                ImGui::TextDisabled("No nodes found matching '%s'", search_query_.c_str());
            } else {
                RenderNodeGrid(results);
                RenderExtensionGrid(extension_results);
                if (!related.empty()) {
                    if (!results.empty() || !extension_results.empty()) ImGui::Spacing();
                    SectionTitle("Related");
                    RenderNodeGrid(related);
                }
            }
        } else if (showing_all_in_category_ &&
                   current_category_ == cyxwiz::NodeCategory::Plugin) {
            RenderExtensionSection(true);
        } else if (showing_all_in_category_) {
            // Showing all nodes in a specific category
            auto nodes = Curated(registry.GetByCategory(current_category_, true));
            nodes = ApplySupportFilter(nodes);
            RenderNodeGrid(nodes);
        } else {
            // Normal category view
            auto categories = registry.GetCategories();
            for (auto category : categories) {
                if (category == cyxwiz::NodeCategory::Plugin) {
                    RenderExtensionSection(false);
                } else {
                    RenderCategorySection(category);
                }
            }
            if (support_filter_mode_ == SupportFilterMode::All) {
                RenderAdvancedSection();
                RenderCantRunSection();
            }
        }

        ImGui::EndChild();
    }
    ImGui::End();
}

void NodeBrowserPanel::RenderHeader() {
    // Breadcrumb navigation: "Nodes" or "Nodes > CategoryName"
    ImGui::PushStyleColor(ImGuiCol_Text, ui::CurrentTokens().text_dim);

    if (showing_all_in_category_) {
        // Clickable "Nodes" to go back
        if (ImGui::SmallButton("Nodes")) {
            NavigateBack();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Go back to all categories (Esc)");
        }
        ImGui::SameLine();
        ImGui::TextUnformatted(">");
        ImGui::SameLine();
        ImGui::TextUnformatted(cyxwiz::GetCategoryDisplayName(current_category_).c_str());
    } else {
        ImGui::TextUnformatted("Nodes");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Node Browser - Drag nodes to canvas\nPress / to search");
        }
    }

    ImGui::PopStyleColor();

    // Grid/List toggle button (top right) - placeholder for future
    ImGui::SameLine(ImGui::GetContentRegionAvail().x - 20);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::SmallButton(ICON_FA_GRIP);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Toggle grid/list view (Coming Soon)");
    }
    ImGui::PopStyleColor();
}

void NodeBrowserPanel::RenderSearchBar() {
    ImGui::PushItemWidth(-1);

    char search_buf[256] = {0};
    strncpy(search_buf, search_query_.c_str(), sizeof(search_buf) - 1);

    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8, 6));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);

    // Focus search bar if requested (from keyboard shortcut)
    if (focus_search_next_frame_) {
        ImGui::SetKeyboardFocusHere();
        focus_search_next_frame_ = false;
    }

    if (ImGui::InputTextWithHint("##NodeSearch", ICON_FA_SEARCH " Search nodes (/)",
                                  search_buf, sizeof(search_buf))) {
        search_query_ = search_buf;
        // Clear category view when searching
        if (!search_query_.empty()) {
            showing_all_in_category_ = false;
        }
    }

    // Tooltip for search bar
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Search nodes by name, keywords, or description\nShortcut: / or Ctrl+F");
    }

    ImGui::PopStyleVar(2);
    ImGui::PopItemWidth();
}

void NodeBrowserPanel::ClearSearch() {
    search_query_.clear();
    showing_all_in_category_ = false;
    current_category_ = cyxwiz::NodeCategory::Unknown;
    provider_filter_.clear();
}

void NodeBrowserPanel::ShowPluginNodes(const std::string& provider_id) {
    ClearSearch();
    support_filter_mode_ = SupportFilterMode::All;
    NavigateToCategory(cyxwiz::NodeCategory::Plugin);
    provider_filter_ = provider_id;
    visible_ = true;
    focus_window_next_frame_ = true;
}

bool NodeBrowserPanel::HandleKeyboardShortcuts() {
    // Only handle shortcuts when panel is visible and no text input is active
    if (!visible_) return false;

    // Check if any text input is active
    if (ImGui::GetIO().WantTextInput) return false;

    // / key - Focus search bar
    if (ImGui::IsKeyPressed(ImGuiKey_Slash) && !ImGui::GetIO().KeyCtrl) {
        FocusSearchBar();
        return true;
    }

    // Ctrl+F - Focus search bar
    if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_F)) {
        FocusSearchBar();
        return true;
    }

    // Escape - Clear search and go back
    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
        if (!search_query_.empty() || showing_all_in_category_) {
            ClearSearch();
            return true;
        }
    }

    return false;
}

void NodeBrowserPanel::RenderStudioSection() {
    const ui::Tokens& t = ui::CurrentTokens();
    SectionTitle("Studio Tools");

    // Calculate card dimensions (same as node cards)
    float available_width = ImGui::GetContentRegionAvail().x;
    grid_columns_ = ColumnsFor(available_width);
    float card_width = (available_width - (grid_columns_ - 1) * GRID_SPACING) / grid_columns_;

    // === Note Card ===
    {
        ImGui::PushID("StudioNote");
        ImVec2 card_size(card_width, NODE_CARD_HEIGHT);
        ImVec2 cursor_start = ImGui::GetCursorScreenPos();

        ImGui::InvisibleButton("##notecard", card_size);
        bool hovered = ImGui::IsItemHovered();

        // Double-click to add annotation
        if (hovered && ImGui::IsMouseDoubleClicked(0)) {
            if (node_editor_) {
                node_editor_->AddAnnotation();
            }
        }

        // Drag-drop source for annotation
        if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
            int annotation_type = 1;  // Just a marker value
            ImGui::SetDragDropPayload("ANNOTATION", &annotation_type, sizeof(int));
            ImGui::Text(ICON_FA_COMMENT " Annotation");
            ImGui::EndDragDropSource();
        }

        // Tooltip
        if (hovered) {
            ImGui::BeginTooltip();
            ImGui::TextColored(ui::CurrentTokens().text_bright, ICON_FA_COMMENT " Annotation");
            ImGui::TextWrapped("Add a sticky note to the canvas for documentation");
            ImGui::Separator();
            ImGui::TextDisabled("Drag to canvas or double-click to add");
            ImGui::TextDisabled("Shortcut: Ctrl+Shift+N");
            ImGui::EndTooltip();
        }

        // Draw card
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        float icon_x = cursor_start.x + (card_width - NODE_ICON_SIZE) * 0.5f;
        float icon_y = cursor_start.y + 4;
        ImVec2 icon_pos(icon_x, icon_y);
        ImVec2 p_max(icon_pos.x + NODE_ICON_SIZE, icon_pos.y + NODE_ICON_SIZE);

        // The canvas note colour
        ImU32 bg_color = DefaultAnnotationColour();
        ImU32 border_color = ui::ToU32(ui::Mix(ImGui::ColorConvertU32ToFloat4(bg_color), t.bg_window, 0.3f));
        draw_list->AddRectFilled(icon_pos, p_max, bg_color, 6.0f);
        draw_list->AddRect(icon_pos, p_max, border_color, 6.0f, 0, 2.0f);

        // Icon
        ImFont* icon_font = ImGui::GetFont();
        float scaled_icon_size = NODE_ICON_SIZE * 0.55f;
        const char* icon = ICON_FA_COMMENT;
        ImVec2 icon_text_size = icon_font->CalcTextSizeA(scaled_icon_size, FLT_MAX, 0.0f, icon);
        ImVec2 icon_text_pos(icon_pos.x + (NODE_ICON_SIZE - icon_text_size.x) * 0.5f,
                            icon_pos.y + (NODE_ICON_SIZE - icon_text_size.y) * 0.5f);
        draw_list->AddText(icon_font, scaled_icon_size, icon_text_pos, ui::ToU32(ui::ReadableOn(t, ImGui::ColorConvertU32ToFloat4(bg_color))), icon);

        // Label
        const char* label = "Note";
        ImVec2 text_size = ImGui::CalcTextSize(label);
        float text_x = cursor_start.x + (card_width - text_size.x) * 0.5f;
        float text_y = icon_y + NODE_ICON_SIZE + 4;
        draw_list->AddText(ImVec2(text_x, text_y), ui::ToU32(t.text), label);

        // Hover highlight
        if (hovered) {
            draw_list->AddRect(cursor_start,
                ImVec2(cursor_start.x + card_width, cursor_start.y + card_size.y),
                ui::ToU32(ui::WithAlpha(t.accent_text, 100.0f / 255.0f)), 4.0f);
        }
        ImGui::PopID();
    }

    if (grid_columns_ > 1) ImGui::SameLine(0, GRID_SPACING);  // one column: the cards stack

    // === Group Card ===
    {
        ImGui::PushID("StudioGroup");
        ImVec2 card_size(card_width, NODE_CARD_HEIGHT);
        ImVec2 cursor_start = ImGui::GetCursorScreenPos();

        ImGui::InvisibleButton("##groupcard", card_size);
        bool hovered = ImGui::IsItemHovered();

        // Double-click to group selected nodes
        if (hovered && ImGui::IsMouseDoubleClicked(0)) {
            if (node_editor_) {
                node_editor_->GroupSelectedNodes();
            }
        }

        // Tooltip
        if (hovered) {
            ImGui::BeginTooltip();
            ImGui::TextColored(ui::CurrentTokens().text_bright, ICON_FA_OBJECT_GROUP " Group");
            ImGui::TextWrapped("Group selected nodes together for organization");
            ImGui::Separator();
            ImGui::TextDisabled("Select nodes first, then double-click");
            ImGui::TextDisabled("Shortcut: Ctrl+G");
            ImGui::EndTooltip();
        }

        // Draw card
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        float icon_x = cursor_start.x + (card_width - NODE_ICON_SIZE) * 0.5f;
        float icon_y = cursor_start.y + 4;
        ImVec2 icon_pos(icon_x, icon_y);
        ImVec2 p_max(icon_pos.x + NODE_ICON_SIZE, icon_pos.y + NODE_ICON_SIZE);

        // Group card: a series colour of the theme
        ImU32 bg_color = ui::ToU32(t.series[1]);
        ImU32 border_color = ui::ToU32(ui::Mix(t.series[1], t.bg_window, 0.3f));
        draw_list->AddRectFilled(icon_pos, p_max, bg_color, 6.0f);
        draw_list->AddRect(icon_pos, p_max, border_color, 6.0f, 0, 2.0f);

        // Icon
        ImFont* icon_font = ImGui::GetFont();
        float scaled_icon_size = NODE_ICON_SIZE * 0.55f;
        const char* icon = ICON_FA_OBJECT_GROUP;
        ImVec2 icon_text_size = icon_font->CalcTextSizeA(scaled_icon_size, FLT_MAX, 0.0f, icon);
        ImVec2 icon_text_pos(icon_pos.x + (NODE_ICON_SIZE - icon_text_size.x) * 0.5f,
                            icon_pos.y + (NODE_ICON_SIZE - icon_text_size.y) * 0.5f);
        draw_list->AddText(icon_font, scaled_icon_size, icon_text_pos, ui::ToU32(ui::ReadableOn(t, ImGui::ColorConvertU32ToFloat4(bg_color))), icon);

        // Label
        const char* label = "Group";
        ImVec2 text_size = ImGui::CalcTextSize(label);
        float text_x = cursor_start.x + (card_width - text_size.x) * 0.5f;
        float text_y = icon_y + NODE_ICON_SIZE + 4;
        draw_list->AddText(ImVec2(text_x, text_y), ui::ToU32(t.text), label);

        // Hover highlight
        if (hovered) {
            draw_list->AddRect(cursor_start,
                ImVec2(cursor_start.x + card_width, cursor_start.y + card_size.y),
                ui::ToU32(ui::WithAlpha(t.accent_text, 100.0f / 255.0f)), 4.0f);
        }
        ImGui::PopID();
    }

    // Frame card (drag to canvas to create visual organization box)
    {
        ImGui::PushID("frame_card");
        ImVec2 card_size(card_width, NODE_ICON_SIZE + 24);
        ImVec2 cursor_start = ImGui::GetCursorScreenPos();
        ImGui::InvisibleButton("##framecard", card_size);
        bool hovered = ImGui::IsItemHovered();

        // Drag source for creating frame on canvas
        if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
            const char* payload = "STUDIO_FRAME";
            ImGui::SetDragDropPayload("STUDIO_FRAME", payload, strlen(payload) + 1);
            ImGui::Text(ICON_FA_SQUARE " Frame");
            ImGui::TextDisabled("Drop on canvas to create");
            ImGui::EndDragDropSource();
        }

        // Tooltip
        if (hovered && !ImGui::GetDragDropPayload()) {
            ImGui::BeginTooltip();
            ImGui::TextColored(ui::CurrentTokens().text_bright, ICON_FA_SQUARE " Frame");
            ImGui::TextWrapped("Visual organization box for arranging nodes");
            ImGui::Separator();
            ImGui::TextDisabled("Drag to canvas to create");
            ImGui::EndTooltip();
        }

        // Draw card
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        float icon_x = cursor_start.x + (card_width - NODE_ICON_SIZE) * 0.5f;
        float icon_y = cursor_start.y + 4;
        ImVec2 icon_pos(icon_x, icon_y);
        ImVec2 p_max(icon_pos.x + NODE_ICON_SIZE, icon_pos.y + NODE_ICON_SIZE);

        // Frame card: a series colour of the theme
        ImU32 bg_color = ui::ToU32(t.series[3]);
        ImU32 border_color = ui::ToU32(ui::Mix(t.series[3], t.bg_window, 0.3f));
        draw_list->AddRectFilled(icon_pos, p_max, bg_color, 6.0f);
        draw_list->AddRect(icon_pos, p_max, border_color, 6.0f, 0, 2.0f);

        // Icon
        ImFont* icon_font = ImGui::GetFont();
        float scaled_icon_size = NODE_ICON_SIZE * 0.55f;
        const char* icon = ICON_FA_SQUARE;
        ImVec2 icon_text_size = icon_font->CalcTextSizeA(scaled_icon_size, FLT_MAX, 0.0f, icon);
        ImVec2 icon_text_pos(icon_pos.x + (NODE_ICON_SIZE - icon_text_size.x) * 0.5f,
                            icon_pos.y + (NODE_ICON_SIZE - icon_text_size.y) * 0.5f);
        draw_list->AddText(icon_font, scaled_icon_size, icon_text_pos, ui::ToU32(ui::ReadableOn(t, ImGui::ColorConvertU32ToFloat4(bg_color))), icon);

        // Label
        const char* label = "Frame";
        ImVec2 text_size = ImGui::CalcTextSize(label);
        float text_x = cursor_start.x + (card_width - text_size.x) * 0.5f;
        float text_y = icon_y + NODE_ICON_SIZE + 4;
        draw_list->AddText(ImVec2(text_x, text_y), ui::ToU32(t.text), label);

        // Hover highlight
        if (hovered) {
            draw_list->AddRect(cursor_start,
                ImVec2(cursor_start.x + card_width, cursor_start.y + card_size.y),
                ui::ToU32(ui::WithAlpha(t.accent_text, 100.0f / 255.0f)), 4.0f);
        }
        ImGui::PopID();
    }

    ImGui::Spacing();
    ImGui::Spacing();

    // === Examples Section ===
    ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
    ImGui::TextUnformatted("Examples");

    // Horizontal separator line
    ImVec2 examples_line_start = ImGui::GetCursorScreenPos();
    examples_line_start.y -= 2;
    float examples_text_width = ImGui::CalcTextSize("Examples").x + 8;
    ImVec2 examples_line_end = ImVec2(examples_line_start.x + ImGui::GetContentRegionAvail().x, examples_line_start.y);
    examples_line_start.x += examples_text_width;
    ImGui::GetWindowDrawList()->AddLine(examples_line_start, examples_line_end, ui::ToU32(t.border_soft), 1.0f);
    ImGui::PopStyleColor();

    // Get pattern library
    auto& pattern_lib = patterns::PatternLibrary::Instance();
    if (!pattern_lib.IsInitialized()) {
        pattern_lib.Initialize();
    }

    auto pattern_categories = pattern_lib.GetAvailableCategories();
    for (auto cat : pattern_categories) {
        // Category tree node
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth;
        const char* cat_icon = patterns::GetPatternCategoryIcon(cat);
        std::string cat_label = std::string(cat_icon) + " " + patterns::PatternCategoryToString(cat);

        if (ImGui::TreeNodeEx(cat_label.c_str(), flags)) {
            const auto cat_patterns = pattern_lib.GetByCategory(cat);
            for (const auto& pattern : cat_patterns) {
                ImGui::PushID(pattern.id.c_str());

                // Selectable item for each pattern
                bool selected = false;
                if (ImGui::Selectable(pattern.name.c_str(), &selected, ImGuiSelectableFlags_AllowDoubleClick)) {
                    if (ImGui::IsMouseDoubleClicked(0)) {
                        // Load pattern into canvas
                        if (node_editor_) {
                            std::vector<MLNode> nodes;
                            std::vector<NodeLink> links;
                            std::map<std::string, std::string> params;  // Default params
                            static int base_id = 20000;
                            int next_node_id = base_id;
                            int next_link_id = base_id * 100;
                            ImVec2 base_pos(200.0f, 200.0f);

                            patterns::NodeCreatorCallback creator = [this](NodeType type, const std::string& name) {
                                MLNode node = node_editor_->CreateNode(type, name);
                                node_editor_->ApplyUserNodeDefaults(node);
                                return node;
                            };

                            if (pattern_lib.InstantiatePatternWithCreator(
                                pattern.id, params, nodes, links, next_node_id, next_link_id, base_pos, creator)) {
                                node_editor_->InsertPattern(nodes, links);
                                base_id = next_node_id + 100;
                            }
                        }
                    }
                }

                // Tooltip
                if (ImGui::IsItemHovered()) {
                    ImGui::BeginTooltip();
                    ImGui::TextColored(ui::CurrentTokens().text_bright, "%s", pattern.name.c_str());
                    if (!pattern.description.empty()) {
                        ImGui::TextWrapped("%s", pattern.description.c_str());
                    }
                    ImGui::Separator();
                    ImGui::TextDisabled("Double-click to load");
                    ImGui::EndTooltip();
                }

                ImGui::PopID();
            }
            ImGui::TreePop();
        }
    }

    if (pattern_categories.empty()) {
        ImGui::TextDisabled("No patterns available");
    }

    ImGui::Spacing();
    ImGui::Spacing();
}

void NodeBrowserPanel::RenderFilterTags() {
    auto& registry = cyxwiz::NodeMetadataRegistry::Instance();

    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6, 3));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0f);

    bool rendered_filter = false;
    if (showing_all_in_category_) {
        std::string cat_name = cyxwiz::GetCategoryDisplayName(current_category_);
        if (ImGui::BeginCombo("##CatFilter", cat_name.c_str(), ImGuiComboFlags_NoArrowButton)) {
            for (auto category : registry.GetCategories()) {
                bool selected = (category == current_category_);
                if (ImGui::Selectable(cyxwiz::GetCategoryDisplayName(category).c_str(), selected)) {
                    NavigateToCategory(category);
                }
            }
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Filter by node category");
        }
        rendered_filter = true;
    }

    if (rendered_filter) {
        ImGui::SameLine();
    }

    if (ImGui::BeginCombo("##SupportFilter", GetSupportFilterLabel(support_filter_mode_),
                          ImGuiComboFlags_NoArrowButton)) {
        const SupportFilterMode modes[] = {
            SupportFilterMode::All,
            SupportFilterMode::Pipeline,
            SupportFilterMode::Training,
            SupportFilterMode::UiOnly,
            SupportFilterMode::Blocked,
            SupportFilterMode::Favourites,
        };
        for (SupportFilterMode mode : modes) {
            const bool selected = support_filter_mode_ == mode;
            if (ImGui::Selectable(GetSupportFilterLabel(mode), selected)) {
                support_filter_mode_ = mode;
            }
        }
        ImGui::EndCombo();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Filter using structured support axes");
    }

    std::vector<const cyxwiz::NodeMetadata*> nodes;
    if (!search_query_.empty()) {
        nodes = registry.Search(search_query_, true);
    } else if (showing_all_in_category_) {
        nodes = registry.GetByCategory(current_category_, true);
    }

    if (!nodes.empty() && support_filter_mode_ != SupportFilterMode::All) {
        const auto filtered_nodes = ApplySupportFilter(nodes);
        ImGui::SameLine();
        ImGui::TextDisabled("%zu/%zu", filtered_nodes.size(), nodes.size());
    }

    ImGui::PopStyleVar(2);
}

std::vector<const cyxwiz::NodeMetadata*> NodeBrowserPanel::Curated(std::vector<const cyxwiz::NodeMetadata*> nodes) const {
    if (support_filter_mode_ != SupportFilterMode::All) return nodes;
    nodes.erase(std::remove_if(nodes.begin(), nodes.end(),
                               [](const cyxwiz::NodeMetadata* m) {
                                   return !m || cyxwiz::palette::PlaceNode(*m).place != cyxwiz::palette::Place::Category;
                               }),
                nodes.end());
    return nodes;
}

void NodeBrowserPanel::RenderAdvancedSection() {
    const ui::Tokens& t = ui::CurrentTokens();
    std::map<std::string, std::vector<const cyxwiz::NodeMetadata*>> by_group;
    size_t count = 0;
    for (const auto* m : cyxwiz::NodeMetadataRegistry::Instance().GetAllMetadata()) {
        if (!m) continue;
        const auto placed = cyxwiz::palette::PlaceNode(*m);
        if (placed.place != cyxwiz::palette::Place::Advanced) continue;
        by_group[placed.advanced_group].push_back(m);
        ++count;
    }
    if (count == 0) return;
    const std::string label = std::string(cyxwiz::palette::kAdvancedOperations) + "  " + std::to_string(count) + "###advanced_operations";
    if (!ImGui::TreeNodeEx(label.c_str(), ImGuiTreeNodeFlags_SpanAvailWidth)) {
        ImGui::Spacing();
        return;
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim, "Table steps a Preparation Recipe holds. Data Studio > Prepare adds them for you; "
                                   "add one here for a graph built step by step.");
    ImGui::PopTextWrapPos();
    for (const auto& group : cyxwiz::palette::AdvancedGroups()) {
        auto& nodes = by_group[group];
        if (nodes.empty()) continue;
        std::sort(nodes.begin(), nodes.end(),
                  [](const cyxwiz::NodeMetadata* a, const cyxwiz::NodeMetadata* b) { return a->name < b->name; });
        ImGui::Spacing();
        ImGui::TextColored(t.text_dim, "%s", group.c_str());
        RenderNodeGrid(nodes);
    }
    ImGui::TreePop();
    ImGui::Spacing();
}

void NodeBrowserPanel::RenderCantRunSection() {
    const ui::Tokens& t = ui::CurrentTokens();
    std::vector<const cyxwiz::NodeMetadata*> nodes;
    for (const auto* m : cyxwiz::NodeMetadataRegistry::Instance().GetAllMetadata())
        if (m && cyxwiz::palette::PlaceNode(*m).place == cyxwiz::palette::Place::CantRun) nodes.push_back(m);
    if (nodes.empty()) return;
    // By category, then name: what is missing where.
    std::sort(nodes.begin(), nodes.end(), [](const cyxwiz::NodeMetadata* a, const cyxwiz::NodeMetadata* b) {
        const std::string ca = cyxwiz::GetCategoryDisplayName(a->category), cb = cyxwiz::GetCategoryDisplayName(b->category);
        return ca != cb ? ca < cb : a->name < b->name;
    });
    const std::string label = std::string(cyxwiz::palette::kCantRunYet) + "  " + std::to_string(nodes.size()) + "###cant_run_yet";
    if (!ImGui::TreeNodeEx(label.c_str(), ImGuiTreeNodeFlags_SpanAvailWidth)) {
        ImGui::Spacing();
        return;
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim, "What the Engine cannot run yet; each leaves this list when it runs. A saved graph that "
                                   "has one still opens and says why it cannot run. Hover a node for the reason.");
    ImGui::PopTextWrapPos();
    RenderNodeGrid(nodes);
    ImGui::TreePop();
    ImGui::Spacing();
}

void NodeBrowserPanel::RenderCategorySection(cyxwiz::NodeCategory category) {
    auto& registry = cyxwiz::NodeMetadataRegistry::Instance();
    auto nodes = Curated(registry.GetByCategory(category, true));
    nodes = ApplySupportFilter(nodes);

    if (nodes.empty()) return;

    const ui::Tokens& t = ui::CurrentTokens();
    std::string cat_name = cyxwiz::GetCategoryDisplayName(category);
    SectionTitle(cat_name.c_str());

    // Determine how many nodes to show (rows of the columns this width fits)
    const int collapsed_count = ColumnsFor(ImGui::GetContentRegionAvail().x) * VISIBLE_ROWS_COLLAPSED;
    int max_visible = category_show_all_[category] ? -1 : collapsed_count;

    // Render nodes in grid
    RenderNodeGrid(nodes, max_visible);

    // "Show all" button if there are more nodes
    if (!category_show_all_[category] && (int)nodes.size() > collapsed_count) {
        float button_x = ImGui::GetContentRegionAvail().x - 60;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + button_x);

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);

        if (ImGui::SmallButton(("Show all##" + cat_name).c_str())) {
            NavigateToCategory(category);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("View all %zu nodes in %s category", nodes.size(), cat_name.c_str());
        }

        // Circle arrow icon next to "Show all"
        ImGui::SameLine();
        ImGui::TextUnformatted(ICON_FA_CHEVRON_RIGHT);

        ImGui::PopStyleColor(2);
    }

    ImGui::Spacing();
    ImGui::Spacing();
}

void NodeBrowserPanel::SyncUserNodes() {
    auto& registry = cyxwiz::NodeMetadataRegistry::Instance();
    auto& engine_config = cyxwiz::core::EngineConfig::Instance();
    if (!user_nodes_loaded_) {
        const cyxwiz::core::NodesPanelConfig saved = engine_config.GetNodesPanel();
        registry.SetUserNodes(NodeTypes(saved.favourites), NodeTypes(saved.recent));
        saved_favourites_ = saved.favourites;
        saved_recent_ = saved.recent;
        user_nodes_loaded_ = true;
    }
    std::vector<int> favourites = TypeNumbers(registry.FavoriteTypes());
    std::vector<int> recent = TypeNumbers(registry.RecentTypes());
    if (favourites == saved_favourites_ && recent == saved_recent_) return;
    cyxwiz::core::NodesPanelConfig config = engine_config.GetNodesPanel();
    config.favourites = favourites;
    config.recent = recent;
    engine_config.SetNodesPanel(config);
    engine_config.Save();
    saved_favourites_ = std::move(favourites);
    saved_recent_ = std::move(recent);
}

void NodeBrowserPanel::RenderFavouritesSection() {
    auto& registry = cyxwiz::NodeMetadataRegistry::Instance();
    const auto favourites = registry.GetFavorites();
    const std::string count = std::to_string(favourites.size());
    SectionTitle(ICON_FA_STAR " Favourites", count.c_str());
    if (favourites.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled("Click the star on a node, or right-click it, to keep it here.");
        ImGui::PopTextWrapPos();
    } else {
        RenderNodeGrid(favourites);
    }
    ImGui::Spacing();
}

void NodeBrowserPanel::RenderRecentSection() {
    const cyxwiz::core::NodesPanelConfig config = cyxwiz::core::EngineConfig::Instance().GetNodesPanel();
    if (!config.show_recent) return;
    const auto recent = cyxwiz::NodeMetadataRegistry::Instance().GetRecent(static_cast<size_t>(config.recent_count));
    if (recent.empty()) return;
    SectionTitle("Recent");
    const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
    for (size_t i = 0; i < recent.size(); ++i) {
        const cyxwiz::NodeMetadata* metadata = recent[i];
        ImGui::PushID(static_cast<int>(metadata->type));
        const float width = ui::ChipButtonWidth(metadata->name.c_str(), true);
        if (i > 0) {
            ImGui::SameLine(0, 6);
            if (ImGui::GetCursorScreenPos().x + width > right) ImGui::NewLine();
        }
        const bool can_add = !metadata->IsTemplate() && !IsSupportBlocked(metadata) && metadata->made_by.empty();
        if (ui::ChipButton(metadata->name.c_str(), GetNodeColor(metadata->category)) && can_add) {
            CreateNodeAtMouse(metadata);
        }
        if (can_add && ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
            ImGui::SetDragDropPayload("NODE_TYPE", &metadata->type, sizeof(cyxwiz::NodeType));
            ImGui::Text("%s %s", metadata->icon.c_str(), metadata->name.c_str());
            ImGui::EndDragDropSource();
        } else if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", can_add ? "Click to add again; drag onto the canvas" : "Can't be added to the graph");
        }
        ImGui::PopID();
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("The last nodes you added, newest first.");
    ImGui::PopTextWrapPos();
    ImGui::Spacing();
}

void NodeBrowserPanel::RenderNodeGrid(const std::vector<const cyxwiz::NodeMetadata*>& nodes, int max_visible) {
    if (nodes.empty()) return;

    float available_width = ImGui::GetContentRegionAvail().x;
    grid_columns_ = ColumnsFor(available_width);
    float card_width = (available_width - (grid_columns_ - 1) * GRID_SPACING) / grid_columns_;

    int visible_count = (max_visible > 0) ? std::min((int)nodes.size(), max_visible) : (int)nodes.size();

    for (int i = 0; i < visible_count; ++i) {
        if (i % grid_columns_ != 0) {
            ImGui::SameLine(0, GRID_SPACING);
        }

        RenderNodeCard(nodes[i], card_width);
    }
}

void NodeBrowserPanel::RenderExtensionGrid(const std::vector<cyxwiz::ExtensionPaletteEntry>& entries) {
    if (entries.empty()) return;
    auto& registry = cyxwiz::ExtensionNodeRegistry::Instance();

    // Card data: the descriptor's metadata, copied (a plugin may unload).
    std::vector<cyxwiz::NodeMetadata> metadata;
    std::vector<const cyxwiz::ExtensionPaletteEntry*> shown;
    metadata.reserve(entries.size());
    for (const auto& entry : entries) {
        const auto descriptor = registry.Find(entry.type_id);
        if (!descriptor) continue;
        metadata.push_back(descriptor->metadata);
        // The engine's plug icon, as the catalog uses for plugin nodes.
        if (metadata.back().icon.empty()) metadata.back().icon = ICON_FA_PLUG;
        shown.push_back(&entry);
    }

    float available_width = ImGui::GetContentRegionAvail().x;
    grid_columns_ = ColumnsFor(available_width);
    float card_width = (available_width - (grid_columns_ - 1) * GRID_SPACING) / grid_columns_;
    for (size_t i = 0; i < metadata.size(); ++i) {
        if (i % grid_columns_ != 0) {
            ImGui::SameLine(0, GRID_SPACING);
        }
        RenderNodeCard(&metadata[i], card_width, shown[i]);
    }
}

void NodeBrowserPanel::RenderExtensionSection(bool show_all) {
    // Only the All filter lists extension nodes: the other filters sort
    // built-in nodes by their support facts, which extension nodes lack.
    if (support_filter_mode_ != SupportFilterMode::All) return;

    auto entries = cyxwiz::BuildExtensionPalette(cyxwiz::ExtensionNodeRegistry::Instance().All());
    // Narrowed to one plugin (Apps > a loaded plugin).
    if (show_all && !provider_filter_.empty()) {
        entries.erase(std::remove_if(entries.begin(), entries.end(),
                                     [&](const cyxwiz::ExtensionPaletteEntry& entry) {
                                         std::string provider_id;
                                         std::string type_name;
                                         return !cyxwiz::SplitExtensionTypeId(entry.type_id, provider_id, type_name) ||
                                                provider_id != provider_filter_;
                                     }),
                      entries.end());
    }
    std::vector<std::pair<std::string, std::string>> provider_names;
    for (const auto& entry : entries) {
        std::string provider_id;
        std::string type_name;
        if (!cyxwiz::SplitExtensionTypeId(entry.type_id, provider_id, type_name)) continue;
        if (const auto* plugin = cyxwiz::plugin::PluginManager::Instance().GetLoadedPlugin(provider_id)) {
            provider_names.emplace_back(provider_id, plugin->manifest.name);
        }
    }
    for (auto& name : cyxwiz::python_ui::BundleNames()) provider_names.push_back(std::move(name));

    // Section header, as other categories draw it, with the summary.
    const std::string cat_name = cyxwiz::GetCategoryDisplayName(cyxwiz::NodeCategory::Plugin);
    ImGui::PushStyleColor(ImGuiCol_Text, ui::CurrentTokens().text_dim);
    ImGui::TextUnformatted(cat_name.c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("%s", cyxwiz::ExtensionPaletteSummary(entries, provider_names).c_str());
    ImGui::PopStyleColor();
    if (show_all && !provider_filter_.empty()) {
        if (cyxwiz::ui::LinkButton("Show all plugins##provider_filter")) provider_filter_.clear();
    }

    if (entries.empty()) {
        ImGui::PushID("no_extension_nodes");
        if (ImGui::BeginChild("##empty", ImVec2(0.0f, 0.0f),
                              ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY |
                                  ImGuiChildFlags_AlwaysUseWindowPadding)) {
            ImGui::TextUnformatted("No extension nodes are loaded");
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextDisabled("Load a plugin that provides nodes and they appear here.");
            ImGui::PopTextWrapPos();
            if (cyxwiz::ui::SecondaryButton("Open Plugin Manager", node_editor_ != nullptr,
                                            "The node editor is not available")) {
                node_editor_->OpenPluginManager();
            }
        }
        ImGui::EndChild();
        ImGui::PopID();
        ImGui::Spacing();
        ImGui::Spacing();
        return;
    }

    // One grid per plugin category ("RL / Simulation", ...).
    size_t shown = 0;
    const size_t limit = show_all ? entries.size() : static_cast<size_t>(ColumnsFor(ImGui::GetContentRegionAvail().x) * VISIBLE_ROWS_COLLAPSED);
    for (size_t i = 0; i < entries.size() && shown < limit;) {
        const std::string& group = entries[i].group;
        std::vector<cyxwiz::ExtensionPaletteEntry> group_entries;
        for (; i < entries.size() && entries[i].group == group && shown < limit; ++i, ++shown) {
            group_entries.push_back(entries[i]);
        }
        ImGui::TextDisabled("%s", group.c_str());
        RenderExtensionGrid(group_entries);
    }

    if (!show_all && entries.size() > limit) {
        if (cyxwiz::ui::LinkButton("Show all##plugins")) {
            NavigateToCategory(cyxwiz::NodeCategory::Plugin);
        }
    }
    ImGui::Spacing();
    ImGui::Spacing();
}

namespace {

// Up to two centred lines of `name` within `width`. A word wider than the
// card breaks after a hyphen or before a capital ("Squeeze-" "Excitation",
// "Instance" "Norm"); what is left past two lines ends in "...".
void DrawCardName(ImDrawList* dl, const std::string& name, float x, float width, float y, ImU32 colour) {
    const float max_width = width - 6.0f;
    const auto fits = [&](size_t begin, size_t end) {
        return ImGui::CalcTextSize(name.c_str() + begin, name.c_str() + end).x <= max_width;
    };
    struct Range {
        size_t begin, end;
    };
    std::vector<Range> pieces;  // words, and the parts of a word too wide for a line
    for (size_t at = 0; at < name.size();) {
        const size_t space = name.find(' ', at);
        const size_t end = space == std::string::npos ? name.size() : space;
        size_t begin = at;
        while (begin < end && !fits(begin, end)) {
            size_t cut = end - 1;
            while (cut > begin + 1 && !fits(begin, cut)) --cut;
            for (size_t i = cut; i > begin + 2; --i) {
                const char c = name[i - 1];
                if (c == '-' || std::isupper(static_cast<unsigned char>(name[i])) != 0) {
                    cut = i;
                    break;
                }
            }
            pieces.push_back({begin, cut});
            begin = cut;
        }
        if (begin < end) pieces.push_back({begin, end});
        at = end + 1;
    }
    std::vector<Range> lines;
    for (const Range& piece : pieces) {
        if (!lines.empty() && fits(lines.back().begin, piece.end)) {
            lines.back().end = piece.end;
        } else {
            lines.push_back(piece);
        }
    }
    if (lines.size() > 2) {
        lines.resize(2);
        lines[1].end = name.size();
    }
    const float line_height = ImGui::GetTextLineHeight();
    for (size_t i = 0; i < lines.size(); ++i) {
        std::string text = name.substr(lines[i].begin, lines[i].end - lines[i].begin);
        if (!fits(lines[i].begin, lines[i].end)) {
            const auto fits_text = [max_width](const std::string& s) { return ImGui::CalcTextSize(s.c_str()).x <= max_width; };
            while (text.size() > 1 && !fits_text(text + "...")) text.pop_back();
            text += "...";
        }
        const float w = ImGui::CalcTextSize(text.c_str()).x;
        dl->AddText(ImVec2(x + (width - w) * 0.5f, y + line_height * static_cast<float>(i)), colour, text.c_str());
    }
}

}  // namespace

void NodeBrowserPanel::RenderNodeCard(const cyxwiz::NodeMetadata* metadata, float card_width,
                                      const cyxwiz::ExtensionPaletteEntry* extension) {
    if (!metadata) return;

    auto& registry = cyxwiz::NodeMetadataRegistry::Instance();

    // Extension nodes share one NodeType: they are keyed by type id.
    if (extension) {
        ImGui::PushID(extension->type_id.c_str());
    } else {
        ImGui::PushID(static_cast<int>(metadata->type));
    }
    const bool pinned = extension ? extension->type_id == pinned_extension_id_
                                  : (pinned_extension_id_.empty() && metadata->type == pinned_node_type_);
    const auto select = [&]() {
        if (extension) {
            pinned_extension_id_ = extension->type_id;
            pinned_node_type_ = cyxwiz::NodeType::Unknown;
            if (on_extension_select_) on_extension_select_(extension->type_id);
        } else {
            pinned_extension_id_.clear();
            pinned_node_type_ = metadata->type;
            if (on_node_select_) on_node_select_(metadata->type);
        }
    };

    ImVec2 card_size(card_width, NODE_CARD_HEIGHT);
    ImVec2 cursor_start = ImGui::GetCursorScreenPos();

    // Invisible button for interaction
    ImGui::InvisibleButton("##card", card_size);

    // Preserve the full layout while avoiding formatting and draw-list work
    // for cards clipped outside the NodeGrid child window.
    if (!ImGui::IsItemVisible()) {
        ImGui::PopID();
        return;
    }

    bool hovered = ImGui::IsItemHovered();
    const bool support_blocked = IsSupportBlocked(metadata);
    const bool can_add_to_graph = !metadata->IsTemplate() && !support_blocked && metadata->made_by.empty();

    // The favourite star, top right (favourites are kept per NodeType;
    // extension nodes share one).
    const bool has_star = !extension && can_add_to_graph;
    const float star_size = ImGui::GetFontSize();
    const ImVec2 star_min(cursor_start.x + card_width - star_size - 4.0f, cursor_start.y + 3.0f);
    const ImVec2 star_max(star_min.x + star_size, star_min.y + star_size);
    const bool star_hovered = has_star && hovered && ImGui::IsMouseHoveringRect(star_min, star_max);
    const char* const favourite_label = metadata->is_favorite ? "Remove from Favourites" : "Add to Favourites";

    // Single click pins the node in the Info Panel; on the star it toggles
    // the favourite.
    if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
        if (star_hovered) {
            registry.ToggleFavorite(metadata->type);
        } else {
            select();
        }
    }

    // Handle double-click to create node
    if (hovered && !star_hovered && ImGui::IsMouseDoubleClicked(0) && can_add_to_graph) {
        CreateNodeAtMouse(metadata, extension);
    }

    // Handle drag-drop source (only for graph-addable nodes)
    if (can_add_to_graph && ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
        if (extension) {
            ImGui::SetDragDropPayload("EXTENSION_NODE", extension->type_id.data(),
                                      extension->type_id.size());
        } else {
            ImGui::SetDragDropPayload("NODE_TYPE", &metadata->type, sizeof(cyxwiz::NodeType));
        }
        ImGui::Text("%s %s", metadata->icon.c_str(), metadata->name.c_str());
        dragging_node_ = metadata;
        ImGui::EndDragDropSource();
    }

    // Context menu (right-click)
    if (ImGui::BeginPopupContextItem("##NodeContextMenu")) {
        ImGui::TextDisabled("%s %s", metadata->icon.c_str(), metadata->name.c_str());
        ImGui::Separator();

        if (can_add_to_graph) {
            if (ImGui::MenuItem(ICON_FA_PLUS " Add to Canvas", "Double-click")) {
                CreateNodeAtMouse(metadata, extension);
            }

            if (has_star && ImGui::MenuItem((std::string(ICON_FA_STAR " ") + favourite_label).c_str())) {
                registry.ToggleFavorite(metadata->type);
            }
        } else if (!metadata->made_by.empty()) {
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 24.0f);
            ImGui::TextUnformatted(metadata->made_by.c_str());
            ImGui::PopTextWrapPos();
        } else if (metadata->IsTemplate()) {
            ImGui::TextDisabled(ICON_FA_CLOCK " Coming Soon");
            if (ImGui::MenuItem(ICON_FA_THUMBS_UP " Vote for this feature")) {
                // TODO: Increment user_votes and save preferences
            }
        } else {
            ImGui::TextDisabled(ICON_FA_TRIANGLE_EXCLAMATION " Blocked");
            const std::string reason = FirstUnsupportedSupportReason(metadata);
            if (!reason.empty()) {
                ImGui::TextWrapped("%s", reason.c_str());
            }
        }

        ImGui::Separator();

        if (ImGui::MenuItem(ICON_FA_CIRCLE_INFO " Show details")) {
            // Pin this node in the Info Panel
            select();
        }

        if (ImGui::MenuItem(ICON_FA_COPY " Copy Node Name")) {
            ImGui::SetClipboardText(metadata->name.c_str());
        }

        ImGui::EndPopup();
    }

    // Tooltip
    if (star_hovered) {
        ImGui::SetTooltip("%s", favourite_label);
    } else if (hovered && !ImGui::IsMouseDragging(0)) {
        RenderNodeTooltip(metadata);
    }

    // Draw the card
    const ui::Tokens& t = ui::CurrentTokens();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    // Icon position (centered horizontally)
    float icon_x = cursor_start.x + (card_width - NODE_ICON_SIZE) * 0.5f;
    float icon_y = cursor_start.y + 4;
    ImVec2 icon_pos(icon_x, icon_y);
    ImVec2 icon_size(NODE_ICON_SIZE, NODE_ICON_SIZE);

    // Draw node icon with color
    RenderNodeIcon(metadata, icon_pos, icon_size, extension);

    // Support: only the exceptions are marked (a "Real" chip sat over every
    // icon), as a dot in the state's colour; the word is in the tooltip.
    if (const auto* support_state = FindSupportState(metadata); support_state && support_state->value != "real") {
        draw_list->AddCircleFilled(ImVec2(cursor_start.x + 7.0f, cursor_start.y + 7.0f), 4.0f,
                                   ui::ToU32(SupportStateBadgeColor(support_state->value)));
    }

    // Draw port indicators
    RenderPortIndicators(metadata, icon_pos, icon_size);

    if (has_star) {
        const ImVec4 star_colour = metadata->is_favorite ? t.accent_text
                                   : star_hovered        ? t.text_dim
                                                         : ui::WithAlpha(t.text_faint, 0.55f);
        draw_list->AddText(nullptr, star_size, star_min, ui::ToU32(star_colour), ICON_FA_STAR);
    }

    // The name below the icon, on up to two lines that fit the card (it was
    // cut after 10 characters: "Deploy to ...", "Export Par...").
    // Dim text for nodes that cannot currently be added.
    const ImU32 text_color = !can_add_to_graph ? ui::ToU32(t.text_faint) : ui::ToU32(t.text);
    DrawCardName(draw_list, metadata->name, cursor_start.x, card_width, icon_y + NODE_ICON_SIZE + 4, text_color);

    // Draw "Coming Soon" badge for template nodes
    if (metadata->IsTemplate()) {
        // Draw small badge in top-right corner
        const char* badge_text = metadata->badge.empty() ? "Soon" : metadata->badge.c_str();
        // Truncate long badges
        std::string badge_display = badge_text;
        if (badge_display.length() > 6) {
            badge_display = badge_display.substr(0, 4) + "..";
        }

        ImVec2 badge_text_size = ImGui::CalcTextSize(badge_display.c_str());
        float badge_padding = 3.0f;
        float badge_width = badge_text_size.x + badge_padding * 2;
        float badge_height = badge_text_size.y + badge_padding;

        float badge_x = cursor_start.x + card_width - badge_width - 2;
        float badge_y = cursor_start.y + 2;

        // Badge background (pending: "Coming Soon")
        draw_list->AddRectFilled(
            ImVec2(badge_x, badge_y),
            ImVec2(badge_x + badge_width, badge_y + badge_height),
            ui::ToU32(ui::WithAlpha(t.pending, 230.0f / 255.0f)),
            3.0f
        );

        // Badge text
        draw_list->AddText(
            ImVec2(badge_x + badge_padding, badge_y + badge_padding * 0.5f),
            ui::ToU32(ui::ReadableOn(t, t.pending)),
            badge_display.c_str()
        );
    }

    // Pinned card: accent outline. Hover: light outline and a preview in the
    // Info Panel (it returns to the pinned node when the hover ends).
    if (pinned) {
        draw_list->AddRect(cursor_start,
                          ImVec2(cursor_start.x + card_width, cursor_start.y + card_size.y),
                          ui::ToU32(t.accent), 4.0f, 0, 2.0f);
    }
    if (hovered) {
        draw_list->AddRect(cursor_start,
                          ImVec2(cursor_start.x + card_width, cursor_start.y + card_size.y),
                          ui::ToU32(ui::WithAlpha(t.accent_text, 110.0f / 255.0f)), 4.0f);
        if (extension) {
            if (on_extension_hover_) on_extension_hover_(extension->type_id);
        } else if (on_node_hover_) {
            on_node_hover_(metadata->type);
        }
    }

    ImGui::PopID();
}

void NodeBrowserPanel::RenderNodeIcon(const cyxwiz::NodeMetadata* metadata, ImVec2 icon_pos, ImVec2 size,
                                      const cyxwiz::ExtensionPaletteEntry* extension) {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    // Use the passed icon position (calculated in RenderNodeCard)
    ImVec2 p_min = icon_pos;
    ImVec2 p_max(icon_pos.x + size.x, icon_pos.y + size.y);

    const ui::Tokens& t = ui::CurrentTokens();
    // Get category color
    ImU32 bg_color = GetNodeColor(metadata->category);
    ImU32 border_color = GetNodeColorDark(metadata->category);

    if (extension && (extension->color >> 24) != 0) {
        bg_color = extension->color;
    }

    // Template nodes are grayed out
    if (metadata->IsTemplate()) {
        bg_color = ui::ToU32(t.bg_raised);
        border_color = ui::ToU32(t.border);
    }

    // Draw rounded rectangle background
    draw_list->AddRectFilled(p_min, p_max, bg_color, 6.0f);
    draw_list->AddRect(p_min, p_max, border_color, 6.0f, 0, 2.0f);

    // Draw icon character in center using the icon font
    if (!metadata->icon.empty()) {
        // FontAwesome icons are merged into the main font
        ImFont* icon_font = ImGui::GetFont();
        float scaled_icon_size = size.y * 0.55f;  // Icon takes 55% of box height
        ImVec2 icon_text_size = icon_font->CalcTextSizeA(scaled_icon_size, FLT_MAX, 0.0f, metadata->icon.c_str());
        ImVec2 icon_text_pos(
            p_min.x + (size.x - icon_text_size.x) * 0.5f,
            p_min.y + (size.y - icon_text_size.y) * 0.5f
        );
        draw_list->AddText(icon_font, scaled_icon_size, icon_text_pos, ui::ToU32(ui::ReadableOn(t, ImGui::ColorConvertU32ToFloat4(bg_color))), metadata->icon.c_str());
    }
}

void NodeBrowserPanel::RenderPortIndicators(const cyxwiz::NodeMetadata* metadata,
                                            ImVec2 icon_pos, ImVec2 icon_size) {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    // Input ports (left side triangles pointing LEFT - into the node)
    if (!metadata->inputs.empty()) {
        float tri_size = 5.0f;
        float y_center = icon_pos.y + icon_size.y * 0.5f;

        // Draw small triangle pointing left (<)
        ImVec2 p1(icon_pos.x - 1, y_center);  // tip points left
        ImVec2 p2(icon_pos.x + tri_size - 1, y_center - tri_size);
        ImVec2 p3(icon_pos.x + tri_size - 1, y_center + tri_size);

        draw_list->AddTriangleFilled(p1, p2, p3, ui::ToU32(ui::CurrentTokens().text_faint));
    }

    // Output ports (right side triangles pointing RIGHT - out of the node)
    if (!metadata->outputs.empty()) {
        float tri_size = 5.0f;
        float y_center = icon_pos.y + icon_size.y * 0.5f;
        float x_right = icon_pos.x + icon_size.x;

        // Draw small triangle pointing right (>)
        ImVec2 p1(x_right - tri_size + 1, y_center - tri_size);
        ImVec2 p2(x_right - tri_size + 1, y_center + tri_size);
        ImVec2 p3(x_right + 1, y_center);  // tip points right

        draw_list->AddTriangleFilled(p1, p2, p3, ui::ToU32(ui::CurrentTokens().text_faint));
    }
}

void NodeBrowserPanel::RenderNodeTooltip(const cyxwiz::NodeMetadata* metadata) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::BeginTooltip();

    // Header with icon and name
    ImGui::TextColored(ui::CurrentTokens().text_bright, "%s %s",
                       metadata->icon.c_str(), metadata->name.c_str());

    // Brief description
    if (!metadata->brief_description.empty()) {
        ImGui::TextWrapped("%s", metadata->brief_description.c_str());
    }

    ImGui::Separator();

    // Inputs/Outputs
    if (!metadata->inputs.empty()) {
        ImGui::TextColored(t.series[3], "Inputs:");
        for (const auto& input : metadata->inputs) {
            ImGui::BulletText("%s", input.name.c_str());
        }
    }

    if (!metadata->outputs.empty()) {
        ImGui::TextColored(t.series[4], "Outputs:");
        for (const auto& output : metadata->outputs) {
            ImGui::BulletText("%s", output.name.c_str());
        }
    }

    if (!metadata->support_axes.empty()) {
        ImGui::Separator();
        if (const auto* support_state = FindSupportState(metadata)) {
            const ImVec4 state_color = support_state->supported
                ? t.success
                : t.error;
            ImGui::TextColored(
                state_color,
                "Support state: %s",
                SupportStateLabel(support_state->value).c_str());
            if (!support_state->reason.empty()) {
                ImGui::TextWrapped("%s", support_state->reason.c_str());
            }
        }
        ImGui::TextColored(t.text_bright, "Support details:");
        for (const auto& axis : metadata->support_axes) {
            const ImVec4 axis_color = axis.supported
                ? t.success
                : t.error;
            const char* icon = axis.supported
                ? ICON_FA_CIRCLE_CHECK
                : ICON_FA_TRIANGLE_EXCLAMATION;

            ImGui::TextColored(axis_color, "%s", icon);
            ImGui::SameLine();
            ImGui::Text("%s:", axis.name.c_str());
            ImGui::SameLine();
            const std::string axis_value = SupportAxisValueLabel(axis.value);
            ImGui::TextDisabled("%s", axis_value.c_str());

            const bool show_supported_reason =
                axis.name == "Task Type" ||
                axis.name == "Training Role" ||
                axis.name == "Workflow Lane" ||
                axis.name == "Implementation Owner";
            if (!axis.reason.empty() &&
                (!axis.supported || show_supported_reason)) {
                ImGui::Indent(16.0f);
                ImGui::TextWrapped("%s", axis.reason.c_str());
                ImGui::Unindent(16.0f);
            }
        }
    }

    // Status info for template nodes
    if (metadata->IsTemplate()) {
        ImGui::Separator();
        ImGui::TextColored(t.pending,
                          ICON_FA_CLOCK " Coming Soon");
        if (metadata->user_votes > 0) {
            ImGui::SameLine();
            ImGui::Text("(%d votes)", metadata->user_votes);
        }
    }

    // Usage hint
    ImGui::Separator();
    if (metadata->IsTemplate()) {
        ImGui::TextDisabled("Template node: not addable yet");
    } else if (IsSupportBlocked(metadata)) {
        const std::string reason = FirstUnsupportedSupportReason(metadata);
        ImGui::TextColored(t.error,
                           ICON_FA_TRIANGLE_EXCLAMATION " Blocked from graph add");
        if (!reason.empty()) {
            ImGui::TextWrapped("%s", reason.c_str());
        }
    } else {
        ImGui::TextDisabled("Double-click or drag to add");
    }

    ImGui::EndTooltip();
}

void NodeBrowserPanel::NavigateToCategory(cyxwiz::NodeCategory category) {
    provider_filter_.clear();
    showing_all_in_category_ = true;
    current_category_ = category;
    category_show_all_[category] = true;
}

void NodeBrowserPanel::NavigateBack() {
    provider_filter_.clear();
    showing_all_in_category_ = false;
    current_category_ = cyxwiz::NodeCategory::Unknown;
    // Reset all category show_all states
    for (auto& pair : category_show_all_) {
        pair.second = false;
    }
}

void NodeBrowserPanel::CreateNodeAtMouse(const cyxwiz::NodeMetadata* metadata,
                                         const cyxwiz::ExtensionPaletteEntry* extension) {
    if (!node_editor_ || !metadata) return;

    // Only create nodes that central support metadata says are graph-addable.
    if (metadata->IsTemplate() || IsSupportBlocked(metadata) || !metadata->made_by.empty()) {
        return;
    }

    // Added at the centre of the canvas view, like the toolbar's menus.
    if (extension) {
        node_editor_->AddNodeFromMenu(cyxwiz::NodeType::PluginCustom, extension->type_id);
    } else {
        node_editor_->AddNodeFromMenu(metadata->type, metadata->name);
    }
}

ImU32 NodeBrowserPanel::GetNodeColor(cyxwiz::NodeCategory category) const {
    return ui::ToU32(NodeCategoryColour(category));
}

std::vector<const cyxwiz::NodeMetadata*> NodeBrowserPanel::ApplySupportFilter(
    const std::vector<const cyxwiz::NodeMetadata*>& nodes) const {
    if (support_filter_mode_ == SupportFilterMode::All) {
        return nodes;
    }

    std::vector<const cyxwiz::NodeMetadata*> filtered;
    filtered.reserve(nodes.size());
    for (const auto* node : nodes) {
        if (NodeMatchesSupportFilter(node)) {
            filtered.push_back(node);
        }
    }
    return filtered;
}

bool NodeBrowserPanel::NodeMatchesSupportFilter(
    const cyxwiz::NodeMetadata* metadata) const {
    if (!metadata) return false;

    const bool blocked = IsSupportBlocked(metadata);
    switch (support_filter_mode_) {
    case SupportFilterMode::Pipeline:
        return HasPipelineSupport(metadata);
    case SupportFilterMode::Training:
        return HasTrainingSupport(metadata);
    case SupportFilterMode::UiOnly:
        return IsUiOnly(metadata);
    case SupportFilterMode::Blocked:
        return blocked;
    case SupportFilterMode::Favourites:
        return metadata->is_favorite;
    case SupportFilterMode::All:
        return true;
    }
    return true;
}

bool NodeBrowserPanel::IsSupportBlocked(
    const cyxwiz::NodeMetadata* metadata) const {
    return metadata != nullptr && cyxwiz::IsNodeSupportBlocked(*metadata);
}

bool NodeBrowserPanel::HasPipelineSupport(
    const cyxwiz::NodeMetadata* metadata) const {
    return metadata != nullptr &&
           metadata->IsImplemented() &&
           !IsSupportBlocked(metadata) &&
           HasSupportedAxis(metadata, "Runtime") &&
           HasSupportedAxis(metadata, "Pipeline Executor");
}

bool NodeBrowserPanel::HasTrainingSupport(
    const cyxwiz::NodeMetadata* metadata) const {
    return metadata != nullptr &&
           metadata->IsImplemented() &&
           !IsSupportBlocked(metadata) &&
           HasSupportedAxis(metadata, "Training Backend") &&
           HasSupportedAxis(metadata, "Compile") &&
           HasSupportedAxis(metadata, "Training");
}

bool NodeBrowserPanel::IsUiOnly(
    const cyxwiz::NodeMetadata* metadata) const {
    return metadata != nullptr &&
           metadata->IsImplemented() &&
           !IsSupportBlocked(metadata) &&
           !IsTrainingCategory(metadata->category) &&
           !HasAxis(metadata, "Runtime") &&
           !HasAxis(metadata, "Pipeline Executor") &&
           !HasAxis(metadata, "Training Backend") &&
           !HasAxis(metadata, "Compile") &&
           !HasAxis(metadata, "Training");
}

const char* NodeBrowserPanel::GetSupportFilterLabel(
    SupportFilterMode mode) const {
    switch (mode) {
    case SupportFilterMode::Pipeline:
        return "Pipeline";
    case SupportFilterMode::Training:
        return "Training";
    case SupportFilterMode::UiOnly:
        return "UI-only";
    case SupportFilterMode::Blocked:
        return "Blocked";
    case SupportFilterMode::Favourites:
        return "Favourites";
    case SupportFilterMode::All:
        return "All";
    }
    return "All";
}

ImU32 NodeBrowserPanel::GetNodeColorDark(cyxwiz::NodeCategory category) const {
    // The category colour a step toward the window, for borders
    return ui::ToU32(ui::Mix(NodeCategoryColour(category), ui::CurrentTokens().bg_window, 0.3f));
}

} // namespace gui
