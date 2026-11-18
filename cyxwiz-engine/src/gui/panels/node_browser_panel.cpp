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
#include "../ui_widgets.h"
#include "../patterns/pattern_library.h"
#include "../icons.h"
#include <imgui.h>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace gui {

namespace ui = cyxwiz::ui;
namespace np = cyxwiz::nodes_panel;

namespace {

constexpr const char* kPlugins = "Plugins";
constexpr const char* kExamples = "Examples";

// A small dimmed title (no rule under it: spacing separates the areas).
void SmallTitle(const char* title) {
    ImGui::PushStyleColor(ImGuiCol_Text, ui::CurrentTokens().text_dim);
    ImGui::TextUnformatted(title);
    ImGui::PopStyleColor();
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

ImVec4 MarkColour(np::Mark mark) {
    const ui::Tokens& t = ui::CurrentTokens();
    switch (mark) {
    case np::Mark::Partial: return t.warning;
    case np::Mark::Blocked: return t.error;
    case np::Mark::Pending: return t.pending;
    case np::Mark::None: break;
    }
    return t.text_faint;
}

ImVec4 ToneColour(np::Tone tone) {
    const ui::Tokens& t = ui::CurrentTokens();
    switch (tone) {
    case np::Tone::Good: return t.success;
    case np::Tone::Partial: return t.warning;
    case np::Tone::Blocked: return t.error;
    case np::Tone::Neutral: break;
    }
    return t.text_dim;
}

std::string PortList(const std::vector<np::PortLine>& ports) {
    std::string out;
    for (const auto& port : ports) {
        if (!out.empty()) out += ", ";
        out += port.name + " (" + port.kind + ")";
    }
    return out;
}

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

// One line of `text` within `width`, ending in "..." when it does not fit.
std::string FitLine(const std::string& text, float width) {
    if (ImGui::CalcTextSize(text.c_str()).x <= width) return text;
    std::string out = text;
    while (out.size() > 1 && ImGui::CalcTextSize((out + "...").c_str()).x > width) out.pop_back();
    return out + "...";
}

}  // namespace

NodeBrowserPanel::NodeBrowserPanel() {
    auto& registry = cyxwiz::NodeMetadataRegistry::Instance();
    if (!registry.IsInitialized()) registry.Initialize();
}

void NodeBrowserPanel::LoadPanelSettings() {
    if (settings_loaded_) return;
    settings_loaded_ = true;
    const cyxwiz::core::NodesPanelConfig config = cyxwiz::core::EngineConfig::Instance().GetNodesPanel();
    list_view_ = config.list_view;
    open_groups_ = std::set<std::string>(config.open_groups.begin(), config.open_groups.end());
}

void NodeBrowserPanel::SaveOpenGroups() {
    auto& engine_config = cyxwiz::core::EngineConfig::Instance();
    cyxwiz::core::NodesPanelConfig config = engine_config.GetNodesPanel();
    config.open_groups.assign(open_groups_.begin(), open_groups_.end());
    engine_config.SetNodesPanel(config);
    engine_config.Save();
}

void NodeBrowserPanel::Render() {
    auto& registry = cyxwiz::NodeMetadataRegistry::Instance();
    if (!registry.IsInitialized()) registry.Initialize();
    // Also while hidden: nodes added from the menus and the canvas count.
    SyncUserNodes();
    if (!visible_) return;
    LoadPanelSettings();

    ImGui::SetNextWindowSizeConstraints(ImVec2(PANEL_MIN_WIDTH, 300), ImVec2(450, FLT_MAX));
    if (focus_window_next_frame_) {
        ImGui::SetNextWindowFocus();
        focus_window_next_frame_ = false;
    }
    if (ImGui::Begin("Nodes", &visible_)) {
        HandleKeys();
        RenderSearchRow();

        const bool searching = search_[0] != '\0';
        std::vector<np::Group> groups;
        np::SearchResult found;
        std::vector<cyxwiz::ExtensionPaletteEntry> extension_results;
        size_t count = 0;
        if (searching) {
            found = np::OrderSearch(registry.Search(search_, true), search_, filter_);
            if (filter_ == np::Filter::All)
                for (auto& entry : cyxwiz::BuildExtensionPalette(cyxwiz::ExtensionNodeRegistry::Instance().All()))
                    if (cyxwiz::ExtensionPaletteEntryMatches(entry, search_)) extension_results.push_back(std::move(entry));
            count = found.named.size() + found.related.size() + extension_results.size();
        } else {
            std::vector<np::CategoryNodes> by_category;
            for (auto category : registry.GetCategories())
                by_category.push_back({category, registry.GetByCategory(category, true)});
            groups = np::BuildGroups(by_category, filter_);
            for (const auto& group : groups) count += group.nodes.size();
        }
        RenderFilterRow(count);
        ImGui::Spacing();

        ImGui::BeginChild("NodeGrid", ImVec2(0, 0), ImGuiChildFlags_None);
        if (searching) {
            RenderSearchResults(found, extension_results);
        } else {
            if (filter_ == np::Filter::All) {
                RenderStudioTools();
                RenderFavourites();
                RenderRecent();
            }
            RenderGroups(groups);
            if (filter_ == np::Filter::All) {
                RenderPluginsGroup();
                RenderAdvancedGroup();
                RenderCantRunGroup();
                RenderExamplesGroup();
            }
        }
        ImGui::EndChild();
    }
    ImGui::End();
}

void NodeBrowserPanel::HandleKeys() {
    // While the panel (or its list) has focus: / goes to the search, Esc
    // clears it. Ctrl+F stays the canvas's Find.
    if (!ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) || ImGui::GetIO().WantTextInput) return;
    if (ImGui::IsKeyPressed(ImGuiKey_Slash, false) && !ImGui::GetIO().KeyCtrl) focus_search_next_frame_ = true;
    if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) ClearSearch();
}

void NodeBrowserPanel::RenderSearchRow() {
    const bool has_text = search_[0] != '\0';
    const float clear_width = has_text ? ui::ButtonWidth(ICON_FA_XMARK, ui::ButtonSize::Small) + 6.0f : 0.0f;
    if (focus_search_next_frame_) {
        ImGui::SetKeyboardFocusHere();
        focus_search_next_frame_ = false;
    }
    ui::SearchField("##NodeSearch", search_, sizeof(search_), "Search nodes (/)",
                    ImGui::GetContentRegionAvail().x - clear_width, ImGuiInputTextFlags_EscapeClearsAll);
    // Esc in the field clears all of it, as Esc on the panel does.
    if (ImGui::IsItemDeactivated() && ImGui::IsKeyPressed(ImGuiKey_Escape, false)) ClearSearch();
    if (ImGui::IsItemHovered() && !has_text)
        ui::Tooltip("Search by name, keyword or description. / focuses the search while the panel has focus; "
                    "Esc clears it.");
    if (has_text) {
        ImGui::SameLine(0.0f, 6.0f);
        if (ui::GhostButton(ICON_FA_XMARK "##clear_search")) ClearSearch();
        if (ImGui::IsItemHovered()) ui::Tooltip("Clear the search (Esc)");
    }
}

void NodeBrowserPanel::RenderFilterRow(size_t count) {
    const ui::Tokens& t = ui::CurrentTokens();
    const std::string label = std::string(np::FilterName(filter_)) + "  " ICON_FA_CHEVRON_DOWN "##filter";
    if (ui::SecondaryButton(label.c_str())) ImGui::OpenPopup("##filter_menu");
    if (ImGui::IsItemHovered()) ui::Tooltip("Show only some of the nodes");
    if (ImGui::BeginPopup("##filter_menu")) {
        for (const auto& info : np::Filters()) {
            ImGui::PushID(info.name);
            const ImVec2 at = ImGui::GetCursorPos();
            const float line = ImGui::GetTextLineHeightWithSpacing();
            if (ImGui::Selectable("##choice", filter_ == info.filter, 0, ImVec2(220.0f, line * 2.0f)))
                filter_ = info.filter;
            ImGui::SetCursorPos(at);
            ImGui::TextUnformatted(info.name);
            ImGui::TextColored(t.text_dim, "%s", info.note);
            ImGui::PopID();
        }
        ImGui::EndPopup();
    }
    ImGui::SameLine(0.0f, 10.0f);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "%s", np::CountLine(count, search_[0] != '\0').c_str());

    const char* views[] = {ICON_FA_TABLE_CELLS, ICON_FA_LIST};
    const float width = ui::SegmentedControlWidth(views, 2);
    ui::SameLineRight(width, 8.0f);
    int view = list_view_ ? 1 : 0;
    if (ui::SegmentedControl("##view", views, 2, &view)) list_view_ = view == 1;
    if (ImGui::IsItemHovered())
        ui::Tooltip("Grid or list. Preferences > Nodes panel sets the view a session starts in.");
}

void NodeBrowserPanel::ClearSearch() {
    search_[0] = '\0';
    provider_filter_.clear();
}

void NodeBrowserPanel::ShowPluginNodes(const std::string& provider_id) {
    ClearSearch();
    filter_ = np::Filter::All;
    LoadPanelSettings();
    open_groups_.insert(kPlugins);  // for this session; a fold click saves the open groups
    provider_filter_ = provider_id;
    scroll_to_plugins_ = true;
    visible_ = true;
    focus_window_next_frame_ = true;
}

void NodeBrowserPanel::RenderStudioTools() {
    const ui::Tokens& t = ui::CurrentTokens();
    SmallTitle("Studio Tools");
    const int selected = node_editor_ ? node_editor_->GetSelectedNodeCount() : 0;
    struct Tool {
        const char* id;
        const char* label;
        ImVec4 colour;
        const char* what;
        const char* how;
        bool enabled;
        const char* off_reason;
    };
    const Tool tools[] = {
        {"note", "Note", ImGui::ColorConvertU32ToFloat4(DefaultAnnotationColour()),
         "A sticky note on the canvas, to explain the graph.", "Double-click, or drag onto the canvas.", true, nullptr},
        {"group", "Group", t.series[1], "Groups the selected nodes into one box (Ctrl+G).",
         "Select nodes on the canvas, then double-click.", selected > 0, "Select nodes on the canvas first."},
        {"frame", "Frame", t.series[3], "A box to arrange nodes in.", "Drag onto the canvas.", true, nullptr},
    };
    const float gap = 6.0f;
    const float width = (ImGui::GetContentRegionAvail().x - gap * 2.0f) / 3.0f;
    const float height = ImGui::GetFrameHeight() + 4.0f;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    for (int i = 0; i < 3; ++i) {
        const Tool& tool = tools[i];
        if (i > 0) ImGui::SameLine(0.0f, gap);
        ImGui::PushID(tool.id);
        const ImVec2 min = ImGui::GetCursorScreenPos();
        ImGui::InvisibleButton("##tool", ImVec2(width, height));
        const bool hovered = ImGui::IsItemHovered();
        if (tool.enabled && hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && node_editor_) {
            if (i == 0) node_editor_->AddAnnotation();
            if (i == 1) node_editor_->GroupSelectedNodes();
        }
        if (i == 0 && ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
            const int marker = 1;
            ImGui::SetDragDropPayload("ANNOTATION", &marker, sizeof(int));
            ImGui::Text(ICON_FA_COMMENT " Note");
            ImGui::EndDragDropSource();
        }
        if (i == 2 && ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
            const char* payload = "STUDIO_FRAME";
            ImGui::SetDragDropPayload("STUDIO_FRAME", payload, strlen(payload) + 1);
            ImGui::Text(ICON_FA_SQUARE " Frame");
            ImGui::EndDragDropSource();
        }
        if (hovered && !ImGui::GetDragDropPayload()) {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 20.0f);
            ImGui::TextColored(t.text_bright, "%s", tool.label);
            ImGui::TextUnformatted(tool.what);
            ImGui::TextColored(tool.enabled ? t.text_dim : t.warning, "%s", tool.enabled ? tool.how : tool.off_reason);
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
        const ImVec2 max(min.x + width, min.y + height);
        dl->AddRectFilled(min, max, ImGui::GetColorU32(hovered && tool.enabled ? ImGuiCol_ButtonHovered : ImGuiCol_Button),
                          t.rounding_field);
        // The tool's colour, then its name; in a narrow panel the name alone,
        // centred (the colour is a 3 px strip at the bottom then).
        const float swatch = 10.0f;
        const float label_width = ImGui::CalcTextSize(tool.label).x;
        const ImVec4 colour = tool.enabled ? tool.colour : ui::Mix(tool.colour, t.bg_window, 0.5f);
        const float text_y = min.y + (height - ImGui::GetTextLineHeight()) * 0.5f;
        if (6.0f + swatch + 5.0f + label_width + 4.0f <= width) {
            const float left = min.x + (width - (swatch + 5.0f + label_width)) * 0.5f;
            dl->AddRectFilled(ImVec2(left, min.y + (height - swatch) * 0.5f),
                              ImVec2(left + swatch, min.y + (height + swatch) * 0.5f), ui::ToU32(colour), 3.0f);
            dl->AddText(ImVec2(left + swatch + 5.0f, text_y), ui::ToU32(tool.enabled ? t.text : t.text_faint), tool.label);
        } else {
            const std::string text = FitLine(tool.label, width - 4.0f);
            dl->AddText(ImVec2(min.x + (width - ImGui::CalcTextSize(text.c_str()).x) * 0.5f, text_y),
                        ui::ToU32(tool.enabled ? t.text : t.text_faint), text.c_str());
            dl->AddRectFilled(ImVec2(min.x + 6.0f, max.y - 3.0f), ImVec2(max.x - 6.0f, max.y), ui::ToU32(colour), 1.5f);
        }
        ImGui::PopID();
    }
    ImGui::Spacing();
    ImGui::Spacing();
}

void NodeBrowserPanel::RenderNodeChip(const cyxwiz::NodeMetadata* metadata) {
    auto& registry = cyxwiz::NodeMetadataRegistry::Instance();
    const bool can_add = np::CanAdd(*metadata);
    if (ui::ChipButton(metadata->name.c_str(), GetNodeColor(metadata->category)) && can_add) AddNode(metadata);
    if (can_add && ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
        ImGui::SetDragDropPayload("NODE_TYPE", &metadata->type, sizeof(cyxwiz::NodeType));
        ImGui::Text("%s %s", metadata->icon.c_str(), metadata->name.c_str());
        ImGui::EndDragDropSource();
    } else if (ImGui::IsItemHovered()) {
        RenderNodeTooltip(metadata, can_add ? "Click to add; drag onto the canvas; right-click for more"
                                            : "Right-click for more");
    }
    if (ImGui::BeginPopupContextItem("##chip_menu")) {
        ImGui::TextDisabled("%s %s", metadata->icon.c_str(), metadata->name.c_str());
        ImGui::Separator();
        if (can_add && ImGui::MenuItem(ICON_FA_PLUS " Add to Canvas", "Click")) AddNode(metadata);
        // A favourite that can no longer be added can still be taken off.
        if ((can_add || metadata->is_favorite) &&
            ImGui::MenuItem(metadata->is_favorite ? ICON_FA_STAR " Remove from Favourites"
                                                  : ICON_FA_STAR " Add to Favourites"))
            registry.ToggleFavorite(metadata->type);
        if (ImGui::MenuItem(ICON_FA_CIRCLE_INFO " Show details")) {
            pinned_extension_id_.clear();
            pinned_node_type_ = metadata->type;
            if (on_node_select_) on_node_select_(metadata->type);
        }
        ImGui::EndPopup();
    }
}

void NodeBrowserPanel::RenderFavourites() {
    const auto favourites = cyxwiz::NodeMetadataRegistry::Instance().GetFavorites();
    SmallTitle(ICON_FA_STAR " Favourites");
    if (favourites.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled("Click the star on a node, or right-click it, to keep it here.");
        ImGui::PopTextWrapPos();
    } else {
        ImGui::PushID("favourites");
        ui::FlowRow row;
        for (const auto* metadata : favourites) {
            ImGui::PushID(static_cast<int>(metadata->type));
            row.Next(ui::ChipButtonWidth(metadata->name.c_str(), true));
            RenderNodeChip(metadata);
            ImGui::PopID();
        }
        ImGui::PopID();
    }
    ImGui::Spacing();
}

void NodeBrowserPanel::RenderRecent() {
    const cyxwiz::core::NodesPanelConfig config = cyxwiz::core::EngineConfig::Instance().GetNodesPanel();
    if (!config.show_recent) return;
    const auto recent = cyxwiz::NodeMetadataRegistry::Instance().GetRecent(static_cast<size_t>(config.recent_count));
    if (recent.empty()) return;
    SmallTitle("Recent");
    if (ImGui::IsItemHovered()) ui::Tooltip("The last nodes you added, newest first. Preferences > Nodes panel sets how many.");
    ImGui::PushID("recent");
    ui::FlowRow row;
    for (const auto* metadata : recent) {
        ImGui::PushID(static_cast<int>(metadata->type));
        row.Next(ui::ChipButtonWidth(metadata->name.c_str(), true));
        RenderNodeChip(metadata);
        ImGui::PopID();
    }
    ImGui::PopID();
    ImGui::Spacing();
    ImGui::Spacing();
}

bool NodeBrowserPanel::FoldHeader(const std::string& name, size_t count, const ImVec4& swatch, const char* note) {
    const ui::Tokens& t = ui::CurrentTokens();
    const bool forced = filter_ != np::Filter::All;  // a filter shows what passes, open
    const bool open = forced || open_groups_.count(name) > 0;
    ImGui::PushID(name.c_str());
    const ImVec2 min = ImGui::GetCursorScreenPos();
    const float height = ImGui::GetFrameHeight();
    const float width = ImGui::GetContentRegionAvail().x;
    if (ImGui::InvisibleButton("##fold", ImVec2(width, height)) && !forced) {
        if (open) open_groups_.erase(name);
        else open_groups_.insert(name);
        SaveOpenGroups();
    }
    const bool hovered = ImGui::IsItemHovered();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    if (hovered && !forced) dl->AddRectFilled(min, ImVec2(min.x + width, min.y + height), ui::ToU32(t.hover), 4.0f);
    const float text_y = min.y + (height - ImGui::GetTextLineHeight()) * 0.5f;
    float x = min.x + 2.0f;
    dl->AddText(nullptr, ImGui::GetFontSize() * 0.8f, ImVec2(x, text_y + ImGui::GetFontSize() * 0.1f),
                ui::ToU32(t.text_dim), open ? ICON_FA_CHEVRON_DOWN : ICON_FA_CHEVRON_RIGHT);
    x += ImGui::GetFontSize() + 2.0f;
    const float box = 8.0f;
    dl->AddRectFilled(ImVec2(x, min.y + (height - box) * 0.5f), ImVec2(x + box, min.y + (height + box) * 0.5f),
                      ui::ToU32(swatch), 2.0f);
    x += box + 8.0f;
    dl->AddText(ImVec2(x, text_y), ui::ToU32(t.text), name.c_str());
    x += ImGui::CalcTextSize(name.c_str()).x + 8.0f;
    const std::string tail = std::to_string(count) + (note ? std::string("  ") + note : std::string());
    dl->AddText(ImVec2(x, text_y), ui::ToU32(t.text_dim), FitLine(tail, min.x + width - x).c_str());
    ImGui::PopID();
    return open;
}

void NodeBrowserPanel::RenderGroups(const std::vector<np::Group>& groups) {
    if (groups.empty() && filter_ != np::Filter::All) {
        ImGui::TextDisabled("No node is %s.", filter_ == np::Filter::Favourites ? "starred" : "listed under this filter");
        return;
    }
    for (const auto& group : groups) {
        if (FoldHeader(group.name, group.nodes.size(), ImGui::ColorConvertU32ToFloat4(GetNodeColor(group.category)))) {
            RenderNodes(group.nodes);
            ImGui::Spacing();
        }
    }
}

void NodeBrowserPanel::RenderPluginsGroup() {
    const ui::Tokens& t = ui::CurrentTokens();
    auto entries = cyxwiz::BuildExtensionPalette(cyxwiz::ExtensionNodeRegistry::Instance().All());
    std::vector<std::pair<std::string, std::string>> provider_names;
    for (const auto& entry : entries) {
        std::string provider_id, type_name;
        if (!cyxwiz::SplitExtensionTypeId(entry.type_id, provider_id, type_name)) continue;
        if (const auto* plugin = cyxwiz::plugin::PluginManager::Instance().GetLoadedPlugin(provider_id))
            provider_names.emplace_back(provider_id, plugin->manifest.name);
    }
    for (auto& name : cyxwiz::python_ui::BundleNames()) provider_names.push_back(std::move(name));
    // Narrowed to one provider (Apps > a loaded plugin).
    if (!provider_filter_.empty()) {
        entries.erase(std::remove_if(entries.begin(), entries.end(),
                                     [&](const cyxwiz::ExtensionPaletteEntry& entry) {
                                         std::string provider_id, type_name;
                                         return !cyxwiz::SplitExtensionTypeId(entry.type_id, provider_id, type_name) ||
                                                provider_id != provider_filter_;
                                     }),
                      entries.end());
    }
    if (scroll_to_plugins_) {
        ImGui::SetScrollHereY(0.0f);
        scroll_to_plugins_ = false;
    }
    if (!FoldHeader(kPlugins, entries.size(), t.text_faint)) return;
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim, "%s", cyxwiz::ExtensionPaletteSummary(entries, provider_names).c_str());
    ImGui::PopTextWrapPos();
    if (!provider_filter_.empty() && ui::LinkButton("Show all plugins##provider_filter")) provider_filter_.clear();
    if (entries.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled("No plugin nodes are loaded. Load a plugin that provides nodes and they appear here.");
        ImGui::PopTextWrapPos();
        if (ui::SecondaryButton("Open Plugin Manager", node_editor_ != nullptr, "The node editor is not available"))
            node_editor_->OpenPluginManager();
    }
    // One list per plugin group ("RL / Simulation", ...).
    for (size_t i = 0; i < entries.size();) {
        const std::string group = entries[i].group;
        std::vector<cyxwiz::ExtensionPaletteEntry> group_entries;
        for (; i < entries.size() && entries[i].group == group; ++i) group_entries.push_back(entries[i]);
        ImGui::TextColored(t.text_dim, "%s", group.c_str());
        RenderExtensionNodes(group_entries);
    }
    ImGui::Spacing();
}

void NodeBrowserPanel::RenderAdvancedGroup() {
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
    if (count == 0 || !FoldHeader(cyxwiz::palette::kAdvancedOperations, count, t.text_faint, "table steps")) return;
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
        RenderNodes(nodes);
    }
    ImGui::Spacing();
}

void NodeBrowserPanel::RenderCantRunGroup() {
    const ui::Tokens& t = ui::CurrentTokens();
    std::vector<const cyxwiz::NodeMetadata*> nodes;
    for (const auto* m : cyxwiz::NodeMetadataRegistry::Instance().GetAllMetadata())
        if (m && cyxwiz::palette::PlaceNode(*m).place == cyxwiz::palette::Place::CantRun) nodes.push_back(m);
    // By category, then name: what is missing where.
    std::sort(nodes.begin(), nodes.end(), [](const cyxwiz::NodeMetadata* a, const cyxwiz::NodeMetadata* b) {
        const std::string ca = cyxwiz::GetCategoryDisplayName(a->category), cb = cyxwiz::GetCategoryDisplayName(b->category);
        return ca != cb ? ca < cb : a->name < b->name;
    });
    if (nodes.empty() || !FoldHeader(cyxwiz::palette::kCantRunYet, nodes.size(), t.text_faint, "with why")) return;
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim, "What the Engine cannot run yet; each leaves this list when it runs. A saved graph that "
                                   "has one still opens and says why it cannot run. Hover a node for the reason.");
    ImGui::PopTextWrapPos();
    RenderNodes(nodes);
    ImGui::Spacing();
}

void NodeBrowserPanel::RenderExamplesGroup() {
    const ui::Tokens& t = ui::CurrentTokens();
    auto& library = patterns::PatternLibrary::Instance();
    if (!library.IsInitialized()) library.Initialize();
    const auto categories = library.GetAvailableCategories();
    if (examples_count_ < 0) {  // GetByCategory copies the patterns: count once, not every frame
        examples_count_ = 0;
        for (auto category : categories) examples_count_ += static_cast<int>(library.GetByCategory(category).size());
    }
    if (examples_count_ == 0 ||
        !FoldHeader(kExamples, static_cast<size_t>(examples_count_), t.text_faint, "small graphs to start from"))
        return;
    for (auto category : categories) {
        const std::string label =
            std::string(patterns::GetPatternCategoryIcon(category)) + " " + patterns::PatternCategoryToString(category);
        if (!ImGui::TreeNodeEx(label.c_str(), ImGuiTreeNodeFlags_SpanAvailWidth)) continue;
        for (const auto& pattern : library.GetByCategory(category)) {
            ImGui::PushID(pattern.id.c_str());
            ImGui::Selectable(pattern.name.c_str(), false, ImGuiSelectableFlags_AllowDoubleClick);
            if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && on_example_insert_)
                on_example_insert_(pattern.id);
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::PushTextWrapPos(ImGui::GetFontSize() * 22.0f);
                ImGui::TextColored(t.text_bright, "%s", pattern.name.c_str());
                if (!pattern.description.empty()) ImGui::TextUnformatted(pattern.description.c_str());
                ImGui::TextColored(t.text_dim, "Double-click to add it to the canvas");
                ImGui::PopTextWrapPos();
                ImGui::EndTooltip();
            }
            ImGui::PopID();
        }
        ImGui::TreePop();
    }
    ImGui::Spacing();
}

void NodeBrowserPanel::RenderSearchResults(const np::SearchResult& found,
                                           const std::vector<cyxwiz::ExtensionPaletteEntry>& extension_results) {
    if (found.named.empty() && found.related.empty() && extension_results.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled("No node is named or about '%s'.", search_);
        ImGui::PopTextWrapPos();
        return;
    }
    RenderNodes(found.named);
    RenderExtensionNodes(extension_results);
    if (!found.related.empty()) {
        if (!found.named.empty() || !extension_results.empty()) ImGui::Spacing();
        SmallTitle("Related");
        if (ImGui::IsItemHovered()) ui::Tooltip("Nodes that mention it in a keyword or their description");
        ImGui::PushID("related");
        RenderNodes(found.related);
        ImGui::PopID();
    }
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

void NodeBrowserPanel::RenderNodes(const std::vector<const cyxwiz::NodeMetadata*>& nodes) {
    if (nodes.empty()) return;
    const float available = ImGui::GetContentRegionAvail().x;
    if (list_view_) {
        for (const auto* metadata : nodes) RenderNodeItem(metadata, available);
        return;
    }
    const int columns = ColumnsFor(available);
    const float width = (available - (columns - 1) * GRID_SPACING) / columns;
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (i % columns != 0) ImGui::SameLine(0, GRID_SPACING);
        RenderNodeItem(nodes[i], width);
    }
}

void NodeBrowserPanel::RenderExtensionNodes(const std::vector<cyxwiz::ExtensionPaletteEntry>& entries) {
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
    const float available = ImGui::GetContentRegionAvail().x;
    const int columns = list_view_ ? 1 : ColumnsFor(available);
    const float width = list_view_ ? available : (available - (columns - 1) * GRID_SPACING) / columns;
    for (size_t i = 0; i < metadata.size(); ++i) {
        if (i % columns != 0) ImGui::SameLine(0, GRID_SPACING);
        RenderNodeItem(&metadata[i], width, shown[i]);
    }
}

void NodeBrowserPanel::RenderNodeItem(const cyxwiz::NodeMetadata* metadata, float width,
                                      const cyxwiz::ExtensionPaletteEntry* extension) {
    if (!metadata) return;
    auto& registry = cyxwiz::NodeMetadataRegistry::Instance();

    // Extension nodes share one NodeType: they are keyed by type id.
    if (extension) ImGui::PushID(extension->type_id.c_str());
    else ImGui::PushID(static_cast<int>(metadata->type));
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

    const float height = list_view_ ? ROW_ICON_SIZE + 8.0f : NODE_CARD_HEIGHT;
    const ImVec2 min = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##item", ImVec2(width, height));
    // Keep the layout, skip the work for items scrolled out of view.
    if (!ImGui::IsItemVisible()) {
        ImGui::PopID();
        return;
    }
    const bool hovered = ImGui::IsItemHovered();
    const bool can_add = np::CanAdd(*metadata);

    // The favourite star (favourites are kept per NodeType; extension nodes
    // share one): top right of a card, the end of a row.
    // A favourite keeps its star when it can no longer be added, so it can
    // be taken off.
    const bool has_star = !extension && (can_add || metadata->is_favorite);
    const float star_size = ImGui::GetFontSize();
    const ImVec2 star_min = list_view_ ? ImVec2(min.x + width - star_size - 6.0f, min.y + (height - star_size) * 0.5f)
                                       : ImVec2(min.x + width - star_size - 4.0f, min.y + 3.0f);
    const bool star_hovered = has_star && hovered &&
                              ImGui::IsMouseHoveringRect(star_min, ImVec2(star_min.x + star_size, star_min.y + star_size));
    const char* const favourite_label = metadata->is_favorite ? "Remove from Favourites" : "Add to Favourites";

    // A click shows the node in the Info panel; on the star it toggles the favourite.
    if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
        if (star_hovered) registry.ToggleFavorite(metadata->type);
        else select();
    }
    if (hovered && !star_hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && can_add)
        AddNode(metadata, extension);
    if (can_add && ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
        if (extension) ImGui::SetDragDropPayload("EXTENSION_NODE", extension->type_id.data(), extension->type_id.size());
        else ImGui::SetDragDropPayload("NODE_TYPE", &metadata->type, sizeof(cyxwiz::NodeType));
        ImGui::Text("%s %s", metadata->icon.c_str(), metadata->name.c_str());
        ImGui::EndDragDropSource();
    }

    if (ImGui::BeginPopupContextItem("##node_menu")) {
        ImGui::TextDisabled("%s %s", metadata->icon.c_str(), metadata->name.c_str());
        ImGui::Separator();
        if (can_add) {
            if (ImGui::MenuItem(ICON_FA_PLUS " Add to Canvas", "Double-click")) AddNode(metadata, extension);
        } else {
            const np::Tooltip tip = np::BuildTooltip(*metadata);
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 24.0f);
            ImGui::TextColored(ToneColour(tip.tone), "%s", tip.runs.c_str());
            ImGui::PopTextWrapPos();
        }
        if (has_star && ImGui::MenuItem((std::string(ICON_FA_STAR " ") + favourite_label).c_str()))
            registry.ToggleFavorite(metadata->type);
        ImGui::Separator();
        if (ImGui::MenuItem(ICON_FA_CIRCLE_INFO " Show details")) select();
        if (ImGui::MenuItem(ICON_FA_COPY " Copy Node Name")) ImGui::SetClipboardText(metadata->name.c_str());
        ImGui::EndPopup();
    }

    if (star_hovered) ui::Tooltip(favourite_label);
    else if (hovered && !ImGui::IsMouseDragging(ImGuiMouseButton_Left)) RenderNodeTooltip(metadata);

    // The node shown in the Info panel: the selection fill, under the item.
    // Hover previews it there (it returns to the kept node when the hover ends).
    const ui::Tokens& t = ui::CurrentTokens();
    const bool pinned = extension ? extension->type_id == pinned_extension_id_
                                  : (pinned_extension_id_.empty() && metadata->type == pinned_node_type_);
    const ImVec2 max(min.x + width, min.y + height);
    if (pinned) ImGui::GetWindowDrawList()->AddRectFilled(min, max, ui::ToU32(t.selection), 6.0f);
    else if (hovered) ImGui::GetWindowDrawList()->AddRectFilled(min, max, ui::ToU32(t.hover), 6.0f);
    if (list_view_) DrawRow(metadata, min, width, can_add, has_star, star_hovered, extension);
    else DrawCard(metadata, min, width, can_add, has_star, star_hovered, extension);
    if (hovered) {
        if (extension) {
            if (on_extension_hover_) on_extension_hover_(extension->type_id);
        } else if (on_node_hover_) {
            on_node_hover_(metadata->type);
        }
    }
    ImGui::PopID();
}

void NodeBrowserPanel::DrawCard(const cyxwiz::NodeMetadata* metadata, ImVec2 min, float width, bool can_add,
                                bool has_star, bool star_hovered, const cyxwiz::ExtensionPaletteEntry* extension) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 icon_min(min.x + (width - NODE_ICON_SIZE) * 0.5f, min.y + 4.0f);
    DrawIcon(metadata, icon_min, NODE_ICON_SIZE, extension);

    // Inputs and outputs: a small triangle each side of the icon.
    const float tri = 5.0f;
    const float mid = icon_min.y + NODE_ICON_SIZE * 0.5f;
    if (!metadata->inputs.empty())
        dl->AddTriangleFilled(ImVec2(icon_min.x - 1, mid), ImVec2(icon_min.x + tri - 1, mid - tri),
                              ImVec2(icon_min.x + tri - 1, mid + tri), ui::ToU32(t.text_faint));
    if (!metadata->outputs.empty()) {
        const float right = icon_min.x + NODE_ICON_SIZE;
        dl->AddTriangleFilled(ImVec2(right - tri + 1, mid - tri), ImVec2(right - tri + 1, mid + tri), ImVec2(right + 1, mid),
                              ui::ToU32(t.text_faint));
    }
    // Only the exceptions are marked (the word is in the tooltip).
    if (const np::Mark mark = np::CardMark(*metadata); mark != np::Mark::None)
        dl->AddCircleFilled(ImVec2(min.x + 7.0f, min.y + 7.0f), 4.0f, ui::ToU32(MarkColour(mark)));
    if (has_star) {
        const ImVec4 colour = metadata->is_favorite ? t.accent_text : star_hovered ? t.text_dim : ui::WithAlpha(t.text_faint, 0.55f);
        dl->AddText(nullptr, ImGui::GetFontSize(), ImVec2(min.x + width - ImGui::GetFontSize() - 4.0f, min.y + 3.0f),
                    ui::ToU32(colour), ICON_FA_STAR);
    }
    DrawCardName(dl, metadata->name, min.x, width, icon_min.y + NODE_ICON_SIZE + 4.0f,
                 ui::ToU32(can_add ? t.text : t.text_faint));

    // A template's badge ("Coming Soon"), whole, along the card's top.
    if (metadata->IsTemplate()) {
        const std::string badge = FitLine(metadata->badge.empty() ? "Coming soon" : metadata->badge, width - 8.0f);
        const ImVec2 size = ImGui::CalcTextSize(badge.c_str());
        const ImVec2 badge_min(min.x + (width - size.x) * 0.5f - 3.0f, min.y + 2.0f);
        dl->AddRectFilled(badge_min, ImVec2(badge_min.x + size.x + 6.0f, badge_min.y + size.y + 2.0f),
                          ui::ToU32(ui::WithAlpha(t.pending, 0.9f)), 3.0f);
        dl->AddText(ImVec2(badge_min.x + 3.0f, badge_min.y + 1.0f), ui::ToU32(ui::ReadableOn(t, t.pending)), badge.c_str());
    }
}

void NodeBrowserPanel::DrawRow(const cyxwiz::NodeMetadata* metadata, ImVec2 min, float width, bool can_add,
                               bool has_star, bool star_hovered, const cyxwiz::ExtensionPaletteEntry* extension) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float height = ROW_ICON_SIZE + 8.0f;
    const ImVec2 icon_min(min.x + 6.0f, min.y + 4.0f);
    DrawIcon(metadata, icon_min, ROW_ICON_SIZE, extension);
    float right = min.x + width - 6.0f;
    if (has_star) {
        const ImVec4 colour = metadata->is_favorite ? t.accent_text : star_hovered ? t.text_dim : ui::WithAlpha(t.text_faint, 0.55f);
        right -= ImGui::GetFontSize();
        dl->AddText(nullptr, ImGui::GetFontSize(), ImVec2(right, min.y + (height - ImGui::GetFontSize()) * 0.5f),
                    ui::ToU32(colour), ICON_FA_STAR);
        right -= 6.0f;
    }
    if (const np::Mark mark = np::CardMark(*metadata); mark != np::Mark::None) {
        right -= 8.0f;
        dl->AddCircleFilled(ImVec2(right + 4.0f, min.y + height * 0.5f), 3.5f, ui::ToU32(MarkColour(mark)));
        right -= 6.0f;
    }
    if (metadata->IsTemplate()) {
        const std::string badge = metadata->badge.empty() ? "Coming soon" : metadata->badge;
        const float w = ImGui::CalcTextSize(badge.c_str()).x;
        right -= w;
        dl->AddText(ImVec2(right, min.y + (height - ImGui::GetTextLineHeight()) * 0.5f), ui::ToU32(t.pending), badge.c_str());
        right -= 8.0f;
    }
    const float text_x = icon_min.x + ROW_ICON_SIZE + 8.0f;
    const std::string name = FitLine(metadata->name, right - text_x);
    dl->AddText(ImVec2(text_x, min.y + (height - ImGui::GetTextLineHeight()) * 0.5f),
                ui::ToU32(can_add ? t.text : t.text_faint), name.c_str());
}

void NodeBrowserPanel::DrawIcon(const cyxwiz::NodeMetadata* metadata, ImVec2 min, float size,
                                const cyxwiz::ExtensionPaletteEntry* extension) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImU32 fill = GetNodeColor(metadata->category);
    if (extension && (extension->color >> 24) != 0) fill = extension->color;
    if (metadata->IsTemplate()) fill = ui::ToU32(t.bg_raised);  // not runnable yet: no family colour
    const ImVec2 max(min.x + size, min.y + size);
    dl->AddRectFilled(min, max, fill, size > 24.0f ? 6.0f : 4.0f);
    if (metadata->icon.empty()) return;
    ImFont* font = ImGui::GetFont();
    const float glyph = size * 0.55f;
    const ImVec2 text = font->CalcTextSizeA(glyph, FLT_MAX, 0.0f, metadata->icon.c_str());
    dl->AddText(font, glyph, ImVec2(min.x + (size - text.x) * 0.5f, min.y + (size - text.y) * 0.5f),
                ui::ToU32(ui::ReadableOn(t, ImGui::ColorConvertU32ToFloat4(fill))), metadata->icon.c_str());
}

void NodeBrowserPanel::RenderNodeTooltip(const cyxwiz::NodeMetadata* metadata, const char* footer) {
    const ui::Tokens& t = ui::CurrentTokens();
    const np::Tooltip tip = np::BuildTooltip(*metadata);
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 24.0f);
    ImGui::TextColored(t.text_bright, "%s %s", metadata->icon.c_str(), tip.title.c_str());
    ImGui::SameLine(0.0f, 8.0f);
    ImGui::TextColored(t.text_dim, "%s", tip.family.c_str());
    if (!tip.description.empty()) ImGui::TextUnformatted(tip.description.c_str());
    ImGui::Spacing();
    const auto row = [&](const char* label, const std::string& value, const ImVec4& colour) {
        if (value.empty()) return;
        ImGui::TextColored(t.text_dim, "%s", label);
        ImGui::SameLine(ImGui::GetFontSize() * 3.5f);
        ImGui::TextColored(colour, "%s", value.c_str());
    };
    row("In", PortList(tip.inputs), t.text);
    row("Out", PortList(tip.outputs), t.text);
    row("Runs", tip.runs, ToneColour(tip.tone));
    ImGui::Spacing();
    ImGui::TextColored(t.text_dim, "%s", footer ? footer : tip.footer.c_str());
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
}

void NodeBrowserPanel::AddNode(const cyxwiz::NodeMetadata* metadata, const cyxwiz::ExtensionPaletteEntry* extension) {
    if (!node_editor_ || !metadata || !np::CanAdd(*metadata)) return;
    // Added at a free spot near the middle of the canvas view.
    if (extension) node_editor_->AddNodeFromMenu(cyxwiz::NodeType::PluginCustom, extension->type_id);
    else node_editor_->AddNodeFromMenu(metadata->type, metadata->name);
}

ImU32 NodeBrowserPanel::GetNodeColor(cyxwiz::NodeCategory category) const {
    return ui::ToU32(NodeCategoryColour(category));
}

} // namespace gui
