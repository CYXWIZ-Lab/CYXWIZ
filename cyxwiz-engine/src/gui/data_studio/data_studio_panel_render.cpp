#include "data_studio_panel.h"

#include "../../core/dataset_catalog.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"

#include <imgui.h>

namespace cyxwiz {

namespace {

std::string Thousands(size_t n) {
    std::string s = std::to_string(n);
    for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3) s.insert(static_cast<size_t>(i), ",");
    return s;
}

// "8,582 rows · 15 cols", "2,000 files · 2 classes".
std::string SizeText(const DatasetEntry& e) {
    std::string unit = "rows";
    if (e.modality == DatasetModality::Image) unit = "files";
    else if (e.modality == DatasetModality::Audio) unit = "clips";
    else if (e.modality == DatasetModality::Text && !e.Has(kBackingArrow)) unit = "texts";
    std::string s = Thousands(e.rows) + " " + unit;
    if (e.columns) s += " \xC2\xB7 " + Thousands(e.columns) + " cols";
    if (e.classes) s += " \xC2\xB7 " + Thousands(e.classes) + " classes";
    return s;
}

const char* OriginText(const DatasetEntry& e) {
    if (e.materialized) return "node result";
    if (e.Has(kBackingLegacy) && e.backings == kBackingLegacy) return "older dataset";
    return "Data Input";
}

}  // namespace

void DataStudioPanel::Render() {
    if (query_editor_) query_editor_->RenderWindows();
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(1200, 800), ImGuiCond_FirstUseEver);
    // Opened from elsewhere (a dashboard): bring the panel forward, also when
    // it is a hidden dock tab (its Begin returns false until focused).
    if (show_profile_) ImGui::SetNextWindowFocus();
    if (ImGui::Begin("Data Studio", &visible_)) {
        RenderToolbar();
        ImGui::Spacing();
        RenderTabBar();
        ImGui::Spacing();
        RenderStatusBar();
    }
    ImGui::End();
}

void DataStudioPanel::RenderToolbar() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Data Studio");
    ImGui::SameLine();
    RenderDatasetSelector();

    ImGui::SameLine();
    if (ui::SecondaryButton(ICON_FA_ROTATE " Refresh", !active_dataset_.empty(), "Pick a dataset first")) SetActiveDataset(active_dataset_);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Read the picked dataset again (after it was re-loaded)");

    // Cross-navigation: one click opens the Engine's Node Editor panel.
    ImGui::SameLine();
    if (open_node_editor_callback_) {
        if (ui::SecondaryButton("Open Node Editor")) open_node_editor_callback_();
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Open the CyxWiz Studio (Node Editor) panel. "
                              "Switch to 'Data Pipeline' execution mode "
                              "there to build visual transformation graphs.");
        }
    } else {
        ImGui::TextColored(t.text_dim, "For visual pipelines, use CyxWiz Studio");
    }
}

void DataStudioPanel::RenderDatasetSelector() {
    const ui::Tokens& t = ui::CurrentTokens();
    // Every dataset the catalog knows: Data Input tables, node results, files.
    const auto entries = DatasetCatalog::Instance().List();
    ImGui::TextColored(t.text_dim, "Dataset");
    ImGui::SameLine();
    std::string preview = "Select a dataset...";
    for (const auto& e : entries)
        if (e.name == active_dataset_) preview = e.Shown() + "  \xC2\xB7  " + OriginText(e) + "  \xC2\xB7  " + SizeText(e);
    if (!active_dataset_.empty() && preview == "Select a dataset...") preview = active_dataset_ + "  (no longer loaded)";
    ImGui::SetNextItemWidth(std::min(520.0f, ImGui::GetContentRegionAvail().x * 0.45f));
    if (ImGui::BeginCombo("##dataset_selector", preview.c_str(), ImGuiComboFlags_HeightLarge)) {
        if (entries.empty()) ImGui::TextColored(t.text_dim, "No datasets yet: load a Data Input in CyxWiz Studio.");
        if (entries.size() > 0 && ImGui::BeginTable("##datasets", 4, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("name", ImGuiTableColumnFlags_WidthStretch, 1.6f);
            ImGui::TableSetupColumn("origin", ImGuiTableColumnFlags_WidthStretch, 0.9f);
            ImGui::TableSetupColumn("kind", ImGuiTableColumnFlags_WidthStretch, 1.3f);
            ImGui::TableSetupColumn("size", ImGuiTableColumnFlags_WidthStretch, 1.4f);
            for (const auto& e : entries) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::PushID(e.name.c_str());
                const bool selected = e.name == active_dataset_;
                if (ImGui::Selectable(e.Shown().c_str(), selected, ImGuiSelectableFlags_SpanAllColumns)) SetActiveDataset(e.name);
                if (selected) ImGui::SetItemDefaultFocus();
                if (!e.label.empty() && ImGui::IsItemHovered()) ImGui::SetTooltip("Dataset name: %s (a query can use either)", e.name.c_str());
                ImGui::PopID();
                ImGui::TableSetColumnIndex(1);
                ImGui::TextColored(t.text_dim, "%s", OriginText(e));
                ImGui::TableSetColumnIndex(2);
                ImGui::TextColored(t.text_dim, "%s", StorageText(e.storage));
                ImGui::TableSetColumnIndex(3);
                ImGui::TextColored(t.text_dim, "%s", SizeText(e).c_str());
            }
            ImGui::EndTable();
        }
        ImGui::EndCombo();
    }
}

void DataStudioPanel::RenderTabBar() {
    // Unified Canvas Phase 5: Removed Pipeline tab (moved to Node Editor)
    if (ImGui::BeginTabBar("DataStudioTabs")) {
        if (ImGui::BeginTabItem("Query")) {
            if (query_editor_) query_editor_->Render();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Profile", nullptr, show_profile_ ? ImGuiTabItemFlags_SetSelected : 0)) {
            show_profile_ = false;
            if (profile_) profile_->Render();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Visualize")) {
            if (visualizer_) visualizer_->Render();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void DataStudioPanel::RenderStatusBar() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!active_dataset_.empty()) ImGui::TextColored(t.text_dim, "Dataset: %s", active_dataset_.c_str());
    else ImGui::TextColored(t.text_dim, "No dataset selected");
    ImGui::SameLine();
    ImGui::TextColored(t.text_faint, "\xC2\xB7");
    ImGui::SameLine();
    if (query_editor_ && query_editor_->IsRunning()) ImGui::TextColored(t.info, "Query running");
    else if (profile_ && profile_->IsRunning()) ImGui::TextColored(t.info, "Profiling");
    else ImGui::TextColored(t.text_dim, "Ready");
}

}  // namespace cyxwiz
