#include "dataset_picker.h"

#include "../core/dataset_catalog.h"
#include "../core/dataset_picker_presentation.h"
#include "ui_tokens.h"

#include <imgui.h>

#include <algorithm>
#include <vector>

namespace cyxwiz::ui {

bool DatasetPicker(const char* id, std::string& selected, const DatasetPickerOptions& options) {
    const Tokens& t = CurrentTokens();
    std::vector<DatasetEntry> entries = DatasetCatalog::Instance().List();
    if (options.tabular_only) {
        entries.erase(std::remove_if(entries.begin(), entries.end(),
                                     [](const DatasetEntry& e) { return !IsTabularDataset(e); }),
                      entries.end());
    }
    const std::string preview = DatasetPickerPreview(entries, selected);
    const float width = options.width > 0.0f ? options.width : std::min(520.0f, ImGui::GetContentRegionAvail().x);
    ImGui::SetNextItemWidth(width);
    bool changed = false;
    ImGui::PushID(id);
    if (ImGui::BeginCombo("##dataset", preview.c_str(), ImGuiComboFlags_HeightLarge)) {
        if (entries.empty()) ImGui::TextColored(t.text_dim, "No datasets yet: load a Data Input in CyxWiz Studio.");
        if (!entries.empty() && ImGui::BeginTable("##datasets", 4, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("name", ImGuiTableColumnFlags_WidthStretch, 1.6f);
            ImGui::TableSetupColumn("origin", ImGuiTableColumnFlags_WidthStretch, 0.9f);
            ImGui::TableSetupColumn("kind", ImGuiTableColumnFlags_WidthStretch, 1.3f);
            ImGui::TableSetupColumn("size", ImGuiTableColumnFlags_WidthStretch, 1.4f);
            for (const auto& e : entries) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::PushID(e.name.c_str());
                const bool is_selected = e.name == selected;
                if (ImGui::Selectable(e.Shown().c_str(), is_selected, ImGuiSelectableFlags_SpanAllColumns) && !is_selected) {
                    selected = e.name;
                    changed = true;
                }
                if (is_selected) ImGui::SetItemDefaultFocus();
                if (!e.label.empty() && ImGui::IsItemHovered()) ImGui::SetTooltip("Dataset name: %s (a query can use either)", e.name.c_str());
                ImGui::PopID();
                ImGui::TableSetColumnIndex(1);
                ImGui::TextColored(t.text_dim, "%s", DatasetOriginText(e));
                ImGui::TableSetColumnIndex(2);
                ImGui::TextColored(t.text_dim, "%s", StorageText(e.storage));
                ImGui::TableSetColumnIndex(3);
                ImGui::TextColored(t.text_dim, "%s", DatasetSizeText(e).c_str());
            }
            ImGui::EndTable();
        }
        ImGui::EndCombo();
    }
    ImGui::PopID();
    return changed;
}

}  // namespace cyxwiz::ui
