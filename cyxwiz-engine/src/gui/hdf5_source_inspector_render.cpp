#include "hdf5_source_inspector.h"
#include "icons.h"
#include "ui_buttons.h"

#include <algorithm>
#include <limits>

namespace gui {
namespace {

bool IconAction(const char* icon, const char* tooltip, bool enabled = true) {
    const bool pressed = cyxwiz::ui::GhostButton(icon, enabled, tooltip);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("%s", tooltip);
    return pressed;
}

std::string ShapeText(const std::vector<uint64_t>& shape) {
    std::string text = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) text += ", ";
        text += std::to_string(shape[i]);
    }
    return text + "]";
}

bool SelectableDataset(const cyxwiz::Hdf5BrowseEntry& entry, bool labels) {
    const auto& shape = entry.dataset.shape;
    return entry.kind == cyxwiz::Hdf5ObjectKind::Dataset &&
        (entry.status == cyxwiz::Hdf5TableStatus::Ok || entry.status == cyxwiz::Hdf5TableStatus::ResourceLimit) &&
        !shape.empty() && shape.size() <= (labels ? 1U : 2U) && shape[0] > 0 &&
        shape[0] <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) &&
        (shape.size() == 1 || (shape[1] > 0 && shape[1] <= 4096));
}
} // namespace

void Hdf5SourceInspector::RenderError() const {
    if (!error_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.35f, 1.0f), "%s", error_.c_str());
        ImGui::PopTextWrapPos();
    }
}

void Hdf5SourceInspector::RenderSettings() {
    Poll();
    ImGui::PushID("hdf5_inspector");
    ImGui::TextUnformatted("HDF5 source inspection");
    if (!cyxwiz::Hdf5TableSupportAvailable()) {
        ImGui::TextWrapped("HDF5 support is not compiled into this build");
        ImGui::PopID();
        return;
    }
    ImGui::TextDisabled("Preview only; loading unavailable");
    ImGui::TextUnformatted("Group");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputText("##group", group_path_, sizeof(group_path_))) {
        Cancel();
        hierarchy_ = {};
        error_.clear();
    }
    if (IconAction(ICON_FA_FOLDER_OPEN, "Open group", !Busy() && !source_path_.empty()))
        Browse(group_path_);
    ImGui::SameLine();
    if (IconAction(ICON_FA_ARROW_UP, "Parent group", !Busy() && std::string(group_path_) != "/")) {
        const std::string group(group_path_);
        const auto slash = group.find_last_of('/');
        Browse(slash == 0 || slash == std::string::npos ? "/" : group.substr(0, slash));
    }
    ImGui::SameLine();
    if (IconAction(ICON_FA_ARROWS_ROTATE, "Refresh source", !Busy() && !source_path_.empty()))
        Browse(group_path_, 0, true);
    ImGui::SameLine();
    if (IconAction(ICON_FA_STOP, "Cancel inspection", Busy())) {
        Cancel();
        error_ = "HDF5 inspection cancelled";
    }
    ImGui::SameLine();
    if (Busy()) ImGui::TextDisabled("Inspecting...");
    else if (hierarchy_.status == cyxwiz::Hdf5TableStatus::Ok)
        ImGui::TextDisabled("%llu entries", static_cast<unsigned long long>(hierarchy_.total_entries));

    const char* roles[] = {"Data", "Labels"};
    cyxwiz::ui::SegmentedControl("selection_role", roles, 2, &selection_role_);
    std::string open_group;
    if (ImGui::BeginTable("hierarchy", 3,
            ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
                ImGuiTableFlags_Resizable,
            ImVec2(0, 180))) {
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch, 2.0f);
        ImGui::TableSetupColumn("Shape", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();
        for (size_t index = 0; index < hierarchy_.entries.size(); ++index) {
            const auto& entry = hierarchy_.entries[index];
            ImGui::PushID(static_cast<int>(index));
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            const bool group = entry.kind == cyxwiz::Hdf5ObjectKind::Group;
            const bool selectable = group || SelectableDataset(entry, selection_role_ == 1);
            const bool selected = entry.path == (selection_role_ == 0 ? data_path_ : label_path_);
            const auto position = ImGui::GetCursorScreenPos();
            ImGui::BeginDisabled(!selectable || Busy());
            const bool clicked = ImGui::Selectable("##entry", selected);
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::BeginTooltip();
                ImGui::PushTextWrapPos(ImGui::GetFontSize() * 32.0f);
                ImGui::TextUnformatted(entry.path.c_str());
                if (!entry.reason.empty()) ImGui::TextUnformatted(entry.reason.c_str());
                ImGui::PopTextWrapPos();
                ImGui::EndTooltip();
            }
            const auto end_position = ImGui::GetCursorScreenPos();
            ImGui::SetCursorScreenPos(position);
            ImGui::TextUnformatted(entry.name.c_str());
            ImGui::SetCursorScreenPos(end_position);
            if (clicked) {
                if (group) open_group = entry.path;
                else {
                    auto selection = Selection();
                    (selection_role_ == 0 ? selection.data_path : selection.label_path) = entry.path;
                    SetSelection(selection);
                }
            }
            ImGui::TableSetColumnIndex(1);
            if (!entry.dataset.shape.empty()) ImGui::TextUnformatted(ShapeText(entry.dataset.shape).c_str());
            ImGui::TableSetColumnIndex(2);
            ImGui::TextUnformatted(group ? "Group" : entry.dataset.source_type.empty()
                ? "Unsupported" : entry.dataset.source_type.c_str());
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    if (!open_group.empty()) Browse(open_group);
    if (IconAction(ICON_FA_CHEVRON_LEFT, "Previous group page", !Busy() && hierarchy_.offset > 0))
        Browse(group_path_, hierarchy_.offset > 64 ? hierarchy_.offset - 64 : 0);
    ImGui::SameLine();
    if (IconAction(ICON_FA_CHEVRON_RIGHT, "Next group page", !Busy() && hierarchy_.has_next))
        Browse(group_path_, hierarchy_.next_offset);
    ImGui::TextUnformatted("Data path");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputText("##data_path", data_path_, sizeof(data_path_))) InvalidateSelection();
    ImGui::TextUnformatted("Label path (optional)");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputText("##label_path", label_path_, sizeof(label_path_))) InvalidateSelection();
    if (cyxwiz::ui::PrimaryButton(ICON_FA_EYE " Preview", !Busy() && data_path_[0] != '\0')) Preview();
    RenderError();
    ImGui::PopID();
}

void Hdf5SourceInspector::RenderPreview() {
    Poll();
    ImGui::PushID("hdf5_source_preview");
    ImGui::TextUnformatted("HDF5 source preview");
    if (!cyxwiz::Hdf5TableSupportAvailable()) {
        ImGui::TextWrapped("HDF5 support is not compiled into this build");
        ImGui::PopID();
        return;
    }
    ImGui::TextWrapped("%s", data_path_);
    if (label_path_[0]) ImGui::TextWrapped("Labels: %s", label_path_);
    if (cyxwiz::ui::PrimaryButton(ICON_FA_EYE " Preview", !Busy() && data_path_[0] != '\0'))
        Preview(row_offset_, column_offset_);
    ImGui::SameLine();
    if (IconAction(ICON_FA_STOP, "Cancel preview", Busy())) {
        Cancel();
        error_ = "HDF5 inspection cancelled";
    }
    ImGui::SetNextItemWidth(90);
    const auto rows_label = std::to_string(rows_per_page_);
    if (ImGui::BeginCombo("Rows/page", rows_label.c_str())) {
        for (int count : {20, 50, 100, 200}) {
            if (ImGui::Selectable(std::to_string(count).c_str(), count == rows_per_page_)) {
                rows_per_page_ = count;
                InvalidateSelection();
            }
        }
        ImGui::EndCombo();
    }
    ImGui::PushID("rows");
    if (IconAction(ICON_FA_CHEVRON_LEFT, "Previous rows", !Busy() && row_offset_ > 0))
        Preview(row_offset_ > static_cast<uint64_t>(rows_per_page_) ? row_offset_ - rows_per_page_ : 0, column_offset_);
    ImGui::SameLine();
    if (IconAction(ICON_FA_CHEVRON_RIGHT, "Next rows", !Busy() && page_.ok && page_.has_next))
        Preview(static_cast<uint64_t>(page_.next_offset), column_offset_);
    ImGui::SameLine();
    ImGui::TextDisabled("Rows %llu-%llu of %lld",
        static_cast<unsigned long long>(page_.rows_returned ? row_offset_ + 1 : 0),
        static_cast<unsigned long long>(row_offset_ + page_.rows_returned),
        static_cast<long long>(page_.total_rows));
    ImGui::PopID();
    const auto columns = data_.shape.empty() ? 0 : data_.shape.size() == 1 ? 1 : data_.shape[1];
    ImGui::PushID("columns");
    if (IconAction(ICON_FA_CHEVRON_LEFT, "Previous columns", !Busy() && column_offset_ > 0))
        Preview(row_offset_, column_offset_ > 32 ? column_offset_ - 32 : 0);
    ImGui::SameLine();
    if (IconAction(ICON_FA_CHEVRON_RIGHT, "Next columns", !Busy() && page_.ok && column_offset_ + 32 < columns))
        Preview(row_offset_, column_offset_ + 32);
    ImGui::SameLine();
    ImGui::TextDisabled("Data columns %llu-%llu of %llu",
        static_cast<unsigned long long>(columns ? column_offset_ + 1 : 0),
        static_cast<unsigned long long>(std::min<uint64_t>(columns, column_offset_ + 32)),
        static_cast<unsigned long long>(columns));
    ImGui::PopID();
    if (Busy()) ImGui::TextDisabled("Inspecting...");
    RenderError();
    if (page_.ok) {
        RenderDataPreviewTable("hdf5_sample", page_.schema, page_.offset, page_.rows_returned,
            [this](int64_t row) -> const DataPreviewRow* {
                const auto local = row - page_.offset;
                return local >= 0 && local < static_cast<int64_t>(page_.rows.size())
                    ? &page_.rows[static_cast<size_t>(local)] : nullptr;
            }, ImVec2(0, std::max(140.0f, ImGui::GetContentRegionAvail().y - 10.0f)), true, nullptr, &view_);
        RenderDataPreviewCellDetails(view_);
    }
    ImGui::PopID();
}
} // namespace gui
