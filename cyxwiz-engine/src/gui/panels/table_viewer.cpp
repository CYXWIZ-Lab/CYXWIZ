#include "table_viewer.h"
#include "../../core/variables_presentation.h"
#include "../../core/data_registry.h"
#include "../../core/dataset_catalog.h"
#include "../../core/dataset_table_slice.h"
#include "../ui_widgets.h"
#include "../ui_fonts.h"
#include "../editor_fonts.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../separate_windows.h"
#include <imgui_internal.h>
#include "visualization_panel.h"
#include "../icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <cfloat>
#include <cstring>
#include <limits>
#include <numeric>
#include <algorithm>
#include <map>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace cyxwiz {

TableViewerPanel::TableViewerPanel()
    : Panel("Table Viewer", false)  // Start hidden
{
}

void TableViewerPanel::Render() {
    // The Plot window stays while the Table Viewer is closed or hidden.
    if (plot_window_) plot_window_->Render();
    if (!visible_) return;

    // Handle deferred tab close
    if (close_tab_index_ >= 0 && close_tab_index_ < static_cast<int>(tabs_.size())) {
        tabs_.erase(tabs_.begin() + close_tab_index_);
        if (active_tab_index_ >= static_cast<int>(tabs_.size())) {
            active_tab_index_ = static_cast<int>(tabs_.size()) - 1;
        }
        close_tab_index_ = -1;
    }

    // Handle keyboard shortcuts
    ImGuiIO& io = ImGui::GetIO();
    TableTab* shortcut_tab = GetActiveTab();
    if (shortcut_tab) {
        // Ctrl+S: Save
        if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_S)) {
            if (shortcut_tab->is_dirty) {
                SaveTable(shortcut_tab);
            }
        }
        // Escape: Cancel editing
        if (ImGui::IsKeyPressed(ImGuiKey_Escape) && shortcut_tab->grid.editing_row >= 0) {
            shortcut_tab->grid.EndCellEdit(false);
        }
    }

    // Not docked yet (layout saved before it had a slot): open at a usable size.
    ImGui::SetNextWindowSize(ImVec2(900.0f, 560.0f), ImGuiCond_FirstUseEver);
    // A size saved while floating can be a sliver (121 px wide was seen):
    // floating, the viewer is never narrower than its toolbar and a few columns.
    ImGui::SetNextWindowSizeConstraints(ImVec2(480.0f, 300.0f), ImVec2(FLT_MAX, FLT_MAX));
    // Collapsed or behind another dock tab: skip the body (TOFIX129 0.6).
    ::gui::NextWindowMayLeave(GetName());
    const bool expanded = ImGui::Begin(GetName(), &visible_);
    ::gui::TabMenu(GetName(), &visible_);
    if (expanded) {
        // Tab bar at top, the live-value line (Data Viewer), the toolbar.
        RenderTabBar();
        RenderLiveHeader(GetActiveTab());
        RenderToolbar();
        ImGui::Dummy(ImVec2(0.0f, 2.0f));

        // Table display
        TableTab* active_tab = GetActiveTab();
        if (active_tab) {
            if (active_tab->grid.table) {
                // 3-pane layout: sidebar + splitter + main table
                if (show_stats_sidebar_) {
                    RenderStatsSidebar(active_tab);
                    ImGui::SameLine();

                    // Draggable splitter: a gap, a line in the surface tone on hover.
                    const ImVec2 split = ImGui::GetCursorScreenPos();
                    ImGui::InvisibleButton("##vsplitter", ImVec2(6.0f, std::max(1.0f, ImGui::GetContentRegionAvail().y - 30.0f)));
                    if (ImGui::IsItemHovered() || ImGui::IsItemActive())
                        ImGui::GetWindowDrawList()->AddLine(ImVec2(split.x + 3.0f, split.y), ImVec2(split.x + 3.0f, ImGui::GetItemRectMax().y),
                                                            ui::ToU32(ui::CurrentTokens().border), 1.0f);
                    if (ImGui::IsItemActive()) {
                        stats_sidebar_width_ += ImGui::GetIO().MouseDelta.x;
                        stats_sidebar_width_ = std::clamp(stats_sidebar_width_, 120.0f, 300.0f);
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
                    }
                    ImGui::SameLine();
                }

                // Main table area
                ImGui::BeginChild("TableContent", ImVec2(0, -30));
                RenderTable();
                ImGui::EndChild();
            } else {
                ImGui::TextWrapped("Failed to load table.");
            }
        } else {
            RenderEmptyState();
        }

        // Status bar
        RenderStatusBar();

    }
    ImGui::End();

    // Render modal dialogs (must be outside main window)
    RenderExportDialog();
    RenderFindDialog();
}

void TableViewerPanel::RenderLiveHeader(TableTab* tab) {
    if (!tab || !tab->live.on) return;
    const ui::Tokens& t = ui::CurrentTokens();
    auto& live = tab->live;
    vars::LiveTable lt;
    lt.name = live.request.name;
    lt.scope_label = live.request.scope.label;
    lt.kind = live.kind;
    lt.shape = live.shape;
    lt.rows = live.rows;
    lt.shown = live.shown;
    lt.columns = static_cast<long long>(tab->GetColumnCount()) - (live.kind == "frame" ? 1 : 0);
    lt.dtype = live.dtypes.empty() ? std::string() : live.dtypes.front();
    lt.slice = live.slice;

    // A tinted line: where the data comes from, and that it is a snapshot.
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float w = ImGui::GetContentRegionAvail().x;
    const std::string header = vars::LiveHeader(lt, live.read_at);
    const char* note = "Edits stay in this view";
    const float right = ImGui::CalcTextSize(note).x + ui::ButtonWidth("Read again", ui::ButtonSize::Small) + 30.0f;
    const bool one_line = 26.0f + ImGui::CalcTextSize(header.c_str()).x + 24.0f + right <= w;
    const float h = (one_line ? 1.0f : 2.0f) * ImGui::GetFrameHeight() + 8.0f;
    dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), ui::ToU32(ui::Mix(t.bg_window, t.accent, 0.10f)), 4.0f);
    ImGui::SetCursorScreenPos(ImVec2(p.x + 10.0f, p.y + 4.0f));
    const float mid = p.y + 4.0f + ImGui::GetFrameHeight() * 0.5f;
    dl->AddCircleFilled(ImVec2(p.x + 14.0f, mid), 3.5f, ui::ToU32(live.problem.empty() ? t.success : t.warning));
    ImGui::SetCursorScreenPos(ImVec2(p.x + 26.0f, p.y + 4.0f));
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(header.c_str());
    if (one_line) ImGui::SameLine(ImGui::GetContentRegionMax().x - right);
    else ImGui::SetCursorScreenPos(ImVec2(p.x + 26.0f, p.y + 4.0f + ImGui::GetFrameHeight()));
    if (live.reading) ImGui::TextColored(t.text_dim, "%s", "Reading...");
    else if (ui::LinkButton("Read again")) ReadLive(tab, live.slice, live.request.max_rows);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("Read the value from Python again");
    ImGui::SameLine(0.0f, 14.0f);
    ImGui::TextColored(t.text_dim, "%s", note);
    ImGui::SetCursorScreenPos(ImVec2(p.x, p.y + h + 4.0f));
    if (!live.problem.empty()) {
        ImGui::TextColored(t.warning, ICON_FA_TRIANGLE_EXCLAMATION " %s", live.problem.c_str());
    }

    // More than two axes: which slice (Up/Down step the first index).
    if (live.kind == "array" && live.shape.size() > 2) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_dim, "%s", "Showing");
        ImGui::SameLine();
        ImFont* mono = gui::GetCodeFont();
        if (mono) ImGui::PushFont(mono);
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted((live.request.name + "[").c_str());
        if (mono) ImGui::PopFont();
        std::vector<int> wanted = live.slice;
        bool step = false;
        const size_t lead = live.shape.size() - 2;
        wanted.resize(lead, 0);
        for (size_t axis = 0; axis < lead; ++axis) {
            ImGui::PushID(static_cast<int>(axis));
            ImGui::SameLine(0.0f, 2.0f);
            if (ui::GhostButton(ICON_FA_MINUS, wanted[axis] > 0 && !live.reading, "First index")) {
                --wanted[axis];
                step = true;
            }
            ImGui::SameLine(0.0f, 4.0f);
            ImGui::AlignTextToFramePadding();
            ImGui::Text("%d", wanted[axis]);
            ImGui::SameLine(0.0f, 4.0f);
            if (ui::GhostButton(ICON_FA_PLUS, wanted[axis] + 1 < live.shape[axis] && !live.reading, "Last index")) {
                ++wanted[axis];
                step = true;
            }
            ImGui::SameLine(0.0f, 2.0f);
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(",");
            ImGui::PopID();
        }
        ImGui::SameLine(0.0f, 6.0f);
        if (mono) ImGui::PushFont(mono);
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(":, :]");
        if (mono) ImGui::PopFont();
        ImGui::SameLine(0.0f, 10.0f);
        const size_t n = live.shape.size();
        ImGui::TextColored(t.text_dim, "\xC2\xB7 %s \xC3\x97 %s \xC2\xB7 Up/Down step the first index",
                           vars::Thousands(live.shape[n - 2]).c_str(), vars::Thousands(live.shape[n - 1]).c_str());
        const bool keys_ours = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) && tab->grid.editing_row < 0;
        if (keys_ours) {
            // Up/Down step the slice; owning them keeps ImGui's keyboard
            // navigation from also moving in the table on the same press.
            const ImGuiID owner = ImGui::GetID("##slice_keys");
            ImGui::SetKeyOwner(ImGuiKey_UpArrow, owner);
            ImGui::SetKeyOwner(ImGuiKey_DownArrow, owner);
        }
        if (keys_ours && !ImGui::IsAnyItemActive() && !live.reading) {
            if (ImGui::IsKeyPressed(ImGuiKey_UpArrow) && wanted[0] > 0) {
                --wanted[0];
                step = true;
            } else if (ImGui::IsKeyPressed(ImGuiKey_DownArrow) && wanted[0] + 1 < live.shape[0]) {
                ++wanted[0];
                step = true;
            }
        }
        if (step) ReadLive(tab, wanted, live.request.max_rows);
    }

    // A stated row limit, with the way past it.
    const std::string limit = vars::LimitText(lt);
    if (!limit.empty()) {
        const ImVec2 q = ImGui::GetCursorScreenPos();
        const float lh = ImGui::GetFrameHeight() + 6.0f;
        dl->AddRectFilled(q, ImVec2(q.x + w, q.y + lh), ui::ToU32(ui::WithAlpha(t.warning, 0.10f)), 4.0f);
        ImGui::SetCursorScreenPos(ImVec2(q.x + 10.0f, q.y + 3.0f));
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.warning, "%s", ICON_FA_TRIANGLE_EXCLAMATION);
        ImGui::SameLine();
        ImGui::TextUnformatted(limit.c_str());
        ImGui::SameLine(0.0f, 14.0f);
        if (!live.reading && ui::LinkButton("Read all rows")) ReadLive(tab, live.slice, 0);
        ImGui::SetCursorScreenPos(ImVec2(q.x, q.y + lh + 4.0f));
    }
    ImGui::Dummy(ImVec2(0.0f, 2.0f));
}

void TableViewerPanel::RenderTabBar() {
    if (tabs_.empty()) return;  // the empty state says what to do

    ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_Reorderable |
                                      ImGuiTabBarFlags_AutoSelectNewTabs |
                                      ImGuiTabBarFlags_TabListPopupButton |
                                      ImGuiTabBarFlags_FittingPolicyScroll;

    if (ImGui::BeginTabBar("TableViewerTabs", tab_bar_flags)) {
        for (int i = 0; i < static_cast<int>(tabs_.size()); i++) {
            auto& tab = tabs_[i];

            // Tab name with loading/dirty indicator
            std::string tab_name = tab->filename;
            if (tab->is_dirty) {
                tab_name += " *";  // Unsaved changes indicator
            }
            tab_name = ICON_FA_TABLE " " + tab_name + "###table_tab" + std::to_string(tab->uid);

            // Make tab closable
            bool tab_open = true;
            ImGuiTabItemFlags tab_flags = ImGuiTabItemFlags_None;

            if (select_tab_ == i) {
                tab_flags |= ImGuiTabItemFlags_SetSelected;
                select_tab_ = -1;
            }
            if (ImGui::BeginTabItem(tab_name.c_str(), &tab_open, tab_flags)) {
                active_tab_index_ = i;
                ImGui::EndTabItem();
            }

            // Handle tab close
            if (!tab_open) {
                close_tab_index_ = i;
            }
        }
        ImGui::EndTabBar();
    }
}

void TableViewerPanel::RenderToolbar() {
    TableTab* active_tab = GetActiveTab();
    if (!active_tab) return;

    // Another loaded dataset, beside the open tabs.
    if (ui::GhostButton(ICON_FA_DATABASE " Open dataset")) open_dataset_popup_requested_ = true;
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Open the first rows of a dataset a Data Input loaded");
    RenderOpenDatasetPopup();
    ImGui::SameLine();

    // Stats sidebar toggle
    if (ui::GhostButton(ICON_FA_CHART_BAR " Stats", true, nullptr, show_stats_sidebar_)) {
        show_stats_sidebar_ = !show_stats_sidebar_;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Toggle Statistics Sidebar");

    ImGui::SameLine();
    ImGui::Checkbox("Data Bars", &show_data_bars_);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Show data bars in numeric columns");

    ImGui::SameLine();
    ImGui::Checkbox("Line #", &show_line_numbers_);

    ImGui::SameLine();
    ImGui::SetNextItemWidth(70);
    if (ImGui::InputInt("##RowsPerPage", &rows_per_page_, 0, 0)) {
        rows_per_page_ = std::clamp(rows_per_page_, 10, 1000);
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Rows per page");

    ImGui::SameLine(0.0f, 18.0f);

    // Filter
    ImGui::Text(ICON_FA_FILTER);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(150);
    bool filter_changed = ImGui::InputText("##Filter", active_tab->grid.filter_buffer, sizeof(active_tab->grid.filter_buffer),
        ImGuiInputTextFlags_EnterReturnsTrue);
    if (filter_changed) active_tab->grid.SetFilter(active_tab->grid.filter_buffer);

    // Filter mode toggle
    ImGui::SameLine();
    if (ImGui::Checkbox("Hide", &active_tab->grid.filter_mode_hide)) {
        if (active_tab->grid.Hiding()) active_tab->grid.ApplyFilter();
        active_tab->grid.current_page = 0;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Hide non-matching rows instead of highlighting");

    // Clear filter button
    if (!active_tab->grid.filter_text.empty()) {
        ImGui::SameLine();
        if (ui::GhostButton(ICON_FA_XMARK "##ClearFilter")) {
            active_tab->grid.ClearFilter();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Clear filter");
    }

    ImGui::SameLine(0.0f, 18.0f);

    // Column freeze control
    ImGui::Text(ICON_FA_LOCK);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Frozen columns");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(50);
    if (ImGui::InputInt("##Freeze", &active_tab->grid.frozen_columns, 0, 0)) {
        active_tab->grid.frozen_columns = std::clamp(active_tab->grid.frozen_columns, 0,
            static_cast<int>(active_tab->grid.table ? active_tab->grid.table->GetColumnCount() : 0));
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Number of columns to freeze");

    ImGui::SameLine(0.0f, 18.0f);

    // Find button
    if (ui::GhostButton(ICON_FA_MAGNIFYING_GLASS " Find")) {
        show_find_dialog_ = true;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Find in table");

    // Save button (only enabled if dirty)
    if (active_tab->grid.table) {
        ImGui::SameLine();
        if (ui::GhostButton(ICON_FA_FLOPPY_DISK " Save", active_tab->is_dirty && !active_tab->live.on,
                            active_tab->live.on ? "A value from Python: export it instead" : "No unsaved changes"))
            SaveTable(active_tab);
        if (active_tab->is_dirty && ImGui::IsItemHovered()) ImGui::SetTooltip("Save changes (Ctrl+S)");

        // Export button
        ImGui::SameLine();
        if (ui::GhostButton(ICON_FA_FILE_EXPORT " Export")) {
            show_export_dialog_ = true;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Export table to file");
    }

    // Close tab button
    ImGui::SameLine();
    if (ui::GhostButton(ICON_FA_XMARK "##close_tab")) {
        close_tab_index_ = active_tab_index_;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Close Tab");
}

void TableViewerPanel::RenderTable() {
    TableTab* tab = GetActiveTab();
    if (!tab || !tab->HasData()) return;
    tab->grid.options.line_numbers = show_line_numbers_;
    tab->grid.options.data_bars = show_data_bars_;
    tab->grid.options.rows_per_page = rows_per_page_;
    // Keyed by the table's name: its column widths are its own (one shared
    // id gave every table the widths last saved for any of them).
    tab->grid.Render(("##grid_" + tab->filename).c_str());
}

void TableViewerPanel::RenderStatusBar() {
    TableTab* tab = GetActiveTab();
    if (!tab) return;

    if (!tab->HasData()) {
        ImGui::TextDisabled("Failed to load table");
        return;
    }

    // Left: Table info
    ImGui::Text(ICON_FA_TABLE " %zu rows x %zu cols",
        tab->GetRowCount(), tab->GetColumnCount());

    // Middle: Selection info
    ImGui::SameLine(0.0f, 18.0f);
    if (tab->grid.selected_row >= 0 && tab->grid.selected_col >= 0) {
        ImGui::Text("Cell: Row %d, Col %d", tab->grid.selected_row + 1, tab->grid.selected_col + 1);
    } else {
        ImGui::TextDisabled("No cell selected");
    }

    // Right: Memory estimate for in-memory table
    ImGui::SameLine(0.0f, 18.0f);
    size_t mem_bytes = tab->GetRowCount() * tab->GetColumnCount() * sizeof(double);
    if (mem_bytes > 1024 * 1024) {
        ImGui::Text(ICON_FA_MEMORY " %.1f MB", mem_bytes / (1024.0 * 1024.0));
    } else {
        ImGui::Text(ICON_FA_MEMORY " %.1f KB", mem_bytes / 1024.0);
    }
}

bool TableViewerPanel::OpenDataset(const std::string& dataset_name, int64_t max_rows, std::string& error) {
    auto& registry = DataRegistry::Instance();
    DatasetSlice slice = ReadDatasetSlice(
        [&registry](const DataPreviewRequest& request) {
            return DataPreviewService::PreviewRegisteredTabular(registry, request);
        },
        dataset_name, max_rows);
    if (!slice.table) {
        error = slice.error;
        return false;
    }
    const size_t shown = slice.table->GetRowCount();
    if (static_cast<int64_t>(shown) < slice.total_rows)
        slice.table->SetName(dataset_name + " \xC2\xB7 first " + std::to_string(shown) + " of " +
                             std::to_string(slice.total_rows) + " rows");
    SetTable(slice.table);
    visible_ = true;
    return true;
}

void TableViewerPanel::RenderEmptyState() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::Spacing();
    {
        ui::FontScope font(ui::Font::Medium);
        ImGui::TextUnformatted("No table open");
    }
    ImGui::PushTextWrapPos(std::min(ImGui::GetContentRegionMax().x, ImGui::GetCursorPosX() + 560.0f));
    ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
    ImGui::TextUnformatted("The Table Viewer shows a table held in memory, to sort, filter, format, edit or plot. "
                           "Tables come here from:");
    ImGui::PopStyleColor();
    for (const char* route : {"a loaded dataset - its first rows, with the button below",
                              "Data Studio - open a query's result",
                              "the Variable Explorer - View data on a Python value",
                              "a notebook or script that shows a table"}) {
        ImGui::Bullet();
        ImGui::TextUnformatted(route);
    }
    ImGui::PopTextWrapPos();
    ImGui::Spacing();
    if (ui::PrimaryButton(ICON_FA_DATABASE " Open a loaded dataset...")) open_dataset_popup_requested_ = true;
    RenderOpenDatasetPopup();
}

void TableViewerPanel::RenderOpenDatasetPopup() {
    if (open_dataset_popup_requested_) {
        open_dataset_popup_requested_ = false;
        open_dataset_error_.clear();
        ImGui::OpenPopup("##open_dataset");
    }
    ImGui::SetNextWindowSizeConstraints(ImVec2(420.0f, 0.0f), ImVec2(640.0f, 560.0f));
    if (!ImGui::BeginPopup("##open_dataset")) return;
    const ui::Tokens& t = ui::CurrentTokens();

    // Tabular datasets the Engine holds (Data Inputs, pipeline results).
    std::vector<DatasetEntry> tables;
    for (auto& entry : DatasetCatalog::Instance().List())
        if (entry.Has(kBackingArrow) || entry.Has(kBackingParquet)) tables.push_back(std::move(entry));

    ui::SettingsSection("Loaded datasets");
    if (tables.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled("No dataset is loaded. Apply a Data Input (or run a pipeline) first.");
        ImGui::PopTextWrapPos();
    } else {
        ImGui::BeginChild("##datasets", ImVec2(0.0f, std::min(240.0f, 28.0f * static_cast<float>(tables.size()) + 8.0f)));
        for (const auto& entry : tables) {
            const std::string detail = std::to_string(entry.rows) + " rows \xC2\xB7 " + std::to_string(entry.columns) +
                                       " columns \xC2\xB7 " + StorageText(entry.storage);
            if (ImGui::Selectable((entry.Shown() + "##" + entry.name).c_str(), open_dataset_pick_ == entry.name))
                open_dataset_pick_ = entry.name;
            ImGui::SameLine();
            ImGui::TextColored(t.text_dim, "%s", detail.c_str());
        }
        ImGui::EndChild();
    }

    ui::FieldLabel("Rows to open");
    ImGui::SetNextItemWidth(160.0f);
    if (ImGui::InputInt("##rows", &open_dataset_rows_, 1000, 10000))
        open_dataset_rows_ = std::clamp(open_dataset_rows_, 1, 1000000);
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 400.0f);
    ImGui::TextDisabled("Reads only these first rows; the dataset stays where its Data Input loaded it.");
    ImGui::PopTextWrapPos();
    if (!open_dataset_error_.empty()) ImGui::TextColored(t.warning, "%s", open_dataset_error_.c_str());

    ImGui::Spacing();
    const bool can_open = !open_dataset_pick_.empty() &&
                          std::any_of(tables.begin(), tables.end(),
                                      [this](const DatasetEntry& e) { return e.name == open_dataset_pick_; });
    if (ui::PrimaryButton("Open", can_open, "Pick a dataset")) {
        if (OpenDataset(open_dataset_pick_, open_dataset_rows_, open_dataset_error_)) ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ui::SecondaryButton("Cancel", true, nullptr, ui::ButtonSize::Regular)) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
}

}  // namespace cyxwiz
