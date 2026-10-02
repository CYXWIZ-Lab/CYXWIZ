#include "table_viewer.h"
#include "../../core/variables_presentation.h"
#include "../editor_fonts.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
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
        if (ImGui::IsKeyPressed(ImGuiKey_Escape) && shortcut_tab->editing_row >= 0) {
            EndCellEdit(shortcut_tab, false);
        }
    }

    // Not docked yet (layout saved before it had a slot): open at a usable size.
    ImGui::SetNextWindowSize(ImVec2(900.0f, 560.0f), ImGuiCond_FirstUseEver);
    // A size saved while floating can be a sliver (121 px wide was seen):
    // floating, the viewer is never narrower than its toolbar and a few columns.
    ImGui::SetNextWindowSizeConstraints(ImVec2(480.0f, 300.0f), ImVec2(FLT_MAX, FLT_MAX));
    // Collapsed or behind another dock tab: skip the body (TOFIX129 0.6).
    const bool expanded = ImGui::Begin(GetName(), &visible_);
    if (expanded) {
        // Tab bar at top, the live-value line (Data Viewer), the toolbar.
        RenderTabBar();
        RenderLiveHeader(GetActiveTab());
        RenderToolbar();
        ImGui::Dummy(ImVec2(0.0f, 2.0f));

        // Table display
        TableTab* active_tab = GetActiveTab();
        if (active_tab) {
            if (active_tab->table) {
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
            ImGui::TextWrapped("No table loaded. Registered datasets are previewed from Asset Browser or Data Input through Data Preview.");
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
        const bool keys_ours = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) && tab->editing_row < 0;
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
    if (tabs_.empty()) {
        ImGui::TextDisabled("No tables open");
        return;
    }

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
            tab_name = ICON_FA_TABLE " " + tab_name;

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
    bool filter_changed = ImGui::InputText("##Filter", active_tab->filter_buffer, sizeof(active_tab->filter_buffer),
        ImGuiInputTextFlags_EnterReturnsTrue);
    if (filter_changed) {
        active_tab->filter_text = active_tab->filter_buffer;
        if (active_tab->filter_mode_hide) {
            ApplyFilter(active_tab);
        }
    }

    // Filter mode toggle
    ImGui::SameLine();
    if (ImGui::Checkbox("Hide", &active_tab->filter_mode_hide)) {
        if (active_tab->filter_mode_hide && !active_tab->filter_text.empty()) {
            ApplyFilter(active_tab);
        } else {
            active_tab->filtered_indices.clear();
        }
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Hide non-matching rows instead of highlighting");

    // Clear filter button
    if (!active_tab->filter_text.empty()) {
        ImGui::SameLine();
        if (ui::GhostButton(ICON_FA_XMARK "##ClearFilter")) {
            ClearFilter(active_tab);
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Clear filter");
    }

    ImGui::SameLine(0.0f, 18.0f);

    // Column freeze control
    ImGui::Text(ICON_FA_LOCK);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Frozen columns");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(50);
    if (ImGui::InputInt("##Freeze", &active_tab->frozen_columns, 0, 0)) {
        active_tab->frozen_columns = std::clamp(active_tab->frozen_columns, 0,
            static_cast<int>(active_tab->table ? active_tab->table->GetColumnCount() : 0));
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Number of columns to freeze");

    ImGui::SameLine(0.0f, 18.0f);

    // Find button
    if (ui::GhostButton(ICON_FA_MAGNIFYING_GLASS " Find")) {
        show_find_dialog_ = true;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Find in table");

    // Save button (only enabled if dirty)
    if (active_tab->table) {
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
    TableTab* active_tab = GetActiveTab();
    if (!active_tab || !active_tab->HasData()) return;

    // Use unified accessors
    size_t row_count = active_tab->GetRowCount();
    size_t col_count = active_tab->GetColumnCount();

    if (row_count == 0 || col_count == 0) {
        ImGui::Text("Table is empty");
        return;
    }

    // Compute column stats if not done
    if (active_tab->column_stats.empty()) {
        ComputeColumnStats(active_tab);
    }

    // Initialize sorted indices if empty
    if (active_tab->sorted_indices.empty()) {
        active_tab->sorted_indices.resize(row_count);
        std::iota(active_tab->sorted_indices.begin(), active_tab->sorted_indices.end(), size_t(0));
    }

    // Determine which indices to display (sorted or filtered)
    const std::vector<size_t>& display_indices = (active_tab->filter_mode_hide && !active_tab->filtered_indices.empty())
        ? active_tab->filtered_indices
        : active_tab->sorted_indices;

    size_t display_count = display_indices.size();

    // Calculate pagination based on display count
    size_t total_pages = (display_count + rows_per_page_ - 1) / rows_per_page_;
    if (total_pages == 0) total_pages = 1;
    size_t start_row = active_tab->current_page * rows_per_page_;
    size_t end_row = std::min(start_row + rows_per_page_, display_count);

    // ImGui table flags
    ImGuiTableFlags flags = ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg |
                           ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX |
                           ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable |
                           ImGuiTableFlags_Hideable | ImGuiTableFlags_Sortable |
                           ImGuiTableFlags_SizingFixedFit;

    int column_count = static_cast<int>(col_count);
    if (show_line_numbers_) {
        column_count++;
    }

    if (ImGui::BeginTable("DataTable", column_count, flags)) {
        // Apply column freeze for horizontal scrolling
        // frozen_columns + 1 if line numbers are shown (line number column + frozen data columns)
        // 1 row frozen for header
        int freeze_cols = active_tab->frozen_columns;
        if (show_line_numbers_ && freeze_cols > 0) {
            freeze_cols++;  // Account for line number column
        }
        ImGui::TableSetupScrollFreeze(freeze_cols, 1);  // Freeze columns + 1 header row

        // Setup columns with type indicators and sort arrows
        if (show_line_numbers_) {
            ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_DefaultSort, 50.0f);
        }

        const auto& headers = active_tab->GetHeaders();
        for (size_t i = 0; i < col_count; i++) {
            std::string header = i < headers.size() ? headers[i] : ("Col" + std::to_string(i));

            // Add type indicator
            if (i < active_tab->column_stats.size()) {
                auto& stats = active_tab->column_stats[i];
                if (stats.type == "Numeric") {
                    header = ICON_FA_HASHTAG " " + header;
                } else {
                    header = ICON_FA_FONT " " + header;
                }
            }

            ImGui::TableSetupColumn(header.c_str(), ImGuiTableColumnFlags_DefaultSort);
        }

        // -----------------------------------------------------------
        // Manual header row with context menu support
        // -----------------------------------------------------------
        // A value from Python shows each column's dtype under its name.
        const bool dtype_row = active_tab->live.on && !active_tab->live.dtypes.empty();
        ImGui::TableNextRow(ImGuiTableRowFlags_Headers, dtype_row ? ImGui::GetTextLineHeight() * 2.0f + 6.0f : 0.0f);

        // Line number column header (if enabled)
        if (show_line_numbers_) {
            ImGui::TableSetColumnIndex(0);
            ImGui::TableHeader("#");
        }

        // Data column headers with context menus
        for (size_t i = 0; i < col_count; i++) {
            int table_col = show_line_numbers_ ? static_cast<int>(i) + 1 : static_cast<int>(i);
            ImGui::TableSetColumnIndex(table_col);

            // Build header text with type indicator
            std::string header = i < headers.size() ? headers[i] : ("Col" + std::to_string(i));
            if (i < active_tab->column_stats.size()) {
                auto& stats = active_tab->column_stats[i];
                if (stats.type == "Numeric") {
                    header = ICON_FA_HASHTAG " " + header;
                } else {
                    header = ICON_FA_FONT " " + header;
                }
            }

            // Render clickable header
            ImGui::TableHeader(header.c_str());
            if (dtype_row && i < active_tab->live.dtypes.size() && active_tab->live.dtypes[i] != "index") {
                const ImVec2 h0 = ImGui::GetItemRectMin();
                ImGui::GetWindowDrawList()->AddText(ImVec2(h0.x + ImGui::GetStyle().CellPadding.x, h0.y + ImGui::GetTextLineHeight() + 3.0f),
                                                    ui::ToU32(ui::CurrentTokens().text_dim), active_tab->live.dtypes[i].c_str());
            }

            // Right-click context menu on header
            if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
                context_menu_column_ = static_cast<int>(i);
            }
            if (ImGui::BeginPopupContextItem(("ColumnContextMenu_" + std::to_string(i)).c_str())) {
                RenderColumnContextMenu(active_tab, static_cast<int>(i));
                ImGui::EndPopup();
            }
        }

        // Handle ImGui's built-in sorting
        if (ImGuiTableSortSpecs* sort_specs = ImGui::TableGetSortSpecs()) {
            if (sort_specs->SpecsDirty && sort_specs->SpecsCount > 0) {
                const ImGuiTableColumnSortSpecs& spec = sort_specs->Specs[0];
                int sort_col = spec.ColumnIndex;
                if (show_line_numbers_) sort_col--;  // Adjust for line number column

                if (sort_col < 0) {
                    // Sorting by line number column - reset to original order
                    active_tab->sort_column = -1;
                    active_tab->sorted_indices.resize(row_count);
                    if (spec.SortDirection == ImGuiSortDirection_Ascending) {
                        std::iota(active_tab->sorted_indices.begin(), active_tab->sorted_indices.end(), size_t(0));
                    } else {
                        // Reverse order
                        for (size_t i = 0; i < row_count; i++) {
                            active_tab->sorted_indices[i] = row_count - 1 - i;
                        }
                    }
                    spdlog::info("Reset to original order ({})",
                        spec.SortDirection == ImGuiSortDirection_Ascending ? "ascending" : "descending");
                } else {
                    active_tab->sort_column = sort_col;
                    active_tab->sort_ascending = (spec.SortDirection == ImGuiSortDirection_Ascending);
                    SortByColumn(active_tab, sort_col);
                }
                sort_specs->SpecsDirty = false;
            }
        }

        // Auto-select first column if none selected
        if (active_tab->selected_column < 0 && !active_tab->column_stats.empty()) {
            active_tab->selected_column = 0;
        }

        // Render rows with clipper for performance using display indices
        ImGuiListClipper clipper;
        clipper.Begin(static_cast<int>(end_row - start_row));

        while (clipper.Step()) {
            for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; i++) {
                size_t display_idx = start_row + i;
                size_t actual_row = display_indices[display_idx];

                ImGui::TableNextRow();

                // Line number column
                int col_idx = 0;
                if (show_line_numbers_) {
                    ImGui::TableSetColumnIndex(col_idx++);
                    ImGui::TextDisabled("%zu", actual_row + 1);
                }

                // Data columns
                for (size_t c = 0; c < col_count; c++) {
                    ImGui::TableSetColumnIndex(col_idx++);

                    std::string cell_text = active_tab->GetCellAsString(actual_row, c);

                    // Apply number formatting if configured for this column
                    if (c < active_tab->column_formats.size() && c < active_tab->column_stats.size() &&
                        active_tab->column_stats[c].type == "Numeric") {
                        auto& fmt = active_tab->column_formats[c];
                        // Only format if any formatting option is set
                        if (fmt.decimal_places >= 0 || fmt.thousands_separator ||
                            fmt.as_percentage || !fmt.prefix.empty() || !fmt.suffix.empty()) {
                            auto cell = active_tab->GetCell(actual_row, c);
                            double val = 0;
                            bool is_numeric = false;
                            if (std::holds_alternative<double>(cell)) {
                                val = std::get<double>(cell);
                                is_numeric = true;
                            } else if (std::holds_alternative<int64_t>(cell)) {
                                val = static_cast<double>(std::get<int64_t>(cell));
                                is_numeric = true;
                            }
                            if (is_numeric) {
                                cell_text = FormatNumber(val, fmt);
                            }
                        }
                    }

                    // Check for colormap on this column
                    bool has_colormap = (c < active_tab->column_colormaps.size() &&
                                        active_tab->column_colormaps[c].type != ColorMapType::None);

                    // Data bar or colormap for numeric columns
                    if ((show_data_bars_ || has_colormap) && c < active_tab->column_stats.size()) {
                        auto& stats = active_tab->column_stats[c];
                        if (stats.type == "Numeric" && stats.max_val > stats.min_val) {
                            auto cell = active_tab->GetCell(actual_row, c);
                            double val = 0;
                            if (std::holds_alternative<double>(cell)) {
                                val = std::get<double>(cell);
                            } else if (std::holds_alternative<int64_t>(cell)) {
                                val = static_cast<double>(std::get<int64_t>(cell));
                            }

                            float norm = static_cast<float>((val - stats.min_val) / (stats.max_val - stats.min_val));
                            norm = std::clamp(norm, 0.0f, 1.0f);

                            ImVec2 pos = ImGui::GetCursorScreenPos();
                            float cell_width = ImGui::GetContentRegionAvail().x;
                            float cell_height = ImGui::GetTextLineHeight();

                            if (has_colormap) {
                                // Use colormap for full cell background
                                auto& cmap = active_tab->column_colormaps[c];
                                ImVec4 color = GetColorMapColor(norm, cmap.type);
                                color.w = 0.4f;  // Semi-transparent background
                                ImGui::GetWindowDrawList()->AddRectFilled(
                                    pos, ImVec2(pos.x + cell_width, pos.y + cell_height),
                                    ImGui::GetColorU32(color));
                            } else if (show_data_bars_) {
                                // Default data bar
                                float bar_width = cell_width * norm;
                                ImU32 bar_color = ui::ToU32(ui::WithAlpha(ui::CurrentTokens().accent, data_bar_alpha_));
                                ImGui::GetWindowDrawList()->AddRectFilled(
                                    pos, ImVec2(pos.x + bar_width, pos.y + cell_height),
                                    bar_color);
                            }
                        }
                    }

                    // Check if cell is selected (single selection or multi-selection)
                    int cell_row = static_cast<int>(actual_row);
                    int cell_col = static_cast<int>(c);
                    bool is_selected = (active_tab->selected_row == cell_row &&
                                       active_tab->selected_col == cell_col);

                    // Check multi-selection ranges
                    bool in_multi_selection = false;
                    for (const auto& sel : active_tab->selections) {
                        if (sel.Contains(cell_row, cell_col)) {
                            in_multi_selection = true;
                            break;
                        }
                    }

                    // Apply filter highlighting color
                    bool has_filter_match = !active_tab->filter_text.empty() &&
                        cell_text.find(active_tab->filter_text) != std::string::npos;

                    // Apply multi-selection background color
                    if (in_multi_selection && !is_selected) {
                        ImGui::PushStyleColor(ImGuiCol_Header, ui::CurrentTokens().selection);
                    }

                    if (has_filter_match) {
                        ImGui::PushStyleColor(ImGuiCol_Text, ui::CurrentTokens().warning);
                    }

                    // Push unique ID for each cell to avoid conflicts
                    ImGui::PushID(static_cast<int>(actual_row * col_count + c));

                    // Check if this cell is being edited
                    bool is_editing = (active_tab->editing_row == cell_row &&
                                      active_tab->editing_col == cell_col);

                    if (is_editing) {
                        // -----------------------------------------------------------
                        // EDITING MODE: Show InputText
                        // -----------------------------------------------------------
                        ImGui::SetNextItemWidth(-1);  // Fill available width

                        // Focus on first frame of editing
                        if (active_tab->edit_just_started) {
                            ImGui::SetKeyboardFocusHere();
                            active_tab->edit_just_started = false;
                        }

                        ImGuiInputTextFlags input_flags = ImGuiInputTextFlags_EnterReturnsTrue |
                                                          ImGuiInputTextFlags_AutoSelectAll;

                        if (ImGui::InputText("##CellEdit", active_tab->edit_buffer,
                                            sizeof(active_tab->edit_buffer), input_flags)) {
                            // Enter pressed - save and end editing
                            EndCellEdit(active_tab, true);
                        }

                        // Handle Escape to cancel
                        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                            EndCellEdit(active_tab, false);
                        }

                        // Handle click outside to save
                        if (!ImGui::IsItemActive() && ImGui::IsMouseClicked(0)) {
                            EndCellEdit(active_tab, true);
                        }
                    } else {
                        // -----------------------------------------------------------
                        // NORMAL MODE: Selectable cell
                        // -----------------------------------------------------------
                        ImGuiSelectableFlags sel_flags = ImGuiSelectableFlags_AllowDoubleClick;

                        if (ImGui::Selectable(cell_text.c_str(), is_selected || in_multi_selection, sel_flags)) {
                            ImGuiIO& io = ImGui::GetIO();

                            // Check for double-click to edit
                            if (ImGui::IsMouseDoubleClicked(0)) {
                                BeginCellEdit(active_tab, cell_row, cell_col);
                            } else if (io.KeyCtrl) {
                                // Ctrl+Click: Add new selection range (single cell)
                                SelectionRange new_sel;
                                new_sel.row_start = new_sel.row_end = cell_row;
                                new_sel.col_start = new_sel.col_end = cell_col;
                                active_tab->selections.push_back(new_sel);
                            } else if (io.KeyShift && active_tab->selected_row >= 0 && active_tab->selected_col >= 0) {
                                // Shift+Click: Extend selection from last selected cell
                                SelectionRange new_sel;
                                new_sel.row_start = std::min(active_tab->selected_row, cell_row);
                                new_sel.row_end = std::max(active_tab->selected_row, cell_row);
                                new_sel.col_start = std::min(active_tab->selected_col, cell_col);
                                new_sel.col_end = std::max(active_tab->selected_col, cell_col);
                                active_tab->selections.push_back(new_sel);
                            } else {
                                // Normal click: Clear multi-selection, set single selection
                                active_tab->selections.clear();
                                active_tab->selected_row = cell_row;
                                active_tab->selected_col = cell_col;
                                active_tab->selected_column = cell_col;  // Update stats sidebar
                            }
                        }

                        // Cell context menu (right-click on cell) - using BeginPopupContextItem
                        if (ImGui::BeginPopupContextItem()) {
                            RenderCellContextMenu(active_tab, static_cast<int>(actual_row), static_cast<int>(c));
                            ImGui::EndPopup();
                        }
                    }

                    ImGui::PopID();

                    if (has_filter_match) {
                        ImGui::PopStyleColor();
                    }
                    if (in_multi_selection && !is_selected) {
                        ImGui::PopStyleColor();
                    }
                }
            }
        }

        ImGui::EndTable();
    }

    // Pagination controls
    if (total_pages > 1) {
        ImGui::Dummy(ImVec2(0.0f, 2.0f));
        ImGui::AlignTextToFramePadding();
        ImGui::Text("Page:");
        ImGui::SameLine();

        if (ui::GhostButton(ICON_FA_ANGLES_LEFT "##First")) {
            active_tab->current_page = 0;
        }
        ImGui::SameLine();

        if (ui::GhostButton(ICON_FA_CHEVRON_LEFT "##Prev")) {
            if (active_tab->current_page > 0) active_tab->current_page--;
        }
        ImGui::SameLine();

        ImGui::Text("%d / %zu", active_tab->current_page + 1, total_pages);
        ImGui::SameLine();

        if (ui::GhostButton(ICON_FA_CHEVRON_RIGHT "##Next")) {
            if (active_tab->current_page < static_cast<int>(total_pages) - 1) {
                active_tab->current_page++;
            }
        }
        ImGui::SameLine();

        if (ui::GhostButton(ICON_FA_ANGLES_RIGHT "##Last")) {
            active_tab->current_page = static_cast<int>(total_pages) - 1;
        }

        ImGui::SameLine();
        if (active_tab->filter_mode_hide && !active_tab->filtered_indices.empty()) {
            ImGui::Text("(rows %zu - %zu of %zu | %zu filtered)",
                start_row + 1, end_row, display_count, row_count - display_count);
        } else {
            ImGui::Text("(rows %zu - %zu of %zu)", start_row + 1, end_row, display_count);
        }
    }
}

void TableViewerPanel::RenderStatusBar() {
    TableTab* tab = GetActiveTab();
    if (!tab) {
        ImGui::TextDisabled("No table loaded");
        return;
    }

    if (!tab->HasData()) {
        ImGui::TextDisabled("Failed to load table");
        return;
    }

    // Left: Table info
    ImGui::Text(ICON_FA_TABLE " %zu rows x %zu cols",
        tab->GetRowCount(), tab->GetColumnCount());

    // Middle: Selection info
    ImGui::SameLine(0.0f, 18.0f);
    if (tab->selected_row >= 0 && tab->selected_col >= 0) {
        ImGui::Text("Cell: Row %d, Col %d", tab->selected_row + 1, tab->selected_col + 1);
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

}  // namespace cyxwiz








