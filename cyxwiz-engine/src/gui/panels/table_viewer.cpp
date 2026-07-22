#include "table_viewer.h"
#include "visualization_panel.h"
#include "../icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numeric>
#include <algorithm>
#include <map>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace fs = std::filesystem;

namespace cyxwiz {

TableViewerPanel::TableViewerPanel()
    : Panel("Table Viewer", false)  // Start hidden
{
}

void TableViewerPanel::Render() {
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
    if (shortcut_tab && !shortcut_tab->is_loading) {
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

    ImGui::Begin(GetName(), &visible_);

    // Tab bar at top
    RenderTabBar();

    ImGui::Separator();

    // Toolbar
    RenderToolbar();

    ImGui::Separator();

    // Table display or loading indicator
    TableTab* active_tab = GetActiveTab();
    if (active_tab) {
        if (active_tab->is_loading) {
            RenderLoadingIndicator();
        } else if (active_tab->table) {
            // 3-pane layout: sidebar + splitter + main table
            if (show_stats_sidebar_) {
                RenderStatsSidebar(active_tab);
                ImGui::SameLine();

                // Draggable splitter
                ImGui::Button("##vsplitter", ImVec2(4.0f, -1));
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

    ImGui::Separator();

    // Status bar
    RenderStatusBar();

    ImGui::End();

    // Render modal dialogs (must be outside main window)
    RenderExportDialog();
    RenderFindDialog();
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
            if (tab->is_loading) {
                tab_name = ICON_FA_SPINNER " " + tab_name;
            } else {
                tab_name = ICON_FA_TABLE " " + tab_name;
            }

            // Make tab closable
            bool tab_open = true;
            ImGuiTabItemFlags tab_flags = ImGuiTabItemFlags_None;

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
    if (ImGui::Button(show_stats_sidebar_ ? ICON_FA_CHART_BAR " Stats" : ICON_FA_CHART_BAR)) {
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

    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();

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
        if (ImGui::Button(ICON_FA_XMARK "##ClearFilter")) {
            ClearFilter(active_tab);
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Clear filter");
    }

    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();

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

    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();

    // Find button
    if (ImGui::Button(ICON_FA_MAGNIFYING_GLASS " Find")) {
        show_find_dialog_ = true;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Find in table");

    // Save button (only enabled if dirty)
    if (active_tab->table && !active_tab->is_loading) {
        ImGui::SameLine();
        if (active_tab->is_dirty) {
            if (ImGui::Button(ICON_FA_FLOPPY_DISK " Save")) {
                SaveTable(active_tab);
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Save changes (Ctrl+S)");
        } else {
            ImGui::BeginDisabled();
            ImGui::Button(ICON_FA_FLOPPY_DISK " Save");
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("No unsaved changes");
            }
        }

        // Export button
        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_FILE_EXPORT " Export")) {
            show_export_dialog_ = true;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Export table to file");
    }

    // Close tab button
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_XMARK)) {
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
    ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
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

        // ═══════════════════════════════════════════════════════════
        // Manual header row with context menu support
        // ═══════════════════════════════════════════════════════════
        ImGui::TableNextRow(ImGuiTableRowFlags_Headers);

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
                                ImU32 bar_color = ImGui::GetColorU32(ImVec4(0.2f, 0.5f, 0.8f, data_bar_alpha_));
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
                        ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.3f, 0.5f, 0.7f, 0.4f));
                    }

                    if (has_filter_match) {
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.8f, 0.0f, 1.0f));
                    }

                    // Push unique ID for each cell to avoid conflicts
                    ImGui::PushID(static_cast<int>(actual_row * col_count + c));

                    // Check if this cell is being edited
                    bool is_editing = (active_tab->editing_row == cell_row &&
                                      active_tab->editing_col == cell_col);

                    if (is_editing) {
                        // ═══════════════════════════════════════════════════════════
                        // EDITING MODE: Show InputText
                        // ═══════════════════════════════════════════════════════════
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
                        // ═══════════════════════════════════════════════════════════
                        // NORMAL MODE: Selectable cell
                        // ═══════════════════════════════════════════════════════════
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

    // Render quick plot popup
    if (show_plot_popup_) {
        RenderQuickPlot();
    }

    // Pagination controls
    if (total_pages > 1) {
        ImGui::Separator();
        ImGui::Text("Page:");
        ImGui::SameLine();

        if (ImGui::Button(ICON_FA_ANGLES_LEFT "##First")) {
            active_tab->current_page = 0;
        }
        ImGui::SameLine();

        if (ImGui::Button(ICON_FA_CHEVRON_LEFT "##Prev")) {
            if (active_tab->current_page > 0) active_tab->current_page--;
        }
        ImGui::SameLine();

        ImGui::Text("%d / %zu", active_tab->current_page + 1, total_pages);
        ImGui::SameLine();

        if (ImGui::Button(ICON_FA_CHEVRON_RIGHT "##Next")) {
            if (active_tab->current_page < static_cast<int>(total_pages) - 1) {
                active_tab->current_page++;
            }
        }
        ImGui::SameLine();

        if (ImGui::Button(ICON_FA_ANGLES_RIGHT "##Last")) {
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

    if (tab->is_loading) {
        ImGui::Text(ICON_FA_SPINNER " Loading: %s", tab->filename.c_str());
        return;
    }

    if (!tab->HasData()) {
        ImGui::TextDisabled("Failed to load table");
        return;
    }

    // Left: Table info with lazy loading indicator
    if (tab->use_lazy_loading) {
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), ICON_FA_BOLT);
        ImGui::SameLine(0, 4);
    }
    ImGui::Text(ICON_FA_TABLE " %zu rows x %zu cols",
        tab->GetRowCount(), tab->GetColumnCount());

    // Show cache info for lazy loading
    if (tab->use_lazy_loading && tab->lazy_table) {
        ImGui::SameLine();
        ImGui::TextDisabled("(cached: %zu)", tab->lazy_table->GetCachedRowCount());
    }

    // Middle: Selection info
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    if (tab->selected_row >= 0 && tab->selected_col >= 0) {
        ImGui::Text("Cell: Row %d, Col %d", tab->selected_row + 1, tab->selected_col + 1);
    } else {
        ImGui::TextDisabled("No cell selected");
    }

    // Right: Memory/file size info
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    if (tab->use_lazy_loading && tab->lazy_table) {
        // Show file size for lazy loading
        size_t file_bytes = tab->lazy_table->GetFileSize();
        if (file_bytes > 1024 * 1024) {
            ImGui::Text(ICON_FA_FILE " %.1f MB (lazy)", file_bytes / (1024.0 * 1024.0));
        } else {
            ImGui::Text(ICON_FA_FILE " %.1f KB (lazy)", file_bytes / 1024.0);
        }
    } else {
        // Memory estimate for in-memory table
        size_t mem_bytes = tab->GetRowCount() * tab->GetColumnCount() * sizeof(double);
        if (mem_bytes > 1024 * 1024) {
            ImGui::Text(ICON_FA_MEMORY " %.1f MB", mem_bytes / (1024.0 * 1024.0));
        } else {
            ImGui::Text(ICON_FA_MEMORY " %.1f KB", mem_bytes / 1024.0);
        }
    }
}

void TableViewerPanel::RenderLoadingIndicator() {
    TableTab* active_tab = GetActiveTab();
    if (!active_tab) return;

    ImGui::Spacing();
    ImGui::Spacing();

    // Center the loading indicator
    float window_width = ImGui::GetWindowWidth();
    float text_width = ImGui::CalcTextSize(active_tab->load_status.c_str()).x;
    ImGui::SetCursorPosX((window_width - text_width) * 0.5f);

    // Animated spinner
    float time = static_cast<float>(ImGui::GetTime());
    const char* spinner_chars = "|/-\\";
    char spinner = spinner_chars[static_cast<int>(time * 10) % 4];

    ImGui::Text("%c %s", spinner, active_tab->load_status.c_str());

    ImGui::Spacing();

    // Progress bar
    ImGui::SetCursorPosX(window_width * 0.2f);
    ImGui::ProgressBar(active_tab->load_progress, ImVec2(window_width * 0.6f, 0.0f));
}
}  // namespace cyxwiz








