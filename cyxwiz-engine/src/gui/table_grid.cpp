// TableGrid (TOFIX148): the Table Viewer's grid, embeddable. Drawing,
// column kinds, sorting, filtering, colour maps and cell editing; the menus,
// copy and number formats are in table_grid_menus.cpp.

#include "table_grid.h"

#include "icons.h"
#include "ui_buttons.h"
#include "ui_tokens.h"

#include <imgui_internal.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>

namespace cyxwiz {

namespace {
// One line for the grid: line breaks and tabs become spaces, and a long
// value shows its start (the whole value is in the cell's tooltip).
constexpr size_t kMaxCellChars = 300;
bool OneLine(std::string& text) {
    bool changed = false;
    for (char& ch : text) {
        if (ch == '\n' || ch == '\r' || ch == '\t') {
            ch = ' ';
            changed = true;
        }
    }
    if (text.size() > kMaxCellChars) {
        size_t cut = kMaxCellChars;
        while (cut > 0 && (static_cast<unsigned char>(text[cut]) & 0xC0) == 0x80) --cut;  // not inside a UTF-8 character
        text.resize(cut);
        text += "\xE2\x80\xA6";
        changed = true;
    }
    return changed;
}
}  // namespace

std::string TableGrid::CellText(size_t row, size_t col) const {
    const DataTable::CellValue v = table->GetCell(row, col);
    if (const auto* s = std::get_if<std::string>(&v)) return *s;
    if (const auto* n = std::get_if<int64_t>(&v)) return std::to_string(*n);
    if (const auto* d = std::get_if<double>(&v)) {
        char buf[32];
        const auto r = std::to_chars(buf, buf + sizeof(buf), *d);
        if (r.ptr - buf <= 12) return std::string(buf, r.ptr);
        std::snprintf(buf, sizeof(buf), "%.10g", *d);
        return buf;
    }
    return "";
}

void TableGrid::ComputeInitialWidths() {
    const size_t cols = table->GetColumnCount();
    const size_t rows = std::min<size_t>(table->GetRowCount(), 200);
    initial_widths_.assign(cols, 0.0f);
    for (size_t c = 0; c < cols; ++c) {
        if (c < column_kinds.size() && column_kinds[c].type == "Numeric") continue;
        float widest = 0.0f;
        for (size_t r = 0; r < rows && widest <= kMaxInitialWidth; ++r) {
            std::string text = CellText(r, c);
            OneLine(text);
            widest = std::max(widest, ImGui::CalcTextSize(text.c_str()).x);
        }
        if (widest > kMaxInitialWidth) initial_widths_[c] = kMaxInitialWidth;
    }
}

void TableGrid::SetTable(std::shared_ptr<DataTable> t) {
    table = std::move(t);
    current_page = 0;
    filter_text.clear();
    std::memset(filter_buffer, 0, sizeof(filter_buffer));
    filter_mode_hide = false;
    filter_case_sensitive = false;
    filter_whole_cell = false;
    filtered_indices.clear();
    column_kinds.clear();
    initial_widths_.clear();
    sort_column = -1;
    sort_ascending = true;
    sorted_indices.clear();
    if (table) {
        sorted_indices.resize(table->GetRowCount());
        std::iota(sorted_indices.begin(), sorted_indices.end(), size_t(0));
    }
    selected_column = -1;
    selected_row = -1;
    selected_col = -1;
    selections.clear();
    column_colormaps.clear();
    column_formats.clear();
    frozen_columns = 0;
    editing_row = -1;
    editing_col = -1;
    std::memset(edit_buffer, 0, sizeof(edit_buffer));
    edit_just_started = false;
}

const std::vector<size_t>& TableGrid::DisplayIndices() const {
    // Hiding with no row matching shows no rows, not all of them.
    return Hiding() ? filtered_indices : sorted_indices;
}

void TableGrid::Render(const char* id, const ImVec2& size) {
    if (!HasData()) return;

    size_t row_count = table->GetRowCount();
    size_t col_count = table->GetColumnCount();

    if (row_count == 0 || col_count == 0) {
        ImGui::Text("Table is empty");
        return;
    }

    // Column kinds and opening widths, once per table
    if (column_kinds.empty()) {
        ComputeColumnKinds();
    }
    if (initial_widths_.size() != col_count) ComputeInitialWidths();

    // Initialize sorted indices if empty
    if (sorted_indices.empty()) {
        sorted_indices.resize(row_count);
        std::iota(sorted_indices.begin(), sorted_indices.end(), size_t(0));
    }

    // Determine which indices to display (sorted or filtered)
    const std::vector<size_t>& display_indices = DisplayIndices();

    size_t display_count = display_indices.size();

    // Calculate pagination based on display count
    size_t total_pages = (display_count + options.rows_per_page - 1) / options.rows_per_page;
    if (total_pages == 0) total_pages = 1;
    // A filter or a page size can leave the page past the last one.
    current_page = std::clamp(current_page, 0, static_cast<int>(total_pages) - 1);
    size_t start_row = current_page * options.rows_per_page;
    size_t end_row = std::min(start_row + options.rows_per_page, display_count);

    // ImGui table flags
    ImGuiTableFlags flags = ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg |
                           ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX |
                           ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable |
                           ImGuiTableFlags_Hideable | ImGuiTableFlags_Sortable | ImGuiTableFlags_SortTristate |
                           ImGuiTableFlags_SizingFixedFit;

    int column_count = static_cast<int>(col_count);
    if (options.line_numbers) {
        column_count++;
    }

    if (column_count > IMGUI_TABLE_MAX_COLUMNS) {
        ImGui::TextColored(ui::CurrentTokens().text_dim, "%zu columns: too many to show here (the grid shows up to %d).",
                           col_count, IMGUI_TABLE_MAX_COLUMNS - 1);
        return;
    }

    // Past one page the pager (or the no-match note) goes under the grid:
    // keep its line free when the grid fills the space left.
    ImVec2 outer = size;
    const bool note_line = total_pages > 1 || (Hiding() && display_count == 0);
    if (note_line && outer.y == 0.0f) outer.y = -(ImGui::GetFrameHeightWithSpacing() + 4.0f);

    if (ImGui::BeginTable(id, column_count, flags, outer)) {
        // The header row, the line numbers and the frozen data columns stay
        // in view while scrolling.
        const int freeze_cols = frozen_columns + (options.line_numbers ? 1 : 0);
        ImGui::TableSetupScrollFreeze(freeze_cols, 1);

        // Setup columns with type indicators and sort arrows
        if (options.line_numbers) {
            ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_DefaultSort, 50.0f);
        }

        const auto& headers = table->GetHeaders();
        for (size_t i = 0; i < col_count; i++) {
            std::string header = i < headers.size() ? headers[i] : ("Col" + std::to_string(i));

            // Add type indicator
            if (i < column_kinds.size()) {
                auto& stats = column_kinds[i];
                if (stats.type == "Numeric") {
                    header = ICON_FA_HASHTAG " " + header;
                } else {
                    header = ICON_FA_FONT " " + header;
                }
            }

            ImGui::TableSetupColumn(header.c_str(), ImGuiTableColumnFlags_None, initial_widths_[i]);
        }

        // -----------------------------------------------------------
        // Manual header row with context menu support
        // -----------------------------------------------------------
        // A value from Python shows each column's dtype under its name.
        const bool dtype_row = !captions.empty();
        ImGui::TableNextRow(ImGuiTableRowFlags_Headers, dtype_row ? ImGui::GetTextLineHeight() * 2.0f + 6.0f : 0.0f);

        // Line number column header (if enabled)
        if (options.line_numbers) {
            ImGui::TableSetColumnIndex(0);
            ImGui::TableHeader("#");
        }

        // Data column headers with context menus
        for (size_t i = 0; i < col_count; i++) {
            int table_col = options.line_numbers ? static_cast<int>(i) + 1 : static_cast<int>(i);
            ImGui::TableSetColumnIndex(table_col);

            // Build header text with type indicator
            std::string header = i < headers.size() ? headers[i] : ("Col" + std::to_string(i));
            if (i < column_kinds.size()) {
                auto& stats = column_kinds[i];
                if (stats.type == "Numeric") {
                    header = ICON_FA_HASHTAG " " + header;
                } else {
                    header = ICON_FA_FONT " " + header;
                }
            }

            // Render clickable header
            ImGui::TableHeader(header.c_str());
            if (dtype_row && i < captions.size() && captions[i] != "index") {
                const ImVec2 h0 = ImGui::GetItemRectMin();
                ImGui::GetWindowDrawList()->AddText(ImVec2(h0.x + ImGui::GetStyle().CellPadding.x, h0.y + ImGui::GetTextLineHeight() + 3.0f),
                                                    ui::ToU32(ui::CurrentTokens().text_dim), captions[i].c_str());
            }

            // Right-click context menu on header
            if (ImGui::BeginPopupContextItem(("ColumnContextMenu_" + std::to_string(i)).c_str())) {
                RenderColumnContextMenu(static_cast<int>(i));
                ImGui::EndPopup();
            }
        }

        // Handle ImGui's built-in sorting: a header click cycles ascending,
        // descending, unsorted (the table's own order; also how it opens).
        if (ImGuiTableSortSpecs* sort_specs = ImGui::TableGetSortSpecs()) {
            if (sort_specs->SpecsDirty && sort_specs->SpecsCount == 0) {
                sort_column = -1;
                sorted_indices.resize(row_count);
                std::iota(sorted_indices.begin(), sorted_indices.end(), size_t(0));
                if (Hiding()) ApplyFilter();
                sort_specs->SpecsDirty = false;
            } else if (sort_specs->SpecsDirty) {
                const ImGuiTableColumnSortSpecs& spec = sort_specs->Specs[0];
                int sort_col = spec.ColumnIndex;
                if (options.line_numbers) sort_col--;  // Adjust for line number column

                if (sort_col < 0) {
                    // Sorting by line number column - reset to original order
                    sort_column = -1;
                    sorted_indices.resize(row_count);
                    if (spec.SortDirection == ImGuiSortDirection_Ascending) {
                        std::iota(sorted_indices.begin(), sorted_indices.end(), size_t(0));
                    } else {
                        // Reverse order
                        for (size_t i = 0; i < row_count; i++) {
                            sorted_indices[i] = row_count - 1 - i;
                        }
                    }
                    if (Hiding()) ApplyFilter();
                    spdlog::info("Reset to original order ({})",
                        spec.SortDirection == ImGuiSortDirection_Ascending ? "ascending" : "descending");
                } else {
                    sort_column = sort_col;
                    sort_ascending = (spec.SortDirection == ImGuiSortDirection_Ascending);
                    SortByColumn(sort_col);
                }
                sort_specs->SpecsDirty = false;
            }
        }

        // Auto-select first column if none selected
        if (selected_column < 0 && !column_kinds.empty()) {
            selected_column = 0;
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
                if (options.line_numbers) {
                    ImGui::TableSetColumnIndex(col_idx++);
                    ImGui::TextDisabled("%zu", actual_row + 1);
                }

                // Data columns
                for (size_t c = 0; c < col_count; c++) {
                    ImGui::TableSetColumnIndex(col_idx++);

                    std::string cell_text = CellText(actual_row, c);
                    const bool is_null = std::holds_alternative<std::monostate>(table->GetCell(actual_row, c));

                    // Apply number formatting if configured for this column
                    if (c < column_formats.size() && c < column_kinds.size() &&
                        column_kinds[c].type == "Numeric") {
                        auto& fmt = column_formats[c];
                        // Only format if any formatting option is set
                        if (fmt.decimal_places >= 0 || fmt.thousands_separator ||
                            fmt.as_percentage || !fmt.prefix.empty() || !fmt.suffix.empty()) {
                            auto cell = table->GetCell(actual_row, c);
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
                    bool has_colormap = (c < column_colormaps.size() &&
                                        column_colormaps[c].type != ColorMapType::None);

                    // Data bar or colormap for numeric columns
                    if ((options.data_bars || has_colormap) && c < column_kinds.size()) {
                        auto& stats = column_kinds[c];
                        if (stats.type == "Numeric" && stats.max_val > stats.min_val) {
                            auto cell = table->GetCell(actual_row, c);
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
                                auto& cmap = column_colormaps[c];
                                ImVec4 color = GetColorMapColor(norm, cmap.type);
                                color.w = 0.4f;  // Semi-transparent background
                                ImGui::GetWindowDrawList()->AddRectFilled(
                                    pos, ImVec2(pos.x + cell_width, pos.y + cell_height),
                                    ImGui::GetColorU32(color));
                            } else if (options.data_bars) {
                                // Default data bar
                                float bar_width = cell_width * norm;
                                ImU32 bar_color = ui::ToU32(ui::WithAlpha(ui::CurrentTokens().accent, options.data_bar_alpha));
                                ImGui::GetWindowDrawList()->AddRectFilled(
                                    pos, ImVec2(pos.x + bar_width, pos.y + cell_height),
                                    bar_color);
                            }
                        }
                    }

                    // Check if cell is selected (single selection or multi-selection)
                    int cell_row = static_cast<int>(actual_row);
                    int cell_col = static_cast<int>(c);
                    bool is_selected = (selected_row == cell_row &&
                                       selected_col == cell_col);

                    // Check multi-selection ranges
                    bool in_multi_selection = false;
                    for (const auto& sel : selections) {
                        if (sel.Contains(cell_row, cell_col)) {
                            in_multi_selection = true;
                            break;
                        }
                    }

                    // Apply filter highlighting color
                    bool has_filter_match = Matches(cell_text);

                    // Apply multi-selection background color
                    if (in_multi_selection && !is_selected) {
                        ImGui::PushStyleColor(ImGuiCol_Header, ui::CurrentTokens().selection);
                    }

                    // Push unique ID for each cell to avoid conflicts
                    ImGui::PushID(static_cast<int>(actual_row * col_count + c));

                    // Check if this cell is being edited
                    bool is_editing = (editing_row == cell_row &&
                                      editing_col == cell_col);

                    if (is_editing) {
                        // -----------------------------------------------------------
                        // EDITING MODE: Show InputText
                        // -----------------------------------------------------------
                        ImGui::SetNextItemWidth(-1);  // Fill available width

                        // Focus on first frame of editing
                        if (edit_just_started) {
                            ImGui::SetKeyboardFocusHere();
                            edit_just_started = false;
                        }

                        ImGuiInputTextFlags input_flags = ImGuiInputTextFlags_EnterReturnsTrue |
                                                          ImGuiInputTextFlags_AutoSelectAll;

                        if (ImGui::InputText("##CellEdit", edit_buffer,
                                            sizeof(edit_buffer), input_flags)) {
                            // Enter pressed - save and end editing
                            EndCellEdit(true);
                        }

                        // Handle Escape to cancel
                        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                            EndCellEdit(false);
                        }

                        // Handle click outside to save
                        if (!ImGui::IsItemActive() && ImGui::IsMouseClicked(0)) {
                            EndCellEdit(true);
                        }
                    } else {
                        // -----------------------------------------------------------
                        // NORMAL MODE: Selectable cell
                        // -----------------------------------------------------------
                        ImGuiSelectableFlags sel_flags = ImGuiSelectableFlags_AllowDoubleClick;

                        const std::string full = cell_text;
                        const bool shortened = OneLine(cell_text);
                        const ImVec2 text_pos = ImGui::GetCursorScreenPos();
                        const char* shown = is_null ? "null" : cell_text.c_str();
                        // Laid out as wide as its text (the column fits it), hit and highlight across the cell.
                        const bool clicked = ImGui::Selectable("##cell", is_selected || in_multi_selection,
                                                               sel_flags | ImGuiSelectableFlags_SpanAvailWidth,
                                                               ImVec2(ImGui::CalcTextSize(shown).x, 0.0f));
                        {
                            const ui::Tokens& tk = ui::CurrentTokens();
                            const ImVec4 colour = is_null ? tk.text_faint : has_filter_match ? tk.warning : tk.text;
                            ImGui::GetWindowDrawList()->AddText(text_pos, ui::ToU32(colour), shown);
                        }
                        // The whole value on hover when the cell shows only part of it.
                        const bool clipped = shortened || ImGui::CalcTextSize(shown).x > ImGui::GetItemRectSize().x;
                        if (clipped && !is_null && ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
                            ImGui::SetNextWindowSizeConstraints(ImVec2(0.0f, 0.0f), ImVec2(640.0f, 420.0f));
                            ImGui::BeginTooltip();
                            ImGui::PushTextWrapPos(620.0f);
                            ImGui::TextUnformatted(full.size() > 4000 ? (full.substr(0, 4000) + "\xE2\x80\xA6").c_str() : full.c_str());
                            ImGui::PopTextWrapPos();
                            ImGui::EndTooltip();
                        }
                        if (clicked) {
                            ImGuiIO& io = ImGui::GetIO();

                            // Check for double-click to edit
                            if (ImGui::IsMouseDoubleClicked(0) && options.editable) {
                                BeginCellEdit(cell_row, cell_col);
                            } else if (io.KeyCtrl) {
                                // Ctrl+Click: Add new selection range (single cell)
                                SelectionRange new_sel;
                                new_sel.row_start = new_sel.row_end = cell_row;
                                new_sel.col_start = new_sel.col_end = cell_col;
                                selections.push_back(new_sel);
                            } else if (io.KeyShift && selected_row >= 0 && selected_col >= 0) {
                                // Shift+Click: Extend selection from last selected cell
                                SelectionRange new_sel;
                                new_sel.row_start = std::min(selected_row, cell_row);
                                new_sel.row_end = std::max(selected_row, cell_row);
                                new_sel.col_start = std::min(selected_col, cell_col);
                                new_sel.col_end = std::max(selected_col, cell_col);
                                selections.push_back(new_sel);
                            } else {
                                // Normal click: Clear multi-selection, set single selection
                                selections.clear();
                                selected_row = cell_row;
                                selected_col = cell_col;
                                selected_column = cell_col;  // Update stats sidebar
                            }
                        }

                        // Cell context menu (right-click on cell) - using BeginPopupContextItem
                        if (ImGui::BeginPopupContextItem()) {
                            RenderCellContextMenu(static_cast<int>(actual_row), static_cast<int>(c));
                            ImGui::EndPopup();
                        }
                    }

                    ImGui::PopID();

                    if (in_multi_selection && !is_selected) {
                        ImGui::PopStyleColor();
                    }
                }
            }
        }

        ImGui::EndTable();
    }
    if (Hiding() && display_count == 0)
        ImGui::TextColored(ui::CurrentTokens().text_dim, "No row contains \"%s\".", filter_text.c_str());

    // Pagination controls
    if (total_pages > 1) {
        ImGui::Dummy(ImVec2(0.0f, 2.0f));
        ImGui::AlignTextToFramePadding();
        ImGui::Text("Page:");
        ImGui::SameLine();

        if (ui::GhostButton(ICON_FA_ANGLES_LEFT "##First")) {
            current_page = 0;
        }
        ImGui::SameLine();

        if (ui::GhostButton(ICON_FA_CHEVRON_LEFT "##Prev")) {
            if (current_page > 0) current_page--;
        }
        ImGui::SameLine();

        ImGui::Text("%d / %zu", current_page + 1, total_pages);
        ImGui::SameLine();

        if (ui::GhostButton(ICON_FA_CHEVRON_RIGHT "##Next")) {
            if (current_page < static_cast<int>(total_pages) - 1) {
                current_page++;
            }
        }
        ImGui::SameLine();

        if (ui::GhostButton(ICON_FA_ANGLES_RIGHT "##Last")) {
            current_page = static_cast<int>(total_pages) - 1;
        }

        ImGui::SameLine();
        if (Hiding()) {
            ImGui::Text("(rows %zu - %zu of %zu | %zu filtered)",
                start_row + 1, end_row, display_count, row_count - display_count);
        } else {
            ImGui::Text("(rows %zu - %zu of %zu)", start_row + 1, end_row, display_count);
        }
    }
}

void TableGrid::ComputeColumnKinds() {
    if (!HasData()) return;
    const size_t cols = table->GetColumnCount();
    const size_t rows = table->GetRowCount();
    column_kinds.assign(cols, TableColumnKind{});
    for (size_t c = 0; c < cols; c++) {
        bool numeric = false;
        double min_v = std::numeric_limits<double>::max();
        double max_v = std::numeric_limits<double>::lowest();
        for (size_t r = 0; r < rows; r++) {
            const auto cell = table->GetCell(r, c);
            double v = 0.0;
            if (const auto* d = std::get_if<double>(&cell)) v = *d;
            else if (const auto* n = std::get_if<int64_t>(&cell)) v = static_cast<double>(*n);
            else continue;
            if (std::isnan(v) || std::isinf(v)) continue;
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
            numeric = true;
        }
        auto& kind = column_kinds[c];
        kind.type = numeric ? "Numeric" : "Text";
        if (numeric) {
            kind.min_val = min_v;
            kind.max_val = max_v;
        }
    }
}

void TableGrid::SortByColumn(int column) {
    if (!HasData() || column < 0) return;

    size_t rows = table->GetRowCount();

    sorted_indices.resize(rows);
    std::iota(sorted_indices.begin(), sorted_indices.end(), size_t(0));

    bool is_numeric = (column < static_cast<int>(column_kinds.size()) &&
                       column_kinds[column].type == "Numeric");
    bool ascending = sort_ascending;

    std::sort(sorted_indices.begin(), sorted_indices.end(),
        [&](size_t a, size_t b) {
            if (is_numeric) {
                auto get_val = [&](size_t r) -> double {
                    auto cell = table->GetCell(r, column);
                    if (std::holds_alternative<double>(cell))
                        return std::get<double>(cell);
                    if (std::holds_alternative<int64_t>(cell))
                        return static_cast<double>(std::get<int64_t>(cell));
                    return ascending ? std::numeric_limits<double>::max()
                                    : std::numeric_limits<double>::lowest();
                };
                double va = get_val(a), vb = get_val(b);
                return ascending ? va < vb : va > vb;
            } else {
                std::string sa = CellText(a, column);
                std::string sb = CellText(b, column);
                return ascending ? sa < sb : sa > sb;
            }
        });

    if (Hiding()) ApplyFilter();  // the shown rows follow the new order
    spdlog::info("Sorted by column {} ({})", column, ascending ? "ascending" : "descending");
}

void TableGrid::ApplyColorMap(int column, ColorMapType type) {
    if (column < 0) return;

    // Ensure vector is sized
    if (column_colormaps.size() <= static_cast<size_t>(column)) {
        column_colormaps.resize(column + 1);
    }

    auto& cmap = column_colormaps[column];
    cmap.type = type;
    cmap.auto_range = true;

    // Get column min/max for normalization
    if (column < static_cast<int>(column_kinds.size())) {
        cmap.min_val = static_cast<float>(column_kinds[column].min_val);
        cmap.max_val = static_cast<float>(column_kinds[column].max_val);
    }

    spdlog::info("Applied colormap {} to column {}", static_cast<int>(type), column);
}

void TableGrid::ClearColorMap(int column) {
    if (column < 0) return;

    if (static_cast<size_t>(column) < column_colormaps.size()) {
        column_colormaps[column].type = ColorMapType::None;
    }
}

ImVec4 TableGrid::GetColorMapColor(double normalized, ColorMapType type) {
    // Clamp to [0, 1]
    float t = std::clamp(static_cast<float>(normalized), 0.0f, 1.0f);

    switch (type) {
        case ColorMapType::Viridis: {
            // Purple → Blue → Green → Yellow
            if (t < 0.25f) {
                float s = t / 0.25f;
                return ImVec4(
                    0.267f + s * (0.192f - 0.267f),
                    0.004f + s * (0.408f - 0.004f),
                    0.329f + s * (0.557f - 0.329f),
                    1.0f);
            } else if (t < 0.50f) {
                float s = (t - 0.25f) / 0.25f;
                return ImVec4(
                    0.192f + s * (0.208f - 0.192f),
                    0.408f + s * (0.718f - 0.408f),
                    0.557f + s * (0.475f - 0.557f),
                    1.0f);
            } else if (t < 0.75f) {
                float s = (t - 0.50f) / 0.25f;
                return ImVec4(
                    0.208f + s * (0.565f - 0.208f),
                    0.718f + s * (0.820f - 0.718f),
                    0.475f + s * (0.251f - 0.475f),
                    1.0f);
            } else {
                float s = (t - 0.75f) / 0.25f;
                return ImVec4(
                    0.565f + s * (0.992f - 0.565f),
                    0.820f + s * (0.906f - 0.820f),
                    0.251f + s * (0.145f - 0.251f),
                    1.0f);
            }
        }

        case ColorMapType::Inferno: {
            // Black → Magenta → Orange → Yellow
            if (t < 0.33f) {
                float s = t / 0.33f;
                return ImVec4(s * 0.737f, s * 0.216f, s * 0.329f, 1.0f);
            } else if (t < 0.66f) {
                float s = (t - 0.33f) / 0.33f;
                return ImVec4(
                    0.737f + s * (0.976f - 0.737f),
                    0.216f + s * (0.557f - 0.216f),
                    0.329f + s * (0.035f - 0.329f),
                    1.0f);
            } else {
                float s = (t - 0.66f) / 0.34f;
                return ImVec4(
                    0.976f + s * (0.988f - 0.976f),
                    0.557f + s * (0.996f - 0.557f),
                    0.035f + s * (0.643f - 0.035f),
                    1.0f);
            }
        }

        case ColorMapType::RdYlGn: {
            // Red → Yellow → Green (diverging)
            if (t < 0.5f) {
                float s = t / 0.5f;
                return ImVec4(
                    0.843f + s * (0.996f - 0.843f),
                    0.188f + s * (0.878f - 0.188f),
                    0.153f + s * (0.545f - 0.153f),
                    1.0f);
            } else {
                float s = (t - 0.5f) / 0.5f;
                return ImVec4(
                    0.996f + s * (0.102f - 0.996f),
                    0.878f + s * (0.596f - 0.878f),
                    0.545f + s * (0.314f - 0.545f),
                    1.0f);
            }
        }

        case ColorMapType::Blues: {
            // Light blue → Dark blue
            return ImVec4(
                0.031f + (1.0f - t) * 0.937f,
                0.188f + (1.0f - t) * 0.757f,
                0.420f + (1.0f - t) * 0.525f,
                1.0f);
        }

        case ColorMapType::Plasma: {
            // Purple → Pink → Orange → Yellow
            if (t < 0.33f) {
                float s = t / 0.33f;
                return ImVec4(
                    0.050f + s * (0.798f - 0.050f),
                    0.030f + s * (0.280f - 0.030f),
                    0.528f + s * (0.469f - 0.528f),
                    1.0f);
            } else if (t < 0.66f) {
                float s = (t - 0.33f) / 0.33f;
                return ImVec4(
                    0.798f + s * (0.973f - 0.798f),
                    0.280f + s * (0.580f - 0.280f),
                    0.469f + s * (0.254f - 0.469f),
                    1.0f);
            } else {
                float s = (t - 0.66f) / 0.34f;
                return ImVec4(
                    0.973f + s * (0.940f - 0.973f),
                    0.580f + s * (0.975f - 0.580f),
                    0.254f + s * (0.131f - 0.254f),
                    1.0f);
            }
        }

        default:
            return ui::SequentialAt(ui::CurrentTokens(), t);
    }
}

// ============================================================================
// Filtering
// ============================================================================

bool TableGrid::Matches(const std::string& text) const {
    if (filter_text.empty()) return false;
    if (filter_case_sensitive)
        return filter_whole_cell ? text == filter_text : text.find(filter_text) != std::string::npos;
    const auto lower = [](std::string s) {
        for (char& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        return s;
    };
    const std::string t = lower(text), f = lower(filter_text);
    return filter_whole_cell ? t == f : t.find(f) != std::string::npos;
}

void TableGrid::SetFilter(const std::string& text) {
    filter_text = text;
    std::memset(filter_buffer, 0, sizeof(filter_buffer));
    std::strncpy(filter_buffer, text.c_str(), sizeof(filter_buffer) - 1);
    current_page = 0;
    if (filter_mode_hide) ApplyFilter();
}

void TableGrid::ApplyFilter() {
    if (!table) return;

    filtered_indices.clear();

    if (filter_text.empty()) {
        // No filter - show all
        filtered_indices = sorted_indices;
        return;
    }

    // Filter matching rows
    for (size_t idx : sorted_indices) {
        bool match = false;
        for (size_t c = 0; c < table->GetColumnCount(); c++) {
            if (Matches(CellText(idx, c))) {
                match = true;
                break;
            }
        }
        if (match) {
            filtered_indices.push_back(idx);
        }
    }

    spdlog::info("Filter applied: {} of {} rows match", filtered_indices.size(), sorted_indices.size());
}

void TableGrid::ClearFilter() {
    filter_text.clear();
    std::memset(filter_buffer, 0, sizeof(filter_buffer));
    filtered_indices.clear();
    filter_mode_hide = false;
}

// ============================================================================
// Cell Editing
// ============================================================================

void TableGrid::BeginCellEdit(int row, int col) {
    if (!table) return;

    // End any current editing
    if (editing_row >= 0 && editing_col >= 0) {
        EndCellEdit(true);  // Save current edit
    }

    // Get current cell value
    std::string current_value = CellText(row, col);

    // Copy to edit buffer
    strncpy(edit_buffer, current_value.c_str(), sizeof(edit_buffer) - 1);
    edit_buffer[sizeof(edit_buffer) - 1] = '\0';

    // Set editing state
    editing_row = row;
    editing_col = col;
    edit_just_started = true;

    spdlog::debug("Started editing cell [{}, {}]: '{}'", row + 1, col + 1, current_value);
}

void TableGrid::EndCellEdit(bool save) {
    if (!IsEditing()) return;

    if (save && table) {
        std::string new_value = edit_buffer;
        std::string old_value = CellText(editing_row, editing_col);

        // Only update if value changed
        if (new_value != old_value) {
            if (table->SetCellFromString(editing_row, editing_col, new_value)) {
                spdlog::info("Cell [{}, {}] changed: '{}' -> '{}'",
                    editing_row + 1, editing_col + 1, old_value, new_value);

                // The column kinds follow the edit; the owner marks itself dirty.
                ComputeColumnKinds();
                if (Hiding()) ApplyFilter();  // the edited row may match now, or no longer
                if (on_edited) on_edited();
            } else {
                spdlog::warn("Failed to update cell [{}, {}]", editing_row + 1, editing_col + 1);
            }
        }
    }

    // Clear editing state
    editing_row = -1;
    editing_col = -1;
    std::memset(edit_buffer, 0, sizeof(edit_buffer));
    edit_just_started = false;
}

}  // namespace cyxwiz
