// TableGrid (TOFIX148): the Table Viewer's grid, embeddable. Drawing,
// column kinds, sorting, filtering, colour maps and cell editing; the menus,
// copy and number formats are in table_grid_menus.cpp.

#include "table_grid.h"

#include "../core/async_task_manager.h"
#include "icons.h"
#include "ui_buttons.h"
#include "ui_tokens.h"

#include <imgui_internal.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
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

// Row work up to this many cells runs in the frame; more runs in a task.
constexpr size_t kFrameCells = 200000;
}  // namespace

TableGrid::~TableGrid() {
    if (order_task_) AsyncTaskManager::Instance().Cancel(order_task_);
}

std::string TableGrid::CellText(size_t row, size_t col) const {
    return TableCellText(*table, row, col);
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

void TableGrid::SetTable(std::shared_ptr<DataTable> t, std::vector<TableColumnKind> kinds) {
    CancelWork();
    ++order_generation_;  // a result still on its way is for the old table
    holds_ = std::make_shared<std::atomic<int>>(0);
    table = std::move(t);
    current_page = 0;
    column_page = 0;
    drawn_column_page_ = -1;
    filter_text.clear();
    std::memset(filter_buffer, 0, sizeof(filter_buffer));
    filter_mode_hide = false;
    filter_case_sensitive = false;
    filter_whole_cell = false;
    filter_ = {};
    filtered_indices.clear();
    showing_filtered_ = false;
    match_cells = -1;
    match_rows = 0;
    column_kinds = table && kinds.size() == table->GetColumnCount() ? std::move(kinds) : std::vector<TableColumnKind>{};
    initial_widths_.clear();
    sort_column = -1;
    sort_ascending = true;
    applied_sort_column_ = -1;
    applied_sort_ascending_ = true;
    sync_sort_header_ = false;
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
    return showing_filtered_ ? filtered_indices : sorted_indices;
}

void TableGrid::Render(const char* id, const ImVec2& size) {
    if (!HasData()) return;

    size_t row_count = table->GetRowCount();
    size_t col_count = table->GetColumnCount();

    if (row_count == 0 || col_count == 0) {
        ImGui::Text("Table is empty");
        return;
    }

    RenderWorkLine();

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

    // The data columns in grid order: all of them, or (wider than the grid
    // holds) the frozen ones and one block of the others.
    const int line = options.line_numbers ? 1 : 0;
    const size_t capacity = static_cast<size_t>(IMGUI_TABLE_MAX_COLUMNS - line);
    std::vector<size_t> grid_columns;
    size_t column_pages = 1, block_first = 0, block_end = col_count;
    if (col_count <= capacity) {
        grid_columns.resize(col_count);
        std::iota(grid_columns.begin(), grid_columns.end(), size_t(0));
    } else {
        const size_t frozen = std::min(static_cast<size_t>(std::max(0, frozen_columns)), capacity - 1);
        const size_t block = std::min(static_cast<size_t>(kColumnBlock), capacity - frozen);
        column_pages = (col_count - frozen + block - 1) / block;
        column_page = std::clamp(column_page, 0, static_cast<int>(column_pages) - 1);
        block_first = frozen + static_cast<size_t>(column_page) * block;
        block_end = std::min(block_first + block, col_count);
        for (size_t c = 0; c < frozen; ++c) grid_columns.push_back(c);
        for (size_t c = block_first; c < block_end; ++c) grid_columns.push_back(c);
    }
    const int column_count = static_cast<int>(grid_columns.size()) + line;

    // Past one page the pager (or the no-match note) goes under the grid,
    // and past one block of columns the column pager: keep their lines free
    // when the grid fills the space left.
    ImVec2 outer = size;
    const int note_lines = (total_pages > 1 || (showing_filtered_ && display_count == 0) ? 1 : 0) + (column_pages > 1 ? 1 : 0);
    if (note_lines > 0 && outer.y == 0.0f)
        outer.y = -(ImGui::GetFrameHeightWithSpacing() * static_cast<float>(note_lines) + 4.0f);

    // A block first drawn (a new ImGui table) reports sort specs of its own;
    // the rows keep the sort they have.
    const bool new_block = drawn_column_page_ != column_page;
    drawn_column_page_ = column_page;
    ImGui::PushID(column_page);  // each block keeps its own column widths and order
    if (ImGui::BeginTable(id, column_count, flags, outer)) {
        // The header row, the line numbers and the frozen data columns stay
        // in view while scrolling.
        const int freeze_cols = frozen_columns + (options.line_numbers ? 1 : 0);
        ImGui::TableSetupScrollFreeze(freeze_cols, 1);

        // Setup columns with type indicators and sort arrows
        if (options.line_numbers) {
            ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed |
                                             (sort_column < 0 ? ImGuiTableColumnFlags_DefaultSort : 0), 50.0f);
        }

        const auto& headers = table->GetHeaders();
        for (const size_t i : grid_columns) {
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

            // The sorted column shows its arrow in every block it is in.
            ImGuiTableColumnFlags column_flags = ImGuiTableColumnFlags_None;
            if (static_cast<int>(i) == sort_column)
                column_flags |= ImGuiTableColumnFlags_DefaultSort | (sort_ascending ? ImGuiTableColumnFlags_PreferSortAscending
                                                                                    : ImGuiTableColumnFlags_PreferSortDescending);
            ImGui::TableSetupColumn(header.c_str(), column_flags, initial_widths_[i]);
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
        for (size_t k = 0; k < grid_columns.size(); k++) {
            const size_t i = grid_columns[k];
            ImGui::TableSetColumnIndex(static_cast<int>(k) + line);

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

        // The header's arrow follows a sort from the column menu or a
        // cancelled one (the rows keep their order).
        if (sync_sort_header_) {
            sync_sort_header_ = false;
            int target = -1;
            if (sort_column >= 0) {
                const auto it = std::find(grid_columns.begin(), grid_columns.end(), static_cast<size_t>(sort_column));
                if (it != grid_columns.end()) target = static_cast<int>(it - grid_columns.begin()) + line;
            } else if (!sort_ascending && line) {
                target = 0;
            }
            if (target >= 0)
                ImGui::TableSetColumnSortDirection(
                    target, sort_ascending ? ImGuiSortDirection_Ascending : ImGuiSortDirection_Descending, false);
            else
                ImGui::TableSetColumnSortDirection(0, ImGuiSortDirection_None, false);
            if (ImGuiTableSortSpecs* specs = ImGui::TableGetSortSpecs()) specs->SpecsDirty = false;
        }

        // Handle ImGui's built-in sorting: a header click cycles ascending,
        // descending, unsorted (the table's own order; also how it opens).
        // The line number column sorts by the table's order, either way.
        if (ImGuiTableSortSpecs* sort_specs = ImGui::TableGetSortSpecs()) {
            if (sort_specs->SpecsDirty && new_block) {
                sort_specs->SpecsDirty = false;
            } else if (sort_specs->SpecsDirty && sort_specs->SpecsCount == 0) {
                SetOrder(-1, true);
                sort_specs->SpecsDirty = false;
            } else if (sort_specs->SpecsDirty) {
                const ImGuiTableColumnSortSpecs& spec = sort_specs->Specs[0];
                int sort_col = spec.ColumnIndex - line;  // the line number column is -1
                if (sort_col >= 0) sort_col = static_cast<int>(grid_columns[static_cast<size_t>(sort_col)]);
                SetOrder(sort_col, spec.SortDirection == ImGuiSortDirection_Ascending);
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

                // Data columns; one scrolled out of view draws nothing.
                for (const size_t c : grid_columns) {
                    if (!ImGui::TableSetColumnIndex(col_idx++)) continue;

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
    ImGui::PopID();
    if (showing_filtered_ && display_count == 0)
        ImGui::TextColored(ui::CurrentTokens().text_dim, "No row contains \"%s\".", filter_text.c_str());

    // The column pager: which block of a wide table is shown.
    if (column_pages > 1) {
        ImGui::Dummy(ImVec2(0.0f, 2.0f));
        ImGui::AlignTextToFramePadding();
        ImGui::Text("Columns:");
        ImGui::SameLine();
        if (ui::GhostButton(ICON_FA_CHEVRON_LEFT "##PrevColumns") && column_page > 0) column_page--;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("The previous columns");
        ImGui::SameLine();
        ImGui::Text("%zu - %zu of %zu", block_first + 1, block_end, col_count);
        ImGui::SameLine();
        if (ui::GhostButton(ICON_FA_CHEVRON_RIGHT "##NextColumns") &&
            column_page < static_cast<int>(column_pages) - 1)
            column_page++;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("The next columns");
        if (frozen_columns > 0) {
            ImGui::SameLine();
            ImGui::TextDisabled("(the frozen columns stay)");
        }
    }

    // Pagination controls
    if (total_pages > 1) {
        if (column_pages <= 1) ImGui::Dummy(ImVec2(0.0f, 2.0f));
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
        if (showing_filtered_) {
            ImGui::Text("(rows %zu - %zu of %zu | %zu filtered)",
                start_row + 1, end_row, display_count, row_count - display_count);
        } else {
            ImGui::Text("(rows %zu - %zu of %zu)", start_row + 1, end_row, display_count);
        }
    }
}

void TableGrid::ComputeColumnKinds() {
    if (HasData()) column_kinds = TableColumnKindsOf(*table);
}

// ============================================================================
// Sorting and filtering
// ============================================================================

void TableGrid::SetOrder(int column, bool ascending) {
    if (!table) return;
    sort_column = column;
    sort_ascending = ascending;
    Requery(true);
}

void TableGrid::Requery(bool reorder) {
    if (!table) return;
    // A sort still running is redone with the new request, not dropped.
    reorder = reorder || (order_task_ && order_task_reorders_);
    CancelWork();
    const uint64_t generation = ++order_generation_;

    TableRowRequest request;
    request.sort_column = sort_column;
    request.ascending = sort_ascending;
    request.numeric = sort_column >= 0 && sort_column < static_cast<int>(column_kinds.size()) &&
                      column_kinds[static_cast<size_t>(sort_column)].type == "Numeric";
    if (!reorder) request.order = sorted_indices;
    request.filter = filter_;
    request.filter_rows = Hiding();
    request.count_matches = count_matches && !filter_.empty();

    const size_t rows = table->GetRowCount();
    const bool filtering = !filter_.empty() && (request.filter_rows || request.count_matches);
    const size_t cells = (reorder && sort_column >= 0 ? rows : 0) + (filtering ? rows * table->GetColumnCount() : 0);
    const int column = sort_column;
    const bool ascending = sort_ascending;
    const bool filter_rows = request.filter_rows;
    const bool counted = request.count_matches;
    if (cells <= kFrameCells) {
        ApplyRows(ComputeTableRowOrder(*table, request), filter_rows, counted, column, ascending);
        return;
    }

    // A large table: a task reads it while the grid keeps its rows.
    const auto& headers = table->GetHeaders();
    const std::string name = table->GetName().empty() ? "the table" : table->GetName();
    if (reorder && column >= 0)
        order_task_label_ = "Sorting by " + (static_cast<size_t>(column) < headers.size() ? headers[column] : "a column");
    else if (reorder)
        order_task_label_ = "Putting the rows in table order";
    else if (counted)
        order_task_label_ = "Finding \"" + filter_text + "\"";
    else
        order_task_label_ = "Filtering for \"" + filter_text + "\"";
    order_task_reorders_ = reorder;
    if (match_cells >= 0 && counted) match_cells = -1;  // counting again

    auto result = std::make_shared<TableRowOrder>();
    std::weak_ptr<int> alive = alive_;
    order_task_ = AsyncTaskManager::Instance().RunAsync(
        order_task_label_ + " in " + name,
        [source = table, request = std::move(request), result](LambdaTask& task) {
            *result = ComputeTableRowOrder(*source, request, [&task](float done) {
                task.ReportProgress(done, "");
                return !task.ShouldStop();
            });
            if (result->stopped || task.ShouldStop()) {
                task.MarkCancelled();
                return;
            }
            task.MarkCompleted(std::to_string((request.filter_rows ? result->filtered : result->sorted).size()) +
                               " rows");
        },
        nullptr,
        [this, alive, result, generation, filter_rows, counted, column, ascending](bool success, const std::string&) {
            if (alive.expired() || generation != order_generation_) return;
            order_task_ = 0;
            if (!success) {
                // Stopped: the rows stay as they are, and so does the header.
                sort_column = applied_sort_column_;
                sort_ascending = applied_sort_ascending_;
                sync_sort_header_ = true;
                return;
            }
            ApplyRows(std::move(*result), filter_rows, counted, column, ascending);
        },
        alive_);
}

void TableGrid::ApplyRows(TableRowOrder rows, bool filter_rows, bool counted, int column, bool ascending) {
    sorted_indices = std::move(rows.sorted);
    filtered_indices = std::move(rows.filtered);
    showing_filtered_ = filter_rows;
    match_cells = counted ? rows.match_cells : -1;
    match_rows = counted ? rows.match_rows : 0;
    applied_sort_column_ = column;
    applied_sort_ascending_ = ascending;
}

void TableGrid::CancelWork() {
    if (!order_task_) return;
    AsyncTaskManager::Instance().Cancel(order_task_);
    order_task_ = 0;
    ++order_generation_;  // its result is not applied
    if (sort_column != applied_sort_column_ || sort_ascending != applied_sort_ascending_) {
        sort_column = applied_sort_column_;
        sort_ascending = applied_sort_ascending_;
        sync_sort_header_ = true;
    }
}

std::shared_ptr<void> TableGrid::HoldTable() {
    if (IsEditing()) EndCellEdit(true);
    holds_->fetch_add(1);
    auto holds = holds_;
    return std::shared_ptr<void>(nullptr, [holds](void*) { holds->fetch_sub(1); });
}

void TableGrid::RenderWorkLine() {
    if (!order_task_) return;
    const ui::Tokens& t = ui::CurrentTokens();
    const auto task = AsyncTaskManager::Instance().GetTask(order_task_);
    const float done = task ? task->GetInfo().progress : 0.0f;
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.accent, ICON_FA_SPINNER);
    ImGui::SameLine();
    ImGui::TextColored(t.text, "%s", order_task_label_.c_str());
    ImGui::SameLine();
    ImGui::TextColored(t.text_dim, "%d%%", static_cast<int>(done * 100.0f + 0.5f));
    ImGui::SameLine();
    if (ui::LinkButton("Cancel##order")) CancelWork();
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

void TableGrid::SetFilter(const std::string& text) {
    filter_text = text;
    std::memset(filter_buffer, 0, sizeof(filter_buffer));
    std::strncpy(filter_buffer, text.c_str(), sizeof(filter_buffer) - 1);
    current_page = 0;
    RefreshFilter();
}

void TableGrid::RefreshFilter() {
    filter_ = TableFilter(filter_text, filter_case_sensitive, filter_whole_cell);
    Requery(false);
}

void TableGrid::SetHide(bool hide) {
    filter_mode_hide = hide;
    current_page = 0;
    Requery(false);
}

void TableGrid::ClearFilter() {
    filter_text.clear();
    std::memset(filter_buffer, 0, sizeof(filter_buffer));
    filter_mode_hide = false;
    RefreshFilter();
}

// ============================================================================
// Cell Editing
// ============================================================================

void TableGrid::BeginCellEdit(int row, int col) {
    if (!table || TableHeld()) return;

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

                // The column's kind follows the edit; the owner marks itself dirty.
                if (static_cast<size_t>(editing_col) < column_kinds.size())
                    column_kinds[static_cast<size_t>(editing_col)] = TableColumnKindOf(*table, static_cast<size_t>(editing_col));
                if (Hiding() || count_matches) Requery(false);  // the edited row may match now, or no longer
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
