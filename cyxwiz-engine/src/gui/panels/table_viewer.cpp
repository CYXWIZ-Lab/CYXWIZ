#include "table_viewer.h"
#include "../../core/async_task_manager.h"
#include "../../core/arrow_dataset.h"
#include "../../core/parquet_backed_dataset.h"
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
    // A removed dataset leaves its tabs as frozen copies of the rows they
    // show; they let go of its Arrow buffers so its memory is freed.
    catalog_subscription_ = DatasetCatalog::Instance().Subscribe(alive_, [this](const DatasetChange& change) {
        if (change.kind != DatasetChange::Kind::Removed) return;
        for (auto& tab : tabs_) {
            if (tab->removed || tab->origin != change.name) continue;
            tab->removed = true;
            tab->source.reset();
            tab->dataset.clear();
            spdlog::info("Table Viewer: '{}' was removed; tab '{}' keeps a frozen copy of its {} rows", change.name,
                         tab->filename, tab->GetRowCount());
        }
    });
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
        RenderRemovedHeader(GetActiveTab());
        RenderToolbar();
        RenderOpening();
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
    RenderFindDialog();
}

void TableViewerPanel::RenderRemovedHeader(TableTab* tab) {
    if (!tab || !tab->removed) return;
    const ui::Tokens& t = ui::CurrentTokens();
    const std::string text = "'" + tab->origin + "' was removed (its Data Input or graph is gone). This tab is a frozen "
                             "copy of the " + std::to_string(tab->GetRowCount()) + " rows it shows.";
    // A tinted line, like the live-value line; the text wraps.
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float w = ImGui::GetContentRegionAvail().x;
    const float wrap = w - 36.0f;
    const float h = ImGui::CalcTextSize(text.c_str(), nullptr, false, wrap).y + 10.0f;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), ui::ToU32(ui::Mix(t.bg_window, t.warning, 0.10f)), 4.0f);
    dl->AddCircleFilled(ImVec2(p.x + 14.0f, p.y + 5.0f + ImGui::GetTextLineHeight() * 0.5f), 3.5f, ui::ToU32(t.warning));
    ImGui::SetCursorScreenPos(ImVec2(p.x + 26.0f, p.y + 5.0f));
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + wrap);
    ImGui::TextColored(t.text, "%s", text.c_str());
    ImGui::PopTextWrapPos();
    ImGui::SetCursorScreenPos(ImVec2(p.x, p.y + h + 4.0f));
    ImGui::Dummy(ImVec2(0.0f, 0.0f));
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

    // Each group goes beside the one before it when it fits, else on the next
    // line (a narrow panel cut Find, Save and Export off).
    const ImGuiStyle& style = ImGui::GetStyle();
    const float right = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
    auto beside = [&](float width, float gap) {
        if (ImGui::GetItemRectMax().x + gap + width <= right) ImGui::SameLine(0.0f, gap);
    };
    auto button = [](const char* label) { return ui::ButtonWidth(label, ui::ButtonSize::Small); };
    auto check = [&](const char* label) {
        return ImGui::GetFrameHeight() + style.ItemInnerSpacing.x + ImGui::CalcTextSize(label, nullptr, true).x;
    };
    auto icon = [](const char* text) { return ImGui::CalcTextSize(text).x; };
    const float gap = style.ItemSpacing.x;

    // Another loaded dataset, beside the open tabs.
    if (ui::GhostButton(ICON_FA_DATABASE " Open dataset")) open_dataset_popup_requested_ = true;
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Open the first rows of a dataset a Data Input loaded");
    RenderOpenDatasetPopup();

    // Stats sidebar toggle
    beside(button(ICON_FA_CHART_BAR " Stats"), gap);
    if (ui::GhostButton(ICON_FA_CHART_BAR " Stats", true, nullptr, show_stats_sidebar_)) {
        show_stats_sidebar_ = !show_stats_sidebar_;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Toggle Statistics Sidebar");

    beside(check("Data Bars"), gap);
    ImGui::Checkbox("Data Bars", &show_data_bars_);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Show data bars in numeric columns");

    beside(check("Line #") + gap + 70.0f, gap);
    ImGui::Checkbox("Line #", &show_line_numbers_);

    ImGui::SameLine();
    ImGui::SetNextItemWidth(70);
    if (ImGui::InputInt("##RowsPerPage", &rows_per_page_, 0, 0)) {
        rows_per_page_ = std::clamp(rows_per_page_, 10, 1000);
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Rows per page");

    // Filter
    const bool filtered = !active_tab->grid.filter_text.empty();
    beside(icon(ICON_FA_FILTER) + gap + 150.0f + gap + check("Hide") +
               (filtered ? gap + button(ICON_FA_XMARK "##ClearFilter") : 0.0f),
           18.0f);
    ImGui::AlignTextToFramePadding();
    ImGui::Text(ICON_FA_FILTER);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(150);
    bool filter_changed = ImGui::InputText("##Filter", active_tab->grid.filter_buffer, sizeof(active_tab->grid.filter_buffer),
        ImGuiInputTextFlags_EnterReturnsTrue);
    if (filter_changed) active_tab->grid.SetFilter(active_tab->grid.filter_buffer);

    // Filter mode toggle
    ImGui::SameLine();
    bool hide = active_tab->grid.filter_mode_hide;
    if (ImGui::Checkbox("Hide", &hide)) active_tab->grid.SetHide(hide);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Hide non-matching rows instead of highlighting");

    // Clear filter button
    if (filtered) {
        ImGui::SameLine();
        if (ui::GhostButton(ICON_FA_XMARK "##ClearFilter")) {
            active_tab->grid.ClearFilter();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Clear filter");
    }

    // Column freeze control
    beside(icon(ICON_FA_LOCK) + gap + 50.0f, 18.0f);
    ImGui::AlignTextToFramePadding();
    ImGui::Text(ICON_FA_LOCK);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Frozen columns");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(50);
    if (ImGui::InputInt("##Freeze", &active_tab->grid.frozen_columns, 0, 0)) {
        active_tab->grid.frozen_columns = std::clamp(active_tab->grid.frozen_columns, 0,
            static_cast<int>(active_tab->grid.table ? active_tab->grid.table->GetColumnCount() : 0));
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Number of columns to freeze");

    // Find button
    beside(button(ICON_FA_MAGNIFYING_GLASS " Find"), 18.0f);
    if (ui::GhostButton(ICON_FA_MAGNIFYING_GLASS " Find")) {
        show_find_dialog_ = true;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Find in table");

    // Save button (only enabled if dirty)
    if (active_tab->grid.table) {
        beside(button(ICON_FA_FLOPPY_DISK " Save"), gap);
        if (ui::GhostButton(ICON_FA_FLOPPY_DISK " Save", active_tab->is_dirty && !active_tab->live.on,
                            active_tab->live.on ? "A value from Python: export it instead" : "No unsaved changes"))
            SaveTable(active_tab);
        if (active_tab->is_dirty && ImGui::IsItemHovered()) ImGui::SetTooltip("Save changes (Ctrl+S)");

        // Export button
        beside(button(ICON_FA_FILE_EXPORT " Export " ICON_FA_CHEVRON_DOWN), gap);
        if (ui::GhostButton(ICON_FA_FILE_EXPORT " Export " ICON_FA_CHEVRON_DOWN)) ImGui::OpenPopup("##export");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Write the table to a file, or copy it");
        RenderExportMenu(active_tab);
    }

    // Close tab button
    beside(button(ICON_FA_XMARK "##close_tab"), gap);
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
    if (!status_note_.empty()) {
        ImGui::SameLine(0.0f, 18.0f);
        const ui::Tokens& t = ui::CurrentTokens();
        ImGui::TextColored(status_note_error_ ? t.error : t.text_dim, "%s", status_note_.c_str());
    }
}

void TableViewerPanel::OpenDataset(const std::string& dataset_name, int64_t max_rows) {
    StartOpen(dataset_name, dataset_name, [dataset_name, max_rows](const SliceProgress& progress) {
        auto& registry = DataRegistry::Instance();
        if (auto dataset = registry.GetArrowDataset(dataset_name)) {
            const auto table = dataset->GetArrowTable();
            return DatasetSliceFromArrow(table, dataset_name, max_rows, table ? table->num_rows() : 0, progress);
        }
        if (auto parquet = registry.GetParquetBackedDataset(dataset_name)) {
            // On disk: read the row groups the rows need (each once), then convert.
            DatasetSlice slice;
            const auto rows = FirstRowsOfRowGroups(
                parquet->GetNumRowGroups(), [&parquet](int group) { return parquet->ReadRowGroup(group); }, max_rows,
                [&progress](float done, const std::string& message) { return progress(done * 0.5f, message); },
                slice.error);
            if (!rows) {
                slice.stopped = slice.error.empty();
                return slice;
            }
            return DatasetSliceFromArrow(rows, dataset_name, max_rows, parquet->GetNumRows(),
                                         [&progress](float done, const std::string& message) {
                                             return progress(0.5f + done * 0.5f, message);
                                         });
        }
        DatasetSlice missing;
        missing.error = "'" + dataset_name + "' is not loaded any more; apply its Data Input again.";
        return missing;
    });
}

void TableViewerPanel::OpenArrow(std::shared_ptr<arrow::Table> table, const std::string& name, int64_t max_rows) {
    if (!table) return;
    StartOpen(name, "", [table, name, max_rows](const SliceProgress& progress) {
        return DatasetSliceFromArrow(table, name, max_rows < 0 ? table->num_rows() : max_rows, table->num_rows(),
                                     progress);
    });
}

void TableViewerPanel::StartOpen(const std::string& name, const std::string& dataset,
                                 std::function<DatasetSlice(const SliceProgress&)> read) {
    struct Result {
        DatasetSlice slice;
        std::vector<TableColumnKind> kinds;
    };
    auto result = std::make_shared<Result>();
    std::weak_ptr<int> alive = alive_;
    const uint64_t task = AsyncTaskManager::Instance().RunAsync(
        "Open " + name + " in the Table Viewer",
        [read = std::move(read), result](LambdaTask& task) {
            result->slice = read([&task](float done, const std::string& message) {
                task.ReportProgress(done * 0.9f, message);
                return !task.ShouldStop();
            });
            auto& slice = result->slice;
            if (slice.stopped || task.ShouldStop()) {
                task.MarkCancelled();
                return;
            }
            if (!slice.table) {
                task.MarkFailed(slice.error.empty() ? "The table could not be read." : slice.error);
                return;
            }
            task.ReportProgress(0.95f, "Column types");
            result->kinds = TableColumnKindsOf(*slice.table);
            task.MarkCompleted(std::to_string(slice.table->GetRowCount()) + " rows \xC2\xB7 " +
                               std::to_string(slice.table->GetColumnCount()) + " columns");
        },
        nullptr,
        [this, alive, result, name, dataset](bool success, const std::string& error) {
            if (alive.expired()) return;
            opening_.erase(std::remove_if(opening_.begin(), opening_.end(),
                                          [&result](const Opening& o) { return o.token == result; }),
                           opening_.end());
            auto& slice = result->slice;
            if (!success || !slice.table) {
                status_note_ = slice.stopped || error.empty() ? "Stopped opening " + name
                                                              : "Could not open " + name + ": " + error;
                status_note_error_ = !slice.stopped && !error.empty();
                return;
            }
            const size_t shown = slice.table->GetRowCount();
            if (static_cast<int64_t>(shown) < slice.total_rows)
                slice.table->SetName(name + " \xC2\xB7 first " + std::to_string(shown) + " of " +
                                     std::to_string(slice.total_rows) + " rows");
            status_note_.clear();
            TableTab* tab = AddTab(slice.table, std::move(result->kinds), slice.source);
            if (tab) tab->origin = dataset;
            if (tab && static_cast<int64_t>(shown) == slice.total_rows) tab->dataset = dataset;
            visible_ = true;
        },
        alive_);
    opening_.push_back({task, name, result});
    visible_ = true;
}

void TableViewerPanel::RenderOpening() {
    if (opening_.empty()) return;
    const ui::Tokens& t = ui::CurrentTokens();
    for (const auto& open : opening_) {
        ImGui::PushID(static_cast<int>(open.task));
        const auto task = AsyncTaskManager::Instance().GetTask(open.task);
        const TaskInfo info = task ? task->GetInfo() : TaskInfo{};
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.accent, ICON_FA_SPINNER);
        ImGui::SameLine();
        ImGui::TextColored(t.text, "Opening %s", open.name.c_str());
        ImGui::SameLine();
        const std::string detail = std::to_string(static_cast<int>(info.progress * 100.0f + 0.5f)) + "%" +
                                   (info.status_message.empty() ? "" : " \xC2\xB7 " + info.status_message);
        ImGui::TextColored(t.text_dim, "%s", detail.c_str());
        ImGui::SameLine();
        if (ui::LinkButton("Cancel")) AsyncTaskManager::Instance().Cancel(open.task);
        ImGui::PopID();
    }
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
            // A click picks it (the popup stays for the row count); a double-click opens it.
            if (ImGui::Selectable((entry.Shown() + "##" + entry.name).c_str(), open_dataset_pick_ == entry.name,
                                  ImGuiSelectableFlags_NoAutoClosePopups | ImGuiSelectableFlags_AllowDoubleClick)) {
                open_dataset_pick_ = entry.name;
                if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                    OpenDataset(open_dataset_pick_, open_dataset_rows_);
                    ImGui::CloseCurrentPopup();
                }
            }
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

    ImGui::Spacing();
    const bool can_open = !open_dataset_pick_.empty() &&
                          std::any_of(tables.begin(), tables.end(),
                                      [this](const DatasetEntry& e) { return e.name == open_dataset_pick_; });
    if (ui::PrimaryButton("Open", can_open, "Pick a dataset")) {
        OpenDataset(open_dataset_pick_, open_dataset_rows_);
        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ui::SecondaryButton("Cancel", true, nullptr, ui::ButtonSize::Regular)) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
}

}  // namespace cyxwiz
