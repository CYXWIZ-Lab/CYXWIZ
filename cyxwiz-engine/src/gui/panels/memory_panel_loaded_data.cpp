// Memory panel: the Loaded data tab (TOFIX160 P1, approved mockup
// D:/bench157/loaded_data_mockup.html). Every dataset the Engine holds, its
// memory (buffers shared by several names counted once, and marked), disk and
// the file it reads, with the Arrow pool now and at its peak. The account
// (DataRegistry::LoadedData) is rebuilt only when the dataset catalog reports
// a change; the pool is read each frame (two counters).

#include "memory_panel.h"

#include "../../core/data_registry.h"
#include "../icons.h"
#include "../ui_fonts.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <arrow/memory_pool.h>
#include <imgui.h>

#include <algorithm>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {

std::string Thousands(int64_t n) {
    std::string digits = std::to_string(n < 0 ? -n : n);
    for (int i = static_cast<int>(digits.size()) - 3; i > 0; i -= 3) digits.insert(static_cast<size_t>(i), ",");
    return n < 0 ? "-" + digits : digits;
}

}  // namespace

void MemoryPanel::RenderLoadedData() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (loaded_stale_) {
        loaded_ = DataRegistry::Instance().LoadedData();
        loaded_stale_ = false;
    }
    if (loaded_.entries.empty()) {
        ui::EmptyState(ICON_FA_DATABASE, "No dataset is loaded", "Apply a Data Input, or run a pipeline, to load one.");
        return;
    }

    // The summary: count, memory (each byte once), disk, what Arrow holds.
    arrow::MemoryPool* pool = arrow::default_memory_pool();
    const size_t count = loaded_.entries.size();
    const std::string figures[4][2] = {
        {std::to_string(count), count == 1 ? "dataset" : "datasets"},
        {FormatBytes(loaded_.memory_bytes), "in memory"},
        {FormatBytes(loaded_.disk_bytes), "on disk"},
        {FormatBytes(static_cast<size_t>(pool->bytes_allocated())), "Arrow holds now"},
    };
    // Its own line: a wrap broke the peak inside the number.
    const std::string peak = "peak " + FormatBytes(static_cast<size_t>(pool->max_memory()));
    ImGui::Spacing();
    if (ImGui::BeginTable("##loaded_summary", 4, ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableNextRow();
        for (int i = 0; i < 4; ++i) {
            ImGui::TableSetColumnIndex(i);
            {
                ui::FontScope medium(ui::Font::Medium);
                ImGui::TextColored(t.text_bright, "%s", figures[i][0].c_str());
            }
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.text_dim, "%s", figures[i][1].c_str());
            ImGui::PopTextWrapPos();
            if (i == 3) ImGui::TextColored(t.text_faint, "%s", peak.c_str());
        }
        ImGui::EndTable();
    }

    const bool any_shared = std::any_of(loaded_.entries.begin(), loaded_.entries.end(),
                                        [](const LoadedDataEntry& e) { return e.shared_bytes > 0; });
    const bool any_files = std::any_of(loaded_.entries.begin(), loaded_.entries.end(), [](const LoadedDataEntry& e) {
        return e.kind == LoadedDataKind::TableOnDisk || e.kind == LoadedDataKind::Images ||
               e.kind == LoadedDataKind::Audio || e.kind == LoadedDataKind::Text;
    });
    ImGui::PushTextWrapPos(0.0f);
    if (any_shared)
        ImGui::TextColored(t.text_dim,
                           "Shared: these names point at the same data, so it is held once. A row with no shared part "
                           "that should be a share is a copy to look at.");
    if (any_files)
        ImGui::TextColored(t.text_dim,
                           "A table on disk is read a row group at a time; it holds no memory of its own. Images, audio "
                           "and text read their files as training goes.");
    ImGui::PopTextWrapPos();
    ImGui::Spacing();

    // The datasets, the largest first.
    std::vector<const LoadedDataEntry*> rows;
    for (const auto& e : loaded_.entries) rows.push_back(&e);
    std::stable_sort(rows.begin(), rows.end(), [](const LoadedDataEntry* a, const LoadedDataEntry* b) {
        return a->memory_bytes + a->disk_bytes > b->memory_bytes + b->disk_bytes;
    });
    const LoadedDataEntry* picked = nullptr;
    for (const auto* e : rows)
        if (e->name == loaded_pick_) picked = e;
    if (!picked) {
        picked = rows.front();
        loaded_pick_ = picked->name;
    }

    const float details_height = ImGui::GetTextLineHeightWithSpacing() * 4.0f + ImGui::GetStyle().WindowPadding.y * 2.0f;
    const float legend_height = ImGui::GetTextLineHeightWithSpacing() * 2.0f;
    const float table_height = std::max(ImGui::GetTextLineHeightWithSpacing() * 4.0f,
                                        ImGui::GetContentRegionAvail().y - details_height - legend_height);
    const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY |
                                  ImGuiTableFlags_SizingFixedFit;
    if (ImGui::BeginTable("##loaded_rows", 7, flags, ImVec2(0.0f, table_height))) {
        ImGui::TableSetupScrollFreeze(1, 1);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch, 2.0f);
        ImGui::TableSetupColumn("Kind");
        ImGui::TableSetupColumn("Rows");
        ImGui::TableSetupColumn("In memory");
        ImGui::TableSetupColumn("Shared");
        ImGui::TableSetupColumn("On disk");
        ImGui::TableSetupColumn("Reads", ImGuiTableColumnFlags_WidthStretch, 3.0f);
        ImGui::TableHeadersRow();
        const auto right = [](const std::string& text, const ImVec4& colour) {
            const float w = ImGui::CalcTextSize(text.c_str()).x;
            const float avail = ImGui::GetContentRegionAvail().x;
            if (avail > w) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + avail - w);
            ImGui::TextColored(colour, "%s", text.c_str());
        };
        for (const auto* e : rows) {
            ImGui::PushID(e->name.c_str());
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            if (ImGui::Selectable(e->name.c_str(), e == picked,
                                  ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap))
                loaded_pick_ = e->name;
            if (e->shared_bytes > 0) {
                ImGui::SameLine();
                ui::Chip("shared", t.warning);
            }
            ImGui::TableSetColumnIndex(1);
            ImGui::TextColored(t.text_dim, "%s", LoadedDataKindName(e->kind));
            ImGui::TableSetColumnIndex(2);
            right(e->rows < 0 ? "-" : Thousands(e->rows), t.text);
            ImGui::TableSetColumnIndex(3);
            right(FormatBytes(e->memory_bytes), t.text);
            ImGui::TableSetColumnIndex(4);
            right(e->shared_bytes ? FormatBytes(e->shared_bytes) : "-", e->shared_bytes ? t.warning : t.text_faint);
            ImGui::TableSetColumnIndex(5);
            right(e->disk_bytes ? FormatBytes(e->disk_bytes) : "-", e->disk_bytes ? t.text : t.text_faint);
            ImGui::TableSetColumnIndex(6);
            ImGui::TextColored(t.text_faint, "%s", e->path.empty() ? "-" : e->path.c_str());
            if (!e->path.empty() && ImGui::IsItemHovered()) ui::Tooltip(e->path.c_str());
            ImGui::PopID();
        }
        ImGui::EndTable();
    }

    // The picked dataset, two facts a line.
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_raised);
    if (ImGui::BeginChild("##loaded_details", ImVec2(0.0f, 0.0f),
                          ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding)) {
        {
            ui::FontScope medium(ui::Font::Medium);
            ImGui::TextColored(t.text_bright, "%s", picked->name.c_str());
        }
        const std::string facts[6][2] = {
            {"Kind", LoadedDataKindName(picked->kind)},
            {"Rows", picked->rows < 0 ? "-" : Thousands(picked->rows)},
            {"In memory", FormatBytes(picked->memory_bytes)},
            {"Shared with others", picked->shared_bytes ? FormatBytes(picked->shared_bytes) : "none"},
            {"On disk", picked->disk_bytes ? FormatBytes(picked->disk_bytes) : "-"},
            {"Reads", picked->path.empty() ? "-" : picked->path},
        };
        if (ImGui::BeginTable("##loaded_facts", 4, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("k1", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            ImGui::TableSetupColumn("v1", ImGuiTableColumnFlags_WidthStretch, 2.0f);
            ImGui::TableSetupColumn("k2", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            ImGui::TableSetupColumn("v2", ImGuiTableColumnFlags_WidthStretch, 2.0f);
            for (int i = 0; i < 6; i += 2) {
                ImGui::TableNextRow();
                for (int j = 0; j < 2; ++j) {
                    ImGui::TableSetColumnIndex(j * 2);
                    ImGui::TextColored(t.text_dim, "%s", facts[i + j][0].c_str());
                    ImGui::TableSetColumnIndex(j * 2 + 1);
                    ImGui::PushTextWrapPos(0.0f);
                    ImGui::TextColored(t.text, "%s", facts[i + j][1].c_str());
                    ImGui::PopTextWrapPos();
                }
            }
            ImGui::EndTable();
        }
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint,
                       "Updates when a dataset is added, replaced or removed (when the Console logs it). Click a row "
                       "for its details.");
    ImGui::PopTextWrapPos();
}

}  // namespace cyxwiz
