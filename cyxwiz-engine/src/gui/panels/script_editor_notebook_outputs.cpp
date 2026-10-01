// Script Editor notebook outputs (TOFIX133 P4 step 4.3c, approved board 5):
// text in the code font, stderr on an amber tint, errors as a block with
// frames you can open, tables with "Open in Table Viewer", plots with their
// own small toolbar.

#include "script_editor.h"

#include "output_renderer.h"
#include "../../core/html_table.h"
#include "../../core/notebook_presentation.h"
#include "../../data/data_table.h"
#include "../../scripting/scripting_engine.h"
#include "../editor_fonts.h"
#include "../icons.h"
#include "../markdown_view.h"
#include "../ui_buttons.h"
#include "../ui_fonts.h"
#include "../ui_tokens.h"

#include <imgui.h>
#include <spdlog/spdlog.h>
#include <stb_image.h>

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {
constexpr int kLongOutputLines = 400;  // longer text shows its start and end until "Show all"

int CountLines(const std::string& s) {
    int n = 1;
    for (char c : s) n += c == '\n';
    return n;
}

std::string TrimTrailingNewlines(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

// Monospace text, wrapped at the content width; a very long output shows
// its first and last lines until expanded.
void MonoText(CellOutput& out, const std::string& text, const ImVec4& colour) {
    ImFont* mono = gui::GetCodeFont();
    if (mono) ImGui::PushFont(mono);
    ImGui::PushStyleColor(ImGuiCol_Text, colour);
    const std::string body = TrimTrailingNewlines(text);
    const int lines = CountLines(body);
    if (lines <= kLongOutputLines || out.expanded) {
        ImGui::TextWrapped("%s", body.c_str());
    } else {
        size_t head_end = 0;
        for (int i = 0; i < 200 && head_end != std::string::npos; ++i) head_end = body.find('\n', head_end + 1);
        size_t tail_start = body.size();
        for (int i = 0; i < 50 && tail_start != std::string::npos && tail_start > 0; ++i) tail_start = body.rfind('\n', tail_start - 1);
        ImGui::TextWrapped("%s", body.substr(0, head_end).c_str());
        ImGui::PopStyleColor();
        if (mono) ImGui::PopFont();
        char more[96];
        std::snprintf(more, sizeof(more), "%d lines not shown. Show all##more", lines - 250);
        if (ui::LinkButton(more)) out.expanded = true;
        if (mono) ImGui::PushFont(mono);
        ImGui::PushStyleColor(ImGuiCol_Text, colour);
        ImGui::TextWrapped("%s", body.substr(tail_start == std::string::npos ? 0 : tail_start + 1).c_str());
    }
    ImGui::PopStyleColor();
    if (mono) ImGui::PopFont();
}

// A tinted block behind content drawn after it: split the draw list so the
// rectangle can go under what the block holds.
struct TintBlock {
    ImVec2 min;
    float width;
    ImVec2 pad;
    explicit TintBlock(float w, ImVec2 padding) : min(ImGui::GetCursorScreenPos()), width(w), pad(padding) {
        ImGui::GetWindowDrawList()->ChannelsSplit(2);
        ImGui::GetWindowDrawList()->ChannelsSetCurrent(1);
        ImGui::SetCursorScreenPos(ImVec2(min.x + pad.x, min.y + pad.y));
        ImGui::BeginGroup();
        ImGui::PushTextWrapPos(min.x + width - pad.x);
    }
    void End(const ImVec4& tint, float rounding) {
        ImGui::PopTextWrapPos();
        ImGui::EndGroup();
        const float bottom = ImGui::GetItemRectMax().y + pad.y;
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->ChannelsSetCurrent(0);
        dl->AddRectFilled(min, ImVec2(min.x + width, bottom), ui::ToU32(tint), rounding);
        dl->ChannelsMerge();
        ImGui::SetCursorScreenPos(ImVec2(min.x, bottom));
        ImGui::Dummy(ImVec2(width, 0.0f));
    }
};

std::shared_ptr<DataTable> ToDataTable(const html::Table& t, const std::string& name) {
    auto table = std::make_shared<DataTable>();
    std::vector<std::string> headers = t.columns;
    const size_t width = std::max(headers.size(), t.rows.empty() ? size_t{0} : t.rows.front().size());
    headers.resize(width);
    if (t.has_index && !headers.empty() && headers[0].empty()) headers[0] = "index";
    table->SetHeaders(headers);
    for (const auto& r : t.rows) {
        DataTable::Row row;
        for (size_t i = 0; i < width; ++i) row.emplace_back(i < r.size() ? r[i] : std::string());
        table->AddRow(std::move(row));
    }
    table->SetName(name);
    return table;
}
}  // namespace

void ScriptEditorPanel::RenderNotebookOutputs(Cell& cell, int index, float width) {
    auto& tab = *tabs_[active_tab_index_];
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::PushID("outputs");
    for (size_t i = 0; i < cell.outputs.size(); ++i) {
        CellOutput& out = cell.outputs[i];
        ImGui::PushID(static_cast<int>(i));
        ImGui::Dummy(ImVec2(0.0f, 2.0f));
        switch (out.type) {
            case OutputType::Stream:
                if (out.name == "stderr") {
                    TintBlock block(width, ImVec2(8.0f, 3.0f));
                    MonoText(out, out.data, t.warning);
                    block.End(ui::WithAlpha(t.warning, 0.07f), 4.0f);
                } else {
                    MonoText(out, out.data, t.text);
                }
                break;
            case OutputType::Text:
                if (out.is_result && !out.html.empty()) {
                    if (!out.table_cache) {
                        auto parsed = std::make_shared<html::Table>();
                        out.table_cache = html::ParseTable(out.html, *parsed) ? parsed : std::make_shared<html::Table>();
                    }
                    if (!out.table_cache->columns.empty() || !out.table_cache->rows.empty()) {
                        RenderTableOutput(cell, out, width);
                        break;
                    }
                }
                MonoText(out, out.data, t.text);
                break;
            case OutputType::Error:
                RenderErrorOutput(cell, index, out, width);
                break;
            case OutputType::Image:
            case OutputType::Plot:
                RenderPlotOutput(cell, out, width);
                break;
            case OutputType::Markdown:
                ui::MarkdownView(out.data);
                break;
            case OutputType::Html:
            case OutputType::Table:
                MonoText(out, out.data, t.text);
                break;
        }
        ImGui::PopID();
    }
    ImGui::PopID();
    (void)tab;
}

void ScriptEditorPanel::RenderErrorOutput(Cell& cell, int index, CellOutput& out, float width) {
    const ui::Tokens& t = ui::CurrentTokens();
    (void)cell;
    (void)index;
    TintBlock block(width, ImVec2(14.0f, 10.0f));
    const std::string ename = out.ename.empty() ? "Error" : out.ename;
    std::string evalue = out.evalue;
    if (out.ename.empty()) {
        // An error from text only (Python could not start, an older notebook).
        evalue = TrimTrailingNewlines(out.data);
        const size_t nl = evalue.rfind('\n');
        if (!out.expanded && nl != std::string::npos) evalue = evalue.substr(nl + 1);
    }
    {
        ui::FontScope bold(ui::Font::Bold);
        ImGui::TextColored(ui::Mix(t.error, t.text_bright, 0.25f), "%s", ename.c_str());
    }
    if (!evalue.empty()) {
        ImGui::SameLine(0.0f, 10.0f);
        ImGui::PushStyleColor(ImGuiCol_Text, ui::Mix(t.error, t.text, 0.65f));
        ImGui::TextWrapped("%s", evalue.c_str());
        ImGui::PopStyleColor();
    }

    // Frames: the user's own first; library frames behind "Show full traceback".
    std::vector<std::string> files;
    for (const auto& f : out.frames) files.push_back(f.file);
    std::vector<int> user = nbview::UserFrames(files);
    std::vector<int> shown;
    if (out.expanded) {
        for (int k = 0; k < static_cast<int>(out.frames.size()); ++k) shown.push_back(k);
    } else {
        shown = user;
        // A chained exception keeps its "Caused by" line even when its first
        // frame is library code (the frame itself waits for the full view).
        for (int k = 0; k < static_cast<int>(out.frames.size()); ++k)
            if (!out.frames[k].cause.empty() && std::find(shown.begin(), shown.end(), k) == shown.end()) shown.push_back(k);
        std::sort(shown.begin(), shown.end());
    }
    auto is_user = [&](int k) { return out.expanded || std::find(user.begin(), user.end(), k) != user.end(); };
    if (!shown.empty()) {
        ImGui::Dummy(ImVec2(0.0f, 2.0f));
        ImFont* mono = gui::GetCodeFont();
        const float link_w = std::min(300.0f, width * 0.38f);
        for (int k : shown) {
            const TraceFrame& f = out.frames[static_cast<size_t>(k)];
            ImGui::PushID(k);
            if (!f.cause.empty()) {
                ImGui::Dummy(ImVec2(0.0f, 2.0f));
                ImGui::TextColored(t.text_dim, "Caused by %s", f.cause.c_str());
            }
            if (!f.file.empty() && is_user(k)) {
                const nbview::FrameLink link = nbview::FrameLinkFor(f.file, f.line);
                const ImVec2 start = ImGui::GetCursorScreenPos();
                if (mono) ImGui::PushFont(mono);
                const bool can_open = link.cell_count > 0 || !link.path.empty();
                if (can_open) {
                    if (ui::LinkButton((link.label + "##frame").c_str())) OpenTraceFrame(link);
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
                        ImGui::SetTooltip("%s", link.path.empty() ? "Go to the cell" : link.path.c_str());
                } else {
                    ImGui::TextColored(t.text_dim, "%s", link.label.c_str());
                }
                ImGui::SameLine();
                ImGui::SetCursorScreenPos(ImVec2(std::max(ImGui::GetCursorScreenPos().x, start.x + link_w), ImGui::GetCursorScreenPos().y));
                ImGui::AlignTextToFramePadding();
                ImGui::TextColored(t.text_dim, "%s", f.code.c_str());
                if (mono) ImGui::PopFont();
            }
            ImGui::PopID();
        }
    }
    if (out.expanded && out.frames.empty() && !out.data.empty()) {
        ImGui::Dummy(ImVec2(0.0f, 2.0f));
        MonoText(out, out.data, ui::Mix(t.error, t.text, 0.65f));
    }

    ImGui::Dummy(ImVec2(0.0f, 2.0f));
    if (ui::SecondaryButton("Copy error")) {
        std::string text = out.data;
        if (!out.ename.empty() && text.find(out.ename) == std::string::npos) text += "\n" + out.ename + ": " + out.evalue;
        ImGui::SetClipboardText(text.c_str());
    }
    const int hidden = static_cast<int>(out.frames.size()) - static_cast<int>(user.size());
    const bool has_more = out.expanded || hidden > 0 || (out.frames.empty() && CountLines(TrimTrailingNewlines(out.data)) > 1);
    if (has_more) {
        ImGui::SameLine();
        if (ui::SecondaryButton(out.expanded ? "Show less" : "Show full traceback")) out.expanded = !out.expanded;
    }
    block.End(ui::WithAlpha(t.error, 0.07f), 6.0f);
}

void ScriptEditorPanel::RenderTableOutput(Cell& cell, CellOutput& out, float width) {
    const ui::Tokens& t = ui::CurrentTokens();
    const html::Table& table = *out.table_cache;
    const int cols = static_cast<int>(std::max(table.columns.size(), table.rows.empty() ? size_t{0} : table.rows.front().size()));
    if (cols <= 0) return;
    const float max_w = std::min(width, 960.0f);
    const int rows = static_cast<int>(table.rows.size());
    const float row_h = ImGui::GetTextLineHeight() + 8.0f;
    const bool scroll = rows > 15;
    ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoBordersInBody |
                            ImGuiTableFlags_Resizable;
    if (scroll) flags |= ImGuiTableFlags_ScrollY;
    ImGui::PushStyleColor(ImGuiCol_TableRowBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_TableRowBgAlt, ui::WithAlpha(t.text, 0.03f));
    ImGui::PushStyleColor(ImGuiCol_TableHeaderBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_TableBorderLight, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_TableBorderStrong, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(10.0f, 4.0f));
    if (ImGui::BeginTable("##result_table", cols, flags, ImVec2(max_w, scroll ? row_h * 16.0f : 0.0f))) {
        if (scroll) ImGui::TableSetupScrollFreeze(0, 1);
        for (int c = 0; c < cols; ++c) {
            const std::string name = c < static_cast<int>(table.columns.size()) ? table.columns[c] : std::string();
            const bool index_col = table.has_index && c == 0;
            ImGui::TableSetupColumn((name + "##c" + std::to_string(c)).c_str(),
                                    index_col ? ImGuiTableColumnFlags_WidthFixed : ImGuiTableColumnFlags_WidthStretch,
                                    index_col ? 64.0f : 1.0f);
        }
        ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
        for (int c = 0; c < cols; ++c) {
            ImGui::TableSetColumnIndex(c);
            ui::FontScope bold(ui::Font::Bold);
            ImGui::TextColored(t.text_dim, "%s", c < static_cast<int>(table.columns.size()) ? table.columns[c].c_str() : "");
        }
        ImGuiListClipper clipper;
        clipper.Begin(rows);
        while (clipper.Step()) {
            for (int r = clipper.DisplayStart; r < clipper.DisplayEnd; ++r) {
                ImGui::TableNextRow();
                const auto& row = table.rows[static_cast<size_t>(r)];
                for (int c = 0; c < cols; ++c) {
                    ImGui::TableSetColumnIndex(c);
                    const char* text = c < static_cast<int>(row.size()) ? row[c].c_str() : "";
                    if (table.has_index && c == 0) ImGui::TextColored(t.text_dim, "%s", text);
                    else ImGui::TextUnformatted(text);
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal) && ImGui::GetItemRectSize().x >= ImGui::GetColumnWidth() - 1.0f)
                        ImGui::SetTooltip("%s", text);
                }
            }
        }
        ImGui::EndTable();
    }
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(5);

    // Footer: the shape and the way to the full table.
    const int data_cols = cols - (table.has_index ? 1 : 0);
    char shape[96];
    std::snprintf(shape, sizeof(shape), "%d rows \xC3\x97 %d columns", rows, data_cols);
    ImGui::AlignTextToFramePadding();
    if (table.truncated && !table.footer.empty())
        ImGui::TextColored(t.text_dim, "Shortened by pandas: %s", table.footer.c_str());
    else
        ImGui::TextColored(t.text_dim, "%s", table.footer.empty() ? shape : table.footer.c_str());
    ImGui::SameLine(0.0f, 14.0f);
    if (ui::LinkButton("Open in Table Viewer")) OpenResultInTableViewer(cell, out);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("Every row of this value, in the Table Viewer");
    if (!table_open_error_.empty() && table_open_error_cell_ == cell.id) {
        ImGui::SameLine();
        ImGui::TextColored(t.warning, "%s", table_open_error_.c_str());
    }
}

void ScriptEditorPanel::OpenResultInTableViewer(Cell& cell, const CellOutput& out) {
    auto& tab = *tabs_[active_tab_index_];
    table_open_error_.clear();
    table_open_error_cell_ = cell.id;
    const std::string name = std::filesystem::path(tab.filename).stem().string() + " Out[" + std::to_string(cell.execution_count) + "]";
    std::shared_ptr<DataTable> table;
    // The whole value from Python (Out[n]); else the rows the output shows.
    std::string why;
    if (scripting_engine_ && cell.execution_count > 0) {
        std::error_code ec;
        const auto dir = std::filesystem::temp_directory_path(ec) / "cyxwiz_notebook_tables";
        std::filesystem::create_directories(dir, ec);
        const auto file = dir / (tab.cell_manager.NamespaceKey() + "_out" + std::to_string(cell.execution_count) + ".csv");
        if (scripting_engine_->ExportNotebookValueToCsv(tab.cell_manager.NamespaceKey(), cell.execution_count, file.string(), &why)) {
            auto loaded = std::make_shared<DataTable>();
            if (loaded->LoadFromCSV(file.string())) {
                loaded->SetName(name);
                table = loaded;
            }
        }
    }
    if (!table && out.table_cache) {
        table = ToDataTable(*out.table_cache, name + " (shown rows)");
        table_open_error_ = why.empty() ? "" : "Showing only the rows above: " + why;
    }
    if (!table) {
        table_open_error_ = why.empty() ? "Nothing to open" : why;
        return;
    }
    if (open_table_callback_) open_table_callback_(table);
    spdlog::info("Opened {} in the Table Viewer ({} rows)", name, table->GetRowCount());
}

void ScriptEditorPanel::RenderPlotOutput(Cell& cell, CellOutput& out, float width) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (out.texture_id == 0 && !out.image_data.empty()) {
        int w = 0, h = 0, channels = 0;
        unsigned char* pixels = stbi_load_from_memory(out.image_data.data(), static_cast<int>(out.image_data.size()), &w, &h, &channels, 4);
        if (pixels) {
            out.texture_id = OutputRenderer::CreateTextureFromRGBA(pixels, w, h);
            out.width = w;
            out.height = h;
            stbi_image_free(pixels);
        }
    }
    if (out.texture_id == 0) {
        ImGui::TextColored(t.text_dim, "The image could not be shown.");
        return;
    }
    const float side = 130.0f;  // the plot's toolbar to its right
    const bool beside = width >= 360.0f + side;
    const float max_w = beside ? width - side : width;
    const float scale = std::min(1.0f, max_w / static_cast<float>(std::max(1, out.width)));
    const ImVec2 size(out.width * scale, out.height * scale);
    ImGui::Image(static_cast<ImTextureID>(static_cast<intptr_t>(out.texture_id)), size);
    if (beside) {
        ImGui::SameLine(0.0f, 10.0f);
        ImGui::BeginGroup();
    }
    const std::string title = out.name.empty() ? "plot" : out.name;
    if (ui::GhostButton("Copy")) OutputRenderer::CopyImageToClipboard(out.image_data);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("Copy the image");
    if (!beside) ImGui::SameLine();
    if (ui::GhostButton("Save as...")) OutputRenderer::SaveImageToFile(out.image_data, title);
    if (!beside) ImGui::SameLine();
    if (ui::GhostButton("Open in window")) {
        PlotWindow w;
        w.id = next_plot_window_++;
        w.title = title;
        w.png = out.image_data;
        plot_windows_.push_back(std::move(w));
    }
    if (!beside) ImGui::SameLine();
    if (ui::GhostButton("Hide output")) cell.output_collapsed = true;
    if (beside) ImGui::EndGroup();
}

void ScriptEditorPanel::RenderPlotWindows() {
    for (auto& w : plot_windows_) {
        if (w.texture == 0 && !w.png.empty()) {
            int channels = 0;
            unsigned char* pixels = stbi_load_from_memory(w.png.data(), static_cast<int>(w.png.size()), &w.width, &w.height, &channels, 4);
            if (pixels) {
                w.texture = OutputRenderer::CreateTextureFromRGBA(pixels, w.width, w.height);
                stbi_image_free(pixels);
            }
        }
        const std::string name = w.title + "###notebook_plot" + std::to_string(w.id);
        ImGui::SetNextWindowSize(ImVec2(static_cast<float>(w.width) + 24.0f, static_cast<float>(w.height) + 64.0f), ImGuiCond_FirstUseEver);
        if (ImGui::Begin(name.c_str(), &w.open, ImGuiWindowFlags_NoCollapse)) {
            if (w.texture != 0) {
                // Fit the window, keeping the aspect ratio.
                const ImVec2 avail = ImGui::GetContentRegionAvail();
                const float s = std::max(0.05f, std::min(avail.x / std::max(1, w.width), (avail.y - ImGui::GetFrameHeightWithSpacing()) / std::max(1, w.height)));
                ImGui::Image(static_cast<ImTextureID>(static_cast<intptr_t>(w.texture)), ImVec2(w.width * s, w.height * s));
                if (ui::GhostButton("Copy")) OutputRenderer::CopyImageToClipboard(w.png);
                ImGui::SameLine();
                if (ui::GhostButton("Save as...")) OutputRenderer::SaveImageToFile(w.png, w.title);
            } else {
                ImGui::TextDisabled("The image could not be shown.");
            }
        }
        ImGui::End();
    }
    for (auto it = plot_windows_.begin(); it != plot_windows_.end();) {
        if (!it->open) {
            if (it->texture != 0) OutputRenderer::DeleteTexture(it->texture);
            it = plot_windows_.erase(it);
        } else {
            ++it;
        }
    }
}

void ScriptEditorPanel::OpenTraceFrame(const nbview::FrameLink& link) {
    auto& tab = *tabs_[active_tab_index_];
    if (link.cell_count > 0) {
        for (int i = 0; i < tab.cell_manager.GetCellCount(); ++i) {
            Cell& c = tab.cell_manager.GetCell(i);
            if (c.type == CellType::Code && c.execution_count == link.cell_count) {
                tab.selected_cell = i;
                tab.editing_cell = i;
                tab.last_editing_cell = i;
                c.SyncEditorFromSource();
                c.editor.GoToLine(std::max(0, link.line - 1));
                c.editor.RequestFocus();
                return;
            }
        }
        spdlog::info("Cell [{}] is not in this notebook any more (run again to renumber)", link.cell_count);
        return;
    }
    // Opened at the start of the next frame: a new tab must not be added
    // while this notebook's cells are being drawn (tabs_ may reallocate).
    if (!link.path.empty()) {
        deferred_open_path_ = link.path;
        deferred_open_line_ = link.line;
    }
}

}  // namespace cyxwiz
