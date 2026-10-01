// Script Editor: files that change on disk while open (TOFIX133 P2 step 2.6,
// approved board 2). Every 2 s each open file's write time is compared with
// the one seen at load or save. An unmodified tab reloads quietly; a tab
// with edits shows a band: Keep my edits / Reload from disk / Compare.

#include "script_editor.h"

#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <fstream>
#include <iterator>
#include <system_error>

namespace cyxwiz {

namespace {
bool ReadDiskText(const std::string& path, std::string& out) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    out.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    return true;
}
}  // namespace

void ScriptEditorPanel::NoteDiskTime(EditorTab& tab) {
    tab.disk_changed = false;
    tab.disk_missing = false;
    if (tab.filepath.empty()) return;
    std::error_code ec;
    const auto time = std::filesystem::last_write_time(tab.filepath, ec);
    tab.disk_time = ec ? std::filesystem::file_time_type{} : time;
}

void ScriptEditorPanel::CheckFilesOnDisk() {
    const double now = ImGui::GetTime();
    if (now - disk_check_time_ < 2.0) return;
    disk_check_time_ = now;
    for (auto& tab : tabs_) {
        if (!tab || tab->filepath.empty() || tab->is_new || tab->is_loading || tab->load_failed || tab->is_large_file ||
            tab->disk_changed)
            continue;
        std::error_code ec;
        const auto time = std::filesystem::last_write_time(tab->filepath, ec);
        if (ec) {
            if (!tab->disk_missing) {
                tab->disk_missing = true;
                tab->disk_changed = true;
                spdlog::warn("{} was deleted or moved on disk", tab->filepath);
            }
            continue;
        }
        if (time == tab->disk_time) continue;
        if (!tab->is_modified) {
            ReloadFromDisk(*tab);
            spdlog::info("Reloaded {} (changed on disk)", tab->filepath);
        } else {
            tab->disk_changed = true;
            spdlog::warn("{} changed on disk while it has unsaved edits", tab->filepath);
        }
    }
}

void ScriptEditorPanel::ReloadFromDisk(EditorTab& tab) {
    std::string raw;
    if (!ReadDiskText(tab.filepath, raw)) return;
    scriptfile::Decoded decoded = scriptfile::Decode(raw);
    tab.format = decoded.format;
    tab.format_changed = false;
    const int line = tab.editor.Doc().Primary().head.line;
    if (scriptfile::IsNotebookJson(tab.filepath)) {
        if (!tab.cell_manager.ParseFromIpynb(decoded.text)) return;  // half-written by another program: try again later
        tab.selected_cell = tab.cell_manager.GetCellCount() > 0 ? 0 : -1;
        tab.editing_cell = -1;
        tab.last_editing_cell = -1;
    } else if (tab.cell_mode) {
        tab.cell_manager.ParseFromCyx(decoded.text);
        tab.selected_cell = tab.cell_manager.GetCellCount() > 0 ? 0 : -1;
        tab.editing_cell = -1;
        tab.last_editing_cell = -1;
    }
    tab.editor.SetText(decoded.text);
    tab.editor.Doc().SetCursor(tab.editor.Doc().Clamp({line, 0}));
    tab.editor.ScrollToCursor();
    tab.is_modified = false;
    NoteDiskTime(tab);
}

void ScriptEditorPanel::RenderDiskChangedBand(EditorTab& tab) {
    if (!tab.disk_changed) return;
    const ui::Tokens& t = ui::CurrentTokens();
    const float width = ImGui::GetContentRegionAvail().x;
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ui::WithAlpha(t.warning, 0.12f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 6.0f));
    ImGui::BeginChild("##disk_changed", ImVec2(width, 0.0f), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding,
                      ImGuiWindowFlags_NoScrollbar);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.warning, "%s", ICON_FA_TRIANGLE_EXCLAMATION);
    ImGui::SameLine();
    ImGui::PushTextWrapPos(width - 360.0f > 200.0f ? ImGui::GetCursorPosX() + width - 380.0f : 0.0f);
    if (tab.disk_missing) ImGui::TextUnformatted("This file was deleted or moved on disk. Saving writes it again.");
    else ImGui::TextUnformatted("This file changed on disk while you have unsaved edits. Your edits are kept until you choose.");
    ImGui::PopTextWrapPos();
    ImGui::SameLine();
    if (ui::SecondaryButton("Keep my edits", true, nullptr)) {
        // The next save writes the edits over the disk version.
        NoteDiskTime(tab);
        tab.disk_changed = false;
    }
    if (!tab.disk_missing) {
        ImGui::SameLine();
        if (ui::DangerButton("Reload from disk", true, nullptr)) ReloadFromDisk(tab);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("Your unsaved edits are lost");
        ImGui::SameLine();
        if (ui::LinkButton("Compare")) OpenDiskCopy(tab);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("Open the version on disk in a read-only tab");
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

void ScriptEditorPanel::OpenDiskCopy(EditorTab& tab) {
    std::string raw;
    if (!ReadDiskText(tab.filepath, raw)) return;
    auto copy = std::make_unique<EditorTab>();
    copy->document_id = next_document_id_++;
    copy->filename = tab.filename + " (on disk)";
    copy->is_new = false;
    copy->is_modified = false;
    ConfigureEditor(copy->editor);
    copy->editor.SetText(scriptfile::Decode(raw).text);
    copy->editor.SetReadOnly(true);
    tabs_.push_back(std::move(copy));
    active_tab_index_ = static_cast<int>(tabs_.size()) - 1;
    request_focus_ = true;
}

}  // namespace cyxwiz
