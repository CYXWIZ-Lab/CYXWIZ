#include "panel_memory.h"

#include "dock_style.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <spdlog/spdlog.h>

#include <cstring>
#include <map>
#include <string>

namespace gui {

namespace {
constexpr const char* kTypeName = "CyxWizPanels";

// Panel name -> open. Filled from imgui.ini, then kept current each frame.
std::map<std::string, bool>& Remembered() {
    static std::map<std::string, bool> remembered;
    return remembered;
}

void* ReadOpen(ImGuiContext*, ImGuiSettingsHandler*, const char* name) {
    return std::strcmp(name, "Open") == 0 ? &Remembered() : nullptr;
}

void ReadLine(ImGuiContext*, ImGuiSettingsHandler*, void*, const char* line) {
    // "Script Editor=1": names may contain spaces, so split at the last '='.
    const char* eq = std::strrchr(line, '=');
    if (!eq || eq == line) return;
    Remembered()[std::string(line, eq)] = eq[1] == '1';
}

void WriteAll(ImGuiContext*, ImGuiSettingsHandler* handler, ImGuiTextBuffer* out) {
    if (Remembered().empty()) return;
    out->appendf("[%s][Open]\n", handler->TypeName);
    for (const auto& [name, open] : Remembered()) out->appendf("%s=%d\n", name.c_str(), open ? 1 : 0);
    out->append("\n");
}
}  // namespace

void InstallPanelMemory() {
    ImGuiSettingsHandler handler;
    handler.TypeName = kTypeName;
    handler.TypeHash = ImHashStr(kTypeName);
    handler.ReadOpenFn = ReadOpen;
    handler.ReadLineFn = ReadLine;
    handler.WriteAllFn = WriteAll;
    ImGui::AddSettingsHandler(&handler);
}

void ApplyRememberedPanels() {
    int applied = 0;
    for (const auto& panel : GetDockStyle().GetPanels()) {
        if (!panel.visible_ptr) continue;
        const auto it = Remembered().find(panel.name);
        if (it == Remembered().end()) continue;
        *panel.visible_ptr = it->second;
        ++applied;
    }
    if (applied > 0) spdlog::info("Restored {} panels as they were in the last session", applied);
}

void TrackPanelChanges() {
    bool changed = false;
    for (const auto& panel : GetDockStyle().GetPanels()) {
        if (!panel.visible_ptr) continue;
        auto [it, inserted] = Remembered().try_emplace(panel.name, *panel.visible_ptr);
        if (inserted || it->second != *panel.visible_ptr) {
            it->second = *panel.visible_ptr;
            changed = true;
        }
    }
    if (changed) ImGui::MarkIniSettingsDirty();
}

void ForgetRememberedPanels() {
    Remembered().clear();
    ImGui::MarkIniSettingsDirty();
}

}  // namespace gui
