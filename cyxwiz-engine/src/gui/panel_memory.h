#pragma once

// Open panels remembered between sessions (TOFIX129 step 0.6). The panels
// registered with the sidebar (DockStyle, which is the panel registry) keep
// their open or closed state in imgui.ini under [CyxWizPanels][Open], next to
// the dock layout they belong to.

namespace gui {

// Once, after ImGui::CreateContext and before the first NewFrame (which
// loads imgui.ini).
void InstallPanelMemory();

// After the panels are registered: applies the remembered state. Panels with
// nothing remembered keep their defaults.
void ApplyRememberedPanels();

// Every frame: notes changes so imgui.ini is saved. Reads the registered
// visibility flags only here, never while ImGui saves, because the panels
// may already be destroyed when ImGui saves at shutdown.
void TrackPanelChanges();

// View > Layout > Reset to Default: forget the remembered state.
void ForgetRememberedPanels();

}  // namespace gui
