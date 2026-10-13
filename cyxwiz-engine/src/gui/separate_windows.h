#pragma once

#include <string>

// Windows of their own (TOFIX129 piece A8). The big windows (Dashboard,
// Plot, Data Studio, Script Editor, ...) may leave the Engine's main window
// and become windows of the operating system, on any monitor, and come
// back; every other window stays inside the main window. ImGui's
// multi-viewport mode does the OS part; this file decides which window may
// leave, keeps the rest inside and offers the two commands.

namespace gui {

// Once, after ImGui::CreateContext and before the backends are set up:
// turns the mode on and installs the frame hook that keeps every window
// that did not ask to leave inside the main window.
void InstallSeparateWindows();

// Once, after the platform backend is set up: a window of its own is titled
// "CyxWiz Engine - <window name>" without the icon glyphs of the tab.
void NameSeparateWindows();
// That title for a window name (the part before "###").
std::string SeparateWindowTitle(const char* name);

// Before ImGui::Begin(name, ...) of a window that may leave. `name` is the
// string given to Begin.
void NextWindowMayLeave(const char* name);

// Before ImGui::Begin of a window the Engine places itself (sidebar, status
// bar, overlays): always drawn in the main window, even when it slides
// partly out of view.
void NextWindowStaysInMain();

// Before ImGui::Begin of an overlay drawn over the current window (minimap,
// search bar, completion list): it goes with that window when the window
// is in a window of its own. Called between the owner's Begin and End.
void NextWindowFollowsCurrent();

// Every frame from the main window's dock space: where "Dock back" returns
// a window to.
void SetMainDockSpace(unsigned int dockspace_id);

// The window is in a window of its own right now.
bool WindowIsOutside(const char* name);

// The window asked NextWindowMayLeave.
bool WindowMayLeave(const char* name);

// The window that had the focus last, popups and menus aside (the name
// given to Begin); empty when none. For View > Window.
std::string FocusedWindowName();

// The two commands, from a tab's menu and from View > Window.
void MoveWindowOut(const char* name);
void MoveWindowBack(const char* name);

// Every window back inside (View > Layout > Reset to Default).
void MoveAllWindowsBack();

// Right after ImGui::Begin of a window that may leave: the menu on its tab
// or title bar with the two commands and Close (`open` is the flag given to
// Begin; may be null).
void TabMenu(const char* name, bool* open);

}  // namespace gui
