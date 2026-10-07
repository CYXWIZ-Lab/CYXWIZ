#include "separate_windows.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <spdlog/spdlog.h>

#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// How it works: with ImGuiConfigFlags_ViewportsEnable, ImGui gives a floating
// window a window of its own as soon as it is no longer fully inside the main
// window ("late create" in Begin), and merges it back when it is dragged
// fully over the main window again. The windows that may leave say so with a
// window class; the hook below keeps every other floating window fully
// inside the main window, so ImGui never creates one for them. A window that
// left keeps ImGuiViewportFlags_NoAutoMerge until it is docked again, so it
// does not fall back inside when it happens to overlap the main window.

namespace gui {
namespace {

const ImGuiID kMayLeaveClass = ImHashStr("cyxwiz.window.may_leave");

ImGuiID g_main_dockspace = 0;
ImGuiID g_focused = 0;                               // the window that had the focus last (not a popup)
std::unordered_set<ImGuiID> g_leaving;               // asked to move out; takes a window of its own at its next Begin
std::unordered_map<ImGuiID, ImGuiID> g_home;         // last dock node inside the main window, per window
struct Pending { ImGuiID window; bool out; };
std::vector<Pending> g_pending;                      // commands, run at the start of the next frame

bool MayLeave(const ImGuiWindow* w) { return w->WindowClass.ClassId == kMayLeaveClass; }

// A dock node (or tree) may leave when every window in it may.
bool NodeMayLeave(const ImGuiDockNode* node) {
    if (!node) return true;
    if (node->IsLeafNode()) {
        for (ImGuiWindow* w : node->Windows)
            if (!MayLeave(w)) return false;
        return true;
    }
    return NodeMayLeave(node->ChildNodes[0]) && NodeMayLeave(node->ChildNodes[1]);
}

bool InMainWindow(const ImGuiWindow* w) { return w->Viewport == ImGui::GetMainViewport(); }

void Undock(ImGuiContext& g, ImGuiWindow* w) {
    if (w->DockNode || w->DockId) ImGui::DockContextProcessUndockWindow(&g, w, true);
}

// Start of the frame: the commands asked for during the last one, while no
// window is between Begin and End.
void RunPending(ImGuiContext* ctx, ImGuiContextHook*) {
    ImGuiContext& g = *ctx;
    for (const Pending& p : g_pending) {
        ImGuiWindow* w = ImGui::FindWindowByID(p.window);
        if (!w) continue;
        if (p.out) {
            if (w->DockId && InMainWindow(w)) g_home[w->ID] = w->DockId;
            // The window lifts off where it is, at the size it had.
            const ImVec2 pos = w->Pos, size = w->Size;
            Undock(g, w);
            ImGui::SetWindowPos(w, pos, ImGuiCond_Always);
            ImGui::SetWindowSize(w, size, ImGuiCond_Always);
            g_leaving.insert(w->ID);
            ImGui::FocusWindow(w);
            spdlog::info("Window '{}' moves to a window of its own", w->Name);
        } else {
            g_leaving.erase(w->ID);
            ImGuiID target = 0;
            if (auto it = g_home.find(w->ID); it != g_home.end() && ImGui::DockContextFindNodeByID(&g, it->second))
                target = it->second;
            if (!target) target = g_main_dockspace;
            if (!target) {
                spdlog::warn("Window '{}': no dock space to return to", w->Name);
                continue;
            }
            ImGui::SetWindowDock(w, target, ImGuiCond_Always);
            ImGui::FocusWindow(w);
            spdlog::info("Window '{}' docks back into the main window", w->Name);
        }
    }
    g_pending.clear();
}

// End of NewFrame: every floating window that may not leave is kept fully
// inside the main window (ImGui only creates a window of its own for one
// that is not). Also notes where the windows that may leave are docked.
void KeepInside(ImGuiContext* ctx, ImGuiContextHook*) {
    ImGuiContext& g = *ctx;
    // The focused window, remembered here because opening a menu moves the
    // focus to the menu itself: the panel behind the focused child (a docked
    // panel is a child of its dock host, so the climb stops there), unless a
    // popup or menu has it.
    if (ImGuiWindow* nav = g.NavWindow) {
        bool popup = false;
        for (ImGuiWindow* w = nav; w; w = (w->Flags & ImGuiWindowFlags_ChildWindow) && !w->DockIsActive ? w->ParentWindow : nullptr) {
            if (w->Flags & (ImGuiWindowFlags_Popup | ImGuiWindowFlags_ChildMenu | ImGuiWindowFlags_Tooltip)) popup = true;
            nav = w;
        }
        if (!popup && !(nav->Flags & ImGuiWindowFlags_DockNodeHost)) g_focused = nav->ID;
    }
    ImGuiViewportP* main = g.Viewports[0];
    const ImRect main_rect = main->GetMainRect();
    for (ImGuiWindow* w : g.Windows) {
        if (!w->WasActive) continue;
        if (w->Flags & (ImGuiWindowFlags_ChildWindow | ImGuiWindowFlags_Popup | ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_ChildMenu))
            continue;
        if (MayLeave(w)) {
            if (w->DockId && InMainWindow(w) && (w->DockIsActive || w->DockNode)) g_home[w->ID] = w->DockId;
            continue;
        }
        if (w->DockIsActive) continue;  // follows its host
        if (!InMainWindow(w) && !w->ViewportOwned) continue;  // placed in another window's viewport on purpose (NextWindowFollowsCurrent)
        if (w->DockNodeAsHost ? NodeMayLeave(w->DockNodeAsHost) : false) continue;
        const ImVec2 size(ImMin(w->Size.x, main->Size.x), ImMin(w->Size.y, main->Size.y));
        const ImVec2 pos(ImClamp(w->Pos.x, main_rect.Min.x, main_rect.Max.x - size.x),
                         ImClamp(w->Pos.y, main_rect.Min.y, main_rect.Max.y - size.y));
        if (size.x != w->Size.x || size.y != w->Size.y) ImGui::SetWindowSize(w, size, ImGuiCond_Always);
        if (pos.x != w->Pos.x || pos.y != w->Pos.y) ImGui::SetWindowPos(w, pos, ImGuiCond_Always);
    }
}

// The platform backend's title setter; ours strips the tab's icon glyphs
// (private-use code points, which the OS title bar cannot draw) and names
// the Engine.
void (*g_backend_set_title)(ImGuiViewport*, const char*) = nullptr;

void SetTitle(ImGuiViewport* vp, const char* name) { g_backend_set_title(vp, SeparateWindowTitle(name).c_str()); }

}  // namespace

std::string SeparateWindowTitle(const char* name) {
    std::string title;
    const char* p = name;
    const char* end = name + strlen(name);
    while (p < end) {
        unsigned int c = 0;
        const int n = ImTextCharFromUtf8(&c, p, end);
        if (n <= 0) break;
        const bool icon = (c >= 0xE000 && c <= 0xF8FF) || c >= 0xF0000;
        if (!icon) title.append(p, static_cast<size_t>(n));
        p += n;
    }
    const size_t first = title.find_first_not_of(' ');
    title = first == std::string::npos ? std::string() : title.substr(first);
    // The window's name first, as documents name their windows; the main
    // window is the one that starts with the Engine's name.
    return title + " - CyxWiz Engine";
}

void InstallSeparateWindows() {
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    io.ConfigViewportsNoDecoration = false;  // a window of its own has the OS title bar and borders
    ImGuiContext& g = *GImGui;
    ImGuiContextHook pre;
    pre.Type = ImGuiContextHookType_NewFramePre;
    pre.Callback = RunPending;
    ImGui::AddContextHook(&g, &pre);
    ImGuiContextHook post;
    post.Type = ImGuiContextHookType_NewFramePost;
    post.Callback = KeepInside;
    ImGui::AddContextHook(&g, &post);
}

void NameSeparateWindows() {
    ImGuiPlatformIO& pio = ImGui::GetPlatformIO();
    if (!pio.Platform_SetWindowTitle || pio.Platform_SetWindowTitle == SetTitle) return;
    g_backend_set_title = pio.Platform_SetWindowTitle;
    pio.Platform_SetWindowTitle = SetTitle;
}

void NextWindowMayLeave(const char* name) {
    ImGuiWindowClass cls;
    cls.ClassId = kMayLeaveClass;
    cls.DockingAllowUnclassed = true;
    ImGuiWindow* w = ImGui::FindWindowByName(name);
    bool outside = false;
    if (w) {
        outside = w->WasActive && !InMainWindow(w);
        if (outside) g_leaving.erase(w->ID);
        outside = outside || g_leaving.count(w->ID);
    } else if (const ImGuiWindowSettings* saved = ImGui::FindWindowSettingsByID(ImHashStr(name))) {
        // Not shown yet this session: outside when the layout left it there,
        // so it does not fall back inside on its first frame.
        outside = saved->ViewportId != 0 && saved->ViewportId != ImGui::GetMainViewport()->ID;
    }
    if (outside) {
        // Outside: its own window until it is docked again, and a tab strip
        // under the OS title bar instead of a second title bar.
        cls.ViewportFlagsOverrideSet |= ImGuiViewportFlags_NoAutoMerge;
        cls.DockingAlwaysTabBar = true;
    }
    ImGui::SetNextWindowClass(&cls);
}

void NextWindowStaysInMain() { ImGui::SetNextWindowViewport(ImGui::GetMainViewport()->ID); }

void NextWindowFollowsCurrent() { ImGui::SetNextWindowViewport(ImGui::GetWindowViewport()->ID); }

void SetMainDockSpace(unsigned int dockspace_id) { g_main_dockspace = dockspace_id; }

bool WindowIsOutside(const char* name) {
    const ImGuiWindow* w = ImGui::FindWindowByName(name);
    return w && w->WasActive && !InMainWindow(w);
}

bool WindowMayLeave(const char* name) {
    const ImGuiWindow* w = ImGui::FindWindowByName(name);
    return w && MayLeave(w);
}

std::string FocusedWindowName() {
    const ImGuiWindow* w = g_focused ? ImGui::FindWindowByID(g_focused) : nullptr;
    return w && w->WasActive ? std::string(w->Name) : std::string();
}

void MoveWindowOut(const char* name) {
    if (const ImGuiWindow* w = ImGui::FindWindowByName(name)) g_pending.push_back({w->ID, true});
}

void MoveWindowBack(const char* name) {
    if (const ImGuiWindow* w = ImGui::FindWindowByName(name)) g_pending.push_back({w->ID, false});
}

void MoveAllWindowsBack() {
    ImGuiContext& g = *GImGui;
    for (ImGuiWindow* w : g.Windows)
        if (MayLeave(w) && w->WasActive && !InMainWindow(w)) g_pending.push_back({w->ID, false});
}

void TabMenu(const char* name, bool* open) {
    // After Begin, the "item" is the window's tab or title bar.
    if (!ImGui::BeginPopupContextItem("##window_home")) return;
    const bool outside = WindowIsOutside(name);
    if (ImGui::MenuItem("Move to its own window", nullptr, false, !outside)) MoveWindowOut(name);
    if (ImGui::MenuItem("Dock back into the main window", nullptr, false, outside)) MoveWindowBack(name);
    if (open) {
        ImGui::Separator();
        if (ImGui::MenuItem("Close")) *open = false;
    }
    ImGui::EndPopup();
}

}  // namespace gui
