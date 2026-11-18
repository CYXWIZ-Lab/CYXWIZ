// Windows of their own (TOFIX129 A8): the contract of gui/separate_windows
// against a headless ImGui context with a pretend platform (one monitor; a
// window of its own is a position, a size and a title).
#include "../src/gui/separate_windows.h"

#include <imgui.h>
#include <imgui_internal.h>

#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>

namespace {

void Check(bool ok, const std::string& message) {
    if (!ok) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

constexpr const char* kDashboard = "Dashboard \xC2\xB7 Text###dash_1";
constexpr const char* kProperties = "Properties";
constexpr ImGuiID kDockSpace = 0x5EED;

bool g_dashboard_open = true;
std::function<void()> g_before_dockspace;  // dock builder calls, inside the next frame
bool g_draw_bar = false;  // a window the Engine places itself (sidebar, status bar)
constexpr const char* kBar = "Bar";

// The pretend platform.
struct FakeWindow { ImVec2 pos, size; };
std::string g_last_title;
FakeWindow* Fake(ImGuiViewport* vp) { return static_cast<FakeWindow*>(vp->PlatformUserData); }

void InstallFakePlatform() {
    ImGuiPlatformIO& pio = ImGui::GetPlatformIO();
    pio.Platform_CreateWindow = [](ImGuiViewport* vp) { vp->PlatformUserData = new FakeWindow{vp->Pos, vp->Size}; };
    pio.Platform_DestroyWindow = [](ImGuiViewport* vp) { delete Fake(vp); vp->PlatformUserData = nullptr; };
    pio.Platform_ShowWindow = [](ImGuiViewport*) {};
    pio.Platform_SetWindowPos = [](ImGuiViewport* vp, ImVec2 pos) { Fake(vp)->pos = pos; };
    pio.Platform_GetWindowPos = [](ImGuiViewport* vp) { return Fake(vp)->pos; };
    pio.Platform_SetWindowSize = [](ImGuiViewport* vp, ImVec2 size) { Fake(vp)->size = size; };
    pio.Platform_GetWindowSize = [](ImGuiViewport* vp) { return Fake(vp)->size; };
    pio.Platform_SetWindowFocus = [](ImGuiViewport*) {};
    pio.Platform_GetWindowFocus = [](ImGuiViewport*) { return true; };
    pio.Platform_GetWindowMinimized = [](ImGuiViewport*) { return false; };
    pio.Platform_SetWindowTitle = [](ImGuiViewport*, const char* title) { g_last_title = title; };
    ImGuiPlatformMonitor monitor;
    monitor.MainPos = monitor.WorkPos = ImVec2(0, 0);
    monitor.MainSize = monitor.WorkSize = ImVec2(2560, 1440);
    pio.Monitors.push_back(monitor);
    ImGuiViewport* main = ImGui::GetMainViewport();
    main->PlatformUserData = new FakeWindow{ImVec2(100, 100), ImGui::GetIO().DisplaySize};  // not at the screen's corner
    main->PlatformHandle = main->PlatformUserData;
}

// One frame of the Engine's shape: a dock space host, a window that may
// leave, a window that may not; the positions apply before Begin.
void Frame(const ImVec2* dashboard_pos = nullptr, const ImVec2* properties_pos = nullptr) {
    ImGui::NewFrame();
    if (g_before_dockspace) {
        g_before_dockspace();
        g_before_dockspace = nullptr;
    }
    const ImGuiViewport* main = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(main->Pos);
    ImGui::SetNextWindowSize(main->Size);
    ImGui::SetNextWindowViewport(main->ID);
    ImGui::Begin("Host", nullptr, ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove);
    gui::SetMainDockSpace(kDockSpace);
    ImGui::DockSpace(kDockSpace);
    ImGui::End();

    if (dashboard_pos) ImGui::SetNextWindowPos(*dashboard_pos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_Once);
    gui::NextWindowMayLeave(kDashboard);
    ImGui::Begin(kDashboard, &g_dashboard_open);
    gui::TabMenu(kDashboard, &g_dashboard_open);
    ImGui::Text("cards");
    ImGui::End();

    if (properties_pos) ImGui::SetNextWindowPos(*properties_pos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(200, 150), ImGuiCond_Once);
    ImGui::Begin(kProperties);
    ImGui::Text("fields");
    ImGui::End();

    if (g_draw_bar) {
        gui::NextWindowStaysInMain();
        ImGui::SetNextWindowSize(ImVec2(120, 60), ImGuiCond_Once);
        ImGui::Begin(kBar);
        ImGui::End();
    }
    gui::WindowMenus();
    ImGui::EndFrame();
    ImGui::UpdatePlatformWindows();
}

ImGuiWindow* Window(const char* name) { return ImGui::FindWindowByName(name); }

void Setup() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = ImVec2(1280, 720);
    io.DeltaTime = 1.0f / 60.0f;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.BackendFlags |= ImGuiBackendFlags_PlatformHasViewports | ImGuiBackendFlags_RendererHasViewports;
    unsigned char* pixels = nullptr;
    int width = 0, height = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    gui::InstallSeparateWindows();
    InstallFakePlatform();
    gui::NameSeparateWindows();
}

void Teardown() {
    ImGui::DestroyPlatformWindows();
    delete Fake(ImGui::GetMainViewport());
    ImGui::GetMainViewport()->PlatformUserData = ImGui::GetMainViewport()->PlatformHandle = nullptr;
    ImGui::DestroyContext();
}

}  // namespace

int main() {
    Setup();
    Check(ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable, "the mode is on");

    // Titles: no icon glyphs, the window's name first.
    Check(gui::SeparateWindowTitle("\xEF\x80\x80 Variable Explorer") == "Variable Explorer - CyxWiz Engine", "icon stripped");
    Check(gui::SeparateWindowTitle("Plot \xC2\xB7 loss") == "Plot \xC2\xB7 loss - CyxWiz Engine", "plain title kept");

    // Both windows start inside the main window.
    const ImVec2 inside(300, 300);
    for (int i = 0; i < 3; ++i) Frame(&inside, &inside);
    Check(ImGui::GetMainViewport()->Pos.x == 100, "main window at its platform position");
    Check(Window(kDashboard)->Viewport == ImGui::GetMainViewport(), "dashboard starts inside");
    Check(gui::WindowMayLeave(kDashboard), "dashboard may leave");
    Check(!gui::WindowMayLeave(kProperties), "properties may not");
    Check(!gui::WindowIsOutside(kDashboard), "dashboard not outside yet");

    // Dragged past the edge: the dashboard gets a window of its own, the
    // properties window is held inside.
    const ImVec2 beyond(1300, 300);  // right of the main window (100 + 1280)
    for (int i = 0; i < 3; ++i) {
        ImGui::SetWindowPos(Window(kProperties), beyond, ImGuiCond_Always);  // as a drag moves it, before the frame
        Frame(&beyond, nullptr);
    }
    Check(Window(kDashboard)->Viewport != ImGui::GetMainViewport(), "dashboard is in a viewport of its own");
    Check(gui::WindowIsOutside(kDashboard), "dashboard reports outside");
    Check(g_last_title == "Dashboard \xC2\xB7 Text - CyxWiz Engine", "its window is titled: " + g_last_title);
    Check(Window(kProperties)->Viewport == ImGui::GetMainViewport(), "properties stays in the main window");
    Check(Window(kProperties)->Pos.x + Window(kProperties)->Size.x <= 100 + 1280 + 0.5f, "properties clamped inside");

    // Back over the main window: the dashboard keeps its own window until docked.
    for (int i = 0; i < 3; ++i) Frame(&inside, nullptr);
    Check(Window(kDashboard)->Viewport != ImGui::GetMainViewport(), "a window that left does not fall back inside");
    Check(Window(kDashboard)->DockNode && Window(kDashboard)->DockNode->IsFloatingNode(), "outside it sits in a floating node (tab strip under the OS title bar)");

    // Dock back: into the main dock space.
    gui::MoveWindowBack(kDashboard);
    for (int i = 0; i < 4; ++i) Frame();
    Check(Window(kDashboard)->DockNode && !Window(kDashboard)->DockNode->IsFloatingNode(), "docked back into the dock space");
    Check(Window(kDashboard)->Viewport == ImGui::GetMainViewport(), "inside after dock back");
    Check(!gui::WindowIsOutside(kDashboard), "not outside after dock back");

    // Move out from the dock: a window of its own where it was.
    gui::MoveWindowOut(kDashboard);
    for (int i = 0; i < 4; ++i) Frame();
    Check(!Window(kDashboard)->DockNode || Window(kDashboard)->DockNode->IsFloatingNode(), "left the main dock space");
    Check(gui::WindowIsOutside(kDashboard), "its own window after the command");
    Check(Window(kDashboard)->Size.x >= 1100, "at the size it had in the dock (" + std::to_string(int(Window(kDashboard)->Size.x)) + ")");

    // Reset layout brings it back.
    gui::MoveAllWindowsBack();
    for (int i = 0; i < 4; ++i) Frame();
    Check(Window(kDashboard)->Viewport == ImGui::GetMainViewport(), "all back inside");

    // The focused window, remembered across a popup.
    ImGui::FocusWindow(Window(kDashboard));
    Frame();
    Check(gui::FocusedWindowName() == kDashboard, "focused window is the dashboard: " + gui::FocusedWindowName());

    // Every window may leave (Preferences > General, A8b). On: the
    // properties window gets a window of its own and keeps it over the main
    // window; a window the Engine places itself still stays; off again: held
    // inside as before.
    gui::SetEveryWindowMayLeave(true);
    g_draw_bar = true;
    for (int i = 0; i < 2; ++i) Frame();
    Check(gui::WindowMayLeave(kProperties), "every window: properties may leave");
    Check(!gui::WindowMayLeave(kBar), "a window the Engine places never leaves");
    for (int i = 0; i < 3; ++i) {
        ImGui::SetWindowPos(Window(kProperties), beyond, ImGuiCond_Always);
        ImGui::SetWindowPos(Window(kBar), beyond, ImGuiCond_Always);
        Frame();
    }
    Check(Window(kProperties)->Viewport != ImGui::GetMainViewport(), "properties in a window of its own");
    Check(gui::WindowIsOutside(kProperties), "properties reports outside");
    Check(Window(kBar)->Viewport == ImGui::GetMainViewport(), "the bar stays in the main window");
    for (int i = 0; i < 3; ++i) {
        ImGui::SetWindowPos(Window(kProperties), inside, ImGuiCond_Always);
        Frame();
    }
    Check(Window(kProperties)->Viewport != ImGui::GetMainViewport(), "it does not fall back in over the main window");
    gui::MoveWindowBack(kProperties);
    for (int i = 0; i < 4; ++i) Frame();
    Check(Window(kProperties)->Viewport == ImGui::GetMainViewport() && Window(kProperties)->DockNode,
          "Dock back works for it too");
    gui::MoveWindowOut(kProperties);
    for (int i = 0; i < 4; ++i) Frame();
    Check(gui::WindowIsOutside(kProperties), "and Move to its own window");
    // Dock back when its place closed as it left: a split dock space, the
    // window alone on one side. It came back into the split root, where it
    // was not shown; it goes to the central area, shown.
    gui::MoveWindowBack(kProperties);
    for (int i = 0; i < 3; ++i) Frame();
    ImGuiID left = 0, right = 0;
    g_before_dockspace = [&]() {
        ImGui::DockBuilderRemoveNode(kDockSpace);
        ImGui::DockBuilderAddNode(kDockSpace, ImGuiDockNodeFlags_DockSpace);
        ImGui::DockBuilderSetNodeSize(kDockSpace, ImGui::GetMainViewport()->Size);
        ImGui::DockBuilderSplitNode(kDockSpace, ImGuiDir_Left, 0.5f, &left, &right);
        ImGui::DockBuilderDockWindow(kDashboard, left);
        ImGui::DockBuilderDockWindow(kProperties, right);
        ImGui::DockBuilderFinish(kDockSpace);
    };
    for (int i = 0; i < 4; ++i) Frame();
    Check(Window(kProperties)->DockNode && Window(kProperties)->DockNode->ID == right, "properties alone on the right");
    gui::MoveWindowOut(kProperties);
    for (int i = 0; i < 4; ++i) Frame();
    Check(gui::WindowIsOutside(kProperties), "out of the split dock space");
    gui::MoveWindowBack(kProperties);
    for (int i = 0; i < 4; ++i) Frame();
    {
        const ImGuiWindow* w = Window(kProperties);
        Check(w->Viewport == ImGui::GetMainViewport() && w->DockNode && w->DockNode->IsLeafNode() && w->DockNode->IsVisible,
              "docked back where it shows (a leaf node of the main dock space)");
    }
    g_before_dockspace = []() { ImGui::DockBuilderRemoveNode(kDockSpace); };  // the plain dock space again below
    for (int i = 0; i < 3; ++i) Frame();

    gui::SetEveryWindowMayLeave(false);
    for (int i = 0; i < 3; ++i) Frame();
    Check(!gui::WindowMayLeave(kProperties), "off: properties may not leave");
    Check(Window(kProperties)->Viewport == ImGui::GetMainViewport(), "off: held inside the main window again");
    g_draw_bar = false;

    // Bring to front: both windows as tabs of one dock node; the latest
    // request wins and its tab shows, whichever window draws later.
    ImGui::DockBuilderDockWindow(kDashboard, kDockSpace);
    ImGui::DockBuilderDockWindow(kProperties, kDockSpace);
    for (int i = 0; i < 4; ++i) Frame();
    Check(Window(kDashboard)->DockNode && Window(kDashboard)->DockNode == Window(kProperties)->DockNode, "one dock node");
    gui::BringToFront(kProperties);
    gui::BringToFront(kDashboard);
    for (int i = 0; i < 2; ++i) Frame();
    Check(Window(kDashboard)->DockTabIsVisible && !Window(kProperties)->DockTabIsVisible, "the latest request's tab shows");
    Check(gui::FocusedWindowName() == kDashboard, "and it has the focus: " + gui::FocusedWindowName());
    gui::BringToFront(kProperties);  // drawn after the dashboard: wins by request, not by draw order
    for (int i = 0; i < 2; ++i) Frame();
    Check(Window(kProperties)->DockTabIsVisible && !Window(kDashboard)->DockTabIsVisible, "the hidden tab comes forward");
    gui::BringToFront("###dash_1");  // by its id part, as a window whose title changes
    for (int i = 0; i < 2; ++i) Frame();
    Check(Window(kDashboard)->DockTabIsVisible, "found by the id part of its name");
    gui::BringToFront("Never drawn");  // let go after a while, no harm
    for (int i = 0; i < 40; ++i) Frame();
    Check(Window(kDashboard)->DockTabIsVisible, "a request for a missing window changes nothing");

    // Between sessions: outside at the end of one, outside at the start of
    // the next (also when its window overlaps the main window).
    gui::MoveWindowOut(kDashboard);
    for (int i = 0; i < 4; ++i) Frame();
    Check(gui::WindowIsOutside(kDashboard), "outside before the layout is saved");
    const std::string ini = ImGui::SaveIniSettingsToMemory();
    Check(ini.find("ViewportId=") != std::string::npos, "the layout records the window's viewport");
    Teardown();
    Setup();
    ImGui::LoadIniSettingsFromMemory(ini.c_str(), ini.size());
    for (int i = 0; i < 4; ++i) Frame();
    Check(gui::WindowIsOutside(kDashboard), "outside again in the next session");
    Check(Window(kDashboard)->DockNode && Window(kDashboard)->DockNode->IsFloatingNode(), "with its tab strip");

    // The same for a window only the setting lets out: outside at the end of
    // a session, outside at the start of the next (over the main window too).
    gui::SetEveryWindowMayLeave(true);
    for (int i = 0; i < 2; ++i) Frame();
    gui::MoveWindowOut(kProperties);
    for (int i = 0; i < 4; ++i) Frame();
    for (int i = 0; i < 3; ++i) Frame(nullptr, &inside);  // over the main window
    Check(gui::WindowIsOutside(kProperties), "properties outside, over the main window");
    const std::string ini2 = ImGui::SaveIniSettingsToMemory();
    Teardown();
    Setup();
    gui::SetEveryWindowMayLeave(true);
    ImGui::LoadIniSettingsFromMemory(ini2.c_str(), ini2.size());
    for (int i = 0; i < 4; ++i) Frame();
    Check(gui::WindowIsOutside(kProperties), "properties outside again in the next session");
    gui::SetEveryWindowMayLeave(false);

    Teardown();
    std::cout << "separate windows contract: ok\n";
    return 0;
}
