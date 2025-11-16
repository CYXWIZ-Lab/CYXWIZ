#include "main_window.h"
#include "node_editor.h"
#include "console.h"
#include "viewport.h"
#include "properties.h"
#include "panels/toolbar.h"
#include "panels/asset_browser.h"
#include "panels/training_dashboard.h"
#include "panels/plot_test_control.h"
#include "panels/command_window.h"
#include "panels/script_editor.h"
#include "../scripting/scripting_engine.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <cyxwiz/cyxwiz.h>
#include <spdlog/spdlog.h>

namespace gui {

MainWindow::MainWindow()
    : show_about_dialog_(false), show_demo_window_(false), first_time_layout_(true) {

    // Original panels
    node_editor_ = std::make_unique<NodeEditor>();
    console_ = std::make_unique<Console>();
    viewport_ = std::make_unique<Viewport>();
    properties_ = std::make_unique<Properties>();

    // Initialize scripting engine (shared resource)
    scripting_engine_ = std::make_shared<scripting::ScriptingEngine>();

    // New panel system
    toolbar_ = std::make_unique<cyxwiz::ToolbarPanel>();
    asset_browser_ = std::make_unique<cyxwiz::AssetBrowserPanel>();
    training_dashboard_ = std::make_unique<cyxwiz::TrainingDashboardPanel>();
    plot_test_control_ = std::make_unique<cyxwiz::PlotTestControlPanel>();
    command_window_ = std::make_unique<cyxwiz::CommandWindowPanel>();
    script_editor_ = std::make_unique<cyxwiz::ScriptEditorPanel>();

    // Set scripting engine for command window and script editor
    command_window_->SetScriptingEngine(scripting_engine_);
    script_editor_->SetScriptingEngine(scripting_engine_);

    // Set up callbacks in the toolbar
    toolbar_->SetResetLayoutCallback([this]() {
        this->ResetDockLayout();
    });

    toolbar_->SetTogglePlotTestControlCallback([this]() {
        if (plot_test_control_) {
            plot_test_control_->Toggle();
        }
    });

    spdlog::info("MainWindow initialized with docking layout system");
}

MainWindow::~MainWindow() = default;

void MainWindow::ResetDockLayout() {
    // Force rebuild of the docking layout
    first_time_layout_ = true;
    spdlog::info("Dock layout reset requested");
}

void MainWindow::Render() {
    RenderDockSpace();

    // Render new panel system - Toolbar (replaces old menu bar)
    if (toolbar_) toolbar_->Render();

    // Render new panels
    if (asset_browser_) asset_browser_->Render();
    if (training_dashboard_) training_dashboard_->Render();
    if (plot_test_control_) plot_test_control_->Render();
    if (command_window_) command_window_->Render();
    if (script_editor_) script_editor_->Render();

    // Render original panels
    if (node_editor_) node_editor_->Render();
    if (console_) console_->Render();
    if (viewport_) viewport_->Render();
    if (properties_) properties_->Render();

    if (show_about_dialog_) {
        ShowAboutDialog();
    }

    if (show_demo_window_) {
        ImGui::ShowDemoWindow(&show_demo_window_);
    }
}

void MainWindow::RenderDockSpace() {
    static bool opt_fullscreen = false;  // Set to false to show native title bar with window controls
    static bool opt_padding = false;
    static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;

    // Always get viewport to fill the available space
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    if (opt_fullscreen) {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    }

    if (!opt_padding)
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("DockSpace", nullptr, window_flags);
    if (!opt_padding)
        ImGui::PopStyleVar();

    if (opt_fullscreen)
        ImGui::PopStyleVar(2);

    // DockSpace
    ImGuiIO& io = ImGui::GetIO();
    if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable) {
        ImGuiID dockspace_id = ImGui::GetID("CyxWizDockSpace");

        // Build the initial layout ONLY if no saved layout exists
        if (first_time_layout_) {
            // Check if dockspace node already exists (loaded from imgui.ini)
            ImGuiDockNode* node = ImGui::DockBuilderGetNode(dockspace_id);
            if (node == nullptr || !node->IsSplitNode()) {
                // No saved layout, build default
                BuildInitialDockLayout();
            }
            first_time_layout_ = false;
        }

        ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
    }

    ImGui::End();
}

void MainWindow::BuildInitialDockLayout() {
    ImGuiID dockspace_id = ImGui::GetID("CyxWizDockSpace");

    // Clear any existing layout
    ImGui::DockBuilderRemoveNode(dockspace_id);
    ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->Size);

    // Create the docking layout structure
    // We'll build this structure:
    // - Left column (Asset Browser) ~15%
    // - Center (Node Editor) - main workspace
    // - Right column (Properties) ~20%
    // - Bottom section split into:
    //   - Bottom-left (Console) ~70% of bottom
    //   - Bottom-right (Inspector/Training Dashboard) ~30% of bottom
    //   - Bottom-bottom (Profiler/Viewport)

    ImGuiID dock_id_left = 0;
    ImGuiID dock_id_right = 0;
    ImGuiID dock_id_center = dockspace_id;
    ImGuiID dock_id_bottom = 0;
    ImGuiID dock_id_bottom_left = 0;
    ImGuiID dock_id_bottom_right = 0;
    ImGuiID dock_id_bottom_bottom = 0;

    // Split left side for Asset Browser (15% width)
    dock_id_left = ImGui::DockBuilderSplitNode(dock_id_center, ImGuiDir_Left, 0.15f, nullptr, &dock_id_center);

    // Split right side for Properties (20% width)
    dock_id_right = ImGui::DockBuilderSplitNode(dock_id_center, ImGuiDir_Right, 0.25f, nullptr, &dock_id_center);

    // Split bottom for Console area (25% height of remaining)
    dock_id_bottom = ImGui::DockBuilderSplitNode(dock_id_center, ImGuiDir_Down, 0.30f, nullptr, &dock_id_center);

    // Split bottom section horizontally: left (Console) and right (Inspector)
    dock_id_bottom_right = ImGui::DockBuilderSplitNode(dock_id_bottom, ImGuiDir_Right, 0.25f, nullptr, &dock_id_bottom_left);

    // Split bottom-left further for Profiler/Viewport
    dock_id_bottom_bottom = ImGui::DockBuilderSplitNode(dock_id_bottom_left, ImGuiDir_Down, 0.40f, nullptr, &dock_id_bottom_left);

    // Dock windows to their designated areas
    // Window names must EXACTLY match the names in ImGui::Begin() calls in each panel
    ImGui::DockBuilderDockWindow("Asset Browser", dock_id_left);
    ImGui::DockBuilderDockWindow("Node Editor", dock_id_center);
    ImGui::DockBuilderDockWindow("Script Editor", dock_id_center); // Tabbed with Node Editor
    ImGui::DockBuilderDockWindow("Properties", dock_id_right);
    ImGui::DockBuilderDockWindow("Console", dock_id_bottom_left);
    ImGui::DockBuilderDockWindow("Command Window", dock_id_bottom_left); // Tabbed with Console
    ImGui::DockBuilderDockWindow("Training Dashboard", dock_id_bottom_right);
    ImGui::DockBuilderDockWindow("Viewport", dock_id_bottom_bottom);

    // Finish the docking layout
    ImGui::DockBuilderFinish(dockspace_id);

    spdlog::info("Initial dock layout built successfully");
    spdlog::info("  - Left: Asset Browser (15%)");
    spdlog::info("  - Center: Node Editor (main workspace)");
    spdlog::info("  - Right: Properties (20%)");
    spdlog::info("  - Bottom-Left: Console");
    spdlog::info("  - Bottom-Right: Training Dashboard (Inspector)");
    spdlog::info("  - Bottom-Bottom: Viewport (Profiler Timeline)");
}

void MainWindow::ShowAboutDialog() {
    if (!ImGui::Begin("About CyxWiz Engine", &show_about_dialog_, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::End();
        return;
    }

    ImGui::Text("CyxWiz Engine");
    ImGui::Text("Version: %s", cyxwiz::GetVersionString());
    ImGui::Separator();
    ImGui::Text("Decentralized ML Compute Platform");
    ImGui::Text("Built with ImGui, ArrayFire, and gRPC");
    ImGui::Separator();
    if (ImGui::Button("OK")) {
        show_about_dialog_ = false;
    }

    ImGui::End();
}

} // namespace gui
