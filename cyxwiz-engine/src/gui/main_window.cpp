#include "main_window.h"
#include "node_editor.h"
#include "console.h"
#include "viewport.h"
#include "properties.h"
#include "dock_style.h"
#include "icons.h"
#include "panels/dataset_panel.h"
#include "panels/toolbar.h"
#include "panels/asset_browser.h"
#include "panels/training_dashboard.h"
#include "panels/training_plot_panel.h"
#include "panels/plot_test_control.h"
#include "panels/command_window.h"
#include "panels/script_editor.h"
#include "panels/table_viewer.h"
#include "panels/connection_dialog.h"
#include "panels/job_status_panel.h"
#include "panels/p2p_training_panel.h"
#include "panels/wallet_panel.h"
#include "../scripting/scripting_engine.h"
#include "../scripting/startup_script_manager.h"
#include "../network/job_manager.h"
#include "../core/project_manager.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <cyxwiz/cyxwiz.h>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <fstream>

namespace gui {

MainWindow::MainWindow()
    : show_about_dialog_(false), show_demo_window_(false), first_time_layout_(true) {

    // Original panels
    node_editor_ = std::make_unique<NodeEditor>();
    console_ = std::make_unique<Console>();
    viewport_ = std::make_unique<Viewport>();
    properties_ = std::make_unique<Properties>();
    dataset_panel_ = std::make_unique<DatasetPanel>();

    // Initialize scripting engine (shared resource)
    scripting_engine_ = std::make_shared<scripting::ScriptingEngine>();

    // New panel system
    toolbar_ = std::make_unique<cyxwiz::ToolbarPanel>();
    asset_browser_ = std::make_unique<cyxwiz::AssetBrowserPanel>();
    // training_dashboard_ = std::make_unique<cyxwiz::TrainingDashboardPanel>();  // Removed - merged into TrainingPlotPanel
    training_plot_panel_ = std::make_unique<cyxwiz::TrainingPlotPanel>();  // Now named "Training Dashboard"
    plot_test_control_ = std::make_unique<cyxwiz::PlotTestControlPanel>();
    command_window_ = std::make_unique<cyxwiz::CommandWindowPanel>();
    script_editor_ = std::make_unique<cyxwiz::ScriptEditorPanel>();
    table_viewer_ = std::make_unique<cyxwiz::TableViewerPanel>();
    job_status_panel_ = std::make_unique<cyxwiz::JobStatusPanel>();
    p2p_training_panel_ = std::make_unique<cyxwiz::P2PTrainingPanel>();
    wallet_panel_ = std::make_unique<gui::WalletPanel>();

    // Set scripting engine for command window and script editor
    command_window_->SetScriptingEngine(scripting_engine_);
    script_editor_->SetScriptingEngine(scripting_engine_);

    // Expose TrainingPlotPanel to Python scripts through the scripting engine
    // This avoids DLL boundary issues by using pybind11 directly
    if (scripting_engine_) {
        scripting_engine_->RegisterTrainingDashboard(training_plot_panel_.get());
    }

    // Connect Viewport to TrainingPlotPanel for real-time metrics display
    viewport_->SetTrainingPanel(training_plot_panel_.get());

    // Connect script editor to command window for output display
    script_editor_->SetCommandWindow(command_window_.get());

    // Connect Node Editor to Script Editor for code generation output
    node_editor_->SetScriptEditor(script_editor_.get());

    // Connect Node Editor to Properties panel for node selection display
    node_editor_->SetPropertiesPanel(properties_.get());

    // Set up callbacks in the toolbar
    toolbar_->SetResetLayoutCallback([this]() {
        this->ResetDockLayout();
    });

    toolbar_->SetSaveLayoutCallback([this]() {
        this->SaveLayout();
    });

    toolbar_->SetSaveProjectSettingsCallback([this]() {
        this->SaveProjectSettings();
    });

    toolbar_->SetTogglePlotTestControlCallback([this]() {
        if (plot_test_control_) {
            plot_test_control_->Toggle();
        }
    });

    // Register callbacks with ProjectManager for project lifecycle events
    cyxwiz::ProjectManager::Instance().SetOnProjectOpened([this](const std::string& project_root) {
        this->OnProjectOpened(project_root);
    });

    cyxwiz::ProjectManager::Instance().SetOnProjectClosed([this](const std::string& project_root) {
        this->OnProjectClosed(project_root);
    });

    // Set up New Script callback (called after script is created to refresh asset browser)
    toolbar_->SetNewScriptCallback([this]() {
        if (asset_browser_) {
            asset_browser_->Refresh();
        }
    });

    // Set up Open Script in Editor callback (called with file path to open)
    toolbar_->SetOpenScriptInEditorCallback([this](const std::string& file_path) {
        if (script_editor_) {
            script_editor_->OpenFile(file_path);
            spdlog::info("Opened script in editor: {}", file_path);
        }
    });

    // Set up Save All callback
    toolbar_->SetSaveAllCallback([this]() {
        SaveAllFiles();
        spdlog::info("All files saved via toolbar");
    });

    // Set up Exit callback (called when user confirms exit)
    toolbar_->SetExitCallback([this]() {
        if (exit_request_callback_) {
            exit_request_callback_();
        }
    });

    // Set up Has Unsaved Changes callback (called to check if confirmation dialog is needed)
    toolbar_->SetHasUnsavedChangesCallback([this]() -> bool {
        return HasUnsavedFiles();
    });

    // Set up Edit menu callbacks for Find/Replace
    toolbar_->SetFindCallback([this](const std::string& text, bool case_sensitive, bool whole_word, bool use_regex) {
        if (script_editor_) {
            script_editor_->FindInEditor(text, case_sensitive, whole_word, use_regex);
        }
    });

    toolbar_->SetReplaceCallback([this](const std::string& find_text, const std::string& replace_text,
                                        bool case_sensitive, bool whole_word, bool use_regex) {
        if (script_editor_) {
            script_editor_->Replace(find_text, replace_text, case_sensitive, whole_word, use_regex);
        }
    });

    toolbar_->SetReplaceAllCallback([this](const std::string& find_text, const std::string& replace_text,
                                           bool case_sensitive, bool whole_word, bool use_regex) {
        if (script_editor_) {
            int count = script_editor_->ReplaceAll(find_text, replace_text, case_sensitive, whole_word, use_regex);
            spdlog::info("Replaced {} occurrences", count);
        }
    });

    // Set up edit operation callbacks
    toolbar_->SetUndoCallback([this]() {
        if (script_editor_) {
            script_editor_->Undo();
        }
    });

    toolbar_->SetRedoCallback([this]() {
        if (script_editor_) {
            script_editor_->Redo();
        }
    });

    toolbar_->SetCutCallback([this]() {
        if (script_editor_) {
            script_editor_->Cut();
        }
    });

    toolbar_->SetCopyCallback([this]() {
        if (script_editor_) {
            script_editor_->Copy();
        }
    });

    toolbar_->SetPasteCallback([this]() {
        if (script_editor_) {
            script_editor_->Paste();
        }
    });

    toolbar_->SetDeleteCallback([this]() {
        if (script_editor_) {
            script_editor_->Delete();
        }
    });

    toolbar_->SetSelectAllCallback([this]() {
        if (script_editor_) {
            script_editor_->SelectAll();
        }
    });

    // Set up comment toggle callbacks
    toolbar_->SetToggleLineCommentCallback([this]() {
        if (script_editor_) {
            script_editor_->ToggleLineComment();
        }
    });

    toolbar_->SetToggleBlockCommentCallback([this]() {
        if (script_editor_) {
            script_editor_->ToggleBlockComment();
        }
    });

    // Set up Go to Line callback
    toolbar_->SetGoToLineCallback([this](int line) {
        if (script_editor_) {
            script_editor_->GoToLine(line);
        }
    });

    // Set up line operation callbacks
    toolbar_->SetDuplicateLineCallback([this]() {
        if (script_editor_) {
            script_editor_->DuplicateLine();
        }
    });

    toolbar_->SetMoveLineUpCallback([this]() {
        if (script_editor_) {
            script_editor_->MoveLineUp();
        }
    });

    toolbar_->SetMoveLineDownCallback([this]() {
        if (script_editor_) {
            script_editor_->MoveLineDown();
        }
    });

    toolbar_->SetIndentCallback([this]() {
        if (script_editor_) {
            script_editor_->Indent();
        }
    });

    toolbar_->SetOutdentCallback([this]() {
        if (script_editor_) {
            script_editor_->Outdent();
        }
    });

    // Set up text transformation callbacks
    toolbar_->SetTransformUppercaseCallback([this]() {
        if (script_editor_) {
            script_editor_->TransformToUppercase();
        }
    });

    toolbar_->SetTransformLowercaseCallback([this]() {
        if (script_editor_) {
            script_editor_->TransformToLowercase();
        }
    });

    toolbar_->SetTransformTitleCaseCallback([this]() {
        if (script_editor_) {
            script_editor_->TransformToTitleCase();
        }
    });

    // Set up sort and join lines callbacks
    toolbar_->SetSortLinesAscCallback([this]() {
        if (script_editor_) {
            script_editor_->SortLinesAscending();
        }
    });

    toolbar_->SetSortLinesDescCallback([this]() {
        if (script_editor_) {
            script_editor_->SortLinesDescending();
        }
    });

    toolbar_->SetJoinLinesCallback([this]() {
        if (script_editor_) {
            script_editor_->JoinLines();
        }
    });

    // Set up editor settings callbacks (Preferences -> Script Editor synchronization)
    toolbar_->SetEditorThemeCallback([this](int theme_index) {
        if (script_editor_) {
            script_editor_->SetTheme(theme_index);
            spdlog::info("Editor theme changed to index {}", theme_index);
        }
    });

    toolbar_->SetEditorTabSizeCallback([this](int tab_size) {
        if (script_editor_) {
            script_editor_->SetTabSize(tab_size);
            spdlog::info("Editor tab size changed to {}", tab_size);
        }
    });

    toolbar_->SetEditorFontScaleCallback([this](float scale) {
        if (script_editor_) {
            script_editor_->SetFontScale(scale);
            spdlog::info("Editor font scale changed to {}", scale);
        }
    });

    toolbar_->SetEditorShowWhitespaceCallback([this](bool show) {
        if (script_editor_) {
            script_editor_->SetShowWhitespace(show);
            spdlog::info("Editor show whitespace changed to {}", show);
        }
    });

    toolbar_->SetEditorWordWrapCallback([this](bool wrap) {
        if (script_editor_) {
            script_editor_->SetWordWrap(wrap);
            spdlog::info("Editor word wrap changed to {}", wrap);
        }
    });

    toolbar_->SetEditorAutoIndentCallback([this](bool indent) {
        if (script_editor_) {
            script_editor_->SetAutoIndent(indent);
            spdlog::info("Editor auto indent changed to {}", indent);
        }
    });

    // Initialize toolbar editor settings from script editor's current values
    if (script_editor_) {
        toolbar_->SetEditorTheme(script_editor_->GetThemeIndex());
        toolbar_->SetEditorTabSize(script_editor_->GetTabSize());
        toolbar_->SetEditorFontScale(script_editor_->GetFontScale());
        toolbar_->SetEditorShowWhitespace(script_editor_->GetShowWhitespace());
        toolbar_->SetEditorWordWrap(script_editor_->GetWordWrap());
        toolbar_->SetEditorAutoIndent(script_editor_->GetAutoIndent());

        // Set up callback for when settings change in Script Editor (View menu)
        // This syncs changes back to the Preferences dialog
        script_editor_->SetOnSettingsChangedCallback([this]() {
            if (toolbar_ && script_editor_) {
                toolbar_->SetEditorTheme(script_editor_->GetThemeIndex());
                toolbar_->SetEditorTabSize(script_editor_->GetTabSize());
                toolbar_->SetEditorFontScale(script_editor_->GetFontScale());
                toolbar_->SetEditorShowWhitespace(script_editor_->GetShowWhitespace());
                toolbar_->SetEditorWordWrap(script_editor_->GetWordWrap());
                toolbar_->SetEditorAutoIndent(script_editor_->GetAutoIndent());
            }
        });
    }

    // Set up asset browser double-click callback to open files in script editor
    asset_browser_->SetOnAssetDoubleClick([this](const cyxwiz::AssetBrowserPanel::AssetItem& item) {
        if (!item.is_directory && script_editor_) {
            // Get file extension
            std::string ext = std::filesystem::path(item.absolute_path).extension().string();

            // Open text-based files in script editor
            if (ext == ".py" || ext == ".cyx" || ext == ".txt" ||
                ext == ".json" || ext == ".md" || ext == ".csv" ||
                ext == ".yaml" || ext == ".yml" || ext == ".toml" ||
                ext == ".ini" || ext == ".cfg" || ext == ".conf") {
                script_editor_->OpenFile(item.absolute_path);
                spdlog::info("Opened file in script editor: {}", item.name);
            }
        }
    });

    // Initialize startup script manager
    startup_script_manager_ = std::make_unique<scripting::StartupScriptManager>(scripting_engine_);

    // Load and execute startup scripts
    if (startup_script_manager_->LoadConfig()) {
        spdlog::info("Executing startup scripts...");
        startup_script_manager_->ExecuteAll(command_window_.get());
    } else {
        spdlog::debug("No startup scripts configured or startup_scripts.txt not found");
    }

    // Install custom dock node handler for Unreal-style tabs
    DockStyle::InstallCustomHandler();

    // Register panels with sidebar for hide/unhide toggles
    RegisterPanelsWithSidebar();

    spdlog::info("MainWindow initialized with docking layout system");
}

MainWindow::~MainWindow() = default;

void MainWindow::SetNetworkComponents(network::GRPCClient* client, network::JobManager* job_manager) {
    // Store job manager reference
    job_manager_ = job_manager;

    // Create connection dialog with network components
    connection_dialog_ = std::make_unique<cyxwiz::ConnectionDialog>(client, job_manager);

    // Set JobManager for JobStatusPanel
    if (job_status_panel_) {
        job_status_panel_->SetJobManager(job_manager);
    }

    // Set JobManager for DatasetPanel (enables training job submission)
    if (dataset_panel_) {
        dataset_panel_->SetJobManager(job_manager);

        // Set TrainingPlotPanel for local training visualization
        dataset_panel_->SetTrainingPlotPanel(training_plot_panel_.get());

        // Set callback to start P2P monitoring when training starts
        dataset_panel_->SetTrainingStartCallback([this](const std::string& job_id) {
            StartJobMonitoring(job_id);
        });
    }

    // Set up callback in toolbar to show connection dialog
    if (toolbar_) {
        toolbar_->SetConnectToServerCallback([this]() {
            if (connection_dialog_) {
                connection_dialog_->Show();
            }
        });
    }

    spdlog::info("Network components set in MainWindow");
}

void MainWindow::StartJobMonitoring(const std::string& job_id) {
    spdlog::info("Starting P2P monitoring for job: {}", job_id);

    monitoring_job_id_ = job_id;

    // Note: The P2PClient won't be available immediately - it's created when
    // the job is assigned to a node. For now, we'll start monitoring with
    // the job_id, and the P2PTrainingPanel will connect when the client is ready.

    if (p2p_training_panel_) {
        // Start monitoring - this sets up the panel state
        // The P2PClient will be connected later when available
        p2p_training_panel_->StartMonitoring(job_id, "");
        p2p_training_panel_->Show();

        spdlog::info("P2PTrainingPanel now monitoring job: {}", job_id);
    }

    // Try to get the P2PClient if it's already available
    if (job_manager_) {
        auto p2p_client = job_manager_->GetP2PClient(job_id);
        if (p2p_client && p2p_training_panel_) {
            p2p_training_panel_->SetP2PClient(p2p_client);
            spdlog::info("P2PClient connected to monitoring panel");
        }
    }
}

bool MainWindow::IsScriptRunning() const {
    if (scripting_engine_) {
        return scripting_engine_->IsScriptRunning();
    }
    return false;
}

void MainWindow::StopRunningScript() {
    if (scripting_engine_) {
        scripting_engine_->StopScript();
    }
}

bool MainWindow::HasUnsavedFiles() const {
    if (script_editor_) {
        return script_editor_->HasUnsavedFiles();
    }
    return false;
}

std::vector<std::string> MainWindow::GetUnsavedFileNames() const {
    if (script_editor_) {
        return script_editor_->GetUnsavedFileNames();
    }
    return {};
}

void MainWindow::SaveAllFiles() {
    if (script_editor_) {
        script_editor_->SaveAllFiles();
    }
}

void MainWindow::ResetDockLayout() {
    // Force rebuild of the docking layout
    first_time_layout_ = true;
    spdlog::info("Dock layout reset requested");
}

void MainWindow::Render() {
    // Handle global keyboard shortcuts
    HandleGlobalShortcuts();

    // Check if we need to connect P2PClient to monitoring panel
    if (!monitoring_job_id_.empty() && job_manager_ && p2p_training_panel_) {
        auto p2p_client = job_manager_->GetP2PClient(monitoring_job_id_);
        if (p2p_client) {
            // Check if P2PTrainingPanel doesn't have the client yet
            // (SetP2PClient is idempotent so we can call it multiple times safely)
            p2p_training_panel_->SetP2PClient(p2p_client);
        }
    }

    RenderDockSpace();

    // Render Unreal-style sidebar for panel toggles
    RenderSidebar();

    // Render new panel system - Toolbar (replaces old menu bar)
    if (toolbar_) toolbar_->Render();

    // Render new panels
    if (asset_browser_) asset_browser_->Render();
    // if (training_dashboard_) training_dashboard_->Render();  // Removed - merged into TrainingPlotPanel
    if (training_plot_panel_) training_plot_panel_->Render();  // Now "Training Dashboard"
    if (plot_test_control_) plot_test_control_->Render();
    if (command_window_) command_window_->Render();
    if (script_editor_) script_editor_->Render();
    if (table_viewer_) table_viewer_->Render();
    if (connection_dialog_) connection_dialog_->Render();
    if (job_status_panel_) job_status_panel_->Render();
    if (p2p_training_panel_) p2p_training_panel_->Render();
    if (wallet_panel_) wallet_panel_->Render();

    // Render original panels
    if (node_editor_) node_editor_->Render();
    if (console_) console_->Render();
    if (viewport_) viewport_->Render();
    if (properties_) properties_->Render();
    if (dataset_panel_) dataset_panel_->Render();

    if (show_about_dialog_) {
        ShowAboutDialog();
    }

    if (show_demo_window_) {
        ImGui::ShowDemoWindow(&show_demo_window_);
    }
}

// Helper function to draw active tab indicator on a dock node
static void DrawDockNodeActiveTabIndicator(ImGuiDockNode* node) {
    if (!node) return;

    // Recursively process child nodes
    if (node->ChildNodes[0]) DrawDockNodeActiveTabIndicator(node->ChildNodes[0]);
    if (node->ChildNodes[1]) DrawDockNodeActiveTabIndicator(node->ChildNodes[1]);

    // Only process leaf nodes with tab bars
    if (!node->TabBar || node->Windows.Size == 0) return;

    ImGuiTabBar* tab_bar = node->TabBar;
    const DockTabStyle& style = GetDockStyle().GetStyle();

    // Only draw if we have an active tab and indicator is enabled
    if (!style.show_active_indicator || tab_bar->VisibleTabId == 0) return;

    ImGuiTabItem* active_tab = ImGui::TabBarFindTabByID(tab_bar, tab_bar->VisibleTabId);
    if (!active_tab) return;

    // Get the draw list for the host window
    ImGuiWindow* host_window = node->HostWindow;
    if (!host_window) return;

    ImDrawList* draw_list = host_window->DrawList;

    // Calculate tab position relative to the tab bar
    ImVec2 tab_bar_min = tab_bar->BarRect.Min;
    ImVec2 tab_min = ImVec2(tab_bar_min.x + active_tab->Offset, tab_bar_min.y);
    ImVec2 tab_max = ImVec2(tab_min.x + active_tab->Width, tab_bar_min.y + style.active_indicator_height);

    // Draw the indicator line at the TOP of the active tab
    ImU32 indicator_color = ImGui::ColorConvertFloat4ToU32(style.active_indicator_color);
    draw_list->AddRectFilled(tab_min, tab_max, indicator_color);
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

        // Draw Unreal-style active tab indicators on all dock nodes
        ImGuiDockNode* root_node = ImGui::DockBuilderGetNode(dockspace_id);
        if (root_node) {
            DrawDockNodeActiveTabIndicator(root_node);
        }
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

void MainWindow::RegisterPanelsWithSidebar() {
    DockStyle& dock_style = GetDockStyle();

    // Clear any existing registrations
    dock_style.ClearPanels();

    // Register main panels with FontAwesome icons

    // Main editing panels
    if (node_editor_) {
        dock_style.RegisterPanel("Node Editor", ICON_FA_DIAGRAM_PROJECT, node_editor_->GetVisiblePtr());
    }
    if (script_editor_) {
        dock_style.RegisterPanel("Script Editor", ICON_FA_CODE, script_editor_->GetVisiblePtr());
    }

    // Side panels
    if (asset_browser_) {
        dock_style.RegisterPanel("Asset Browser", ICON_FA_IMAGES, asset_browser_->GetVisiblePtr());
    }
    if (properties_) {
        dock_style.RegisterPanel("Properties", ICON_FA_SLIDERS, properties_->GetVisiblePtr());
    }

    // Bottom panels
    if (console_) {
        dock_style.RegisterPanel("Console", ICON_FA_TERMINAL, console_->GetVisiblePtr());
    }
    if (command_window_) {
        dock_style.RegisterPanel("Command", ICON_FA_CHEVRON_RIGHT, command_window_->GetVisiblePtr());
    }
    if (training_plot_panel_) {
        dock_style.RegisterPanel("Training", ICON_FA_CHART_LINE, training_plot_panel_->GetVisiblePtr());
    }
    if (viewport_) {
        dock_style.RegisterPanel("Viewport", ICON_FA_CUBES, viewport_->GetVisiblePtr());
    }

    // Additional panels (less commonly used)
    if (dataset_panel_) {
        dock_style.RegisterPanel("Dataset", ICON_FA_DATABASE, dataset_panel_->GetVisiblePtr());
    }
    if (job_status_panel_) {
        dock_style.RegisterPanel("Jobs", ICON_FA_LIST_CHECK, job_status_panel_->GetVisiblePtr());
    }
    if (wallet_panel_) {
        dock_style.RegisterPanel("Wallet", ICON_FA_WALLET, wallet_panel_->GetVisiblePtr());
    }

    spdlog::info("Registered {} panels with sidebar", dock_style.GetPanels().size());
}

void MainWindow::RenderSidebar() {
    // Render the Unreal-style sidebar (auto-hides, appears on hover)
    GetDockStyle().RenderSidebarToggles();
}


void MainWindow::HandleGlobalShortcuts() {
    ImGuiIO& io = ImGui::GetIO();

    bool ctrl = io.KeyCtrl;
    bool shift = io.KeyShift;
    bool alt = io.KeyAlt;

    // Don't capture shortcuts if a dialog is already open
    if (toolbar_) {
        if (toolbar_->IsFindDialogOpen() || toolbar_->IsReplaceDialogOpen() ||
            toolbar_->IsFindInFilesDialogOpen() || toolbar_->IsReplaceInFilesDialogOpen()) {
            return;
        }
    }

    // Find (Ctrl+F)
    if (ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_F)) {
        if (toolbar_) {
            toolbar_->OpenFindDialog();
            spdlog::info("Opened Find dialog via Ctrl+F");
        }
    }

    // Replace (Ctrl+H)
    if (ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_H)) {
        if (toolbar_) {
            toolbar_->OpenReplaceDialog();
            spdlog::info("Opened Replace dialog via Ctrl+H");
        }
    }

    // Find in Files (Ctrl+Shift+F)
    if (ctrl && shift && !alt && ImGui::IsKeyPressed(ImGuiKey_F)) {
        if (toolbar_) {
            toolbar_->OpenFindInFilesDialog();
            spdlog::info("Opened Find in Files dialog via Ctrl+Shift+F");
        }
    }

    // Replace in Files (Ctrl+Shift+H)
    if (ctrl && shift && !alt && ImGui::IsKeyPressed(ImGuiKey_H)) {
        if (toolbar_) {
            toolbar_->OpenReplaceInFilesDialog();
            spdlog::info("Opened Replace in Files dialog via Ctrl+Shift+H");
        }
    }

    // Toggle Line Comment (Ctrl+/)
    if (ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_Slash)) {
        if (script_editor_) {
            script_editor_->ToggleLineComment();
            spdlog::info("Toggled line comment via Ctrl+/");
        }
    }

    // Toggle Block Comment (Shift+Alt+A)
    if (!ctrl && shift && alt && ImGui::IsKeyPressed(ImGuiKey_A)) {
        if (script_editor_) {
            script_editor_->ToggleBlockComment();
            spdlog::info("Toggled block comment via Shift+Alt+A");
        }
    }

    // Go to Line (Ctrl+G) - Opens dialog in toolbar
    if (ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_G)) {
        // Note: This opens the Go to Line dialog in the toolbar
        // The toolbar handles the dialog rendering
    }

    // Duplicate Line (Ctrl+D)
    if (ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_D)) {
        if (script_editor_) {
            script_editor_->DuplicateLine();
            spdlog::info("Duplicated line via Ctrl+D");
        }
    }

    // Move Line Up (Alt+Up)
    if (!ctrl && !shift && alt && ImGui::IsKeyPressed(ImGuiKey_UpArrow)) {
        if (script_editor_) {
            script_editor_->MoveLineUp();
            spdlog::info("Moved line up via Alt+Up");
        }
    }

    // Move Line Down (Alt+Down)
    if (!ctrl && !shift && alt && ImGui::IsKeyPressed(ImGuiKey_DownArrow)) {
        if (script_editor_) {
            script_editor_->MoveLineDown();
            spdlog::info("Moved line down via Alt+Down");
        }
    }

    // Join Lines (Ctrl+J)
    if (ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_J)) {
        if (script_editor_) {
            script_editor_->JoinLines();
            spdlog::info("Joined lines via Ctrl+J");
        }
    }
}

// ============================================================================
// Project Settings Persistence
// ============================================================================

void MainWindow::SaveLayout() {
    // Save to the default imgui.ini in the executable directory
    // This ensures consistent layout across all projects
    ImGui::SaveIniSettingsToDisk("imgui.ini");
    spdlog::info("Saved layout to imgui.ini");
}

void MainWindow::LoadLayout() {
    // Layout is loaded automatically from imgui.ini by ImGui
    // This function is kept for API compatibility but doesn't need to do anything
    // Per-project layouts are disabled to avoid dock corruption issues
}

void MainWindow::SaveProjectSettings() {
    auto& pm = cyxwiz::ProjectManager::Instance();
    if (!pm.HasActiveProject()) {
        spdlog::warn("Cannot save project settings: no active project");
        return;
    }

    // Get current editor settings from script editor
    cyxwiz::EditorSettings& settings = pm.GetConfig().editor_settings;
    if (script_editor_) {
        settings.theme = script_editor_->GetThemeIndex();
        settings.font_scale = script_editor_->GetFontScale();
        settings.tab_size = script_editor_->GetTabSize();
        settings.show_whitespace = script_editor_->GetShowWhitespace();
    }

    // Save layout file
    SaveLayout();

    // Save project file (includes editor settings)
    pm.SaveProject();
    spdlog::info("Saved project settings");
}

void MainWindow::LoadProjectSettings() {
    auto& pm = cyxwiz::ProjectManager::Instance();
    if (!pm.HasActiveProject()) {
        return;
    }

    const cyxwiz::EditorSettings& settings = pm.GetConfig().editor_settings;

    // Apply editor settings to script editor
    if (script_editor_) {
        script_editor_->SetTheme(settings.theme);
        script_editor_->SetFontScale(settings.font_scale);
        script_editor_->SetTabSize(settings.tab_size);
        script_editor_->SetShowWhitespace(settings.show_whitespace);
    }

    // Sync settings to toolbar/preferences
    if (toolbar_) {
        toolbar_->SetEditorTheme(settings.theme);
        toolbar_->SetEditorTabSize(settings.tab_size);
        toolbar_->SetEditorFontScale(settings.font_scale);
        toolbar_->SetEditorShowWhitespace(settings.show_whitespace);
    }

    // Load layout file
    LoadLayout();

    // Restore open scripts
    const auto& open_scripts = pm.GetConfig().open_scripts;
    for (const auto& script_path : open_scripts) {
        if (script_editor_ && std::filesystem::exists(script_path)) {
            script_editor_->OpenFile(script_path);
        }
    }

    spdlog::info("Loaded project settings (theme={}, font_scale={:.1f}, tab_size={})",
                 settings.theme, settings.font_scale, settings.tab_size);
}

void MainWindow::OnProjectOpened(const std::string& project_root) {
    spdlog::info("Project opened: {}", project_root);

    // Load project settings and layout
    LoadProjectSettings();

    // Set project root and refresh asset browser to show project files
    if (asset_browser_) {
        asset_browser_->SetProjectRoot(project_root);
        asset_browser_->Refresh();
    }
}

void MainWindow::OnProjectClosed(const std::string& project_root) {
    spdlog::info("Project closed: {}", project_root);

    // Save current settings before closing
    // Note: This is called AFTER project state is cleared, so we can't save here
    // Settings should be saved before CloseProject() is called
}

} // namespace gui
