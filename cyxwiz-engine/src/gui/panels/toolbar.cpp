#include "toolbar.h"
#include "plot_window.h"
#include "plot_test_panel.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <fstream>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#endif

namespace cyxwiz {

ToolbarPanel::ToolbarPanel()
    : Panel("Toolbar", true)
    , show_new_project_dialog_(false)
    , show_about_dialog_(false)
{
    memset(project_name_buffer_, 0, sizeof(project_name_buffer_));
    memset(project_path_buffer_, 0, sizeof(project_path_buffer_));
}

void ToolbarPanel::Render() {
    if (!visible_) return;

    if (ImGui::BeginMainMenuBar()) {
        RenderFileMenu();
        RenderEditMenu();
        RenderViewMenu();
        RenderNodesMenu();
        RenderTrainMenu();
        RenderDatasetMenu();
        RenderScriptMenu();
        RenderPlotsMenu();
        RenderDeployMenu();
        RenderHelpMenu();

        ImGui::EndMainMenuBar();
    }

    // Render all plot windows
    for (auto& plot_window : plot_windows_) {
        if (plot_window) {
            plot_window->Render();
        }
    }

    // Render dialogs if open
    if (show_new_project_dialog_) {
        ImGui::OpenPopup("New Project");
        if (ImGui::BeginPopupModal("New Project", &show_new_project_dialog_, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Create a new CyxWiz project");
            ImGui::Separator();

            ImGui::InputText("Project Name", project_name_buffer_, sizeof(project_name_buffer_));

            ImGui::InputText("Location", project_path_buffer_, sizeof(project_path_buffer_));
            ImGui::SameLine();
            if (ImGui::Button("Browse...")) {
                std::string selected_folder = OpenFolderDialog();
                if (!selected_folder.empty()) {
                    strncpy(project_path_buffer_, selected_folder.c_str(), sizeof(project_path_buffer_) - 1);
                    project_path_buffer_[sizeof(project_path_buffer_) - 1] = '\0';
                }
            }

            ImGui::Separator();

            if (ImGui::Button("Create", ImVec2(120, 0))) {
                std::string proj_name = project_name_buffer_;
                std::string proj_path = project_path_buffer_;

                if (!proj_name.empty() && !proj_path.empty()) {
                    if (CreateProjectOnDisk(proj_name, proj_path)) {
                        spdlog::info("Project created: {}/{}", proj_path, proj_name);
                        show_new_project_dialog_ = false;
                        // Clear buffers
                        memset(project_name_buffer_, 0, sizeof(project_name_buffer_));
                        memset(project_path_buffer_, 0, sizeof(project_path_buffer_));
                    }
                } else {
                    spdlog::warn("Project name and location are required");
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                show_new_project_dialog_ = false;
                // Clear buffers
                memset(project_name_buffer_, 0, sizeof(project_name_buffer_));
                memset(project_path_buffer_, 0, sizeof(project_path_buffer_));
            }

            ImGui::EndPopup();
        }
    }

    if (show_about_dialog_) {
        ImGui::OpenPopup("About CyxWiz");
        if (ImGui::BeginPopupModal("About CyxWiz", &show_about_dialog_, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("CyxWiz Engine");
            ImGui::Text("Version 0.1.0");
            ImGui::Separator();
            ImGui::Text("Decentralized ML Compute Platform");
            ImGui::Text("Built with C++, ImGui, ArrayFire, and Solana");
            ImGui::Separator();

            if (ImGui::Button("OK", ImVec2(120, 0))) {
                show_about_dialog_ = false;
            }

            ImGui::EndPopup();
        }
    }
}

void ToolbarPanel::RenderFileMenu() {
    if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("New Project", "Ctrl+N")) {
            show_new_project_dialog_ = true;
        }

        if (ImGui::MenuItem("Open Project...", "Ctrl+O")) {
            // TODO: Open file dialog
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Save", "Ctrl+S")) {
            // TODO: Save current project
        }

        if (ImGui::MenuItem("Save As...", "Ctrl+Shift+S")) {
            // TODO: Save project with new name
        }

        ImGui::Separator();

        if (ImGui::BeginMenu("Import")) {
            if (ImGui::MenuItem("ONNX Model...")) {
                // TODO: Import ONNX
            }
            if (ImGui::MenuItem("PyTorch Model...")) {
                // TODO: Import PyTorch
            }
            if (ImGui::MenuItem("TensorFlow Model...")) {
                // TODO: Import TF
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Export")) {
            if (ImGui::MenuItem("ONNX...")) {
                // TODO: Export ONNX
            }
            if (ImGui::MenuItem("GGUF...")) {
                // TODO: Export GGUF
            }
            if (ImGui::MenuItem("LoRA Adapter...")) {
                // TODO: Export LoRA
            }
            ImGui::EndMenu();
        }

        ImGui::Separator();

        if (ImGui::BeginMenu("Recent Projects")) {
            ImGui::MenuItem("project1.cyxwiz");
            ImGui::MenuItem("project2.cyxwiz");
            ImGui::EndMenu();
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Exit", "Alt+F4")) {
            // TODO: Exit application
        }

        ImGui::EndMenu();
    }
}

void ToolbarPanel::RenderEditMenu() {
    if (ImGui::BeginMenu("Edit")) {
        if (ImGui::MenuItem("Undo", "Ctrl+Z")) {
            // TODO: Undo
        }

        if (ImGui::MenuItem("Redo", "Ctrl+Y")) {
            // TODO: Redo
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Cut", "Ctrl+X")) {
            // TODO: Cut
        }

        if (ImGui::MenuItem("Copy", "Ctrl+C")) {
            // TODO: Copy
        }

        if (ImGui::MenuItem("Paste", "Ctrl+V")) {
            // TODO: Paste
        }

        if (ImGui::MenuItem("Delete", "Delete")) {
            // TODO: Delete
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Select All", "Ctrl+A")) {
            // TODO: Select all
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Preferences...")) {
            // TODO: Open preferences
        }

        ImGui::EndMenu();
    }
}

void ToolbarPanel::RenderViewMenu() {
    if (ImGui::BeginMenu("View")) {
        // Panel visibility toggles
        // TODO: Get references to actual panels and toggle their visibility
        ImGui::MenuItem("Asset Browser", nullptr, true);
        ImGui::MenuItem("Node Editor", nullptr, true);
        ImGui::MenuItem("Properties", nullptr, true);
        ImGui::MenuItem("Console", nullptr, true);
        ImGui::MenuItem("Training Dashboard", nullptr, true);
        ImGui::MenuItem("Viewport (Profiler)", nullptr, true);

        ImGui::Separator();

        if (ImGui::BeginMenu("Layout")) {
            if (ImGui::MenuItem("Reset to Default Layout")) {
                // Call the reset layout callback if set
                if (reset_layout_callback_) {
                    reset_layout_callback_();
                }
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Save Layout...")) {
                // TODO: Save current layout to file
            }
            if (ImGui::MenuItem("Load Layout...")) {
                // TODO: Load saved layout from file
            }
            ImGui::EndMenu();
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Fullscreen", "F11")) {
            // TODO: Toggle fullscreen mode
        }

        ImGui::EndMenu();
    }
}

void ToolbarPanel::RenderNodesMenu() {
    if (ImGui::BeginMenu("Nodes")) {
        if (ImGui::BeginMenu("Add Layer")) {
            if (ImGui::MenuItem("Dense/Linear")) {
                // TODO: Add dense layer
            }
            if (ImGui::MenuItem("Convolutional")) {
                // TODO: Add conv layer
            }
            if (ImGui::MenuItem("Pooling")) {
                // TODO: Add pooling layer
            }
            if (ImGui::MenuItem("Dropout")) {
                // TODO: Add dropout
            }
            if (ImGui::MenuItem("Batch Normalization")) {
                // TODO: Add batch norm
            }
            if (ImGui::MenuItem("Attention")) {
                // TODO: Add attention
            }
            ImGui::EndMenu();
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Group Selected", "Ctrl+G")) {
            // TODO: Group nodes
        }

        if (ImGui::MenuItem("Ungroup", "Ctrl+Shift+G")) {
            // TODO: Ungroup nodes
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Duplicate", "Ctrl+D")) {
            // TODO: Duplicate selected
        }

        if (ImGui::MenuItem("Delete Selected", "Delete")) {
            // TODO: Delete selected nodes
        }

        ImGui::EndMenu();
    }
}

void ToolbarPanel::RenderTrainMenu() {
    if (ImGui::BeginMenu("Train")) {
        if (ImGui::MenuItem("Start Training", "F5")) {
            // TODO: Start training
        }

        if (ImGui::MenuItem("Pause", "F6")) {
            // TODO: Pause training
        }

        if (ImGui::MenuItem("Stop", "Shift+F5")) {
            // TODO: Stop training
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Training Settings...")) {
            // TODO: Open training settings
        }

        if (ImGui::MenuItem("Optimizer Settings...")) {
            // TODO: Open optimizer settings
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Resume from Checkpoint...")) {
            // TODO: Resume training
        }

        ImGui::EndMenu();
    }
}

void ToolbarPanel::RenderDatasetMenu() {
    if (ImGui::BeginMenu("Dataset")) {
        if (ImGui::MenuItem("Import Dataset...")) {
            // TODO: Import dataset
        }

        if (ImGui::MenuItem("Create Custom Dataset...")) {
            // TODO: Create dataset
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Preprocess...")) {
            // TODO: Preprocess dataset
        }

        if (ImGui::MenuItem("Tokenize...")) {
            // TODO: Tokenize dataset
        }

        if (ImGui::MenuItem("Augment...")) {
            // TODO: Data augmentation
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Dataset Statistics")) {
            // TODO: Show statistics
        }

        ImGui::EndMenu();
    }
}

void ToolbarPanel::RenderScriptMenu() {
    if (ImGui::BeginMenu("Script")) {
        if (ImGui::MenuItem("Open Python Console", "F12")) {
            // TODO: Show Python console
        }

        ImGui::Separator();

        if (ImGui::MenuItem("New Script...")) {
            // TODO: Create new script
        }

        if (ImGui::MenuItem("Run Script...", "Ctrl+R")) {
            // TODO: Run script
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Script Editor...")) {
            // TODO: Open script editor
        }

        ImGui::EndMenu();
    }
}

void ToolbarPanel::RenderPlotsMenu() {
    if (ImGui::BeginMenu("Plots")) {
        // 2D Plots submenu
        if (ImGui::BeginMenu("2D Plots")) {
            if (ImGui::MenuItem("Line Plot")) {
                CreatePlotWindow("Line Plot", PlotWindow::PlotWindowType::Line2D);
            }
            if (ImGui::MenuItem("Scatter Plot")) {
                CreatePlotWindow("Scatter Plot", PlotWindow::PlotWindowType::Scatter2D);
            }
            if (ImGui::MenuItem("Bar Chart")) {
                CreatePlotWindow("Bar Chart", PlotWindow::PlotWindowType::Bar);
            }
            if (ImGui::MenuItem("Stem Plot")) {
                CreatePlotWindow("Stem Plot", PlotWindow::PlotWindowType::Stem);
            }
            if (ImGui::MenuItem("Stair Plot")) {
                CreatePlotWindow("Stair Plot", PlotWindow::PlotWindowType::Stair);
            }
            if (ImGui::MenuItem("Histogram")) {
                CreatePlotWindow("Histogram", PlotWindow::PlotWindowType::Histogram);
            }
            if (ImGui::MenuItem("Pie Chart")) {
                CreatePlotWindow("Pie Chart", PlotWindow::PlotWindowType::PieChart);
            }
            if (ImGui::MenuItem("Box Plot")) {
                CreatePlotWindow("Box Plot", PlotWindow::PlotWindowType::BoxPlot);
            }
            ImGui::EndMenu();
        }

        // 3D Plots submenu
        if (ImGui::BeginMenu("3D Plots")) {
            if (ImGui::MenuItem("Surface Plot")) {
                CreatePlotWindow("Surface Plot", PlotWindow::PlotWindowType::Surface3D);
            }
            if (ImGui::MenuItem("3D Scatter")) {
                CreatePlotWindow("3D Scatter", PlotWindow::PlotWindowType::Scatter3D);
            }
            if (ImGui::MenuItem("3D Line")) {
                CreatePlotWindow("3D Line", PlotWindow::PlotWindowType::Line3D);
            }
            ImGui::EndMenu();
        }

        // Specialized submenu
        if (ImGui::BeginMenu("Specialized")) {
            if (ImGui::MenuItem("Polar Plot")) {
                CreatePlotWindow("Polar Plot", PlotWindow::PlotWindowType::Polar);
            }
            if (ImGui::MenuItem("Heatmap")) {
                CreatePlotWindow("Heatmap", PlotWindow::PlotWindowType::Heatmap);
            }
            if (ImGui::MenuItem("Parametric Plot")) {
                CreatePlotWindow("Parametric Plot", PlotWindow::PlotWindowType::Parametric);
            }
            ImGui::EndMenu();
        }

        ImGui::Separator();

        // Plot Test Panel (toggle visibility)
        if (ImGui::MenuItem("Plot Test Panel")) {
            if (plot_test_panel_) {
                plot_test_panel_->SetVisible(!plot_test_panel_->IsVisible());
                spdlog::info("Plot Test Panel visibility toggled to: {}", plot_test_panel_->IsVisible());
            } else {
                spdlog::warn("Plot Test Panel not available - needs to be set via SetPlotTestPanel()");
            }
        }

        ImGui::EndMenu();
    }
}

void ToolbarPanel::RenderDeployMenu() {
    if (ImGui::BeginMenu("Deploy")) {
        if (ImGui::BeginMenu("Export Model")) {
            if (ImGui::MenuItem("ONNX Format")) {
                // TODO: Export ONNX
            }
            if (ImGui::MenuItem("GGUF Format")) {
                // TODO: Export GGUF
            }
            if (ImGui::MenuItem("LoRA Adapter")) {
                // TODO: Export LoRA
            }
            if (ImGui::MenuItem("Safetensors")) {
                // TODO: Export safetensors
            }
            ImGui::EndMenu();
        }

        ImGui::Separator();

        if (ImGui::BeginMenu("Quantize")) {
            if (ImGui::MenuItem("INT8")) {
                // TODO: Quantize INT8
            }
            if (ImGui::MenuItem("INT4")) {
                // TODO: Quantize INT4
            }
            if (ImGui::MenuItem("FP16")) {
                // TODO: Quantize FP16
            }
            ImGui::EndMenu();
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Deploy to Server Node...")) {
            // TODO: Deploy to node
        }

        if (ImGui::MenuItem("Publish to Marketplace...")) {
            // TODO: Publish model
        }

        ImGui::EndMenu();
    }
}

void ToolbarPanel::RenderHelpMenu() {
    if (ImGui::BeginMenu("Help")) {
        if (ImGui::MenuItem("Documentation", "F1")) {
            // TODO: Open docs
        }

        if (ImGui::MenuItem("Keyboard Shortcuts")) {
            // TODO: Show shortcuts
        }

        if (ImGui::MenuItem("API Reference")) {
            // TODO: Open API docs
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Report Issue...")) {
            // TODO: Open issue tracker
        }

        if (ImGui::MenuItem("Check for Updates...")) {
            // TODO: Check updates
        }

        ImGui::Separator();

        if (ImGui::MenuItem("About CyxWiz")) {
            show_about_dialog_ = true;
        }

        ImGui::EndMenu();
    }
}

std::string ToolbarPanel::OpenFolderDialog() {
#ifdef _WIN32
    BROWSEINFO bi = { 0 };
    bi.lpszTitle = "Select Project Location";
    bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;

    LPITEMIDLIST pidl = SHBrowseForFolder(&bi);
    if (pidl != nullptr) {
        char path[MAX_PATH];
        if (SHGetPathFromIDList(pidl, path)) {
            CoTaskMemFree(pidl);
            return std::string(path);
        }
        CoTaskMemFree(pidl);
    }
    return "";
#else
    // For Linux/Mac, we'd need a different implementation
    // For now, return empty string
    spdlog::warn("Folder dialog not implemented for this platform");
    return "";
#endif
}

bool ToolbarPanel::CreateProjectOnDisk(const std::string& project_name, const std::string& project_path) {
    try {
        namespace fs = std::filesystem;

        // Create project directory
        fs::path project_dir = fs::path(project_path) / project_name;
        if (fs::exists(project_dir)) {
            spdlog::error("Project directory already exists: {}", project_dir.string());
            return false;
        }

        fs::create_directories(project_dir);
        spdlog::info("Created project directory: {}", project_dir.string());

        // Create subdirectories
        fs::create_directories(project_dir / "models");
        fs::create_directories(project_dir / "datasets");
        fs::create_directories(project_dir / "scripts");
        fs::create_directories(project_dir / "checkpoints");
        fs::create_directories(project_dir / "exports");

        // Create project file (.cyxwiz)
        fs::path project_file = project_dir / (project_name + ".cyxwiz");
        std::ofstream file(project_file);
        if (!file.is_open()) {
            spdlog::error("Failed to create project file: {}", project_file.string());
            return false;
        }

        // Write basic project configuration (JSON format)
        file << "{\n";
        file << "  \"name\": \"" << project_name << "\",\n";
        file << "  \"version\": \"0.1.0\",\n";
        file << "  \"created\": \"" << std::time(nullptr) << "\",\n";
        file << "  \"description\": \"CyxWiz Machine Learning Project\",\n";
        file << "  \"models\": [],\n";
        file << "  \"datasets\": [],\n";
        file << "  \"scripts\": []\n";
        file << "}\n";
        file.close();

        spdlog::info("Created project file: {}", project_file.string());

        // Create README.md
        fs::path readme = project_dir / "README.md";
        std::ofstream readme_file(readme);
        if (readme_file.is_open()) {
            readme_file << "# " << project_name << "\n\n";
            readme_file << "CyxWiz Machine Learning Project\n\n";
            readme_file << "## Directory Structure\n\n";
            readme_file << "- `models/` - Neural network models\n";
            readme_file << "- `datasets/` - Training and validation datasets\n";
            readme_file << "- `scripts/` - Python scripts for training and inference\n";
            readme_file << "- `checkpoints/` - Training checkpoints\n";
            readme_file << "- `exports/` - Exported models (ONNX, GGUF, etc.)\n";
            readme_file.close();
        }

        return true;

    } catch (const std::exception& e) {
        spdlog::error("Failed to create project: {}", e.what());
        return false;
    }
}

void ToolbarPanel::CreatePlotWindow(const std::string& title, PlotWindow::PlotWindowType type) {
    // Create new plot window with auto-generated data
    auto plot_window = std::make_shared<PlotWindow>(title, type, true);
    plot_windows_.push_back(plot_window);
    spdlog::info("Created new plot window: {}", title);
}

} // namespace cyxwiz
