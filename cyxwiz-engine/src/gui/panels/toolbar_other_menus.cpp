#include "toolbar.h"
#include "plot_window.h"
#include "../theme.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <fstream>
#include <cstring>
#include "../dock_style.h"
#include "../../core/project_manager.h"
#include "../icons.h"

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#include <commdlg.h>
#endif

namespace cyxwiz {

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
            if (import_dataset_callback_) {
                import_dataset_callback_();
            }
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
        // Test Control - Interactive option
        if (ImGui::MenuItem("Test Control")) {
            if (toggle_plot_test_control_callback_) {
                toggle_plot_test_control_callback_();
            }
        }

        ImGui::Separator();
        ImGui::TextDisabled("Available Plot Types (View Only)");
        ImGui::Separator();

        // Basic 2D Plots (from cheatsheet)
        if (ImGui::BeginMenu("Basic 2D", false)) { ImGui::EndMenu(); }
        ImGui::Indent();
        ImGui::TextDisabled("plot() - Line plot");
        ImGui::TextDisabled("scatter() - Scatter plot");
        ImGui::TextDisabled("bar() / barh() - Bar chart");
        ImGui::TextDisabled("imshow() - Image display");
        ImGui::TextDisabled("contour() / contourf() - Contour plot");
        ImGui::TextDisabled("pcolormesh() - Pseudocolor plot");
        ImGui::TextDisabled("quiver() - Vector field");
        ImGui::TextDisabled("pie() - Pie chart");
        ImGui::TextDisabled("fill_between() - Filled area");
        ImGui::Unindent();

        ImGui::Separator();

        // Advanced 2D Plots
        if (ImGui::BeginMenu("Advanced 2D", false)) { ImGui::EndMenu(); }
        ImGui::Indent();
        ImGui::TextDisabled("step() - Step plot");
        ImGui::TextDisabled("boxplot() - Box plot");
        ImGui::TextDisabled("errorbar() - Error bar plot");
        ImGui::TextDisabled("hist() - Histogram");
        ImGui::TextDisabled("violinplot() - Violin plot");
        ImGui::TextDisabled("barbs() - Barbs plot");
        ImGui::TextDisabled("eventplot() - Event plot");
        ImGui::TextDisabled("hexbin() - Hexagonal binning");
        ImGui::Unindent();

        ImGui::Separator();

        // 3D Plots
        if (ImGui::BeginMenu("3D Plots", false)) { ImGui::EndMenu(); }
        ImGui::Indent();
        ImGui::TextDisabled("plot3D() - 3D line plot");
        ImGui::TextDisabled("scatter3D() - 3D scatter");
        ImGui::TextDisabled("plot_surface() - Surface plot");
        ImGui::TextDisabled("plot_wireframe() - Wireframe");
        ImGui::TextDisabled("contour3D() - 3D contour");
        ImGui::Unindent();

        ImGui::Separator();

        // Polar Plots
        if (ImGui::BeginMenu("Polar", false)) { ImGui::EndMenu(); }
        ImGui::Indent();
        ImGui::TextDisabled("polar() - Polar plot");
        ImGui::Unindent();

        ImGui::Separator();

        // Statistical Plots
        if (ImGui::BeginMenu("Statistical", false)) { ImGui::EndMenu(); }
        ImGui::Indent();
        ImGui::TextDisabled("hist() - Histogram");
        ImGui::TextDisabled("boxplot() - Box plot");
        ImGui::TextDisabled("violinplot() - Violin plot");
        ImGui::TextDisabled("kde plot - Density estimation");
        ImGui::Unindent();

        ImGui::Separator();

        // Specialized Plots
        if (ImGui::BeginMenu("Specialized", false)) { ImGui::EndMenu(); }
        ImGui::Indent();
        ImGui::TextDisabled("heatmap - Heat map");
        ImGui::TextDisabled("streamplot() - Stream plot");
        ImGui::TextDisabled("specgram() - Spectrogram");
        ImGui::TextDisabled("spy() - Sparse matrix viz");
        ImGui::Unindent();

        ImGui::EndMenu();
    }
}

void ToolbarPanel::RenderDeployMenu() {
    if (ImGui::BeginMenu("Deploy")) {
        if (ImGui::MenuItem("Connect to Server...")) {
            if (connect_to_server_callback_) {
                connect_to_server_callback_();
            }
        }

        ImGui::Separator();

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

void ToolbarPanel::CreatePlotWindow(const std::string& title, PlotWindow::PlotWindowType type) {
    // Create new plot window with auto-generated data
    auto plot_window = std::make_shared<PlotWindow>(title, type, true);
    plot_windows_.push_back(plot_window);
    spdlog::info("Created new plot window: {}", title);
}

} // namespace cyxwiz
