#include "tutorial_system.h"
#include "../icons.h"
#include <imgui.h>
#include <imgui_internal.h>
#include <spdlog/spdlog.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <cmath>

namespace cyxwiz {

TutorialSystem& TutorialSystem::Instance() {
    static TutorialSystem instance;
    return instance;
}

TutorialSystem::TutorialSystem() {
    RegisterBuiltInTutorials();
    LoadProgress();
}

void TutorialSystem::StartTutorial(const std::string& tutorial_id) {
    auto it = tutorial_index_.find(tutorial_id);
    if (it == tutorial_index_.end()) {
        spdlog::error("Tutorial not found: {}", tutorial_id);
        return;
    }

    current_tutorial_ = &tutorials_[it->second];
    current_step_index_ = 0;
    step_start_time_ = ImGui::GetTime();
    is_active_ = true;
    show_browser_ = false;

    spdlog::info("Started tutorial: {} ({} steps)", current_tutorial_->name, current_tutorial_->steps.size());
}

void TutorialSystem::StopTutorial() {
    if (current_tutorial_) {
        spdlog::info("Stopped tutorial: {}", current_tutorial_->name);
    }
    is_active_ = false;
    current_tutorial_ = nullptr;
    current_step_index_ = 0;
}

void TutorialSystem::NextStep() {
    if (!is_active_ || !current_tutorial_) return;

    current_step_index_++;
    step_start_time_ = ImGui::GetTime();

    if (current_step_index_ >= static_cast<int>(current_tutorial_->steps.size())) {
        // Tutorial complete
        if (std::find(completed_tutorials_.begin(), completed_tutorials_.end(),
            current_tutorial_->id) == completed_tutorials_.end()) {
            completed_tutorials_.push_back(current_tutorial_->id);
            current_tutorial_->completed = true;
            SaveProgress();
        }

        spdlog::info("Tutorial completed: {}", current_tutorial_->name);
        StopTutorial();
    }
}

void TutorialSystem::PreviousStep() {
    if (!is_active_ || !current_tutorial_) return;

    if (current_step_index_ > 0) {
        current_step_index_--;
        step_start_time_ = ImGui::GetTime();
    }
}

void TutorialSystem::SkipTutorial() {
    spdlog::info("Tutorial skipped");
    StopTutorial();
}

void TutorialSystem::Render() {
    // Update animation
    highlight_pulse_ += ImGui::GetIO().DeltaTime * 3.0f;
    if (highlight_pulse_ > 6.28318f) highlight_pulse_ -= 6.28318f;

    // Render tutorial browser if requested
    if (show_browser_) {
        ShowTutorialBrowser();
    }

    // Render active tutorial
    if (is_active_ && current_tutorial_) {
        RenderOverlay();
    }
}

void TutorialSystem::RenderOverlay() {
    if (!current_tutorial_ || current_step_index_ >= static_cast<int>(current_tutorial_->steps.size())) {
        return;
    }

    const TutorialStep& step = current_tutorial_->steps[current_step_index_];

    // Get target window rect
    ImRect target_rect;
    bool has_target = false;

    if (!step.target_element.empty() && window_rect_callback_) {
        target_rect = window_rect_callback_(step.target_element);
        has_target = (target_rect.GetWidth() > 0 && target_rect.GetHeight() > 0);
    }

    // Apply offset and size override
    if (has_target) {
        target_rect.Min.x += step.highlight_offset.x;
        target_rect.Min.y += step.highlight_offset.y;
        if (step.highlight_size.x > 0) {
            target_rect.Max.x = target_rect.Min.x + step.highlight_size.x;
        }
        if (step.highlight_size.y > 0) {
            target_rect.Max.y = target_rect.Min.y + step.highlight_size.y;
        }
    }

    // No dark overlay - keep everything visible!
    // Just highlight the focus area with a bright border/glow
    ImGuiIO& io = ImGui::GetIO();

    if (has_target) {
        // Draw highlight border around the target area (no darkening)
        RenderHighlight(target_rect);
    }
    // If no target, just show the tooltip without any overlay

    // Render step tooltip
    RenderStepTooltip(step, has_target ? target_rect : ImRect(io.DisplaySize.x * 0.3f,
        io.DisplaySize.y * 0.3f, io.DisplaySize.x * 0.7f, io.DisplaySize.y * 0.5f));

    // Render progress bar
    RenderProgressBar();

    // Check completion condition
    if (step.completion_condition && step.completion_condition()) {
        NextStep();
    }

    // Auto-advance for non-interactive steps
    if (!step.requires_interaction && step.delay_seconds > 0) {
        float elapsed = ImGui::GetTime() - step_start_time_;
        if (elapsed >= step.delay_seconds) {
            NextStep();
        }
    }
}

void TutorialSystem::RenderHighlight(const ImRect& target_rect) {
    ImDrawList* draw_list = ImGui::GetForegroundDrawList();

    // Pulsing glow effect - more pronounced since no dark overlay
    float pulse_alpha = 0.7f + 0.3f * sinf(highlight_pulse_);
    float pulse_size = 3.0f + 5.0f * sinf(highlight_pulse_);

    // Bright outer glow - more visible layers
    for (int i = 5; i >= 0; i--) {
        float expand = pulse_size + i * 3.0f;
        float alpha = pulse_alpha * (1.0f - i * 0.15f);
        ImVec4 glow_color = highlight_color_;
        glow_color.w = alpha;

        draw_list->AddRect(
            ImVec2(target_rect.Min.x - expand, target_rect.Min.y - expand),
            ImVec2(target_rect.Max.x + expand, target_rect.Max.y + expand),
            ImGui::ColorConvertFloat4ToU32(glow_color),
            6.0f, 0, 2.5f);
    }

    // Solid inner border - thicker and brighter
    draw_list->AddRect(target_rect.Min, target_rect.Max,
        IM_COL32(0, 220, 255, 255), 6.0f, 0, 4.0f);

    // Add a subtle filled highlight behind the target area
    draw_list->AddRectFilled(
        ImVec2(target_rect.Min.x - 2, target_rect.Min.y - 2),
        ImVec2(target_rect.Max.x + 2, target_rect.Max.y + 2),
        IM_COL32(0, 180, 220, 30),  // Very subtle cyan tint
        6.0f);
}

void TutorialSystem::RenderStepTooltip(const TutorialStep& step, const ImRect& target_rect) {
    ImGuiIO& io = ImGui::GetIO();

    // Calculate tooltip dimensions
    const float tooltip_width = 380.0f;
    const float tooltip_padding = 20.0f;
    const float tooltip_margin = 25.0f;

    // Calculate position
    ImVec2 tooltip_pos;
    float space_below = io.DisplaySize.y - target_rect.Max.y;
    float space_above = target_rect.Min.y;

    // Estimate height
    ImVec2 desc_size = ImGui::CalcTextSize(step.description.c_str(), nullptr, false, tooltip_width - 2 * tooltip_padding);
    float estimated_height = tooltip_padding * 2 + desc_size.y + 180.0f;

    if (space_below > estimated_height + tooltip_margin) {
        tooltip_pos = ImVec2(target_rect.GetCenter().x - tooltip_width / 2, target_rect.Max.y + tooltip_margin);
    } else if (space_above > estimated_height + tooltip_margin) {
        tooltip_pos = ImVec2(target_rect.GetCenter().x - tooltip_width / 2, target_rect.Min.y - estimated_height - tooltip_margin);
    } else {
        tooltip_pos = ImVec2(io.DisplaySize.x / 2 - tooltip_width / 2, io.DisplaySize.y / 2 - estimated_height / 2);
    }

    // Clamp to screen
    tooltip_pos.x = std::max(10.0f, std::min(tooltip_pos.x, io.DisplaySize.x - tooltip_width - 10));
    tooltip_pos.y = std::max(10.0f, std::min(tooltip_pos.y, io.DisplaySize.y - estimated_height - 10));

    // Create a real ImGui window with full styling - this handles input properly
    ImGui::SetNextWindowPos(tooltip_pos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(tooltip_width, 0), ImGuiCond_Always);  // Auto height
    ImGui::SetNextWindowBgAlpha(1.0f);

    // Push bright, opaque styling - solid dark background for readability
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.10f, 0.10f, 0.14f, 1.0f));  // Solid dark background
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.0f, 0.86f, 1.0f, 1.0f));  // Bright cyan
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.47f, 0.86f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.27f, 0.59f, 0.98f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.12f, 0.35f, 0.7f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Separator, ImVec4(0.3f, 0.3f, 0.4f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(tooltip_padding, tooltip_padding));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 12.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 3.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 6));

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize;

    if (ImGui::Begin("##TutorialTooltip", nullptr, window_flags)) {
        // Step counter - bright cyan
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "Step %d of %d",
            current_step_index_ + 1, static_cast<int>(current_tutorial_->steps.size()));

        ImGui::Spacing();

        // Title - pure white, larger
        ImGui::PushFont(ImGui::GetFont());  // Could use a bold font here
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 1.0f, 1.0f), "%s", step.title.c_str());
        ImGui::PopFont();

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Description - white, wrapped
        ImGui::PushTextWrapPos(tooltip_width - tooltip_padding * 2);
        ImGui::TextColored(ImVec4(0.95f, 0.95f, 0.95f, 1.0f), "%s", step.description.c_str());
        ImGui::PopTextWrapPos();

        // Action hint - bright yellow
        if (!step.action_hint.empty()) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(1.0f, 0.9f, 0.3f, 1.0f), "%s %s",
                ICON_FA_CIRCLE_INFO, step.action_hint.c_str());
        }

        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Navigation buttons
        float button_width = 90.0f;

        if (current_step_index_ > 0) {
            if (ImGui::Button(ICON_FA_ARROW_LEFT " Previous", ImVec2(button_width, 0))) {
                PreviousStep();
            }
            ImGui::SameLine();
        }

        if (!step.completion_condition) {
            if (current_step_index_ < static_cast<int>(current_tutorial_->steps.size()) - 1) {
                if (ImGui::Button("Next " ICON_FA_ARROW_RIGHT, ImVec2(button_width, 0))) {
                    NextStep();
                }
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.7f, 0.3f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.8f, 0.4f, 1.0f));
                if (ImGui::Button(ICON_FA_CHECK " Complete", ImVec2(button_width, 0))) {
                    NextStep();
                }
                ImGui::PopStyleColor(2);
            }
            ImGui::SameLine();
        }

        // Skip button - right aligned
        float avail = ImGui::GetContentRegionAvail().x;
        if (avail > button_width) {
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + avail - button_width);
        }
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.2f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.7f, 0.3f, 0.3f, 1.0f));
        if (ImGui::Button(ICON_FA_XMARK " Skip", ImVec2(button_width, 0))) {
            SkipTutorial();
        }
        ImGui::PopStyleColor(2);
    }
    ImGui::End();

    ImGui::PopStyleVar(5);
    ImGui::PopStyleColor(7);

    // Get the actual window position and size after rendering
    ImGuiWindow* tooltip_window = ImGui::FindWindowByName("##TutorialTooltip");
    if (tooltip_window) {
        ImVec2 win_pos = tooltip_window->Pos;
        ImVec2 win_size = tooltip_window->Size;
        ImVec2 win_max(win_pos.x + win_size.x, win_pos.y + win_size.y);

        // Draw solid backing rectangle behind the window to prevent bleed-through
        // Use the background draw list so it's behind everything
        ImDrawList* bg_draw = ImGui::GetBackgroundDrawList();
        bg_draw->AddRectFilled(
            ImVec2(win_pos.x - 2, win_pos.y - 2),
            ImVec2(win_max.x + 2, win_max.y + 2),
            IM_COL32(20, 20, 28, 255),  // Solid dark background
            14.0f);  // Rounded corners

        // Draw bright cyan glow border on foreground
        ImDrawList* fg_draw = ImGui::GetForegroundDrawList();
        for (int i = 2; i >= 0; i--) {
            float expand = i * 2.0f;
            float alpha = 0.8f - i * 0.2f;
            fg_draw->AddRect(
                ImVec2(win_pos.x - expand, win_pos.y - expand),
                ImVec2(win_max.x + expand, win_max.y + expand),
                IM_COL32(0, 220, 255, (int)(alpha * 255)),
                12.0f + expand, 0, 2.0f);
        }
    }
}

void TutorialSystem::RenderProgressBar() {
    if (!current_tutorial_ || current_tutorial_->steps.empty()) return;

    ImGuiIO& io = ImGui::GetIO();
    ImDrawList* draw_list = ImGui::GetForegroundDrawList();

    float progress = static_cast<float>(current_step_index_) / current_tutorial_->steps.size();
    float bar_width = 300.0f;
    float bar_height = 6.0f;
    float bar_x = (io.DisplaySize.x - bar_width) / 2;
    float bar_y = io.DisplaySize.y - 40.0f;

    // Background
    draw_list->AddRectFilled(
        ImVec2(bar_x, bar_y),
        ImVec2(bar_x + bar_width, bar_y + bar_height),
        ImGui::ColorConvertFloat4ToU32(ImVec4(0.2f, 0.2f, 0.2f, 0.8f)),
        bar_height / 2);

    // Progress
    draw_list->AddRectFilled(
        ImVec2(bar_x, bar_y),
        ImVec2(bar_x + bar_width * progress, bar_y + bar_height),
        ImGui::ColorConvertFloat4ToU32(highlight_color_),
        bar_height / 2);

    // Step indicators
    for (size_t i = 0; i < current_tutorial_->steps.size(); i++) {
        float dot_x = bar_x + (bar_width * (i + 0.5f) / current_tutorial_->steps.size());
        float dot_y = bar_y + bar_height / 2;
        float dot_radius = 4.0f;

        ImU32 dot_color;
        if (static_cast<int>(i) < current_step_index_) {
            dot_color = ImGui::ColorConvertFloat4ToU32(ImVec4(0.3f, 0.8f, 0.3f, 1.0f)); // Completed
        } else if (static_cast<int>(i) == current_step_index_) {
            dot_color = ImGui::ColorConvertFloat4ToU32(highlight_color_); // Current
        } else {
            dot_color = ImGui::ColorConvertFloat4ToU32(ImVec4(0.5f, 0.5f, 0.5f, 1.0f)); // Pending
        }

        draw_list->AddCircleFilled(ImVec2(dot_x, dot_y), dot_radius, dot_color);
    }
}

void TutorialSystem::ShowTutorialBrowser() {
    ImGui::SetNextWindowSize(ImVec2(500, 400), ImGuiCond_FirstUseEver);

    // Push explicit styling for visibility
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.12f, 0.12f, 0.16f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.2f, 0.4f, 0.6f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.3f, 0.5f, 0.7f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.5f, 0.8f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.6f, 0.9f, 1.0f));

    if (!ImGui::Begin("Interactive Tutorials", &show_browser_)) {
        ImGui::End();
        ImGui::PopStyleColor(6);
        return;
    }

    // Header with icon
    ImGui::TextColored(ImVec4(0.0f, 0.9f, 1.0f, 1.0f), ICON_FA_BOOK " Available Tutorials");
    ImGui::Separator();
    ImGui::Spacing();

    // Group tutorials by category
    std::map<std::string, std::vector<Tutorial*>> by_category;
    for (auto& tutorial : tutorials_) {
        by_category[tutorial.category].push_back(&tutorial);
    }

    // Show count for debugging
    if (tutorials_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "No tutorials registered!");
    }

    for (auto& [category, tuts] : by_category) {
        // Category header
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.9f, 0.5f, 1.0f));
        bool open = ImGui::CollapsingHeader(category.c_str(), ImGuiTreeNodeFlags_DefaultOpen);
        ImGui::PopStyleColor();

        if (open) {
            ImGui::Indent(10.0f);
            for (auto* tutorial : tuts) {
                ImGui::PushID(tutorial->id.c_str());

                bool is_completed = std::find(completed_tutorials_.begin(), completed_tutorials_.end(),
                    tutorial->id) != completed_tutorials_.end();

                // Status icon
                if (is_completed) {
                    ImGui::TextColored(ImVec4(0.3f, 0.9f, 0.3f, 1.0f), ICON_FA_CIRCLE_CHECK);
                } else {
                    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), ICON_FA_CIRCLE);
                }
                ImGui::SameLine();

                // Tutorial name - bright white
                ImGui::TextColored(ImVec4(1.0f, 1.0f, 1.0f, 1.0f), "%s", tutorial->name.c_str());
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.7f, 1.0f), "(%zu steps)", tutorial->steps.size());

                // Start button
                ImGui::SameLine(ImGui::GetWindowWidth() - 90);
                if (ImGui::Button("Start", ImVec2(70, 0))) {
                    StartTutorial(tutorial->id);
                }

                // Description on hover
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("%s", tutorial->description.c_str());
                }

                ImGui::PopID();
            }
            ImGui::Unindent(10.0f);
            ImGui::Spacing();
        }
    }

    ImGui::End();
    ImGui::PopStyleColor(6);
}

bool TutorialSystem::IsTutorialComplete(const std::string& tutorial_id) const {
    return std::find(completed_tutorials_.begin(), completed_tutorials_.end(), tutorial_id)
        != completed_tutorials_.end();
}

int TutorialSystem::GetTotalSteps() const {
    return current_tutorial_ ? static_cast<int>(current_tutorial_->steps.size()) : 0;
}

void TutorialSystem::RegisterTutorial(Tutorial tutorial) {
    tutorial_index_[tutorial.id] = tutorials_.size();
    tutorials_.push_back(std::move(tutorial));
}

void TutorialSystem::SaveProgress() {
    try {
        nlohmann::json j;
        j["completed"] = completed_tutorials_;
        j["welcome_shown"] = has_shown_welcome_;

        std::ofstream file("tutorial_progress.json");
        if (file.is_open()) {
            file << j.dump(2);
            file.close();
            spdlog::debug("Tutorial progress saved");
        }
    } catch (const std::exception& e) {
        spdlog::error("Failed to save tutorial progress: {}", e.what());
    }
}

void TutorialSystem::LoadProgress() {
    try {
        std::ifstream file("tutorial_progress.json");
        if (file.is_open()) {
            nlohmann::json j;
            file >> j;
            file.close();

            if (j.contains("completed")) {
                completed_tutorials_ = j["completed"].get<std::vector<std::string>>();
            }
            if (j.contains("welcome_shown")) {
                has_shown_welcome_ = j["welcome_shown"].get<bool>();
            }

            // Mark tutorials as completed
            for (const auto& id : completed_tutorials_) {
                auto it = tutorial_index_.find(id);
                if (it != tutorial_index_.end()) {
                    tutorials_[it->second].completed = true;
                }
            }

            spdlog::debug("Tutorial progress loaded: {} completed tutorials", completed_tutorials_.size());
        }
    } catch (const std::exception& e) {
        spdlog::debug("No tutorial progress found or failed to load: {}", e.what());
    }
}

void TutorialSystem::RegisterBuiltInTutorials() {
    // Getting Started Tutorial (10 steps)
    {
        Tutorial tutorial;
        tutorial.id = "getting_started";
        tutorial.name = "Getting Started";
        tutorial.description = "Learn the basics of CyxWiz Engine - creating projects, navigating the interface, and understanding the workspace.";
        tutorial.category = "Getting Started";

        tutorial.steps = {
            {
                "welcome",
                "Welcome to CyxWiz Engine!",
                "CyxWiz is a visual machine learning platform that lets you design, train, and deploy neural networks using an intuitive node-based editor.\n\nThis tutorial will guide you through all the main features of the interface.",
                "",
                "Click Next to continue",
                nullptr,
                false,
                0.0f
            },
            {
                "menu_bar",
                "The Menu Bar",
                "The top menu bar provides access to all features:\n\n• File - Create/open projects, scripts\n• Edit - Undo/redo, find/replace\n• View - Show/hide panels, themes\n• Nodes - Add layers and operations\n• Train - Start training locally or on the network\n• Help - Tutorials and documentation",
                "",
                "Explore the menu options",
                nullptr,
                false,
                0.0f
            },
            {
                "asset_browser",
                "Asset Browser",
                "The Asset Browser on the left shows your project files organized by type. You can:\n\n• Create new scripts and models\n• Import datasets (CSV, NumPy, HDF5)\n• Organize files with drag-and-drop\n• Filter by file type using the toolbar",
                "Asset Browser",
                "Click on the Asset Browser to explore",
                nullptr,
                false,
                0.0f
            },
            {
                "node_editor",
                "Node Editor",
                "The Node Editor is the heart of CyxWiz. Here you design neural networks visually:\n\n• Right-click to add nodes\n• Drag between pins to connect\n• Use Ctrl+G to group nodes\n• Ctrl+F to search nodes",
                "Node Editor",
                "Right-click to see the node context menu",
                nullptr,
                false,
                0.0f
            },
            {
                "properties",
                "Properties Panel",
                "The Properties panel shows details about selected nodes:\n\n• Layer parameters (units, kernel size)\n• Activation functions\n• Input/output tensor shapes\n• Memory usage estimates",
                "Properties",
                "Click on a node to see its properties",
                nullptr,
                false,
                0.0f
            },
            {
                "console",
                "Python Console",
                "The Console provides an interactive Python REPL:\n\n• Run quick experiments\n• Test tensor operations\n• Access loaded datasets\n• Debug your models",
                "Console",
                "Try typing a Python command",
                nullptr,
                false,
                0.0f
            },
            {
                "script_editor",
                "Script Editor",
                "The Script Editor lets you write Python and CyxWiz scripts:\n\n• Syntax highlighting\n• Code completion\n• Run scripts with F5\n• Multiple file tabs",
                "Script Editor",
                "Open a script file to edit",
                nullptr,
                false,
                0.0f
            },
            {
                "viewport",
                "Viewport",
                "The Viewport displays system information and compute device status:\n\n• Available GPUs\n• Backend (CUDA/OpenCL/CPU)\n• Device capabilities\n• Memory information",
                "Viewport",
                "Check your available compute devices",
                nullptr,
                false,
                0.0f
            },
            {
                "training_dashboard",
                "Training Dashboard",
                "The Training Dashboard monitors your training in real-time:\n\n• Loss and accuracy curves\n• Per-epoch statistics\n• Learning rate schedules\n• Export training logs",
                "Training Dashboard",
                "View training metrics here",
                nullptr,
                false,
                0.0f
            },
            {
                "complete",
                "You're Ready!",
                "Congratulations! You now know all the main components of CyxWiz Engine.\n\nNext steps:\n• Try 'Your First Model' tutorial\n• Load a dataset and train\n• Explore the Pattern Library for pre-built architectures",
                "",
                "Click Complete to finish",
                nullptr,
                false,
                0.0f
            }
        };

        RegisterTutorial(std::move(tutorial));
    }

    // Your First Model Tutorial
    {
        Tutorial tutorial;
        tutorial.id = "first_model";
        tutorial.name = "Your First Model";
        tutorial.description = "Create a simple neural network for image classification step by step.";
        tutorial.category = "Getting Started";

        tutorial.steps = {
            {
                "intro",
                "Creating Your First Model",
                "In this tutorial, you'll build a simple image classifier. We'll create a network with input, hidden, and output layers.",
                "",
                "Click Next to begin",
                nullptr,
                false,
                0.0f
            },
            {
                "add_input",
                "Add an Input Layer",
                "First, we need an input layer to receive data. Right-click in the Node Editor and select 'Data > DataInput' to add an input node.",
                "Node Editor",
                "Add a DataInput node from the context menu",
                nullptr,
                false,
                0.0f
            },
            {
                "add_dense",
                "Add a Dense Layer",
                "Now add a Dense (fully connected) layer. Right-click and select 'Layers > Dense'. This layer will learn patterns in your data.",
                "Node Editor",
                "Add a Dense layer from the context menu",
                nullptr,
                false,
                0.0f
            },
            {
                "connect",
                "Connect the Nodes",
                "Connect the DataInput to the Dense layer by dragging from the output pin to the input pin. This defines the data flow.",
                "Node Editor",
                "Drag from output to input to connect nodes",
                nullptr,
                false,
                0.0f
            },
            {
                "add_output",
                "Add Output Layer",
                "Add another Dense layer for the output. Set its units to match the number of classes you want to classify (e.g., 10 for digits).",
                "Node Editor",
                "Add another Dense layer for output",
                nullptr,
                false,
                0.0f
            },
            {
                "complete",
                "Model Complete!",
                "Congratulations! You've created your first neural network. Next, try loading a dataset and training your model.",
                "",
                "Click Complete to finish",
                nullptr,
                false,
                0.0f
            }
        };

        RegisterTutorial(std::move(tutorial));
    }

    // Training Basics Tutorial
    {
        Tutorial tutorial;
        tutorial.id = "training_basics";
        tutorial.name = "Training Basics";
        tutorial.description = "Learn how to configure training parameters, monitor progress, and evaluate your model.";
        tutorial.category = "Training";

        tutorial.steps = {
            {
                "intro",
                "Training Your Model",
                "Training is the process of teaching your neural network to recognize patterns. You'll need a dataset, loss function, and optimizer.",
                "",
                "Click Next to learn about training",
                nullptr,
                false,
                0.0f
            },
            {
                "dataset",
                "Load a Dataset",
                "Go to Dataset > Import Dataset or drag a dataset file into the Asset Browser. CyxWiz supports CSV, NPZ, and other formats.",
                "Asset Browser",
                "Import or select a dataset",
                nullptr,
                false,
                0.0f
            },
            {
                "optimizer",
                "Configure Training",
                "Add an Optimizer node (Training > Optimizer) and connect it to your network. The optimizer controls how the model learns.",
                "Node Editor",
                "Add an Optimizer node",
                nullptr,
                false,
                0.0f
            },
            {
                "loss",
                "Add a Loss Function",
                "Add a Loss node (Training > Loss). The loss function measures how wrong your model's predictions are.",
                "Node Editor",
                "Add a Loss node",
                nullptr,
                false,
                0.0f
            },
            {
                "start",
                "Start Training",
                "Click the Train button in the toolbar or press Ctrl+T. Watch the Training Dashboard to monitor loss and accuracy.",
                "Training Dashboard",
                "Click Train to start training",
                nullptr,
                false,
                0.0f
            },
            {
                "complete",
                "Well Done!",
                "You now understand the basics of training. Experiment with different optimizers, learning rates, and architectures!",
                "",
                "Click Complete to finish",
                nullptr,
                false,
                0.0f
            }
        };

        RegisterTutorial(std::move(tutorial));
    }

    spdlog::info("Registered {} built-in tutorials", tutorials_.size());
}

} // namespace cyxwiz
