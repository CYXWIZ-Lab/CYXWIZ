#include "python_setup_wizard.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include "../../core/engine_config.h"

namespace cyxwiz {

PythonSetupWizard::PythonSetupWizard() {
    StartDetection();
}

void PythonSetupWizard::StartDetection() {
    detection_started_ = true;
    detection_status_ = "Scanning system for Python installations...";
    detection_progress_ = 0.1f;

    // Run detection (synchronous for now - could be async later)
    found_pythons_ = cyxwiz::core::PythonDetector::FindAllPythonInstallations();

    detection_progress_ = 1.0f;
    detection_status_ = "Detection complete";

    OnDetectionComplete();
}

void PythonSetupWizard::OnDetectionComplete() {
    if (found_pythons_.empty()) {
        state_ = State::NoPythonFound;
        spdlog::warn("No Python 3.12+ installations found");
    } else if (found_pythons_.size() == 1) {
        state_ = State::SingleFound;
        selected_index_ = 0;
        selected_python_path_ = found_pythons_[0].executable_path;
        spdlog::info("Found Python {}: {}", found_pythons_[0].version, found_pythons_[0].executable_path);
    } else {
        state_ = State::MultipleFound;
        selected_index_ = 0;
        selected_python_path_ = found_pythons_[0].executable_path;
        spdlog::info("Found {} Python installations", found_pythons_.size());
    }
}

bool PythonSetupWizard::Render() {
    if (result_ != Result::InProgress) {
        return false;  // Wizard completed or cancelled
    }

    // Center the modal window
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImVec2 center = viewport->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(600, 400), ImGuiCond_Appearing);

    // Make it modal (blocks interaction with other windows)
    if (ImGui::BeginPopupModal("Python Setup Wizard", nullptr,
                               ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove)) {

        switch (state_) {
            case State::Detecting:
                RenderDetecting();
                break;
            case State::NoPythonFound:
                RenderNoPythonFound();
                break;
            case State::SingleFound:
                RenderSingleFound();
                break;
            case State::MultipleFound:
                RenderMultipleFound();
                break;
        }

        ImGui::EndPopup();
    }

    // Open the popup on first frame
    if (!ImGui::IsPopupOpen("Python Setup Wizard")) {
        ImGui::OpenPopup("Python Setup Wizard");
    }

    return result_ == Result::InProgress;
}

void PythonSetupWizard::RenderDetecting() {
    ImGui::TextWrapped("Welcome to CyxWiz Engine!");
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::TextWrapped("Detecting Python 3.12 or newer on your system...");
    ImGui::Spacing();

    ImGui::Text("Status: %s", detection_status_.c_str());
    ImGui::ProgressBar(detection_progress_, ImVec2(-1, 0));

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::TextWrapped("CyxWiz Engine requires Python 3.12+ with the venv module installed.");
}

void PythonSetupWizard::RenderNoPythonFound() {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.6f, 0.0f, 1.0f));
    ImGui::TextWrapped("No Python 3.12+ Installation Found");
    ImGui::PopStyleColor();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::TextWrapped("CyxWiz Engine requires Python 3.12 or newer to run.");
    ImGui::Spacing();

    ImGui::TextWrapped("Please install Python 3.12 or 3.13 from:");
    ImGui::Spacing();
    ImGui::Indent();
    ImGui::BulletText("Windows: https://www.python.org/downloads/");
    ImGui::BulletText("macOS: brew install python@3.12");
    ImGui::BulletText("Linux: sudo apt install python3.12 python3.12-venv");
    ImGui::Unindent();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::TextWrapped("After installing Python, restart CyxWiz Engine.");

    ImGui::Spacing();
    ImGui::Spacing();

    // Exit button
    float button_width = 120.0f;
    float window_width = ImGui::GetWindowWidth();
    ImGui::SetCursorPosX((window_width - button_width) * 0.5f);

    if (ImGui::Button("Exit", ImVec2(button_width, 0))) {
        result_ = Result::Cancelled;
        ImGui::CloseCurrentPopup();
    }
}

void PythonSetupWizard::RenderSingleFound() {
    const auto& python = found_pythons_[0];

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 1.0f, 0.4f, 1.0f));
    ImGui::TextWrapped("Python %s Found!", python.version.c_str());
    ImGui::PopStyleColor();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Version:");
    ImGui::SameLine(150);
    ImGui::TextWrapped("%s", python.version.c_str());

    ImGui::Text("Location:");
    ImGui::SameLine(150);
    ImGui::TextWrapped("%s", python.executable_path.c_str());

    ImGui::Text("Home:");
    ImGui::SameLine(150);
    ImGui::TextWrapped("%s", python.home.c_str());

    ImGui::Spacing();

    ImGui::Text("Features:");
    ImGui::SameLine(150);
    ImGui::TextWrapped("venv: %s, pip: %s",
                       python.has_venv_module ? "Yes" : "No",
                       python.has_pip ? "Yes" : "No");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    if (!python.has_venv_module) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.6f, 0.0f, 1.0f));
        ImGui::TextWrapped("Warning: venv module not found. Please install it:");
        ImGui::PopStyleColor();
        ImGui::Indent();
        ImGui::BulletText("Windows: Reinstall Python with \"pip\" option");
        ImGui::BulletText("Linux: sudo apt install python3-venv");
        ImGui::Unindent();
        ImGui::Spacing();
    }

    ImGui::TextWrapped("This Python installation will be used to create virtual environments for your projects.");

    ImGui::Spacing();
    ImGui::Spacing();

    // Buttons
    float button_width = 120.0f;
    float spacing = 20.0f;
    float total_width = button_width * 2 + spacing;
    float window_width = ImGui::GetWindowWidth();
    ImGui::SetCursorPosX((window_width - total_width) * 0.5f);

    bool can_continue = python.has_venv_module;

    if (!can_continue) {
        ImGui::BeginDisabled();
    }

    if (ImGui::Button("Continue", ImVec2(button_width, 0))) {
        // Save to config
        auto& config = cyxwiz::core::EngineConfig::Instance();
        config.SetSystemPythonPath(selected_python_path_);
        config.Save();

        result_ = Result::Completed;
        ImGui::CloseCurrentPopup();

        spdlog::info("Python configured: {}", selected_python_path_);
    }

    if (!can_continue) {
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("Cannot continue without venv module");
        }
    }

    ImGui::SameLine(0, spacing);

    if (ImGui::Button("Cancel", ImVec2(button_width, 0))) {
        result_ = Result::Cancelled;
        ImGui::CloseCurrentPopup();
    }
}

void PythonSetupWizard::RenderMultipleFound() {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 1.0f, 0.4f, 1.0f));
    ImGui::TextWrapped("Multiple Python Installations Found");
    ImGui::PopStyleColor();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::TextWrapped("Please select which Python installation to use:");
    ImGui::Spacing();

    // List of Python installations
    if (ImGui::BeginChild("PythonList", ImVec2(0, 200), true)) {
        for (int i = 0; i < static_cast<int>(found_pythons_.size()); ++i) {
            const auto& python = found_pythons_[i];

            bool is_selected = (i == selected_index_);

            // Format the label
            std::string label = "Python " + python.version;
            if (!python.has_venv_module) {
                label += " (missing venv)";
            }

            // Color based on validity
            if (!python.has_venv_module) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.6f, 1.0f));
            }

            if (ImGui::Selectable(label.c_str(), is_selected)) {
                selected_index_ = i;
                selected_python_path_ = python.executable_path;
            }

            if (!python.has_venv_module) {
                ImGui::PopStyleColor();
            }

            // Show details on same line
            ImGui::SameLine(200);
            ImGui::TextDisabled("%s", python.executable_path.c_str());
        }
    }
    ImGui::EndChild();

    ImGui::Spacing();

    // Show details of selected Python
    if (selected_index_ >= 0 && selected_index_ < static_cast<int>(found_pythons_.size())) {
        const auto& python = found_pythons_[selected_index_];

        ImGui::Text("Version: %s", python.version.c_str());
        ImGui::Text("Location: %s", python.executable_path.c_str());
        ImGui::Text("Features: venv: %s, pip: %s",
                   python.has_venv_module ? "Yes" : "No",
                   python.has_pip ? "Yes" : "No");

        if (!python.has_venv_module) {
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.6f, 0.0f, 1.0f));
            ImGui::TextWrapped("Warning: This Python installation is missing the venv module.");
            ImGui::PopStyleColor();
        }
    }

    ImGui::Spacing();
    ImGui::Spacing();

    // Buttons
    float button_width = 120.0f;
    float spacing = 20.0f;
    float total_width = button_width * 2 + spacing;
    float window_width = ImGui::GetWindowWidth();
    ImGui::SetCursorPosX((window_width - total_width) * 0.5f);

    bool can_continue = false;
    if (selected_index_ >= 0 && selected_index_ < static_cast<int>(found_pythons_.size())) {
        can_continue = found_pythons_[selected_index_].has_venv_module;
    }

    if (!can_continue) {
        ImGui::BeginDisabled();
    }

    if (ImGui::Button("Continue", ImVec2(button_width, 0))) {
        // Save to config
        auto& config = cyxwiz::core::EngineConfig::Instance();
        config.SetSystemPythonPath(selected_python_path_);
        config.Save();

        result_ = Result::Completed;
        ImGui::CloseCurrentPopup();

        spdlog::info("Python configured: {}", selected_python_path_);
    }

    if (!can_continue) {
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("Selected Python must have venv module");
        }
    }

    ImGui::SameLine(0, spacing);

    if (ImGui::Button("Cancel", ImVec2(button_width, 0))) {
        result_ = Result::Cancelled;
        ImGui::CloseCurrentPopup();
    }
}

} // namespace cyxwiz
