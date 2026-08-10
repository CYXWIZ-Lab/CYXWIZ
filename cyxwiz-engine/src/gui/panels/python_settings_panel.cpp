#include "python_settings_panel.h"
#include "../../core/engine_config.h"
#include "../../core/python_detector.h"
#include "../../core/file_dialogs.h"
#include "../icons.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <cstring>
#include <sstream>
#include <algorithm>

namespace gui {

PythonSettingsPanel::PythonSettingsPanel() {
    LoadSettings();
}

void PythonSettingsPanel::LoadSettings() {
    auto& config = cyxwiz::core::EngineConfig::Instance();

    // Load system Python path
    std::string python_path = config.GetSystemPythonPath();
    strncpy(system_python_path_, python_path.c_str(), sizeof(system_python_path_) - 1);

    // Load auto-create venv setting
    auto_create_venv_ = config.GetAutoCreateVenv();

    // Load default packages
    const auto& packages = config.GetDefaultVenvPackages();
    std::ostringstream oss;
    for (size_t i = 0; i < packages.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << packages[i];
    }
    std::string packages_str = oss.str();
    strncpy(default_packages_, packages_str.c_str(), sizeof(default_packages_) - 1);

    // Validate current Python
    if (!python_path.empty()) {
        auto detector = cyxwiz::core::PythonDetector();
        auto installation = detector.ValidatePythonInstallation(python_path);
        if (installation) {
            python_valid_ = true;
            python_version_ = installation->version;
            python_error_.clear();
        } else {
            python_valid_ = false;
            python_version_.clear();
            python_error_ = "Invalid or inaccessible Python installation";
        }
    } else {
        python_valid_ = false;
        python_version_.clear();
        python_error_ = "No Python configured";
    }

    is_modified_ = false;
}

void PythonSettingsPanel::ApplySettings() {
    auto& config = cyxwiz::core::EngineConfig::Instance();

    // Save system Python path
    config.SetSystemPythonPath(std::string(system_python_path_));

    // Save auto-create venv setting
    config.SetAutoCreateVenv(auto_create_venv_);

    // Parse and save default packages
    std::vector<std::string> packages;
    std::string packages_str(default_packages_);
    std::istringstream iss(packages_str);
    std::string package;
    while (std::getline(iss, package, ',')) {
        // Trim whitespace
        package.erase(0, package.find_first_not_of(" \t"));
        package.erase(package.find_last_not_of(" \t") + 1);
        if (!package.empty()) {
            packages.push_back(package);
        }
    }
    config.SetDefaultVenvPackages(packages);

    // Save config to disk
    if (config.Save()) {
        spdlog::info("Python settings saved successfully");
        is_modified_ = false;
    } else {
        spdlog::error("Failed to save Python settings");
    }
}

void PythonSettingsPanel::RevertSettings() {
    LoadSettings();
    is_modified_ = false;
}

void PythonSettingsPanel::Render() {
    ImGui::TextWrapped("Configure Python interpreter and virtual environment settings.");
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ========== System Python Path ==========
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), ICON_FA_CODE " System Python");
    ImGui::Spacing();

    ImGui::Text("Python Executable:");
    ImGui::SetNextItemWidth(-140);
    if (ImGui::InputText("##PythonPath", system_python_path_, sizeof(system_python_path_))) {
        is_modified_ = true;
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_FOLDER_OPEN " Browse", ImVec2(120, 0))) {
        BrowsePythonExecutable();
    }

    ImGui::Spacing();

    // Auto-detect button
    if (ImGui::Button(ICON_FA_MAGNIFYING_GLASS " Auto-Detect Python", ImVec2(200, 0))) {
        RunAutoDetection();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Scan system for Python 3.12+ installations");
    }

    ImGui::Spacing();

    // Python status
    if (!std::string(system_python_path_).empty()) {
        if (python_valid_) {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), ICON_FA_CHECK " Python %s", python_version_.c_str());
            ImGui::SameLine();
            ImGui::TextDisabled("(Valid)");
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), ICON_FA_XMARK " %s", python_error_.c_str());
        }
    } else {
        ImGui::TextDisabled("No Python configured");
    }

    // Auto-detection results
    if (show_detection_results_ && !detected_pythons_.empty()) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Detected Python Installations:");
        ImGui::Spacing();

        if (ImGui::BeginChild("DetectedPythons", ImVec2(0, 150), true)) {
            for (int i = 0; i < static_cast<int>(detected_pythons_.size()); ++i) {
                const auto& python = detected_pythons_[i];

                bool is_selected = (i == selected_detected_index_);
                std::string label = python.version + " - " + python.path;

                if (ImGui::Selectable(label.c_str(), is_selected)) {
                    selected_detected_index_ = i;
                }

                if (ImGui::IsItemHovered()) {
                    std::string tooltip = "Path: " + python.path + "\n";
                    tooltip += "Version: " + python.version + "\n";
                    tooltip += "Has venv: " + std::string(python.has_venv ? "Yes" : "No") + "\n";
                    tooltip += "Has pip: " + std::string(python.has_pip ? "Yes" : "No");
                    ImGui::SetTooltip("%s", tooltip.c_str());
                }
            }
        }
        ImGui::EndChild();

        ImGui::Spacing();
        ImGui::BeginDisabled(selected_detected_index_ < 0);
        if (ImGui::Button(ICON_FA_CHECK " Use Selected Python", ImVec2(200, 0))) {
            if (selected_detected_index_ >= 0 && selected_detected_index_ < static_cast<int>(detected_pythons_.size())) {
                const auto& selected = detected_pythons_[selected_detected_index_];
                strncpy(system_python_path_, selected.path.c_str(), sizeof(system_python_path_) - 1);
                python_version_ = selected.version;
                python_valid_ = true;
                python_error_.clear();
                is_modified_ = true;
                show_detection_results_ = false;
            }
        }
        ImGui::EndDisabled();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ========== Virtual Environment Settings ==========
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), ICON_FA_CUBE " Virtual Environment");
    ImGui::Spacing();

    if (ImGui::Checkbox("Auto-create virtual environment for new projects", &auto_create_venv_)) {
        is_modified_ = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("When enabled, a Python venv will be automatically created\nfor each new project in <project>/python/");
    }

    ImGui::Spacing();

    ImGui::Text("Default packages to install in new venvs:");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Comma-separated list of packages to install automatically.\nExample: numpy, torch, scikit-learn");
    }

    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputTextMultiline("##DefaultPackages", default_packages_, sizeof(default_packages_), ImVec2(-1, 60))) {
        is_modified_ = true;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ========== Apply/Revert Buttons ==========
    ImGui::BeginDisabled(!is_modified_);
    if (ImGui::Button(ICON_FA_FLOPPY_DISK " Apply", ImVec2(100, 0))) {
        ApplySettings();
    }
    ImGui::EndDisabled();

    ImGui::SameLine();

    ImGui::BeginDisabled(!is_modified_);
    if (ImGui::Button(ICON_FA_ROTATE_LEFT " Revert", ImVec2(100, 0))) {
        RevertSettings();
    }
    ImGui::EndDisabled();

    if (is_modified_) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), ICON_FA_TRIANGLE_EXCLAMATION " Unsaved changes");
    }
}

void PythonSettingsPanel::RunAutoDetection() {
    spdlog::info("Running Python auto-detection...");
    detected_pythons_.clear();
    selected_detected_index_ = -1;

    auto detector = cyxwiz::core::PythonDetector();
    auto installations = detector.FindAllPythonInstallations();

    for (const auto& installation : installations) {
        DetectedPython python;
        python.path = installation.executable_path;
        python.version = installation.version;
        python.has_venv = installation.has_venv_module;
        python.has_pip = installation.has_pip;
        detected_pythons_.push_back(python);
    }

    if (detected_pythons_.empty()) {
        spdlog::warn("No Python 3.12+ installations found");
    } else {
        spdlog::info("Found {} Python installation(s)", detected_pythons_.size());
        show_detection_results_ = true;
    }
}

void PythonSettingsPanel::BrowsePythonExecutable() {
    if (auto selected = cyxwiz::FileDialogs::OpenFile(
            "Select Python Executable",
            {{"Python Executables", "python,python3,python.exe"}, {"All Files", "*"}},
            system_python_path_[0] == '\0' ? nullptr : system_python_path_)) {
        strncpy(system_python_path_, selected->c_str(), sizeof(system_python_path_) - 1);
        system_python_path_[sizeof(system_python_path_) - 1] = '\0';
        is_modified_ = true;

        // Validate the selected Python
        auto detector = cyxwiz::core::PythonDetector();
        auto installation = detector.ValidatePythonInstallation(*selected);
        if (installation) {
            python_valid_ = true;
            python_version_ = installation->version;
            python_error_.clear();
            spdlog::info("Selected Python: {} ({})", *selected, python_version_);
        } else {
            python_valid_ = false;
            python_version_.clear();
            python_error_ = "Invalid Python installation";
            spdlog::error("Selected file is not a valid Python 3.12+ installation: {}", *selected);
        }
    }
}

} // namespace gui
