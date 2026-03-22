#include "project_selection_dialog.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include "../../core/engine_config.h"
#include "../../core/project_manager.h"
#include <filesystem>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#else
#include <cstdlib>
#endif

namespace cyxwiz {

ProjectSelectionDialog::ProjectSelectionDialog() {
    LoadRecentProjects();

    // Set default project location to user's documents folder
#ifdef _WIN32
    if (auto* userprofile = std::getenv("USERPROFILE")) {
        std::filesystem::path default_loc = std::filesystem::path(userprofile) / "Documents" / "CyxWiz Projects";
        strncpy(project_location_buf_, default_loc.string().c_str(), sizeof(project_location_buf_) - 1);
    }
#else
    if (auto* home = std::getenv("HOME")) {
        std::filesystem::path default_loc = std::filesystem::path(home) / "CyxWiz Projects";
        strncpy(project_location_buf_, default_loc.string().c_str(), sizeof(project_location_buf_) - 1);
    }
#endif
}

void ProjectSelectionDialog::LoadRecentProjects() {
    auto& config = cyxwiz::core::EngineConfig::Instance();
    recent_projects_ = config.GetRecentProjects();

    // Filter out projects that no longer exist
    recent_projects_.erase(
        std::remove_if(recent_projects_.begin(), recent_projects_.end(),
            [](const std::string& path) {
                return !std::filesystem::exists(path);
            }),
        recent_projects_.end()
    );
}

bool ProjectSelectionDialog::Render() {
    if (result_ != Result::InProgress) {
        return false;  // Dialog completed
    }

    // Center the modal window
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImVec2 center = viewport->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(700, 500), ImGuiCond_Appearing);

    // Make it modal (blocks interaction with other windows)
    if (ImGui::BeginPopupModal("Select Project", nullptr,
                               ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove)) {

        ImGui::TextWrapped("Welcome to CyxWiz Engine!");
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::TextWrapped("Please select a project to open or create a new one.");
        ImGui::Spacing();

        // Recent Projects Section
        if (!recent_projects_.empty()) {
            ImGui::Text("Recent Projects:");
            ImGui::Spacing();
            RenderRecentProjects();
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
        }

        // Action Buttons
        RenderButtons();

        // Nested dialogs for create/open
        if (show_create_dialog_) {
            ShowCreateProjectDialog();
        }

        if (show_open_dialog_) {
            ShowOpenProjectDialog();
        }

        ImGui::EndPopup();
    }

    // Open the popup on first frame
    if (!ImGui::IsPopupOpen("Select Project")) {
        ImGui::OpenPopup("Select Project");
    }

    return result_ == Result::InProgress;
}

void ProjectSelectionDialog::RenderRecentProjects() {
    if (ImGui::BeginChild("RecentProjects", ImVec2(0, 200), true)) {
        for (int i = 0; i < static_cast<int>(recent_projects_.size()); ++i) {
            const auto& project_path = recent_projects_[i];

            bool is_selected = (i == selected_recent_index_);

            // Extract project name from path
            std::filesystem::path path(project_path);
            std::string project_name = path.filename().string();

            if (ImGui::Selectable(project_name.c_str(), is_selected, ImGuiSelectableFlags_AllowDoubleClick)) {
                selected_recent_index_ = i;

                // Double-click to open
                if (ImGui::IsMouseDoubleClicked(0)) {
                    SelectProject(project_path);
                }
            }

            // Show full path on hover
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s", project_path.c_str());
            }
        }
    }
    ImGui::EndChild();

    // Open selected recent project button
    ImGui::Spacing();
    ImGui::BeginDisabled(selected_recent_index_ < 0);
    if (ImGui::Button("Open Selected", ImVec2(150, 0))) {
        if (selected_recent_index_ >= 0 && selected_recent_index_ < static_cast<int>(recent_projects_.size())) {
            SelectProject(recent_projects_[selected_recent_index_]);
        }
    }
    ImGui::EndDisabled();
}

void ProjectSelectionDialog::RenderButtons() {
    float button_width = 150.0f;
    float spacing = 20.0f;
    float total_width = button_width * 3 + spacing * 2;
    float window_width = ImGui::GetWindowWidth();
    ImGui::SetCursorPosX((window_width - total_width) * 0.5f);

    if (ImGui::Button("Create New Project", ImVec2(button_width, 0))) {
        show_create_dialog_ = true;
    }

    ImGui::SameLine(0, spacing);

    if (ImGui::Button("Open Existing", ImVec2(button_width, 0))) {
        show_open_dialog_ = true;
    }

    ImGui::SameLine(0, spacing);

    if (ImGui::Button("Exit", ImVec2(button_width, 0))) {
        result_ = Result::Exit;
        ImGui::CloseCurrentPopup();
    }
}

void ProjectSelectionDialog::ShowCreateProjectDialog() {
    ImGui::OpenPopup("Create New Project");

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(500, 300), ImGuiCond_Appearing);

    if (ImGui::BeginPopupModal("Create New Project", &show_create_dialog_,
                               ImGuiWindowFlags_NoResize)) {

        ImGui::Text("Project Name:");
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##ProjectName", project_name_buf_, sizeof(project_name_buf_));

        ImGui::Spacing();

        ImGui::Text("Project Location:");
        ImGui::SetNextItemWidth(-50);
        ImGui::InputText("##ProjectLocation", project_location_buf_, sizeof(project_location_buf_));
        ImGui::SameLine();
        if (ImGui::Button("...##Browse")) {
            // TODO: Open folder browser dialog
            spdlog::warn("Folder browser not yet implemented");
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::Checkbox("Create Python virtual environment", &create_venv_);

        ImGui::Spacing();
        ImGui::Spacing();

        // Preview full path
        if (strlen(project_name_buf_) > 0) {
            std::filesystem::path full_path = std::filesystem::path(project_location_buf_) / project_name_buf_;
            ImGui::TextWrapped("Project will be created at:");
            ImGui::TextWrapped("%s", full_path.string().c_str());
        }

        ImGui::Spacing();
        ImGui::Spacing();

        // Buttons
        float button_width = 100.0f;
        float spacing = 20.0f;
        float total_width = button_width * 2 + spacing;
        float window_width = ImGui::GetWindowWidth();
        ImGui::SetCursorPosX((window_width - total_width) * 0.5f);

        bool can_create = strlen(project_name_buf_) > 0 && strlen(project_location_buf_) > 0;

        ImGui::BeginDisabled(!can_create);
        if (ImGui::Button("Create", ImVec2(button_width, 0))) {
            if (CreateNewProject(project_name_buf_, project_location_buf_)) {
                show_create_dialog_ = false;
            }
        }
        ImGui::EndDisabled();

        ImGui::SameLine(0, spacing);

        if (ImGui::Button("Cancel", ImVec2(button_width, 0))) {
            show_create_dialog_ = false;
        }

        ImGui::EndPopup();
    }
}

void ProjectSelectionDialog::ShowOpenProjectDialog() {
#ifdef _WIN32
    // Use Windows native file dialog
    char filename[MAX_PATH] = "";

    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = nullptr;
    ofn.lpstrFilter = "CyxWiz Project (*.cyxwiz)\0*.cyxwiz\0All Files (*.*)\0*.*\0";
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = "Open CyxWiz Project";
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;

    if (GetOpenFileNameA(&ofn)) {
        SelectProject(filename);
    }

    show_open_dialog_ = false;
#else
    // TODO: Implement for Linux/macOS (use zenity/kdialog or ImGui file browser)
    spdlog::warn("File dialog not yet implemented for this platform");
    show_open_dialog_ = false;
#endif
}

bool ProjectSelectionDialog::CreateNewProject(const std::string& name, const std::string& location) {
    if (name.empty() || location.empty()) {
        spdlog::error("Project name and location cannot be empty");
        return false;
    }

    std::filesystem::path project_path = std::filesystem::path(location) / name;

    // Check if project already exists
    if (std::filesystem::exists(project_path)) {
        spdlog::error("Project already exists at: {}", project_path.string());
        return false;
    }

    // Create project using ProjectManager
    auto& pm = cyxwiz::ProjectManager::Instance();
    if (!pm.CreateProject(name, location)) {
        spdlog::error("Failed to create project: {}", project_path.string());
        return false;
    }

    // Select the newly created project
    SelectProject(project_path.string());
    return true;
}

void ProjectSelectionDialog::SelectProject(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        spdlog::error("Project does not exist: {}", path);
        return;
    }

    // Validate that it's a valid project directory
    // (Check for .cyxwiz file or project structure)
    std::filesystem::path project_file = std::filesystem::path(path) / (std::filesystem::path(path).filename().string() + ".cyxwiz");

    if (!std::filesystem::exists(project_file)) {
        // Try to find any .cyxwiz file in the directory
        bool found = false;
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            if (entry.path().extension() == ".cyxwiz") {
                project_file = entry.path();
                found = true;
                break;
            }
        }

        if (!found) {
            spdlog::error("No .cyxwiz file found in: {}", path);
            return;
        }
    }

    selected_project_path_ = project_file.string();
    result_ = Result::ProjectSelected;

    // Add to recent projects
    auto& config = cyxwiz::core::EngineConfig::Instance();
    config.AddRecentProject(selected_project_path_);
    config.Save();

    spdlog::info("Selected project: {}", selected_project_path_);

    ImGui::CloseCurrentPopup();
}

} // namespace cyxwiz
