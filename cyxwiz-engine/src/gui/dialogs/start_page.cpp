#include "start_page.h"
#include "../../core/engine_config.h"
#include "../../core/project_manager.h"
#include "../../core/file_dialogs.h"
#include "../icons.h"
#include "../theme.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <ctime>
#include <string>
#include <system_error>

namespace cyxwiz {

namespace {

bool LocalTime(std::time_t time, std::tm& out) {
#ifdef _WIN32
    return localtime_s(&out, &time) == 0;
#else
    return localtime_r(&time, &out) != nullptr;
#endif
}

std::string FormatRecentProjectTime(std::time_t time) {
    std::tm tm_info{};
    if (!LocalTime(time, tm_info)) {
        return "Unknown time";
    }

    char time_str[64] = {};
    if (std::strftime(time_str, sizeof(time_str), "%m/%d/%Y %I:%M %p", &tm_info) == 0) {
        return "Unknown time";
    }
    return time_str;
}

} // namespace

StartPage::StartPage() {
    LoadRecentProjects();
    GroupProjectsByTime();

    // Set default project location
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

void StartPage::LoadRecentProjects() {
    auto& pm = ProjectManager::Instance();
    const auto& recent = pm.GetRecentProjects();

    all_projects_.clear();
    for (const auto& rp : recent) {
        if (std::filesystem::exists(rp.path)) {
            RecentProject proj;
            proj.name = rp.name;
            proj.path = rp.path;
            proj.last_opened = rp.last_opened;
            all_projects_.push_back(proj);
        }
    }
}

void StartPage::GroupProjectsByTime() {
    this_week_.clear();
    this_month_.clear();
    older_.clear();

    for (const auto& proj : all_projects_) {
        // Apply search filter if active
        if (search_buffer_[0] != '\0') {
            std::string search_lower = search_buffer_;
            std::string name_lower = proj.name;
            std::transform(search_lower.begin(), search_lower.end(), search_lower.begin(), ::tolower);
            std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);

            if (name_lower.find(search_lower) == std::string::npos) {
                continue;  // Skip if doesn't match search
            }
        }

        if (IsThisWeek(proj.last_opened)) {
            this_week_.push_back(proj);
        } else if (IsThisMonth(proj.last_opened)) {
            this_month_.push_back(proj);
        } else {
            older_.push_back(proj);
        }
    }
}

bool StartPage::IsThisWeek(std::time_t time) const {
    std::time_t now = std::time(nullptr);
    double diff_seconds = std::difftime(now, time);
    return diff_seconds < (7 * 24 * 60 * 60);  // 7 days
}

bool StartPage::IsThisMonth(std::time_t time) const {
    std::time_t now = std::time(nullptr);
    std::tm now_tm{};
    std::tm time_tm{};
    if (!LocalTime(now, now_tm) || !LocalTime(time, time_tm)) {
        return false;
    }

    return (now_tm.tm_year == time_tm.tm_year &&
            now_tm.tm_mon == time_tm.tm_mon);
}

bool StartPage::Render() {
    if (result_ != Result::InProgress) {
        return false;  // Page completed
    }

    // Full-screen window
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                             ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    bool open = ImGui::Begin("##StartPage", nullptr, flags);
    ImGui::PopStyleVar();

    if (!open) {
        result_ = Result::Exit;
        ImGui::End();
        return false;
    }

    // Custom styling for start page
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(12, 12));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(40, 30));

    // Title
    ImGui::SetCursorPos(ImVec2(40, 30));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
    ImGui::SetWindowFontScale(1.8f);
    ImGui::Text("Get started");
    ImGui::SetWindowFontScale(1.0f);
    ImGui::PopStyleVar();

    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 10);

    // Two-column layout
    float window_width = ImGui::GetContentRegionAvail().x;
    float left_width = window_width * 0.65f;
    float right_width = window_width * 0.30f;

    ImGui::BeginGroup();
    {
        ImGui::BeginChild("LeftColumn", ImVec2(left_width, -60), false, ImGuiWindowFlags_NoScrollbar);
        {
            RenderSearchBar();
            ImGui::Spacing();
            RenderRecentProjects();
        }
        ImGui::EndChild();
    }
    ImGui::EndGroup();

    ImGui::SameLine();
    ImGui::SetCursorPosX(40 + left_width + 40);

    ImGui::BeginGroup();
    {
        ImGui::BeginChild("RightColumn", ImVec2(right_width, -60), false);
        {
            RenderActionCards();
        }
        ImGui::EndChild();
    }
    ImGui::EndGroup();

    // Bottom bar
    RenderBottomBar();

    if (show_create_dialog_) {
        RenderCreateProjectDialog();
    }

    ImGui::PopStyleVar(2);
    ImGui::End();

    return true;
}

void StartPage::RenderSearchBar() {
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.15f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(12, 10));

    ImGui::SetNextItemWidth(-1);

    // Search icon
    ImGui::Text(ICON_FA_MAGNIFYING_GLASS);
    ImGui::SameLine();
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() - 8);

    if (ImGui::InputTextWithHint("##Search", "Search recent (Alt+S)", search_buffer_, sizeof(search_buffer_))) {
        GroupProjectsByTime();  // Re-group with filter
    }

    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

void StartPage::RenderRecentProjects() {
    ImGui::BeginChild("##RecentProjects", ImVec2(0, 0), false);

    // This week
    if (!this_week_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0, 0, 0, 0));
        if (ImGui::CollapsingHeader(ICON_FA_CHEVRON_DOWN " This week", &show_this_week_, ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Indent(20);
            for (const auto& proj : this_week_) {
                ImGui::PushID(proj.path.c_str());

                // Project icon
                ImGui::Text(ICON_FA_FOLDER);
                ImGui::SameLine();

                // Clickable project name
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
                if (ImGui::Selectable(proj.name.c_str(), false, ImGuiSelectableFlags_AllowDoubleClick)) {
                    if (ImGui::IsMouseDoubleClicked(0)) {
                        OpenProject(proj.path);
                    }
                }
                ImGui::PopStyleColor();

                // Path and timestamp
                ImGui::Indent(30);
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
                ImGui::TextWrapped("%s", proj.path.c_str());

                const std::string time_str = FormatRecentProjectTime(proj.last_opened);
                ImGui::Text("%s", time_str.c_str());
                ImGui::PopStyleColor();
                ImGui::Unindent(30);

                ImGui::Spacing();
                ImGui::PopID();
            }
            ImGui::Unindent(20);
        }
        ImGui::PopStyleColor();
        ImGui::Spacing();
    }

    // This month
    if (!this_month_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0, 0, 0, 0));
        if (ImGui::CollapsingHeader(ICON_FA_CHEVRON_DOWN " This month", &show_this_month_, ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Indent(20);
            for (const auto& proj : this_month_) {
                ImGui::PushID(proj.path.c_str());

                ImGui::Text(ICON_FA_FOLDER);
                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
                if (ImGui::Selectable(proj.name.c_str(), false, ImGuiSelectableFlags_AllowDoubleClick)) {
                    if (ImGui::IsMouseDoubleClicked(0)) {
                        OpenProject(proj.path);
                    }
                }
                ImGui::PopStyleColor();

                ImGui::Indent(30);
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
                ImGui::TextWrapped("%s", proj.path.c_str());

                const std::string time_str = FormatRecentProjectTime(proj.last_opened);
                ImGui::Text("%s", time_str.c_str());
                ImGui::PopStyleColor();
                ImGui::Unindent(30);

                ImGui::Spacing();
                ImGui::PopID();
            }
            ImGui::Unindent(20);
        }
        ImGui::PopStyleColor();
        ImGui::Spacing();
    }

    // Older
    if (!older_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0, 0, 0, 0));
        if (ImGui::CollapsingHeader(ICON_FA_CHEVRON_DOWN " Older", &show_older_)) {
            ImGui::Indent(20);
            for (const auto& proj : older_) {
                ImGui::PushID(proj.path.c_str());

                ImGui::Text(ICON_FA_FOLDER);
                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
                if (ImGui::Selectable(proj.name.c_str(), false, ImGuiSelectableFlags_AllowDoubleClick)) {
                    if (ImGui::IsMouseDoubleClicked(0)) {
                        OpenProject(proj.path);
                    }
                }
                ImGui::PopStyleColor();

                ImGui::Indent(30);
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
                ImGui::TextWrapped("%s", proj.path.c_str());

                const std::string time_str = FormatRecentProjectTime(proj.last_opened);
                ImGui::Text("%s", time_str.c_str());
                ImGui::PopStyleColor();
                ImGui::Unindent(30);

                ImGui::Spacing();
                ImGui::PopID();
            }
            ImGui::Unindent(20);
        }
        ImGui::PopStyleColor();
    }

    // No projects message
    if (all_projects_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
        ImGui::TextWrapped("No recent projects");
        ImGui::TextWrapped("Create a new project or open an existing one to get started");
        ImGui::PopStyleColor();
    }

    ImGui::EndChild();
}

void StartPage::RenderActionCards() {
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(15, 12));
    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.0f, 0.5f));

    float button_width = ImGui::GetContentRegionAvail().x;

    // Create new project
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.4f, 0.8f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.5f, 0.9f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.15f, 0.35f, 0.75f, 1.0f));
    if (ImGui::Button(ICON_FA_PLUS " Create a new project", ImVec2(button_width, 0))) {
        CreateNewProject();
    }
    ImGui::PopStyleColor(3);

    ImGui::Spacing();

    // Open project
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.15f, 0.15f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.25f, 0.25f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
    if (ImGui::Button(ICON_FA_FOLDER_OPEN " Open a project or solution", ImVec2(button_width, 0))) {
        OpenExistingProject();
    }
    ImGui::PopStyleColor(3);

    ImGui::Spacing();

    // Open folder
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.15f, 0.15f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.25f, 0.25f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
    if (ImGui::Button(ICON_FA_FOLDER " Open a project folder", ImVec2(button_width, 0))) {
        OpenProjectFolder();
    }
    ImGui::PopStyleColor(3);

    ImGui::Spacing();

    // Clone repository
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.15f, 0.15f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.25f, 0.25f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
    ImGui::BeginDisabled();
    ImGui::Button(ICON_FA_CLOUD_ARROW_DOWN " Clone a repository (planned)", ImVec2(button_width, 0));
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("Repository cloning is not wired yet");
    }
    ImGui::PopStyleColor(3);

    ImGui::PopStyleVar(2);
}

void StartPage::RenderBottomBar() {
    // Position button at bottom-right of window content area
    ImVec2 window_size = ImGui::GetWindowSize();
    float button_width = 220.0f;
    float button_height = 30.0f;
    float margin = 20.0f;

    ImGui::SetCursorPos(ImVec2(window_size.x - button_width - margin, window_size.y - button_height - margin));

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.6f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.15f, 0.15f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.25f, 0.25f, 1.0f));
    if (ImGui::Button("Continue without project", ImVec2(button_width, button_height))) {
        ContinueWithoutProject();
    }
    ImGui::PopStyleColor(3);
}

void StartPage::OpenProject(const std::string& path) {
    auto& pm = ProjectManager::Instance();
    if (pm.OpenProject(path)) {
        selected_project_path_ = path;
        result_ = Result::ProjectSelected;
        spdlog::info("Opening project: {}", path);
    } else {
        spdlog::error("Failed to open project: {}", path);
    }
}

void StartPage::CreateNewProject() {
    show_create_dialog_ = true;
}

void StartPage::RenderCreateProjectDialog() {
    ImGui::OpenPopup("Create New Project");

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(500, 260), ImGuiCond_Appearing);

    if (ImGui::BeginPopupModal("Create New Project", &show_create_dialog_, ImGuiWindowFlags_NoResize)) {
        ImGui::Text("Project Name:");
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##ProjectName", project_name_buf_, sizeof(project_name_buf_));

        ImGui::Spacing();

        ImGui::Text("Project Location:");
        ImGui::SetNextItemWidth(-50);
        ImGui::InputText("##ProjectLocation", project_location_buf_, sizeof(project_location_buf_));
        ImGui::SameLine();
        if (ImGui::Button("...##BrowseProjectLocation")) {
            auto selected_folder = FileDialogs::SelectFolder("Select Project Location", project_location_buf_);
            if (selected_folder) {
                strncpy(project_location_buf_, selected_folder->c_str(), sizeof(project_location_buf_) - 1);
                project_location_buf_[sizeof(project_location_buf_) - 1] = '\0';
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (strlen(project_name_buf_) > 0 && strlen(project_location_buf_) > 0) {
            std::filesystem::path full_path = std::filesystem::path(project_location_buf_) / project_name_buf_;
            ImGui::TextWrapped("Project will be created at:");
            ImGui::TextWrapped("%s", full_path.string().c_str());
        }

        ImGui::Spacing();
        ImGui::Spacing();

        float button_width = 100.0f;
        float spacing = 20.0f;
        float total_width = button_width * 2 + spacing;
        float window_width = ImGui::GetWindowWidth();
        ImGui::SetCursorPosX((window_width - total_width) * 0.5f);

        bool can_create = strlen(project_name_buf_) > 0 && strlen(project_location_buf_) > 0;
        ImGui::BeginDisabled(!can_create);
        if (ImGui::Button("Create", ImVec2(button_width, 0))) {
            auto& pm = ProjectManager::Instance();
            if (pm.CreateProject(project_name_buf_, project_location_buf_)) {
                selected_project_path_ = pm.GetProjectFilePath();
                result_ = Result::ProjectSelected;
                show_create_dialog_ = false;
                spdlog::info("Created project from start page: {}", selected_project_path_);
            } else {
                spdlog::error("Failed to create project from start page");
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

void StartPage::OpenExistingProject() {
    auto result = FileDialogs::OpenProject();
    if (result) {
        OpenProject(*result);
    }
}

void StartPage::OpenProjectFolder() {
    auto selected_folder = FileDialogs::SelectFolder("Open CyxWiz Project Folder");
    if (!selected_folder) {
        return;
    }

    std::filesystem::path folder(*selected_folder);
    std::vector<std::filesystem::path> project_files;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(folder, ec)) {
        if (entry.is_regular_file() && entry.path().extension() == ".cyxwiz") {
            project_files.push_back(entry.path());
        }
    }

    if (ec) {
        spdlog::warn("Could not scan selected project folder {}: {}", folder.string(), ec.message());
        return;
    }

    if (project_files.empty()) {
        spdlog::warn("No .cyxwiz project file found in selected folder: {}", folder.string());
        return;
    }

    std::sort(project_files.begin(), project_files.end());
    if (project_files.size() > 1) {
        spdlog::warn("Multiple .cyxwiz project files found in {}; opening {}", folder.string(), project_files.front().string());
    }

    OpenProject(project_files.front().string());
}

void StartPage::ContinueWithoutProject() {
    result_ = Result::ContinueWithout;
    spdlog::info("Continuing without project");
}

} // namespace cyxwiz
