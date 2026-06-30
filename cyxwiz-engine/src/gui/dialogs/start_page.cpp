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
#include <array>
#include <ctime>
#include <string>

namespace cyxwiz {

namespace {

struct ProjectTemplateDefinition {
    const char* name;
    const char* description;
    const char* default_project_name;
};

static constexpr std::array<ProjectTemplateDefinition, 6> kProjectTemplates = {{
    {"Blank project", "General CyxWiz workspace.", "New CyxWiz Project"},
    {"Classic ML workflow", "Tabular-first workflow for data inspection, preprocessing, feature engineering, classical baselines, and evaluation.", "Classic ML Project"},
    {"Deep Learning workflow", "Neural-network workflow for dataset setup, shape-safe batching, model layers, loss, optimizer, training, and export.", "Deep Learning Project"},
    {"Tabular project", "Tables, preprocessing, classical ML, and evaluation.", "Tabular ML Project"},
    {"Vision project", "Image datasets, augmentation, model training, and evaluation.", "Vision ML Project"},
    {"NLP project", "Text datasets, tokenization, vectorization, and language workflows.", "NLP Project"}
}};

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

std::string ResolveStarterGraphPath(const char* filename) {
    namespace fs = std::filesystem;

    std::array<fs::path, 7> roots = {
        fs::current_path(),
        fs::current_path().parent_path(),
        fs::current_path().parent_path().parent_path(),
        fs::current_path().parent_path().parent_path().parent_path(),
        fs::current_path().parent_path().parent_path().parent_path().parent_path(),
        fs::current_path() / "cyxwiz-engine",
        fs::current_path() / ".." / "cyxwiz-engine"
    };

    if (auto* launch_cwd = std::getenv("CYXWIZ_LAUNCH_CWD")) {
        fs::path candidate = fs::path(launch_cwd) / "examples" / "cyxgraph" / filename;
        std::error_code ec;
        if (fs::exists(candidate, ec)) {
            return fs::weakly_canonical(candidate, ec).string();
        }
    }

    for (const auto& root : roots) {
        fs::path candidate = root / "examples" / "cyxgraph" / filename;
        std::error_code ec;
        if (fs::exists(candidate, ec)) {
            return fs::weakly_canonical(candidate, ec).string();
        }
    }

    return {};
}

} // namespace

StartPage::StartPage() {
    LoadRecentProjects();
    LoadStarterGraphs();
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

void StartPage::LoadStarterGraphs() {
    struct StarterDefinition {
        const char* title;
        const char* description;
        const char* domain;
        const char* icon;
        const char* filename;
    };

    static constexpr std::array<StarterDefinition, 5> starters = {{
        {"Binary image classification", "Cats-vs-dogs training graph for a two-class image dataset.", "Binary classification", ICON_FA_IMAGES, "cats_dogs_classifier.cyxgraph"},
        {"Multiclass image classification", "MNIST MLP graph for a tabular digit dataset with ten classes.", "Multiclass classification", ICON_FA_IMAGE, "mnist_mlp.cyxgraph"},
        {"Text classification", "Call-center sentiment graph for customer conversation labels.", "Text classification", ICON_FA_COMMENTS, "call_center_sentiment.cyxgraph"},
        {"Audio classification", "Speech-command graph for labeled command utterances.", "Audio classification", ICON_FA_WAVE_SQUARE, "speech_command_classifier.cyxgraph"},
        {"Time-series forecasting", "Airline-passengers dense forecaster using a real time-series training graph.", "Forecasting / regression", ICON_FA_CHART_LINE, "timeseries/airline_passengers_dense.cyxgraph"}
    }};

    starter_graphs_.clear();
    for (const auto& starter : starters) {
        std::string path = ResolveStarterGraphPath(starter.filename);
        if (path.empty()) {
            continue;
        }

        starter_graphs_.push_back({
            starter.title,
            starter.description,
            starter.domain,
            starter.icon,
            path
        });
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

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const ImVec2 win_pos = ImGui::GetWindowPos();
    const ImVec2 win_size = ImGui::GetWindowSize();
    draw_list->AddRectFilledMultiColor(
        win_pos,
        ImVec2(win_pos.x + win_size.x, win_pos.y + win_size.y),
        IM_COL32(8, 13, 24, 255),
        IM_COL32(9, 28, 48, 255),
        IM_COL32(5, 8, 16, 255),
        IM_COL32(13, 20, 38, 255));
    draw_list->AddCircleFilled(
        ImVec2(win_pos.x + win_size.x * 0.78f, win_pos.y + 120.0f),
        180.0f,
        IM_COL32(0, 115, 255, 24),
        64);
    draw_list->AddCircleFilled(
        ImVec2(win_pos.x + 140.0f, win_pos.y + win_size.y * 0.82f),
        220.0f,
        IM_COL32(0, 210, 190, 14),
        64);

    // Custom styling for start page
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(12, 12));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(40, 30));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 14.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 9.0f);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.90f, 0.95f, 1.0f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.045f, 0.070f, 0.120f, 0.86f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.18f, 0.34f, 0.52f, 0.55f));
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.07f, 0.11f, 0.18f, 0.96f));

    // Title
    ImGui::SetCursorPos(ImVec2(40, 30));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
    ImGui::SetWindowFontScale(2.1f);
    ImGui::TextColored(ImVec4(0.92f, 0.97f, 1.0f, 1.0f), "Get started");
    ImGui::SetWindowFontScale(1.0f);
    ImGui::TextColored(ImVec4(0.45f, 0.74f, 1.0f, 1.0f),
                       "Build, train, debug, and export ML workflows from one engine workspace.");
    ImGui::PopStyleVar();

    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 10);

    // Two-column layout
    float window_width = ImGui::GetContentRegionAvail().x;
    float left_width = window_width * 0.65f;
    float right_width = window_width * 0.30f;

    ImGui::BeginGroup();
    {
        ImGui::BeginChild("LeftColumn", ImVec2(left_width, -60), true, ImGuiWindowFlags_NoScrollbar);
        {
            RenderSearchBar();
            ImGui::Spacing();
            RenderStarterGraphs();
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
        ImGui::BeginChild("RightColumn", ImVec2(right_width, -60), true);
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

    ImGui::PopStyleColor(4);
    ImGui::PopStyleVar(4);
    ImGui::End();

    return true;
}

void StartPage::RenderSearchBar() {
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.08f, 0.13f, 0.21f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.18f, 0.45f, 0.72f, 0.75f));
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
    ImGui::PopStyleColor(2);
}

void StartPage::RenderStarterGraphs() {
    if (starter_graphs_.empty()) {
        return;
    }

    ImGui::TextColored(ImVec4(0.92f, 0.97f, 1.0f, 1.0f), "Task starter graphs");
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.58f, 0.70f, 0.82f, 1.0f));
    ImGui::TextWrapped("Open a real CyxGraph template by prediction task.");
    ImGui::PopStyleColor();

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.035f, 0.055f, 0.095f, 0.58f));
    ImGui::BeginChild("##StarterGraphs", ImVec2(0, 230), true);

    const float button_width = 96.0f;
    const float row_height = 54.0f;
    const float button_x = ImGui::GetWindowContentRegionMax().x - button_width;
    const float text_width = std::max(120.0f, ImGui::GetContentRegionAvail().x - button_width - 28.0f);

    for (const auto& starter : starter_graphs_) {
        ImGui::PushID(starter.path.c_str());

        ImGui::BeginGroup();
        ImGui::TextColored(ImVec4(0.30f, 0.70f, 1.0f, 1.0f),
                           "%s", starter.icon.c_str());
        ImGui::SameLine();

        ImGui::BeginGroup();
        ImGui::Text("%s", starter.title.c_str());
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.30f, 0.72f, 1.0f, 1.0f));
        ImGui::Text("%s", starter.domain.c_str());
        ImGui::PopStyleColor();
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + text_width);
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.62f, 0.72f, 0.82f, 1.0f));
        ImGui::TextWrapped("%s", starter.description.c_str());
        ImGui::PopStyleColor();
        ImGui::PopTextWrapPos();
        ImGui::EndGroup();
        ImGui::EndGroup();

        ImGui::SameLine(button_x);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.04f, 0.36f, 0.88f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.07f, 0.48f, 1.0f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.02f, 0.26f, 0.70f, 1.0f));
        if (ImGui::Button(ICON_FA_DIAGRAM_PROJECT " Open", ImVec2(button_width, row_height))) {
            OpenStarterGraph(starter);
        }
        ImGui::PopStyleColor(3);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", starter.path.c_str());
        }

        ImGui::Separator();
        ImGui::PopID();
    }

    ImGui::EndChild();
    ImGui::PopStyleColor();
}

void StartPage::RenderRecentProjects() {
    ImGui::SeparatorText("Recent Projects");
    ImGui::BeginChild("##RecentProjects", ImVec2(0, 0), false);

    // This week
    if (!this_week_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0, 0, 0, 0));
        if (ImGui::CollapsingHeader(ICON_FA_CHEVRON_DOWN " This week", &show_this_week_, ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Indent(20);
            for (const auto& proj : this_week_) {
                ImGui::PushID(proj.path.c_str());

                // Project icon
                ImGui::TextColored(ImVec4(0.34f, 0.72f, 1.0f, 1.0f), ICON_FA_FOLDER);
                ImGui::SameLine();

                // Clickable project name
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.88f, 0.94f, 1.0f, 1.0f));
                if (ImGui::Selectable(proj.name.c_str(), false, ImGuiSelectableFlags_AllowDoubleClick)) {
                    if (ImGui::IsMouseDoubleClicked(0)) {
                        OpenProject(proj.path);
                    }
                }
                ImGui::PopStyleColor();

                // Path and timestamp
                ImGui::Indent(30);
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.50f, 0.60f, 0.72f, 1.0f));
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

                ImGui::TextColored(ImVec4(0.34f, 0.72f, 1.0f, 1.0f), ICON_FA_FOLDER);
                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.88f, 0.94f, 1.0f, 1.0f));
                if (ImGui::Selectable(proj.name.c_str(), false, ImGuiSelectableFlags_AllowDoubleClick)) {
                    if (ImGui::IsMouseDoubleClicked(0)) {
                        OpenProject(proj.path);
                    }
                }
                ImGui::PopStyleColor();

                ImGui::Indent(30);
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.50f, 0.60f, 0.72f, 1.0f));
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

                ImGui::TextColored(ImVec4(0.34f, 0.72f, 1.0f, 1.0f), ICON_FA_FOLDER);
                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.88f, 0.94f, 1.0f, 1.0f));
                if (ImGui::Selectable(proj.name.c_str(), false, ImGuiSelectableFlags_AllowDoubleClick)) {
                    if (ImGui::IsMouseDoubleClicked(0)) {
                        OpenProject(proj.path);
                    }
                }
                ImGui::PopStyleColor();

                ImGui::Indent(30);
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.50f, 0.60f, 0.72f, 1.0f));
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
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.50f, 0.60f, 0.72f, 1.0f));
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

    ImGui::TextColored(ImVec4(0.92f, 0.97f, 1.0f, 1.0f), "Launch workspace");
    ImGui::TextColored(ImVec4(0.58f, 0.70f, 0.82f, 1.0f),
                       "Start from a workflow lane, domain starter, or existing project.");
    ImGui::Spacing();

    // Create new project
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.02f, 0.34f, 0.92f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.05f, 0.48f, 1.0f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.01f, 0.25f, 0.72f, 1.0f));
    if (ImGui::Button(ICON_FA_PLUS " Create a new project", ImVec2(button_width, 0))) {
        CreateNewProject();
    }
    ImGui::PopStyleColor(3);

    ImGui::Spacing();

    ImGui::SeparatorText("Workflow lanes");
    RenderProjectTemplateButton(ICON_FA_CHART_LINE " Classic ML workflow", 1);
    RenderProjectTemplateButton(ICON_FA_NETWORK_WIRED " Deep Learning workflow", 2);

    ImGui::Spacing();

    ImGui::SeparatorText("Domain starters");
    RenderProjectTemplateButton(ICON_FA_TABLE " Tabular project", 3);
    RenderProjectTemplateButton(ICON_FA_IMAGE " Vision project", 4);
    RenderProjectTemplateButton(ICON_FA_COMMENTS " NLP project", 5);

    ImGui::Spacing();

    // Open project
    ImGui::SeparatorText("File actions");
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.08f, 0.13f, 0.21f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.12f, 0.20f, 0.32f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.06f, 0.10f, 0.18f, 1.0f));
    if (ImGui::Button(ICON_FA_FOLDER_OPEN " Open a project or solution", ImVec2(button_width, 0))) {
        OpenExistingProject();
    }
    ImGui::PopStyleColor(3);

    ImGui::Spacing();

    // Open folder
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.08f, 0.13f, 0.21f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.12f, 0.20f, 0.32f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.06f, 0.10f, 0.18f, 1.0f));
    if (ImGui::Button(ICON_FA_FOLDER " Open a project folder", ImVec2(button_width, 0))) {
        OpenProjectFolder();
    }
    ImGui::PopStyleColor(3);

    ImGui::Spacing();

    // Clone repository
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.08f, 0.13f, 0.21f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.12f, 0.20f, 0.32f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.06f, 0.10f, 0.18f, 1.0f));
    ImGui::BeginDisabled();
    ImGui::Button(ICON_FA_CLOUD_ARROW_DOWN " Clone a repository (planned)", ImVec2(button_width, 0));
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("Repository cloning is not wired yet");
    }
    ImGui::PopStyleColor(3);

    ImGui::PopStyleVar(2);
}

void StartPage::RenderProjectTemplateButton(const char* label, int template_index) {
    float button_width = ImGui::GetContentRegionAvail().x;
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.07f, 0.12f, 0.20f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.10f, 0.22f, 0.36f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.05f, 0.16f, 0.28f, 1.0f));
    if (ImGui::Button(label, ImVec2(button_width, 0))) {
        CreateNewProjectFromTemplate(template_index);
    }
    ImGui::PopStyleColor(3);
    if (ImGui::IsItemHovered() &&
        template_index >= 0 &&
        template_index < static_cast<int>(kProjectTemplates.size())) {
        ImGui::SetTooltip("%s", kProjectTemplates[template_index].description);
    }
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

void StartPage::OpenStarterGraph(const StarterGraph& starter) {
    selected_graph_path_ = starter.path;
    result_ = Result::ExampleGraphSelected;
    spdlog::info("Opening starter graph: {}", starter.path);
}

void StartPage::CreateNewProject() {
    selected_project_template_index_ = 0;
    show_create_dialog_ = true;
}

void StartPage::CreateNewProjectFromTemplate(int template_index) {
    selected_project_template_index_ = std::clamp(
        template_index,
        0,
        static_cast<int>(kProjectTemplates.size()) - 1);

    if (project_name_buf_[0] == '\0') {
        strncpy(
            project_name_buf_,
            kProjectTemplates[selected_project_template_index_].default_project_name,
            sizeof(project_name_buf_) - 1);
        project_name_buf_[sizeof(project_name_buf_) - 1] = '\0';
    }

    show_create_dialog_ = true;
}

void StartPage::RenderCreateProjectDialog() {
    ImGui::OpenPopup("Create New Project");

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(520, 340), ImGuiCond_Appearing);

    if (ImGui::BeginPopupModal("Create New Project", &show_create_dialog_, ImGuiWindowFlags_NoResize)) {
        ImGui::Text("Project Template:");
        const char* template_names[kProjectTemplates.size()] = {};
        for (size_t i = 0; i < kProjectTemplates.size(); ++i) {
            template_names[i] = kProjectTemplates[i].name;
        }
        ImGui::SetNextItemWidth(-1);
        ImGui::Combo(
            "##ProjectTemplate",
            &selected_project_template_index_,
            template_names,
            static_cast<int>(kProjectTemplates.size()));
        ImGui::TextDisabled("%s", kProjectTemplates[selected_project_template_index_].description);

        ImGui::Spacing();

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
                pm.GetConfig().description = kProjectTemplates[selected_project_template_index_].description;
                pm.SaveProject();
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

    auto project_file = ProjectManager::ResolveProjectFilePath(*selected_folder);
    if (!project_file) {
        spdlog::warn("No .cyxwiz project file found in selected folder: {}", *selected_folder);
        return;
    }

    OpenProject(*project_file);
}

void StartPage::ContinueWithoutProject() {
    result_ = Result::ContinueWithout;
    spdlog::info("Continuing without project");
}

} // namespace cyxwiz
