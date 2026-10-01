#include "start_page.h"

#include "../../core/file_dialogs.h"
#include "../../core/project_manager.h"
#include "../../core/start_page_presentation.h"
#include "../../core/window_manager.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_fonts.h"
#include "../ui_platform.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <cyxwiz/cyxwiz.h>
#include <imgui.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <ctime>
#include <filesystem>

namespace cyxwiz {

namespace {

std::string ResolveStarterGraphPath(const char* filename) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (const char* launch_cwd = std::getenv("CYXWIZ_LAUNCH_CWD")) {
        const fs::path candidate = fs::path(launch_cwd) / "examples" / "cyxgraph" / filename;
        if (fs::exists(candidate, ec)) return fs::weakly_canonical(candidate, ec).string();
    }
    const fs::path cwd = fs::current_path(ec);
    const std::array<fs::path, 7> roots = {cwd,
                                           cwd.parent_path(),
                                           cwd.parent_path().parent_path(),
                                           cwd.parent_path().parent_path().parent_path(),
                                           cwd.parent_path().parent_path().parent_path().parent_path(),
                                           cwd / "cyxwiz-engine",
                                           cwd / ".." / "cyxwiz-engine"};
    for (const auto& root : roots) {
        const fs::path candidate = root / "examples" / "cyxgraph" / filename;
        if (fs::exists(candidate, ec)) return fs::weakly_canonical(candidate, ec).string();
    }
    return {};
}

// Panel: a filled, bordered child with a padded header row.
bool BeginPanel(const char* id, float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, t.rounding_card);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 1.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(t.space_lg, t.space_lg));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_panel);
    ImGui::PushStyleColor(ImGuiCol_Border, t.border_soft);
    return ImGui::BeginChild(id, ImVec2(0.0f, height), ImGuiChildFlags_Borders | ImGuiChildFlags_AlwaysUseWindowPadding);
}

void EndPanel() {
    ImGui::EndChild();
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(3);
}

}  // namespace

StartPage::StartPage() {
    LoadRecentProjects();
    LoadStarterGraphs();
}

void StartPage::LoadRecentProjects() {
    recent_.clear();
    std::error_code ec;
    for (const auto& rp : ProjectManager::Instance().GetRecentProjects()) {
        if (std::filesystem::exists(rp.path, ec)) recent_.push_back({rp.name, rp.path, static_cast<long long>(rp.last_opened)});
    }
}

void StartPage::LoadStarterGraphs() {
    struct Def {
        const char* title;
        const char* description;
        const char* domain;
        const char* icon;
        const char* filename;
    };
    static constexpr std::array<Def, 5> defs = {{
        {"Binary image classification", "Cats and dogs: a small convolutional classifier.", "Vision", ICON_FA_IMAGES, "cats_dogs_classifier.cyxgraph"},
        {"Multiclass image classification", "MNIST digits with a dense network.", "Vision", ICON_FA_IMAGE, "mnist_mlp.cyxgraph"},
        {"Text classification", "Call-centre sentiment from conversation transcripts.", "Text", ICON_FA_COMMENTS, "call_center_sentiment.cyxgraph"},
        {"Audio classification", "Speech commands from short labelled recordings.", "Audio", ICON_FA_WAVE_SQUARE, "speech_command_classifier.cyxgraph"},
        {"Time-series forecasting", "Airline passengers with a dense forecaster.", "Time series", ICON_FA_CHART_LINE, "timeseries/airline_passengers_dense.cyxgraph"}}};
    starter_graphs_.clear();
    for (const auto& d : defs) {
        const std::string path = ResolveStarterGraphPath(d.filename);
        if (!path.empty()) starter_graphs_.push_back({d.title, d.description, d.domain, d.icon, path});
    }
}

void StartPage::SetStatus(std::string text, bool problem) {
    status_ = std::move(text);
    status_problem_ = problem;
}

// ---------------------------------------------------------------------------

bool StartPage::Render() {
    if (result_ != Result::InProgress) return false;

    const ui::Tokens& t = ui::CurrentTokens();
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, t.bg_window);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus |
                                   ImGuiWindowFlags_NoDocking;
    ImGui::Begin("##StartPage", nullptr, flags);
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);

    RenderHeader();

    const float footer_h = ImGui::GetFrameHeight() + t.space_sm;
    const float margin_x = t.space_xl * 2.0f;
    ImGui::SetCursorPos(ImVec2(margin_x, ImGui::GetCursorPosY() + t.space_xl));
    const float body_h = ImGui::GetContentRegionAvail().y - footer_h - t.space_lg;
    const float body_w = ImGui::GetContentRegionAvail().x - margin_x;

    if (ImGui::BeginTable("##body", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings,
                          ImVec2(body_w, body_h))) {
        ImGui::TableSetupColumn("recent", ImGuiTableColumnFlags_WidthStretch, 1.4f);
        ImGui::TableSetupColumn("start", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        RenderRecentProjects(body_h);
        ImGui::TableNextColumn();
        const float column_top = ImGui::GetCursorPosY();
        RenderStartActions();
        ImGui::Spacing();
        RenderStarterGraphs(body_h - (ImGui::GetCursorPosY() - column_top));
        ImGui::EndTable();
    }

    RenderFooter();
    HandleKeys();

    if (create_dialog_.Render() == CreateProjectDialog::Result::Created) {
        selected_project_path_ = create_dialog_.CreatedProjectPath();
        result_ = Result::ProjectSelected;
    }

    ImGui::End();
    return result_ == Result::InProgress;
}

void StartPage::RenderHeader() {
    const ui::Tokens& t = ui::CurrentTokens();
    const float height = ImGui::GetFrameHeight() + t.space_xl * 1.5f;
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float width = ImGui::GetWindowWidth();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(origin, ImVec2(origin.x + width, origin.y + height), ui::ToU32(t.bg_bar));
    dl->AddLine(ImVec2(origin.x, origin.y + height - 1.0f), ImVec2(origin.x + width, origin.y + height - 1.0f), ui::ToU32(t.border_soft));

    const float margin_x = t.space_xl * 2.0f;
    ImGui::SetCursorScreenPos(ImVec2(origin.x + margin_x, origin.y + (height - ImGui::GetFrameHeight()) * 0.5f));
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.accent_text, "%s", ICON_FA_DIAGRAM_PROJECT);
    ImGui::SameLine(0.0f, t.space_md);
    {
        ui::FontScope medium(ui::Font::Medium);
        ImGui::TextColored(t.text_bright, "CyxWiz Engine");
    }
    ImGui::SameLine(0.0f, t.space_md);
    ImGui::TextColored(t.text_dim, "Version %s", GetVersionString());

    if (!python_.text.empty()) {
        const ImVec4 dot = python_.level == 0 ? t.success : (python_.level == 1 ? t.running : t.warning);
        const float chip_w = ImGui::CalcTextSize(python_.text.c_str()).x + t.space_xl * 2.0f + 8.0f;
        ImGui::SameLine(width - margin_x - chip_w);
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const ImVec2 size(chip_w, ImGui::GetFrameHeight());
        if (ImGui::InvisibleButton("##python_status", size) && python_.on_click) python_.on_click();
        const bool hovered = ImGui::IsItemHovered();
        dl->AddRectFilled(p, ImVec2(p.x + size.x, p.y + size.y), ui::ToU32(hovered ? t.hover : ImVec4(0, 0, 0, 0)), size.y * 0.5f);
        dl->AddRect(p, ImVec2(p.x + size.x, p.y + size.y), ui::ToU32(t.border), size.y * 0.5f);
        dl->AddCircleFilled(ImVec2(p.x + t.space_lg + 4.0f, p.y + size.y * 0.5f), 4.0f, ui::ToU32(dot));
        dl->AddText(ImVec2(p.x + t.space_lg + 8.0f + t.space_sm, p.y + (size.y - ImGui::GetTextLineHeight()) * 0.5f),
                    ui::ToU32(t.text), python_.text.c_str());
        if (hovered) ImGui::SetTooltip("Python for scripts, the Python console and project environments. Click for details.");
    }
    ImGui::SetCursorScreenPos(ImVec2(origin.x, origin.y + height));
}

void StartPage::RenderRecentProjects(float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!BeginPanel("##recent_panel", height)) {
        EndPanel();
        return;
    }
    std::vector<startpage::RecentEntry> entries;
    for (const auto& r : recent_) entries.push_back({r.name, r.path, static_cast<std::time_t>(r.last_opened)});
    const startpage::RecentView view = startpage::BuildRecentView(entries, search_, std::time(nullptr));

    ui::SectionHeader("Recent projects", (std::to_string(view.total) + (view.total == 1 ? " project" : " projects")).c_str());
    ImGui::SameLine();
    const float search_w = std::min(260.0f, ImGui::GetContentRegionAvail().x * 0.5f);
    ImGui::SetCursorPosX(ImGui::GetWindowContentRegionMax().x - search_w);
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() - t.space_sm);
    ui::SearchField("##recent_search", search_, sizeof(search_), "Search by name or folder", search_w);
    ImGui::Spacing();

    const float hint_h = ImGui::GetTextLineHeightWithSpacing() + t.space_md;
    ImGui::BeginChild("##recent_list", ImVec2(0.0f, -hint_h), ImGuiChildFlags_None);
    if (view.no_projects) {
        ui::EmptyState(ICON_FA_FOLDER_OPEN, "No recent projects", "Create a new project or open an existing one to get started.");
    } else if (view.no_match) {
        ui::EmptyState(ICON_FA_MAGNIFYING_GLASS, (std::string("No recent project matches \"") + search_ + "\"").c_str(), nullptr);
    }
    const bool searching = search_[0] != '\0';
    for (const auto& group : view.groups) {
        ImGui::PushID(group.key.c_str());
        const bool collapsed = !searching && collapsed_groups_.count(group.key) > 0;
        const std::string header = std::string(collapsed ? ICON_FA_CHEVRON_RIGHT : ICON_FA_CHEVRON_DOWN) + "  " + group.label +
                                   "   " + std::to_string(group.rows.size());
        ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
        if (ImGui::Selectable(header.c_str(), false, ImGuiSelectableFlags_None)) {
            if (collapsed) collapsed_groups_.erase(group.key);
            else collapsed_groups_.insert(group.key);
        }
        ImGui::PopStyleColor();
        if (!collapsed) {
            for (const auto& row : group.rows) {
                ImGui::PushID(row.path.c_str());
                const bool selected = selected_path_ == row.path;
                const float row_h = ImGui::GetTextLineHeight() * 2.0f + t.space_md * 2.0f;
                const float actions_w = selected ? ui::ButtonWidth("Open", ui::ButtonSize::Small) +
                                                       ui::ButtonWidth("Actions", ui::ButtonSize::Small) + t.space_md * 2.0f
                                                 : 0.0f;
                const ImVec2 start = ImGui::GetCursorScreenPos();
                const float full_w = ImGui::GetContentRegionAvail().x;
                if (ImGui::Selectable("##row", selected, ImGuiSelectableFlags_AllowDoubleClick | ImGuiSelectableFlags_AllowOverlap,
                                      ImVec2(full_w, row_h))) {
                    selected_path_ = row.path;
                    menu_for_path_.clear();
                    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) OpenProject(row.path);
                    else SetStatus("Selected " + row.name + ". Press Enter or Open.");
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", row.path.c_str());
                ImDrawList* dl = ImGui::GetWindowDrawList();
                const float text_x = start.x + t.space_md + ImGui::GetFontSize() * 1.5f;
                dl->AddText(ImVec2(start.x + t.space_md, start.y + t.space_md), ui::ToU32(selected ? t.accent_text : t.text_dim), ICON_FA_FOLDER);
                const float when_w = ImGui::CalcTextSize(row.when.c_str()).x;
                const float text_right = start.x + full_w - actions_w - when_w - t.space_lg * 2.0f;
                ImGui::PushClipRect(start, ImVec2(text_right, start.y + row_h), true);
                dl->AddText(ImVec2(text_x, start.y + t.space_md), ui::ToU32(t.text_bright), row.name.c_str());
                dl->AddText(ImVec2(text_x, start.y + t.space_md + ImGui::GetTextLineHeight()), ui::ToU32(t.text_dim), row.folder.c_str());
                ImGui::PopClipRect();
                dl->AddText(ImVec2(start.x + full_w - actions_w - when_w - t.space_lg, start.y + (row_h - ImGui::GetTextLineHeight()) * 0.5f),
                            ui::ToU32(t.text_dim), row.when.c_str());
                if (selected) {
                    ImGui::SetCursorScreenPos(ImVec2(start.x + full_w - actions_w + t.space_md,
                                                     start.y + (row_h - ImGui::GetFrameHeight()) * 0.5f));
                    if (ui::PrimaryButton("Open", true, nullptr, ui::ButtonSize::Small)) OpenProject(row.path);
                    ImGui::SameLine(0.0f, t.space_md);
                    if (ui::SecondaryButton("Actions")) ImGui::OpenPopup("##row_actions");
                    if (ImGui::BeginPopup("##row_actions")) {
                        if (ImGui::MenuItem("Open in new window")) {
                            if (core::WindowManager::LaunchWindow(row.path)) SetStatus("Opened " + row.name + " in a new window.");
                            else SetStatus("Could not start a new window for " + row.name + ".", true);
                        }
                        if (ImGui::MenuItem("Show in folder")) {
                            if (!ui::ShowInFileManager(row.path)) SetStatus("Could not open the file manager for " + row.folder + ".", true);
                        }
                        ImGui::Separator();
                        ImGui::PushStyleColor(ImGuiCol_Text, t.error);
                        if (ImGui::MenuItem("Remove from this list")) {
                            ProjectManager::Instance().RemoveRecentProject(row.path);
                            LoadRecentProjects();
                            selected_path_.clear();
                            SetStatus("Removed " + row.name + " from the list. The project itself is untouched.");
                        }
                        ImGui::PopStyleColor();
                        ImGui::EndPopup();
                    }
                    ImGui::SetCursorScreenPos(ImVec2(start.x, start.y + row_h + ImGui::GetStyle().ItemSpacing.y));
                    ImGui::Dummy(ImVec2(0.0f, 0.0f));
                }
                ImGui::PopID();
                if (result_ != Result::InProgress) break;
            }
        }
        ImGui::PopID();
        if (result_ != Result::InProgress) break;
    }
    ImGui::EndChild();
    ImGui::TextColored(t.text_dim, "Enter opens the selected project. Double-click a row to open it at once.");
    EndPanel();
}

void StartPage::RenderStartActions() {
    const ui::Tokens& t = ui::CurrentTokens();
    const float h = ImGui::GetFrameHeight() * 3.0f + ImGui::GetTextLineHeightWithSpacing() * 2.5f + t.space_lg * 4.0f;
    if (BeginPanel("##start_panel", h)) {
        ui::SectionHeader("Start");
        const float full = ImGui::GetContentRegionAvail().x;
        if (ui::PrimaryButton("Create a new project...  (Ctrl+N)", true, nullptr, ui::ButtonSize::Regular, full)) create_dialog_.Open(0);
        const float half = (full - t.space_md) * 0.5f;
        if (ui::SecondaryButton("Open a project...  (Ctrl+O)", true, nullptr, ui::ButtonSize::Regular, half)) OpenExistingProject();
        ImGui::SameLine(0.0f, t.space_md);
        if (ui::SecondaryButton("Open a project folder...", true, nullptr, ui::ButtonSize::Regular, half)) OpenProjectFolder();
        ImGui::Spacing();
        if (ui::LinkButton("Continue without a project")) {
            result_ = Result::ContinueWithout;
            spdlog::info("Continuing without project");
        }
        ui::Tooltip("Opens the workspace with no project. Saving will ask where to put your work.");
        ImGui::SameLine(0.0f, t.space_xl);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_dim, "Clone a repository");
        ImGui::SameLine(0.0f, t.space_sm);
        ui::StatusChip(ui::Status::NotVerifiedYet, "Planned");
    }
    EndPanel();
}

void StartPage::RenderStarterGraphs(float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!BeginPanel("##starter_panel", height)) {
        EndPanel();
        return;
    }
    ui::SectionHeader("Task starter graphs", "One CyxGraph template per prediction task");
    const char* hint = "A starter opens in the Studio without a project. Save it into a project from File > Save.";
    const float hint_h = ImGui::CalcTextSize(hint, nullptr, false, ImGui::GetContentRegionAvail().x).y +
                         ImGui::GetStyle().ItemSpacing.y * 2.0f + t.space_md;
    ImGui::BeginChild("##starter_list", ImVec2(0.0f, -hint_h), ImGuiChildFlags_None);
    if (starter_graphs_.empty()) {
        ui::EmptyState(ICON_FA_DIAGRAM_PROJECT, "No starter graphs found",
                       "The examples/cyxgraph folder is not next to this Engine.");
    }
    for (const auto& s : starter_graphs_) {
        ImGui::PushID(s.path.c_str());
        const float open_w = ui::ButtonWidth("Open", ui::ButtonSize::Small);
        if (ImGui::BeginTable("##starter", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings)) {
            ImGui::TableSetupColumn("text", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("action", ImGuiTableColumnFlags_WidthFixed, open_w);
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextColored(t.text_dim, "%s", s.icon.c_str());
            ImGui::SameLine(0.0f, t.space_sm);
            ImGui::TextColored(t.text_bright, "%s", s.title.c_str());
            ImGui::SameLine(0.0f, t.space_sm);
            ImGui::TextColored(t.accent_text, "%s", s.domain.c_str());
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.text_dim, "%s", s.description.c_str());
            ImGui::PopTextWrapPos();
            ImGui::TableNextColumn();
            if (ui::SecondaryButton("Open")) {
                selected_graph_path_ = s.path;
                result_ = Result::ExampleGraphSelected;
                spdlog::info("Opening starter graph: {}", s.path);
            }
            ui::Tooltip(s.path.c_str());
            ImGui::EndTable();
        }
        ImGui::Spacing();
        ImGui::PopID();
    }
    ImGui::EndChild();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim, "%s", hint);
    ImGui::PopTextWrapPos();
    EndPanel();
}

void StartPage::RenderFooter() {
    const ui::Tokens& t = ui::CurrentTokens();
    const float h = ImGui::GetFrameHeight() + t.space_sm;
    const ImVec2 win = ImGui::GetWindowPos();
    const ImVec2 size = ImGui::GetWindowSize();
    const ImVec2 origin(win.x, win.y + size.y - h);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(origin, ImVec2(origin.x + size.x, origin.y + h), ui::ToU32(t.bg_bar));
    dl->AddLine(origin, ImVec2(origin.x + size.x, origin.y), ui::ToU32(t.border_soft));
    const float margin_x = t.space_xl * 2.0f;
    const float text_y = origin.y + (h - ImGui::GetTextLineHeight()) * 0.5f;
    const char* keys = "Ctrl+N new project   Ctrl+O open   Enter open selected";
    const float keys_w = ImGui::CalcTextSize(keys).x;
    ImGui::PushClipRect(origin, ImVec2(origin.x + size.x - margin_x - keys_w - t.space_xl, origin.y + h), true);
    dl->AddText(ImVec2(origin.x + margin_x, text_y), ui::ToU32(status_problem_ ? t.warning : t.text_dim), status_.c_str());
    ImGui::PopClipRect();
    dl->AddText(ImVec2(origin.x + size.x - margin_x - keys_w, text_y), ui::ToU32(t.text_dim), keys);
}

void StartPage::HandleKeys() {
    if (create_dialog_.IsOpen() || ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId)) return;
    const ImGuiIO& io = ImGui::GetIO();
    if (io.KeyCtrl && !io.KeyShift && ImGui::IsKeyPressed(ImGuiKey_N, false)) create_dialog_.Open(0);
    if (io.KeyCtrl && !io.KeyShift && ImGui::IsKeyPressed(ImGuiKey_O, false)) OpenExistingProject();
    if (!io.KeyCtrl && !io.WantTextInput && !selected_path_.empty() &&
        (ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false)))
        OpenProject(selected_path_);
}

// ---------------------------------------------------------------------------

void StartPage::OpenProject(const std::string& path) {
    if (ProjectManager::Instance().OpenProject(path)) {
        selected_project_path_ = path;
        result_ = Result::ProjectSelected;
        spdlog::info("Opening project: {}", path);
    } else {
        SetStatus("Could not open " + path + ". The file may be missing or damaged; details are in the log.", true);
        spdlog::error("Failed to open project: {}", path);
    }
}

void StartPage::OpenExistingProject() {
    if (auto result = FileDialogs::OpenProject()) OpenProject(*result);
}

void StartPage::OpenProjectFolder() {
    auto folder = FileDialogs::SelectFolder("Open CyxWiz Project Folder");
    if (!folder) return;
    auto project_file = ProjectManager::ResolveProjectFilePath(*folder);
    if (!project_file) {
        SetStatus("No .cyxwiz project file in " + *folder + ". Choose the folder that holds one, or create a project.", true);
        spdlog::warn("No .cyxwiz project file found in selected folder: {}", *folder);
        return;
    }
    OpenProject(*project_file);
}

}  // namespace cyxwiz
