#include "start_page.h"

#include "../../core/file_dialogs.h"
#include "../../core/project_manager.h"
#include "../../core/start_page_presentation.h"
#include "../../core/window_manager.h"
#include "../icons.h"
#include "../ui_fonts.h"
#include "../ui_platform.h"
#include "../separate_windows.h"

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

// The start page keeps its own launch look (owner choice, TOFIX129 A2):
// deep navy gradient, blue actions. Every colour of the page is here.
struct StartPalette {
    ImVec4 title = ImVec4(0.92f, 0.97f, 1.0f, 1.0f);
    ImVec4 subtitle = ImVec4(0.45f, 0.74f, 1.0f, 1.0f);
    ImVec4 text = ImVec4(0.90f, 0.95f, 1.0f, 1.0f);
    ImVec4 dim = ImVec4(0.58f, 0.70f, 0.82f, 1.0f);
    ImVec4 faint = ImVec4(0.42f, 0.52f, 0.64f, 1.0f);
    ImVec4 domain = ImVec4(0.30f, 0.72f, 1.0f, 1.0f);
    ImVec4 icon = ImVec4(0.30f, 0.70f, 1.0f, 1.0f);
    ImVec4 field = ImVec4(0.08f, 0.13f, 0.21f, 1.0f);
    ImVec4 row_hover = ImVec4(0.10f, 0.18f, 0.30f, 0.70f);
    ImVec4 row_selected = ImVec4(0.04f, 0.30f, 0.70f, 0.35f);
    ImVec4 primary = ImVec4(0.02f, 0.34f, 0.92f, 1.0f);
    ImVec4 primary_hover = ImVec4(0.05f, 0.48f, 1.0f, 1.0f);
    ImVec4 primary_active = ImVec4(0.01f, 0.25f, 0.72f, 1.0f);
    ImVec4 lane = ImVec4(0.07f, 0.12f, 0.20f, 1.0f);
    ImVec4 lane_hover = ImVec4(0.10f, 0.22f, 0.36f, 1.0f);
    ImVec4 lane_active = ImVec4(0.05f, 0.16f, 0.28f, 1.0f);
    ImVec4 popup = ImVec4(0.06f, 0.10f, 0.17f, 0.98f);
    ImVec4 ok = ImVec4(0.24f, 0.84f, 0.55f, 1.0f);
    ImVec4 busy = ImVec4(0.45f, 0.74f, 1.0f, 1.0f);
    ImVec4 warn = ImVec4(0.95f, 0.70f, 0.30f, 1.0f);
    ImVec4 danger = ImVec4(1.0f, 0.48f, 0.45f, 1.0f);
};
const StartPalette kPal;

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

ImU32 U32(const ImVec4& c) { return ImGui::ColorConvertFloat4ToU32(c); }

// Full-width button in the launch column (icon + label, left aligned).
bool LaneButton(const char* label, const ImVec4& bg, const ImVec4& hover, const ImVec4& active, bool enabled = true) {
    ImGui::PushStyleColor(ImGuiCol_Button, bg);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hover);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, active);
    if (!enabled) ImGui::BeginDisabled();
    const bool clicked = ImGui::Button(label, ImVec2(ImGui::GetContentRegionAvail().x, 0.0f));
    if (!enabled) ImGui::EndDisabled();
    ImGui::PopStyleColor(3);
    return clicked;
}

bool BlueButton(const char* label, const ImVec2& size = ImVec2(0.0f, 0.0f)) {
    ImGui::PushStyleColor(ImGuiCol_Button, kPal.primary);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, kPal.primary_hover);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, kPal.primary_active);
    const bool clicked = ImGui::Button(label, size);
    ImGui::PopStyleColor(3);
    return clicked;
}

float ContentWidth(float window_width) { return std::min(1240.0f, std::max(640.0f, window_width - 128.0f)); }

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
        {"Binary image classification", "Cats-vs-dogs training graph for a two-class image dataset.", "Binary classification", ICON_FA_IMAGES, "cats_dogs_classifier.cyxgraph"},
        {"Multiclass image classification", "MNIST MLP graph for a tabular digit dataset with ten classes.", "Multiclass classification", ICON_FA_IMAGE, "mnist_mlp.cyxgraph"},
        {"Text classification", "Call-center sentiment graph for customer conversation labels.", "Text classification", ICON_FA_COMMENTS, "call_center_sentiment.cyxgraph"},
        {"Audio classification", "Speech-command graph for labeled command utterances.", "Audio classification", ICON_FA_WAVE_SQUARE, "speech_command_classifier.cyxgraph"},
        {"Time-series forecasting", "Airline-passengers dense forecaster using a real time-series training graph.", "Forecasting / regression", ICON_FA_CHART_LINE, "timeseries/airline_passengers_dense.cyxgraph"}}};
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

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ::gui::NextWindowStaysInMain();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus |
                                   ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoDocking;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("##StartPage", nullptr, flags);
    ImGui::PopStyleVar();

    // Background: deep navy gradient with two soft glows.
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 win_pos = ImGui::GetWindowPos();
    const ImVec2 win_size = ImGui::GetWindowSize();
    dl->AddRectFilledMultiColor(win_pos, ImVec2(win_pos.x + win_size.x, win_pos.y + win_size.y), IM_COL32(8, 13, 24, 255),
                                IM_COL32(9, 28, 48, 255), IM_COL32(5, 8, 16, 255), IM_COL32(13, 20, 38, 255));
    dl->AddCircleFilled(ImVec2(win_pos.x + win_size.x * 0.78f, win_pos.y + 120.0f), 180.0f, IM_COL32(0, 115, 255, 24), 64);
    dl->AddCircleFilled(ImVec2(win_pos.x + 140.0f, win_pos.y + win_size.y * 0.82f), 220.0f, IM_COL32(0, 210, 190, 14), 64);

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(12.0f, 10.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 9.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, 8.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, 8.0f);
    ImGui::PushStyleColor(ImGuiCol_Text, kPal.text);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_FrameBg, kPal.field);
    ImGui::PushStyleColor(ImGuiCol_PopupBg, kPal.popup);
    ImGui::PushStyleColor(ImGuiCol_Header, kPal.row_selected);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, kPal.row_hover);
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, kPal.row_selected);
    ImGui::PushStyleColor(ImGuiCol_ScrollbarBg, ImVec4(0, 0, 0, 0));

    // Centred content: left 60 % (starters, recents), gap, launch column.
    const float content_width = ContentWidth(win_size.x);
    const float content_x = (win_size.x - content_width) * 0.5f;
    const float gap = 64.0f;
    const float left_width = content_width * 0.60f;
    const float right_width = content_width - left_width - gap;
    const float bottom_bar = 60.0f;

    RenderHeader();

    const float columns_top = ImGui::GetCursorPosY() + 8.0f;
    const float columns_h = std::max(120.0f, win_size.y - columns_top - bottom_bar);

    // Both columns scroll, so nothing is cut off at small window sizes.
    ImGui::SetCursorPos(ImVec2(content_x, columns_top));
    ImGui::BeginChild("##LeftColumn", ImVec2(left_width, columns_h), ImGuiChildFlags_None);
    RenderStarterGraphs(0.0f);
    RenderRecentProjects(0.0f);
    ImGui::EndChild();

    // The launch column starts under the version row, beside the title.
    const float right_top = 76.0f;
    ImGui::SetCursorPos(ImVec2(content_x + left_width + gap, right_top));
    ImGui::BeginChild("##RightColumn", ImVec2(right_width, std::max(120.0f, win_size.y - right_top - bottom_bar)),
                      ImGuiChildFlags_None);
    RenderStartActions();
    ImGui::EndChild();

    RenderFooter();
    HandleKeys();

    if (create_dialog_.Render() == CreateProjectDialog::Result::Created) {
        selected_project_path_ = create_dialog_.CreatedProjectPath();
        result_ = Result::ProjectSelected;
    }

    ImGui::PopStyleColor(9);
    ImGui::PopStyleVar(4);
    ImGui::End();
    return result_ == Result::InProgress;
}

void StartPage::RenderHeader() {
    const ImVec2 win_size = ImGui::GetWindowSize();
    const float content_width = ContentWidth(win_size.x);
    const float content_x = (win_size.x - content_width) * 0.5f;
    const float right_edge = content_x + content_width;

    // Title (heading font, so it stays sharp) and subtitle.
    ImGui::SetCursorPos(ImVec2(content_x, 28.0f));
    {
        ui::FontScope heading(ui::Font::Heading);
        ImGui::SetWindowFontScale(1.55f);
        ImGui::TextColored(kPal.title, "Get started");
        ImGui::SetWindowFontScale(1.0f);
    }
    ImGui::SetCursorPosX(content_x);
    ImGui::TextColored(kPal.subtitle, "Build, train, debug, and export ML workflows from one engine workspace.");
    const float after_subtitle = ImGui::GetCursorPosY();

    // Top right: version and the Python status chip.
    const std::string version = std::string("Version ") + GetVersionString();
    const float version_w = ImGui::CalcTextSize(version.c_str()).x;
    const float chip_y = 36.0f;
    float version_x = right_edge - version_w;
    if (!python_.text.empty()) {
        const float chip_w = ImGui::CalcTextSize(python_.text.c_str()).x + 40.0f;
        const float chip_x = right_edge - chip_w;
        version_x = chip_x - 16.0f - version_w;
        ImGui::SetCursorPos(ImVec2(chip_x, chip_y));
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const ImVec2 size(chip_w, ImGui::GetFrameHeight());
        if (ImGui::InvisibleButton("##python_status", size) && python_.on_click) python_.on_click();
        const bool hovered = ImGui::IsItemHovered();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(p, ImVec2(p.x + size.x, p.y + size.y), U32(hovered ? kPal.lane_hover : kPal.lane), size.y * 0.5f);
        const ImVec4 dot = python_.level == 0 ? kPal.ok : (python_.level == 1 ? kPal.busy : kPal.warn);
        dl->AddCircleFilled(ImVec2(p.x + 16.0f, p.y + size.y * 0.5f), 4.0f, U32(dot));
        dl->AddText(ImVec2(p.x + 28.0f, p.y + (size.y - ImGui::GetTextLineHeight()) * 0.5f), U32(kPal.text), python_.text.c_str());
        if (hovered) ImGui::SetTooltip("Python for scripts, the Python console and project environments. Click for details.");
    }
    ImGui::SetCursorPos(ImVec2(version_x, chip_y + (ImGui::GetFrameHeight() - ImGui::GetTextLineHeight()) * 0.5f));
    ImGui::TextColored(kPal.faint, "%s", version.c_str());

    // Search across the left column, under the title.
    ImGui::SetCursorPos(ImVec2(content_x, after_subtitle + 6.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(12.0f, 10.0f));
    ImGui::SetNextItemWidth(content_width * 0.60f);
    ImGui::InputTextWithHint("##recent_search", ICON_FA_MAGNIFYING_GLASS "  Search recent projects by name or folder", search_,
                             sizeof(search_));
    ImGui::PopStyleVar();
}

void StartPage::RenderStarterGraphs(float) {
    ImGui::TextColored(kPal.title, "Task starter graphs");
    ImGui::TextColored(kPal.dim, "Open a real CyxGraph template by prediction task.");
    if (starter_graphs_.empty()) {
        ImGui::TextColored(kPal.faint, "No starter graphs found: the examples/cyxgraph folder is not next to this Engine.");
        return;
    }
    const float button_w = 96.0f;
    const float button_h = 40.0f;
    for (const auto& s : starter_graphs_) {
        ImGui::PushID(s.path.c_str());
        const float button_x = ImGui::GetWindowContentRegionMax().x - button_w - 4.0f;
        const float row_y = ImGui::GetCursorPosY();
        ImGui::BeginGroup();
        ImGui::TextColored(kPal.icon, "%s", s.icon.c_str());
        ImGui::SameLine();
        ImGui::BeginGroup();
        ImGui::TextUnformatted(s.title.c_str());
        ImGui::SameLine();
        ImGui::TextColored(kPal.domain, "%s", s.domain.c_str());
        ImGui::PushTextWrapPos(button_x - 16.0f);
        ImGui::TextColored(kPal.dim, "%s", s.description.c_str());
        ImGui::PopTextWrapPos();
        ImGui::EndGroup();
        ImGui::EndGroup();
        const float row_h = std::max(ImGui::GetItemRectSize().y, button_h);
        ImGui::SetCursorPos(ImVec2(button_x, row_y + (row_h - button_h) * 0.5f));
        if (BlueButton(ICON_FA_DIAGRAM_PROJECT " Open", ImVec2(button_w, button_h))) {
            selected_graph_path_ = s.path;
            result_ = Result::ExampleGraphSelected;
            spdlog::info("Opening starter graph: {}", s.path);
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Opens in the Studio without a project.\n%s", s.path.c_str());
        ImGui::SetCursorPos(ImVec2(0.0f, row_y + row_h + 12.0f));
        ImGui::PopID();
    }
}

void StartPage::RenderRecentProjects(float) {
    std::vector<startpage::RecentEntry> entries;
    for (const auto& r : recent_) entries.push_back({r.name, r.path, static_cast<std::time_t>(r.last_opened)});
    const startpage::RecentView view = startpage::BuildRecentView(entries, search_, std::time(nullptr));

    ImGui::Spacing();
    ImGui::TextColored(kPal.title, "Recent Projects");
    ImGui::SameLine();
    ImGui::TextColored(kPal.faint, "%d", view.total);
    if (view.no_projects) {
        ImGui::TextColored(kPal.dim, "No recent projects. Create a new project or open an existing one to get started.");
        return;
    }
    if (view.no_match) {
        ImGui::TextColored(kPal.dim, "No recent project matches \"%s\".", search_);
        return;
    }

    const bool searching = search_[0] != '\0';
    for (const auto& group : view.groups) {
        ImGui::PushID(group.key.c_str());
        const bool collapsed = !searching && collapsed_groups_.count(group.key) > 0;
        const std::string header = std::string(collapsed ? ICON_FA_CHEVRON_RIGHT : ICON_FA_CHEVRON_DOWN) + "  " + group.label +
                                   "  (" + std::to_string(group.rows.size()) + ")";
        ImGui::PushStyleColor(ImGuiCol_Text, kPal.dim);
        if (ImGui::Selectable(header.c_str(), false)) {
            if (collapsed) collapsed_groups_.erase(group.key);
            else collapsed_groups_.insert(group.key);
        }
        ImGui::PopStyleColor();
        if (!collapsed) {
            for (const auto& row : group.rows) {
                ImGui::PushID(row.path.c_str());
                const bool selected = selected_path_ == row.path;
                const float line = ImGui::GetTextLineHeight();
                const float row_h = line * 2.0f + 12.0f;
                const float full_w = ImGui::GetContentRegionAvail().x;
                const ImGuiStyle& style = ImGui::GetStyle();
                const float actions_w = selected ? ImGui::CalcTextSize("Open").x + ImGui::CalcTextSize("Actions").x +
                                                       style.FramePadding.x * 4.0f + 24.0f
                                                 : 0.0f;
                const ImVec2 start = ImGui::GetCursorScreenPos();
                if (ImGui::Selectable("##row", selected, ImGuiSelectableFlags_AllowDoubleClick | ImGuiSelectableFlags_AllowOverlap,
                                      ImVec2(full_w, row_h))) {
                    selected_path_ = row.path;
                    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) OpenProject(row.path);
                    else SetStatus("Selected " + row.name + ". Press Enter or Open.");
                }
                if (ImGui::IsItemHovered() && !selected) ImGui::SetTooltip("%s\nDouble-click to open.", row.path.c_str());
                const ImVec2 after = ImGui::GetCursorScreenPos();
                ImDrawList* dl = ImGui::GetWindowDrawList();
                dl->AddText(ImVec2(start.x + 8.0f, start.y + 6.0f), U32(kPal.icon), ICON_FA_FOLDER);
                const float text_x = start.x + 8.0f + ImGui::GetFontSize() * 1.6f;
                const float when_w = ImGui::CalcTextSize(row.when.c_str()).x;
                const float when_x = start.x + full_w - actions_w - when_w - 12.0f;
                ImGui::PushClipRect(start, ImVec2(when_x - 12.0f, start.y + row_h), true);
                dl->AddText(ImVec2(text_x, start.y + 6.0f), U32(kPal.text), row.name.c_str());
                dl->AddText(ImVec2(text_x, start.y + 6.0f + line), U32(kPal.dim), row.folder.c_str());
                ImGui::PopClipRect();
                dl->AddText(ImVec2(when_x, start.y + (row_h - line) * 0.5f), U32(kPal.faint), row.when.c_str());
                if (selected) {
                    ImGui::SetCursorScreenPos(ImVec2(start.x + full_w - actions_w + 8.0f, start.y + (row_h - ImGui::GetFrameHeight()) * 0.5f));
                    if (BlueButton("Open")) OpenProject(row.path);
                    ImGui::SameLine(0.0f, 8.0f);
                    ImGui::PushStyleColor(ImGuiCol_Button, kPal.lane);
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, kPal.lane_hover);
                    ImGui::PushStyleColor(ImGuiCol_ButtonActive, kPal.lane_active);
                    if (ImGui::Button("Actions")) ImGui::OpenPopup("##row_actions");
                    ImGui::PopStyleColor(3);
                    if (ImGui::BeginPopup("##row_actions")) {
                        if (ImGui::MenuItem(ICON_FA_WINDOW_RESTORE "  Open in new window")) {
                            if (core::WindowManager::LaunchWindow(row.path)) SetStatus("Opened " + row.name + " in a new window.");
                            else SetStatus("Could not start a new window for " + row.name + ".", true);
                        }
                        if (ImGui::MenuItem(ICON_FA_FOLDER_OPEN "  Show in folder")) {
                            if (!ui::ShowInFileManager(row.path)) SetStatus("Could not open the file manager for " + row.folder + ".", true);
                        }
                        ImGui::Separator();
                        ImGui::PushStyleColor(ImGuiCol_Text, kPal.danger);
                        if (ImGui::MenuItem(ICON_FA_XMARK "  Remove from this list")) {
                            ProjectManager::Instance().RemoveRecentProject(row.path);
                            LoadRecentProjects();
                            selected_path_.clear();
                            SetStatus("Removed " + row.name + " from the list. The project itself is untouched.");
                        }
                        ImGui::PopStyleColor();
                        ImGui::EndPopup();
                    }
                    ImGui::SetCursorScreenPos(after);
                }
                ImGui::PopID();
                if (result_ != Result::InProgress) break;
            }
        }
        ImGui::PopID();
        if (result_ != Result::InProgress) break;
    }
}

void StartPage::RenderStartActions() {
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(15.0f, 9.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.0f, 0.5f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(12.0f, 8.0f));

    ImGui::TextColored(kPal.title, "Launch workspace");
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(kPal.dim, "Start from a workflow lane, domain starter, or existing project.");
    ImGui::PopTextWrapPos();
    ImGui::Spacing();

    if (LaneButton(ICON_FA_PLUS "  Create a new project", kPal.primary, kPal.primary_hover, kPal.primary_active)) create_dialog_.Open(0);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Ctrl+N");
    ImGui::Spacing();

    const auto& templates = startpage::ProjectTemplates();
    auto lane = [&](const char* label, int index) {
        if (LaneButton(label, kPal.lane, kPal.lane_hover, kPal.lane_active)) create_dialog_.Open(index);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", templates[index].description);
    };
    ImGui::TextColored(kPal.dim, "Workflow lanes");
    lane(ICON_FA_CHART_LINE "  Classic ML workflow", 1);
    lane(ICON_FA_NETWORK_WIRED "  Deep Learning workflow", 2);
    ImGui::Spacing();
    ImGui::TextColored(kPal.dim, "Domain starters");
    lane(ICON_FA_TABLE "  Tabular project", 3);
    lane(ICON_FA_IMAGE "  Vision project", 4);
    lane(ICON_FA_COMMENTS "  NLP project", 5);
    ImGui::Spacing();
    ImGui::TextColored(kPal.dim, "File actions");
    if (LaneButton(ICON_FA_FOLDER_OPEN "  Open a project", kPal.field, kPal.lane_hover, kPal.lane_active)) OpenExistingProject();
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Open a .cyxwiz project file (Ctrl+O)");
    if (LaneButton(ICON_FA_FOLDER "  Open a project folder", kPal.field, kPal.lane_hover, kPal.lane_active)) OpenProjectFolder();
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Open the folder that holds a .cyxwiz project file");
    LaneButton(ICON_FA_CLOUD_ARROW_DOWN "  Clone a repository (planned)", kPal.field, kPal.lane_hover, kPal.lane_active, false);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("Planned: cloning a repository is not available yet.");

    ImGui::PopStyleVar(3);
}

void StartPage::RenderFooter() {
    const ImVec2 win_size = ImGui::GetWindowSize();
    const float content_width = ContentWidth(win_size.x);
    const float content_x = (win_size.x - content_width) * 0.5f;
    const float button_w = 220.0f;
    const float button_h = 30.0f;
    const float y = win_size.y - button_h - 16.0f;

    // Status line on the left: what just happened, or what went wrong.
    ImGui::SetCursorPos(ImVec2(content_x, y + (button_h - ImGui::GetTextLineHeight()) * 0.5f));
    ImGui::PushTextWrapPos(content_x + content_width - button_w - 24.0f);
    ImGui::TextColored(status_problem_ ? kPal.warn : kPal.faint, "%s", status_.c_str());
    ImGui::PopTextWrapPos();

    ImGui::SetCursorPos(ImVec2(content_x + content_width - button_w, y));
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.75f, 0.80f, 0.88f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.17f, 0.21f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.27f, 0.32f, 1.0f));
    if (ImGui::Button("Continue without project", ImVec2(button_w, button_h))) {
        result_ = Result::ContinueWithout;
        spdlog::info("Continuing without project");
    }
    ImGui::PopStyleColor(3);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Opens the workspace with no project. Saving asks where to put your work.");
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
