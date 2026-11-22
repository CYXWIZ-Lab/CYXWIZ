#include "tutorial_system.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"
#include "../../core/engine_config.h"
#include <imgui.h>
#include <imgui_internal.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <utility>

namespace cyxwiz {

namespace {

constexpr const char* kResetTitle = "Reset tutorial progress?";
constexpr float kCardWidth = 380.0f;

std::filesystem::path ProgressPath() {
    return core::EngineConfig::Instance().GetUserConfigDir() / "tutorial_progress.json";
}

}  // namespace

TutorialSystem& TutorialSystem::Instance() {
    static TutorialSystem instance;
    return instance;
}

TutorialSystem::TutorialSystem() : tutorials_(tutorial::BuiltInTutorials()) {
    LoadProgress();
}

void TutorialSystem::StartTutorial(const std::string& tutorial_id) {
    const auto it = std::find_if(tutorials_.begin(), tutorials_.end(),
                                 [&](const tutorial::Tutorial& t) { return t.id == tutorial_id; });
    if (it == tutorials_.end()) {
        spdlog::error("Tutorial not found: {}", tutorial_id);
        return;
    }
    const int index = static_cast<int>(it - tutorials_.begin());
    if (current_ != index) {
        // Resume: the step it was closed on; a new start from the first
        const auto paused = paused_.find(tutorial_id);
        current_step_index_ = paused != paused_.end() ? paused->second : 0;
    }
    current_ = index;
    focus_card_ = true;
    show_browser_ = false;
    spdlog::info("Started tutorial: {} ({} steps)", it->name, it->steps.size());
}

void TutorialSystem::StopTutorial() {
    if (current_ >= 0) {
        spdlog::info("Stopped tutorial: {}", tutorials_[current_].name);
        paused_[tutorials_[current_].id] = current_step_index_;
    }
    current_ = -1;
    current_step_index_ = 0;
}

void TutorialSystem::NextStep() {
    if (current_ < 0) return;
    const tutorial::Tutorial& t = tutorials_[current_];
    ++current_step_index_;
    focus_card_ = true;
    if (current_step_index_ >= static_cast<int>(t.steps.size())) {
        if (!IsTutorialComplete(t.id)) {
            completed_tutorials_.push_back(t.id);
            SaveProgress();
        }
        spdlog::info("Tutorial completed: {}", t.name);
        const std::string id = t.id;
        StopTutorial();
        paused_.erase(id);
    }
}

void TutorialSystem::PreviousStep() {
    if (current_ < 0 || current_step_index_ == 0) return;
    --current_step_index_;
    focus_card_ = true;
}

void TutorialSystem::Render() {
    highlight_pulse_ += ImGui::GetIO().DeltaTime * 3.0f;
    if (highlight_pulse_ > 6.28318f) highlight_pulse_ -= 6.28318f;
    if (show_browser_) RenderBrowser();
    if (current_ >= 0) RenderStep();
}

void TutorialSystem::RenderStep() {
    const tutorial::Tutorial& t = tutorials_[current_];
    if (current_step_index_ >= static_cast<int>(t.steps.size())) return;
    const tutorial::Step& step = t.steps[current_step_index_];
    const ui::Tokens& tokens = ui::CurrentTokens();

    // The window it points at, when that window is shown
    ImRect target;
    if (!step.target.empty() && window_rect_callback_) target = window_rect_callback_(step.target);
    const bool has_target = target.GetWidth() > 0.0f && target.GetHeight() > 0.0f;
    // Nothing is drawn over a modal dialog
    const bool modal_open = ImGui::GetTopMostPopupModal() != nullptr;
    if (has_target && !modal_open) RenderHighlight(target);

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const tutorial::Box box{target.Min.x - viewport->Pos.x, target.Min.y - viewport->Pos.y,
                            target.Max.x - viewport->Pos.x, target.Max.y - viewport->Pos.y};
    const tutorial::Point at = tutorial::PlaceCard(has_target ? &box : nullptr, card_size_.x, card_size_.y,
                                                   viewport->Size.x, viewport->Size.y);
    ImGui::SetNextWindowPos(ImVec2(viewport->Pos.x + at.x, viewport->Pos.y + at.y), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(kCardWidth, 0.0f), ImGuiCond_Always);
    if (focus_card_) {
        ImGui::SetNextWindowFocus();
        focus_card_ = false;
    }
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(tokens.space_lg, tokens.space_lg));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, tokens.rounding_card);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoCollapse |
                                   ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDocking |
                                   ImGuiWindowFlags_NoNavInputs;  // the arrows and Esc are the tutorial's
    const bool open = ImGui::Begin("##TutorialStep", nullptr, flags);
    ImGui::PopStyleVar(2);
    if (open) {
        const int total = static_cast<int>(t.steps.size());
        ImGui::TextColored(tokens.accent_text, "Step %d of %d", current_step_index_ + 1, total);
        ImGui::SameLine();
        ImGui::TextColored(tokens.text_dim, "%s", t.name.c_str());
        ui::SameLineRight(ImGui::GetFrameHeight());
        bool stop = ui::GhostButton(ICON_FA_XMARK);
        ui::Tooltip("Close the tutorial (Esc). It resumes from this step in the list.");

        ImGui::Spacing();
        ImGui::TextColored(tokens.text_bright, "%s", step.title.c_str());
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(tokens.text, "%s", step.description.c_str());
        ImGui::PopTextWrapPos();
        ImGui::Spacing();
        ui::NoticeLine(step.hint, tokens.info);
        if (!step.target.empty() && !has_target) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(tokens.text_dim, "That window is closed: open it from View > Panels to see it.");
            ImGui::PopTextWrapPos();
        }
        ImGui::Spacing();

        bool previous = ui::SecondaryButton(ICON_FA_ARROW_LEFT " Previous", current_step_index_ > 0,
                                            "This is the first step", ui::ButtonSize::Regular);
        const bool last = current_step_index_ == total - 1;
        const char* next_label = last ? ICON_FA_CHECK " Complete" : "Next " ICON_FA_ARROW_RIGHT;
        ui::SameLineRight(ui::ButtonWidth(next_label, ui::ButtonSize::Regular));
        bool next = ui::PrimaryButton(next_label);

        // Keys while the card has the focus: arrows step, Esc closes
        if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) && !ImGui::GetIO().WantTextInput) {
            if (ImGui::IsKeyPressed(ImGuiKey_RightArrow, false)) next = true;
            if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow, false)) previous = current_step_index_ > 0;
            if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) stop = true;
        }
        card_size_ = ImGui::GetWindowSize();
        ImGui::End();

        if (stop) {
            StopTutorial();
            return;
        }
        if (previous) PreviousStep();
        if (next) NextStep();
    } else {
        ImGui::End();
    }
    if (current_ >= 0 && !modal_open) RenderStepDots();
}

void TutorialSystem::RenderHighlight(const ImRect& target_rect) {
    ImDrawList* draw_list = ImGui::GetForegroundDrawList();
    const ui::Tokens& t = ui::CurrentTokens();
    const float pulse = 0.5f + 0.5f * std::sin(highlight_pulse_);
    const auto grown = [&target_rect](float by) {
        return std::pair<ImVec2, ImVec2>(ImVec2(target_rect.Min.x - by, target_rect.Min.y - by),
                                         ImVec2(target_rect.Max.x + by, target_rect.Max.y + by));
    };
    // A soft glow that breathes, then the outline in the accent
    const auto glow = grown(7.0f);
    draw_list->AddRect(glow.first, glow.second, ui::ToU32(ui::WithAlpha(t.accent, 0.12f + 0.18f * pulse)),
                       t.rounding_card, 0, 6.0f);
    const auto line = grown(3.0f);
    draw_list->AddRect(line.first, line.second, ui::ToU32(t.accent), t.rounding_card, 0, 2.0f);
}

void TutorialSystem::RenderStepDots() {
    const tutorial::Tutorial& t = tutorials_[current_];
    if (t.steps.empty()) return;
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImDrawList* draw_list = ImGui::GetForegroundDrawList();
    const ui::Tokens& tokens = ui::CurrentTokens();
    const float spacing = 18.0f;
    const float width = spacing * static_cast<float>(t.steps.size() - 1);
    const float y = viewport->Pos.y + viewport->Size.y - 40.0f;
    const float x0 = viewport->Pos.x + (viewport->Size.x - width) * 0.5f;
    for (size_t i = 0; i < t.steps.size(); ++i) {
        const int k = static_cast<int>(i);
        const ImU32 colour = ui::ToU32(k < current_step_index_    ? tokens.success
                                       : k == current_step_index_ ? tokens.accent
                                                                  : tokens.pending);
        draw_list->AddCircleFilled(ImVec2(x0 + spacing * static_cast<float>(i), y), k == current_step_index_ ? 5.0f : 4.0f,
                                   colour);
    }
}

void TutorialSystem::RenderBrowser() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::SetNextWindowSize(ImVec2(460, 460), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Interactive Tutorials", &show_browser_)) {
        ImGui::End();
        return;
    }
    std::vector<std::string> categories;
    for (const auto& tutorial : tutorials_)
        if (std::find(categories.begin(), categories.end(), tutorial.category) == categories.end())
            categories.push_back(tutorial.category);

    std::string start;
    for (const std::string& category : categories) {
        ImGui::Spacing();
        ImGui::TextColored(t.text_dim, "%s", category.c_str());
        for (size_t i = 0; i < tutorials_.size(); ++i) {
            const tutorial::Tutorial& tut = tutorials_[i];
            if (tut.category != category) continue;
            const bool running = current_ == static_cast<int>(i);
            const auto paused = paused_.find(tut.id);
            const int step = running ? current_step_index_ : paused != paused_.end() ? paused->second : -1;
            const tutorial::ListRow row = tutorial::DescribeListRow(tut, IsTutorialComplete(tut.id), step);
            ImGui::PushID(tut.id.c_str());
            ImGui::TextColored(row.done ? t.success : t.pending, "%s", row.done ? ICON_FA_CIRCLE_CHECK : ICON_FA_CIRCLE);
            ImGui::SameLine();
            ImGui::TextColored(t.text_bright, "%s", tut.name.c_str());
            ImGui::SameLine();
            ImGui::TextColored(step >= 0 ? t.success : t.text_dim, "%s", row.status.c_str());
            ui::SameLineRight(ui::ButtonWidth(row.action.c_str(), ui::ButtonSize::Small));
            if (ui::SecondaryButton(row.action.c_str())) start = tut.id;
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.text_dim, "%s", tut.description.c_str());
            ImGui::PopTextWrapPos();
            ImGui::Spacing();
            ImGui::PopID();
        }
    }
    ImGui::Spacing();
    const char* kReset = "Reset progress...";
    ui::SameLineRight(ui::ButtonWidth(kReset, ui::ButtonSize::Small));
    if (ui::DangerButton(kReset, !completed_tutorials_.empty(), "No tutorial is done yet"))
        ImGui::OpenPopup(kResetTitle);
    if (ui::ConfirmDialog(kResetTitle, "Every tutorial reads as not done again. The Engine and your projects are not changed.",
                          "Reset progress", true) == ui::DialogResult::Primary)
        ResetProgress();
    ImGui::End();
    if (!start.empty()) StartTutorial(start);
}

bool TutorialSystem::IsTutorialComplete(const std::string& tutorial_id) const {
    return std::find(completed_tutorials_.begin(), completed_tutorials_.end(), tutorial_id) != completed_tutorials_.end();
}

void TutorialSystem::ResetProgress() {
    completed_tutorials_.clear();
    paused_.clear();
    SaveProgress();
    spdlog::info("Tutorial progress reset");
}

void TutorialSystem::SaveProgress() {
    const auto path = ProgressPath();
    try {
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
        nlohmann::json j;
        j["completed"] = completed_tutorials_;
        std::ofstream file(path);
        if (!file.is_open()) {
            spdlog::warn("Tutorial progress could not be saved to {}", path.string());
            return;
        }
        file << j.dump(2);
    } catch (const std::exception& e) {
        spdlog::error("Failed to save tutorial progress: {}", e.what());
    }
}

void TutorialSystem::LoadProgress() {
    const auto path = ProgressPath();
    std::ifstream file(path);
    if (!file.is_open()) return;  // nothing done yet
    try {
        nlohmann::json j;
        file >> j;
        completed_tutorials_.clear();
        if (j.contains("completed") && j["completed"].is_array())
            for (const auto& id : j["completed"])
                if (id.is_string()) completed_tutorials_.push_back(id.get<std::string>());
        spdlog::debug("Tutorial progress loaded: {} completed tutorials", completed_tutorials_.size());
    } catch (const std::exception& e) {
        spdlog::warn("Tutorial progress in {} could not be read: {}", path.string(), e.what());
    }
}

} // namespace cyxwiz
