#pragma once

#include "../../core/tutorial_content.h"
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <imgui.h>
#include <imgui_internal.h>

namespace cyxwiz {

/**
 * Interactive Tutorials (Help > Interactive Tutorials): the list of tutorials
 * and, while one runs, a step card beside the window it points at, that
 * window outlined and the step dots. Content and placement come from
 * core/tutorial_content. Progress is kept in the user's settings folder.
 */
class TutorialSystem {
public:
    static TutorialSystem& Instance();

    // Tutorial lifecycle
    void StartTutorial(const std::string& tutorial_id);
    void StopTutorial();
    void NextStep();
    void PreviousStep();

    // Render the list and the running tutorial (call in main render loop)
    void Render();

    bool IsTutorialComplete(const std::string& tutorial_id) const;
    const std::vector<tutorial::Tutorial>& GetAvailableTutorials() const { return tutorials_; }

    void OpenTutorialBrowser() { show_browser_ = true; }

    // The screen rectangle of a window that is shown now; empty when it is
    // closed, a hidden tab or not drawn.
    using WindowRectCallback = std::function<ImRect(const std::string& window_name)>;
    void SetWindowRectCallback(WindowRectCallback callback) { window_rect_callback_ = std::move(callback); }

private:
    TutorialSystem();
    ~TutorialSystem() = default;

    void SaveProgress();
    void LoadProgress();
    void ResetProgress();

    void RenderBrowser();
    void RenderStep();
    void RenderHighlight(const ImRect& target_rect);
    void RenderStepDots();

    std::vector<tutorial::Tutorial> tutorials_;
    std::vector<std::string> completed_tutorials_;

    // The tutorial that runs (index into tutorials_, -1: none) and its step
    int current_ = -1;
    int current_step_index_ = 0;
    std::map<std::string, int> paused_;  // closed before the end: the step to resume on (this session)
    bool focus_card_ = false;  // the card takes the keyboard after a start or a step
    ImVec2 card_size_ = ImVec2(380.0f, 240.0f);  // measured last frame, for its place

    bool show_browser_ = false;
    float highlight_pulse_ = 0.0f;
    WindowRectCallback window_rect_callback_;
};

} // namespace cyxwiz
