#pragma once

#include <string>
#include <vector>
#include <functional>
#include <map>
#include <imgui.h>
#include <imgui_internal.h>

namespace cyxwiz {

/**
 * A single step in a tutorial sequence
 */
struct TutorialStep {
    std::string id;                    // Unique step identifier
    std::string title;                 // Step title shown in tooltip
    std::string description;           // Detailed description
    std::string target_element;        // Window/element to highlight (e.g., "Node Editor", "Asset Browser")
    std::string action_hint;           // What the user should do (e.g., "Click the Add Node button")
    std::function<bool()> completion_condition;  // Returns true when step is complete
    bool requires_interaction = true;  // If false, auto-advance after delay
    float delay_seconds = 0.0f;        // Auto-advance delay (if !requires_interaction)
    ImVec2 highlight_offset = ImVec2(0, 0);  // Offset for highlight position
    ImVec2 highlight_size = ImVec2(0, 0);    // Size override for highlight (0 = auto)
};

/**
 * A complete tutorial sequence
 */
struct Tutorial {
    std::string id;                    // Tutorial identifier
    std::string name;                  // Display name
    std::string description;           // Brief description
    std::string category;              // Category (e.g., "Getting Started", "Advanced")
    std::vector<TutorialStep> steps;
    bool completed = false;            // Has user completed this tutorial?
};

/**
 * Interactive Tutorial System
 * Provides guided walkthroughs with visual highlights and step-by-step instructions.
 */
class TutorialSystem {
public:
    static TutorialSystem& Instance();

    // Tutorial lifecycle
    void StartTutorial(const std::string& tutorial_id);
    void StopTutorial();
    void NextStep();
    void PreviousStep();
    void SkipTutorial();

    // Render the tutorial overlay (call in main render loop)
    void Render();

    // State queries
    bool IsActive() const { return is_active_; }
    bool IsTutorialComplete(const std::string& tutorial_id) const;
    const Tutorial* GetCurrentTutorial() const { return current_tutorial_; }
    int GetCurrentStepIndex() const { return current_step_index_; }
    int GetTotalSteps() const;

    // Tutorial management
    void RegisterTutorial(Tutorial tutorial);
    const std::vector<Tutorial>& GetAvailableTutorials() const { return tutorials_; }
    const std::vector<std::string>& GetCompletedTutorials() const { return completed_tutorials_; }

    // Show tutorial selection dialog
    void ShowTutorialBrowser();
    void OpenTutorialBrowser() { show_browser_ = true; }
    void CloseTutorialBrowser() { show_browser_ = false; }

    // First-launch detection
    bool ShouldShowWelcome() const { return !has_shown_welcome_; }
    void MarkWelcomeShown() { has_shown_welcome_ = true; }

    // Persistence
    void SaveProgress();
    void LoadProgress();

    // Set window position callback (to get window rect for highlighting)
    using WindowRectCallback = std::function<ImRect(const std::string& window_name)>;
    void SetWindowRectCallback(WindowRectCallback callback) { window_rect_callback_ = callback; }

private:
    TutorialSystem();
    ~TutorialSystem() = default;

    // Rendering helpers
    void RenderOverlay();
    void RenderHighlight(const ImRect& target_rect);
    void RenderStepTooltip(const TutorialStep& step, const ImRect& target_rect);
    void RenderProgressBar();

    // Built-in tutorials
    void RegisterBuiltInTutorials();

    // Tutorial registry
    std::vector<Tutorial> tutorials_;
    std::map<std::string, size_t> tutorial_index_;  // id -> index mapping

    // Current state
    bool is_active_ = false;
    Tutorial* current_tutorial_ = nullptr;
    int current_step_index_ = 0;
    float step_start_time_ = 0.0f;

    // Progress tracking
    std::vector<std::string> completed_tutorials_;
    bool has_shown_welcome_ = false;

    // UI state
    bool show_browser_ = false;
    float highlight_pulse_ = 0.0f;  // Animation state

    // Callbacks
    WindowRectCallback window_rect_callback_;

    // Colors - solid opaque for clear visibility
    ImVec4 overlay_color_ = ImVec4(0.0f, 0.0f, 0.0f, 0.75f);
    ImVec4 highlight_color_ = ImVec4(0.3f, 0.7f, 1.0f, 1.0f);
    ImVec4 tooltip_bg_color_ = ImVec4(0.12f, 0.12f, 0.18f, 1.0f);  // Fully opaque
};

} // namespace cyxwiz
