#pragma once

#include <string>
#include <vector>
#include <optional>
#include "../../core/python_detector.h"

namespace cyxwiz {

/**
 * @brief First-launch wizard for Python detection and configuration.
 *
 * Guides the user through detecting supported Python versions on their system.
 * Must be shown on first launch if no system Python is configured.
 *
 * Flow:
 * 1. Detecting Python... (scan system)
 * 2a. Python 3.12-3.13 found → Show details → Save to config
 * 2b. No Python found → Show download instructions → Exit
 * 2c. Multiple found → Let user choose → Save to config
 */
class PythonSetupWizard {
public:
    enum class Result {
        InProgress,   // Wizard still running
        Completed,    // Python configured successfully
        Cancelled,    // User cancelled (should exit app)
    };

    PythonSetupWizard();
    ~PythonSetupWizard() = default;

    // Render the wizard dialog (call every frame)
    // Returns true if the wizard is still active
    bool Render();

    // Get the result of the wizard
    Result GetResult() const { return result_; }

    // Get the selected Python path (valid only if result == Completed)
    std::string GetSelectedPythonPath() const { return selected_python_path_; }

private:
    enum class State {
        Detecting,       // Scanning for Python installations
        NoPythonFound,   // No supported Python found
        SingleFound,     // One supported Python found
        MultipleFound,   // Multiple supported Pythons found
    };

    void RenderDetecting();
    void RenderNoPythonFound();
    void RenderSingleFound();
    void RenderMultipleFound();

    void StartDetection();
    void OnDetectionComplete();

    State state_ = State::Detecting;
    Result result_ = Result::InProgress;

    // Detection results
    std::vector<cyxwiz::core::PythonDetector::PythonInstallation> found_pythons_;
    int selected_index_ = 0;
    std::string selected_python_path_;

    // UI state
    bool detection_started_ = false;
    float detection_progress_ = 0.0f;
    std::string detection_status_ = "Initializing...";
};

} // namespace cyxwiz
