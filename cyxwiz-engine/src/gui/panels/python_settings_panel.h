#pragma once

#include <string>
#include <vector>

namespace gui {

/**
 * @brief Python Settings panel for Edit → Settings menu.
 *
 * Allows users to configure Python interpreter settings:
 * - View/change system Python path
 * - Auto-detect Python installations
 * - Configure default packages for new venvs
 * - Toggle auto-create venv for new projects
 */
class PythonSettingsPanel {
public:
    PythonSettingsPanel();
    ~PythonSettingsPanel() = default;

    /**
     * @brief Render the Python settings panel.
     *
     * Call this inside a settings window or as a tab in settings dialog.
     */
    void Render();

    /**
     * @brief Check if settings were modified.
     *
     * @return true if user changed any settings
     */
    bool IsModified() const { return is_modified_; }

    /**
     * @brief Apply and save the current settings.
     */
    void ApplySettings();

    /**
     * @brief Revert to saved settings (discard changes).
     */
    void RevertSettings();

private:
    /**
     * @brief Load settings from EngineConfig.
     */
    void LoadSettings();

    /**
     * @brief Run Python auto-detection and populate detected_pythons_.
     */
    void RunAutoDetection();

    /**
     * @brief Open file browser to manually select Python executable.
     */
    void BrowsePythonExecutable();

    // Current settings (editable)
    char system_python_path_[512] = {};
    bool auto_create_venv_ = true;
    char default_packages_[1024] = {};  // Comma-separated package names

    // Auto-detection results
    struct DetectedPython {
        std::string path;
        std::string version;
        bool has_venv;
        bool has_pip;
    };
    std::vector<DetectedPython> detected_pythons_;
    bool detection_in_progress_ = false;
    bool show_detection_results_ = false;

    // Python status (for current path)
    std::string python_version_;
    bool python_valid_ = false;
    std::string python_error_;

    // UI state
    bool is_modified_ = false;
    int selected_detected_index_ = -1;
};

} // namespace gui
