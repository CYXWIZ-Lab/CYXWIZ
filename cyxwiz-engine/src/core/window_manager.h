#pragma once

#include <string>
#include <vector>
#include <map>

namespace cyxwiz {
namespace core {

/**
 * @brief Manages multiple CyxWiz Engine instances (multi-window support).
 *
 * Each project runs in a separate engine process for complete isolation.
 * This is similar to VS Code/PyCharm's multi-window architecture.
 *
 * Features:
 * - Launch new engine window with a specific project
 * - Launch new window with project selection dialog
 * - Switch current window to a different project
 * - Track all running engine instances
 */
class WindowManager {
public:
    /**
     * @brief Launch a new engine window with the specified project.
     *
     * Spawns a new CyxWiz Engine process with the given project path.
     * Uses CreateProcessA() on Windows or fork()+execv() on Unix.
     *
     * @param project_path Path to .cyxproj file or project directory
     * @return true if successfully launched, false otherwise
     */
    static bool LaunchWindow(const std::string& project_path);

    /**
     * @brief Launch a new engine window with project selection dialog.
     *
     * Spawns a new CyxWiz Engine process without specifying a project,
     * which will show the project selection dialog on startup.
     *
     * @return true if successfully launched, false otherwise
     */
    static bool LaunchWindowWithDialog();

    /**
     * @brief Restart the current engine instance with the specified project.
     *
     * This will launch a new engine process with the given project and
     * expect the caller to exit the current instance afterwards.
     *
     * @param project_path Path to .cyxwiz file or project directory
     * @return true if successfully launched new instance, false otherwise
     */
    static bool RestartEngine(const std::string& project_path);

    /**
     * @brief Switch the current window to a different project.
     *
     * This will:
     * 1. Save the current project (if needed)
     * 2. Close the current window
     * 3. Launch a new engine instance with the new project
     *
     * @param new_project_path Path to .cyxproj file or project directory
     */
    static void SwitchProject(const std::string& new_project_path);

    /**
     * @brief Get all running CyxWiz Engine instances.
     *
     * Returns a list of project paths currently open in other windows.
     * Implementation varies by platform (Process enumeration on Windows,
     * /proc on Linux, ps on macOS).
     *
     * @return Vector of project paths for running instances
     */
    static std::vector<std::string> GetRunningProjects();

    /**
     * @brief Get the executable path for the current engine instance.
     *
     * Used to spawn new instances of the same executable.
     *
     * @return Full path to the engine executable
     */
    static std::string GetExecutablePath();

private:
    /**
     * @brief Launch a new process with the given command-line arguments.
     *
     * Platform-specific implementation:
     * - Windows: Uses CreateProcessA()
     * - Linux/macOS: Uses fork() + execv()
     *
     * @param executable_path Path to the engine executable
     * @param args Command-line arguments (including --project if applicable)
     * @return true if successfully launched, false otherwise
     */
    static bool LaunchProcess(const std::string& executable_path,
                             const std::vector<std::string>& args);
};

} // namespace core
} // namespace cyxwiz
