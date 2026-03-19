#pragma once

#include <string>
#include <vector>
#include <optional>

namespace cyxwiz {
namespace core {

/**
 * @brief Detects and validates Python installations on the system.
 *
 * Searches for Python 3.12+ installations in common locations and validates
 * that they meet CyxWiz requirements (venv module, pip).
 */
class PythonDetector {
public:
    /**
     * @brief Information about a Python installation.
     */
    struct PythonInstallation {
        std::string executable_path;  // e.g., C:\Python312\python.exe
        std::string version;          // e.g., "3.12.1"
        int major = 0;                // 3
        int minor = 0;                // 12
        int micro = 0;                // 1
        bool has_venv_module = false; // Can create virtual environments?
        bool has_pip = false;         // Has pip installed?
        std::string home;             // Python home directory (sys.prefix)

        // Comparison for sorting
        bool operator<(const PythonInstallation& other) const {
            if (major != other.major) return major > other.major;
            if (minor != other.minor) return minor > other.minor;
            return micro > other.micro;
        }
    };

    /**
     * @brief Find all Python installations on the system.
     *
     * Searches in:
     * - Windows: py.exe launcher, PATH, common install locations, Microsoft Store
     * - Linux/macOS: PATH, /usr/bin, /usr/local/bin, Homebrew
     *
     * @return Vector of all detected Python installations (may include duplicates)
     */
    static std::vector<PythonInstallation> FindAllPythonInstallations();

    /**
     * @brief Find the best Python installation for CyxWiz.
     *
     * Prefers:
     * 1. Python 3.12 or 3.13 (stable versions)
     * 2. Has venv module
     * 3. Has pip
     * 4. Newest version
     *
     * @return Best Python installation, or nullopt if none found
     */
    static std::optional<PythonInstallation> FindBestPython();

    /**
     * @brief Check if a specific Python executable is valid.
     *
     * Runs `python -c "..."` to get version and capabilities.
     *
     * @param path Path to Python executable (or command like "py -3.12")
     * @return Python installation info, or nullopt if invalid
     */
    static std::optional<PythonInstallation> CheckPython(const std::string& path);

    /**
     * @brief Validate that a Python installation path is valid and get its info.
     *
     * Alias for CheckPython() - validates the path and returns installation info.
     *
     * @param path Path to Python executable
     * @return Python installation info, or nullopt if invalid
     */
    static std::optional<PythonInstallation> ValidatePythonInstallation(const std::string& path) {
        return CheckPython(path);
    }

    /**
     * @brief Validate that a Python installation meets CyxWiz requirements.
     *
     * Requirements:
     * - Python >= 3.12
     * - Has venv module
     *
     * @param python Python installation to validate
     * @return true if meets requirements, false otherwise
     */
    static bool MeetsRequirements(const PythonInstallation& python);

    /**
     * @brief Get human-readable description of why Python doesn't meet requirements.
     *
     * @param python Python installation to check
     * @return Error message, or empty string if valid
     */
    static std::string GetRequirementError(const PythonInstallation& python);

private:
    /**
     * @brief Get list of candidate Python paths to check.
     *
     * Platform-specific list of common Python installation locations.
     *
     * @return Vector of paths to check
     */
    static std::vector<std::string> GetCandidatePaths();

    /**
     * @brief Execute command and capture output.
     *
     * @param command Command to execute
     * @param output Output will be written here
     * @return Exit code (0 = success)
     */
    static int ExecuteCommand(const std::string& command, std::string& output);
};

} // namespace core
} // namespace cyxwiz
