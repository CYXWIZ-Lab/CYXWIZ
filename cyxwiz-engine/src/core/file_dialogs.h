// file_dialogs.h - Cross-platform native file dialog utilities
// Uses nativefiledialog-extended for Windows, Linux, and macOS support
#pragma once

#include <string>
#include <vector>
#include <optional>
#include <utility>

namespace cyxwiz {

/**
 * Cross-platform native file dialog utilities.
 *
 * Uses native dialogs on each platform:
 * - Windows: Win32 dialogs
 * - Linux: GTK3 dialogs
 * - macOS: Cocoa dialogs
 *
 * Usage:
 *   auto result = FileDialogs::OpenFile("Open Project", {
 *       {"CyxWiz Projects", "cyxwiz"},
 *       {"All Files", "*"}
 *   });
 *   if (result) {
 *       std::string path = *result;
 *   }
 */
class FileDialogs {
public:
    // Filter specification: {name, extension}
    // Example: {"Image Files", "png,jpg,jpeg"} or {"All Files", "*"}
    using Filter = std::pair<std::string, std::string>;
    using FilterList = std::vector<Filter>;

    /**
     * Open a single file selection dialog.
     *
     * @param title Dialog window title
     * @param filters List of file filters (name, extensions)
     * @param default_path Optional starting directory
     * @return Selected file path, or nullopt if cancelled
     */
    static std::optional<std::string> OpenFile(
        const char* title,
        const FilterList& filters = {},
        const char* default_path = nullptr
    );

    /**
     * Open a multiple file selection dialog.
     *
     * @param title Dialog window title
     * @param filters List of file filters
     * @param default_path Optional starting directory
     * @return Vector of selected file paths (empty if cancelled)
     */
    static std::vector<std::string> OpenMultiple(
        const char* title,
        const FilterList& filters = {},
        const char* default_path = nullptr
    );

    /**
     * Open a save file dialog.
     *
     * @param title Dialog window title
     * @param filters List of file filters
     * @param default_path Optional starting directory
     * @param default_name Optional default filename
     * @return Selected save path, or nullopt if cancelled
     */
    static std::optional<std::string> SaveFile(
        const char* title,
        const FilterList& filters = {},
        const char* default_path = nullptr,
        const char* default_name = nullptr
    );

    /**
     * Open a folder selection dialog.
     *
     * @param title Dialog window title
     * @param default_path Optional starting directory
     * @return Selected folder path, or nullopt if cancelled
     */
    static std::optional<std::string> SelectFolder(
        const char* title,
        const char* default_path = nullptr
    );

    // =========================================================================
    // Convenience methods for common CyxWiz file types
    // =========================================================================

    /** Open a CyxWiz project file (.cyxwiz) */
    static std::optional<std::string> OpenProject(const char* default_path = nullptr);

    /** Save a CyxWiz project file (.cyxwiz) */
    static std::optional<std::string> SaveProject(const char* default_path = nullptr);

    /** Open a CyxWiz graph file (.cyxgraph) */
    static std::optional<std::string> OpenGraph(const char* default_path = nullptr);

    /** Save a CyxWiz graph file (.cyxgraph) */
    static std::optional<std::string> SaveGraph(const char* default_path = nullptr);

    /** Open a Python/CyxWiz script file (.py, .cyx) */
    static std::optional<std::string> OpenScript(const char* default_path = nullptr);

    /** Save a Python/CyxWiz script file (.py, .cyx) */
    static std::optional<std::string> SaveScript(const char* default_path = nullptr);

    /** Open a model file (ONNX, PyTorch, etc.) */
    static std::optional<std::string> OpenModel(const char* default_path = nullptr);

    /** Save a model file (ONNX, etc.) */
    static std::optional<std::string> SaveModel(const char* default_path = nullptr);

    /** Open a dataset file (CSV, HDF5, etc.) */
    static std::optional<std::string> OpenDataset(const char* default_path = nullptr);

    /** Open an image file */
    static std::optional<std::string> OpenImage(const char* default_path = nullptr);

    /** Open multiple image files */
    static std::vector<std::string> OpenImages(const char* default_path = nullptr);

    /** Select a folder for datasets */
    static std::optional<std::string> SelectDatasetFolder(const char* default_path = nullptr);

    /** Select a folder for project output */
    static std::optional<std::string> SelectOutputFolder(const char* default_path = nullptr);

private:
    // Initialize NFD (called automatically)
    static void EnsureInitialized();
    static bool initialized_;
};

} // namespace cyxwiz
