// Script Editor file workflow orchestration.

#include "script_editor.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include <spdlog/spdlog.h>

namespace cyxwiz {

void ScriptEditorPanel::NewFile() {
    auto tab = std::make_unique<EditorTab>();
    tab->filename = "Untitled" + std::to_string(tabs_.size() + 1) + ".cyx";
    tab->filepath = "";
    tab->is_new = true;
    tab->is_modified = false;

    auto lang = CreatePythonLanguage();
    tab->editor.SetLanguageDefinition(lang);
    // Apply current theme
    switch (current_theme_) {
        case EditorTheme::Dark: tab->editor.SetPalette(TextEditor::GetDarkPalette()); break;
        case EditorTheme::Light: tab->editor.SetPalette(TextEditor::GetLightPalette()); break;
        case EditorTheme::RetroBlu: tab->editor.SetPalette(TextEditor::GetRetroBluePalette()); break;
    }
    tab->editor.SetShowWhitespaces(show_whitespace_);
    tab->editor.SetTabSize(4);
    tab->editor.SetImGuiChildIgnored(false);
    tab->editor.SetReadOnly(false);

    tabs_.push_back(std::move(tab));
    active_tab_index_ = static_cast<int>(tabs_.size()) - 1;
    request_focus_ = true;

    spdlog::info("Created new script file: {}", tabs_[active_tab_index_]->filename);
}

void ScriptEditorPanel::OpenFile(const std::string& filepath) {
    std::string path = filepath;

    // If no path provided, show file dialog
    if (path.empty()) {
        path = OpenFileDialog();
        if (path.empty()) return;  // User cancelled
    }

    // Check if file is already open
    for (int i = 0; i < static_cast<int>(tabs_.size()); i++) {
        if (tabs_[i]->filepath == path) {
            active_tab_index_ = i;
            request_focus_ = true;
            request_window_focus_ = true;  // Focus the Script Editor window
            spdlog::info("File already open: {}", path);
            return;
        }
    }

    // Check if we can replace an existing empty untitled tab
    int empty_tab_index = -1;
    for (int i = 0; i < static_cast<int>(tabs_.size()); i++) {
        auto& tab = tabs_[i];
        std::string tab_text = tab->editor.GetText();
        tab_text.erase(0, tab_text.find_first_not_of(" \t\n\r"));
        tab_text.erase(tab_text.find_last_not_of(" \t\n\r") + 1);

        if (tab->is_new && !tab->is_modified && tab_text.empty()) {
            empty_tab_index = i;
            break;
        }
    }

    // Use the empty tab or create a new one
    int tab_index;
    if (empty_tab_index >= 0) {
        tab_index = empty_tab_index;
        auto& tab = tabs_[tab_index];
        tab->filename = std::filesystem::path(path).filename().string();
        tab->filepath = path;
        tab->is_new = false;
        tab->is_modified = false;
        tab->is_loading = true;
        tab->load_progress = 0.0f;
        tab->load_status = "Loading...";
    } else {
        // Create new tab with loading state
        auto tab = std::make_unique<EditorTab>();
        tab->filename = std::filesystem::path(path).filename().string();
        tab->filepath = path;
        tab->is_new = false;
        tab->is_modified = false;
        tab->is_loading = true;
        tab->load_progress = 0.0f;
        tab->load_status = "Loading...";
        tabs_.push_back(std::move(tab));
        tab_index = static_cast<int>(tabs_.size()) - 1;
    }

    active_tab_index_ = tab_index;
    request_focus_ = true;
    request_window_focus_ = true;

    // Load file asynchronously
    OpenFileAsync(path);
}

void ScriptEditorPanel::OpenFileAsync(const std::string& filepath) {
    std::string path = filepath;
    std::string filename = std::filesystem::path(path).filename().string();

    // Find the tab that's loading this file
    int tab_index = -1;
    for (int i = 0; i < static_cast<int>(tabs_.size()); i++) {
        if (tabs_[i]->filepath == path && tabs_[i]->is_loading) {
            tab_index = i;
            break;
        }
    }

    if (tab_index < 0) {
        spdlog::error("OpenFileAsync: Could not find loading tab for {}", path);
        return;
    }

    spdlog::info("Starting async load of script: {}", filename);

    AsyncTaskManager::Instance().RunAsync(
        "Loading: " + filename,
        [this, tab_index, path](LambdaTask& task) {
            task.ReportProgress(0.1f, "Opening file...");

            // Read file content in background thread
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file.is_open()) {
                task.MarkFailed("Could not open file");
                return;
            }

            task.ReportProgress(0.3f, "Reading content...");

            std::streamsize size = file.tellg();
            file.seekg(0, std::ios::beg);

            std::string content;
            content.resize(static_cast<size_t>(size));

            if (!file.read(&content[0], size)) {
                task.MarkFailed("Failed to read file content");
                return;
            }

            task.ReportProgress(0.8f, "Finalizing...");

            // Store content for main thread to finalize
            if (tab_index < static_cast<int>(tabs_.size())) {
                tabs_[tab_index]->pending_content = std::move(content);
            }

            task.ReportProgress(1.0f, "Complete");
            task.MarkCompleted();
        },
        [this, tab_index](float progress, const std::string& status) {
            // Progress callback - update tab
            if (tab_index < static_cast<int>(tabs_.size())) {
                tabs_[tab_index]->load_progress = progress;
                tabs_[tab_index]->load_status = status;
            }
        },
        [this, tab_index, path](bool success, const std::string& error) {
            // Completion callback
            if (tab_index < static_cast<int>(tabs_.size())) {
                auto& tab = tabs_[tab_index];
                if (success) {
                    // Finalize on main thread
                    FinalizeAsyncLoad(tab_index);
                    spdlog::info("Async script load completed: {}", path);
                } else {
                    tab->is_loading = false;
                    tab->load_status = "Failed: " + error;
                    spdlog::error("Async script load failed: {} - {}", path, error);
                }
            }
        }
    );
}

void ScriptEditorPanel::FinalizeAsyncLoad(int tab_index) {
    if (tab_index < 0 || tab_index >= static_cast<int>(tabs_.size())) return;

    auto& tab = tabs_[tab_index];
    if (!tab->is_loading) return;

    auto lang = CreatePythonLanguage();
    tab->editor.SetLanguageDefinition(lang);

    // Apply current theme
    switch (current_theme_) {
        case EditorTheme::Dark: tab->editor.SetPalette(TextEditor::GetDarkPalette()); break;
        case EditorTheme::Light: tab->editor.SetPalette(TextEditor::GetLightPalette()); break;
        case EditorTheme::RetroBlu: tab->editor.SetPalette(TextEditor::GetRetroBluePalette()); break;
        case EditorTheme::Monokai: tab->editor.SetPalette(GetMonokaiPalette()); break;
        case EditorTheme::Dracula: tab->editor.SetPalette(GetDraculaPalette()); break;
        case EditorTheme::OneDark: tab->editor.SetPalette(GetOneDarkPalette()); break;
        case EditorTheme::GitHub: tab->editor.SetPalette(GetGitHubPalette()); break;
    }

    tab->editor.SetShowWhitespaces(show_whitespace_);
    tab->editor.SetTabSize(tab_size_);
    tab->editor.SetImGuiChildIgnored(false);
    tab->editor.SetReadOnly(false);
    tab->editor.SetText(tab->pending_content);

    // Clear loading state
    tab->is_loading = false;
    tab->load_progress = 1.0f;
    tab->load_status.clear();
    tab->pending_content.clear();

    request_focus_ = true;
}

void ScriptEditorPanel::LoadGeneratedCode(const std::string& code, const std::string& framework_name) {
    std::string target_filename = "generated_" + framework_name + ".py";

    // Check if a tab with this filename already exists
    int existing_tab_index = -1;
    for (int i = 0; i < static_cast<int>(tabs_.size()); i++) {
        if (tabs_[i]->filename == target_filename) {
            existing_tab_index = i;
            break;
        }
    }

    if (existing_tab_index >= 0) {
        // Update existing tab
        auto& tab = tabs_[existing_tab_index];
        tab->editor.SetText(code);
        tab->is_modified = true;
        active_tab_index_ = existing_tab_index;
        request_focus_ = true;
        request_window_focus_ = true;
        spdlog::info("Updated existing {} code tab", framework_name);
    } else {
        // Create new tab with generated code
        auto tab = std::make_unique<EditorTab>();
        tab->filename = target_filename;
        tab->filepath = "";  // Not saved yet
        tab->is_new = true;
        tab->is_modified = true;  // Has content, mark as modified

        auto lang = CreatePythonLanguage();
        tab->editor.SetLanguageDefinition(lang);
        // Apply current theme
        switch (current_theme_) {
            case EditorTheme::Dark: tab->editor.SetPalette(TextEditor::GetDarkPalette()); break;
            case EditorTheme::Light: tab->editor.SetPalette(TextEditor::GetLightPalette()); break;
            case EditorTheme::RetroBlu: tab->editor.SetPalette(TextEditor::GetRetroBluePalette()); break;
        }
        tab->editor.SetShowWhitespaces(show_whitespace_);
        tab->editor.SetTabSize(4);
        tab->editor.SetImGuiChildIgnored(false);
        tab->editor.SetReadOnly(false);
        tab->editor.SetText(code);

        tabs_.push_back(std::move(tab));
        active_tab_index_ = static_cast<int>(tabs_.size()) - 1;
        request_focus_ = true;
        request_window_focus_ = true;

        spdlog::info("Loaded generated {} code into new tab", framework_name);
    }
}

void ScriptEditorPanel::SaveFile() {
    if (active_tab_index_ < 0) return;

    auto& tab = tabs_[active_tab_index_];

    // If new file without path, use Save As
    if (tab->is_new || tab->filepath.empty()) {
        SaveFileAs();
        return;
    }

    // Save to existing path
    std::string content = tab->editor.GetText();
    if (SaveFileContent(tab->filepath, content)) {
        tab->is_modified = false;
        spdlog::info("Saved file: {}", tab->filepath);
    } else {
        spdlog::error("Failed to save file: {}", tab->filepath);
    }
}

void ScriptEditorPanel::SaveFileAs() {
    if (active_tab_index_ < 0) return;

    auto& tab = tabs_[active_tab_index_];

    // Check if script is empty before showing save dialog
    std::string content = tab->editor.GetText();
    // Trim whitespace to check if truly empty
    bool is_empty = content.empty() ||
                    content.find_first_not_of(" \t\n\r") == std::string::npos;
    if (is_empty) {
        show_empty_script_warning_ = true;
        spdlog::warn("Cannot save empty script - no content present");
        return;
    }

    // Show save dialog
    std::string path = SaveFileDialog();
    if (path.empty()) return;  // User cancelled

    // Ensure .cyx extension
    std::filesystem::path fspath(path);
    if (fspath.extension() != ".cyx") {
        path += ".cyx";
    }

    // Save content (already have content from empty check above)
    if (SaveFileContent(path, content)) {
        tab->filepath = path;
        tab->filename = std::filesystem::path(path).filename().string();
        tab->is_new = false;
        tab->is_modified = false;
        spdlog::info("Saved file as: {}", path);
    } else {
        spdlog::error("Failed to save file: {}", path);
    }
}

void ScriptEditorPanel::CloseFile(int tab_index) {
    if (tab_index < 0 || tab_index >= static_cast<int>(tabs_.size())) return;

    // Check if file has unsaved changes
    if (tabs_[tab_index]->is_modified || tabs_[tab_index]->is_new) {
        // Don't close immediately - show confirmation dialog
        pending_close_tab_index_ = tab_index;
        show_save_before_close_dialog_ = true;
        spdlog::info("File has unsaved changes, showing save dialog: {}", tabs_[tab_index]->filename);
        return;
    }

    // File is saved, close directly
    DoCloseFile(tab_index);
}

void ScriptEditorPanel::DoCloseFile(int tab_index) {
    if (tab_index < 0 || tab_index >= static_cast<int>(tabs_.size())) return;

    spdlog::info("Closing file: {}", tabs_[tab_index]->filename);
    tabs_.erase(tabs_.begin() + tab_index);

    // Adjust active tab index
    if (tabs_.empty()) {
        // Create new empty tab if all closed
        NewFile();
    } else if (active_tab_index_ >= static_cast<int>(tabs_.size())) {
        active_tab_index_ = static_cast<int>(tabs_.size()) - 1;
    }

    // Reset pending close state
    pending_close_tab_index_ = -1;
}

} // namespace cyxwiz
