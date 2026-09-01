// Script Editor file workflow orchestration.

#include "script_editor.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include <spdlog/spdlog.h>

namespace cyxwiz {

void ScriptEditorPanel::NewFile() {
    auto tab = std::make_unique<EditorTab>();
    tab->document_id = next_document_id_++;
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
        case EditorTheme::Monokai: tab->editor.SetPalette(GetMonokaiPalette()); break;
        case EditorTheme::Dracula: tab->editor.SetPalette(GetDraculaPalette()); break;
        case EditorTheme::OneDark: tab->editor.SetPalette(GetOneDarkPalette()); break;
        case EditorTheme::GitHub: tab->editor.SetPalette(GetGitHubPalette()); break;
    }
    tab->editor.SetShowWhitespaces(show_whitespace_);
    tab->editor.SetColorizerEnable(syntax_highlighting_);
    tab->editor.SetTabSize(tab_size_);
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

    std::error_code size_error;
    const auto file_size = std::filesystem::file_size(path, size_error);
    if (size_error) {
        spdlog::error("Could not inspect file '{}': {}", path, size_error.message());
        return;
    }
    const bool use_large_file_view =
        static_cast<std::uint64_t>(file_size) > kEditableFileLimitBytes;

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
        tab->document_id = next_document_id_++;
        tab->filename = std::filesystem::path(path).filename().string();
        tab->filepath = path;
        tab->is_new = false;
        tab->is_modified = false;
        tab->is_loading = true;
        tab->load_progress = 0.0f;
        tab->load_status = use_large_file_view ? "Indexing large file..." : "Loading...";
        tab->is_large_file = use_large_file_view;
    } else {
        // Create new tab with loading state
        auto tab = std::make_unique<EditorTab>();
        tab->document_id = next_document_id_++;
        tab->filename = std::filesystem::path(path).filename().string();
        tab->filepath = path;
        tab->is_new = false;
        tab->is_modified = false;
        tab->is_loading = true;
        tab->load_progress = 0.0f;
        tab->load_status = use_large_file_view ? "Indexing large file..." : "Loading...";
        tab->is_large_file = use_large_file_view;
        tabs_.push_back(std::move(tab));
        tab_index = static_cast<int>(tabs_.size()) - 1;
    }

    active_tab_index_ = tab_index;
    request_focus_ = true;
    request_window_focus_ = true;

    const auto document_id = tabs_[tab_index]->document_id;
    if (use_large_file_view) {
        spdlog::info(
            "Opening '{}' in bounded large-file view ({} bytes)", path, file_size);
        OpenLargeFileAsync(document_id, path);
    } else {
        OpenFileAsync(document_id, path);
    }
}

void ScriptEditorPanel::OpenFileAsync(
    std::uint64_t document_id,
    const std::string& filepath) {
    std::string path = filepath;
    std::string filename = std::filesystem::path(path).filename().string();

    const int tab_index = FindTabIndex(document_id);
    if (tab_index < 0) {
        spdlog::error("OpenFileAsync: Could not find loading tab for {}", path);
        return;
    }

    spdlog::info("Starting async load of script: {}", filename);

    struct LoadResult {
        std::string content;
    };
    auto result = std::make_shared<LoadResult>();
    std::weak_ptr<std::atomic<bool>> owner_alive = async_owner_alive_;

    const std::uint64_t task_id = AsyncTaskManager::Instance().RunAsync(
        "Loading: " + filename,
        [path, result](LambdaTask& task) {
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file");
            }

            std::streamsize size = file.tellg();
            if (size < 0) {
                throw std::runtime_error("Could not read file size");
            }
            file.seekg(0, std::ios::beg);

            result->content.resize(static_cast<std::size_t>(size));
            constexpr std::streamsize chunk_size = 256 * 1024;
            std::streamsize offset = 0;
            while (offset < size) {
                if (task.ShouldStop()) {
                    return;
                }
                const auto count = std::min(chunk_size, size - offset);
                file.read(result->content.data() + offset, count);
                if (file.gcount() != count) {
                    throw std::runtime_error("Failed to read file content");
                }
                offset += count;
                const float progress = size == 0
                    ? 1.0f
                    : static_cast<float>(
                        static_cast<double>(offset) / static_cast<double>(size));
                task.ReportProgress(progress, "Reading content...");
            }
        },
        nullptr,
        [this, owner_alive, document_id, path, result](
            bool success,
            const std::string& error) mutable {
            const auto alive = owner_alive.lock();
            if (!alive || !alive->load()) {
                return;
            }

            const int current_index = FindTabIndex(document_id);
            if (current_index < 0) {
                return;
            }

            if (success) {
                FinalizeAsyncLoad(document_id, std::move(result->content));
                spdlog::info("Async script load completed: {}", path);
            } else {
                auto& tab = tabs_[current_index];
                tab->is_loading = false;
                tab->load_status = error.empty() ? "Cancelled" : "Failed: " + error;
                spdlog::warn("Async script load stopped: {} - {}", path, tab->load_status);
            }
        }
    );

    tabs_[tab_index]->load_task_id = task_id;
}

void ScriptEditorPanel::FinalizeAsyncLoad(
    std::uint64_t document_id,
    std::string content) {
    const int tab_index = FindTabIndex(document_id);
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
    tab->editor.SetColorizerEnable(syntax_highlighting_);
    tab->editor.SetTabSize(tab_size_);
    tab->editor.SetImGuiChildIgnored(false);
    tab->editor.SetReadOnly(false);
    tab->editor.SetText(content);

    const std::filesystem::path loaded_path(tab->filepath);
    if (loaded_path.extension() == ".cyx" && CellManager::HasCellMarkers(content)) {
        tab->cell_mode = true;
        tab->cell_manager.SetScriptingEngine(scripting_engine_);
        tab->cell_manager.ParseFromCyx(content);
        tab->cell_manager.ApplyTabSize(tab_size_);
        tab->cell_manager.ApplyEditorPalette(tab->editor.GetPalette());
        tab->cell_manager.ApplySyntaxHighlighting(syntax_highlighting_);
        tab->selected_cell = tab->cell_manager.GetCellCount() > 0 ? 0 : -1;
        tab->editing_cell = -1;
        tab->last_editing_cell = -1;
        spdlog::info("Opened .cyx notebook in cell mode: {}", tab->filename);
    } else {
        tab->cell_mode = false;
        tab->selected_cell = -1;
        tab->editing_cell = -1;
        tab->last_editing_cell = -1;
    }

    // Clear loading state
    tab->is_loading = false;
    tab->load_progress = 1.0f;
    tab->load_status.clear();
    tab->load_task_id = 0;

    request_focus_ = true;
}

int ScriptEditorPanel::FindTabIndex(std::uint64_t document_id) const {
    for (int index = 0; index < static_cast<int>(tabs_.size()); ++index) {
        if (tabs_[index] && tabs_[index]->document_id == document_id) {
            return index;
        }
    }
    return -1;
}

bool ScriptEditorPanel::IsActiveTabEditable() const {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return false;
    }
    const auto& tab = tabs_[active_tab_index_];
    return tab && !tab->is_loading && !tab->is_large_file;
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
        tab->document_id = next_document_id_++;
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
            case EditorTheme::Monokai: tab->editor.SetPalette(GetMonokaiPalette()); break;
            case EditorTheme::Dracula: tab->editor.SetPalette(GetDraculaPalette()); break;
            case EditorTheme::OneDark: tab->editor.SetPalette(GetOneDarkPalette()); break;
            case EditorTheme::GitHub: tab->editor.SetPalette(GetGitHubPalette()); break;
        }
        tab->editor.SetShowWhitespaces(show_whitespace_);
        tab->editor.SetColorizerEnable(syntax_highlighting_);
        tab->editor.SetTabSize(tab_size_);
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
    if (!IsActiveTabEditable()) {
        spdlog::warn("Save ignored: active Script Editor tab is not editable");
        return;
    }

    auto& tab = tabs_[active_tab_index_];

    // If new file without path, use Save As
    if (tab->is_new || tab->filepath.empty()) {
        SaveFileAs();
        return;
    }

    // Save to existing path
    std::string content = GetTabContentForPersistence(*tab);
    if (SaveFileContent(tab->filepath, content)) {
        tab->is_modified = false;
        spdlog::info("Saved file: {}", tab->filepath);
    } else {
        spdlog::error("Failed to save file: {}", tab->filepath);
    }
}

void ScriptEditorPanel::SaveFileAs() {
    if (!IsActiveTabEditable()) {
        spdlog::warn("Save As ignored: active Script Editor tab is not editable");
        return;
    }

    auto& tab = tabs_[active_tab_index_];

    // Check if script is empty before showing save dialog
    std::string content = GetTabContentForPersistence(*tab);
    bool is_empty = IsTabContentBlank(*tab);
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
    CancelTabTasks(*tabs_[tab_index]);
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
