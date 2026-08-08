// Script Editor script execution and section helpers.

#include "script_editor.h"
#include "command_window.h"
#include "../../scripting/scripting_engine.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

namespace cyxwiz {

void ScriptEditorPanel::RunScript() {
    if (!IsActiveTabEditable() || !scripting_engine_) return;

    // Don't start if already running
    if (script_running_) {
        spdlog::warn("Script already running");
        return;
    }

    auto& tab = tabs_[active_tab_index_];

    // Check if file is unsaved (new or modified) - prompt to save first
    if (tab->is_new || tab->is_modified) {
        spdlog::info("Script is unsaved, prompting to save before run");
        show_save_before_run_dialog_ = true;
        return;
    }

    // File is saved, run it directly
    DoRunScript();
}

void ScriptEditorPanel::DoRunScript() {
    if (!IsActiveTabEditable() || !scripting_engine_) return;

    auto& tab = tabs_[active_tab_index_];

    spdlog::info("Running script asynchronously: {}", tab->filename);

    // Use the active UI model as the source of truth. In notebook mode the
    // visible cells, not the hidden plain-text editor, own the current content.
    std::string script_text = GetTabExecutableText(*tab);

    // Strip out %% section markers before executing (always, regardless of source)
    std::string script;
    std::istringstream stream(script_text);
    std::string line;
    while (std::getline(stream, line)) {
        // Only skip lines that are ONLY a %% marker (with optional whitespace)
        std::string trimmed = line;
        size_t start = trimmed.find_first_not_of(" \t");
        size_t end = trimmed.find_last_not_of(" \t\r\n");
        if (start != std::string::npos && end != std::string::npos) {
            trimmed = trimmed.substr(start, end - start + 1);
        } else {
            trimmed = "";
        }
        // Keep the line unless it's exactly "%%"
        if (trimmed != "%%") {
            script += line + "\n";
        }
    }

    // Show running indicator in command window
    if (command_window_) {
        command_window_->DisplayScriptOutput(tab->filename, "Script started...", false);
    }

    // Execute asynchronously
    scripting_engine_->ExecuteScriptAsync(script);
    script_running_ = true;
    running_indicator_time_ = 0.0f;
}

void ScriptEditorPanel::SyncActiveCellEditor(EditorTab& tab) {
    if (!tab.cell_mode) {
        return;
    }

    if (tab.editing_cell >= 0 && tab.editing_cell < static_cast<int>(tab.cell_manager.GetCellCount())) {
        tab.cell_manager.GetCell(tab.editing_cell).SyncSourceFromEditor();
    }
}

std::string ScriptEditorPanel::GetTabContentForPersistence(EditorTab& tab) {
    if (!tab.cell_mode) {
        return tab.editor.GetText();
    }

    SyncActiveCellEditor(tab);
    return tab.cell_manager.SerializeToCyx();
}

std::string ScriptEditorPanel::GetTabExecutableText(EditorTab& tab) {
    if (!tab.cell_mode) {
        return tab.editor.GetText();
    }

    SyncActiveCellEditor(tab);

    std::string script;
    for (const auto& cell : tab.cell_manager.GetCells()) {
        if (cell.type != CellType::Code) {
            continue;
        }

        if (cell.source.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }

        script += cell.source;
        if (!script.empty() && script.back() != '\n') {
            script += '\n';
        }
        script += '\n';
    }

    return script;
}

bool ScriptEditorPanel::IsTabContentBlank(EditorTab& tab) const {
    if (!tab.cell_mode) {
        const std::string text = tab.editor.GetText();
        return text.find_first_not_of(" \t\r\n") == std::string::npos;
    }

    for (const auto& cell : tab.cell_manager.GetCells()) {
        if (cell.source.find_first_not_of(" \t\r\n") != std::string::npos) {
            return false;
        }
    }
    return true;
}
std::string ScriptEditorPanel::DedentCode(const std::string& code) {
    std::vector<std::string> lines;
    std::istringstream stream(code);
    std::string line;

    // Split into lines
    while (std::getline(stream, line)) {
        lines.push_back(line);
    }

    if (lines.empty()) return code;

    // Find minimum indentation (ignoring empty lines and whitespace-only lines)
    size_t min_indent = std::string::npos;
    for (const auto& l : lines) {
        if (l.empty()) continue;
        size_t first_non_space = l.find_first_not_of(" \t");
        if (first_non_space == std::string::npos) continue;  // Whitespace-only line
        if (first_non_space < min_indent) min_indent = first_non_space;
    }

    // No dedent needed
    if (min_indent == 0 || min_indent == std::string::npos) return code;

    // Remove common indentation
    std::string result;
    for (const auto& l : lines) {
        if (l.empty()) {
            result += "\n";
        } else {
            size_t first_non_space = l.find_first_not_of(" \t");
            if (first_non_space == std::string::npos) {
                // Whitespace-only line, keep it empty
                result += "\n";
            } else {
                result += l.substr(min_indent) + "\n";
            }
        }
    }

    // Remove trailing newline if original didn't have one
    if (!code.empty() && code.back() != '\n' && !result.empty() && result.back() == '\n') {
        result.pop_back();
    }

    return result;
}

void ScriptEditorPanel::RunSelection() {
    if (!IsActiveTabEditable() || !scripting_engine_) return;
    if (script_running_) return;  // Already running

    auto& tab = tabs_[active_tab_index_];
    std::string selected_text = tab->editor.GetSelectedText();

    if (selected_text.empty()) {
        spdlog::warn("No text selected");
        if (command_window_) {
            command_window_->DisplayScriptOutput(tab->filename, "No text selected", true);
        } else {
            last_execution_output_ = "No text selected";
            show_output_notification_ = true;
            output_notification_time_ = 0.0f;
        }
        return;
    }

    spdlog::info("Running selection asynchronously");
    if (command_window_) {
        command_window_->DisplayScriptOutput(tab->filename + " (selection)", "Running...", false);
    }

    // Dedent and execute asynchronously for plot capture support
    std::string dedented = DedentCode(selected_text);
    spdlog::debug("Dedented selection:\n{}", dedented);
    scripting_engine_->ExecuteScriptAsync(dedented);
    script_running_ = true;
    running_indicator_time_ = 0.0f;
}

void ScriptEditorPanel::RunCurrentSection() {
    if (!IsActiveTabEditable() || !scripting_engine_) return;
    if (script_running_) return;  // Already running

    auto& tab = tabs_[active_tab_index_];
    Section section = GetCurrentSection();

    if (section.code.empty()) {
        spdlog::warn("No section found at cursor");
        if (command_window_) {
            command_window_->DisplayScriptOutput(tab->filename, "No section found at cursor", true);
        } else {
            last_execution_output_ = "No section found at cursor";
            show_output_notification_ = true;
            output_notification_time_ = 0.0f;
        }
        return;
    }

    std::string section_name = tab->filename + " (lines " +
                              std::to_string(section.start_line) + "-" +
                              std::to_string(section.end_line) + ")";

    spdlog::info("Running section {} asynchronously", section_name);
    if (command_window_) {
        command_window_->DisplayScriptOutput(section_name, "Running...", false);
    }

    // Dedent and execute asynchronously for plot capture support
    std::string dedented = DedentCode(section.code);
    spdlog::debug("Dedented section:\n{}", dedented);
    scripting_engine_->ExecuteScriptAsync(dedented);
    script_running_ = true;
    running_indicator_time_ = 0.0f;
}

std::vector<ScriptEditorPanel::Section> ScriptEditorPanel::ParseSections(const std::string& text) {
    std::vector<Section> sections;
    std::istringstream stream(text);
    std::string line;

    int line_num = 0;
    Section current_section;
    current_section.start_line = 0;
    bool in_section = false;

    while (std::getline(stream, line)) {
        // Check for section delimiter %%
        // Each %% both ENDS the previous section AND STARTS the next one (MATLAB-style)
        if (line.find("%%") != std::string::npos) {
            if (in_section && !current_section.code.empty()) {
                // End current section (only if it has content)
                current_section.end_line = line_num - 1;
                sections.push_back(current_section);
            }
            // Start new section after this %%
            current_section = Section();
            current_section.start_line = line_num + 1;
            in_section = true;
        } else if (in_section) {
            current_section.code += line + "\n";
        }

        line_num++;
    }

    // Add final section if still open
    if (in_section && !current_section.code.empty()) {
        current_section.end_line = line_num - 1;
        sections.push_back(current_section);
    }

    // If no %% markers found, treat entire file as one section
    if (sections.empty() && !text.empty()) {
        Section whole_file;
        whole_file.start_line = 0;
        whole_file.end_line = line_num - 1;
        whole_file.code = text;
        sections.push_back(whole_file);
    }

    return sections;
}

ScriptEditorPanel::Section ScriptEditorPanel::GetCurrentSection() {
    Section empty_section;

    if (active_tab_index_ < 0) {
        return empty_section;
    }

    auto& tab = tabs_[active_tab_index_];
    auto cursor_pos = tab->editor.GetCursorPosition();
    int current_line = cursor_pos.mLine;

    // Get all text and parse sections
    std::string text = tab->editor.GetText();
    std::vector<Section> sections = ParseSections(text);

    spdlog::debug("GetCurrentSection: cursor at line {}, found {} sections", current_line, sections.size());
    for (size_t i = 0; i < sections.size(); i++) {
        spdlog::debug("  Section {}: lines {}-{}", i, sections[i].start_line, sections[i].end_line);
    }

    // Find section containing cursor
    for (const auto& section : sections) {
        if (current_line >= section.start_line && current_line <= section.end_line) {
            spdlog::debug("  -> Found section containing cursor at lines {}-{}", section.start_line, section.end_line);
            return section;
        }
    }

    // If cursor is on a %% marker line, find the nearest section
    // Check if current line contains %%
    std::istringstream stream(text);
    std::string line;
    int line_num = 0;
    while (std::getline(stream, line) && line_num <= current_line) {
        if (line_num == current_line && line.find("%%") != std::string::npos) {
            // Cursor is on a %% line, return the section after it
            for (const auto& section : sections) {
                if (section.start_line > current_line) {
                    return section;
                }
            }
        }
        line_num++;
    }

    return empty_section;
}

} // namespace cyxwiz
