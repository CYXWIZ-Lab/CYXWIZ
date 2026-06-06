// Script Editor find and replace operations.

#include "script_editor.h"

#include <algorithm>
#include <cctype>
#include <regex>
#include <string>

#include <spdlog/spdlog.h>

namespace cyxwiz {

// ==================== Find/Replace Operations ====================

bool ScriptEditorPanel::FindInEditor(const std::string& search_text, bool case_sensitive, bool whole_word, bool use_regex) {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return false;
    }

    if (search_text.empty()) {
        return false;
    }

    // Store search parameters for FindNext/FindPrevious
    last_search_text_ = search_text;
    last_case_sensitive_ = case_sensitive;
    last_whole_word_ = whole_word;
    last_use_regex_ = use_regex;

    auto& editor = tabs_[active_tab_index_]->editor;
    std::string text = editor.GetText();

    // Get current cursor position
    auto cursor = editor.GetCursorPosition();
    int start_pos = 0;

    // Convert cursor position to character offset
    auto lines = editor.GetTextLines();
    for (int i = 0; i < cursor.mLine && i < static_cast<int>(lines.size()); ++i) {
        start_pos += static_cast<int>(lines[i].length()) + 1;  // +1 for newline
    }
    start_pos += cursor.mColumn;

    // If there's a selection that matches the search text, skip past it
    if (editor.HasSelection()) {
        std::string selected = editor.GetSelectedText();
        std::string match_text = selected;
        std::string search_check = search_text;

        if (!case_sensitive) {
            std::transform(match_text.begin(), match_text.end(), match_text.begin(), ::tolower);
            std::transform(search_check.begin(), search_check.end(), search_check.begin(), ::tolower);
        }

        if (match_text == search_check) {
            // Skip past the current selection
            start_pos += static_cast<int>(selected.length());
        }
    }

    // Search from current position
    size_t found_pos = std::string::npos;
    size_t match_len = search_text.length();

    if (use_regex) {
        try {
            std::regex::flag_type flags = std::regex::ECMAScript;
            if (!case_sensitive) flags |= std::regex::icase;

            std::regex re(search_text, flags);
            std::smatch match;
            std::string search_area = text.substr(start_pos);

            if (std::regex_search(search_area, match, re)) {
                found_pos = start_pos + match.position(0);
                match_len = match.length(0);
            } else {
                // Wrap around and search from beginning
                if (std::regex_search(text, match, re)) {
                    found_pos = match.position(0);
                    match_len = match.length(0);
                }
            }
        } catch (const std::regex_error& e) {
            spdlog::warn("Invalid regex: {}", e.what());
            return false;
        }
    } else {
        std::string search_text_lower = search_text;
        std::string text_lower = text;

        if (!case_sensitive) {
            std::transform(search_text_lower.begin(), search_text_lower.end(),
                           search_text_lower.begin(), ::tolower);
            std::transform(text_lower.begin(), text_lower.end(),
                           text_lower.begin(), ::tolower);
        }

        // Search from current position
        found_pos = text_lower.find(search_text_lower, start_pos);

        // Wrap around if not found
        if (found_pos == std::string::npos) {
            found_pos = text_lower.find(search_text_lower, 0);
        }

        // Check whole word boundary
        if (found_pos != std::string::npos && whole_word) {
            bool start_ok = (found_pos == 0) || !std::isalnum(static_cast<unsigned char>(text_lower[found_pos - 1]));
            bool end_ok = (found_pos + search_text_lower.length() >= text_lower.length()) ||
                          !std::isalnum(static_cast<unsigned char>(text_lower[found_pos + search_text_lower.length()]));
            if (!start_ok || !end_ok) {
                found_pos = std::string::npos;
            }
        }
    }

    if (found_pos != std::string::npos) {
        // Convert character offset to line/column
        int line = 0;
        int col = 0;
        size_t pos = 0;

        for (const auto& line_text : lines) {
            if (pos + line_text.length() >= found_pos) {
                col = static_cast<int>(found_pos - pos);
                break;
            }
            pos += line_text.length() + 1;  // +1 for newline
            line++;
        }

        // Select the found text
        TextEditor::Coordinates start_coord(line, col);
        TextEditor::Coordinates end_coord(line, col + static_cast<int>(match_len));

        // Handle multi-line match
        size_t end_pos = found_pos + match_len;
        pos = 0;
        for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
            if (pos + lines[i].length() >= end_pos) {
                end_coord.mLine = i;
                end_coord.mColumn = static_cast<int>(end_pos - pos);
                break;
            }
            pos += lines[i].length() + 1;
        }

        editor.SetSelection(start_coord, end_coord);
        editor.SetCursorPosition(start_coord);

        spdlog::info("Found '{}' at line {}, col {}", search_text, line + 1, col + 1);
        return true;
    }

    spdlog::info("'{}' not found", search_text);
    return false;
}

bool ScriptEditorPanel::FindNext() {
    if (last_search_text_.empty()) {
        return false;
    }

    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return false;
    }

    auto& editor = tabs_[active_tab_index_]->editor;

    // If there's a selection (current match), move cursor past it before searching
    if (editor.HasSelection()) {
        // Get selection end and move cursor there
        auto cursor = editor.GetCursorPosition();
        auto text = editor.GetText();
        auto lines = editor.GetTextLines();

        // Calculate character offset of current position
        int current_offset = 0;
        for (int i = 0; i < cursor.mLine && i < static_cast<int>(lines.size()); ++i) {
            current_offset += static_cast<int>(lines[i].length()) + 1;
        }
        current_offset += cursor.mColumn;

        // Move cursor forward by the length of the search text to skip current match
        int new_offset = current_offset + static_cast<int>(last_search_text_.length());

        // Convert back to coordinates
        int line = 0;
        int col = 0;
        int pos = 0;
        for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
            if (pos + static_cast<int>(lines[i].length()) >= new_offset) {
                line = i;
                col = new_offset - pos;
                break;
            }
            pos += static_cast<int>(lines[i].length()) + 1;
            line = i + 1;
            col = 0;
        }

        // Clear selection and set cursor to end of current match
        editor.SetCursorPosition(TextEditor::Coordinates(line, col));
    }

    return FindInEditor(last_search_text_, last_case_sensitive_, last_whole_word_, last_use_regex_);
}

bool ScriptEditorPanel::FindPrevious() {
    if (last_search_text_.empty()) {
        return false;
    }

    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return false;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    std::string text = editor.GetText();
    auto lines = editor.GetTextLines();

    // Get current cursor position
    auto cursor = editor.GetCursorPosition();
    int end_search_pos = 0;

    // Convert cursor position to character offset
    for (int i = 0; i < cursor.mLine && i < static_cast<int>(lines.size()); ++i) {
        end_search_pos += static_cast<int>(lines[i].length()) + 1;  // +1 for newline
    }
    end_search_pos += cursor.mColumn;

    // If there's a selection (current match), search before it
    if (editor.HasSelection()) {
        // Don't include current match in search
        end_search_pos = std::max(0, end_search_pos - 1);
    }

    // Search backward from current position
    size_t found_pos = std::string::npos;
    size_t match_len = last_search_text_.length();

    if (last_use_regex_) {
        try {
            std::regex::flag_type flags = std::regex::ECMAScript;
            if (!last_case_sensitive_) flags |= std::regex::icase;

            std::regex re(last_search_text_, flags);
            std::smatch match;

            // Search in the text before cursor
            std::string search_area = text.substr(0, end_search_pos);

            // Find the last match by iterating through all matches
            auto begin = std::sregex_iterator(search_area.begin(), search_area.end(), re);
            auto end_it = std::sregex_iterator();

            std::smatch last_match;
            bool found_any = false;
            for (auto it = begin; it != end_it; ++it) {
                last_match = *it;
                found_any = true;
            }

            if (found_any) {
                found_pos = last_match.position(0);
                match_len = last_match.length(0);
            } else {
                // Wrap around: search from end of document
                auto wrap_begin = std::sregex_iterator(text.begin(), text.end(), re);
                for (auto it = wrap_begin; it != end_it; ++it) {
                    last_match = *it;
                    found_any = true;
                }
                if (found_any) {
                    found_pos = last_match.position(0);
                    match_len = last_match.length(0);
                }
            }
        } catch (const std::regex_error& e) {
            spdlog::warn("Invalid regex: {}", e.what());
            return false;
        }
    } else {
        std::string search_text_lower = last_search_text_;
        std::string text_lower = text;

        if (!last_case_sensitive_) {
            std::transform(search_text_lower.begin(), search_text_lower.end(),
                           search_text_lower.begin(), ::tolower);
            std::transform(text_lower.begin(), text_lower.end(),
                           text_lower.begin(), ::tolower);
        }

        // Search backward from current position
        if (end_search_pos > 0) {
            found_pos = text_lower.rfind(search_text_lower, end_search_pos - 1);
        }

        // Wrap around if not found
        if (found_pos == std::string::npos) {
            found_pos = text_lower.rfind(search_text_lower);
        }

        // Check whole word boundary
        if (found_pos != std::string::npos && last_whole_word_) {
            bool start_ok = (found_pos == 0) || !std::isalnum(static_cast<unsigned char>(text_lower[found_pos - 1]));
            bool end_ok = (found_pos + search_text_lower.length() >= text_lower.length()) ||
                          !std::isalnum(static_cast<unsigned char>(text_lower[found_pos + search_text_lower.length()]));
            if (!start_ok || !end_ok) {
                // Try to find another match that satisfies whole word
                while (found_pos != std::string::npos && found_pos > 0) {
                    found_pos = text_lower.rfind(search_text_lower, found_pos - 1);
                    if (found_pos != std::string::npos) {
                        start_ok = (found_pos == 0) || !std::isalnum(static_cast<unsigned char>(text_lower[found_pos - 1]));
                        end_ok = (found_pos + search_text_lower.length() >= text_lower.length()) ||
                                 !std::isalnum(static_cast<unsigned char>(text_lower[found_pos + search_text_lower.length()]));
                        if (start_ok && end_ok) break;
                    }
                }
            }
        }
    }

    if (found_pos != std::string::npos) {
        // Convert character offset to line/column
        int line = 0;
        int col = 0;
        size_t pos = 0;

        for (const auto& line_text : lines) {
            if (pos + line_text.length() >= found_pos) {
                col = static_cast<int>(found_pos - pos);
                break;
            }
            pos += line_text.length() + 1;  // +1 for newline
            line++;
        }

        // Select the found text
        TextEditor::Coordinates start_coord(line, col);
        TextEditor::Coordinates end_coord(line, col + static_cast<int>(match_len));

        // Handle multi-line match
        size_t end_pos = found_pos + match_len;
        pos = 0;
        for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
            if (pos + lines[i].length() >= end_pos) {
                end_coord.mLine = i;
                end_coord.mColumn = static_cast<int>(end_pos - pos);
                break;
            }
            pos += lines[i].length() + 1;
        }

        editor.SetSelection(start_coord, end_coord);
        editor.SetCursorPosition(start_coord);

        spdlog::info("Found previous '{}' at line {}, col {}", last_search_text_, line + 1, col + 1);
        return true;
    }

    spdlog::info("'{}' not found (backward)", last_search_text_);
    return false;
}

bool ScriptEditorPanel::Replace(const std::string& search_text, const std::string& replace_text,
                                 bool case_sensitive, bool whole_word, bool use_regex) {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return false;
    }

    auto& editor = tabs_[active_tab_index_]->editor;

    // If there's a selection matching the search text, replace it
    if (editor.HasSelection()) {
        std::string selected = editor.GetSelectedText();
        std::string match_text = selected;
        std::string search_check = search_text;

        if (!case_sensitive) {
            std::transform(match_text.begin(), match_text.end(), match_text.begin(), ::tolower);
            std::transform(search_check.begin(), search_check.end(), search_check.begin(), ::tolower);
        }

        if (match_text == search_check) {
            // Replace the selection
            editor.Delete();

            auto cursor = editor.GetCursorPosition();
            editor.InsertText(replace_text);

            tabs_[active_tab_index_]->is_modified = true;

            spdlog::info("Replaced '{}' with '{}'", search_text, replace_text);

            // Find next occurrence
            FindInEditor(search_text, case_sensitive, whole_word, use_regex);
            return true;
        }
    }

    // No valid selection, find next occurrence
    return FindInEditor(search_text, case_sensitive, whole_word, use_regex);
}

int ScriptEditorPanel::ReplaceAll(const std::string& search_text, const std::string& replace_text,
                                   bool case_sensitive, bool whole_word, bool use_regex) {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return 0;
    }

    if (search_text.empty()) {
        return 0;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    std::string text = editor.GetText();
    int count = 0;

    if (use_regex) {
        try {
            std::regex::flag_type flags = std::regex::ECMAScript;
            if (!case_sensitive) flags |= std::regex::icase;

            std::regex re(search_text, flags);

            // Count matches
            std::string temp = text;
            std::smatch match;
            while (std::regex_search(temp, match, re)) {
                count++;
                temp = match.suffix().str();
            }

            // Replace all
            std::string result = std::regex_replace(text, re, replace_text);
            editor.SetText(result);

        } catch (const std::regex_error& e) {
            spdlog::warn("Invalid regex: {}", e.what());
            return 0;
        }
    } else {
        std::string search_lower = search_text;
        std::string text_lower = text;

        if (!case_sensitive) {
            std::transform(search_lower.begin(), search_lower.end(), search_lower.begin(), ::tolower);
            std::transform(text_lower.begin(), text_lower.end(), text_lower.begin(), ::tolower);
        }

        // Find and replace all occurrences
        std::string result;
        size_t pos = 0;
        size_t last_pos = 0;

        while ((pos = text_lower.find(search_lower, last_pos)) != std::string::npos) {
            bool match_ok = true;

            if (whole_word) {
                bool start_ok = (pos == 0) || !std::isalnum(static_cast<unsigned char>(text_lower[pos - 1]));
                bool end_ok = (pos + search_lower.length() >= text_lower.length()) ||
                              !std::isalnum(static_cast<unsigned char>(text_lower[pos + search_lower.length()]));
                match_ok = start_ok && end_ok;
            }

            if (match_ok) {
                result += text.substr(last_pos, pos - last_pos);
                result += replace_text;
                count++;
            } else {
                result += text.substr(last_pos, pos - last_pos + search_text.length());
            }

            last_pos = pos + search_text.length();
        }

        result += text.substr(last_pos);

        if (count > 0) {
            editor.SetText(result);
        }
    }

    if (count > 0) {
        tabs_[active_tab_index_]->is_modified = true;
        spdlog::info("Replaced {} occurrences of '{}' with '{}'", count, search_text, replace_text);
    }

    return count;
}
} // namespace cyxwiz
