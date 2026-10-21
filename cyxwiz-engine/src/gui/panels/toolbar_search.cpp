// Toolbar find-in-files helpers.

#include "toolbar.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

namespace cyxwiz {

// Helper function to check if a file matches any of the given patterns
static bool MatchesFilePattern(const std::string& filename, const std::string& patterns) {
    if (patterns.empty()) return true;

    // Split patterns by semicolon
    std::vector<std::string> pattern_list;
    std::stringstream ss(patterns);
    std::string pattern;
    while (std::getline(ss, pattern, ';')) {
        // Trim whitespace
        size_t start = pattern.find_first_not_of(" \t");
        size_t end = pattern.find_last_not_of(" \t");
        if (start != std::string::npos && end != std::string::npos) {
            pattern_list.push_back(pattern.substr(start, end - start + 1));
        }
    }

    // Check if filename matches any pattern
    for (const auto& pat : pattern_list) {
        // Convert glob pattern to regex
        std::string regex_pattern;
        for (char c : pat) {
            switch (c) {
                case '*': regex_pattern += ".*"; break;
                case '?': regex_pattern += "."; break;
                case '.': regex_pattern += "\\."; break;
                default: regex_pattern += c; break;
            }
        }
        regex_pattern = "^" + regex_pattern + "$";

        try {
            std::regex re(regex_pattern, std::regex::icase);
            if (std::regex_match(filename, re)) {
                return true;
            }
        } catch (const std::regex_error&) {
            // If regex fails, try simple extension match
            if (pat.length() > 1 && pat[0] == '*') {
                std::string ext = pat.substr(1);
                if (filename.length() >= ext.length() &&
                    filename.substr(filename.length() - ext.length()) == ext) {
                    return true;
                }
            }
        }
    }

    return false;
}

// Helper function to search in a single line
static bool SearchInLine(const std::string& line, const std::string& search_text,
                         bool case_sensitive, bool whole_word, bool use_regex,
                         int& match_start, int& match_length) {
    if (search_text.empty()) return false;

    if (use_regex) {
        try {
            std::regex::flag_type flags = std::regex::ECMAScript;
            if (!case_sensitive) flags |= std::regex::icase;

            std::regex re(search_text, flags);
            std::smatch match;
            if (std::regex_search(line, match, re)) {
                match_start = static_cast<int>(match.position(0));
                match_length = static_cast<int>(match.length(0));
                return true;
            }
        } catch (const std::regex_error& e) {
            spdlog::warn("Invalid regex pattern: {}", e.what());
            return false;
        }
    } else {
        std::string search_line = line;
        std::string search_term = search_text;

        if (!case_sensitive) {
            std::transform(search_line.begin(), search_line.end(), search_line.begin(), ::tolower);
            std::transform(search_term.begin(), search_term.end(), search_term.begin(), ::tolower);
        }

        size_t pos = search_line.find(search_term);
        if (pos != std::string::npos) {
            if (whole_word) {
                // Check word boundaries
                bool start_ok = (pos == 0) || !std::isalnum(static_cast<unsigned char>(search_line[pos - 1]));
                bool end_ok = (pos + search_term.length() >= search_line.length()) ||
                              !std::isalnum(static_cast<unsigned char>(search_line[pos + search_term.length()]));
                if (!start_ok || !end_ok) {
                    return false;
                }
            }
            match_start = static_cast<int>(pos);
            match_length = static_cast<int>(search_term.length());
            return true;
        }
    }

    return false;
}

void ToolbarPanel::SearchInFiles(const std::string& search_text, const std::string& search_path,
                                  const std::string& file_patterns, bool case_sensitive,
                                  bool whole_word, bool use_regex) {
    search_results_.clear();
    search_in_progress_ = true;

    if (search_text.empty() || search_path.empty()) {
        search_in_progress_ = false;
        return;
    }

    namespace fs = std::filesystem;

    try {
        size_t files_searched = 0;
        constexpr size_t kMaxResults = 1000;  // Prevent UI slowdown.

        for (const auto& entry : fs::recursive_directory_iterator(search_path,
                fs::directory_options::skip_permission_denied)) {
            if (!entry.is_regular_file()) continue;

            std::string filename = entry.path().filename().string();
            if (!MatchesFilePattern(filename, file_patterns)) continue;

            files_searched++;

            // Read file and search
            std::ifstream file(entry.path());
            if (!file.is_open()) continue;

            std::string line;
            int line_number = 0;

            while (std::getline(file, line) &&
                   search_results_.size() < kMaxResults) {
                line_number++;

                int match_start = 0, match_length = 0;
                if (SearchInLine(line, search_text, case_sensitive, whole_word, use_regex,
                                 match_start, match_length)) {
                    SearchResult result;
                    result.file_path = entry.path().string();
                    result.line_number = line_number;
                    result.line_content = line;
                    result.match_start = match_start;
                    result.match_length = match_length;

                    // Truncate line if too long
                    if (result.line_content.length() > 200) {
                        result.line_content = result.line_content.substr(0, 200) + "...";
                    }

                    search_results_.push_back(result);
                }
            }

            if (search_results_.size() >= kMaxResults) {
                spdlog::info("Search stopped: max results ({}) reached",
                             kMaxResults);
                break;
            }
        }

        spdlog::info("Search complete: found {} results in {} files",
                     search_results_.size(), files_searched);

    } catch (const fs::filesystem_error& e) {
        spdlog::error("Filesystem error during search: {}", e.what());
    }

    search_in_progress_ = false;
}
// Replace in Files (TOFIX129 G3): every match in every file under the folder
// that matches the patterns. Files open in the Script Editor with unsaved
// edits are skipped so no edit is lost; open unmodified files are reloaded.
void ToolbarPanel::ReplaceInFiles(const std::string& search_text, const std::string& replace_text,
                                  const std::string& search_path, const std::string& file_patterns,
                                  bool case_sensitive, bool whole_word, bool use_regex) {
    namespace fs = std::filesystem;
    replace_in_files_summary_.clear();
    if (search_text.empty() || search_path.empty()) return;

    size_t files_changed = 0;
    size_t replacements = 0;
    size_t skipped_unsaved = 0;
    size_t failed = 0;

    try {
        for (const auto& entry : fs::recursive_directory_iterator(search_path,
                fs::directory_options::skip_permission_denied)) {
            if (!entry.is_regular_file()) continue;
            if (!MatchesFilePattern(entry.path().filename().string(), file_patterns)) continue;
            const std::string path = entry.path().string();

            std::ifstream in(entry.path(), std::ios::binary);
            if (!in.is_open()) continue;
            std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            in.close();

            // Replace line by line so the whole-word and regex rules of the
            // search apply the same way; line endings are kept as they are.
            std::string out;
            out.reserve(content.size());
            size_t file_replacements = 0;
            size_t pos = 0;
            while (pos <= content.size()) {
                size_t end = content.find('\n', pos);
                const bool last = end == std::string::npos;
                std::string line = content.substr(pos, last ? std::string::npos : end - pos);
                std::string rebuilt;
                size_t offset = 0;
                int match_start = 0, match_length = 0;
                while (offset < line.size() &&
                       SearchInLine(line.substr(offset), search_text, case_sensitive, whole_word, use_regex,
                                    match_start, match_length) && match_length > 0) {
                    rebuilt += line.substr(offset, match_start);
                    rebuilt += replace_text;
                    offset += static_cast<size_t>(match_start + match_length);
                    ++file_replacements;
                }
                rebuilt += line.substr(std::min(offset, line.size()));
                out += rebuilt;
                if (last) break;
                out += '\n';
                pos = end + 1;
            }
            if (file_replacements == 0) continue;

            if (file_has_unsaved_changes_callback_ && file_has_unsaved_changes_callback_(path)) {
                ++skipped_unsaved;
                spdlog::warn("Replace in Files: skipped {} (unsaved edits in the editor)", path);
                continue;
            }
            std::ofstream file_out(entry.path(), std::ios::binary | std::ios::trunc);
            if (!file_out.is_open()) {
                ++failed;
                spdlog::error("Replace in Files: cannot write {}", path);
                continue;
            }
            file_out << out;
            file_out.close();
            ++files_changed;
            replacements += file_replacements;
            if (reload_open_file_callback_) reload_open_file_callback_(path);
        }
    } catch (const fs::filesystem_error& e) {
        spdlog::error("Filesystem error during replace: {}", e.what());
    }

    replace_in_files_summary_ = std::to_string(replacements) + " replacement" + (replacements == 1 ? "" : "s") +
                                " in " + std::to_string(files_changed) + " file" + (files_changed == 1 ? "" : "s");
    if (skipped_unsaved > 0)
        replace_in_files_summary_ += "; " + std::to_string(skipped_unsaved) + " open file(s) with unsaved edits skipped";
    if (failed > 0) replace_in_files_summary_ += "; " + std::to_string(failed) + " could not be written";
    spdlog::info("Replace in Files: {}", replace_in_files_summary_);

    // Refresh the results list so it shows what is left.
    SearchInFiles(search_text, search_path, file_patterns, case_sensitive, whole_word, use_regex);
}

} // namespace cyxwiz
