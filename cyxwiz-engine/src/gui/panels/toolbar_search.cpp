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
        int files_searched = 0;
        int max_results = 1000;  // Limit results to prevent UI slowdown

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

            while (std::getline(file, line) && search_results_.size() < max_results) {
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

            if (search_results_.size() >= max_results) {
                spdlog::info("Search stopped: max results ({}) reached", max_results);
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
} // namespace cyxwiz
