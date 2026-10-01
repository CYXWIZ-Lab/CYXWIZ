// Find and replace over editor text (TOFIX133 P0 item 13), and the byte
// offset <-> visual column mapping editors need (tabs, UTF-8). No ImGui.
#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz::textsearch {

struct Options {
    bool case_sensitive = false;
    bool whole_word = false;   // letters, digits and _ are word characters
    bool regex = false;        // ECMAScript; replacement may use $1, $&
};

struct Match {
    size_t pos = 0;
    size_t len = 0;
};

// Every match in order. Empty regex matches are skipped (they would replace
// nothing and stall a loop). `error` is set for an invalid regex.
std::vector<Match> FindAll(const std::string& text, const std::string& pattern, const Options& options,
                           std::string* error = nullptr);

// First match starting at or after `from`, wrapping to the start.
std::optional<Match> FindNext(const std::string& text, const std::string& pattern, size_t from, const Options& options,
                              std::string* error = nullptr);
// Last match ending at or before `before`, wrapping to the end.
std::optional<Match> FindPrevious(const std::string& text, const std::string& pattern, size_t before,
                                  const Options& options, std::string* error = nullptr);

// The text that replaces `matched` (regex groups expanded); nullopt when
// `matched` as a whole is not a match (Replace then only moves to the next one).
std::optional<std::string> ReplacementFor(const std::string& matched, const std::string& pattern,
                                          const std::string& replacement, const Options& options);

// Replaces every match; `count` receives how many.
std::string ReplaceAll(const std::string& text, const std::string& pattern, const std::string& replacement,
                       const Options& options, int* count, std::string* error = nullptr);

// Line/column of a byte offset in '\n'-separated text, as the editor counts
// columns: tabs advance to the next tab stop, a UTF-8 character is one column.
struct Position {
    int line = 0;
    int column = 0;
};
Position ToPosition(const std::string& text, size_t offset, int tab_size);
// Byte offset of an editor line/column (clamped to the line).
size_t ToOffset(const std::string& text, Position position, int tab_size);

}  // namespace cyxwiz::textsearch
