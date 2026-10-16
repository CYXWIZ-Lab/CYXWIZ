// CSV records as RFC 4180 writes them: quoted fields may hold commas, line
// breaks and doubled quotes. Used by DataTable (Table Viewer). No ImGui.
#pragma once

#include <string>
#include <vector>

namespace cyxwiz::csv {

// Every record of the text (a UTF-8 byte-order mark and CRLF are fine).
// Unquoted fields are trimmed of surrounding spaces; quoted ones are kept.
std::vector<std::vector<std::string>> ReadRecords(const std::string& text);

// A field for writing: quoted when it holds a comma, quote or line break.
std::string Field(const std::string& value);

// True when the whole of `text` is an integer / a number (no "3 rows").
bool ParseInt(const std::string& text, long long& out);
bool ParseDouble(const std::string& text, double& out);

}  // namespace cyxwiz::csv
