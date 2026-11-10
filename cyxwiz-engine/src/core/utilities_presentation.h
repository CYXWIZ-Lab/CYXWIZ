#pragma once

// Tools > Utilities (TOFIX129 D12): what the one Utilities window shows of
// cyxwiz::Utilities' results, worked out without ImGui so the window and its
// tests agree. Six tabs: Calculator, Units, Random, Hash, JSON, Regex.

#include <cyxwiz/utilities.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::utilities_view {

enum class Tab { Calculator, Units, Random, Hash, Json, Regex };

constexpr Tab kTabs[] = {Tab::Calculator, Tab::Units, Tab::Random, Tab::Hash, Tab::Json, Tab::Regex};

const char* TabLabel(Tab tab);
// The tab's id in menus and Preferences ("calculator", "units", ...).
const char* TabId(Tab tab);
// The tab of an id; Calculator for one it does not know.
Tab TabFromId(const std::string& id);

// `v` to `digits` significant digits (1 .. 17), without "-0".
std::string FormatNumber(double v, int digits);
// The shortest text that reads back as exactly `v`.
std::string FullNumber(double v);
// "1,000"
std::string FormatCount(uint64_t n);

// ---- Units
std::string CategoryLabel(const std::string& category);  // "length" -> "Length"
// "kilometre per hour (km/h)"; the label alone when it is its symbol.
std::string UnitLabel(const UnitDefinition& unit);
// The units of a category, in the table's order.
std::vector<const UnitDefinition*> UnitsOf(const std::string& category);
// What a category's factors are defined by, for the line under its table.
std::string CategoryNote(const std::string& category);

// ---- Hash
struct HashCheck {
    enum class Kind { Nothing, Match, Differs };
    Kind kind = Kind::Nothing;
    std::string text;  // "Match", "Differs at character 7", "Differs: 40 characters, the hash has 64"
};
// The expected hash against the computed one, case and spaces ignored.
HashCheck CheckHash(const std::string& expected, const std::string& computed);
std::string HashOf(const HashResult& result, const std::string& algorithm);
std::string AlgorithmLabel(const std::string& algorithm);  // "sha256" -> "SHA-256"
std::string SizeText(uint64_t bytes);                       // "143.8 MB", "13 bytes"

// ---- JSON
// "2 objects · 1 array · 7 keys · depth 2 · 208 characters formatted · 157 minified"
std::string JsonSummary(const JSONResult& result);
struct JsonErrorView {
    std::string sentence;                         // "Line 4, column 3: ..."
    std::vector<std::pair<int, std::string>> lines;  // the line before and the error's line
    int caret_column = 0;                         // 1-based, in characters
};
JsonErrorView JsonError(const JSONResult& result);

// ---- Regex
struct RegexRow {
    std::string text;
    int line = 0;      // 1-based
    int column = 0;    // 1-based, in characters
    int start = 0;     // characters from the start of the text
    int end = 0;
    std::vector<std::string> groups;
};
// The matches with lines, columns and character positions (the backend's are
// bytes of UTF-8).
std::vector<RegexRow> RegexRows(const RegexResult& result, const std::string& text);
std::string RegexSummary(const RegexResult& result);  // "3 matches", "No match", the pattern's error

// ---- Random
struct Figure {
    std::string label, value;
};
std::vector<Figure> RandomFigures(const RandomNumberResult& result, int digits);
// "1,000 values · Normal(10, 2) · seed 42"
std::string RandomTitle(const RandomNumberResult& result, const std::vector<double>& parameters);
std::string DistributionLabel(const std::string& distribution);
// The values one per line, for Copy and a CSV.
std::string ValuesText(const std::vector<double>& values, const std::string& header);

}  // namespace cyxwiz::utilities_view
