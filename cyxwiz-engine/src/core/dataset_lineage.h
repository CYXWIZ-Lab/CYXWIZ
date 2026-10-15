#pragma once

// Dataset lineage (TOFIX148 Publish, approved mockup v8): what made a file the
// Data Pipeline wrote, kept beside it as <file>.lineage.json. Written by the
// executor when an export node (Export CSV / JSON / Parquet, Data Output)
// finishes, so every run records it; Data Studio's Publish shows it.
//
// It names the graph, the steps above the export in run order (type, name,
// settings, outcome), the recipe and its revision, the input files with their
// SHA-256, the result (rows, columns, schema, file SHA-256) and the checks.

#include <nlohmann/json.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

struct LineageStep {
    int id = -1;
    std::string type;    // pipeline type name ("SQLQuery", "RowCountCheck")
    std::string name;
    std::string recipe;  // the Preparation Recipe it belongs to ("" outside one)
    std::map<std::string, std::string> parameters;
    std::vector<int> inputs;
    std::string outcome;  // the operator's summary (a check's "passed: ...")
};

struct LineageFile {
    std::string path;
    uint64_t bytes = 0;
    std::string sha256;  // "" when it could not be read
};

struct LineageCheck {
    std::string name;
    std::string outcome;
    bool passed = false;
};

struct DatasetLineage {
    int version = 1;
    std::string written_at;  // local time, "2026-10-15 14:28:41"
    std::string engine;      // "CyxWiz Engine"
    std::string graph;       // the graph file the run came from ("" for an unsaved graph)
    LineageFile output;      // the file written
    std::string format;      // "parquet", "csv", "json"
    int64_t rows = -1;
    int64_t columns = -1;
    std::vector<std::pair<std::string, std::string>> schema;  // column name, Arrow type
    std::vector<LineageFile> inputs;                          // the Data Inputs' files
    std::vector<LineageStep> steps;                           // above the export, run order
    std::string recipe;                                       // the recipe the export reads ("" none)
    std::string recipe_revision;                              // 12 hex: its steps' types and settings
    std::vector<LineageCheck> checks;
};

// "<output>.lineage.json"
std::string LineagePathFor(const std::string& output_path);

// The recipe's revision: SHA-256 of its steps' types and settings in run order
// (names and outcomes left out), first 12 hex digits; "" without steps.
std::string RecipeRevision(const std::vector<LineageStep>& steps, const std::string& recipe);

// The checks among the steps (RowCountCheck), passed when the outcome starts "passed".
std::vector<LineageCheck> LineageChecks(const std::vector<LineageStep>& steps);

nlohmann::json LineageToJson(const DatasetLineage& lineage);
bool LineageFromJson(const nlohmann::json& json, DatasetLineage& lineage, std::string& error);

// Written to a temporary file, then renamed over the old one.
bool WriteLineage(const DatasetLineage& lineage, std::string& error);
bool ReadLineage(const std::string& output_path, DatasetLineage& lineage, std::string& error);

// What Publish shows for it, one line each.
struct LineageSummary {
    std::string made_by;  // "recipe tier1_paragraphs, revision 3f2a9c1b0d4e (6 steps)"
    std::string inputs;   // "tier1_books.zip - 2.1 MB - sha256 9c4e...a1"
    std::string result;   // "15,349 rows - 13 columns - sha256 1b7f...0e"
    std::string checks;   // "4 of 4 pass" / "3 of 4 pass: 'Check: Law' failed" / "no checks"
    bool checks_ok = true;
    std::string written;  // "2026-10-15 14:28:41 from berean_v3_tier1_paragraphs.cyxgraph"
};
LineageSummary SummarizeLineage(const DatasetLineage& lineage);

// A short digest for display: first 4 and last 2 hex digits ("9c4e...a1").
std::string ShortDigest(const std::string& sha256);

}  // namespace cyxwiz
