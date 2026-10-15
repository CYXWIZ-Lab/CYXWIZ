#pragma once

// Data Studio's shell (TOFIX148 plan step 3, approved mockup v8): the open
// graph's data work as seven jobs, the data rail, the "In the graph" node
// strip per job and the last-run line. Pure data in and out: the panel reads
// the graph, the run facts and the dataset catalog into Input, draws View.

#include "graph_model.h"

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace cyxwiz::studio {

enum class Job { BringIn, Explore, Prepare, Validate, Ready, Publish, Results };
constexpr size_t kJobCount = 7;
const char* JobName(Job job);

// The job a node belongs to; false for nodes no job shows (layers, losses).
bool JobOf(gui::NodeType type, Job& job);

struct GraphNode {
    int id = -1;
    gui::NodeType type{};
    std::string name;
    std::string file_path;     // a Data Input's source or an export's target
    std::string dataset_name;  // a Data Input's registry name once loaded ("documents_train")
    bool recipe = false;       // a Preparation Recipe (a Subgraph)
    int recipe_steps = 0;
    std::vector<int> input_from;  // the node linked to each input pin, in pin order (-1: none)
    std::string balance;       // a Data Loader's balance mode
    std::map<std::string, std::string> parameters;  // checks and validators: their settings (Validate)
    // Its latest run in this session.
    bool ran = false;
    int64_t rows = -1, columns = -1;
    double seconds = 0.0;
    std::string outcome, error, output_dataset, finished_at;
};

struct Dataset {
    std::string name;          // registry name ("ds_sql_101")
    std::string label;         // what the graph calls it ("Tier 1 books (ZIP text)"), may be empty
    size_t rows = 0, columns = 0;
    bool tabular = true;
};

struct Input {
    std::string graph_name;    // the graph file's stem
    std::vector<GraphNode> nodes;
    std::vector<Dataset> datasets;
};

enum class Tone { Plain, Ok, Warn, Error };

struct RailItem {
    enum class Kind { DataInput, NodeResult, Published, Other };
    Kind kind = Kind::Other;
    std::string label;
    std::string meta;          // rows, "pass", "not loaded"
    std::string dataset;       // what selecting it shows ("" when nothing to show yet)
    std::string tip;
    int node_id = -1;
    Tone tone = Tone::Plain;
};

struct RailSection {
    std::string title;
    std::vector<RailItem> items;
};

struct NodeChip {
    std::string name;          // "Data Input", "Check"
    std::string note;          // "× 4", "tier1_paragraphs"
    std::vector<int> ids;
};

struct JobView {
    Job job = Job::BringIn;
    std::string name;
    std::string state;         // the line under the tab: "4 Data Inputs", "4 of 4 pass"
    std::vector<NodeChip> chips;
    std::vector<int> node_ids; // every node of the job (Show on canvas)
};

// Bring in: one row per Data Input node (approved mockup v8).
struct SourceRow {
    int node_id = -1;
    std::string name;
    std::string file;          // the node's file_path as written ("datasets/.../tier1_books.zip")
    std::string size;          // rows x columns ("7 x 6" with a times sign); "" until loaded
    std::string state;         // "loaded", "not loaded", "failed"
    std::string detail;        // the error when it failed
    std::string dataset;       // its registry name when loaded
    Tone tone = Tone::Plain;
};

// ML readiness: splits to compare. A Data Split's inputs (train, validation,
// test in pin order) or catalog datasets named <stem>_train / _validation /
// _test.
struct SplitPart {
    std::string split;         // "train", "validation", "test"
    std::string dataset;       // registry name
    std::string label;         // what the graph calls it
};

struct SplitSet {
    std::string name;          // the Data Split's name, or the stem ("documents")
    int node_id = -1;          // the Data Split (-1: from the catalog's names)
    std::vector<SplitPart> parts;
    std::string note;          // why a split is missing ("validation: Data Input not loaded")
};

struct View {
    std::string graph_label;   // "berean_v3_tier1_paragraphs · recipe tier1_paragraphs"
    std::vector<RailSection> rail;
    std::vector<SourceRow> sources;  // Bring in
    std::vector<SplitSet> split_sets;  // ML readiness
    std::string balance;       // the graph's Data Loader balance mode ("" without one)
    std::array<JobView, kJobCount> jobs;
    std::string last_run;      // "tier1_paragraphs · 15,349 rows · 4/4 checks · 3.1 s"; "" before any run
    Tone last_run_tone = Tone::Plain;
};

View Build(const Input& input);

// Prepare: a recipe's steps in run order as its step list shows them.
struct StepRow {
    int id = -1;
    std::string name;          // "tier1_paragraphs"
    std::string kind;          // "SQL Query", "Check", "step node"
    std::string meta;          // "15,349 rows · 3.1 s", "pass", "failed", "not run"
    std::string tip;           // the error or a check's outcome
    Tone tone = Tone::Plain;
    bool sql = false;          // edited in place (an SQL Query step)
};
std::vector<StepRow> StepRows(const std::vector<GraphNode>& steps);

// The split a dataset name ends in ("documents_train" -> "train", stem
// "documents"); "" when none.
std::string SplitOfName(const std::string& name, std::string* stem = nullptr);

// "15,349"
std::string Thousands(int64_t n);

}  // namespace cyxwiz::studio
