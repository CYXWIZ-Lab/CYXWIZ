#pragma once

// The compile report (TOFIX129 B3): what Compile (F7) found, why Train or
// Local Debug did not start, where each layer runs and the configuration the
// compile produced, plus the memory-risk question before training. Pure data
// in and out: no ImGui, no registry, so the popup and Copy report say the
// same words and the wording is tested.

#include "graph_compiler.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::compile_report {

enum class Mode { Compile, TrainBlocked, DebugBlocked, CheckpointBlocked };

enum class Tone { Ok, Warning, Error, Info, Neutral };

// Where the compiled dataset lives (the Backing line of the configuration).
enum class Backing { Unknown, NotLoaded, InMemoryArrow, DiskParquet, OtherLoaded };

struct ConfigFacts {
    size_t layers = 0;
    size_t input_size = 0;
    std::vector<size_t> input_shape;
    size_t output_size = 0;

    std::string dataset;            // registry name, or the Data Input's name when not loaded
    Backing backing = Backing::Unknown;
    std::string backing_label;      // OtherLoaded: "text, loaded lazily"
    double backing_megabytes = 0.0; // Arrow: resident; Parquet: on disk
    long long rows = -1;
    long long row_groups = -1;

    bool has_split = false;
    std::string split_node;
    double train_ratio = 0.0, val_ratio = 0.0, test_ratio = 0.0;

    bool has_loader = false;
    std::string loader_node;
    int batch_size = 0;
    int epochs = 0;
    bool shuffle = false;
    bool drop_last = false;
    int num_workers = 0;
    int prefetch_factor = 0;
    int log_interval = 0;
    int validation_freq = 0;
    unsigned long long seed = 0;
    int grad_accum_steps = 1;

    std::string loss;
    std::string optimizer;
    double learning_rate = 0.0;
    bool normalize = false;
    double norm_mean = 0.0, norm_std = 0.0;
    std::vector<float> norm_channel_mean, norm_channel_std;  // per channel (TOFIX178 1.4)
    bool norm_imagenet = false;
    bool one_hot = false;
    int num_classes = 0;

    // Learnable parameters summed from the layers (the AS COMPILED cards'
    // counts); -1 while not counted.
    long long parameters = -1;
};

struct Input {
    Mode mode = Mode::Compile;
    std::string graph_name;     // "fasttext_check"
    std::string stop_reason;    // why Train / Debug / the checkpoint stopped; empty for compile errors
    bool compiled_ok = false;   // no Error-level finding
    std::vector<ValidationIssue> issues;
    std::vector<BackendPlacementEntry> placements;
    bool has_config = false;
    ConfigFacts config;
};

struct Chip {
    std::string text;
    Tone tone = Tone::Neutral;
};

// A node the finding's fix adds after its node (the renderer creates it).
// AudioFeatures offers Mel Spectrogram, Spectrogram and MFCC.
enum class AddNodeFix { None, TextTokenizer, Resize, AudioFeatures };

struct Finding {
    IssueLevel level = IssueLevel::Info;
    Tone tone = Tone::Info;
    int node_id = -1;
    std::string node_name;
    std::string message;
    std::string code;           // empty for the generic code (C-1)
    bool open_dialog = false;   // the fix is in the node's dialog (data not loaded or failed to load)
    bool open_data_studio = false;
    AddNodeFix add_node = AddNodeFix::None;
};

struct PlacementRow {
    int node_id = -1;
    std::string layer;          // node name, else its type
    std::string runs_on;        // "GPU", "CPU", "GPU and CPU", "At risk", "Unsupported", "Unknown"
    Tone tone = Tone::Neutral;
    std::string engine;         // "ArrayFire", "native provider cyxwiz.nvidia-cublas-cell"
    std::string why;            // a short plain sentence; with the suggested action when it needs attention
    bool attention = false;
    std::vector<std::pair<std::string, std::string>> details;
};

struct Tile {
    std::string value;
    std::string label;
};

struct ConfigGroup {
    std::string title;
    std::vector<std::pair<std::string, std::string>> rows;
};

struct Report {
    std::string title;          // window title
    std::string verdict;
    Tone verdict_tone = Tone::Neutral;
    std::string subtitle;
    bool ready = false;
    size_t errors = 0, warnings = 0, notes = 0;
    std::vector<Chip> chips;
    std::vector<Finding> findings;  // errors, then warnings, then notes; compile order within a level
    std::string placement_sentence;
    std::vector<Chip> placement_chips;
    std::vector<PlacementRow> placements;  // rows that need attention first
    std::vector<Tile> tiles;
    std::vector<ConfigGroup> config;
    bool offer_train = false;
    std::string copy_text;
};

Report BuildReport(const Input& input);

// "1 error", "3 warnings".
std::string Count(size_t n, const char* one, const char* many);

// Before training: a step's estimated memory against what is free.
struct MemoryRiskInput {
    std::string node_name;
    int node_id = -1;
    std::string risk;                       // "risky" or "warning"
    uint64_t estimated_bytes = 0;
    uint64_t available_bytes = 0;           // free memory now
    uint64_t budget_bytes = 0;              // the safe budget
    std::string detail;                     // the preflight's own sentence
};

struct MemoryRiskView {
    std::string title;                      // window title
    std::string verdict;
    Tone tone = Tone::Warning;
    std::string subtitle;
    std::vector<std::pair<std::string, std::string>> rows;
    double estimate_share = 0.0;            // estimate / available, 0..1 for the bar
    std::string advice;
    std::string detail;
};

MemoryRiskView BuildMemoryRiskView(const MemoryRiskInput& input);

}  // namespace cyxwiz::compile_report
