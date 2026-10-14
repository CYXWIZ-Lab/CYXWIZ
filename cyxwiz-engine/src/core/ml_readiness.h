#pragma once

// Data Studio's ML readiness job (TOFIX148 step 3, approved mockup v8, D5):
// a dataset's splits side by side before training. Per split: rows, groups,
// IDs, text length, class balance; across splits: groups, IDs and identical
// texts found in two splits (leakage). Roles come from the train split's
// contract (Explore > Profile sets them; a Group column is never inferred).
// Measure runs plain SQL through a QueryRunner (the session query service in
// the Engine, an engine of its own in the test); Build words the result.

#include "data_studio_presentation.h"
#include "dataset_contract.h"
#include "dataset_profiler.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::readiness {

// The columns the checks read ("" when the train split has none).
struct Roles {
    std::string text, id, group, target;
};
// One column per role: one you set first, then the contract, then inferred.
Roles PickRoles(const DatasetContract& contract);

struct SplitTable {
    std::string split;    // "train", "validation", "test"
    std::string dataset;  // the table's catalog name
    std::string shown;    // what the graph calls it
};

struct SplitFacts {
    SplitTable table;
    int64_t rows = -1, groups = -1, ids = -1;
    double words_median = std::numeric_limits<double>::quiet_NaN();
    double words_mean = std::numeric_limits<double>::quiet_NaN();
    std::vector<std::pair<std::string, int64_t>> classes;  // target value -> rows, most first
    std::string error;
};

// Values found in both splits a and b (-1: not counted, no such role).
struct Overlap {
    size_t a = 0, b = 0;
    int64_t groups = -1, ids = -1, texts = -1;
};

struct Facts {
    std::string name;          // the split set ("Frozen train and validation families", "documents")
    Roles roles;
    std::vector<SplitFacts> splits;
    std::vector<Overlap> overlaps;
    std::string balance;       // the graph's Data Loader balance mode ("none", "oversample"); "" without one
    double elapsed_ms = 0;
    std::string error;
};

// The SQL Measure runs (exposed for the test).
std::string SplitSql(const std::string& table, const Roles& roles);
std::string OverlapSql(const std::string& a, const std::string& b, const std::string& column);
std::string ClassSql(const std::string& table, const std::string& target, size_t limit);

// Every split's facts and every pair's overlaps. Blocking: run it off the UI
// thread. `should_stop` returning true stops between queries.
Facts Measure(const std::string& name, const std::vector<SplitTable>& splits, const Roles& roles, const QueryRunner& run,
              const std::function<bool()>& should_stop = {});

struct Line {
    std::string text;
    studio::Tone tone = studio::Tone::Plain;
    bool faint = false;        // a note under the figures
};

struct Card {
    std::string title;         // "Split sizes"
    std::vector<std::string> header;               // a table, when it has one
    std::vector<std::vector<std::string>> rows;
    std::vector<Line> lines;                       // under the table
};

struct View {
    std::string title;         // the split set
    std::string splits;        // "train · validation · test"
    std::string verdict;       // "No leakage found", "Leakage: 3 groups in two splits", "Leakage not checked"
    studio::Tone verdict_tone = studio::Tone::Plain;
    std::string verdict_tip;
    std::vector<Card> cards;
};

View Build(const Facts& facts);

}  // namespace cyxwiz::readiness
