#pragma once

// What the Dashboard shows of sparse features (TOFIX134 P3, approved board
// 19): the Count / TF-IDF Vectorizer's CSR matrix read directly (not through
// SQL), exact. Pure: the caller hands the arrays; a label filter keeps the
// rows of the chosen classes (a click on the label bar).

#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::dashboard {

struct SparseInput {
    size_t rows = 0, features = 0;
    const int32_t* offsets = nullptr;   // rows + 1
    const int32_t* indices = nullptr;   // nnz
    const float* values = nullptr;      // nnz
    std::vector<std::string> feature_names;  // empty: "f0", "f1", ...
    std::vector<std::string> labels;         // one per row (empty: no labels)
    size_t bytes = 0;                        // the matrix in memory
};

struct SparseSummary {
    size_t rows_all = 0, rows = 0;   // rows, and the rows kept by the filter
    size_t features = 0;
    size_t nnz = 0;                  // non-zero values in the kept rows
    double density = 0;              // nnz / (rows x features)
    double memory_mb = 0;
    // Classes: rows per label, all rows and kept rows (label order: most rows first).
    std::vector<std::string> classes;
    std::vector<size_t> class_rows_all, class_rows;
    // Top features by total weight in the kept rows.
    std::vector<std::pair<std::string, double>> top_features;
    // Per kept row: how many features it uses; per feature used: how many kept rows use it.
    std::vector<double> features_per_row, rows_per_feature;
    // Top features by class: the mean weight per row of each class
    // (by_class_features x classes, row major).
    std::vector<std::string> by_class_features;
    std::vector<double> by_class_weight;
    // The first kept rows: row number, label, features used, the strongest four.
    struct Row {
        size_t row = 0;
        std::string label;
        size_t used = 0;
        std::string strongest;  // "feel 0.42, like 0.31, ..."
    };
    std::vector<Row> sample;
};

// `keep`: the labels whose rows count (empty: every row).
SparseSummary SummarizeSparse(const SparseInput& in, const std::set<std::string>& keep = {}, size_t top = 15, size_t by_class_top = 10,
                              size_t sample_rows = 20);

}  // namespace cyxwiz::dashboard
