#pragma once

// The rows an analysis node fits on (K-Means, DBSCAN, Hierarchical, Gaussian
// mixture, PCA, t-SNE, UMAP; TOFIX129 D5): the rows with every feature present
// and finite, read as doubles, optionally standardised by the population
// standard deviation as Explore > Groups and Projection do. The other rows are
// left out of the fit; the columns the node adds are null there, so the output
// keeps every input row in order.

#include <arrow/api.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

struct FeatureRows {
    std::vector<std::vector<double>> data;  // [fitted row][feature]
    std::vector<int64_t> rows;              // each fitted row's index in the table
    int64_t total = 0;                      // rows in the table
};

// Reads `names` (numeric columns) from `table`. Refuses a feature with one
// value in every fitted row when standardising (it cannot be scaled).
arrow::Status ReadFeatureRows(const std::shared_ptr<arrow::Table>& table, const std::vector<std::string>& names,
                              const std::string& op_name, bool standardise, FeatureRows& out,
                              const std::function<bool()>& cancelled = {});

// Adds an int32 column: `values[i]` on row `rows[i]`, null on the rows left out.
arrow::Result<std::shared_ptr<arrow::Table>> AppendFittedInt32(const std::shared_ptr<arrow::Table>& table, const std::string& name,
                                                               const std::vector<int64_t>& rows, const std::vector<int>& values);

// Adds float64 columns `names[k]`: `values[i][k]` on row `rows[i]`, null elsewhere.
arrow::Result<std::shared_ptr<arrow::Table>> AppendFittedColumns(const std::shared_ptr<arrow::Table>& table,
                                                                 const std::vector<std::string>& names,
                                                                 const std::vector<int64_t>& rows,
                                                                 const std::vector<std::vector<double>>& values);

}  // namespace cyxwiz
