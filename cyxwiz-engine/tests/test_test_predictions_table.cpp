// BuildTestPredictionsTable (TOFIX140): the latest test run's per-sample
// results as a table for the ROC / PR nodes and panels - columns, binary
// "score" (positive class), multi-class "score_<class>" with duplicate-name
// fallback, and no score columns when the scores were not kept; (TOFIX157 B2)
// the row_id column and source metadata, and the regression table.
#include "../src/core/test_predictions_table.h"

#include <arrow/api.h>
#include <arrow/util/key_value_metadata.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using cyxwiz::BuildTestPredictionsTable;

namespace {

int failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        ++failures;
    }
}

std::vector<std::string> Names(const std::shared_ptr<arrow::Table>& table) {
    std::vector<std::string> names;
    for (const auto& field : table->schema()->fields()) names.push_back(field->name());
    return names;
}

double DoubleAt(const std::shared_ptr<arrow::Table>& table, const std::string& column, int row) {
    const auto chunk = std::static_pointer_cast<arrow::DoubleArray>(table->GetColumnByName(column)->chunk(0));
    return chunk->Value(row);
}

}  // namespace

int main() {
    // Binary: score = P(class 1).
    const auto binary = BuildTestPredictionsTable({0, 1, 1}, {0, 1, 0}, {0.9f, 0.8f, 0.6f},
                                                  {0.9f, 0.1f, 0.2f, 0.8f, 0.6f, 0.4f}, 2, {"no", "yes"});
    Check(binary != nullptr, "binary table built");
    if (binary) {
        Check(binary->num_rows() == 3, "binary rows");
        Check(Names(binary) == std::vector<std::string>({"true_class", "predicted_class", "confidence", "score"}),
              "binary columns");
        Check(std::abs(DoubleAt(binary, "score", 1) - 0.8) < 1e-6, "binary score is the positive class");
    }

    // Multi-class: one score per class, named; a repeated name falls back to the index.
    const auto multi = BuildTestPredictionsTable({2, 0}, {2, 1}, {0.7f, 0.5f},
                                                 {0.1f, 0.2f, 0.7f, 0.3f, 0.5f, 0.2f}, 3, {"cat", "cat", "dog"});
    Check(multi != nullptr, "multi-class table built");
    if (multi) {
        Check(Names(multi) == std::vector<std::string>(
                                  {"true_class", "predicted_class", "confidence", "score_cat", "score_1", "score_dog"}),
              "multi-class columns");
        Check(std::abs(DoubleAt(multi, "score_dog", 0) - 0.7) < 1e-6, "multi-class score per class");
    }

    // Scores not kept: the class columns only.
    const auto plain = BuildTestPredictionsTable({0, 1}, {0, 1}, {1.0f, 1.0f}, {}, 0, {});
    Check(plain != nullptr && plain->num_columns() == 3, "no score columns without scores");

    // Mismatched lengths are refused.
    Check(BuildTestPredictionsTable({0, 1}, {0}, {1.0f, 1.0f}, {}, 0, {}) == nullptr, "length mismatch refused");

    // The rows of the tested table (TOFIX157 B2): row_id first, the table's
    // name in the schema metadata; row ids that do not match the rows are left out.
    const auto sourced = BuildTestPredictionsTable({0, 1}, {0, 1}, {1.0f, 1.0f}, {}, 0, {}, {{7, 2}, "iris"});
    Check(sourced && Names(sourced) == std::vector<std::string>({"row_id", "true_class", "predicted_class", "confidence"}),
          "row_id column first");
    if (sourced) {
        const auto ids = std::static_pointer_cast<arrow::Int64Array>(sourced->GetColumnByName("row_id")->chunk(0));
        Check(ids->Value(0) == 7 && ids->Value(1) == 2, "row ids kept in test order");
        const auto metadata = sourced->schema()->metadata();
        Check(metadata && metadata->FindKey(cyxwiz::kTestPredictionsSourceKey) >= 0 &&
                  metadata->value(metadata->FindKey(cyxwiz::kTestPredictionsSourceKey)) == "iris",
              "source dataset in the metadata");
    }
    const auto unsourced = BuildTestPredictionsTable({0, 1}, {0, 1}, {1.0f, 1.0f}, {}, 0, {}, {{7}, "iris"});
    Check(unsourced && unsourced->num_columns() == 3, "a row id per row or none");

    // Regression: one target, then two.
    const auto one = cyxwiz::BuildRegressionPredictionsTable({1.0f, 2.0f}, {1.5f, 1.0f}, 1, {{0, 1}, "houses"});
    Check(one && Names(one) == std::vector<std::string>({"row_id", "true_value", "predicted_value", "error"}),
          "regression columns");
    if (one) Check(std::abs(DoubleAt(one, "error", 1) + 1.0) < 1e-6, "error = predicted - true");
    const auto two = cyxwiz::BuildRegressionPredictionsTable({1, 2, 3, 4}, {1, 2, 3, 5}, 2);
    Check(two && Names(two) == std::vector<std::string>({"true_0", "predicted_0", "error_0", "true_1", "predicted_1",
                                                         "error_1"}),
          "two targets");
    if (two) Check(two->num_rows() == 2 && std::abs(DoubleAt(two, "error_1", 1) - 1.0) < 1e-6, "second target per row");
    Check(cyxwiz::BuildRegressionPredictionsTable({1, 2, 3}, {1, 2, 3}, 2) == nullptr, "values not a whole number of rows");

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("test predictions table: columns, scores and refusals as expected. OK\n");
    return 0;
}
