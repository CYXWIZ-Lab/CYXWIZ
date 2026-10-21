#include "core/node_executors/text_feature_matrix.h"
#include "core/sparse_feature_dataset.h"

#include <arrow/api.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void CheckNear(float actual, float expected, const std::string& message) {
    Check(std::fabs(actual - expected) < 1e-6f, message);
}

cyxwiz::TextFeatureMatrix MakeMatrix() {
    cyxwiz::TextFeatureMatrix matrix;
    matrix.num_rows = 3;
    matrix.num_features = 4;
    matrix.feature_names = {"alpha", "beta", "delta", "gamma"};
    matrix.labels = {1, 0, 1};
    matrix.label_name = "y";
    Check(cyxwiz::AppendNormalizedTextFeatureRow(
              matrix, {{3, 0.5f}, {0, 1.0f}}, "none").ok(),
          "could not append first row");
    Check(cyxwiz::AppendNormalizedTextFeatureRow(
              matrix, {}, "none").ok(),
          "could not append empty row");
    Check(cyxwiz::AppendNormalizedTextFeatureRow(
              matrix, {{1, 2.0f}}, "none").ok(),
          "could not append final row");
    return matrix;
}

void TestSharedDenseAndSparseEmission() {
    const auto matrix = MakeMatrix();
    auto dense_result =
        cyxwiz::BuildDenseTextFeatureTable(matrix, "tfidf_");
    Check(dense_result.ok(), dense_result.status().ToString());
    const auto dense = dense_result.ValueOrDie();
    Check(dense->num_rows() == 3 && dense->num_columns() == 5,
          "dense output shape mismatch");

    auto sparse_result = cyxwiz::BuildSparseTextFeatureDataset(
        matrix, "shared_sparse");
    Check(sparse_result.ok(), sparse_result.status().ToString());
    const auto sparse = sparse_result.ValueOrDie();
    Check(sparse->GetRowOffsets() ==
              std::vector<int32_t>({0, 2, 2, 3}),
          "sparse row offsets mismatch");
    Check(sparse->GetColumnIndices() ==
              std::vector<int32_t>({0, 3, 1}),
          "sparse column ordering mismatch");
    Check(sparse->GetFeatureNames() == matrix.feature_names,
          "sparse vocabulary mismatch");

    for (int64_t row = 0; row < dense->num_rows(); ++row) {
        for (int64_t column = 0; column < matrix.num_features; ++column) {
            const auto dense_column =
                std::static_pointer_cast<arrow::FloatArray>(
                    dense->column(static_cast<int>(column))->chunk(0));
            float sparse_value = 0.0f;
            const int32_t begin = sparse->GetRowOffsets()[
                static_cast<size_t>(row)];
            const int32_t end = sparse->GetRowOffsets()[
                static_cast<size_t>(row + 1)];
            for (int32_t index = begin; index < end; ++index) {
                if (sparse->GetColumnIndices()[static_cast<size_t>(index)] ==
                    column) {
                    sparse_value =
                        sparse->GetValues()[static_cast<size_t>(index)];
                }
            }
            CheckNear(dense_column->Value(row), sparse_value,
                      "dense/CSR numerical mismatch");
        }
    }
}

void TestNormalizationAndValidation() {
    cyxwiz::TextFeatureMatrix matrix;
    matrix.num_rows = 1;
    matrix.num_features = 3;
    matrix.feature_names = {"a", "b", "c"};
    Check(cyxwiz::AppendNormalizedTextFeatureRow(
              matrix, {{2, 4.0f}, {0, 3.0f}}, "l2").ok(),
          "l2 row append failed");
    Check(matrix.column_indices == std::vector<int32_t>({0, 2}),
          "normalization should preserve canonical column ordering");
    CheckNear(matrix.values[0], 0.6f, "l2 first value mismatch");
    CheckNear(matrix.values[1], 0.8f, "l2 second value mismatch");

    cyxwiz::TextFeatureMatrix duplicate;
    duplicate.num_rows = 1;
    duplicate.num_features = 2;
    duplicate.feature_names = {"a", "b"};
    Check(!cyxwiz::AppendNormalizedTextFeatureRow(
               duplicate, {{0, 1.0f}, {0, 2.0f}}, "none").ok(),
          "duplicate columns should fail closed");

    const auto estimate =
        cyxwiz::EstimateSparseTextFeatureMemory(3, 3, true);
    Check(!estimate.overflow && estimate.raw_output_bytes == 52,
          "sparse memory estimate mismatch");
}

} // namespace

int main() {
    TestSharedDenseAndSparseEmission();
    TestNormalizationAndValidation();
    std::cout << "Shared dense/CSR text feature emission tests passed\n";
    return 0;
}
