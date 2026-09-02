#include "../src/core/sparse_feature_dataset.h"

#include <arrow/api.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::shared_ptr<arrow::ChunkedArray> MakeIntLabels(
    const std::vector<int32_t>& values) {
    arrow::Int32Builder builder;
    Check(builder.AppendValues(values).ok(), "could not append labels");
    std::shared_ptr<arrow::Array> array;
    Check(builder.Finish(&array).ok(), "could not finish labels");
    return std::make_shared<arrow::ChunkedArray>(array);
}

cyxwiz::SparseFeatureDataset::Contents ValidContents() {
    cyxwiz::SparseFeatureDataset::Contents contents;
    contents.name = "tiny_tfidf";
    contents.num_rows = 3;
    contents.num_features = 4;
    contents.row_offsets = {0, 2, 2, 3};
    contents.column_indices = {0, 3, 1};
    contents.values = {1.0f, 0.5f, 2.0f};
    contents.feature_names = {"alpha", "beta", "delta", "gamma"};
    contents.labels = MakeIntLabels({1, 0, 1});
    contents.label_name = "y";
    return contents;
}

void ExpectInvalid(cyxwiz::SparseFeatureDataset::Contents contents,
                   const std::string& expected_message) {
    auto result = cyxwiz::SparseFeatureDataset::Create(std::move(contents));
    Check(!result.ok(), "invalid CSR unexpectedly succeeded");
    Check(result.status().message().find(expected_message) != std::string::npos,
          "unexpected validation message: " + result.status().ToString());
}

void TestValidCanonicalCsr() {
    auto result = cyxwiz::SparseFeatureDataset::Create(ValidContents());
    Check(result.ok(), result.status().ToString());
    const auto dataset = result.ValueOrDie();

    Check(dataset->GetName() == "tiny_tfidf", "name mismatch");
    Check(dataset->GetNumRows() == 3, "row count mismatch");
    Check(dataset->GetNumFeatures() == 4, "feature count mismatch");
    Check(dataset->GetNnz() == 3, "nnz mismatch");
    Check(std::abs(dataset->GetDensity() - 0.25) < 1e-12,
          "density mismatch");
    Check(dataset->GetFeatureStorageBytes() == 40,
          "CSR feature byte estimate mismatch");
    Check(dataset->GetDenseFeatureBytes() == 48,
          "dense feature byte estimate mismatch");
    Check(dataset->GetLabelStorageBytes() > 0,
          "label storage should be reported");
    Check(dataset->GetEstimatedHostMemoryBytes() >=
              dataset->GetFeatureStorageBytes() +
                  dataset->GetLabelStorageBytes(),
          "estimated host memory should include feature and label storage");
    Check(dataset->GetRowOffsets()[2] == 2,
          "empty middle row was not preserved");
    Check(dataset->GetLabels() && dataset->GetLabels()->length() == 3,
          "label payload mismatch");
}

void TestOptionalMetadataAndEmptyRows() {
    cyxwiz::SparseFeatureDataset::Contents contents;
    contents.name = "empty_rows";
    contents.num_rows = 2;
    contents.num_features = 3;
    contents.row_offsets = {0, 0, 0};

    auto result = cyxwiz::SparseFeatureDataset::Create(std::move(contents));
    Check(result.ok(), result.status().ToString());
    Check(result.ValueOrDie()->GetDensity() == 0.0,
          "all-empty CSR density should be zero");
    Check(result.ValueOrDie()->GetFeatureStorageBytes() == 12,
          "empty CSR should still account for row offsets");
}

void TestStructuralValidation() {
    auto contents = ValidContents();
    contents.name.clear();
    ExpectInvalid(std::move(contents), "name must not be empty");

    contents = ValidContents();
    contents.num_rows = -1;
    ExpectInvalid(std::move(contents), "num_rows must be non-negative");

    contents = ValidContents();
    contents.num_features = 0;
    ExpectInvalid(std::move(contents), "num_features must be positive");

    contents = ValidContents();
    contents.num_rows =
        static_cast<int64_t>((std::numeric_limits<int32_t>::max)()) + 1;
    ExpectInvalid(std::move(contents), "num_rows exceeds the int32 CSR contract");

    contents = ValidContents();
    contents.row_offsets = {0, 2, 3};
    ExpectInvalid(std::move(contents), "row_offsets length");

    contents = ValidContents();
    contents.row_offsets.front() = 1;
    ExpectInvalid(std::move(contents), "row_offsets must start at zero");

    contents = ValidContents();
    contents.row_offsets = {0, 2, 1, 3};
    ExpectInvalid(std::move(contents), "monotonic");

    contents = ValidContents();
    contents.row_offsets.back() = 2;
    ExpectInvalid(std::move(contents), "final row offset must equal nnz");

    contents = ValidContents();
    contents.column_indices.pop_back();
    ExpectInvalid(std::move(contents), "equal length");
}

void TestValueAndColumnValidation() {
    auto contents = ValidContents();
    contents.column_indices[1] = 4;
    ExpectInvalid(std::move(contents), "outside [0, num_features)");

    contents = ValidContents();
    contents.column_indices[1] = 0;
    ExpectInvalid(std::move(contents), "strictly increasing");

    contents = ValidContents();
    contents.values[0] = 0.0f;
    ExpectInvalid(std::move(contents), "finite and nonzero");

    contents = ValidContents();
    contents.values[0] = (std::numeric_limits<float>::infinity)();
    ExpectInvalid(std::move(contents), "finite and nonzero");
}

void TestMetadataAndLabelValidation() {
    auto contents = ValidContents();
    contents.feature_names.pop_back();
    ExpectInvalid(std::move(contents), "exactly num_features");

    contents = ValidContents();
    contents.label_name.clear();
    ExpectInvalid(std::move(contents), "label_name is required");

    contents = ValidContents();
    contents.labels = MakeIntLabels({1, 0});
    ExpectInvalid(std::move(contents), "label length must equal num_rows");

    contents = ValidContents();
    contents.labels.reset();
    ExpectInvalid(std::move(contents), "label_name must be empty");
}

} // namespace

int main() {
    TestValidCanonicalCsr();
    TestOptionalMetadataAndEmptyRows();
    TestStructuralValidation();
    TestValueAndColumnValidation();
    TestMetadataAndLabelValidation();
    std::cout << "SparseFeatureDataset contract tests passed\n";
    return 0;
}
