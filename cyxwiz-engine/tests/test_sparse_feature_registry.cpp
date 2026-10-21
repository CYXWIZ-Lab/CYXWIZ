#include "core/data_registry.h"
#include "core/sparse_feature_dataset.h"

#include <arrow/api.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::shared_ptr<arrow::ChunkedArray> MakeLabels(
    const std::vector<int32_t>& values) {
    arrow::Int32Builder builder;
    Check(builder.AppendValues(values).ok(), "could not append labels");
    std::shared_ptr<arrow::Array> array;
    Check(builder.Finish(&array).ok(), "could not finish labels");
    return std::make_shared<arrow::ChunkedArray>(array);
}

std::shared_ptr<cyxwiz::SparseFeatureDataset> MakeDataset(
    const std::string& name,
    float first_value = 1.0f) {
    cyxwiz::SparseFeatureDataset::Contents contents;
    contents.name = name;
    contents.num_rows = 2;
    contents.num_features = 3;
    contents.row_offsets = {0, 2, 3};
    contents.column_indices = {0, 2, 1};
    contents.values = {first_value, 2.0f, 3.0f};
    contents.feature_names = {"alpha", "beta", "gamma"};
    contents.labels = MakeLabels({1, 0});
    contents.label_name = "label";
    auto result = cyxwiz::SparseFeatureDataset::Create(std::move(contents));
    Check(result.ok(), result.status().ToString());
    return result.ValueOrDie();
}

void TestRegistrationInspectionAndReplacement() {
    auto& registry = cyxwiz::DataRegistry::Instance();
    registry.ClearAllSparseFeatureDatasets();

    Check(!registry.RegisterSparseFeatureDataset(nullptr),
          "null registration should fail");

    const auto original = MakeDataset("registry_sparse");
    Check(registry.RegisterSparseFeatureDataset(original),
          "valid sparse registration should succeed");
    Check(registry.IsSparseFeatureDataset("registry_sparse"),
          "registered sparse dataset should be discoverable");
    Check(registry.GetSparseFeatureDataset("registry_sparse") == original,
          "registry should return the published immutable dataset");

    const auto info = registry.InspectSparseFeatureDataset("registry_sparse");
    Check(info.has_value(), "inspection snapshot should exist");
    Check(info->num_rows == 2 && info->num_features == 3 && info->nnz == 3,
          "inspection shape or nnz mismatch");
    Check(info->density == 0.5, "inspection density mismatch");
    Check(info->feature_storage_bytes == 36,
          "inspection CSR byte count mismatch");
    Check(info->has_labels && info->label_name == "label" &&
              info->label_type == "int32" && info->label_null_count == 0,
          "inspection label metadata mismatch");

    const auto replacement = MakeDataset("registry_sparse", 7.0f);
    Check(registry.RegisterSparseFeatureDataset(replacement),
          "replacement registration should succeed");
    Check(registry.GetSparseFeatureDataset("registry_sparse") == replacement,
          "replacement should publish atomically under the same name");
    Check(original->GetValues()[0] == 1.0f,
          "an existing reader snapshot should remain immutable");
}

void TestListingCascadeAndClear() {
    auto& registry = cyxwiz::DataRegistry::Instance();
    registry.ClearAllSparseFeatureDatasets();

    Check(registry.RegisterSparseFeatureDataset(MakeDataset("source")),
          "source registration failed");
    Check(registry.RegisterSparseFeatureDataset(
              MakeDataset("source__materialized")),
          "materialized registration failed");
    Check(registry.RegisterSparseFeatureDataset(MakeDataset("other")),
          "other registration failed");

    const auto listed = registry.ListSparseFeatureDatasets();
    Check(listed.size() == 3, "sparse listing count mismatch");
    Check(listed[0].name == "other" && listed[1].name == "source" &&
              listed[2].name == "source__materialized",
          "sparse listing should be deterministic by dataset name");

    Check(registry.UnregisterSparseFeatureDataset("source"),
          "source unregister should report removal");
    Check(!registry.IsSparseFeatureDataset("source") &&
              !registry.IsSparseFeatureDataset("source__materialized"),
          "source unregister should cascade to its materialized dataset");
    Check(registry.IsSparseFeatureDataset("other"),
          "unrelated sparse dataset should remain registered");
    Check(!registry.UnregisterSparseFeatureDataset("missing"),
          "missing unregister should report no removal");

    Check(registry.ClearAllSparseFeatureDatasets() == 1,
          "clear should report the remaining sparse dataset");
    Check(registry.ListSparseFeatureDatasets().empty(),
          "clear should remove all sparse datasets");
}

} // namespace

int main() {
    TestRegistrationInspectionAndReplacement();
    TestListingCascadeAndClear();
    std::cout << "Sparse feature registry lifecycle tests passed\n";
    return 0;
}
