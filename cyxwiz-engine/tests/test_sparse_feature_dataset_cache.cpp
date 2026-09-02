#include "core/sparse_feature_dataset.h"
#include "core/sparse_feature_dataset_cache.h"

#include <arrow/api.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const auto nonce = std::chrono::steady_clock::now()
            .time_since_epoch().count();
        path_ = fs::temp_directory_path() /
            ("cyxwiz_sparse_cache_test_" + std::to_string(nonce));
        std::error_code error;
        Check(fs::create_directories(path_, error) && !error,
              "could not create test directory");
    }

    ~TemporaryDirectory() {
        std::error_code ignored;
        fs::remove_all(path_, ignored);
    }

    const fs::path& Path() const noexcept { return path_; }

private:
    fs::path path_;
};

std::shared_ptr<arrow::ChunkedArray> MakeChunkedLabels() {
    arrow::Int32Builder first_builder;
    Check(first_builder.Append(1).ok(), "could not append first label");
    Check(first_builder.AppendNull().ok(), "could not append null label");
    std::shared_ptr<arrow::Array> first;
    Check(first_builder.Finish(&first).ok(),
          "could not finish first label chunk");

    arrow::Int32Builder second_builder;
    Check(second_builder.Append(0).ok(), "could not append second label");
    std::shared_ptr<arrow::Array> second;
    Check(second_builder.Finish(&second).ok(),
          "could not finish second label chunk");
    return std::make_shared<arrow::ChunkedArray>(
        arrow::ArrayVector{first, second});
}

std::shared_ptr<cyxwiz::SparseFeatureDataset> MakeDataset(
    const std::string& name,
    float first_value) {
    cyxwiz::SparseFeatureDataset::Contents contents;
    contents.name = name;
    contents.num_rows = 3;
    contents.num_features = 4;
    contents.row_offsets = {0, 2, 2, 3};
    contents.column_indices = {0, 3, 1};
    contents.values = {first_value, 0.5f, 2.0f};
    contents.feature_names = {"alpha", "beta", "delta", "gamma"};
    contents.labels = MakeChunkedLabels();
    contents.label_name = "target";
    auto result = cyxwiz::SparseFeatureDataset::Create(std::move(contents));
    Check(result.ok(), result.status().ToString());
    return result.ValueOrDie();
}

void CheckEquivalent(const cyxwiz::SparseFeatureDataset& actual,
                     const cyxwiz::SparseFeatureDataset& expected) {
    Check(actual.GetName() == expected.GetName(), "cache name mismatch");
    Check(actual.GetNumRows() == expected.GetNumRows() &&
              actual.GetNumFeatures() == expected.GetNumFeatures(),
          "cache shape mismatch");
    Check(actual.GetRowOffsets() == expected.GetRowOffsets(),
          "cache row offsets mismatch");
    Check(actual.GetColumnIndices() == expected.GetColumnIndices(),
          "cache column indices mismatch");
    Check(actual.GetValues() == expected.GetValues(),
          "cache values mismatch");
    Check(actual.GetFeatureNames() == expected.GetFeatureNames(),
          "cache feature names mismatch");
    Check(actual.GetLabelName() == expected.GetLabelName(),
          "cache label name mismatch");
    Check(actual.GetLabels() && expected.GetLabels() &&
              actual.GetLabels()->Equals(*expected.GetLabels()),
          "cache labels mismatch");
}

void TestRoundTripAndAtomicReplacement() {
    TemporaryDirectory directory;
    const fs::path cache_path = directory.Path() / "features.cyxcsr";

    const auto original = MakeDataset("cached_sparse", 1.0f);
    auto status = cyxwiz::SparseFeatureDatasetCache::SaveAtomically(
        *original, cache_path.string());
    Check(status.ok(), status.ToString());

    auto loaded = cyxwiz::SparseFeatureDatasetCache::Load(
        cache_path.string());
    Check(loaded.ok(), loaded.status().ToString());
    CheckEquivalent(*loaded.ValueOrDie(), *original);

    const auto replacement = MakeDataset("cached_sparse", 7.0f);
    status = cyxwiz::SparseFeatureDatasetCache::SaveAtomically(
        *replacement, cache_path.string());
    Check(status.ok(), status.ToString());
    loaded = cyxwiz::SparseFeatureDatasetCache::Load(cache_path.string());
    Check(loaded.ok(), loaded.status().ToString());
    CheckEquivalent(*loaded.ValueOrDie(), *replacement);
    Check(loaded.ValueOrDie()->GetValues()[0] == 7.0f,
          "atomic replacement did not publish the new cache");

    size_t temporary_count = 0;
    for (const auto& entry : fs::directory_iterator(directory.Path())) {
        if (entry.path().filename().string().find(".tmp.") !=
            std::string::npos) {
            ++temporary_count;
        }
    }
    Check(temporary_count == 0,
          "successful cache writes should not leave temporary files");
}

void TestOptionalFieldsAndMalformedInput() {
    TemporaryDirectory directory;
    const fs::path cache_path = directory.Path() / "optional.cyxcsr";

    cyxwiz::SparseFeatureDataset::Contents contents;
    contents.name = "optional_sparse";
    contents.num_rows = 2;
    contents.num_features = 3;
    contents.row_offsets = {0, 0, 0};
    auto dataset = cyxwiz::SparseFeatureDataset::Create(std::move(contents));
    Check(dataset.ok(), dataset.status().ToString());
    Check(cyxwiz::SparseFeatureDatasetCache::SaveAtomically(
              *dataset.ValueOrDie(), cache_path.string()).ok(),
          "optional-field cache save failed");
    auto loaded = cyxwiz::SparseFeatureDatasetCache::Load(cache_path.string());
    Check(loaded.ok(), loaded.status().ToString());
    Check(!loaded.ValueOrDie()->GetLabels() &&
              loaded.ValueOrDie()->GetFeatureNames().empty(),
          "absent optional fields should remain absent");

    {
        std::ofstream corrupt(cache_path, std::ios::binary | std::ios::trunc);
        corrupt << "not an Arrow IPC cache";
    }
    loaded = cyxwiz::SparseFeatureDatasetCache::Load(cache_path.string());
    Check(!loaded.ok(), "malformed cache should fail closed");
    Check(!cyxwiz::SparseFeatureDatasetCache::Load("").ok(),
          "empty cache path should fail closed");
}

} // namespace

int main() {
    TestRoundTripAndAtomicReplacement();
    TestOptionalFieldsAndMalformedInput();
    std::cout << "Sparse feature dataset cache tests passed\n";
    return 0;
}
