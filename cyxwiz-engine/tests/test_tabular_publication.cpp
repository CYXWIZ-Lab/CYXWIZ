#include "../src/core/arrow_dataset.h"
#include "../src/core/data_registry.h"
#include "../src/core/parquet_backed_dataset.h"
#include "../src/core/sparse_feature_dataset.h"

#include <arrow/api.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

int RunStagedPublicationTests();

namespace {

std::atomic<int> checks{0};

void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

class RegistryFixture {
public:
    RegistryFixture() {
        auto& registry = cyxwiz::DataRegistry::Instance();
        registry.SetOnDatasetLoaded({});
        registry.UnloadAll();
        registry.ClearAllTabularDatasets();
        registry.ClearAllSparseFeatureDatasets();
    }

    ~RegistryFixture() {
        auto& registry = cyxwiz::DataRegistry::Instance();
        registry.SetOnDatasetLoaded({});
        registry.UnloadAll();
        registry.ClearAllTabularDatasets();
        registry.ClearAllSparseFeatureDatasets();
        for (const auto& path : files_) {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    }

    std::filesystem::path TempPath(const std::string& suffix) {
        auto path = std::filesystem::temp_directory_path() /
            ("cyxwiz_tabular_publication_" +
             std::to_string(std::chrono::steady_clock::now()
                                .time_since_epoch()
                                .count()) +
             "_" + suffix);
        files_.push_back(path);
        return path;
    }

private:
    std::vector<std::filesystem::path> files_;
};

std::shared_ptr<arrow::Array> Int64Array(const std::vector<int64_t>& values) {
    arrow::Int64Builder builder;
    Check(builder.AppendValues(values).ok(), "append int64 values");
    std::shared_ptr<arrow::Array> array;
    Check(builder.Finish(&array).ok(), "finish int64 array");
    return array;
}

std::shared_ptr<arrow::Array> LabelArray(const std::vector<std::string>& values) {
    arrow::StringBuilder builder;
    for (const auto& value : values) {
        Check(builder.Append(value).ok(), "append label");
    }
    std::shared_ptr<arrow::Array> array;
    Check(builder.Finish(&array).ok(), "finish label array");
    return array;
}

std::shared_ptr<arrow::Table> MakeTable(int64_t base, int64_t rows = 3) {
    std::vector<int64_t> ids;
    std::vector<int64_t> values;
    std::vector<std::string> labels;
    for (int64_t row = 0; row < rows; ++row) {
        ids.push_back(row + 1);
        values.push_back(base + row);
        labels.push_back("class_" + std::to_string(row % 2));
    }
    return arrow::Table::Make(
        arrow::schema({
            arrow::field("id", arrow::int64()),
            arrow::field("value", arrow::int64()),
            arrow::field("label", arrow::utf8()),
        }),
        {Int64Array(ids), Int64Array(values), LabelArray(labels)},
        rows);
}

std::shared_ptr<arrow::Table> MakeEmptyTable() {
    return arrow::Table::Make(
        arrow::schema({arrow::field("value", arrow::int64())}),
        {Int64Array({})},
        0);
}

std::shared_ptr<cyxwiz::ArrowDataset> MakeCandidate(
    const std::string& name,
    int64_t base,
    int64_t rows = 3) {
    return std::make_shared<cyxwiz::ArrowDataset>(MakeTable(base, rows), name);
}

std::shared_ptr<cyxwiz::SparseFeatureDataset> MakeSparseDataset(
    const std::string& name) {
    cyxwiz::SparseFeatureDataset::Contents contents;
    contents.name = name;
    contents.num_rows = 2;
    contents.num_features = 2;
    contents.row_offsets = {0, 1, 2};
    contents.column_indices = {0, 1};
    contents.values = {1.0f, 2.0f};
    contents.feature_names = {"a", "b"};
    auto result = cyxwiz::SparseFeatureDataset::Create(std::move(contents));
    Check(result.ok(), result.status().ToString());
    return result.ValueOrDie();
}

std::shared_ptr<cyxwiz::ParquetBackedDataset> MakeParquetDataset(
    RegistryFixture& fixture,
    const std::string& name) {
    const auto csv_path = fixture.TempPath(name + ".csv");
    const auto parquet_path = fixture.TempPath(name + ".parquet");
    {
        std::ofstream csv(csv_path, std::ios::binary | std::ios::trunc);
        csv << "id,value\n1,10\n2,20\n";
    }
    Check(cyxwiz::ParquetBackedDataset::ConvertCsvToParquet(
              csv_path.string(), parquet_path.string()),
          "convert Parquet-backed fixture");
    auto parquet =
        cyxwiz::ParquetBackedDataset::Open(parquet_path.string(), name);
    Check(parquet != nullptr, "open Parquet-backed fixture");
    return parquet;
}

void ExpectSourcePath(const std::string& name, const std::string& source) {
    const auto recorded =
        cyxwiz::DataRegistry::Instance().GetTabularSourcePath(name);
    const auto normalized = std::filesystem::weakly_canonical(
        std::filesystem::absolute(source)).lexically_normal().string();
    Check(recorded.has_value() && *recorded == normalized,
          "source path should be recorded by dataset name for " + name);
}

void ExpectNoTabular(const std::string& name) {
    auto& registry = cyxwiz::DataRegistry::Instance();
    Check(!registry.IsArrowDataset(name) &&
              !registry.IsParquetBackedDataset(name) &&
              !registry.GetTabularSourcePath(name).has_value(),
          "tabular slot should remain absent for " + name);
}

void TestAbsentSuccessAndDefaultInvalid() {
    RegistryFixture fixture;
    auto& registry = cyxwiz::DataRegistry::Instance();
    std::string error;
    const auto source = fixture.TempPath("source_a.csv").string();

    cyxwiz::DataRegistry::TabularPublicationToken invalid;
    Check(!registry.TryPublishArrowTable(
              invalid, MakeCandidate("default_invalid", 10), source, error),
          "default publication token must be invalid");
    ExpectNoTabular("default_invalid");

    auto token = registry.CaptureTabularPublication("absent_success");
    Check(registry.TryPublishArrowTable(
              token, MakeCandidate("absent_success", 20), source, error),
          "absent captured slot should publish a matching Arrow candidate: " + error);
    Check(registry.IsArrowDataset("absent_success"),
          "successful publication should register Arrow backing");
    ExpectSourcePath("absent_success", source);
    Check(!registry.FindTabularDatasetBySourcePath(source).has_value(),
          "published source must not populate filename-only reverse lookup");
}

void TestInvalidInputsRetainPriorState() {
    RegistryFixture fixture;
    auto& registry = cyxwiz::DataRegistry::Instance();
    std::string error;
    Check(registry.TryPublishArrowTable(
              registry.CaptureTabularPublication("stable"),
              MakeCandidate("stable", 30), fixture.TempPath("old.csv").string(), error),
          "initial stable publish");
    const auto prior = registry.GetArrowDataset("stable");
    const auto old_source = registry.GetTabularSourcePath("stable").value_or("");

    auto TryAndCheckRejected = [&](const std::string& label,
                                   cyxwiz::DataRegistry::TabularPublicationToken token,
                                   std::shared_ptr<cyxwiz::ArrowDataset> candidate,
                                   const std::string& source) {
        error.clear();
        Check(!registry.TryPublishArrowTable(token, std::move(candidate), source, error),
              label + " should reject");
        Check(!error.empty(), label + " should explain rejection");
        Check(registry.GetArrowDataset("stable") == prior,
              label + " must retain prior Arrow backing");
        ExpectSourcePath("stable", old_source);
    };

    TryAndCheckRejected("null candidate",
                        registry.CaptureTabularPublication("stable"),
                        nullptr,
                        "new.csv");
    TryAndCheckRejected("null table",
                        registry.CaptureTabularPublication("stable"),
                        std::make_shared<cyxwiz::ArrowDataset>(nullptr, "stable"),
                        "new.csv");
    auto malformed = arrow::Table::Make(
        arrow::schema({arrow::field("value", arrow::int64())}),
        {Int64Array({1})}, 3);
    TryAndCheckRejected("malformed table",
                        registry.CaptureTabularPublication("stable"),
                        std::make_shared<cyxwiz::ArrowDataset>(malformed, "stable"),
                        "new.csv");
    TryAndCheckRejected("empty table",
                        registry.CaptureTabularPublication("stable"),
                        std::make_shared<cyxwiz::ArrowDataset>(
                            MakeEmptyTable(), "stable"),
                        "new.csv");
    TryAndCheckRejected("mismatched candidate name",
                        registry.CaptureTabularPublication("stable"),
                        MakeCandidate("different", 40),
                        "new.csv");

    bool empty_name_rejected = false;
    try {
        (void)registry.CaptureTabularPublication("");
    } catch (const std::invalid_argument&) {
        empty_name_rejected = true;
    }
    Check(empty_name_rejected, "empty dataset name should reject");
    ExpectNoTabular("");
}

void TestSnapshotInvalidationAndABA() {
    RegistryFixture fixture;
    auto& registry = cyxwiz::DataRegistry::Instance();
    std::string error;
    const auto v1_source = fixture.TempPath("versioned_v1.csv").string();
    const auto restored_source = fixture.TempPath("versioned_restored.csv").string();

    auto token = registry.CaptureTabularPublication("versioned");
    Check(registry.TryPublishArrowTable(
              token, MakeCandidate("versioned", 70), v1_source, error),
          "first versioned publish");
    auto stale = token;
    Check(!registry.TryPublishArrowTable(
              stale, MakeCandidate("versioned", 80),
              fixture.TempPath("versioned_v2.csv").string(), error),
          "reusing a consumed token must fail");
    ExpectSourcePath("versioned", v1_source);

    const auto old_snapshot = registry.GetArrowDataset("versioned");
    auto restore_token = registry.CaptureTabularPublication("versioned");
    registry.RestoreTabularDataset("versioned", old_snapshot, nullptr, restored_source);
    Check(!registry.TryPublishArrowTable(
              restore_token, MakeCandidate("versioned", 90),
              fixture.TempPath("after_restore.csv").string(), error),
          "same-pointer RestoreTabularDataset ABA must invalidate token");
    Check(registry.GetArrowDataset("versioned") == old_snapshot,
          "ABA failure should keep restored pointer");
    ExpectSourcePath("versioned", restored_source);

    auto absent = registry.CaptureTabularPublication("aba_absent");
    Check(registry.RegisterArrowTable(MakeTable(100), "aba_absent") != nullptr,
          "old API register for absent ABA");
    registry.UnregisterTabularDataset("aba_absent");
    Check(!registry.TryPublishArrowTable(
              absent, MakeCandidate("aba_absent", 110), "aba.csv", error),
          "absent-register-unregister ABA must invalidate token");
    ExpectNoTabular("aba_absent");

    auto empty_token = registry.CaptureTabularPublication("clear_empty");
    registry.ClearAllTabularDatasets();
    Check(!registry.TryPublishArrowTable(
              empty_token, MakeCandidate("clear_empty", 120), "clear.csv", error),
          "clearing registry must invalidate an empty-slot token");
    ExpectNoTabular("clear_empty");
}

void TestUnrelatedNameDoesNotInvalidate() {
    RegistryFixture fixture;
    auto& registry = cyxwiz::DataRegistry::Instance();
    std::string error;
    auto token = registry.CaptureTabularPublication("target");
    Check(registry.RegisterArrowTable(MakeTable(130), "other") != nullptr,
          "old API mutation of other name");
    cyxwiz::DataRegistry::TextDatasetEntry other_text;
    other_text.source_path = "text.csv";
    registry.RegisterTextDataset("other_text", other_text);
    const auto target_source = fixture.TempPath("target.csv").string();
    Check(registry.TryPublishArrowTable(
              token, MakeCandidate("target", 140), target_source, error),
          "other-name mutations must not invalidate captured slot: " + error);
    ExpectSourcePath("target", target_source);
}

void TestConcurrentSameToken() {
    RegistryFixture fixture;
    auto& registry = cyxwiz::DataRegistry::Instance();
    std::atomic<int> callbacks{0};
    registry.SetOnDatasetLoaded(
        [&](const std::string& name, const cyxwiz::DatasetInfo&) {
            if (name == "race") callbacks.fetch_add(1);
        });

    auto token = registry.CaptureTabularPublication("race");
    std::atomic<bool> go{false};
    std::atomic<int> successes{0};
    std::string errors[2];
    auto run = [&](int index) {
        while (!go.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        if (registry.TryPublishArrowTable(
                token,
                MakeCandidate("race", 200 + index),
                "race_" + std::to_string(index) + ".csv",
                errors[index])) {
            successes.fetch_add(1);
        }
    };

    std::thread a(run, 0);
    std::thread b(run, 1);
    go.store(true, std::memory_order_release);
    a.join();
    b.join();
    registry.SetOnDatasetLoaded({});

    Check(successes.load() == 1, "same-token race should have exactly one winner");
    Check(callbacks.load() == 1,
          "successful concurrent publication should invoke callback once");
    Check(registry.IsArrowDataset("race"), "race winner should publish Arrow");
}

void TestSameSourcePathIsSelectionAware() {
    RegistryFixture fixture;
    auto& registry = cyxwiz::DataRegistry::Instance();
    std::string error;
    const auto shared_source = fixture.TempPath("shared_source.csv").string();
    Check(registry.TryPublishArrowTable(
              registry.CaptureTabularPublication("source_a"),
              MakeCandidate("source_a", 300),
              shared_source,
              error),
          "publish first shared-source dataset");
    Check(registry.TryPublishArrowTable(
              registry.CaptureTabularPublication("source_b"),
              MakeCandidate("source_b", 310),
              shared_source,
              error),
          "publish second shared-source dataset");

    ExpectSourcePath("source_a", shared_source);
    ExpectSourcePath("source_b", shared_source);
    Check(!registry.FindTabularDatasetBySourcePath(shared_source).has_value(),
          "shared source path must not collapse selections through reverse lookup");
    Check(registry.IsArrowDataset("source_a") && registry.IsArrowDataset("source_b"),
          "same source path should retain both named selections");
}

void TestLegacySourceLookupOwnership() {
    RegistryFixture fixture;
    auto& registry = cyxwiz::DataRegistry::Instance();
    const auto source = fixture.TempPath("shared.h5").string();
    const auto other_source = fixture.TempPath("other.h5").string();
    registry.RestoreTabularDataset("legacy", MakeCandidate("legacy", 350), nullptr, source);
    auto legacy_token = registry.CaptureTabularPublication("legacy");
    std::string error;
    Check(registry.TryPublishArrowTable(
              registry.CaptureTabularPublication("selection"),
              MakeCandidate("selection", 360), source, error),
          "publish selection beside legacy source lookup");
    registry.UnregisterTabularDataset("selection");
    Check(registry.FindTabularDatasetBySourcePath(source) == "legacy",
          "forgetting selection preserves another name's reverse lookup");
    Check(registry.TryPublishArrowTable(
              registry.CaptureTabularPublication("selection"),
              MakeCandidate("selection", 370), source, error),
          "republish selection beside legacy source lookup");
    registry.RestoreTabularDataset("selection", MakeCandidate("selection", 380), nullptr, other_source);
    Check(registry.FindTabularDatasetBySourcePath(source) == "legacy",
          "moving selection preserves another name's reverse lookup");
    ExpectSourcePath("legacy", source);
    registry.RestoreTabularDataset("new_owner", MakeCandidate("new_owner", 390), nullptr, source);
    Check(!registry.TryPublishArrowTable(
              legacy_token, MakeCandidate("legacy", 395), source, error),
          "legacy source association reassignment invalidates its prior owner's token");
    Check(registry.FindTabularDatasetBySourcePath(source) == "new_owner" &&
              !registry.GetTabularSourcePath("legacy"),
          "legacy reverse reassignment retains the new owner's association");
}

void TestSourcePathUpdatesAndClear() {
    RegistryFixture fixture;
    auto& registry = cyxwiz::DataRegistry::Instance();
    std::string error;
    const auto old_source = fixture.TempPath("moving_old.csv").string();
    const auto new_source = fixture.TempPath("moving_new.csv").string();
    Check(registry.TryPublishArrowTable(
              registry.CaptureTabularPublication("moving"),
              MakeCandidate("moving", 400),
              old_source,
              error),
          "publish moving old source");
    Check(registry.TryPublishArrowTable(
              registry.CaptureTabularPublication("moving"),
              MakeCandidate("moving", 410),
              new_source,
              error),
          "publish moving new source");
    ExpectSourcePath("moving", new_source);
    Check(registry.TryPublishArrowTable(
              registry.CaptureTabularPublication("moving"),
              MakeCandidate("moving", 420),
              "",
              error),
          "publishing with an empty source path should clear provenance: " + error);
    Check(registry.IsArrowDataset("moving") &&
              !registry.GetTabularSourcePath("moving").has_value(),
          "empty source path should clear source path without dropping backing");
    registry.UnregisterTabularDataset("moving");
    Check(!registry.GetTabularSourcePath("moving").has_value(),
          "unregister should clear source path by name");
}

void TestPriorParquetReplacement() {
    RegistryFixture fixture;
    auto& registry = cyxwiz::DataRegistry::Instance();
    auto parquet = MakeParquetDataset(fixture, "prior_pq");
    registry.RegisterParquetBacked("prior_pq", parquet);
    Check(registry.IsParquetBackedDataset("prior_pq"),
          "prior Parquet backing should be registered");

    std::string error;
    Check(registry.TryPublishArrowTable(
              registry.CaptureTabularPublication("prior_pq"),
              MakeCandidate("prior_pq", 500),
              fixture.TempPath("arrow_source.csv").string(),
              error),
          "publication should replace prior Parquet backing: " + error);
    Check(registry.IsArrowDataset("prior_pq") &&
              !registry.IsParquetBackedDataset("prior_pq"),
          "Arrow publication should retire prior Parquet backing");
    Check(registry.GetTabularSourcePath("prior_pq").has_value(),
          "Arrow replacement should retain a source path");
}

void TestNamespaceCollisionsReject() {
    RegistryFixture fixture;
    auto& registry = cyxwiz::DataRegistry::Instance();
    std::string error;

    cyxwiz::DataRegistry::TextDatasetEntry text;
    text.source_path = "text.csv";
    registry.RegisterTextDataset("mixed_text", text);
    Check(!registry.TryPublishArrowTable(
              registry.CaptureTabularPublication("mixed_text"),
              MakeCandidate("mixed_text", 600),
              "mixed.csv",
              error),
          "text namespace collision must reject Arrow publication");
    Check(registry.IsTextDataset("mixed_text") &&
              !registry.IsArrowDataset("mixed_text"),
          "text collision failure should retain text entry");

    cyxwiz::DataRegistry::ImageDatasetEntry image;
    image.folder_path = "images";
    image.num_images = 2;
    registry.RegisterImageDataset("mixed_image", image);
    Check(!registry.TryPublishArrowTable(
              registry.CaptureTabularPublication("mixed_image"),
              MakeCandidate("mixed_image", 610),
              "mixed.csv",
              error),
          "image namespace collision must reject Arrow publication");
    Check(registry.IsImageDataset("mixed_image") &&
              !registry.IsArrowDataset("mixed_image"),
          "image collision failure should retain image entry");

    cyxwiz::DataRegistry::AudioDatasetEntry audio;
    audio.folder_path = "audio";
    audio.num_samples = 2;
    registry.RegisterAudioDataset("mixed_audio", audio);
    Check(!registry.TryPublishArrowTable(
              registry.CaptureTabularPublication("mixed_audio"),
              MakeCandidate("mixed_audio", 620),
              "mixed.csv",
              error),
          "audio namespace collision must reject Arrow publication");
    Check(registry.IsAudioDataset("mixed_audio") &&
              !registry.IsArrowDataset("mixed_audio"),
          "audio collision failure should retain audio entry");

    Check(registry.RegisterSparseFeatureDataset(MakeSparseDataset("mixed_sparse")),
          "sparse collision fixture should register");
    Check(!registry.TryPublishArrowTable(
              registry.CaptureTabularPublication("mixed_sparse"),
              MakeCandidate("mixed_sparse", 630),
              "mixed.csv",
              error),
          "sparse namespace collision must reject Arrow publication");
    Check(registry.IsSparseFeatureDataset("mixed_sparse") &&
              !registry.IsArrowDataset("mixed_sparse"),
          "sparse collision failure should retain sparse entry");

    const auto mixed_arrow = registry.RegisterArrowTable(MakeTable(640), "mixed_text");
    Check(!registry.TryPublishArrowTable(
              registry.CaptureTabularPublication("mixed_text"),
              MakeCandidate("mixed_text", 650),
              "mixed.csv",
              error),
          "mixed Arrow/text namespace collision must reject publication");
    Check(registry.IsTextDataset("mixed_text") &&
              registry.GetArrowDataset("mixed_text") == mixed_arrow,
          "mixed collision must retain both prior entries");
}

void TestCallbacksOutsideMutexAndThrowingCallbacks() {
    RegistryFixture fixture;
    auto& registry = cyxwiz::DataRegistry::Instance();
    std::atomic<int> callbacks{0};
    std::atomic<bool> queried{false};
    std::atomic<bool> recaptured{false};
    registry.SetOnDatasetLoaded(
        [&](const std::string& name, const cyxwiz::DatasetInfo&) {
            if (name != "callback_query") return;
            callbacks.fetch_add(1);
            queried.store(registry.IsArrowDataset(name), std::memory_order_release);
            auto token = registry.CaptureTabularPublication(name);
            (void)token;
            recaptured.store(true, std::memory_order_release);
        });

    std::string error;
    const auto callback_source = fixture.TempPath("callback.csv").string();
    Check(registry.TryPublishArrowTable(
              registry.CaptureTabularPublication("callback_query"),
              MakeCandidate("callback_query", 710),
              callback_source,
              error),
          "callback query publication should succeed: " + error);
    registry.SetOnDatasetLoaded({});
    Check(callbacks.load() == 1 && queried.load() && recaptured.load(),
          "callback should run outside mutex and be able to query/recapture");
    ExpectSourcePath("callback_query", callback_source);

    registry.SetOnDatasetLoaded(
        [](const std::string&, const cyxwiz::DatasetInfo&) {
            throw std::runtime_error("injected callback failure");
        });
    error.clear();
    const auto throwing_source = fixture.TempPath("throwing.csv").string();
    Check(registry.TryPublishArrowTable(
              registry.CaptureTabularPublication("throwing_callback"),
              MakeCandidate("throwing_callback", 720),
              throwing_source,
              error),
          "throwing callback must not turn a committed publish into failure");
    registry.SetOnDatasetLoaded({});
    Check(registry.IsArrowDataset("throwing_callback"),
          "throwing callback publication should remain committed");
    ExpectSourcePath("throwing_callback", throwing_source);
}

} // namespace

int main() {
    TestAbsentSuccessAndDefaultInvalid();
    TestInvalidInputsRetainPriorState();
    TestSnapshotInvalidationAndABA();
    TestUnrelatedNameDoesNotInvalidate();
    TestConcurrentSameToken();
    TestSameSourcePathIsSelectionAware();
    TestLegacySourceLookupOwnership();
    TestSourcePathUpdatesAndClear();
    TestPriorParquetReplacement();
    TestNamespaceCollisionsReject();
    TestCallbacksOutsideMutexAndThrowingCallbacks();
    checks += RunStagedPublicationTests();

    std::cout << "Tabular publication tests passed: " << checks.load()
              << " checks\n";
    return 0;
}
