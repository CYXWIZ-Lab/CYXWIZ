#include "../src/core/arrow_dataset.h"
#include "../src/core/async_task_manager.h"
#include "../src/core/hdf5_source_identity.h"
#include "../src/gui/hdf5_loaded_source_verifier.h"

#include <arrow/api.h>
#include <arrow/util/key_value_metadata.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace cyxwiz;
namespace fs = std::filesystem;
int checks = 0;
void Check(bool value, const std::string& message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}

struct Workspace {
    fs::path path = fs::temp_directory_path() / ("cyxwiz_hdf5_identity_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Workspace() { Check(fs::create_directory(path), "create identity workspace"); }
    ~Workspace() { std::error_code error; fs::remove_all(path, error); }
};

// Identity verification deliberately uses only a file stamp and Arrow metadata;
// real HDF5 preparation is covered by test_hdf5_prepared_source.
Hdf5RegisteredSourceRequest Snapshot(const fs::path& path) {
    if (!fs::exists(path)) { std::ofstream output(path, std::ios::binary); output << "stamp fixture"; }
    std::string error;
    const auto stamp = ReadHdf5SourceStamp(path.string(), error);
    Check(stamp.has_value(), "read fixture stamp: " + error);
    Hdf5RegisteredSourceRequest request;
    request.dataset_name = "verified";
    request.resolved_path = path.string();
    request.registered_source_path = stamp->canonical_path;
    request.settings.selection.label_path = "/labels";
    auto metadata = arrow::key_value_metadata(
        {"hdf5.source_path", "hdf5.source_size", "hdf5.source_modified", "hdf5.import_mode",
         "hdf5.data_path", "hdf5.label_path", "hdf5.numeric_policy", "hdf5.max_materialized_bytes", "label_column"},
        {stamp->canonical_path, std::to_string(stamp->size), std::to_string(stamp->modified), "numeric_table",
         "/data", "/labels", "preserve", std::to_string(request.settings.max_materialized_bytes), "label"});
    arrow::Int64Builder builder;
    Check(builder.AppendValues({1, 2, 3}).ok(), "append fixture values");
    auto values = builder.Finish().ValueOrDie();
    auto table = arrow::Table::Make(arrow::schema({arrow::field("value", arrow::int64()),
        arrow::field("label", arrow::int64())}, metadata), {values, values});
    request.dataset = std::make_shared<ArrowDataset>(table, request.dataset_name);
    return request;
}

void ChangeMetadata(Hdf5RegisteredSourceRequest& request, const std::string& key,
                    const std::string& value, bool erase = false, bool duplicate = false) {
    const auto table = request.dataset->GetArrowTable();
    const auto metadata = table->schema()->metadata();
    std::vector<std::string> keys, values;
    for (int64_t i = 0; i < metadata->size(); ++i) {
        if (erase && metadata->key(i) == key) continue;
        keys.push_back(metadata->key(i));
        values.push_back(metadata->key(i) == key ? value : metadata->value(i));
    }
    if (duplicate) { keys.push_back(key); values.push_back(value); }
    request.dataset = std::make_shared<ArrowDataset>(
        table->ReplaceSchemaMetadata(arrow::key_value_metadata(keys, values)), request.dataset_name);
}

void Refused(const Hdf5RegisteredSourceResult& result, Hdf5TableStatus status = Hdf5TableStatus::ReadFailed) {
    Check(result.status == status, "expected refusal: " + result.error);
    Check(!result.error.empty(), "refusal explains cause");
    Check(result.rows == 0 && result.columns == 0 && result.bytes == 0, "failure exposes no loaded-state counters");
}

void TestMetadata(const fs::path& path) {
    auto request = Snapshot(path);
    const auto accepted = VerifyHdf5RegisteredSource(request);
    Check(accepted.status == Hdf5TableStatus::Ok && accepted.rows == 3 && accepted.columns == 2 && accepted.bytes > 0,
          "matching provenance restores counts");
    for (const auto* key : {"hdf5.source_path", "hdf5.source_size", "hdf5.source_modified", "hdf5.import_mode",
                           "hdf5.data_path", "hdf5.label_path", "hdf5.numeric_policy", "hdf5.max_materialized_bytes", "label_column"}) {
        auto invalid = request;
        ChangeMetadata(invalid, key, "wrong");
        Refused(VerifyHdf5RegisteredSource(invalid));
        invalid = request;
        ChangeMetadata(invalid, key, "", true);
        Refused(VerifyHdf5RegisteredSource(invalid));
    }
    auto invalid = request;
    const auto table = request.dataset->GetArrowTable();
    invalid.dataset = std::make_shared<ArrowDataset>(table->ReplaceSchemaMetadata(nullptr), request.dataset_name);
    Refused(VerifyHdf5RegisteredSource(invalid));
    invalid.dataset = std::make_shared<ArrowDataset>(nullptr, request.dataset_name);
    Refused(VerifyHdf5RegisteredSource(invalid));
    invalid.dataset = std::make_shared<ArrowDataset>(table->RemoveColumn(1).ValueOrDie(), request.dataset_name);
    Refused(VerifyHdf5RegisteredSource(invalid));
    invalid.settings.selection.label_path.clear();
    ChangeMetadata(invalid, "hdf5.label_path", "");
    ChangeMetadata(invalid, "label_column", "");
    Check(VerifyHdf5RegisteredSource(invalid).status == Hdf5TableStatus::Ok,
          "unlabeled registered HDF5 source verifies");
    invalid = request;
    ChangeMetadata(invalid, "hdf5.data_path", "/data", false, true);
    Refused(VerifyHdf5RegisteredSource(invalid));
    invalid = request;
    invalid.dataset.reset();
    Refused(VerifyHdf5RegisteredSource(invalid));
    invalid = request;
    invalid.dataset_name = "other";
    Refused(VerifyHdf5RegisteredSource(invalid));
    invalid = request;
    invalid.registered_source_path.clear();
    Refused(VerifyHdf5RegisteredSource(invalid));
    invalid = request;
    invalid.settings.selection.data_path = "/other";
    Refused(VerifyHdf5RegisteredSource(invalid));
    invalid = request;
    invalid.settings.selection.label_path.clear();
    Refused(VerifyHdf5RegisteredSource(invalid));
    invalid = request;
    invalid.settings.numeric_policy = Hdf5NumericPolicy::Float64;
    Refused(VerifyHdf5RegisteredSource(invalid));
    invalid = request;
    invalid.settings.max_materialized_bytes /= 2;
    Refused(VerifyHdf5RegisteredSource(invalid));
    invalid.settings.max_materialized_bytes = 0;
    Refused(VerifyHdf5RegisteredSource(invalid), Hdf5TableStatus::InvalidSelection);
    Refused(VerifyHdf5RegisteredSource(request, [] { return true; }), Hdf5TableStatus::Cancelled);
    int calls = 0;
    Check(VerifyHdf5RegisteredSource(request, [&] { ++calls; return false; }).status == Hdf5TableStatus::Ok,
          "count verification cancellation boundaries");
    int current = 0;
    Refused(VerifyHdf5RegisteredSource(request, [&] { return ++current == calls - 1; }), Hdf5TableStatus::Cancelled);
    current = 0;
    const auto original_time = fs::last_write_time(path);
    const auto changed = VerifyHdf5RegisteredSource(request, [&] {
        if (++current == calls - 1) fs::last_write_time(path, original_time + std::chrono::seconds(2));
        return false;
    });
    Refused(changed);
    Check(changed.error.find("changed during") != std::string::npos, "final stamp catches mid-verification mutation");
    Refused(VerifyHdf5RegisteredSource(request));
    fs::last_write_time(path, original_time);
    { std::ofstream output(path, std::ios::app); output << "more"; }
    Refused(VerifyHdf5RegisteredSource(request));
    fs::remove(path);
    Refused(VerifyHdf5RegisteredSource(request), Hdf5TableStatus::InvalidFile);
}

struct ManagerFixture {
    ManagerFixture() {
        AsyncTaskManager::Instance().Shutdown(std::chrono::seconds(5));
        AsyncTaskManager::Instance().Initialize(1);
    }
    ~ManagerFixture() { AsyncTaskManager::Instance().Shutdown(std::chrono::seconds(5)); }
};

template <typename Predicate>
void Wait(Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!predicate() && std::chrono::steady_clock::now() < deadline) {
        AsyncTaskManager::Instance().ProcessCompletedCallbacks();
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    Check(predicate(), "async verification finishes within bounded wait");
}

struct Blocker {
    std::shared_ptr<std::promise<void>> release = std::make_shared<std::promise<void>>();
    Blocker() {
        auto entered = std::make_shared<std::promise<void>>();
        auto ready = entered->get_future();
        auto wait = release->get_future().share();
        AsyncTaskManager::Instance().Submit(std::make_shared<LambdaTask>("Block verification worker",
            [entered, wait](LambdaTask& task) { entered->set_value(); wait.wait(); task.MarkCompleted(); }));
        Check(ready.wait_for(std::chrono::seconds(5)) == std::future_status::ready, "blocker starts");
    }
    ~Blocker() { release->set_value(); }
};

void TestController(const fs::path& path, const fs::path& other_path) {
    ManagerFixture manager;
    auto request = Snapshot(path);
    gui::Hdf5LoadedSourceVerifier verifier;
    verifier.Start(request);
    Wait([&] { verifier.Poll(request); return !verifier.Busy(); });
    Check(verifier.Result().status == Hdf5TableStatus::Ok && verifier.Result().rows == 3,
          "controller delivers matching result");
    Check(!verifier.Poll(request), "result is delivered only once");

    for (int change = 0; change < 6; ++change) {
        verifier.Start(request);
        Wait([] { return AsyncTaskManager::Instance().GetActiveTaskCount() == 0; });
        auto current = request;
        if (change == 0) current.dataset = Snapshot(path).dataset;
        if (change == 1) current.settings.selection.data_path = "/other";
        if (change == 2) current.settings.numeric_policy = Hdf5NumericPolicy::Float64;
        if (change == 3) current.settings.max_materialized_bytes /= 2;
        if (change == 4) current.registered_source_path = "other";
        if (change == 5) current.resolved_path = other_path.string();
        Check(verifier.Poll(current), "changed binding retires completed verification");
        Refused(verifier.Result());
    }

    auto owner = std::make_shared<int>(1);
    verifier.Start(request, owner);
    Wait([] { return AsyncTaskManager::Instance().GetActiveTaskCount() == 0; });
    owner.reset();
    Check(!verifier.OwnerAlive(), "expired project owner is visible before node access");
    Check(verifier.Poll(request), "expired owner retires completed result");
    Refused(verifier.Result());

    std::weak_ptr<const void> expired;
    { auto temporary = std::make_shared<int>(2); expired = temporary; }
    verifier.Start(request, expired);
    Check(!verifier.Busy(), "expired owner cannot start verification");
    Refused(verifier.Result(), Hdf5TableStatus::Cancelled);

    {
        Blocker blocker;
        auto queued = Snapshot(path);
        std::weak_ptr<ArrowDataset> weak = queued.dataset;
        verifier.Start(std::move(queued));
        verifier.Cancel();
        Check(!verifier.Busy() && weak.expired(), "queued cancellation immediately releases the private dataset reference");
        queued = Snapshot(path);
        weak = queued.dataset;
        {
            gui::Hdf5LoadedSourceVerifier closed;
            closed.Start(std::move(queued));
        }
        Check(weak.expired(), "dialog destruction releases queued dataset despite task history");
        queued = Snapshot(path);
        weak = queued.dataset;
        verifier.Start(std::move(queued));
        request = Snapshot(other_path);
        verifier.Start(request);
        Check(weak.expired(), "new generation drops old queued table");
    }
    Wait([&] { verifier.Poll(request); return !verifier.Busy(); });
    Check(verifier.Result().status == Hdf5TableStatus::Ok, "late old generation cannot replace the new result");
    Wait([] { return AsyncTaskManager::Instance().GetActiveTaskCount() == 0; });
    std::weak_ptr<ArrowDataset> released = request.dataset;
    request.dataset.reset();
    Check(released.expired(), "completed task history and controller do not retain the verified table");
}
} // namespace

int main() try {
    Workspace workspace;
    TestMetadata(workspace.path / "source.h5");
    TestController(workspace.path / "source.h5", workspace.path / "other.h5");
    std::cout << "HDF5 loaded source verifier: " << checks << " checks passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
