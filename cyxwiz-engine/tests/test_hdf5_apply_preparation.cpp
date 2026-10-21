#include <iostream>

#ifdef CYXWIZ_HAS_HDF5
#include "../src/core/hdf5_apply_preparation.h"
#include "../src/core/arrow_dataset.h"
#include "../src/core/async_task_manager.h"
#include "../src/core/dataset_audit.h"

#include <arrow/api.h>
#include <highfive/highfive.hpp>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <thread>

namespace {
using namespace cyxwiz;
namespace fs = std::filesystem;
int checks = 0;
void Check(bool value, const std::string& message) {
    ++checks;
    if (!value) throw std::runtime_error("Apply preparation: " + message);
}

struct Fixture {
    const std::string name = "cyxwiz_hdf5_apply_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const fs::path directory = fs::temp_directory_path() / name;
    const fs::path path = directory / "source.h5";
    std::shared_ptr<ArrowDataset> baseline;
    Fixture() {
        Check(fs::create_directory(directory), "create fixture");
        {
            HighFive::File file(path.string(), HighFive::File::Overwrite);
            const std::vector<uint64_t> values{9007199254740993ULL, 4, 7, 12};
            const std::vector<int32_t> labels{0, 1, 0, 1}, single_class{1, 1, 1, 1};
            file.createDataSet<uint64_t>("/data", HighFive::DataSpace::From(values)).write(values);
            file.createDataSet<int32_t>("/labels", HighFive::DataSpace::From(labels)).write(labels);
            file.createDataSet<int32_t>("/single_class", HighFive::DataSpace::From(single_class)).write(single_class);
            // A constant feature is a degenerate column: the dataset audit refuses it.
            const std::vector<uint64_t> constant(4, 7);
            file.createDataSet<uint64_t>("/constant", HighFive::DataSpace::From(constant)).write(constant);
        }
        arrow::Int64Builder builder;
        Check(builder.AppendValues({-5, 6}).ok(), "make baseline");
        auto table = arrow::Table::Make(arrow::schema({arrow::field("old", arrow::int64())}),
                                       {builder.Finish().ValueOrDie()});
        baseline = DataRegistry::Instance().RegisterArrowTable(table, name);
        Check(baseline != nullptr, "register baseline");
    }
    ~Fixture() {
        DataRegistry::Instance().UnregisterTabularDataset(name);
        std::error_code error;
        fs::remove_all(directory, error);
    }
    Hdf5ApplyPreparationRequest Request() const {
        Hdf5ApplyPreparationRequest request;
        request.dataset_name = name;
        request.source.path = path.string();
        request.source.settings.selection = {"/data", "/labels"};
        request.publication = DataRegistry::Instance().CaptureTabularPublication(name);
        request.parameters = {{"file_path", "data/source.h5"}, {"data_loaded", "false"},
                              {"dataset_role", "test"}, {"unrelated", "kept"}};
        return request;
    }
    void Unchanged() const {
        Check(DataRegistry::Instance().GetArrowDataset(name) == baseline,
              "private preparation never changes registration");
        Check(!DataRegistry::Instance().GetTabularSourcePath(name), "source association unchanged");
    }
};

std::shared_ptr<Hdf5ApplyPreparationState> Run(Hdf5ApplyPreparationRequest request,
                                             Hdf5TableStatus status) {
    auto state = std::make_shared<Hdf5ApplyPreparationState>();
    auto task = MakeHdf5ApplyPreparationTask(std::move(request), state);
    task->Execute();
    Check(state->done.load(std::memory_order_acquire), "worker publishes done");
    Check(state->result.status == status, "expected status " + std::to_string(static_cast<int>(status)) +
          ", got " + std::to_string(static_cast<int>(state->result.status)) + ": " + state->result.error);
    return state;
}

void NoPayload(const Hdf5ApplyPreparationResult& result) {
    Check(!result.publication && !result.source && result.parameters.empty(),
          "failure exposes no publication, source receipt, or parameter commit");
    Check(!result.error.empty(), "failure explains its cause");
}

void TestPreparation() {
    Fixture fixture;
    auto request = fixture.Request();
    const auto original_parameters = request.parameters;
    auto ready = Run(request, Hdf5TableStatus::Ok);
    const auto& result = ready->result;
    Check(result.publication && result.source && result.audit && !result.audit->HasErrors(),
          "success contains a staged handoff, verified source, and completed audit");
    Check(request.parameters == original_parameters, "caller parameters unchanged");
    Check(result.parameters.at("loaded_rows") == "4" && result.parameters.at("loaded_cols") == "2" &&
          result.parameters.at("data_loaded") == "true" && result.parameters.at("dataset_name") == fixture.name,
          "private parameters contain matching loaded-state fields");
    Check(result.parameters.at("label_column") == "label" && result.parameters.at("unrelated") == "kept" &&
          result.parameters.at("file_path") == "data/source.h5" && !result.parameters.count("dataset_role"),
          "preserve common settings, set label contract, and remove obsolete role");
    const auto settings = ReadHdf5InputSettings(result.parameters);
    Check(settings.ok && settings.settings.selection.data_path == "/data" &&
          settings.settings.selection.label_path == "/labels", "canonical settings match prepared selection");
    fixture.Unchanged();

    auto invalid = request;
    invalid.source.settings.selection.data_path = "/missing";
    NoPayload(Run(invalid, Hdf5TableStatus::MissingDataset)->result);
    invalid = request;
    invalid.source.settings.selection.label_path = "/single_class";
    Check(Run(invalid, Hdf5TableStatus::Ok)->result.audit->HasWarnings(), "single-class labels warn but prepare");
    fixture.Unchanged();
    invalid = request;
    invalid.source.settings.selection.data_path = "/constant";
    auto refused = Run(invalid, Hdf5TableStatus::ReadFailed);
    NoPayload(refused->result);
    Check(refused->result.audit && refused->result.audit->HasErrors(), "audit refusal retains diagnostics only");
    invalid = request;
    invalid.dataset_name.clear();
    NoPayload(Run(invalid, Hdf5TableStatus::InvalidSelection)->result);
    invalid = request;
    invalid.publication = {};
    NoPayload(Run(invalid, Hdf5TableStatus::ReadFailed)->result);
    invalid = request;
    invalid.source.cancel_requested = [] { return true; };
    NoPayload(Run(invalid, Hdf5TableStatus::Cancelled)->result);

    for (const float boundary : {0.1f, 0.92f, 0.98f}) {
        auto state = std::make_shared<Hdf5ApplyPreparationState>();
        auto task = MakeHdf5ApplyPreparationTask(request, state);
        task->SetProgressCallback([weak = std::weak_ptr<AsyncTask>(task), boundary](float progress, const std::string&) {
            if (progress >= boundary) if (auto task = weak.lock()) task->RequestCancel();
        });
        task->Execute();
        Check(state->done.load(std::memory_order_acquire) && state->result.status == Hdf5TableStatus::Cancelled,
              "cancellation at read/stage/final-stamp boundary");
        NoPayload(state->result);
    }
    auto changed = std::make_shared<Hdf5ApplyPreparationState>();
    const auto original_time = fs::last_write_time(fixture.path);
    auto task = MakeHdf5ApplyPreparationTask(request, changed);
    task->SetProgressCallback([&](float progress, const std::string&) {
        if (progress >= 0.98f) fs::last_write_time(fixture.path, original_time + std::chrono::seconds(2));
    });
    task->Execute();
    Check(changed->done.load(std::memory_order_acquire) && changed->result.source_changed,
          "final source check covers staging and map allocation");
    NoPayload(changed->result);
    fs::last_write_time(fixture.path, original_time);
    auto allocation = std::make_shared<Hdf5ApplyPreparationState>();
    task = MakeHdf5ApplyPreparationTask(request, allocation);
    task->SetProgressCallback([](float progress, const std::string&) {
        if (progress >= 0.92f) throw std::bad_alloc{};
    });
    task->Execute();
    Check(allocation->result.status == Hdf5TableStatus::ResourceLimit, "staging allocation failure classified");
    NoPayload(allocation->result);
    fixture.Unchanged();

    // The tested worker produces the exact handoff used by the registry commit.
    auto parameters = original_parameters;
    std::string error;
    Check(DataRegistry::Instance().TryPublishPreparedArrowTable(*ready->result.publication, error),
          "commit prepared publication: " + error);
    parameters.swap(ready->result.parameters);
    ready->result.publication->Notify();
    auto registered = DataRegistry::Instance().GetArrowDataset(fixture.name);
    Check(registered && registered != fixture.baseline && parameters.at("data_loaded") == "true",
          "explicit commit publishes both registration and prebuilt settings");
    auto values = std::static_pointer_cast<arrow::UInt64Array>(registered->GetArrowTable()->column(0)->chunk(0));
    Check(values->Value(0) == 9007199254740993ULL, "staged Apply preserves integer precision");

    auto stale = Run(request, Hdf5TableStatus::Ok);
    const auto saved_parameters = parameters;
    Check(!DataRegistry::Instance().TryPublishPreparedArrowTable(*stale->result.publication, error),
          "consumed ownership token refuses stale handoff");
    Check(parameters == saved_parameters && DataRegistry::Instance().GetArrowDataset(fixture.name) == registered,
          "refusal preserves committed node settings and registration");
}

void TestTaskOwnership() {
    Fixture fixture;
    auto request = fixture.Request();
    auto state = std::make_shared<Hdf5ApplyPreparationState>();
    auto task = MakeHdf5ApplyPreparationTask(request, state);
    task->Execute();
    Check(state->result.status == Hdf5TableStatus::Ok, "prepare lifetime fixture");
    std::weak_ptr<Hdf5ApplyPreparationState> weak = state;
    state.reset();
    Check(weak.expired(), "completed task handle cannot retain unpublished table/handoff");
    fixture.Unchanged();

    state = std::make_shared<Hdf5ApplyPreparationState>();
    bool executed = false;
    request.source.cancel_requested = [&] { executed = true; return false; };
    task = MakeHdf5ApplyPreparationTask(request, state);
    state.reset();
    task->Execute();
    Check(!executed, "missing owner storage prevents any preparation work");

    auto& manager = AsyncTaskManager::Instance();
    manager.Initialize(1);
    state = std::make_shared<Hdf5ApplyPreparationState>();
    task = MakeHdf5ApplyPreparationTask(request, state);
    task->RequestCancel();
    manager.Submit(task);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (manager.GetActiveTaskCount() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    manager.Shutdown(std::chrono::seconds(5));
    Check(task->GetState() == TaskState::Cancelled && !state->done.load() && !executed,
          "queued cancellation skips worker without exposing a handoff");
    weak = state;
    state.reset();
    Check(weak.expired(), "cancelled task history does not retain result storage");
}
} // namespace
#endif

void TestHdf5ApplyPreparation() {
#ifdef CYXWIZ_HAS_HDF5
    TestPreparation();
    TestTaskOwnership();
    std::cout << "HDF5 Apply preparation: " << checks << " checks passed\n";
#endif
}
