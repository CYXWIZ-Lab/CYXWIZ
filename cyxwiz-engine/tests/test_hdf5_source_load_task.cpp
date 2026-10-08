#include "../src/core/hdf5_source_load_task.h"
#include "../src/core/async_task_manager.h"

#ifdef CYXWIZ_HAS_HDF5
#include "../src/core/dataset_audit.h"
#include <arrow/api.h>
#include <arrow/util/key_value_metadata.h>
#include <highfive/highfive.hpp>
#endif

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using namespace cyxwiz;
int checks = 0;

void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

struct Workspace {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("cyxwiz_hdf5_source_load_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    Workspace() { Check(std::filesystem::create_directory(path), "unique workspace"); }
    ~Workspace() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};

void CheckNoPayload(const Hdf5SourceLoadResult& result, bool audit_refusal = false) {
    const auto& read = result.read;
    Check(!read.table && !result.source && read.data.path.empty() && read.data.shape.empty() &&
              read.data.source_type.empty() && !read.labels && read.estimated_materialized_bytes == 0 &&
              read.row_offset == 0 && read.column_offset == 0, "failure clears every payload field");
    Check(static_cast<bool>(result.audit) == audit_refusal, "only audit refusal retains diagnostics on failure");
}

void CheckTerminal(const std::shared_ptr<AsyncTask>& task,
                   const std::shared_ptr<Hdf5SourceLoadTaskResult>& state, Hdf5TableStatus status,
                   bool audit_refusal = false) {
    Check(state->done.load(std::memory_order_acquire), "worker publishes done");
    Check(state->result.read.status == status, "expected load status: " + state->result.read.error);
    const auto terminal = status == Hdf5TableStatus::Ok ? TaskState::Completed :
        (status == Hdf5TableStatus::Cancelled ? TaskState::Cancelled : TaskState::Failed);
    Check(task->GetState() == terminal, "task reaches matching terminal state");
    if (status != Hdf5TableStatus::Ok) {
        Check(!state->result.read.error.empty(), "failure explains its cause");
        CheckNoPayload(state->result, audit_refusal);
    }
#ifdef CYXWIZ_HAS_HDF5
    if (status == Hdf5TableStatus::Ok)
        Check(state->result.audit && !state->result.audit->cancelled && !state->result.audit->HasErrors(),
              "successful load carries a completed accepting audit");
    if (audit_refusal)
        Check(status == Hdf5TableStatus::ReadFailed && state->result.audit &&
                  !state->result.audit->cancelled && state->result.audit->HasErrors() && !state->result.source_changed,
              "audit refusal carries completed error diagnostics only");
#endif
    if (terminal == TaskState::Failed)
        Check(task->GetErrorMessage() == state->result.read.error, "task and result failure agree");
}

Hdf5SourceLoadResult Run(const Hdf5SourceLoadRequest& request, Hdf5TableStatus status,
                         bool audit_refusal = false) {
    auto state = std::make_shared<Hdf5SourceLoadTaskResult>();
    auto task = MakeHdf5SourceLoadTask(request, state);
    Check(task->IsCancellable() && !state->done.load(), "new task is cancellable and unpublished");
    task->Execute();
    CheckTerminal(task, state, status, audit_refusal);
    auto result = std::move(state->result);
    std::weak_ptr<Hdf5SourceLoadTaskResult> observer = state;
    state.reset();
    Check(observer.expired(), "completed task does not retain result storage");
    const auto terminal = task->GetState();
    task->Execute();
    Check(task->GetState() == terminal, "repeated Execute is a no-op");
    return result;
}

void TestIdentity(const Workspace& workspace) {
    const auto path = (workspace.path / "identity.txt").string();
    { std::ofstream file(path, std::ios::binary); file << "abc"; }
    std::string error = "old error";
    const auto stamp = ReadHdf5SourceStamp(path, error);
    Check(stamp && error.empty() && stamp->size == 3 &&
              stamp->canonical_path == std::filesystem::canonical(path).string(),
          "identity is filesystem-only, including without HDF5");
    Check(ReadHdf5SourceStamp((workspace.path / "." / "identity.txt").string(), error) == stamp,
          "equivalent filesystem paths share canonical identity");
    for (const auto& bad : {std::string{}, path + '\0' + "suffix", path + ".missing", workspace.path.string()}) {
        Check(!ReadHdf5SourceStamp(bad, error) && !error.empty(),
              "invalid identity fails with an error: " + bad);
    }
}

#ifdef CYXWIZ_HAS_HDF5
void CreateFixture(const std::string& path) {
    HighFive::File file(path, HighFive::File::Overwrite);
    file.createGroup("/nested");
    const std::vector<uint64_t> values{std::numeric_limits<uint64_t>::max(), 9007199254740993ULL, 7};
    file.createDataSet<uint64_t>("/nested/values", HighFive::DataSpace::From(values)).write(values);
    const std::vector<uint8_t> labels{2, 1, 0};
    file.createDataSet<uint8_t>("/labels", HighFive::DataSpace::From(labels)).write(labels);
    const std::vector<uint64_t> exact_labels{9007199254740992ULL, 9007199254740993ULL,
                                             std::numeric_limits<uint64_t>::max()};
    file.createDataSet<uint64_t>("/exact_labels", HighFive::DataSpace::From(exact_labels)).write(exact_labels);
    const std::vector<std::vector<int16_t>> matrix{{-3, 4}, {5, -6}, {7, 8}};
    file.createDataSet<int16_t>("/matrix", HighFive::DataSpace::From(matrix)).write(matrix);
    const std::vector<float> floats{1.25f, -2.5f, 3.75f};
    file.createDataSet<float>("/floats", HighFive::DataSpace::From(floats)).write(floats);
    file.createDataSet<float>("/rank3", HighFive::DataSpace(std::vector<size_t>{2, 3, 4}));
    const std::vector<uint64_t> slabs(70000, 9007199254740993ULL);
    file.createDataSet<uint64_t>("/slabs", HighFive::DataSpace::From(slabs)).write(slabs);
}

void ChangeTime(const std::string& path) {
    std::error_code ec;
    const auto before = std::filesystem::last_write_time(path, ec);
    Check(!ec, "read fixture modification time");
    std::filesystem::last_write_time(path, before + std::chrono::seconds(2), ec);
    Check(!ec, "change modification time deterministically");
}

void CheckMetadata(const std::shared_ptr<arrow::Table>& table,
                   const std::string& key, const std::string& expected) {
    const auto metadata = table->schema()->metadata();
    Check(metadata != nullptr, "schema metadata present");
    const auto value = metadata->Get(key);
    Check(value.ok() && value.ValueOrDie() == expected, "metadata matches: " + key);
}

void CheckChanged(const Hdf5SourceLoadResult& result) {
    Check(result.source_changed && result.read.status == Hdf5TableStatus::ReadFailed &&
              result.read.error.find("Refresh") != std::string::npos, "source change requests refresh");
    CheckNoPayload(result);
}

void TestValuesAndOwnership(const std::string& path) {
    Hdf5SourceLoadRequest request;
    request.path = path;
    request.settings.selection = {"/nested/values", "/labels"};
    std::string error;
    const auto stamp = ReadHdf5SourceStamp(path, error);
    Check(stamp.has_value(), "fixture source stamp");
    request.expected_source = stamp;
    auto state = std::make_shared<Hdf5SourceLoadTaskResult>();
    auto task = MakeHdf5SourceLoadTask(request, state);
    request.path += ".missing";
    request.settings.selection = {"/missing", ""};
    request.settings.numeric_policy = Hdf5NumericPolicy::Float64;
    request.settings.max_materialized_bytes = 1;
    ++request.expected_source->size;
    request.cancel_requested = [] { return true; };
    task->Execute();
    CheckTerminal(task, state, Hdf5TableStatus::Ok);
    const auto& result = state->result;
    const auto table = result.read.table;
    Check(table && table->ValidateFull().ok() && table->num_rows() == 3 && table->num_columns() == 2,
          "copied request produces full valid table");
    Check(result.source == stamp && !result.source_changed && result.read.estimated_materialized_bytes > 0,
          "successful load retains verified identity and byte estimate");
    Check(result.read.data.path == "/nested/values" && result.read.data.shape == std::vector<uint64_t>{3} &&
              result.read.labels && result.read.labels->path == "/labels", "source descriptors retained");
    Check(table->field(0)->name() == "value" && table->field(0)->type()->id() == arrow::Type::UINT64 &&
              !table->field(0)->nullable() && table->field(1)->name() == "label" &&
              table->field(1)->type()->id() == arrow::Type::UINT8 && !table->field(1)->nullable(),
          "native schema and label nullability preserved");
    const auto values = std::static_pointer_cast<arrow::UInt64Array>(table->column(0)->chunk(0));
    Check(values->Value(0) == std::numeric_limits<uint64_t>::max() && values->Value(1) == 9007199254740993ULL,
          "uint64 values are exact beyond float64 precision");
    const auto labels = std::static_pointer_cast<arrow::UInt8Array>(table->column(1)->chunk(0));
    Check(labels->Value(0) == 2 && labels->Value(1) == 1 && labels->Value(2) == 0, "labels remain aligned");
    CheckMetadata(table, "hdf5.data_path", "/nested/values");
    CheckMetadata(table, "hdf5.label_path", "/labels");
    CheckMetadata(table, "hdf5.numeric_policy", "preserve");
    CheckMetadata(table, "label_column", "label");
    CheckMetadata(table, "hdf5.source_path", stamp->canonical_path);
    CheckMetadata(table, "hdf5.source_size", std::to_string(stamp->size));
    CheckMetadata(table, "hdf5.source_modified", std::to_string(stamp->modified));
    CheckMetadata(table, "hdf5.import_mode", "numeric_table");

    request = {};
    request.path = path;
    request.settings.selection = {"/matrix", "/labels"};
    auto matrix = Run(request, Hdf5TableStatus::Ok).read.table;
    Check(matrix->num_columns() == 3 && matrix->field(0)->name() == "col_0" &&
              matrix->field(1)->name() == "col_1" && matrix->field(0)->type()->id() == arrow::Type::INT16 &&
              std::static_pointer_cast<arrow::Int16Array>(matrix->column(1)->chunk(0))->Value(1) == -6,
          "rank2 signed integers retain column order and width");
    request.settings.selection = {"/floats", ""};
    const auto floats = Run(request, Hdf5TableStatus::Ok).read.table;
    Check(floats->field(0)->type()->id() == arrow::Type::FLOAT &&
              std::static_pointer_cast<arrow::FloatArray>(floats->column(0)->chunk(0))->Value(1) == -2.5f,
          "float32 remains float32");
    CheckMetadata(floats, "label_column", "");
    request.settings.selection = {"/nested/values", "/labels"};
    request.settings.numeric_policy = Hdf5NumericPolicy::Float64;
    const auto doubles = Run(request, Hdf5TableStatus::Ok).read.table;
    Check(doubles->field(0)->type()->id() == arrow::Type::DOUBLE &&
              doubles->field(1)->type()->id() == arrow::Type::DOUBLE &&
              std::static_pointer_cast<arrow::DoubleArray>(doubles->column(0)->chunk(0))->Value(1) ==
                  static_cast<double>(9007199254740993ULL), "explicit Float64 permits precision loss");
    CheckMetadata(doubles, "hdf5.numeric_policy", "float64");

    state = std::make_shared<Hdf5SourceLoadTaskResult>();
    std::weak_ptr<Hdf5SourceLoadTaskResult> observer = state;
    auto cancel_owner = std::make_shared<int>(42);
    std::weak_ptr<int> cancel_observer = cancel_owner;
    request.cancel_requested = [cancel_owner] { return *cancel_owner != 42; };
    task = MakeHdf5SourceLoadTask(request, state);
    request.cancel_requested = {};
    cancel_owner.reset();
    state.reset();
    bool observed_during_execute = false;
    task->SetProgressCallback([&](float, const std::string&) {
        const auto retained = observer.lock();
        Check(retained != nullptr && !retained->done.load(), "worker owns detached state during Execute");
        Check(!cancel_observer.expired(), "worker owns external cancel closure during Execute");
        observed_during_execute = true;
    });
    task->Execute();
    Check(observed_during_execute && task->GetState() == TaskState::Completed, "detached worker completes");
    Check(observer.expired() && cancel_observer.expired(), "task history retains neither result nor cancel closure");
    task->Execute();
    Check(task->GetState() == TaskState::Completed && observer.expired(), "detached task is one-shot");
    task->SetProgressCallback({});

    state = std::make_shared<Hdf5SourceLoadTaskResult>();
    task = MakeHdf5SourceLoadTask(request, state);
    task->Execute();
    CheckTerminal(task, state, Hdf5TableStatus::Ok);
    const auto retained_table = state->result.read.table;
    observer = state;
    state.reset();
    Check(observer.expired(), "caller table ownership does not retain result storage");
    task.reset();
    Check(retained_table && retained_table->ValidateFull().ok(), "caller-owned table outlives task and result storage");
}

bool HasAuditIssue(const Hdf5SourceLoadResult& result, DatasetAuditSeverity severity, const std::string& code) {
    if (!result.audit) return false;
    for (const auto& issue : result.audit->issues)
        if (issue.severity == severity && issue.code == code) return true;
    return false;
}

void TestAudit(const std::string& path) {
    Hdf5SourceLoadRequest request;
    request.path = path;
    request.settings.selection = {"/floats", ""};
    const auto unlabeled = Run(request, Hdf5TableStatus::Ok);
    Check(HasAuditIssue(unlabeled, DatasetAuditSeverity::Warning, "missing_label_column"),
          "unlabeled source is accepted with a warning");
    Check(unlabeled.audit->dataset_name == "HDF5 source" && unlabeled.audit->sample_count == 3 &&
              unlabeled.audit->feature_count == 1, "private candidate audit reports counts");
    request.settings.selection.label_path = "/exact_labels";
    const auto labeled = Run(request, Hdf5TableStatus::Ok);
    Check(labeled.audit->class_count == 3 &&
              !HasAuditIssue(labeled, DatasetAuditSeverity::Warning, "missing_label_column"),
          "adjacent uint64 labels above float64 precision remain distinct audit classes");
    const auto labels = std::static_pointer_cast<arrow::UInt64Array>(labeled.read.table->column(1)->chunk(0));
    Check(labels->Value(0) == 9007199254740992ULL && labels->Value(1) == 9007199254740993ULL &&
              labels->Value(2) == std::numeric_limits<uint64_t>::max(), "audit does not coerce label values");
    CheckMetadata(labeled.read.table, "label_column", "label");
    CheckMetadata(labeled.read.table, "hdf5.label_path", "/exact_labels");

    request.settings.selection = {"/slabs", ""};
    const auto refused = Run(request, Hdf5TableStatus::ReadFailed, true);
    Check(HasAuditIssue(refused, DatasetAuditSeverity::Error, "too_many_degenerate_columns") &&
              refused.read.error.find("audit") != std::string::npos, "constant source refused with audit diagnostics");

    bool auditing = false;
    int audit_polls = 0;
    request.cancel_requested = [&] { return auditing && ++audit_polls >= 3; };
    auto state = std::make_shared<Hdf5SourceLoadTaskResult>();
    auto task = MakeHdf5SourceLoadTask(request, state);
    task->SetProgressCallback([&](float, const std::string& message) {
        if (message == "Auditing columns") auditing = true;
    });
    task->Execute();
    Check(audit_polls >= 3, "cancellation reaches the exact constant-column scan");
    CheckTerminal(task, state, Hdf5TableStatus::Cancelled);
    request.cancel_requested = {};

    for (bool cancel : {false, true}) {
        state = std::make_shared<Hdf5SourceLoadTaskResult>();
        task = MakeHdf5SourceLoadTask(request, state);
        bool acted = false;
        float last_progress = 0.0f;
        task->SetProgressCallback([&](float progress, const std::string& message) {
            Check(progress >= last_progress, "audit progress remains monotonic");
            last_progress = progress;
            if (!acted && message == "Audit complete") {
                Check(progress >= 0.91f && progress <= 0.93f, "audit completes within mapped progress window");
                acted = true;
                if (cancel) task->RequestCancel();
                else ChangeTime(path);
            }
        });
        task->Execute();
        Check(acted, "reached completed audit before publication");
        CheckTerminal(task, state, cancel ? Hdf5TableStatus::Cancelled : Hdf5TableStatus::ReadFailed);
        if (!cancel) CheckChanged(state->result);
    }
}

void TestFailures(const std::string& path, const Workspace& workspace) {
    Hdf5SourceLoadRequest request;
    request.path = path;
    request.settings.selection = {"/nested/values", "/labels"};
    const auto valid = request.settings;
    for (const auto& data_path : {std::string{}, std::string("relative"), std::string("/nested/../values")}) {
        request.settings.selection.data_path = data_path;
        Run(request, Hdf5TableStatus::InvalidSelection);
    }
    request.settings = valid;
    request.settings.selection.label_path = "relative";
    Run(request, Hdf5TableStatus::InvalidSelection);
    request.settings = valid;
    request.settings.numeric_policy = static_cast<Hdf5NumericPolicy>(99);
    Run(request, Hdf5TableStatus::InvalidSelection);
    request.settings = valid;
    for (uint64_t budget : std::vector<uint64_t>{0, 256ULL * 1024 * 1024 + 1}) {
        request.settings.max_materialized_bytes = budget;
        Run(request, Hdf5TableStatus::InvalidSelection);
    }
    request.settings.max_materialized_bytes = 1;
    Run(request, Hdf5TableStatus::ResourceLimit);
    request.settings = valid;
    request.settings.selection = {"/rank3", ""};
    Run(request, Hdf5TableStatus::UnsupportedRank);
    request.settings.selection.data_path = "/missing";
    Run(request, Hdf5TableStatus::MissingDataset);
    request.settings = valid;
    request.settings.selection.label_path = "/matrix";
    Run(request, Hdf5TableStatus::UnsupportedRank);
    request.settings.selection.label_path = "/slabs";
    Run(request, Hdf5TableStatus::LabelRowMismatch);
    request.settings = valid;
    request.path = (workspace.path / "missing.h5").string();
    Run(request, Hdf5TableStatus::InvalidFile);
    request.path = path + '\0' + "suffix";
    Run(request, Hdf5TableStatus::InvalidFile);
    request.path = (workspace.path / "fake.h5").string();
    { std::ofstream fake(request.path, std::ios::binary); fake << "not HDF5"; }
    Run(request, Hdf5TableStatus::InvalidFile);
    request.path = path;
    std::string error;
    const auto stamp = ReadHdf5SourceStamp(path, error);
    Check(stamp.has_value(), "source stamp before mismatch tests");
    for (int field = 0; field < 3; ++field) {
        request.expected_source = stamp;
        if (field == 0) request.expected_source->canonical_path += ".different";
        else if (field == 1) ++request.expected_source->size;
        else ++request.expected_source->modified;
        CheckChanged(Run(request, Hdf5TableStatus::ReadFailed));
    }
    request.expected_source = stamp;
    request.path += ".missing";
    CheckChanged(Run(request, Hdf5TableStatus::ReadFailed));
    request.path = path;
    for (float threshold : {0.05f, 0.15f, 0.75f, 0.78f, 0.90f, 0.95f}) {
        request.expected_source = ReadHdf5SourceStamp(path, error);
        auto state = std::make_shared<Hdf5SourceLoadTaskResult>();
        auto task = MakeHdf5SourceLoadTask(request, state);
        bool changed = false;
        task->SetProgressCallback([&](float progress, const std::string&) {
            if (!changed && progress >= threshold) { changed = true; ChangeTime(path); }
        });
        task->Execute();
        Check(changed, "reached requested identity mutation stage");
        CheckTerminal(task, state, Hdf5TableStatus::ReadFailed);
        CheckChanged(state->result);
    }
    request.expected_source.reset();
    auto state = std::make_shared<Hdf5SourceLoadTaskResult>();
    auto task = MakeHdf5SourceLoadTask(request, state);
    bool resized = false;
    task->SetProgressCallback([&](float progress, const std::string&) {
        if (!resized && progress >= 0.75f) {
            resized = true;
            std::ofstream append(path, std::ios::binary | std::ios::app);
            append << "source changed";
            Check(static_cast<bool>(append), "append after the reader closes the file");
        }
    });
    task->Execute();
    Check(resized, "source-size change injected after materialization");
    CheckTerminal(task, state, Hdf5TableStatus::ReadFailed);
    CheckChanged(state->result);
}

void TestCancellation(const std::string& path) {
    Hdf5SourceLoadRequest request;
    request.path = path;
    request.settings.selection = {"/slabs", ""};
    auto state = std::make_shared<Hdf5SourceLoadTaskResult>();
    auto task = MakeHdf5SourceLoadTask(request, state);
    task->RequestCancel();
    Check(!state->done.load(), "cancel request does not write worker result");
    task->Execute();
    CheckTerminal(task, state, Hdf5TableStatus::Cancelled);
    request.cancel_requested = [] { return true; };
    Run(request, Hdf5TableStatus::Cancelled);
    request.cancel_requested = {};
    for (float threshold : {0.15f, 0.75f, 0.95f}) {
        state = std::make_shared<Hdf5SourceLoadTaskResult>();
        task = MakeHdf5SourceLoadTask(request, state);
        bool requested = false;
        task->SetProgressCallback([&](float progress, const std::string&) {
            if (!requested && progress >= threshold) { requested = true; task->RequestCancel(); }
        });
        task->Execute();
        Check(requested, "reached cancellation stage");
        CheckTerminal(task, state, Hdf5TableStatus::Cancelled);
    }
    bool reading = false;
    bool validating = false;
    int polls = 0;
    request.cancel_requested = [&] { return reading && ++polls >= 6; };
    state = std::make_shared<Hdf5SourceLoadTaskResult>();
    task = MakeHdf5SourceLoadTask(request, state);
    task->SetProgressCallback([&](float progress, const std::string&) {
        if (progress >= 0.15f) reading = true;
        if (progress >= 0.75f) validating = true;
    });
    task->Execute();
    CheckTerminal(task, state, Hdf5TableStatus::Cancelled);
    Check(polls >= 6 && !validating, "adapter polls cancellation during materialization");
    bool verifying = false;
    int verification_polls = 0;
    request.cancel_requested = [&] { return verifying && ++verification_polls == 2; };
    state = std::make_shared<Hdf5SourceLoadTaskResult>();
    task = MakeHdf5SourceLoadTask(request, state);
    task->SetProgressCallback([&](float progress, const std::string&) {
        if (progress >= 0.95f) verifying = true;
    });
    task->Execute();
    CheckTerminal(task, state, Hdf5TableStatus::Cancelled);
    Check(verification_polls == 2, "external cancellation honored after final identity read");
    for (int exception = 0; exception < 3; ++exception) {
        request.cancel_requested = [exception]() -> bool {
            if (exception == 0) throw std::bad_alloc{};
            if (exception == 1) throw std::runtime_error("injected failure");
            throw 7;
        };
        Run(request, exception == 0 ? Hdf5TableStatus::ResourceLimit : Hdf5TableStatus::ReadFailed);
    }
    request.cancel_requested = {};
    state = std::make_shared<Hdf5SourceLoadTaskResult>();
    task = MakeHdf5SourceLoadTask(request, state);
    task->SetProgressCallback([](float progress, const std::string&) {
        if (progress >= 0.95f) throw std::runtime_error("late progress failure");
    });
    task->Execute();
    CheckTerminal(task, state, Hdf5TableStatus::ReadFailed);
}
#endif
} // namespace

int main() try {
    Workspace workspace;
    TestIdentity(workspace);
    bool rejected_null_state = false;
    try { MakeHdf5SourceLoadTask({}, nullptr); }
    catch (const std::invalid_argument&) { rejected_null_state = true; }
    Check(rejected_null_state, "null result storage rejected");
#ifdef CYXWIZ_HAS_HDF5
    const auto path = (workspace.path / "source.h5").string();
    CreateFixture(path);
    TestValuesAndOwnership(path);
    TestAudit(path);
    TestFailures(path, workspace);
    TestCancellation(path);
#else
    Check(!Hdf5TableSupportAvailable(), "HDF5 support absent");
    Hdf5SourceLoadRequest request;
    request.path = "missing/unavailable.h5";
    request.expected_source = Hdf5SourceStamp{"unavailable", 1, 2};
    request.cancel_requested = []() -> bool { throw std::runtime_error("must not inspect source"); };
    auto result = Run(request, Hdf5TableStatus::DependencyUnavailable);
    Check(!result.source_changed, "dependency failure precedes filesystem identity");
    request.settings.selection.data_path = "invalid";
    request.settings.max_materialized_bytes = 0;
    Run(request, Hdf5TableStatus::DependencyUnavailable);
#endif
    std::cout << "HDF5 source load task: " << checks << " checks passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
