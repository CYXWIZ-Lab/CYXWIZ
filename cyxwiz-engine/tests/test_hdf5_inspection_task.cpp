#include "../src/core/hdf5_inspection_task.h"
#include "../src/core/async_task_manager.h"

#ifdef CYXWIZ_HAS_HDF5
#include <highfive/highfive.hpp>
#endif

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

using namespace cyxwiz;
int checks = 0;

void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

void CheckNoPayload(const Hdf5InspectionResult& result) {
    Check(!result.source && result.hierarchy.entries.empty() && !result.hierarchy.has_next &&
              result.preview.rows.empty() && result.preview.schema.empty() && !result.preview.ok &&
              result.preview.total_rows == 0 && result.preview.total_columns == 0 &&
              result.data.path.empty() && result.data.shape.empty() && !result.labels,
          "cancelled or stale inspection must clear all payload");
}

void CheckTerminal(const std::shared_ptr<AsyncTask>& task,
                   const std::shared_ptr<Hdf5InspectionTaskResult>& state, Hdf5TableStatus status) {
    Check(state->done.load(std::memory_order_acquire), "worker publishes done");
    Check(state->result.status == status, "expected inspection status: " + state->result.error);
    const auto terminal = status == Hdf5TableStatus::Ok ? TaskState::Completed :
        (status == Hdf5TableStatus::Cancelled ? TaskState::Cancelled : TaskState::Failed);
    Check(task->GetState() == terminal, "task reaches the matching terminal state");
    if (status != Hdf5TableStatus::Ok) Check(!state->result.error.empty(), "failure explains its cause");
    if (terminal == TaskState::Failed)
        Check(task->GetErrorMessage() == state->result.error, "task and result share the failure reason");
}

Hdf5InspectionResult Run(const Hdf5InspectionRequest& request, Hdf5TableStatus status) {
    auto state = std::make_shared<Hdf5InspectionTaskResult>();
    auto task = MakeHdf5InspectionTask(request, state);
    Check(task->IsCancellable() && !state->done.load(), "new task is cancellable and unpublished");
    task->Execute();
    CheckTerminal(task, state, status);
    auto result = std::move(state->result);
    std::weak_ptr<Hdf5InspectionTaskResult> observer = state;
    state.reset();
    Check(observer.expired(), "completed task does not retain result storage");
    const auto terminal = task->GetState();
    task->Execute();
    Check(task->GetState() == terminal, "repeated Execute is a no-op");
    return result;
}

#ifdef CYXWIZ_HAS_HDF5
struct Workspace {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("cyxwiz_hdf5_inspection_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    Workspace() { Check(std::filesystem::create_directory(path), "unique workspace"); }
    ~Workspace() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};

void ChangeTime(const std::string& path) {
    std::error_code ec;
    const auto before = std::filesystem::last_write_time(path, ec);
    Check(!ec, "read fixture modification time");
    std::filesystem::last_write_time(path, before + std::chrono::seconds(2), ec);
    Check(!ec, "change fixture modification time without waiting");
}

void CreateFixture(const std::string& path) {
    HighFive::File file(path, HighFive::File::Overwrite);
    const std::vector<uint64_t> values{std::numeric_limits<uint64_t>::max(), 9007199254740993ULL,
                                       9007199254740995ULL, 7};
    file.createDataSet<uint64_t>("/values", HighFive::DataSpace::From(values)).write(values);
    const std::vector<uint8_t> labels{2, 1, 0, 3};
    file.createDataSet<uint8_t>("/labels", HighFive::DataSpace::From(labels)).write(labels);
    const std::vector<std::vector<int16_t>> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
    file.createDataSet<int16_t>("/matrix", HighFive::DataSpace::From(matrix)).write(matrix);
    file.createDataSet<float>("/rank3", HighFive::DataSpace(std::vector<size_t>{2, 3, 4}));
    file.createDataSet<double>("/huge", HighFive::DataSpace(std::vector<size_t>{20000000}));
    HighFive::DataSetCreateProps properties;
    properties.add(HighFive::Chunking(std::vector<hsize_t>{3 * 1024 * 1024}));
    file.createDataSet<double>("/chunked", HighFive::DataSpace(std::vector<size_t>{3 * 1024 * 1024}), properties);
    file.createGroup("/group");
    file.createGroup("/group/z");
    file.createGroup("/group/a");
}

void CheckChanged(const Hdf5InspectionResult& result) {
    Check(result.status == Hdf5TableStatus::ReadFailed && result.source_changed &&
              result.error.find("Refresh") != std::string::npos, "source mismatch requests a refresh");
    CheckNoPayload(result);
}
#endif

} // namespace

int main() try {
    bool rejected_null = false;
    try { MakeHdf5InspectionTask({}, nullptr); }
    catch (const std::invalid_argument&) { rejected_null = true; }
    Check(rejected_null, "task requires result storage");
    Hdf5InspectionRequest request;
#ifdef CYXWIZ_HAS_HDF5
    Workspace workspace;
    const auto path = (workspace.path / "input.h5").string();
    CreateFixture(path);
    request.path = path;
    request.browse.group_path = "/group";
    request.browse.limit = 1;
    auto state = std::make_shared<Hdf5InspectionTaskResult>();
    auto task = MakeHdf5InspectionTask(request, state);
    request.path = "changed-after-queue";
    request.browse.group_path = "/missing";
    request.browse.limit = 0;
    request.kind = static_cast<Hdf5InspectionKind>(99);
    task->Execute();
    CheckTerminal(task, state, Hdf5TableStatus::Ok);
    const auto& hierarchy = state->result.hierarchy;
    Check(hierarchy.entries.size() == 1 && hierarchy.entries[0].name == "a" &&
              hierarchy.total_entries == 2 && hierarchy.offset == 0 && hierarchy.next_offset == 1 &&
              hierarchy.has_next, "hierarchy uses copied request and bounded name-ordered page");
    Check(state->result.preview.rows.empty() && state->result.data.path.empty(), "hierarchy does not produce table payload");
    Check(state->result.source && state->result.source->canonical_path == std::filesystem::canonical(path).string() &&
              state->result.source->size == std::filesystem::file_size(path), "worker captures canonical source identity");
    const auto stamp = *state->result.source;

    request = {};
    request.path = path;
    request.kind = Hdf5InspectionKind::Preview;
    request.selection = {"/values", "/labels"};
    request.window = {1, 2, 0, 1};
    request.expected_source = stamp;
    state = std::make_shared<Hdf5InspectionTaskResult>();
    task = MakeHdf5InspectionTask(request, state);
    request.selection = {"/missing", ""};
    request.window.row_offset = 99;
    request.expected_source->size++;
    task->Execute();
    CheckTerminal(task, state, Hdf5TableStatus::Ok);
    const auto& preview = state->result.preview;
    Check(preview.ok && preview.status == DataPreviewStatus::Ready && preview.backend == "HDF5 source" &&
              preview.total_rows == 4 && preview.total_columns == 2 && preview.offset == 1 &&
              preview.rows_returned == 2 && preview.has_next && preview.next_offset == 3,
          "preview reports full source totals and sampled window");
    Check(preview.schema.size() == 2 && preview.schema[0].name == "value" && preview.schema[0].type == "uint64" &&
              !preview.schema[0].nullable && preview.schema[1].name == "label" && preview.schema[1].type == "uint8" &&
              !preview.schema[1].nullable && preview.schema[0].sampled_values == 2 && preview.schema[0].sampled_nulls == 0,
          "preview retains native schema and sample counts");
    Check(preview.rows == std::vector<std::vector<std::string>>{{"9007199254740993", "1"}, {"9007199254740995", "0"}},
          "uint64 stringification is exact above float64 precision");
    Check(state->result.data.path == "/values" && state->result.data.shape == std::vector<uint64_t>({4}) &&
              state->result.labels && state->result.labels->path == "/labels" &&
              state->result.labels->shape == std::vector<uint64_t>({4}), "full descriptors survive sampled preview");

    request = {};
    request.path = path;
    request.kind = Hdf5InspectionKind::Preview;
    request.selection = {"/values", "/labels"};
    request.window = {0, 1, 0, 1};
    Check(Run(request, Hdf5TableStatus::Ok).preview.rows[0][0] == "18446744073709551615", "maximum uint64 remains exact");
    request.selection.data_path = "/matrix";
    request.window = {2, 2, 1, 1};
    auto result = Run(request, Hdf5TableStatus::Ok);
    Check(result.preview.total_columns == 4 && result.preview.schema.size() == 2 &&
              result.preview.schema[0].name == "col_1" && result.preview.schema[0].type == "int16" &&
              result.preview.rows == std::vector<std::vector<std::string>>{{"8", "0"}, {"11", "3"}} &&
              !result.preview.has_next && result.preview.next_offset == 4, "column window preserves full totals and aligned labels");
    request.window.row_offset = 4;
    result = Run(request, Hdf5TableStatus::Ok);
    Check(result.preview.rows.empty() && result.preview.rows_returned == 0 && result.preview.schema.size() == 2 &&
              result.preview.offset == 4 && result.preview.next_offset == 4 && !result.preview.has_next,
          "end-of-source window retains schema with no rows");
    request.selection = {"/huge", ""};
    request.window = {0, 1, 0, 1};
    result = Run(request, Hdf5TableStatus::Ok);
    Check(result.preview.total_rows == 20000000 && result.preview.rows_returned == 1, "large source reads only requested sample");
    request.selection.data_path = "/chunked";
    result = Run(request, Hdf5TableStatus::ResourceLimit);
    Check(result.data.path == "/chunked" && result.preview.rows.empty(), "fixed 16 MiB policy rejects a 24 MiB decoded chunk");
    request.selection.data_path = "/rank3";
    result = Run(request, Hdf5TableStatus::UnsupportedRank);
    Check(result.data.shape == std::vector<uint64_t>({2, 3, 4}) && result.data.source_type == "float" &&
              result.preview.rows.empty(), "unsupported selection retains inspectable descriptors");
    request.selection.data_path = "/missing";
    Run(request, Hdf5TableStatus::MissingDataset);
    request.selection.data_path = "values";
    Run(request, Hdf5TableStatus::InvalidSelection);
    request.selection.data_path = "/values";
    for (const auto window : {Hdf5TablePreviewRequest{0, 0, 0, 1}, {0, 201, 0, 1}, {0, 1, 0, 0}, {0, 1, 0, 65}}) {
        request.window = window;
        Run(request, Hdf5TableStatus::InvalidSelection);
    }
    request.window = {0, 4, 0, 1};

    state = std::make_shared<Hdf5InspectionTaskResult>();
    std::weak_ptr<Hdf5InspectionTaskResult> observer = state;
    auto cancel_owner = std::make_shared<int>(42);
    std::weak_ptr<int> cancel_observer = cancel_owner;
    request.browse.cancel_requested = [cancel_owner] { return *cancel_owner != 42; };
    task = MakeHdf5InspectionTask(request, state);
    request.browse.cancel_requested = {};
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

    for (int field = 0; field < 3; ++field) {
        request.expected_source = stamp;
        if (field == 0) request.expected_source->canonical_path += ".different";
        else if (field == 1) ++request.expected_source->size;
        else ++request.expected_source->modified;
        CheckChanged(Run(request, Hdf5TableStatus::ReadFailed));
    }
    request.expected_source = stamp;
    ChangeTime(path);
    CheckChanged(Run(request, Hdf5TableStatus::ReadFailed));
    request.expected_source.reset();
    for (const auto kind : {Hdf5InspectionKind::Hierarchy, Hdf5InspectionKind::Preview}) {
        request.kind = kind;
        state = std::make_shared<Hdf5InspectionTaskResult>();
        task = MakeHdf5InspectionTask(request, state);
        bool changed = false;
        task->SetProgressCallback([&](float progress, const std::string&) {
            if (!changed && progress >= 0.15f) { changed = true; ChangeTime(path); }
        });
        task->Execute();
        CheckTerminal(task, state, Hdf5TableStatus::ReadFailed);
        CheckChanged(state->result);
    }
    request.kind = Hdf5InspectionKind::Preview;
    state = std::make_shared<Hdf5InspectionTaskResult>();
    task = MakeHdf5InspectionTask(request, state);
    bool changed_size = false;
    task->SetProgressCallback([&](float progress, const std::string&) {
        if (!changed_size && progress >= 0.60f) {
            changed_size = true;
            std::ofstream append(path, std::ios::binary | std::ios::app);
            append << "changed";
            Check(static_cast<bool>(append), "change source size after sampling");
        }
    });
    task->Execute();
    CheckTerminal(task, state, Hdf5TableStatus::ReadFailed);
    CheckChanged(state->result);

    state = std::make_shared<Hdf5InspectionTaskResult>();
    task = MakeHdf5InspectionTask(request, state);
    task->RequestCancel();
    Check(!state->done.load(), "RequestCancel does not publish worker-owned result storage");
    task->Execute();
    CheckTerminal(task, state, Hdf5TableStatus::Cancelled);
    CheckNoPayload(state->result);
    for (const auto kind : {Hdf5InspectionKind::Hierarchy, Hdf5InspectionKind::Preview}) {
        request.kind = kind;
        request.browse.group_path = "/group";
        request.selection = {"/matrix", "/labels"};
        request.window = {0, 4, 0, 3};
        bool reading = false;
        bool formatting = false;
        int polls = 0;
        request.browse.cancel_requested = [&] { return reading && ++polls >= (kind == Hdf5InspectionKind::Hierarchy ? 4 : 6); };
        state = std::make_shared<Hdf5InspectionTaskResult>();
        task = MakeHdf5InspectionTask(request, state);
        task->SetProgressCallback([&](float progress, const std::string&) {
            if (progress >= 0.15f) reading = true;
            if (progress >= 0.60f) formatting = true;
        });
        task->Execute();
        CheckTerminal(task, state, Hdf5TableStatus::Cancelled);
        CheckNoPayload(state->result);
        Check(!formatting && polls >= 4, "cancellation reaches enumeration/materialization before formatting");
    }
    request.browse.cancel_requested = {};
    request.kind = Hdf5InspectionKind::Preview;
    for (float threshold : {0.61f, 0.95f}) {
        state = std::make_shared<Hdf5InspectionTaskResult>();
        task = MakeHdf5InspectionTask(request, state);
        bool requested = false;
        task->SetProgressCallback([&](float progress, const std::string&) {
            if (!requested && progress >= threshold) { requested = true; task->RequestCancel(); }
        });
        task->Execute();
        Check(requested, "test reached stringification or final identity stage");
        CheckTerminal(task, state, Hdf5TableStatus::Cancelled);
        CheckNoPayload(state->result);
    }
    for (bool unknown : {false, true}) {
        request.browse.cancel_requested = [unknown]() -> bool {
            if (unknown) throw 7;
            throw std::runtime_error("injected inspection failure");
        };
        result = Run(request, Hdf5TableStatus::ReadFailed);
        CheckNoPayload(result);
    }
    request.browse.cancel_requested = {};
    state = std::make_shared<Hdf5InspectionTaskResult>();
    task = MakeHdf5InspectionTask(request, state);
    task->SetProgressCallback([](float progress, const std::string&) {
        if (progress > 0.60f) throw std::runtime_error("stringification progress failed");
    });
    task->Execute();
    CheckTerminal(task, state, Hdf5TableStatus::ReadFailed);
    CheckNoPayload(state->result);
    request.path = path + '\0' + "suffix";
    Run(request, Hdf5TableStatus::InvalidFile);
    request.path = (workspace.path / "missing.h5").string();
    Run(request, Hdf5TableStatus::InvalidFile);
    request.expected_source = stamp;
    CheckChanged(Run(request, Hdf5TableStatus::ReadFailed));
    request.expected_source.reset();
    request.path = (workspace.path / "fake.h5").string();
    { std::ofstream fake(request.path, std::ios::binary); fake << "not HDF5"; }
    Run(request, Hdf5TableStatus::InvalidFile);
    request.kind = static_cast<Hdf5InspectionKind>(99);
    Run(request, Hdf5TableStatus::InvalidSelection);
#else
    request.path = "missing/unavailable.h5";
    request.expected_source = Hdf5SourceStamp{"unavailable", 1, 2};
    for (const auto kind : {Hdf5InspectionKind::Hierarchy, Hdf5InspectionKind::Preview}) {
        request.kind = kind;
        const auto result = Run(request, Hdf5TableStatus::DependencyUnavailable);
        Check(!result.source_changed, "missing dependency takes precedence over filesystem identity");
        CheckNoPayload(result);
    }
#endif
    std::cout << "HDF5 inspection task: " << checks << " checks passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
