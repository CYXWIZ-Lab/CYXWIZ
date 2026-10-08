#include <iostream>

#ifdef CYXWIZ_HAS_HDF5
#include "../src/core/arrow_dataset.h"
#include "../src/core/async_task_manager.h"
#include "../src/core/data_preview_service.h"
#include "../src/core/data_registry.h"
#include "../src/core/dataset_audit.h"
#include "../src/core/hdf5_source_load_task.h"

#include <arrow/api.h>
#include <arrow/util/key_value_metadata.h>
#include <highfive/highfive.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace fs = std::filesystem;
int checks = 0;

void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

struct Workspace {
    const std::string name = "cyxwiz_hdf5_prepared_" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const fs::path path = fs::temp_directory_path() / name;
    const std::string second_name = name + "_second";

    Workspace() { Check(fs::create_directory(path), "create unique fixture workspace"); }
    ~Workspace() {
        cyxwiz::DataRegistry::Instance().UnregisterTabularDataset(name);
        cyxwiz::DataRegistry::Instance().UnregisterTabularDataset(second_name);
        std::error_code error;
        fs::remove_all(path, error);
        if (error) std::cerr << "Fixture cleanup failed: " << error.message() << '\n';
    }
};

std::shared_ptr<arrow::Table> MakeBaseline() {
    arrow::Int64Builder builder;
    Check(builder.Append(-11).ok(), "append first baseline value");
    Check(builder.Append(22).ok(), "append second baseline value");
    std::shared_ptr<arrow::Array> values;
    Check(builder.Finish(&values).ok(), "finish baseline values");
    return arrow::Table::Make(
        arrow::schema({arrow::field("baseline", arrow::int64())}), {values});
}

void CheckBaseline(cyxwiz::DataRegistry& registry, const std::string& name,
                   const std::shared_ptr<cyxwiz::ArrowDataset>& baseline,
                   const std::shared_ptr<arrow::Table>& table,
                   const std::string& context) {
    const auto current = registry.GetArrowDataset(name);
    Check(current == baseline, context + ": registry dataset pointer unchanged");
    Check(current->GetArrowTable() == table, context + ": baseline table pointer unchanged");
    cyxwiz::DataPreviewRequest request;
    request.dataset_name = name;
    const auto page = cyxwiz::DataPreviewService::PreviewRegisteredTabular(registry, request);
    Check(page.ok, context + ": baseline preview succeeds: " + page.reason);
    Check(page.schema.size() == 1 && page.schema[0].name == "baseline" &&
              page.schema[0].type == "int64", context + ": baseline schema unchanged");
    Check(page.rows == std::vector<std::vector<std::string>>{{"-11"}, {"22"}},
          context + ": baseline values unchanged");
}

cyxwiz::Hdf5SourceLoadResult Prepare(const cyxwiz::Hdf5SourceLoadRequest& request,
                                    cyxwiz::Hdf5TableStatus expected,
                                    const std::string& context) {
    auto state = std::make_shared<cyxwiz::Hdf5SourceLoadTaskResult>();
    auto task = cyxwiz::MakeHdf5SourceLoadTask(request, state);
    Check(task && !state->done.load(), context + ": task starts unpublished");
    task->Execute();
    Check(state->done.load(std::memory_order_acquire), context + ": task publishes completion");
    Check(state->result.read.status == expected,
          context + ": expected read status: " + state->result.read.error);
    const auto terminal = expected == cyxwiz::Hdf5TableStatus::Ok
        ? cyxwiz::TaskState::Completed
        : (expected == cyxwiz::Hdf5TableStatus::Cancelled
            ? cyxwiz::TaskState::Cancelled : cyxwiz::TaskState::Failed);
    Check(task->GetState() == terminal, context + ": matching terminal task state");
    if (expected == cyxwiz::Hdf5TableStatus::Ok) {
        Check(state->result.read.table && state->result.source,
              context + ": successful preparation retains table and verified source");
        Check(state->result.audit && !state->result.audit->HasErrors() &&
                  !state->result.audit->cancelled,
              context + ": successful preparation retains a completed audit without errors");
    } else {
        Check(!state->result.read.table && !state->result.source,
              context + ": failure exposes no prepared table or source stamp");
        Check(!state->result.read.error.empty(), context + ": failure explains its cause");
    }
    return std::move(state->result);
}

void CheckMetadata(const std::shared_ptr<arrow::Table>& table,
                   const std::string& key, const std::string& expected) {
    const auto metadata = table->schema()->metadata();
    Check(metadata != nullptr, "registered table retains schema metadata");
    const auto value = metadata->Get(key);
    Check(value.ok() && value.ValueOrDie() == expected, "registered metadata: " + key);
}

void CheckSourceMetadata(const std::shared_ptr<arrow::Table>& table,
                         const cyxwiz::Hdf5SourceStamp& source,
                         const cyxwiz::Hdf5InputSettings& settings) {
    CheckMetadata(table, "hdf5.source_path", source.canonical_path);
    CheckMetadata(table, "hdf5.source_size", std::to_string(source.size));
    CheckMetadata(table, "hdf5.source_modified", std::to_string(source.modified));
    CheckMetadata(table, "hdf5.import_mode", "numeric_table");
    CheckMetadata(table, "hdf5.data_path", settings.selection.data_path);
    CheckMetadata(table, "hdf5.label_path", settings.selection.label_path);
    CheckMetadata(table, "hdf5.numeric_policy", "preserve");
    CheckMetadata(table, "label_column", "label");
}

void TestPreparedSource() {
    Workspace workspace;
    const auto path = (workspace.path / "source.h5").string();
    const std::vector<uint64_t> values{
        9007199254740993ULL, std::numeric_limits<uint64_t>::max(), 9007199254740995ULL, 0};
    const std::vector<int32_t> labels{-7, 42, -3, 42};
    const std::vector<uint64_t> second_values{0, 9007199254740995ULL,
        std::numeric_limits<uint64_t>::max(), 9007199254740993ULL};
    {
        HighFive::File file(path, HighFive::File::Overwrite);
        file.createDataSet<uint64_t>("/data", HighFive::DataSpace::From(values)).write(values);
        file.createDataSet<int32_t>("/labels", HighFive::DataSpace::From(labels)).write(labels);
        file.createDataSet<uint64_t>("/other", HighFive::DataSpace::From(second_values)).write(second_values);
        const std::vector<int32_t> single_labels(4, 5);
        file.createDataSet<int32_t>("/single_labels", HighFive::DataSpace::From(single_labels)).write(single_labels);
        const std::vector<uint64_t> constant(4, 7);
        file.createDataSet<uint64_t>("/constant", HighFive::DataSpace::From(constant)).write(constant);
    }

    auto& registry = cyxwiz::DataRegistry::Instance();
    const auto baseline_table = MakeBaseline();
    const auto baseline = registry.RegisterArrowTable(baseline_table, workspace.name);
    Check(baseline != nullptr, "register preexisting named Arrow baseline");
    const auto publication = registry.CaptureTabularPublication(workspace.name);

    cyxwiz::Hdf5SourceLoadRequest request;
    request.path = path;
    request.settings.selection = {"/data", "/labels"};
    std::string error;
    request.expected_source = cyxwiz::ReadHdf5SourceStamp(path, error);
    Check(request.expected_source.has_value(), "capture fixture identity: " + error);
    const auto prepared = Prepare(request, cyxwiz::Hdf5TableStatus::Ok, "successful preparation");
    Check(prepared.read.table && prepared.read.table->ValidateFull().ok(),
          "preparation produces a valid private Arrow table");
    Check(prepared.source == request.expected_source && !prepared.source_changed,
          "preparation retains the verified source identity");
    CheckBaseline(registry, workspace.name, baseline, baseline_table, "successful preparation");

    // Exercise persisted settings independently of any GUI restoration or Apply wiring.
    std::map<std::string, std::string> parameters{
        {"file_path", path}, {"dataset_name", workspace.name}};
    Check(cyxwiz::WriteHdf5InputSettings(request.settings, parameters, error),
          "serialize canonical HDF5 settings: " + error);
    const auto copied_parameters = parameters;
    const auto restored = cyxwiz::ReadHdf5InputSettings(copied_parameters);
    Check(restored.ok && !restored.migrated_legacy, "read copied canonical settings: " + restored.error);
    Check(restored.settings.selection.data_path == request.settings.selection.data_path &&
              restored.settings.selection.label_path == request.settings.selection.label_path &&
              restored.settings.numeric_policy == request.settings.numeric_policy &&
              restored.settings.max_materialized_bytes == request.settings.max_materialized_bytes,
          "settings round trip preserves selection, policy and byte budget");
    Check(copied_parameters.at("file_path") == path &&
              copied_parameters.at("dataset_name") == workspace.name,
          "settings serialization preserves source path and dataset name");
    auto restored_request = request;
    restored_request.path = copied_parameters.at("file_path");
    restored_request.settings = restored.settings;
    const auto restored_prepared = Prepare(restored_request, cyxwiz::Hdf5TableStatus::Ok, "restored settings");
    Check(restored_prepared.source == prepared.source &&
              restored_prepared.read.table->Equals(*prepared.read.table, true),
          "restored settings prepare identical values, schema and source identity metadata");
    CheckSourceMetadata(restored_prepared.read.table, *request.expected_source, restored.settings);
    CheckBaseline(registry, workspace.name, baseline, baseline_table, "restored settings");

    auto failed = request;
    failed.settings.selection.data_path = "/missing";
    Prepare(failed, cyxwiz::Hdf5TableStatus::MissingDataset, "missing dataset path");
    CheckBaseline(registry, workspace.name, baseline, baseline_table, "missing dataset path");

    failed = request;
    failed.path = (workspace.path / "missing.h5").string();
    failed.expected_source.reset();
    Prepare(failed, cyxwiz::Hdf5TableStatus::InvalidFile, "missing file path");
    CheckBaseline(registry, workspace.name, baseline, baseline_table, "missing file path");

    failed = request;
    failed.settings.max_materialized_bytes = 1;
    Prepare(failed, cyxwiz::Hdf5TableStatus::ResourceLimit, "tiny materialization budget");
    CheckBaseline(registry, workspace.name, baseline, baseline_table, "tiny materialization budget");

    failed = request;
    failed.cancel_requested = [] { return true; };
    Prepare(failed, cyxwiz::Hdf5TableStatus::Cancelled, "cancelled preparation");
    CheckBaseline(registry, workspace.name, baseline, baseline_table, "cancelled preparation");

    failed = request;
    failed.settings.selection.data_path = "/constant";
    const auto refused = Prepare(failed, cyxwiz::Hdf5TableStatus::ReadFailed, "degenerate audit refusal");
    Check(refused.audit && refused.audit->HasErrors() && !refused.audit->cancelled,
          "audit refusal retains error diagnostics without table or source payload");
    bool degenerate_issue = false;
    for (const auto& issue : refused.audit->issues) {
        if (issue.code == "too_many_degenerate_columns" &&
            issue.severity == cyxwiz::DatasetAuditSeverity::Error) degenerate_issue = true;
    }
    Check(degenerate_issue, "audit explains the degenerate-feature refusal");
    CheckBaseline(registry, workspace.name, baseline, baseline_table, "degenerate audit refusal");

    // Test-only publication exercises the registry contract, not a production Apply path.
    const auto registered = std::make_shared<cyxwiz::ArrowDataset>(prepared.read.table, workspace.name);
    CheckBaseline(registry, workspace.name, baseline, baseline_table, "private dataset construction");
    auto staged = registry.PrepareArrowTablePublication(
        publication, registered, prepared.source->canonical_path, error);
    Check(staged != nullptr, "stage private publication: " + error);
    CheckBaseline(registry, workspace.name, baseline, baseline_table, "publication preparation");
    std::map<std::string, std::string> committed_parameters{{"data_loaded", "false"}};
    auto next_parameters = parameters;
    next_parameters["data_loaded"] = "true";
    next_parameters["label_column"] = "label";
    next_parameters["loaded_rows"] = "4";
    next_parameters["loaded_cols"] = "2";
    const auto expected_parameters = next_parameters;
    int notifications = 0;
    bool observed_committed_state = false;
    registry.SetOnDatasetLoaded([&](const std::string& name, const cyxwiz::DatasetInfo&) {
        ++notifications;
        observed_committed_state = name == workspace.name &&
            committed_parameters == expected_parameters && registry.GetArrowDataset(name) == registered;
    });
    const bool published = registry.TryPublishPreparedArrowTable(*staged, error);
    Check(published, "publish using the token captured before preparation: " + error);
    Check(notifications == 0 && committed_parameters.at("data_loaded") == "false",
          "bounded publication defers notification and does not mutate copied node state");
    static_assert(noexcept(committed_parameters.swap(next_parameters)));
    committed_parameters.swap(next_parameters);
    staged->Notify();
    staged->Notify();
    registry.SetOnDatasetLoaded({});
    Check(notifications == 1 && observed_committed_state,
          "notification observes both committed HDF5 settings and registered backing exactly once");
    Check(registered && registered != baseline && registry.GetArrowDataset(workspace.name) == registered,
          "conditional publication replaces the named baseline");
    Check(registry.GetTabularSourcePath(workspace.name) == prepared.source->canonical_path,
          "published dataset has its canonical forward source association");
    Check(!registry.FindTabularDatasetBySourcePath(path),
          "HDF5 selection cannot be reused by filename alone");
    const auto table = registered->GetArrowTable();
    Check(table && table->ValidateFull().ok() && table->num_rows() == 4 && table->num_columns() == 2,
          "registered table retains its shape and validity");
    Check(table->schema()->Equals(*prepared.read.table->schema(), true),
          "registration preserves the complete schema including metadata");
    Check(table->field(0)->name() == "value" && table->field(0)->type()->id() == arrow::Type::UINT64 &&
              table->field(1)->name() == "label" && table->field(1)->type()->id() == arrow::Type::INT32,
          "registration preserves unsigned data and signed label primitives");
    for (int64_t row = 0; row < 4; ++row) {
        const auto value = table->column(0)->GetScalar(row);
        const auto label = table->column(1)->GetScalar(row);
        Check(value.ok() && label.ok(), "registered numeric cells are readable");
        const auto typed_value = std::dynamic_pointer_cast<arrow::UInt64Scalar>(value.ValueOrDie());
        const auto typed_label = std::dynamic_pointer_cast<arrow::Int32Scalar>(label.ValueOrDie());
        Check(typed_value && typed_value->is_valid && typed_value->value == values[row],
              "registered uint64 value is exact at row " + std::to_string(row));
        Check(typed_label && typed_label->is_valid && typed_label->value == labels[row],
              "registered signed label remains aligned at row " + std::to_string(row));
    }
    CheckSourceMetadata(table, *request.expected_source, request.settings);

    cyxwiz::DataPreviewRequest preview;
    preview.dataset_name = workspace.name;
    preview.row_limit = 2;
    preview.selected_columns = {"value", "label"};
    const auto first = cyxwiz::DataPreviewService::PreviewRegisteredTabular(registry, preview);
    Check(first.ok && first.status == cyxwiz::DataPreviewStatus::Ready, first.reason);
    Check(first.backend == "Arrow" && first.total_rows == 4 && first.rows_returned == 2 &&
              first.offset == 0 && first.has_next && first.next_offset == 2,
          "registered HDF5 table supports bounded Arrow preview paging");
    Check(first.schema.size() == 2 && first.schema[0].name == "value" &&
              first.schema[0].type == "uint64" && first.schema[1].name == "label" &&
              first.schema[1].type == "int32", "preview retains primitive schema");
    Check(first.rows == std::vector<std::vector<std::string>>{
              {"9007199254740993", "-7"}, {"18446744073709551615", "42"}},
          "preview renders exact uint64 strings with aligned signed labels");
    preview.offset = first.next_offset;
    const auto tail = cyxwiz::DataPreviewService::PreviewRegisteredTabular(registry, preview);
    Check(tail.ok && tail.status == cyxwiz::DataPreviewStatus::Ready, tail.reason);
    Check(tail.offset == 2 && tail.rows_returned == 2 && !tail.has_next && tail.next_offset == 4,
          "preview tail ends at the source row count");
    Check(tail.rows == std::vector<std::vector<std::string>>{
              {"9007199254740995", "-3"}, {"0", "42"}},
          "later preview rows retain exact values and label alignment");

    const auto second_publication = registry.CaptureTabularPublication(workspace.second_name);
    auto second_request = request;
    second_request.settings.selection = {"/other", "/single_labels"};
    const auto second_prepared = Prepare(second_request, cyxwiz::Hdf5TableStatus::Ok, "second selection");
    Check(second_prepared.audit->HasWarnings(), "single-class warnings permit preparation");
    Check(!registry.GetArrowDataset(workspace.second_name) &&
              registry.GetArrowDataset(workspace.name) == registered,
          "preparing a second selection neither registers it nor replaces the first");
    const auto second = std::make_shared<cyxwiz::ArrowDataset>(second_prepared.read.table, workspace.second_name);
    const bool second_published = registry.TryPublishArrowTable(
        second_publication, second, second_prepared.source->canonical_path, error);
    Check(second_published, "publish separately owned second selection: " + error);
    Check(registry.GetArrowDataset(workspace.name) == registered &&
              registry.GetArrowDataset(workspace.second_name) == second,
          "both selections retain their own registered dataset pointers");
    Check(registry.GetTabularSourcePath(workspace.name) == prepared.source->canonical_path &&
              registry.GetTabularSourcePath(workspace.second_name) == prepared.source->canonical_path,
          "both selections retain forward associations to the same canonical HDF5 file");
    Check(!registry.FindTabularDatasetBySourcePath(path),
          "two selections of one HDF5 file cannot collapse into filename-only reuse");
    CheckSourceMetadata(registered->GetArrowTable(), *prepared.source, request.settings);
    CheckSourceMetadata(second->GetArrowTable(), *prepared.source, second_request.settings);
    Check(second->GetArrowTable()->schema()->Equals(*second_prepared.read.table->schema(), true),
          "second publication preserves its complete primitive schema and metadata");
    preview.dataset_name = workspace.second_name;
    preview.offset = 0;
    preview.row_limit = 4;
    const auto second_page = cyxwiz::DataPreviewService::PreviewRegisteredTabular(registry, preview);
    Check(second_page.ok && second_page.rows == std::vector<std::vector<std::string>>{
              {"0", "5"}, {"9007199254740995", "5"},
              {"18446744073709551615", "5"}, {"9007199254740993", "5"}},
          "second selection has its own exact values and aligned labels: " + second_page.reason);
    preview.dataset_name = workspace.name;
    const auto first_again = cyxwiz::DataPreviewService::PreviewRegisteredTabular(registry, preview);
    Check(first_again.ok && first_again.rows == std::vector<std::vector<std::string>>{
              {"9007199254740993", "-7"}, {"18446744073709551615", "42"},
              {"9007199254740995", "-3"}, {"0", "42"}},
          "publishing another selection preserves the first selection's exact preview");

    const auto stale = registry.CaptureTabularPublication(workspace.name);
    auto stale_staged = registry.PrepareArrowTablePublication(
        stale, registered, prepared.source->canonical_path, error);
    Check(stale_staged != nullptr, "stage candidate before competing registration");
    const auto newer_table = MakeBaseline();
    const auto newer = registry.RegisterArrowTable(newer_table, workspace.name);
    Check(newer && newer != registered, "install a newer registration after token capture");
    const auto newer_source = registry.GetTabularSourcePath(workspace.name);
    error.clear();
    Check(!registry.TryPublishPreparedArrowTable(*stale_staged, error),
          "stale publication token must reject an otherwise valid prepared dataset");
    Check(!error.empty(), "stale publication explains rejection");
    CheckBaseline(registry, workspace.name, newer, newer_table, "stale publication rejection");
    Check(registry.GetTabularSourcePath(workspace.name) == newer_source,
          "stale publication preserves the newer source association");
    Check(registry.GetArrowDataset(workspace.second_name) == second &&
              registry.GetTabularSourcePath(workspace.second_name) == prepared.source->canonical_path,
          "replacement and stale rejection preserve the other selection and its source association");
}
} // namespace
#endif

int main() {
#ifdef CYXWIZ_HAS_HDF5
    try {
        TestPreparedSource();
        std::cout << "HDF5 prepared source test passed: " << checks << " checks\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL after " << checks << " checks: " << error.what() << '\n';
        return 1;
    }
#else
    std::cout << "SKIP: HDF5 prepared source integration requires CYXWIZ_HAS_HDF5 (0 checks)\n";
    return 0;
#endif
}
