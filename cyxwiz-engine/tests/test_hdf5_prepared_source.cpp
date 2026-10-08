#include <iostream>

#ifdef CYXWIZ_HAS_HDF5
#include "../src/core/arrow_dataset.h"
#include "../src/core/async_task_manager.h"
#include "../src/core/data_preview_service.h"
#include "../src/core/data_registry.h"
#include "../src/core/hdf5_source_load_task.h"

#include <arrow/api.h>
#include <arrow/util/key_value_metadata.h>
#include <highfive/highfive.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <limits>
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

    Workspace() { Check(fs::create_directory(path), "create unique fixture workspace"); }
    ~Workspace() {
        cyxwiz::DataRegistry::Instance().UnregisterTabularDataset(name);
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
    if (expected != cyxwiz::Hdf5TableStatus::Ok) {
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

void TestPreparedSource() {
    Workspace workspace;
    const auto path = (workspace.path / "source.h5").string();
    const std::vector<uint64_t> values{
        9007199254740993ULL, std::numeric_limits<uint64_t>::max(), 9007199254740995ULL, 0};
    const std::vector<int32_t> labels{-7, 42, -3, 42};
    {
        HighFive::File file(path, HighFive::File::Overwrite);
        file.createDataSet<uint64_t>("/data", HighFive::DataSpace::From(values)).write(values);
        file.createDataSet<int32_t>("/labels", HighFive::DataSpace::From(labels)).write(labels);
    }

    auto& registry = cyxwiz::DataRegistry::Instance();
    const auto baseline_table = MakeBaseline();
    const auto baseline = registry.RegisterArrowTable(baseline_table, workspace.name);
    Check(baseline != nullptr, "register preexisting named Arrow baseline");

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

    // Test-only publication exercises registry/preview compatibility, not a production Apply path.
    const auto registered = registry.RegisterArrowTable(prepared.read.table, workspace.name);
    Check(registered && registered != baseline && registry.GetArrowDataset(workspace.name) == registered,
          "explicit test registration replaces the named baseline");
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
    CheckMetadata(table, "hdf5.source_path", request.expected_source->canonical_path);
    CheckMetadata(table, "hdf5.source_size", std::to_string(request.expected_source->size));
    CheckMetadata(table, "hdf5.source_modified", std::to_string(request.expected_source->modified));
    CheckMetadata(table, "hdf5.import_mode", "numeric_table");
    CheckMetadata(table, "hdf5.data_path", "/data");
    CheckMetadata(table, "hdf5.label_path", "/labels");
    CheckMetadata(table, "hdf5.numeric_policy", "preserve");
    CheckMetadata(table, "label_column", "label");

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
